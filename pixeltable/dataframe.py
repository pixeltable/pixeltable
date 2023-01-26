import base64
import io
import os
from typing import List, Optional, Any, Dict, Generator, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import sqlalchemy as sql
from PIL import Image
import copy

from pixeltable import catalog
from pixeltable.env import Env
from pixeltable.type_system import ColumnType
from pixeltable import exprs
from pixeltable import exceptions as exc

__all__ = [
    'DataFrame'
]


def _format_img(img: object) -> str:
    """
    Create <img> tag for Image object.
    """
    assert isinstance(img, Image.Image), f'Wrong type: {type(img)}'
    with io.BytesIO() as buffer:
        img.save(buffer, 'jpeg')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{img_base64}">'

def _format_video(video_file_path: str) -> str:
    # turn absolute video_file_path into relative path, absolute paths don't work
    p = Path(video_file_path)
    root = Path(os.getcwd())
    #print(root)
    #return f'<video controls><source src="{video_file_path}" type="video/mp4"></video>'
    try:
        rel_path = p.relative_to(root)
        return f'<video controls><source src="{rel_path}" type="video/mp4"></video>'
    except ValueError:
        # display path as string
        return video_file_path

class DataFrameResultSet:
    def __init__(self, rows: List[List], col_names: List[str], col_types: List[ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def __len__(self) -> int:
        return len(self.rows)

    def _repr_html_(self) -> str:
        img_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type.is_image_type()]
        video_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type.is_video_type()]
        formatters = {self.col_names[i]: _format_img for i in img_col_idxs}
        formatters.update({self.col_names[i]: _format_video for i in video_col_idxs})
        # escape=False: make sure <img> tags stay intact
        # TODO: why does mypy complain about formatters having an incorrect type?
        return self.to_pandas().to_html(formatters=formatters, escape=False, index=False)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.to_pandas().to_string()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, tuple):
            if len(index) != 2 or not isinstance(index[0], int) or not isinstance(index[1], int):
                raise exc.OperationalError(f'Bad index: {index}')
            return self.rows[index[0]][index[1]]


class AnalysisInfo:
    def __init__(self):
        # output of the SQL scan stage
        self.sql_scan_output_exprs: List[exprs.Expr] = []
        # output of the agg stage
        self.agg_output_exprs: List[exprs.Expr] = []
        # select list providing the input to the SQL scan stage
        self.sql_select_list: List[sql.sql.expression.ClauseElement] = []
        # Where clause of the Select stmt of the SQL scan stage
        self.sql_where_clause: Optional[sql.sql.expression.ClauseElement] = None
        # filter predicate applied to input rows of the SQL scan stage
        self.filter: Optional[exprs.Predicate] = None
        self.similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None
        self.agg_fn_calls: List[exprs.FunctionCall] = []  # derived from unique_exprs

        self.unique_exprs = exprs.UniqueExprSet()
        self.next_data_row_idx = 0

    @property
    def num_materialized(self) -> int:
        return self.next_data_row_idx

    def assign_idxs(self, expr_list: List[exprs.Expr]) -> None:
        """
        Assign data/sql_row_idx to exprs in expr_list and all their subcomponents.
        An expr with to_sql() != None is assumed to be materialized fully via SQL; its components
        aren't materialized and don't receive idxs.
        """
        for e in expr_list:
            self._assign_idxs_aux(e)
        self.agg_fn_calls = [e for e in self.unique_exprs if isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call]

    def _assign_idxs_aux(self, expr: exprs.Expr) -> None:
        if not self.unique_exprs.add(expr):
            # nothing left to do
            return

        sql_expr = expr.sql_expr()
        # if this can be materialized via SQL we don't need to look at its components;
        # we special-case Literals because we don't want to have to materialize them via SQL
        if sql_expr is not None and not isinstance(expr, exprs.Literal):
            assert expr.data_row_idx < 0
            expr.data_row_idx = self.next_data_row_idx
            self.next_data_row_idx += 1
            expr.sql_row_idx = len(self.sql_select_list)
            self.sql_select_list.append(sql_expr)
            return

        # expr value needs to be computed via Expr.eval()
        for c in expr.components:
            self._assign_idxs_aux(c)
        assert expr.data_row_idx < 0
        expr.data_row_idx = self.next_data_row_idx
        self.next_data_row_idx += 1


class DataFrame:
    def __init__(
            self, tbl: catalog.Table,
            select_list: Optional[List[exprs.Expr]] = None,
            where_clause: Optional[exprs.Predicate] = None):
        self.tbl = tbl
        # self.select_list and self.where_clause contain execution state and therefore cannot be shared
        self.select_list: Optional[List[exprs.Expr]] = None  # None: implies all cols
        if select_list is not None:
            self.select_list = [e.copy() for e in select_list]
        self.where_clause: Optional[exprs.Predicate] = None
        if where_clause is not None:
            self.where_clause = where_clause.copy()
        self.group_by_clause: Optional[List[exprs.Expr]] = None
        self.analysis_info: Optional[AnalysisInfo] = None

    def analyze(self) -> None:
        """
        Populates self.analysis_info.
        """
        info = self.analysis_info = AnalysisInfo()
        if self.where_clause is not None:
            info.sql_where_clause, info.filter = self.where_clause.extract_sql_predicate()
            if info.filter is not None:
                similarity_clauses, info.filter = info.filter.split_conjuncts(
                    lambda e: isinstance(e, exprs.ImageSimilarityPredicate))
                if len(similarity_clauses) > 1:
                    raise exc.OperationalError(f'More than one nearest() or matches() not supported')
                if len(similarity_clauses) == 1:
                    info.similarity_clause = similarity_clauses[0]
                    img_col = info.similarity_clause.img_col_ref.col
                    if not img_col.is_indexed:
                        raise exc.OperationalError(
                            f'nearest()/matches() not available for unindexed column {img_col.name}')

        if info.filter is not None:
            info.assign_idxs([info.filter])
        if len(self.group_by_clause) > 0:
            info.assign_idxs(self.group_by_clause)
            for e in self.group_by_clause:
                self._analyze_group_by(e, True)
        info.assign_idxs(self.select_list)
        grouping_expr_idxs = set([e.data_row_idx for e in self.group_by_clause])
        item_is_agg = [self._analyze_select_list(e, grouping_expr_idxs)[0]  for e in self.select_list]

        if self.is_agg():
            # this is an aggregation
            if item_is_agg.count(False) > 0:
                raise exc.Error(f'Invalid non-aggregate in select list: {self.select_list[item_is_agg.find(False)]}')
            # the agg stage materializes select list items that haven't already been provided by SQL
            info.agg_output_exprs = [e for e in self.select_list if e.sql_row_idx == -1]
            # our sql scan stage needs to materialize: grouping exprs, arguments of agg fn calls
            info.sql_scan_output_exprs = copy.copy(self.group_by_clause)
            unique_args: Set[int] = set()
            for fn_call in info.agg_fn_calls:
                for c in fn_call.components:
                    unique_args.add(c.data_row_idx)
            all_exprs = {e.data_row_idx: e for e in info.unique_exprs}
            info.sql_scan_output_exprs.extend([all_exprs[idx] for idx in unique_args])
        else:
            info.sql_scan_output_exprs = self.select_list

    def is_agg(self) -> bool:
        return len(self.group_by_clause) > 0 \
            or (self.analysis_info is not None and len(self.analysis_info.agg_fn_calls) > 0)

    def _is_agg_fn_call(self, e: exprs.Expr) -> bool:
        return isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call

    def _analyze_group_by(self, e: exprs.Expr, check_sql: bool) -> None:
        """
        Make sure that group-by exprs don't contain aggregates.
        """
        if e.sql_row_idx == -1 and check_sql:
            raise exc.Error(f'Invalid grouping expr, needs to be expressible in SQL: {e}')
        if self._is_agg_fn_call(e):
            raise exc.Error(f'Cannot group by aggregate function: {e}')
        for c in e.components:
            self._analyze_group_by(c, False)

    def _analyze_select_list(self, e: exprs.Expr, grouping_exprs: Set[int]) -> Tuple[bool, bool]:
        """
        Analyzes select list item. Returns (list item is output of agg stage, item is output of scan stage).
        Collects agg fn calls in self.analysis_info.
        """
        if e.data_row_idx in grouping_exprs:
            return True, True
        elif self._is_agg_fn_call(e):
            for c in e.components:
                _, is_scan_output = self._analyze_select_list(c, grouping_exprs)
                if not is_scan_output:
                    raise exc.Error(f'Invalid nested aggregates: {e}')
            return True, False
        elif isinstance(e, exprs.Literal):
            return True, True
        elif isinstance(e, exprs.ColumnRef):
            # we already know that this isn't a grouping expr
            return False, True
        else:
            # an expression such as <grouping expr 1> + <grouping expr 2> can be the output of both
            # the agg stage and the scan stage
            component_is_agg: List[bool] = []
            component_is_scan: List[bool] = []
            for c in e.components:
                is_agg, is_scan = self._analyze_select_list(c, grouping_exprs)
                component_is_agg.append(is_agg)
                component_is_scan.append(is_scan)
            is_agg = component_is_agg.count(True) == len(e.components)
            is_scan = component_is_scan.count(True) == len(e.components)
            if not is_agg and not is_scan:
                raise exc.Error(f'Invalid expression, mixes aggregate with non-aggregate: {e}')
            return is_agg, is_scan

    def exec(self, n: int = 20, select_pk: bool = False) -> Generator[List[Any], None, None]:
        """
        Returned value: list of select list values.
        If select_pk == True, also selects the primary key of the storage table (which is rowid and v_min).
        """
        if self.select_list is None:
            self.select_list = [exprs.ColumnRef(col) for col in self.tbl.columns]
        if self.group_by_clause is None:
            self.group_by_clause = []
        for item in self.select_list:
            item.bind_rel_paths(None)
        if self.analysis_info is None:
            self.analyze()
        if self.analysis_info.similarity_clause is not None and n > 100:
            raise exc.OperationalError(f'nearest()/matches() requires show(n <= 100): n={n}')

        # determine order_by clause for window functions or grouping, if present
        window_fn_calls = [
            e for e in self.analysis_info.unique_exprs
            if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
        ]
        if len(window_fn_calls) > 0 and self.is_agg():
            raise exc.Error(f'Cannot combine window functions with non-windowed aggregation')
        order_by_exprs: List[exprs.Expr] = []
        # TODO: check compatibility of window clauses
        if len(window_fn_calls) > 0:
            order_by_exprs = window_fn_calls[0].get_window_sort_exprs()
        elif self.is_agg():
            # TODO: collect aggs with order-by and analyze for compatibility
            order_by_exprs = self.group_by_clause + self.analysis_info.agg_fn_calls[0].get_agg_order_by()
        order_by_clause = [e.sql_expr() for e in order_by_exprs]
        for i in range(len(order_by_exprs)):
            if order_by_clause[i] is None:
                raise exc.Error(f'order_by element cannot be expressed in SQL: {order_by_exprs[i]}')

        idx_rowids: List[int] = []  # rowids returned by index lookup
        if self.analysis_info.similarity_clause is not None:
            # do index lookup
            assert self.analysis_info.similarity_clause.img_col_ref.col.idx is not None
            embed = self.analysis_info.similarity_clause.embedding()
            idx_rowids = self.analysis_info.similarity_clause.img_col_ref.col.idx.search(embed, n, self.tbl.valid_rowids)

        with Env.get().engine.connect() as conn:
            stmt = self._create_select_stmt(
                self.analysis_info.sql_select_list, self.analysis_info.sql_where_clause, idx_rowids, select_pk,
                order_by_clause)
            num_rows = 0
            sql_scan_evaluator = exprs.ExprEvaluator(
                self.analysis_info.sql_scan_output_exprs, self.analysis_info.filter)
            agg_evaluator = exprs.ExprEvaluator(self.analysis_info.agg_output_exprs, None)

            current_group: Optional[List[Any]] = None  # for grouping agg, the values of the group-by exprs
            for row in conn.execute(stmt):
                sql_row = row._data
                data_row: List[Any] = [None] * self.analysis_info.num_materialized
                if not sql_scan_evaluator.eval(sql_row, data_row):
                    continue

                # copy select list results into contiguous array
                result_row: Optional[List[Any]] = None
                if self.is_agg():
                    group = [data_row[e.data_row_idx] for e in self.group_by_clause]
                    if current_group is None:
                        current_group = group
                    if group != current_group:
                        # we're entering a new group, emit a row for the last one
                        agg_evaluator.eval(last_sql_row, last_data_row)
                        result_row = [last_data_row[e.data_row_idx] for e in self.select_list]
                        current_group = group
                        for fn_call in self.analysis_info.agg_fn_calls:
                            fn_call.reset_agg()
                    for fn_call in self.analysis_info.agg_fn_calls:
                        fn_call.update(data_row)
                else:
                    result_row = [data_row[e.data_row_idx] for e in self.select_list]
                    if select_pk:
                        result_row.extend(sql_row[-2:])

                last_data_row = data_row
                last_sql_row = row._data
                if result_row is not None:
                    yield result_row
                    num_rows += 1
                    if n > 0 and num_rows == n:
                        break

            if self.is_agg():
                # we need to emit the output row for the current group
                agg_evaluator.eval(sql_row, data_row)
                result_row = [data_row[e.data_row_idx] for e in self.select_list]
                yield result_row

    def show(self, n: int = 20) -> DataFrameResultSet:
        data_rows = [row for row in self.exec(n)]
        col_names = [expr.display_name() for expr in self.select_list]
        # replace ''
        col_names = [n if n != '' else f'col_{i}' for i, n in enumerate(col_names)]
        return DataFrameResultSet(data_rows, col_names, [expr.col_type for expr in self.select_list])

    def count(self) -> int:
        """
        TODO: implement as part of DataFrame.agg()
        """
        stmt = sql.select(sql.func.count('*')).select_from(self.tbl.sa_tbl) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if self.where_clause is not None:
            sql_where_clause = self.where_clause.sql_expr()
            assert sql_where_clause is not None
            stmt = stmt.where(sql_where_clause)
        with Env.get().engine.connect() as conn:
            result: int = conn.execute(stmt).scalar_one()
            assert isinstance(result, int)
            return result

    def categorical_map(self) -> Dict[str, int]:
        """
        Return map of distinct values in string ColumnRef to increasing integers.
        TODO: implement as part of DataFrame.agg()
        """
        if self.select_list is None or len(self.select_list) != 1 \
            or not isinstance(self.select_list[0], exprs.ColumnRef) \
            or not self.select_list[0].col_type.is_string_type():
            raise exc.OperationalError(f'categoricals_map() can only be applied to an individual string column')
        assert isinstance(self.select_list[0], exprs.ColumnRef)
        col = self.select_list[0].col
        stmt = sql.select(sql.distinct(col.sa_col)) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version) \
            .order_by(col.sa_col)
        if self.where_clause is not None:
            sql_where_clause = self.where_clause.sql_expr()
            assert sql_where_clause is not None
            stmt = stmt.where(sql_where_clause)
        with Env.get().engine.connect() as conn:
            result = {row._data[0]: i for i, row in enumerate(conn.execute(stmt))}
            return result

    def __getitem__(self, index: object) -> 'DataFrame':
        """
        Allowed:
        - [<Predicate>]: filter operation
        - [List[Expr]]/[Tuple[Expr]]: setting the select list
        - [Expr]: setting a single-col select list
        """
        if isinstance(index, exprs.Predicate):
            return DataFrame(self.tbl, select_list=self.select_list, where_clause=index)
        if isinstance(index, tuple):
            index = list(index)
        if isinstance(index, exprs.Expr):
            index = [index]
        if isinstance(index, list):
            if self.select_list is not None:
                raise exc.OperationalError(f'[] for column selection is only allowed once')
            # analyze select list; wrap literals with the corresponding expressions and update it in place
            for i in range(len(index)):
                expr = index[i]
                if isinstance(expr, dict):
                    index[i] = expr = exprs.InlineDict(expr)
                if isinstance(expr, list):
                    index[i] = expr = exprs.InlineArray(tuple(expr))
                if not isinstance(expr, exprs.Expr):
                    raise exc.OperationalError(f'Invalid expression in []: {expr}')
                if expr.col_type.is_invalid_type():
                    raise exc.OperationalError(f'Invalid type: {expr}')
                # TODO: check that ColumnRefs in expr refer to self.tbl
            return DataFrame(self.tbl, select_list=index, where_clause=self.where_clause)
        raise TypeError(f'Invalid index type: {type(index)}')

    def group_by(self, *expr_list: Tuple[exprs.Expr]) -> 'DataFrame':
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise exc.Error(f'Invalid expr in group_by(): {e}')
        self.group_by_clause = [e.copy() for e in expr_list]
        return self

    def _create_select_stmt(
            self, select_list: List[sql.sql.expression.ClauseElement],
            where_clause: Optional[sql.sql.expression.ClauseElement],
            valid_rowids: List[int],
            select_pk: bool,
            order_by_exprs: List[sql.sql.expression.ClauseElement]
    ) -> sql.sql.expression.Select:
        pk_cols = [self.tbl.rowid_col, self.tbl.v_min_col] if select_pk else []
        # we add pk_cols at the end so that the already-computed sql row indices remain correct
        stmt = sql.select(*select_list, *pk_cols) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        if len(valid_rowids) > 0:
            stmt = stmt.where(self.tbl.rowid_col.in_(valid_rowids))
        if len(order_by_exprs) > 0:
            stmt = stmt.order_by(*order_by_exprs)
        return stmt
