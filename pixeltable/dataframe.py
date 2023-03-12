import base64
import io
import os
import sys
from typing import List, Optional, Any, Dict, Generator, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import sqlalchemy as sql
from PIL import Image
import copy
import traceback
from types import TracebackType

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
                raise exc.RuntimeError(f'Bad index: {index}')
            return self.rows[index[0]][index[1]]


class AnalysisInfo:
    def __init__(self, tbl: catalog.Table):
        self.tbl = tbl
        # output of the SQL scan stage
        self.sql_scan_output_exprs: List[exprs.Expr] = []
        # output of the agg stage
        self.agg_output_exprs: List[exprs.Expr] = []
        # Where clause of the Select stmt of the SQL scan stage
        self.sql_where_clause: Optional[sql.sql.expression.ClauseElement] = None
        # filter predicate applied to input rows of the SQL scan stage
        self.filter: Optional[exprs.Predicate] = None
        self.similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None
        self.agg_fn_calls: List[exprs.FunctionCall] = []  # derived from unique_exprs
        self.has_frame_col: bool = False

        self.evaluator: Optional[exprs.Evaluator] = None
        self.sql_scan_eval_ctx: List[exprs.Expr] = []  # needed to materialize output of SQL scan stage
        self.agg_eval_ctx: List[exprs.Expr] = []  # needed to materialize output of agg stage
        self.filter_eval_ctx: List[exprs.Expr] = []
        self.group_by_eval_ctx: List[exprs.Expr] = []

    def prepare_exec(self) -> None:
        """
        Call prepare() on all collected Exprs.
        """
        exprs.Expr.prepare_list(self.sql_scan_output_exprs)
        exprs.Expr.prepare_list(self.agg_output_exprs)
        if self.filter is not None:
            self.filter.prepare()

    def finalize_exec(self) -> None:
        """
        Call finalize() on all collected Exprs.
        """
        exprs.Expr.finalize_list(self.sql_scan_output_exprs)
        exprs.Expr.finalize_list(self.agg_output_exprs)
        if self.filter is not None:
            self.filter.finalize()


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
        info = self.analysis_info = AnalysisInfo(self.tbl)
        if self.where_clause is not None:
            info.sql_where_clause, info.filter = self.where_clause.extract_sql_predicate()
            if info.filter is not None:
                similarity_clauses, info.filter = info.filter.split_conjuncts(
                    lambda e: isinstance(e, exprs.ImageSimilarityPredicate))
                if len(similarity_clauses) > 1:
                    raise exc.RuntimeError(f'More than one nearest() or matches() not supported')
                if len(similarity_clauses) == 1:
                    info.similarity_clause = similarity_clauses[0]
                    img_col = info.similarity_clause.img_col_ref.col
                    if not img_col.is_indexed:
                        raise exc.RuntimeError(
                            f'nearest()/matches() not available for unindexed column {img_col.name}')

        if self.tbl.frame_col is not None and not self.tbl.frame_col.is_stored:
            # we need to replace ColumnRefs for the frame col with FrameColumnRefs
            col_ref = exprs.ColumnRef(self.tbl.frame_col)
            frame_col_ref = exprs.FrameColumnRef(self.tbl.frame_col)
            exprs.Expr.list_substitute(self.select_list, col_ref, frame_col_ref)
        self._substitute_unstored_cols()

        # set up execution state
        eval_exprs = self.select_list.copy()
        if info.filter is not None:
            eval_exprs.append(info.filter)
        if len(self.group_by_clause) > 0:
            eval_exprs.extend(self.group_by_clause)
        info.evaluator = exprs.Evaluator(eval_exprs)
        unique_exprs = info.evaluator.unique_exprs

        # set up eval ctxs
        filter_ids = [info.filter.data_row_idx] if info.filter is not None else []
        info.filter_eval_ctx = info.evaluator.get_eval_ctx(filter_ids)
        group_by_ids = [e.data_row_idx for e in self.group_by_clause] if len(self.group_by_clause) > 0 else []
        info.group_by_eval_ctx = info.evaluator.get_eval_ctx(group_by_ids)
        frame_col_ref_subexprs = [
            e for e in exprs.Expr.list_subexprs(eval_exprs) if isinstance(e, exprs.FrameColumnRef)
        ]
        info.has_frame_col = len(frame_col_ref_subexprs) > 0
        info.agg_fn_calls = [e for e in unique_exprs if isinstance(e, exprs.FunctionCall) and e.is_agg_fn_call]

        if len(self.group_by_clause) > 0:
            for e in self.group_by_clause:
                self._analyze_group_by(e, True)

        grouping_expr_idxs = set([e.data_row_idx for e in self.group_by_clause])
        item_is_agg = [self._analyze_select_list(e, grouping_expr_idxs)[0] for e in self.select_list]

        if self.is_agg():
            # this is an aggregation
            if item_is_agg.count(False) > 0:
                raise exc.Error(f'Invalid non-aggregate in select list: {self.select_list[item_is_agg.find(False)]}')

            # the agg stage materializes select list items that haven't already been provided by SQL
            info.agg_output_exprs = [e for e in self.select_list if e.sql_row_idx == -1]
            info.agg_eval_ctx = info.evaluator.get_eval_ctx([e.data_row_idx for e in info.agg_output_exprs])

            # our sql scan stage needs to materialize: grouping exprs, arguments of agg fn calls
            sql_scan_expr_ids = set([e.data_row_idx for e in self.group_by_clause])
            for fn_call in info.agg_fn_calls:
                for c in fn_call.components:
                    sql_scan_expr_ids.add(c.data_row_idx)
            info.sql_scan_eval_ctx = info.evaluator.get_eval_ctx(list(sql_scan_expr_ids))
            info.sql_scan_output_exprs = [info.evaluator.unique_exprs[id] for id in sql_scan_expr_ids]
        else:
            info.sql_scan_eval_ctx = info.evaluator.get_eval_ctx([e.data_row_idx for e in self.select_list])
            # this assumes we don't need to call finalize() on the where clause
            info.sql_scan_output_exprs = self.select_list

    def _substitute_unstored_cols(self) -> None:
        """
        Replace references to unstored cols (stored set to False, not None) that aren't extracted frame cols
        with their respective value_exprs.
        """
        # select list
        while True:
            subexprs = exprs.Expr.list_subexprs(self.select_list)
            # don't use isinstance() here, it'll also pick up FrameColumnRefs
            unstored_col_refs = [e for e in subexprs if type(e) == exprs.ColumnRef and e.col.stored == False]
            if len(unstored_col_refs) == 0:
                break
            for col_ref in unstored_col_refs:
                assert col_ref.col.value_expr is not None
                exprs.Expr.list_substitute(self.select_list, col_ref, col_ref.col.value_expr)

        # filter
        if self.analysis_info.filter is not None:
            while True:
                subexprs = self.analysis_info.filter.subexprs()
                unstored_col_refs = [
                    e for e in subexprs if isinstance(e, exprs.ColumnRef) and e.col.stored == False
                ]
                if len(unstored_col_refs) == 0:
                    break
                for col_ref in unstored_col_refs:
                    assert col_ref.col.value_expr is not None
                    self.analysis_info.filter = self.analysis_info.filter.substitute(col_ref, col_ref.col.value_expr)

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

    def _reset_agg_state(self, row_num: int) -> None:
        for fn_call in self.analysis_info.agg_fn_calls:
            try:
                fn_call.reset_agg()
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'init() function of the aggregate {fn_call}'
                raise exc.ExprEvalError(fn_call, expr_msg, e, exc_tb, [], row_num)

    def _update_agg_state(self, data_row: List[Any], row_num: int) -> None:
        for fn_call in self.analysis_info.agg_fn_calls:
            try:
                fn_call.update(data_row)
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'update() function of the aggregate {fn_call}'
                input_vals = [data_row[d.data_row_idx] for d in fn_call.dependencies()]
                raise exc.ExprEvalError(fn_call, expr_msg, e, exc_tb, input_vals, row_num)

    def _eval_agg_fns(self, data_row: List[Any], row_num: int) -> None:
        info = self.analysis_info
        exc_tb = info.evaluator.eval(data_row, info.agg_eval_ctx)
        if exc_tb is not None:
            # first expr with exception
            exc_idx = next(idx for idx, val in enumerate(data_row) if isinstance(val, Exception))
            exc_expr = self.analysis_info.evaluator.unique_exprs[exc_idx]
            expr_msg = f'value() function of the aggregate {exc_expr}'
            raise exc.ExprEvalError(exc_expr, expr_msg, data_row[exc_idx], exc_tb, [], row_num)

    def _eval_sql_scan(self, data_row: List[Any], row_num: int, ignore_errors: bool) -> bool:
        info = self.analysis_info
        exc_tb: Optional[TracebackType] = None
        passes_filter = True
        if info.filter is not None:
            exc_tb = info.evaluator.eval(data_row, info.filter_eval_ctx)
            if exc_tb is None:
                passes_filter = data_row[info.filter.data_row_idx]
        if exc_tb is None and passes_filter:
            exc_tb = info.evaluator.eval(data_row, info.sql_scan_eval_ctx)

        if exc_tb is not None and not ignore_errors:
            # first expr with exception
            exc_idx = next(idx for idx, val in enumerate(data_row) if isinstance(val, Exception))
            exc_expr = self.analysis_info.evaluator.unique_exprs[exc_idx]
            expr_msg = f'expression {exc_expr}'
            input_vals = [data_row[d.data_row_idx] for d in exc_expr.dependencies()]
            raise exc.ExprEvalError(exc_expr, expr_msg, data_row[exc_idx], exc_tb, input_vals, row_num)
        return passes_filter

    def exec(
            self, n: int = 20, select_pk: bool = False, ignore_errors: bool = False
    ) -> Generator[List[Any], None, None]:
        """
        Returned value: list of select list values.
        If select_pk == True, also selects the primary key of the storage table (which is rowid and v_min).
        ignore_errors == False: if any expr raises an exception, raises ExprEvalError.
        ignore_errors == True: exception is returned in result row for each select list item that encountered an exc.
        """
        if self.select_list is None:
            self.select_list = [exprs.ColumnRef(col) for col in self.tbl.columns]
        if self.group_by_clause is None:
            self.group_by_clause = []
        for item in self.select_list:
            item.bind_rel_paths(None)
        if self.analysis_info is None:
            self.analyze()
        info = self.analysis_info
        if info.similarity_clause is not None and n > 100:
            raise exc.RuntimeError(f'nearest()/matches() requires show(n <= 100): n={n}')

        # determine order_by clause for window functions or grouping, if present
        window_fn_calls = [
            e for e in info.evaluator.unique_exprs if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
        ]
        if len(window_fn_calls) > 0 and self.is_agg():
            raise exc.Error(f'Cannot combine window functions with non-windowed aggregation')
        order_by_exprs: List[exprs.Expr] = []
        # TODO: check compatibility of window clauses
        if len(window_fn_calls) > 0:
            order_by_exprs = window_fn_calls[0].get_window_sort_exprs()
        elif info.has_frame_col:
            # we're materializing extracted frames and need to order by the frame src and idx cols
            order_by_exprs = [exprs.ColumnRef(self.tbl.frame_src_col()), exprs.ColumnRef(self.tbl.frame_idx_col())]
        elif self.is_agg():
            # TODO: collect aggs with order-by and analyze for compatibility
            order_by_exprs = self.group_by_clause + info.agg_fn_calls[0].get_agg_order_by()
        order_by_clause = [e.sql_expr() for e in order_by_exprs]
        for i in range(len(order_by_exprs)):
            if order_by_clause[i] is None:
                raise exc.Error(f'order_by element cannot be expressed in SQL: {order_by_exprs[i]}')

        idx_rowids: List[int] = []  # rowids returned by index lookup
        if info.similarity_clause is not None:
            # do index lookup
            assert info.similarity_clause.img_col_ref.col.idx is not None
            embed = info.similarity_clause.embedding()
            idx = info.similarity_clause.img_col_ref.col.idx
            idx_rowids = idx.search(embed, n, self.tbl.valid_rowids)

        with Env.get().engine.connect() as conn:
            info.prepare_exec()
            # if we're retrieving an extracted frame column that is not stored, we also need the PK in order to
            # access the file cache
            stmt = self._create_select_stmt(
                info.evaluator.sql_select_list, info.sql_where_clause, idx_rowids, select_pk or info.has_frame_col,
                order_by_clause)
            num_rows = 0  # number of output rows

            current_group: Optional[List[Any]] = None  # for grouping agg, the values of the group-by exprs
            sql_rows = conn.execute(stmt)  # this might raise an exception
            data_row: List[Any] = []
            for row_num, row in enumerate(sql_rows):
                sql_row = row._data
                last_data_row = data_row
                data_row = info.evaluator.prepare(sql_row, info.has_frame_col)
                passes_filter = self._eval_sql_scan(data_row, row_num, ignore_errors)
                if not passes_filter:
                    continue

                # copy select list results into contiguous array
                result_row: Optional[List[Any]] = None
                if self.is_agg():
                    group = [data_row[e.data_row_idx] for e in self.group_by_clause]
                    if current_group is None:
                        current_group = group
                        self._reset_agg_state(row_num)
                    if group != current_group:
                        # we're entering a new group, emit a row for the last one
                        self._eval_agg_fns(last_data_row, row_num)
                        result_row = [last_data_row[e.data_row_idx] for e in self.select_list]
                        current_group = group
                        self._reset_agg_state(row_num)
                    self._update_agg_state(data_row, row_num)
                else:
                    result_row = [data_row[e.data_row_idx] for e in self.select_list]
                    if select_pk:
                        result_row.extend(sql_row[-2:])

                if result_row is not None:
                    yield result_row
                    num_rows += 1
                    if n > 0 and num_rows == n:
                        break

            if self.is_agg():
                # we need to emit the output row for the current group
                self._eval_agg_fns(data_row, row_num)
                result_row = [data_row[e.data_row_idx] for e in self.select_list]
                yield result_row
            info.finalize_exec()

    def show(self, n: int = 20) -> DataFrameResultSet:
        try:
            data_rows = [row for row in self.exec(n)]
        except exc.ExprEvalError as e:
            msg = (f'In row {e.row_num} the {e.expr_msg} encountered exception '
                   f'{type(e.exc).__name__}:\n{str(e.exc)}')
            if len(e.input_vals) > 0:
                input_msgs = [
                    f"'{d}' = {d.col_type.print_value(e.input_vals[i])}"
                    for i, d in enumerate(e.expr.dependencies())
                ]
                msg += f'\nwith {", ".join(input_msgs)}'
            assert e.exc_tb is not None
            stack_trace = traceback.format_tb(e.exc_tb)
            if len(stack_trace) > 2:
                # append a stack trace if the exception happened in user code
                # (frame 0 is ExprEvaluator and frame 1 is some expr's eval()
                nl = '\n'
                # [-1:0:-1]: leave out entry 0 and reverse order, so that the most recent frame is at the top
                msg += f'\nStack:\n{nl.join(stack_trace[-1:1:-1])}'
            raise exc.Error(msg)
        except sql.exc.DBAPIError as e:
            raise exc.Error(f'Error during SQL execution:\n{e}')

        col_names = [expr.display_name() for expr in self.select_list]
        # replace ''
        col_names = [n if n != '' else f'col_{i}' for i, n in enumerate(col_names)]
        return DataFrameResultSet(data_rows, col_names, [expr.col_type for expr in self.select_list])

    def count(self) -> int:
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
            raise exc.RuntimeError(f'categoricals_map() can only be applied to an individual string column')
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
                raise exc.RuntimeError(f'[] for column selection is only allowed once')
            # analyze select list; wrap literals with the corresponding expressions and update it in place
            for i in range(len(index)):
                expr = index[i]
                if isinstance(expr, dict):
                    index[i] = expr = exprs.InlineDict(expr)
                if isinstance(expr, list):
                    index[i] = expr = exprs.InlineArray(tuple(expr))
                if not isinstance(expr, exprs.Expr):
                    raise exc.RuntimeError(f'Invalid expression in []: {expr}')
                if expr.col_type.is_invalid_type():
                    raise exc.RuntimeError(f'Invalid type: {expr}')
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
        # we append pk_cols so that the already-computed sql row indices remain correct
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
