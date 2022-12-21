import base64
from io import BytesIO
from typing import List, Optional, Any, Dict
import pandas as pd
import sqlalchemy as sql
from PIL import Image

from pixeltable import catalog, env
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
    with BytesIO() as buffer:
        img.save(buffer, 'jpeg')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{img_base64}">'


class DataFrameResultSet:
    def __init__(self, rows: List[List], col_names: List[str], col_types: List[ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def __len__(self) -> int:
        return len(self.rows)

    def _repr_html_(self) -> str:
        img_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type.is_image_type()]
        formatters = {self.col_names[i]: _format_img for i in img_col_idxs}
        # escape=False: make sure <img> tags stay intact
        # TODO: why does mypy complain about formatters having an incorrect type?
        return self.to_pandas().to_html(formatters=formatters, escape=False)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.to_pandas().to_string()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, tuple):
            if len(index) != 2 or not isinstance(index[0], int) or not isinstance(index[1], int):
                raise exc.OperationalError(f'Bad index: {index}')
            return self.rows[index[0]][index[1]]


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
        self.eval_ctx: Optional[exprs.ExprEvalCtx] = None

    def show(self, n: int = 20) -> DataFrameResultSet:
        sql_where_clause: Optional[sql.sql.expression.ClauseElement] = None
        remaining_where_clause: Optional[exprs.Predicate] = None
        similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None
        if self.where_clause is not None:
            sql_where_clause, remaining_where_clause = self.where_clause.extract_sql_predicate()
            if remaining_where_clause is not None:
                similarity_clauses, remaining_where_clause = remaining_where_clause.split_conjuncts(
                    lambda e: isinstance(e, exprs.ImageSimilarityPredicate))
                if len(similarity_clauses) > 1:
                    raise exc.OperationalError(f'More than one nearest() or matches() not supported')
                if len(similarity_clauses) == 1:
                    if not self.tbl.is_indexed:
                        raise exc.OperationalError(f'nearest()/matches() not available for unindexed table')
                    similarity_clause = similarity_clauses[0]
                    if n > 100:
                        raise exc.OperationalError(f'nearest()/matches() requires show(n <= 100): n={n}')

        select_list = self.select_list
        if select_list is None:
            select_list = [exprs.ColumnRef(col) for col in self.tbl.columns()]
        for item in select_list:
            item.bind_rel_paths(None)
        if self.eval_ctx is None:
            # constructing the EvalCtx is not idempotent
            self.eval_ctx = exprs.ExprEvalCtx(select_list, remaining_where_clause)
        # we materialize everything needed for select_list into data_rows
        data_rows: List[List] = []

        idx_rowids: List[int] = []  # rowids returned by index lookup
        if similarity_clause is not None:
            # do index lookup
            assert similarity_clause.img_col.col.idx is not None
            embed = similarity_clause.embedding()
            idx_rowids = similarity_clause.img_col.col.idx.search(embed, n, self.tbl.valid_rowids)

        with env.get_engine().connect() as conn:
            stmt = self._create_select_stmt(self.eval_ctx.sql_exprs, sql_where_clause, idx_rowids)
            num_rows = 0
            evaluator = exprs.ExprEvaluator(select_list, remaining_where_clause)

            for row in conn.execute(stmt):
                data_row: List[Any] = [None] * self.eval_ctx.num_materialized()
                if not evaluator.eval(row._data, data_row):
                    continue

                # copy select list results into contiguous array
                # TODO: make this unnecessary
                result_row = [data_row[e.data_row_idx] for e in select_list]
                data_rows.append(result_row)
                num_rows += 1
                if n > 0 and num_rows == n:
                    break

        col_names = [expr.display_name() for expr in select_list]
        # replace ''
        col_names = [n if n != '' else f'col_{i}' for i, n in enumerate(col_names)]
        return DataFrameResultSet(data_rows, col_names, [expr.col_type for expr in select_list])

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
        with env.get_engine().connect() as conn:
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
        with env.get_engine().connect() as conn:
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

    def _create_select_stmt(
            self, select_list: List[sql.sql.expression.ClauseElement],
            where_clause: Optional[sql.sql.expression.ClauseElement],
            valid_rowids: List[int]) -> sql.sql.expression.Select:
        stmt = sql.select(*select_list) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        if len(valid_rowids) > 0:
            #stmt = stmt.where(sql.text(f'{str(self.tbl.rowid_col)} IN ({",".join(valid_rowids)})'))
            stmt = stmt.where(self.tbl.rowid_col.in_(valid_rowids))
        return stmt
