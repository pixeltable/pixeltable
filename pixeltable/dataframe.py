import base64
from io import BytesIO
from typing import List, Optional, Any, Tuple
import pandas as pd
import sqlalchemy as sql
from PIL import Image

from pixeltable import catalog, store
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

    def _repr_html_(self) -> str:
        img_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type == ColumnType.IMAGE]
        formatters = {self.col_names[i]: _format_img for i in img_col_idxs}
        # escape=False: make sure <img> tags stay intact
        # TODO: why does mypy complain about formatters having an incorrect type?
        return self._create_df().to_html(formatters=formatters, escape=False)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self._create_df().to_string()

    def _create_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)


class EvalCtx:
    """
    Represents the parameters necessary to materialize List[Expr] from a sql query result row
    into a data row.

    Data row:
    - List[Any]
    - contains slots for all *materialized* exprs (ie, not for predicates that turn into the SQL Where clause):
    a) every DataFrame.select_list expr; those occupy the first len(select_list) slots
    b) the parts of the where clause predicate that cannot be evaluated in SQL
    b) every child expr of a) and b), recursively
    - IMAGE columns are materialized immediately as a PIL.Image.Image

    ex.: the select list [<img col 1>.alpha_composite(<img col 2>), <text col 3>]
    - sql row composition: [<file path col 1>, <file path col 2>, <text col 3>]
    - data row composition: [Image, str, Image, Image]
    - copy_exprs: [
        ColumnRef(data_row_idx: 2, sql_row_idx: 0, col: <col 1>)
        ColumnRef(data_row_idx: 3, sql_row_idx: 1, col: <col 2>)
        ColumnRef(data_row_idx: 1, sql_row_idx: 2, col: <col 3>)
      ]
    - eval_exprs: [ImageMethodCall(data_row_idx: 0, sql_row_id: -1)]
    """

    def __init__(self, select_list: List[exprs.Expr], where_clause: Optional[exprs.Predicate]):
        """
        Init for list of materialized exprs
        """

        # exprs needed to materialize the SQL result row
        self.sql_exprs: List[sql.sql.expression.ClauseElement] = []
        # TODO: add self.literal_exprs so that we don't need to retrieve those from SQL
        # exprs that are materialized directly via SQL query and for which results can be copied from sql row
        # into data row
        self.filter_copy_exprs: List[exprs.Expr] = []
        self.select_copy_exprs: List[exprs.Expr] = []
        # exprs for which we need to call eval() to compute the value; must be called in the order stored here
        self.filter_eval_exprs: List[exprs.Expr] = []
        self.select_eval_exprs: List[exprs.Expr] = []

        # we want to avoid duplicate expr evaluation, so we keep track of unique exprs (duplicates share the
        # same data_row_idx); however, __eq__() doesn't work for sets, so we use a list here
        self.unique_exprs: List[exprs.Expr] = []

        self.next_data_row_idx = 0
        # analyze where_clause first, so that it can be evaluated before the select list
        if where_clause is not None:
            self._analyze_expr(where_clause, self.filter_copy_exprs, self.filter_eval_exprs)
        for expr in select_list:
            self._analyze_expr(expr, self.select_copy_exprs, self.select_eval_exprs)

    def num_materialized(self) -> int:
        return self.next_data_row_idx

    def _is_unique_expr(self, expr: exprs.Expr) -> bool:
        """
        If False, sets expr.data_row_idx to that of the already-recorded duplicate.
        """
        try:
            existing = next(e for e in self.unique_exprs if e.equals(expr))
            expr.data_row_idx = existing.data_row_idx
            return False
        except StopIteration:
            return True

    def _analyze_expr(self, expr: exprs.Expr, copy_exprs: List[exprs.Expr], eval_exprs: List[exprs.Expr]) -> None:
        """
        Assign Expr.data_row_idx and Expr.sql_row_idx and update sql/copy/eval_exprs accordingly.
        """
        if not self._is_unique_expr(expr):
            # nothing left to do
            return
        self.unique_exprs.append(expr)

        sql_expr = expr.sql_expr()
        if sql_expr is not None:
            if expr.data_row_idx < 0:
                expr.data_row_idx = self.next_data_row_idx
                self.next_data_row_idx += 1
            expr.sql_row_idx = len(self.sql_exprs)
            self.sql_exprs.append(sql_expr)
            copy_exprs.append(expr)
            return

        # expr value needs to be computed via Expr.eval()
        child_exprs = expr.child_exprs()
        # analyze children before expr, to make sure they are eval()'d first
        for child_expr in child_exprs:
            self._analyze_expr(child_expr, copy_exprs, eval_exprs)
        if expr.data_row_idx < 0:
            expr.data_row_idx = self.next_data_row_idx
            self.next_data_row_idx += 1
        eval_exprs.append(expr)


class DataFrame:
    def __init__(
            self, tbl: catalog.Table,
            select_list: Optional[List[exprs.Expr]] = None,
            where_clause: Optional[exprs.Predicate] = None):
        self.tbl = tbl
        self.select_list = select_list  # None: implies all cols
        self.where_clause = where_clause

    def _analyze_select_list(self) -> EvalCtx:
        if self.select_list is None:
            self.select_list = [exprs.ColumnRef(col) for col in self.tbl.columns()]
        return EvalCtx(self.select_list)

    def _copy_to_data_row(self, exprs: List[exprs.Expr], sql_row: Tuple[Any], data_row: List[Any]):
        """
        Copy expr values from sql to data row.
        """
        for expr in exprs:
            if expr.col_type == ColumnType.IMAGE:
                # row contains a file path that we need to open
                file_path = sql_row[expr.sql_row_idx]
                try:
                    img = Image.open(file_path)
                    img.thumbnail((128, 128))
                    data_row[expr.data_row_idx] = img
                except:
                    raise exc.OperationalError(f'Error reading image file: {file_path}')
            else:
                data_row[expr.data_row_idx] = sql_row[expr.sql_row_idx]

    def show(self, n: int = 20) -> DataFrameResultSet:
        sql_where_clause: Optional[sql.sql.expression.ClauseElement] = None
        remaining_where_clause: Optional[exprs.Predicate] = None
        if self.where_clause is not None:
            sql_where_clause, remaining_where_clause = self.where_clause.extract_sql_predicate()

        select_list = self.select_list
        if select_list is None:
            select_list = [exprs.ColumnRef(col) for col in self.tbl.columns()]
        eval_ctx = EvalCtx(select_list, remaining_where_clause)
        num_items = len(select_list)
        # we materialize everything needed for select_list into data_rows
        data_rows: List[List] = []

        with store.engine.connect() as conn:
            stmt = self._create_select_stmt(eval_ctx.sql_exprs, sql_where_clause)
            num_rows = 0

            for row in conn.execute(stmt):
                data_row: List[Any] = [None] * eval_ctx.num_materialized()

                if remaining_where_clause is not None:
                    # we need to evaluate the remaining filter predicate first
                    self._copy_to_data_row(eval_ctx.filter_copy_exprs, row._data, data_row)
                    for expr in eval_ctx.filter_eval_exprs:
                        expr.eval(data_row)
                    if data_row[remaining_where_clause.data_row_idx] == False:
                        continue

                # materialize the select list
                self._copy_to_data_row(eval_ctx.select_copy_exprs, row._data, data_row)
                for expr in eval_ctx.select_eval_exprs:
                    expr.eval(data_row)

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
        stmt = sql.select(sql.func.count('*')).select_from(self.tbl.sa_tbl) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if self.where_clause is not None:
            sql_where_clause = self.where_clause.sql_expr()
            assert sql_where_clause is not None
            stmt = stmt.where(sql_where_clause)
        with store.engine.connect() as conn:
            result: int = conn.execute(stmt).scalar_one()
            assert isinstance(result, int)
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
            for expr in index:
                if not isinstance(expr, exprs.Expr):
                    raise exc.OperationalError(f'Invalid expression in []: {expr}')
            return DataFrame(self.tbl, select_list=index, where_clause=self.where_clause)
        raise TypeError(f'Invalid index type: {type(index)}')

    def _create_select_stmt(
            self, select_list: List[sql.sql.expression.ClauseElement],
            where_clause: Optional[sql.sql.expression.ClauseElement]) -> sql.sql.expression.Select:
        stmt = sql.select(*select_list) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        return stmt
