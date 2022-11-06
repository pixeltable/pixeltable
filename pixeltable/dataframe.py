import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Tuple, List, Any, Optional
import pandas as pd
import sqlalchemy as sql
from PIL import Image

from pixeltable import catalog, store
from pixeltable import type_system as pt_types
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
    def __init__(self, rows: List[List], col_names: List[str], col_types: List[pt_types.ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def _repr_html_(self) -> str:
        img_col_idxs = [i for i, col_type in enumerate(self.col_types) if col_type == pt_types.ColumnType.IMAGE]
        formatters = {self.col_names[i]: _format_img for i in img_col_idxs}
        # escape=False: make sure <img> tags stay intact
        # TODO: why does mypy complain about formatters having an incorrect type?
        return self._create_df().to_html(formatters=formatters, escape=False)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self._create_df().to_string()

    def _create_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)


@dataclass
class SelectListInfo:
    """
    Represents the parameters necessary to materialize DataFrame.select_list from a sql query result row
    into a data row.
    Composition of data row:
    - type: List[Any]
    - first len(DataFrame.select_list) elements contain values for select list items
      (an IMAGE column is materialized as a PIL.Image.Image)
    - remaining elements contain required intermediate values

    ex.: the select list [<img col 1>.alpha_composite(<img col 2>), <text col 3>] requires 4 slots in each data row:
    - data row composition: [Image, str, Image, Image]
    - sql_select_list = [<file path col 1>, <file path col 2>, <text col 3>]
    - copy_idxs == [(2, 1)]: str can be copied as-is
    - src_img_idxs == [(0, 2), (1, 3)]: these are intermediate results needed for the alpha_composite() call
    - eval_params == [(FunctionCallExpr, 0, [2, 3])]
    """
    # select list run against the stored table
    sql_select_list: List[sql.sql.expression.ClauseElement] = field(default_factory=list)
    # copied directly into materialized data; Tuple[src_idx, dst_idx]
    copy_idxs: List[Tuple[int, int]] = field(default_factory=list)
    # indices into result row; contain image paths that get materialized as Images via Image.open()
    src_img_idxs: List[Tuple[int, int]] = field(default_factory=list)
    # select list items that require eval() to materialize
    # 1. what to evaluate
    # 2. the destination index (into the data row)
    # 3. the indices into the sql result row that are passed as args to eval()
    eval_params: List[Tuple[exprs.Expr, int, List[int]]] = field(default_factory=list)


class DataFrame:
    def __init__(
            self, tbl: catalog.Table,
            select_list: Optional[List[exprs.Expr]] = None,
            where_clause: Optional[exprs.Predicate] = None):
        self.tbl = tbl
        self.select_list = select_list  # None: implies all cols
        self.where_clause = where_clause

    def _analyze_select_list(self) -> SelectListInfo:
        if self.select_list is None:
            self.select_list = [exprs.ColumnRef(col) for col in self.tbl.columns()]

        result = SelectListInfo()
        num_items = len(self.select_list)
        num_intermediate_vals = 0  # those not needed in the select list
        for i, expr in enumerate(self.select_list):
            eval_info = expr.get_eval_info()
            if isinstance(eval_info, sql.sql.expression.ClauseElement):
                result.sql_select_list.append(eval_info)
                if expr.col_type == pt_types.ColumnType.IMAGE:
                    result.src_img_idxs.append((len(result.sql_select_list) - 1, i))
                else:
                    result.copy_idxs.append((len(result.sql_select_list) - 1, i))

            if isinstance(eval_info, list):
                # we need to materialize the referenced cols in the data row
                for i, col_ref in enumerate(eval_info):
                    assert isinstance(col_ref, exprs.ColumnRef)
                    result.sql_select_list.append(col_ref.get_eval_info())
                    if col_ref.col_type == pt_types.ColumnType.IMAGE:
                        result.src_img_idxs.append((len(result.sql_select_list) - 1, num_items + i))
                    else:
                        result.copy_idxs.append((len(result.sql_select_list) - 1, num_items + i))
                eval_params = (expr, i, [i + num_items for i in range(len(eval_info))])
                result.eval_params.append(eval_params)

        return result

    def show(self, n: int = 20) -> DataFrameResultSet:
        sel_list_info = self._analyze_select_list()
        num_materialized_vals = \
            len(sel_list_info.copy_idxs) + len(sel_list_info.src_img_idxs) + len(sel_list_info.eval_params)
        num_items = len(self.select_list)
        # we materialize everything needed for select_list into data_rows
        data_rows: List[List] = []

        with store.engine.connect() as conn:
            stmt = self._create_select_stmt(sel_list_info.sql_select_list)
            num_rows = 0

            for row in conn.execute(stmt):
                data_row = [0] * num_materialized_vals

                # slots we simply copy
                for src, dst in sel_list_info.copy_idxs:
                    # TODO: using row._data here: is there a cleaner way to get the data?
                    data_row[dst] = row._data[src]

                # slots for image paths that we need to open
                for src, dst in sel_list_info.src_img_idxs:
                    file_path = row._data[src]
                    try:
                        img = Image.open(file_path)
                        img.thumbnail((128, 128))
                    except:
                        return f'Error reading image file: {file_path}'
                    data_row[dst] = img

                # select list items that require evaluation
                for expr, dst, arg_idxs in sel_list_info.eval_params:
                    args = [data_row[i] for i in arg_idxs]
                    val = expr.eval(*args)
                    data_row[dst] = val

                data_rows.append(data_row[:num_items])  # get rid of intermediate values we don't need for the result
                num_rows += 1
                if n > 0 and num_rows == n:
                    break

        # TODO: col names
        return DataFrameResultSet(
            data_rows, [f'col{i}' for i in range(len(self.select_list))], [expr.col_type for expr in self.select_list])

    def count(self) -> int:
        stmt = sql.select(sql.func.count('*')).select_from(self.tbl.sa_tbl) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        with store.engine.connect() as conn:
            result: int = conn.execute(stmt).scalar_one()
            assert isinstance(result, int)
            return result

    def __getitem__(self, index: object) -> 'DataFrame':
        """
        Allowed:
        - [<Predicate>]: filter operation
        - [List[Expr]]: setting the select list
        - [Expr]: setting a single-col select list
        """
        if isinstance(index, exprs.Predicate):
            return DataFrame(self.tbl, select_list=self.select_list, where_clause=index)
        if isinstance(index, exprs.ColumnRef):
            index = [index]
        if isinstance(index, list):
            if self.select_list is not None:
                raise exc.OperationalError(f'[] for column selection is only allowed once')
            for expr in index:
                if not isinstance(expr, exprs.Expr):
                    raise exc.OperationalError(f'Invalid expression in []: {expr}')
            return DataFrame(self.tbl, select_list=index, where_clause=self.where_clause)
        raise TypeError(f'Invalid index type: {type(index)}')

    def _create_select_stmt(self, select_list: List[sql.sql.expression.ClauseElement]) -> sql.sql.expression.Select:
        stmt = sql.select(*select_list) \
            .where(self.tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.v_max_col > self.tbl.version)
        if self.where_clause is not None:
            stmt = stmt.where(self.where_clause.to_sql_expr())
        return stmt
