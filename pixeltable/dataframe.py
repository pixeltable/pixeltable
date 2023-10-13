from __future__ import annotations
import base64
import io
import os
from typing import List, Optional, Any, Dict, Generator, Tuple
from pathlib import Path
import pandas as pd
from  pandas.io.formats.style import Styler
import sqlalchemy as sql
from PIL import Image
import traceback
import copy

from pixeltable import catalog
from pixeltable.env import Env
from pixeltable.type_system import ColumnType
from pixeltable import exprs
from pixeltable import exceptions as exc
from pixeltable.plan import Planner

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
    def __init__(self, rows: List[List[Any]], col_names: List[str], col_types: List[ColumnType]):
        self.rows = rows
        self.col_names = col_names
        self.col_types = col_types

    def __len__(self) -> int:
        return len(self.rows)

    def column_names(self) -> List[str]:
        return self.col_names

    def column_types(self) -> List[ColumnType]:
        return self.col_types

    def __repr__(self):
        return self.to_pandas().__repr__()

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
        if not isinstance(index, tuple) or len(index) != 2 \
                or not isinstance(index[0], int) or not isinstance(index[1], int):
            raise exc.Error(f'Bad index, expected tuple (<row idx>, <col idx>): {index}')
        return self.rows[index[0]][index[1]]

    def __eq__(self, other):
        if not isinstance(other, DataFrameResultSet):
            return False
        return self.to_pandas().equals(other.to_pandas())

class AnalysisInfo:
    def __init__(self, tbl: catalog.TableVersion):
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
        self.has_frame_col: bool = False  # True if we're referencing the frame col

        self.evaluator: Optional[exprs.Evaluator] = None
        self.sql_scan_eval_ctx: List[exprs.Expr] = []  # needed to materialize output of SQL scan stage
        self.agg_eval_ctx: List[exprs.Expr] = []  # needed to materialize output of agg stage
        self.filter_eval_ctx: List[exprs.Expr] = []
        self.group_by_eval_ctx: List[exprs.Expr] = []

    def finalize_exec(self) -> None:
        """
        Call release() on all collected Exprs.
        """
        exprs.Expr.release_list(self.sql_scan_output_exprs)
        exprs.Expr.release_list(self.agg_output_exprs)
        if self.filter is not None:
            self.filter.release()

def _expand_select_list(tbl : catalog.TableVersion, 
                        select_list : Optional[List[Tuple[exprs.Expr, Optional[str]]]],
                        ) -> List[Tuple[exprs.Expr, str]]:
    """
    Return the expanded select list, i.e. the select list with all columns expanded.
    """
    if select_list is None:
        expanded_list = []
        for col in tbl.columns():
            if tbl.is_frame_col(col):
                expanded_list.append((exprs.FrameColumnRef(col), None))
            else:
                expanded_list.append((exprs.ColumnRef(col), None))
    else:
        expanded_list = select_list
    out = []
    for i, tup in enumerate(expanded_list):
        (expr, alias) = tup
        if alias is not None:
            out.append((expr, alias)) # user provided alias
        else:
            name = expr.display_name()
            if name != "":
                out.append((expr, name)) # inferred name eg. from column
            else:
                out.append((expr, f"col_{i}")) # default name.
    return out

def _select_list_check_rep(select_list: Optional[List[Tuple[exprs.Expr, Optional[str]]]],
                           ) -> None:
    """ validate basic select list types. 
    """
    if select_list is not None: # basic check for valid select list
        assert len(select_list) > 0
        for ent in select_list:
            assert isinstance(ent, tuple)
            assert len(ent) == 2
            assert isinstance(ent[0], exprs.Expr)
            assert ent[1] is None or isinstance(ent[1], str)
            if isinstance(ent[1], str):
                assert ent[1].isidentifier()

class DataFrame:
    def __init__(
            self, tbl: catalog.TableVersion,
            select_list:  Optional[List[Tuple[exprs.Expr, Optional[str]]]] = None,
            where_clause: Optional[exprs.Predicate] = None,
            group_by_clause: Optional[List[exprs.Expr]] = None,
            order_by_clause: Optional[List[Tuple[exprs.Expr, bool]]] = None,  # List[(expr, asc)]
            limit: Optional[int] = None,):
        self.tbl = tbl
        _select_list_check_rep(select_list)
        self.select_list = copy.deepcopy(select_list)
        self.where_clause = copy.deepcopy(where_clause)
        self.group_by_clause = copy.deepcopy(group_by_clause)
        self.order_by_clause = copy.deepcopy(order_by_clause)
        self.limit_val = limit

    def _exec(self) -> Generator[exprs.DataRow, None, None]:
        """Returned value: list of select list values"""
        select_list = _expand_select_list(self.tbl, self.select_list)
        if self.group_by_clause is None:
            self.group_by_clause = []
        if self.order_by_clause is None:
            self.order_by_clause = []
        for item, _ in select_list:
            item.bind_rel_paths(None)
        select_list_exprs = [e for (e, _) in select_list]
        plan, select_list_tmp = Planner.create_query_plan(
            self.tbl, select_list_exprs, 
            where_clause=self.where_clause, 
            group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause,
            # limit_val == 0: no limit_val
            limit=self.limit_val if self.limit_val is not None else 0)
        plan.open()
        try:
            result = next(plan)
            for data_row in result:
                result_row = [data_row[e.slot_idx] for e in select_list_tmp]
                yield result_row
        finally:
            plan.close()
        return

    def show(self, n: int = 20) -> DataFrameResultSet:
        assert n is not None
        return self.limit(n).collect()

    def head(self, n: int = 20) -> DataFrameResultSet:
        return self.show(n)
    
    def _column_names(self):
        return [alias for (_, alias) in _expand_select_list(self.tbl, self.select_list)]

    def collect(self) -> DataFrameResultSet:
        try:
            data_rows = [row for row in self._exec()]
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

        expanded_select_list = _expand_select_list(self.tbl, self.select_list)
        col_names = [alias for (_, alias) in expanded_select_list]
        col_types = [expr.col_type for expr, _ in expanded_select_list]
        return DataFrameResultSet(data_rows, col_names, col_types)

    def count(self) -> int:
        from pixeltable.plan import Planner
        stmt = Planner.create_count_stmt(self.tbl, self.where_clause)
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
            or not isinstance(self.select_list[0][0], exprs.ColumnRef) \
            or not self.select_list[0][0].col_type.is_string_type():
            raise exc.Error(f'categoricals_map() can only be applied to an individual string column')
        assert isinstance(self.select_list[0][0], exprs.ColumnRef)
        col = self.select_list[0][0].col
        stmt = sql.select(sql.distinct(col.sa_col)) \
            .where(self.tbl.store_tbl.v_min_col <= self.tbl.version) \
            .where(self.tbl.store_tbl.v_max_col > self.tbl.version) \
            .order_by(col.sa_col)
        if self.where_clause is not None:
            sql_where_clause = self.where_clause.sql_expr()
            assert sql_where_clause is not None
            stmt = stmt.where(sql_where_clause)
        with Env.get().engine.connect() as conn:
            result = {row._data[0]: i for i, row in enumerate(conn.execute(stmt))}
            return result

    def _description(self) -> pd.DataFrame:
        """Return a description of this DataFrame as a pandas DataFrame.
        The DataFrame has two columns, heading and info, which list the contents of each 'component'
        (select list, where clause, ...) vertically when printed.
        """
        heading_vals: List[str] = []
        info_vals: List[str] = []
        if self.select_list is not None:
            assert len(self.select_list) > 0
            heading_vals.append('Select')
            heading_vals.extend([''] * (len(self.select_list) - 1))
            info_vals.extend(self._column_names())
        if self.where_clause is not None:
            heading_vals.append('Where')
            info_vals.append(self.where_clause.display_str(inline=False))
        if self.group_by_clause is not None:
            heading_vals.append('Group By')
            heading_vals.extend([''] * (len(self.group_by_clause) - 1))
            info_vals.extend([e.display_str(inline=False) for e in self.group_by_clause])
        if self.order_by_clause is not None:
            heading_vals.append('Order By')
            heading_vals.extend([''] * (len(self.order_by_clause) - 1))
            info_vals.extend([
                f'{e[0].display_str(inline=False)} {"asc" if e[1] else "desc"}' for e in self.order_by_clause
            ])
        if self.limit_val is not None:
            heading_vals.append('Limit')
            info_vals.append(str(self.limit_val))
        assert len(heading_vals) > 0
        assert len(info_vals) > 0
        assert len(heading_vals) == len(info_vals)
        return pd.DataFrame({'Heading': heading_vals, 'Info': info_vals})

    def _description_html(self) -> Styler:
        """Return the description in an ipython-friendly manner."""
        pd_df = self._description()
        # white-space: pre-wrap: print \n as newline
        # th: center-align headings
        return pd_df.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) \
            .hide(axis='index').hide(axis='columns')

    def describe(self) -> None:
        try:
            __IPYTHON__
            from IPython.display import display
            display(self._description_html())
        except NameError:
            print(self.__repr__())

    def __repr__(self) -> str:
        return self._description().to_string(header=False, index=False)

    def _repr_html_(self) -> str:
        return self._description_html()._repr_html_()

    def select(self, *items: exprs.Expr, **named_items : exprs.Expr) -> DataFrame:
        if self.select_list is not None:
            raise exc.Error(f'Select list already specified')
        for (alias, _) in named_items.items():
            if not isinstance(alias, str) or not alias.isidentifier():
                raise exc.Error(f'Invalid alias: {alias}')
        base_list = [(expr, None) for expr in items] + [(expr, k) for (k,expr) in named_items.items()]
        if len(base_list) == 0:
            raise exc.Error(f'Empty select list')
        # analyze select list; wrap literals with the corresponding expressions
        select_list = []
        for raw_expr, alias in base_list:
            if isinstance(raw_expr, exprs.Expr):
                select_list.append((raw_expr, alias))
            elif isinstance(raw_expr, dict):
                select_list.append((exprs.InlineDict(raw_expr), alias))
            elif isinstance(raw_expr, list):
                select_list.append((exprs.InlineArray(raw_expr), alias))
            else:
                select_list.append((exprs.Literal(raw_expr), alias))
            expr = select_list[-1][0]
            if expr.col_type.is_invalid_type():
                raise exc.Error(f'Invalid type: {raw_expr}')
            # TODO: check that ColumnRefs in expr refer to self.tbl
        # check user provided aliases would not conflict among themselves 
        # or with auto-generated ones
        seen = set() 
        for (_, alias) in _expand_select_list(self.tbl, select_list):
            if alias in seen:
                raise exc.Error(f'Repeated alias: {alias}')
            seen.add(alias)
        return DataFrame(
            self.tbl, select_list=select_list,
            where_clause=self.where_clause, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause, limit=self.limit_val)

    def where(self, pred: exprs.Predicate) -> DataFrame:
        return DataFrame(
            self.tbl, select_list=self.select_list,
            where_clause=pred, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause, limit=self.limit_val)

    def group_by(self, *expr_list: exprs.Expr) -> DataFrame:
        if self.group_by_clause is not None:
            raise exc.Error(f'Group-by already specified')
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise exc.Error(f'Invalid expression in group_by(): {e}')
        self.group_by_clause = [e.copy() for e in expr_list]
        return DataFrame(
            self.tbl, select_list=self.select_list,
            where_clause=self.where_clause, group_by_clause=expr_list,
            order_by_clause=self.order_by_clause, limit=self.limit_val)

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> DataFrame:
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise exc.Error(f'Invalid expression in order_by(): {e}')
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend([(e.copy(), asc) for e in expr_list])
        return DataFrame(
            self.tbl, select_list=self.select_list,
            where_clause=self.where_clause, group_by_clause=self.group_by_clause,
            order_by_clause=order_by_clause, limit=self.limit_val)

    def limit(self, n: int) -> DataFrame:
        assert n is not None and isinstance(n, int)
        return DataFrame(
            self.tbl, select_list=self.select_list,
            where_clause=self.where_clause, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause, limit=n)

    def __getitem__(self, index: object) -> DataFrame:
        """
        Allowed:
        - [<Predicate>]: filter operation
        - [List[Expr]]/[Tuple[Expr]]: setting the select list
        - [Expr]: setting a single-col select list
        """
        if isinstance(index, exprs.Predicate):
            return self.where(index)
        if isinstance(index, tuple):
            index = list(index)
        if isinstance(index, exprs.Expr):
            index = [index]
        if isinstance(index, list):
            return self.select(*index)
        raise TypeError(f'Invalid index type: {type(index)}')
