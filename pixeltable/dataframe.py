from __future__ import annotations
import base64
import io
import os
from typing import List, Optional, Any, Dict, Generator, Tuple
from pathlib import Path
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

import json
import hashlib
import pyarrow as pa

from pixeltable.utils.dirs import transactional_folder


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

    def to_pandas(self) -> 'pandas.DataFrame':
        import pandas as pd
        return pd.DataFrame.from_records(self.rows, columns=self.col_names)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, tuple):
            if len(index) != 2 or not isinstance(index[0], int) or not isinstance(index[1], int):
                raise exc.Error(f'Bad index: {index}')
            return self.rows[index[0]][index[1]]

    def __eq__(self, other):
        if not isinstance(other, DataFrameResultSet):
            return False
        return self.rows == other.rows and self.col_names == other.col_names and self.col_types == other.col_types


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


class DataFrame:
    def __init__(
            self, tbl: catalog.TableVersion,
            select_list: Optional[List[exprs.Expr]] = None,
            where_clause: Optional[exprs.Predicate] = None,
            group_by_clause: Optional[List[exprs.Expr]] = None,
            order_by_clause: Optional[List[Tuple[exprs.Expr, bool]]] = None):  # List[(expr, asc)]
        self.tbl = tbl
        # exprs contain execution state and therefore cannot be shared
        self.select_list = copy.deepcopy(select_list)  # None: implies all cols
        self._manual_column_names = {}
        self.where_clause = copy.deepcopy(where_clause)
        self.group_by_clause = copy.deepcopy(group_by_clause)
        self.order_by_clause = copy.deepcopy(order_by_clause)

    def exec(self, n: int = 20, image_format : str = 'pil') -> Generator[List, None, None]:
        """Returned value: list of select list values"""
        assert image_format in ['pil', 'bytes'], image_format
        self.select_list = self._get_select_list()
        columns = self.get_column_names()
        types = self.get_column_types()
        if self.group_by_clause is None:
            self.group_by_clause = []
        if self.order_by_clause is None:
            self.order_by_clause = []
        for item in self.select_list:
            item.bind_rel_paths(None)
        plan, self.select_list = Planner.create_query_plan(
            self.tbl, self.select_list, where_clause=self.where_clause, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause, limit=n)
        plan.open()
        try:
            result = next(plan)
            for data_row in result:
                result_row = []

                for (i, e) in enumerate(self.select_list):
                    val = data_row[e.slot_idx] 

                    # TODO: file urls
                    if types[columns[i]].is_image_type() and image_format == 'bytes':
                        if data_row.file_paths[e.slot_idx] is not None:
                            val = open(data_row.file_paths[e.slot_idx], 'rb').read()
                        elif isinstance(val, Image.Image):
                            # result of image transform without column
                            buf = io.BytesIO()
                            val.save(buf, format='PNG')
                            val = buf.getvalue()
                        else:
                            assert False, f'unknown image type {type(val)}'

                        assert isinstance(val, bytes)

                    result_row.append(val)
                yield result_row
        finally:
            plan.close()
        return
    
    def set_column_names(self, names : Dict[int, str]):
        # TODO validate position within range
        self._manual_column_names.update(names)

    def get_column_names(self) -> List[str]:
        table_names = [expr.display_name() for expr in self._get_select_list()]
        col_names = [self._manual_column_names.get(i, n if n != '' else f'col_{i}') for i, n in enumerate(table_names)]
        return col_names
    
    def _get_select_list(self) -> List[exprs.Expr]:
        if self.select_list is None:
            return [
                exprs.FrameColumnRef(col) if self.tbl.is_frame_col(col) else exprs.ColumnRef(col)
                for col in self.tbl.columns()
            ]
        else:
            return self.select_list
    
    def get_column_types(self) -> Dict[str, ColumnType]:
        slist = self._get_select_list()
        return {n: e.col_type for n, e in zip(self.get_column_names(), slist)}
    
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

        col_names = self.get_column_names()
        return DataFrameResultSet(data_rows, col_names, [expr.col_type for expr in self.select_list])

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
            or not isinstance(self.select_list[0], exprs.ColumnRef) \
            or not self.select_list[0].col_type.is_string_type():
            raise exc.Error(f'categoricals_map() can only be applied to an individual string column')
        assert isinstance(self.select_list[0], exprs.ColumnRef)
        col = self.select_list[0].col
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

    def select(self, *items: exprs.Expr) -> DataFrame:
        if self.select_list is not None:
            raise exc.Error(f'Select list already specified')

        # analyze select list; wrap literals with the corresponding expressions and update it in place
        select_list = list(items)
        for i in range(len(select_list)):
            expr = items[i]
            if isinstance(expr, dict):
                select_list[i] = expr = exprs.InlineDict(expr)
            elif isinstance(expr, list):
                select_list[i] = expr = exprs.InlineArray(tuple(expr))
            elif not isinstance(expr, exprs.Expr):
                select_list[i] = expr = exprs.Literal(expr)
            if expr.col_type.is_invalid_type():
                raise exc.Error(f'Invalid type: {expr}')
            # TODO: check that ColumnRefs in expr refer to self.tbl
        return DataFrame(
            self.tbl, select_list=select_list, where_clause=self.where_clause, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause)

    def where(self, pred: exprs.Predicate) -> DataFrame:
        return DataFrame(
            self.tbl, select_list=self.select_list, where_clause=pred, group_by_clause=self.group_by_clause,
            order_by_clause=self.order_by_clause)

    def group_by(self, *expr_list: exprs.Expr) -> DataFrame:
        if self.group_by_clause is not None:
            raise exc.Error(f'Group-by already specified')
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise exc.Error(f'Invalid expression in group_by(): {e}')
        self.group_by_clause = [e.copy() for e in expr_list]
        return DataFrame(
            self.tbl, select_list=self.select_list, where_clause=self.where_clause, group_by_clause=expr_list,
            order_by_clause=self.order_by_clause)

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> DataFrame:
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise exc.Error(f'Invalid expression in order_by(): {e}')
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend([(e.copy(), asc) for e in expr_list])
        return DataFrame(
            self.tbl, select_list=self.select_list, where_clause=self.where_clause,
            group_by_clause=self.group_by_clause, order_by_clause=order_by_clause)

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
    
    def _json_key(self) -> Dict[str, Any]:
        ''' json information that fully reconstructs the values of the Dataframe.
            most important property: if two Dataframes have the same json_key, they hold the same data.
            It is okay if two Dataframes have different json_keys but hold the same data.
            
            NB: this is also a way to name a dataframe in a way someone else on the same system can reproduce, 
            not sure if this is a good idea though.
        '''
        d = {
            'type': 'DataFrame', ## TODO: how about tbl itself has a method to get the right json.
            'tbl_id': str(self.tbl.id),
            'tbl_version': self.tbl.version,
            'select_list': [e.as_dict() for e in self.select_list] if self.select_list is not None else None,
            'where_clause': self.where_clause.as_dict() if self.where_clause is not None else None
        }

        return json.loads(json.dumps(d)) # check that it is serializable

    def _hash_key(self) -> str:
        ''' A value dependent fixed length key used for caching.
        '''
        k = self._json_key()
        mstr = json.dumps(k, sort_keys=True)
        return hashlib.sha256(mstr.encode()).hexdigest()
    
    def _to_parquet(self, partition_size=2000) -> str:
        ''' Export the dataframe to parquet format.
            Uses cached version when available.
            TODO: partition size is a parameter here for testing.
            TODO: ideally we want to limit size of chunks to tens of MBs, which depends on data and.
            can probably use arrow batch to get estimates without.
        '''
        import pyarrow as pa
        import pyarrow.parquet as pq

        assert (self.count() / partition_size) < 10000 # partition names run out of digits currently

        name = self._hash_key()
        dest_path = (Env.get()._cache_dir / name).with_suffix('.parquet')
        if dest_path.exists() and dest_path.is_dir(): # already cached and folder fully formed.
            return str(dest_path)
        
        column_names = self.get_column_names()

        # store the changes atomically
        with transactional_folder(dest_path) as temp_path:
            # dump metadata json file so we can inspect what was the source of the parquet file later on.
            json.dump(self._json_key(), (temp_path / '.pixeltable.json').open('w'))

            batch_num = 0
            row_batch = []
            def flush(row_batch, batch_num):
                pydict = {column_names[i]: [row[i] for row in row_batch] for i in range(len(row_batch[0]))}
                tab = pa.Table.from_pydict(pydict)
                output_path =temp_path / f'part-{batch_num:04d}.parquet'
                pq.write_table(tab, output_path)


            for data_row in self.exec(n=None, image_format='bytes'):                    
                row_batch.append(data_row)
                if len(row_batch) == partition_size: # flush parquet chunk
                    flush(row_batch, batch_num)
                    batch_num += 1
                    row_batch = []

            flush(row_batch, batch_num)

        return str(dest_path)
    
    def to_pytorch_dataset(self, image_format : str = 'pt') -> 'torch.utils.data.IterableDataset':
        ''' return an object with the torch iterator interface
            image_format : 'np', 'pt'
        '''
        from pixeltable.utils.pytorch import PixeltablePytorchDataset
        pqpath = Path(self._to_parquet())
        import os
        import pyarrow.parquet as pq

        def _get_part_metadata(pqpath):
            files = sorted([f for f in os.listdir(pqpath) if f.endswith('.parquet')])
            rows_per_file = {}

            for file in files:
                file_path = os.path.join(pqpath, file)
                parquet_file = pq.ParquetFile(file_path)
                rows_per_file[file] = parquet_file.metadata.num_rows

            return [(file, num_rows) for file, num_rows in rows_per_file.items()]

        part_metadata = _get_part_metadata(pqpath)
        return PixeltablePytorchDataset(column_types=self.get_column_types(), parquet_path=pqpath, image_format=image_format, 
                                        part_metadata=part_metadata)
