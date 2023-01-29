from typing import Optional, List, Set, Dict, Any, Type, Union, Callable
import re
import inspect
import io
import os
import dataclasses

import PIL, cv2
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
import pathlib

import pandas as pd
import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable import store
from pixeltable.env import Env
from pixeltable import exceptions as exc
from pixeltable.type_system import ColumnType
from pixeltable.utils import clip, video
from pixeltable import utils
from pixeltable.index import VectorIndex
from pixeltable.function import Function, FunctionRegistry


_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


class Column:
    def __init__(
            self, name: str, col_type: Optional[ColumnType] = None,
            computed_with: Optional[Union['Expr', Callable]] = None,
            primary_key: bool = False, nullable: bool = True, col_id: Optional[int] = None,
            value_expr_str: Optional[str] = None, indexed: bool = False):
        """
        Computed columns: those have a non-None computed_with argument
        - when constructed by the user: 'computed_with' was constructed explicitly and is passed in;
          'value_expr_str' is None and col_type is None
        - when loaded from store: 'value_expr_str' is the serialized form and col_type is set;
          'computed_with' is None
        Computed_with is a Callable:
        - the callable's parameter names must correspond to existing columns in the table for which this Column
          is being used
        - col_type needs to be set to the callable's return type

        indexed: only valid for image columns; if true, maintains an NN index for this column
        """
        from pixeltable import exprs
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{name}'")
        self.name = name
        if col_type is None and computed_with is None:
            raise exc.Error(f'Column {name}: col_type is required if computed_with is not specified')
        assert not(value_expr_str is not None and computed_with is not None)

        self.value_expr: Optional['Expr'] = None
        self.compute_func: Optional[Callable] = None
        if computed_with is not None:
            value_expr = exprs.Expr.from_object(computed_with)
            if value_expr is None:
                # computed_with needs to be a Callable
                if not isinstance(computed_with, Callable):
                    raise exc.Error(
                        f'Column {name}: computed_with needs to be either a Pixeltable expression or a Callable, '
                        f'but it is a {type(computed_with)}')
                if col_type is None:
                    raise exc.Error(f'Column {name}: col_type is required if computed_with is a Callable')
                # we need to turn the computed_with function into an Expr, but this requires resolving
                # column name references and for that we need to wait until we're assigned to a Table
                self.compute_func = computed_with
            else:
                self.value_expr = value_expr.copy()
                self.col_type = self.value_expr.col_type

        if col_type is not None:
            self.col_type = col_type
        assert self.col_type is not None

        self.value_expr_str = value_expr_str  # stored here so it's easily accessible for the Table c'tor
        self.dependent_cols: List[Column] = []  # cols with value_exprs that reference us
        self.id = col_id
        self.primary_key = primary_key
        # computed cols are always nullable
        self.nullable = nullable or computed_with is not None or value_expr_str is not None
        self.sa_col: Optional[sql.schema.Column] = None

        if indexed and not self.col_type.is_image_type():
            raise exc.Error(f'Column {name}: indexed=True requires ImageType')
        self.is_indexed = indexed
        self.idx: Optional[VectorIndex] = None

    def to_sql(self) -> str:
        return f'{self.storage_name()} {self.col_type.to_sql()}'

    @property
    def is_computed(self) -> bool:
        return self.compute_func is not None or self.value_expr is not None

    def create_sa_col(self) -> None:
        """
        This needs to be recreated for every new table schema version.
        """
        self.sa_col = sql.Column(self.storage_name(), self.col_type.to_sa_type(), nullable=self.nullable)

    def set_idx(self, idx: VectorIndex) -> None:
        self.idx = idx

    def storage_name(self) -> str:
        assert self.id is not None
        return f'col_{self.id}'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return False
        if self.sa_col is None or other.sa_col is None:
            return False
        # if they point to the same table column, they're the same
        return str(self.sa_col) == str(other.sa_col)


# base class of all addressable objects within a Db
class SchemaObject:
    def __init__(self, obj_id: int):
        self.id = obj_id

    @classmethod
    def display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        assert False
        return ''


class DirBase(SchemaObject):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)

    @classmethod
    def display_name(cls) -> str:
        return 'directory'


# contains only MutableTables
class Dir(DirBase):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)


# contains only TableSnapshots
class SnapshotDir(DirBase):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)


class NamedFunction(SchemaObject):
    """
    Contains references to functions that are named and have a path within a db.
    The Function itself is stored in the FunctionRegistry.
    """
    def __init__(self, id: int, dir_id: int, name: str):
        super().__init__(id)
        self.dir_id = dir_id
        self.name = name


class Table(SchemaObject):
    #def __init__(self, tbl_record: store.Table, schema: List[Column]):
    def __init__(
            self, db_id: int, tbl_id: int, dir_id: int, name: str, version: int, cols: List[Column]):
        super().__init__(tbl_id)
        self.db_id = db_id
        self.dir_id = dir_id
        # TODO: this will be out-of-date after a rename()
        self.name = name
        for pos, col in enumerate(cols):
            if re.fullmatch(_ID_RE, col.name) is None:
                raise exc.BadFormatError(f"Invalid column name: '{col.name}'")
            assert col.id is not None
        self.cols = cols
        self.cols_by_name = {col.name: col for col in cols}
        self.cols_by_id = {col.id: col for col in cols}
        self.version = version

        # we can't call _load_valid_rowids() here because the storage table may not exist yet
        self.valid_rowids: Set[int] = set()

        # sqlalchemy-related metadata; used to insert and query the storage table
        self.sa_md = sql.MetaData()
        self._create_sa_tbl()
        self.is_dropped = False

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        for col in self.cols:
            if col.value_expr is not None or col.value_expr_str is not None:
                self._record_value_expr(col)

    def _record_value_expr(self, col: Column) -> None:
        """
        Update Column.dependent_cols for all cols referenced in col.value_expr.
        Creates col.value_expr if it doesn't exist yet.
        """
        from pixeltable.exprs import Expr, ColumnRef
        if col.value_expr is None:
            assert col.value_expr_str is not None
            col.value_expr = Expr.deserialize(col.value_expr_str, self)

        refd_col_ids = [e.col.id for e in col.value_expr.subexprs() if isinstance(e, ColumnRef)]
        refd_cols = [self.cols_by_id[id] for id in refd_col_ids]
        for refd_col in refd_cols:
            refd_col.dependent_cols.append(col)

    def _load_valid_rowids(self) -> None:
        if not any(col.col_type.is_image_type() for col in self.cols):
            return
        stmt = sql.select(self.rowid_col) \
            .where(self.v_min_col <= self.version) \
            .where(self.v_max_col > self.version)
        with Env.get().engine.begin() as conn:
            rows = conn.execute(stmt)
            for row in rows:
                rowid = row[0]
                self.valid_rowids.add(rowid)

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        if col_name not in self.cols_by_name:
            raise AttributeError(f'Column {col_name} unknown')
        col = self.cols_by_name[col_name]
        from pixeltable.exprs import ColumnRef
        return ColumnRef(col)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        if isinstance(index, str):
            # basically <tbl>.<colname>
            return self.__getattr__(index)
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self)

    def show(self, *args, **kwargs) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        return self.df().show(*args, **kwargs)

    def count(self) -> int:
        return self.df().count()

    @property
    def columns(self) -> List[Column]:
        return self.cols

    def storage_name(self) -> str:
        return f'tbl_{self.id}'

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.OperationalError('Table has been dropped')

    def _create_sa_tbl(self) -> None:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(store.Table.MAX_VERSION))
        sa_cols = [self.rowid_col, self.v_min_col, self.v_max_col]
        # re-create sql.Columns for each column, regardless of whether it already has sa_col set: it was bound
        # to the last sql.Table version we created and cannot be reused
        for col in self.cols:
            col.create_sa_col()
        sa_cols.extend([col.sa_col for col in self.cols])
        if hasattr(self, 'sa_tbl'):
            self.sa_md.remove(self.sa_tbl)
        self.sa_tbl = sql.Table(self.storage_name(), self.sa_md, *sa_cols)

    @classmethod
    def _vector_idx_name(cls, tbl_id: int, col: Column) -> str:
        return f'{tbl_id}_{col.id}'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def load_cols(cls, tbl_id: int, schema_version: int, session: orm.Session) -> List[Column]:
        """
        Returns loaded cols.
        """
        col_records = session.query(store.SchemaColumn) \
            .where(store.SchemaColumn.tbl_id == tbl_id) \
            .where(store.SchemaColumn.schema_version == schema_version) \
            .order_by(store.SchemaColumn.pos.asc()).all()
        cols = [
            Column(
                r.name, ColumnType.deserialize(r.col_type), primary_key=r.is_pk, nullable=r.is_nullable,
                col_id=r.col_id, value_expr_str=r.value_expr, indexed=r.is_indexed)
            for r in col_records
        ]
        for col in [col for col in cols if col.col_type.is_image_type()]:
            if col.is_indexed:
                col.set_idx(VectorIndex.load(cls._vector_idx_name(tbl_id, col), dim=512))
        return cols


class TableSnapshot(Table):
    def __init__(self, snapshot_record: store.TableSnapshot, cols: List[Column]):
        assert snapshot_record.db_id is not None
        assert snapshot_record.id is not None
        assert snapshot_record.dir_id is not None
        assert snapshot_record.name is not None
        assert snapshot_record.tbl_version is not None
        # the id of this SchemaObject is TableSnapshot.tbl_id, not TableSnapshot.id: we use tbl_id to construct
        # the name of the data table
        super().__init__(
            snapshot_record.db_id, snapshot_record.tbl_id, snapshot_record.dir_id, snapshot_record.name,
            snapshot_record.tbl_version, cols)
        self.snapshot_tbl_id = snapshot_record.id
        # it's safe to call _load_valid_rowids() here because the storage table already exists
        self._load_valid_rowids()

    def __repr__(self) -> str:
        return f'TableSnapshot(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table snapshot'

@dataclasses.dataclass
class TableParameters:
    # garbage-collect old versions beyond this point, unless they are referenced in a snapshot
    num_retained_versions: int

    # parameters for frame extraction
    frame_src_col: int  # column id
    frame_col: int  # column id
    frame_idx_col: int  # column id
    extraction_fps: int


class MutableTable(Table):
    def __init__(self, tbl_record: store.Table, schema_version: int, cols: List[Column]):
        assert tbl_record.db_id is not None
        assert tbl_record.id is not None
        assert tbl_record.dir_id is not None
        assert tbl_record.name is not None
        assert tbl_record.current_version is not None
        super().__init__(
            tbl_record.db_id, tbl_record.id, tbl_record.dir_id, tbl_record.name, tbl_record.current_version, cols)
        assert tbl_record.next_col_id is not None
        self.next_col_id = tbl_record.next_col_id
        assert tbl_record.next_row_id is not None
        self.next_row_id = tbl_record.next_row_id
        self.schema_version = schema_version
        self.parameters = TableParameters(**tbl_record.parameters)

    def __repr__(self) -> str:
        return f'MutableTable(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    def add_column(self, c: Column) -> None:
        self._check_is_dropped()
        if re.fullmatch(_ID_RE, c.name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{c.name}'")
        if c.name in self.cols_by_name:
            raise exc.DuplicateNameError(f'Column {c.name} already exists')
        assert self.next_col_id is not None
        c.id = self.next_col_id
        self.next_col_id += 1

        if c.compute_func is not None:
            # create value_expr from compute_func
            self._create_value_expr(c, self.cols_by_name)
        if c.value_expr is not None:
            self._record_value_expr(c)

        self.cols.append(c)
        self.cols_by_name[c.name] = c
        self.cols_by_id[c.id] = c

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version,
                        store.Table.next_col_id: self.next_col_id
                    })
                    .where(store.Table.id == self.id))
            conn.execute(
                sql.insert(store.TableSchemaVersion.__table__)
                    .values(
                        tbl_id=self.id, schema_version=self.schema_version,
                        preceding_schema_version=preceding_schema_version))
            conn.execute(
                sql.insert(store.StorageColumn.__table__)
                    .values(tbl_id=self.id, col_id=c.id, schema_version_add=self.schema_version))
            self._create_col_md(conn)
            stmt = f'ALTER TABLE {self.storage_name()} ADD COLUMN {c.to_sql()}'
            conn.execute(sql.text(stmt))
        self._create_sa_tbl()

        if not c.is_computed or self.count() == 0:
            return
        # backfill the existing rows
        from pixeltable.dataframe import DataFrame
        # use copy to avoid reusing existing execution state
        query = DataFrame(self, [c.value_expr.copy()])
        with Env.get().engine.begin() as conn:
            with tqdm(total=self.count()) as progress_bar:
                for result_row in query.exec(n=0, select_pk=True):
                    column_val, rowid, v_min = result_row
                    column_val = self._convert_to_stored(c, column_val, rowid)
                    conn.execute(
                        sql.update(self.sa_tbl)
                            .values({c.sa_col: column_val})
                            .where(self.rowid_col == rowid)
                            .where(self.v_min_col == v_min))
                    progress_bar.update(1)

    def drop_column(self, name: str) -> None:
        self._check_is_dropped()
        if name not in self.cols_by_name:
            raise exc.UnknownEntityError
        col = self.cols_by_name[name]
        if len(col.dependent_cols) > 0:
            raise exc.Error(
                f'Cannot drop column {name} because the following columns depend on it:\n',
                f'{", ".join([c.name for c in col.dependent_cols])}')
        if col.id == self.parameters.frame_col or col.id == self.parameters.frame_idx_col:
            src_col_name = self.cols_by_id[self.parameters.frame_src_col].name
            raise exc.Error(
                f'Cannot drop column {name} because it is used for frame extraction on column {src_col_name}')
        if col.id == self.parameters.frame_src_col:
            # we also need to reset the frame extraction table parameters
            self.parameters.frame_src_col = None
            self.parameters.frame_col = None
            self.parameters.frame_idx_col = None
            self.parameters.extraction_fps = None

        if col.value_expr is not None:
            # update Column.dependent_cols
            for c in self.cols:
                if c == col:
                    break
                try:
                    c.dependent_cols.remove(col)
                except ValueError:
                    # ignore
                    pass

        self.cols.remove(col)
        del self.cols_by_name[name]
        del self.cols_by_id[col.id]


        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.parameters: dataclasses.asdict(self.parameters),
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version
                    })
                    .where(store.Table.id == self.id))
            conn.execute(
                sql.insert(store.TableSchemaVersion.__table__)
                    .values(
                        tbl_id=self.id, schema_version=self.schema_version,
                        preceding_schema_version=preceding_schema_version))
            conn.execute(
                sql.update(store.StorageColumn.__table__)
                    .values({store.StorageColumn.schema_version_drop: self.schema_version})
                    .where(store.StorageColumn.tbl_id == self.id)
                    .where(store.StorageColumn.col_id == col.id))
            self._create_col_md(conn)
        self._create_sa_tbl()

    def rename_column(self, old_name: str, new_name: str) -> None:
        self._check_is_dropped()
        if old_name not in self.cols_by_name:
            raise exc.UnknownEntityError(f'Unknown column: {old_name}')
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise exc.DuplicateNameError(f'Column {new_name} already exists')
        col = self.cols_by_name[old_name]
        del self.cols_by_name[old_name]
        col.name = new_name
        self.cols_by_name[new_name] = col

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version
                    })
                    .where(store.Table.id == self.id))
            conn.execute(
                sql.insert(store.TableSchemaVersion.__table__)
                    .values(tbl_id=self.id, schema_version=self.schema_version,
                            preceding_schema_version=preceding_schema_version))
            self._create_col_md(conn)

    def _create_col_md(self, conn: sql.engine.base.Connection) -> None:
        for pos, c in enumerate(self.cols):
            value_expr_str = c.value_expr.serialize() if c.value_expr is not None else None
            conn.execute(
                sql.insert(store.SchemaColumn.__table__)
                .values(
                    tbl_id=self.id, schema_version=self.version, col_id=c.id, pos=pos, name=c.name,
                    col_type=c.col_type.serialize(), is_nullable=c.nullable, is_pk=c.primary_key,
                    value_expr=value_expr_str, is_indexed=c.is_indexed))

    def _convert_to_stored(self, col: Column, val: Any, rowid: int) -> Any:
        """
        Convert column value 'val' into a store-compatible format, if needed:
        - images are stored as files
        - arrays are stored as serialized ndarrays
        """
        if col.col_type.is_image_type():
            # replace PIL.Image.Image with file path
            img = val
            img_path = utils.get_computed_img_path(self.id, col.id, self.version, rowid)
            img.save(img_path)
            return str(img_path)
        elif col.col_type.is_array_type():
            # serialize numpy array
            np_array = val
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()
        else:
            return val

    def insert_rows(self, rows: List[List[Any]], columns: List[str] = []) -> None:
        """
        Insert rows into table. 'Columns' is a list of column names that specify the columns present in 'rows'.
        'Columns' == empty: all columns are present in 'rows'.
        """
        assert len(rows) > 0
        if len(rows[0]) != len(self.cols) and len(columns) == 0:
            raise exc.Error(
                f'Table {self.name} has {len(self.cols)} columns, but the data only contains {len(rows[0])} columns. '
                f"In this case, you need to specify the column names with the 'columns' parameter.")

        # make sure that each row contains the same number of values
        num_col_vals = len(rows[0])
        for i in range(1, len(rows)):
            if len(rows[i]) != num_col_vals:
                raise exc.Error(
                    f'Inconsistent number of column values in rows: row 0 has {len(rows[0])}, '
                    f'row {i} has {len(rows[i])}')

        if len(columns) == 0:
            columns = [c.name for c in self.cols]
        if len(rows[0]) != len(columns):
            raise exc.Error(
                f'The number of column values in rows ({len(rows[0])}) does not match the given number of column names '
                f'({len(columns)}')

        pd_df = pd.DataFrame.from_records(rows, columns=columns)
        self.insert_pandas(pd_df)

    def insert_pandas(self, data: pd.DataFrame) -> None:
        """
        If self.parameters.frame_src_col != None:
        - each row (containing a video) is expanded into one row per extracted frame (at the rate of the fps parameter)
        - parameters.frame_col is the image column that receives the extracted frame
        - parameters.frame_idx_col is the integer column that receives the frame index (starting at 0)
        """
        self._check_is_dropped()
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = {col.name for col in self.cols if not col.nullable and col.value_expr is None}
        if self.parameters.frame_src_col is not None:
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_col].name)
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_idx_col].name)
        given_col_names = set(data.columns)
        if not(reqd_col_names <= given_col_names):
            raise exc.InsertError(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')
        if not(given_col_names <= all_col_names):
            raise exc.InsertError(f'Unknown columns: {", ".join(given_col_names - all_col_names)}')
        computed_col_names = {col.name for col in self.cols if col.value_expr is not None}
        if self.parameters.frame_src_col is not None:
            computed_col_names.add(self.cols_by_id[self.parameters.frame_col].name)
            computed_col_names.add(self.cols_by_id[self.parameters.frame_idx_col].name)
        if len(computed_col_names & given_col_names) > 0:
            raise exc.InsertError(
                f'Provided values for computed columns: {", ".join(computed_col_names & given_col_names)}')

        # check types
        provided_cols = [self.cols_by_name[name] for name in data.columns]
        for col in provided_cols:
            if col.col_type.is_string_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires string data but contains {data.dtypes[col.name]}')
            if col.col_type.is_int_type() and not pd.api.types.is_integer_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires integer data but contains {data.dtypes[col.name]}')
            if col.col_type.is_float_type() and not pd.api.types.is_numeric_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires numerical data but contains {data.dtypes[col.name]}')
            if col.col_type.is_bool_type() and not pd.api.types.is_bool_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires boolean data but contains {data.dtypes[col.name]}')
            if col.col_type.is_timestamp_type() and not pd.api.types.is_datetime64_any_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires datetime data but contains {data.dtypes[col.name]}')
            if col.col_type.is_json_type() and not pd.api.types.is_object_dtype(data.dtypes[col.name]):
                raise exc.InsertError(
                    f'Column {col.name} requires dictionary data but contains {data.dtypes[col.name]}')
            if col.col_type.is_array_type() and not pd.api.types.is_object_dtype(data.dtypes[col.name]):
                raise exc.InsertError(
                    f'Column {col.name} requires array data but contains {data.dtypes[col.name]}')
            if col.col_type.is_image_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.InsertError(
                    f'Column {col.name} requires local file paths but contains {data.dtypes[col.name]}')
            if col.col_type.is_video_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.InsertError(
                    f'Column {col.name} requires local file paths but contains {data.dtypes[col.name]}')

        # check data
        data_cols = [self.cols_by_name[name] for name in data.columns]
        for col in data_cols:
            # image cols: make sure file path points to a valid image file
            if col.col_type.is_image_type():
                for _, path_str in data[col.name].items():
                    try:
                        _ = Image.open(path_str)
                    except FileNotFoundError:
                        raise exc.OperationalError(f'Column {col.name}: file does not exist: {path_str}')
                    except PIL.UnidentifiedImageError:
                        raise exc.OperationalError(f'Column {col.name}: not a valid image file: {path_str}')

            # image cols: make sure file path points to a valid image file; build index if col is indexed
            if col.col_type.is_video_type():
                for _, path_str in data[col.name].items():
                    cap = cv2.VideoCapture(path_str)
                    success = cap.isOpened()
                    cap.release()
                    if not success:
                        raise exc.Error(f'Column {col.name}: could not open video file {path_str}')

            if col.col_type.is_json_type():
                for idx, d in data[col.name].items():
                    if not isinstance(d, dict) and not isinstance(d, list):
                        raise exc.OperationalError(
                            f'Value for column {col.name} in row {idx} requires a dictionary or list: {d} ')

        # we're creating a new version
        self.version += 1

        # frame extraction from videos
        if self.parameters.frame_src_col is not None:
            video_col = self.cols_by_id[self.parameters.frame_src_col]
            frame_col = self.cols_by_id[self.parameters.frame_col]
            frame_idx_col = self.cols_by_id[self.parameters.frame_idx_col]

            # check data: video_column needs to contain valid file paths
            for idx, path_str in data[video_col.name].items():
                path = pathlib.Path(path_str)
                if not path.is_file():
                    raise exc.OperationalError(
                        f'For frame extraction, value for column {col.name} in row {idx} requires a valid '
                        f'file path: {path}')

            # expand each row in 'data' into one row per frame, adding columns frame_column and frame_idx_column
            expanded_rows: List[Dict] = []
            for input_row_idx, input_tuple in enumerate(data.itertuples(index=False)):
                input_row = input_tuple._asdict()
                path = input_row[video_col.name]
                # we need to generate a unique prefix for each set of frames corresponding to a single video
                frame_path_prefix = utils.get_extracted_frame_path(
                    self.id, video_col.id, self.version, self.next_row_id + input_row_idx)
                frame_paths = video.extract_frames(path, frame_path_prefix, self.parameters.extraction_fps)
                frame_rows = [
                    {frame_col.name: p, frame_idx_col.name: i, **input_row} for i, p in enumerate(frame_paths)
                ]
                expanded_rows.extend(frame_rows)
            data = pd.DataFrame.from_dict(expanded_rows, orient='columns')

        rowids = range(self.next_row_id, self.next_row_id + len(data))

        # update image indices
        data_cols = [self.cols_by_name[name] for name in data.columns]
        for col in [c for c in data_cols if c.is_indexed]:
            embeddings = np.zeros((len(data), 512))
            for i, (_, path_str) in enumerate(data[col.name].items()):
                try:
                    img = Image.open(path_str)
                    embeddings[i] = clip.encode_image(img)
                except FileNotFoundError:
                    raise exc.OperationalError(f'Column {col.name}: file does not exist: {path_str}')
                except PIL.UnidentifiedImageError:
                    raise exc.OperationalError(f'Column {col.name}: not a valid image file: {path_str}')
            assert col.idx is not None
            col.idx.insert(embeddings, np.array(rowids))

        # prepare state for computed cols
        from pixeltable import exprs
        eval_ctx: Optional[exprs.ComputedColEvalCtx] = None
        evaluator: Optional[exprs.ExprEvaluator] = None
        input_col_refs: List[exprs.ColumnRef] = []  # columns needed as input for computing value_exprs
        computed_cols = [col for col in self.cols if col.value_expr is not None]
        value_exprs: List[exprs.Expr] = []  # for computed_cols
        window_sort_exprs: List[exprs.Expr] = []
        if len(computed_cols) > 0:
            # create copies to avoid reusing past execution state; eval ctx and evaluator need to share these copies
            value_exprs = [c.value_expr.copy() for c in computed_cols]
            eval_ctx = exprs.ComputedColEvalCtx(
                [(exprs.ColumnRef(computed_cols[i]), value_exprs[i]) for i in range(len(computed_cols))])
            evaluator = exprs.ExprEvaluator(value_exprs, None, with_sql=False)
            input_col_refs = [
                e for e in evaluator.output_eval_exprs
                # we're looking for ColumnRefs to Columns that aren't themselves computed
                if isinstance(e, exprs.ColumnRef) and e.col.value_expr is None
            ]

            # determine order_by clause for window functions, if any
            window_fn_calls = [
                e for e in exprs.Expr.list_subexprs(value_exprs)
                if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
            ]
            window_sort_exprs = window_fn_calls[0].get_window_sort_exprs() if len(window_fn_calls) > 0 else []

        # construct new df with the storage column names, in order to iterate over it more easily
        stored_data = {col.storage_name(): data[col.name] for col in data_cols}
        stored_data_df = pd.DataFrame(data=stored_data)
        if len(window_sort_exprs) > 0:
            # need to sort data in order to compute windowed agg functions
            storage_col_names = [e.col.storage_name() for e in window_sort_exprs]
            stored_data_df.sort_values(storage_col_names, axis=0, inplace=True)
        insert_values: List[Dict[str, Any]] = []
        with tqdm(total=len(stored_data_df)) as progress_bar:
            for row_idx, row in enumerate(stored_data_df.itertuples(index=False)):
                row_dict = {'rowid': rowids[row_idx], 'v_min': self.version, **row._asdict()}

                if len(computed_cols) > 0:
                    # materialize computed column values
                    data_row = [None] * eval_ctx.num_materialized
                    # copy inputs
                    for col_ref in input_col_refs:
                        data_row[col_ref.data_row_idx] = row_dict[col_ref.col.storage_name()]
                        # load image, if this is a file path
                        if col_ref.col_type.is_image_type():
                            data_row[col_ref.data_row_idx] = PIL.Image.open(data_row[col_ref.data_row_idx])
                    evaluator.eval((), data_row)

                    # convert data values to storage format where necessary
                    for col_idx in range(len(computed_cols)):
                        val = data_row[value_exprs[col_idx].data_row_idx]
                        data_row[value_exprs[col_idx].data_row_idx] = \
                            self._convert_to_stored(computed_cols[col_idx], val, rowids[row_idx])

                    computed_vals_dict = {
                        computed_cols[i].storage_name(): data_row[value_exprs[i].data_row_idx]
                        for i in range(len(computed_cols))
                    }
                    row_dict.update(computed_vals_dict)

                insert_values.append(row_dict)
                progress_bar.update(1)

        with Env.get().engine.begin() as conn:
            conn.execute(sql.insert(self.sa_tbl), insert_values)
            self.next_row_id += len(data)
            conn.execute(
                sql.update(store.Table.__table__)
                    .values({store.Table.current_version: self.version, store.Table.next_row_id: self.next_row_id})
                    .where(store.Table.id == self.id))

        self.valid_rowids.update(rowids)

    def insert_csv(self, file_path: str) -> None:
        pass

    # TODO: update() signature?
    #def update(self, data: pd.DataFrame) -> None:

    # TODO: delete() signature?
    #def delete(self, data: DataFrame) -> None:

    def _delete_computed_imgs(self, version: int) -> None:
        """
        Delete image files computed for given version.
        """
        img_paths = utils.computed_imgs(tbl_id=self.id, version=version)
        for p in img_paths:
            os.remove(p)
        return

    def _delete_extracted_frames(self, version: int) -> None:
        """
        Delete extracted frames for given version.
        """
        frame_paths = utils.extracted_frames(tbl_id=self.id, version=version)
        for p in frame_paths:
            os.remove(p)

    def revert(self) -> None:
        self._check_is_dropped()
        if self.version == 0:
            raise exc.OperationalError('Cannot revert version 0')
        # check if the current version is referenced by a snapshot
        with orm.Session(Env.get().engine) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(store.TableSnapshot.id)) \
                .where(store.TableSnapshot.db_id == self.db_id) \
                .where(store.TableSnapshot.tbl_id == self.id) \
                .where(store.TableSnapshot.tbl_version == self.version) \
                .scalar()
            if num_references > 0:
                raise exc.OperationalError(
                    f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')

            conn = session.connection()
            # delete newly-added data
            self._delete_computed_imgs(self.version)
            self._delete_extracted_frames(self.version)
            conn.execute(sql.delete(self.sa_tbl).where(self.sa_tbl.c.v_min == self.version))
            # revert new deletions
            conn.execute(
                sql.update(self.sa_tbl).values({self.sa_tbl.c.v_max: store.Table.MAX_VERSION})
                    .where(self.sa_tbl.c.v_max == self.version))

            if self.version == self.schema_version:
                # the current version involved a schema change:
                # we need to determine the preceding schema version and reload the schema
                preceding_schema_version = session.query(store.TableSchemaVersion.preceding_schema_version) \
                    .where(store.TableSchemaVersion.tbl_id == self.id) \
                    .where(store.TableSchemaVersion.schema_version == self.schema_version) \
                    .scalar()
                self.cols = self.load_cols(self.id, preceding_schema_version, session)
                conn.execute(
                    sql.delete(store.TableSchemaVersion.__table__)
                        .where(store.TableSchemaVersion.tbl_id == self.id)
                        .where(store.TableSchemaVersion.schema_version == self.schema_version))
                self.schema_version = preceding_schema_version

            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version
                    })
                    .where(store.Table.id == self.id))

            session.commit()
        self.version -= 1

    # MODULE-LOCAL, NOT PUBLIC
    def rename(self, new_name: str) -> None:
        self._check_is_dropped()
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Table.__table__).values({store.Table.name: new_name})
                    .where(store.Table.id == self.id))

    # MODULE-LOCAL, NOT PUBLIC
    def drop(self) -> None:
        self._check_is_dropped()
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Table.__table__).values({store.Table.is_mutable: False})
                    .where(store.Table.id == self.id))

    @classmethod
    def _create_value_expr(cls, col: Column, existing_cols: Dict[str, Column]) -> None:
        """
        Create col.value_expr, given col.compute_func.
        Interprets compute_func's parameters to be references to columns and construct ColumnRefs as args.
        Does not update Column.dependent_cols.
        """
        assert col.value_expr is None
        assert col.compute_func is not None
        from pixeltable import exprs
        params = inspect.signature(col.compute_func).parameters
        args: List[exprs.ColumnRef] = []
        for param_name in params:
            if param_name not in existing_cols:
                raise exc.Error(
                    f'Column {col.name}: compute_with parameter refers to an unknown column: {param_name}')
            args.append(exprs.ColumnRef(existing_cols[param_name]))
        fn = Function(col.col_type, [arg.col_type for arg in args], eval_fn=col.compute_func)
        col.value_expr = exprs.FunctionCall(fn, args)

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
        cls, db_id: int, dir_id: int, name: str, cols: List[Column],
        num_retained_versions: int,
        extract_frames_from: Optional[str], extracted_frame_col: Optional[str], extracted_frame_idx_col: Optional[str],
        extracted_fps: Optional[int]
    ) -> 'MutableTable':
        # make sure col names are unique (within the table) and assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, c in enumerate(cols):
            if c.name in cols_by_name:
                raise exc.DuplicateNameError(f'Duplicate column: {c.name}')
            c.id = pos
            cols_by_name[c.name] = c

        # check frame extraction params, if present
        if extract_frames_from is not None:
            assert extracted_frame_col is not None and extracted_frame_idx_col is not None and extracted_fps is not None
            if extract_frames_from is not None and extract_frames_from not in cols_by_name:
                raise exc.BadFormatError(f'Unknown column in extract_frames_from: {extract_frames_from}')
            col_type = cols_by_name[extract_frames_from].col_type
            is_nullable = cols_by_name[extract_frames_from].nullable
            if not col_type.is_video_type():
                raise exc.BadFormatError(
                    f'extract_frames_from requires the name of a column of type video, but {extract_frames_from} has '
                    f'type {col_type}')
            if extracted_frame_col is not None and extracted_frame_col not in cols_by_name:
                raise exc.BadFormatError(f'Unknown column in extracted_frame_col: {extracted_frame_col}')
            col_type = cols_by_name[extracted_frame_col].col_type
            if not col_type.is_image_type():
                raise exc.BadFormatError(
                    f'extracted_frame_col requires the name of a column of type image, but {extracted_frame_col} has '
                    f'type {col_type}')
            # the src column determines whether the frame column is nullable
            cols_by_name[extracted_frame_col].nullable = is_nullable
            if extracted_frame_idx_col is not None and extracted_frame_idx_col not in cols_by_name:
                raise exc.BadFormatError(f'Unknown column in extracted_frame_idx_col: {extracted_frame_idx_col}')
            col_type = cols_by_name[extracted_frame_idx_col].col_type
            if not col_type.is_int_type():
                raise exc.BadFormatError(
                    f'extracted_frame_idx_col requires the name of a column of type int, but {extracted_frame_idx_col} '
                    f'has type {col_type}')
            # the src column determines whether the frame idx column is nullable
            cols_by_name[extracted_frame_idx_col].nullable = is_nullable

        params = TableParameters(
            num_retained_versions,
            cols_by_name[extract_frames_from].id if extract_frames_from is not None else None,
            cols_by_name[extracted_frame_col].id if extracted_frame_col is not None else None,
            cols_by_name[extracted_frame_idx_col].id if extracted_frame_idx_col is not None else None,
            extracted_fps)

        with orm.Session(Env.get().engine) as session:
            tbl_record = store.Table(
                db_id=db_id, dir_id=dir_id, name=name, parameters=dataclasses.asdict(params), current_version=0,
                current_schema_version=0, is_mutable=True, next_col_id=len(cols), next_row_id=0)
            session.add(tbl_record)
            session.flush()  # sets tbl_record.id

            tbl_version_record = store.TableSchemaVersion(
                tbl_id=tbl_record.id, schema_version=0, preceding_schema_version=0)
            session.add(tbl_version_record)
            session.flush()  # avoid FK violations in Postgres
            print(f'creating table {name}, id={tbl_record.id}')

            cols_by_name: Dict[str, Column] = {}  # records the cols we have seen so far
            for pos, col in enumerate(cols):
                session.add(store.StorageColumn(tbl_id=tbl_record.id, col_id=col.id, schema_version_add=0))
                session.flush()  # avoid FK violations in Postgres
                if col.value_expr is None and col.compute_func is not None:
                    cls._create_value_expr(col, cols_by_name)
                # Column.dependent_cols for existing cols is wrong at this point, but Table.init() will set it correctly
                value_expr_str = col.value_expr.serialize() if col.value_expr is not None else None
                session.add(
                    store.SchemaColumn(
                        tbl_id=tbl_record.id, schema_version=0, col_id=col.id, pos=pos, name=col.name,
                        col_type=col.col_type.serialize(), is_nullable=col.nullable, is_pk=col.primary_key,
                        value_expr=value_expr_str, is_indexed=col.is_indexed
                    )
                )
                session.flush()  # avoid FK violations in Postgres

                # for image cols, add VectorIndex for kNN search
                if col.is_indexed and col.col_type.is_image_type():
                    col.set_idx(VectorIndex.create(Table._vector_idx_name(tbl_record.id, col), 512))

                cols_by_name[col.name] = col
            session.flush()

            assert tbl_record.id is not None
            tbl = MutableTable(tbl_record, 0, cols)
            tbl.sa_md.create_all(bind=session.connection())
            session.commit()
            return tbl


class Path:
    def __init__(self, path: str, empty_is_valid: bool=False):
        if path == '' and not empty_is_valid or path != '' and re.fullmatch(_PATH_RE, path) is None:
            raise exc.BadFormatError(f"Invalid path format: '{path}'")
        self.components = path.split('.')

    @property
    def len(self) -> int:
        return 0 if self.is_root else len(self.components)

    @property
    def name(self) -> str:
        assert len(self.components) > 0
        return self.components[-1]

    @property
    def is_root(self) -> bool:
        return self.components[0] == ''

    @property
    def parent(self) -> 'Path':
        if len(self.components) == 1:
            if self.is_root:
                return self
            else:
                return Path('', empty_is_valid=True)
        else:
            return Path('.'.join(self.components[:-1]))

    def append(self, name: str) -> 'Path':
        if self.is_root:
            return Path(name)
        else:
            return Path(f'{str(self)}.{name}')

    def is_ancestor(self, other: 'Path', is_parent: bool = False) -> bool:
        """
        True if self as an ancestor path of other.
        """
        if self.len >= other.len or other.is_root:
            return False
        if self.is_root and (other.len == 1 or not is_parent):
            return True
        is_prefix = self.components == other.components[:self.len]
        return is_prefix and (self.len == (other.len - 1) or not is_parent)

    def __str__(self) -> str:
        return '.'.join(self.components)


class PathDict:
    def __init__(self) -> None:
        # *not* Dict[Path, SchemaObject]
        self.paths: Dict[str, SchemaObject] = {}  # all paths

    def __getitem__(self, path: Path) -> SchemaObject:
        return self.paths[str(path)]

    def __setitem__(self, path: Path, val: SchemaObject) -> None:
        self.paths[str(path)] = val

    def __delitem__(self, path: Path) -> None:
        del self.paths[str(path)]

    def update(self, paths: Dict[str, SchemaObject]) -> None:
        self.paths.update(paths)

    # checks that the parent of path exists and is a Dir
    # and that the object of path has 'expected' type
    def check_is_valid(
            self, path: Path, expected: Optional[Type[SchemaObject]],
            expected_parent_type: Type[DirBase] = DirBase) -> None:
        path_str = str(path)
        # check for existence
        if expected is not None:
            if path_str not in self.paths:
                raise exc.UnknownEntityError(path_str)
            obj = self.paths[path_str]
            if not isinstance(obj, expected):
                raise exc.UnknownEntityError(f'{path_str} needs to be a {expected.display_name()}')
        if expected is None and path_str in self.paths:
            raise exc.DuplicateNameError(f'{path_str} already exists')
        # check for containing directory
        parent_path = path.parent
        if str(parent_path) not in self.paths:
            raise exc.UnknownEntityError(f'Directory {str(parent_path)}')
        parent = self.paths[str(parent_path)]
        if not isinstance(parent, expected_parent_type):
            raise exc.UnknownEntityError(f'{str(parent_path)} needs to be a {expected_parent_type.display_name()}')

    def get(self, path_type: Type[SchemaObject]) -> List[Path]:
        return [obj for _, obj in self.paths.items() if isinstance(obj, path_type)]

    def get_children(self, parent: Path, child_type: Optional[Type[SchemaObject]], recursive: bool) -> List[Path]:
        candidates = [
            Path(path, empty_is_valid=True)
            for path, obj in self.paths.items() if child_type is None or isinstance(obj, child_type)
        ]
        result = [path for path in candidates if parent.is_ancestor(path, is_parent=(not recursive))]
        return result


class Db:
    def __init__(self, db_id: int, name: str):
        self.id = db_id
        self.name = name
        self.paths = PathDict()
        self.paths.update(self._load_dirs())
        self.paths.update(self._load_tables())
        self.paths.update(self._load_function_md())

    def create_table(
            self, path_str: str, schema: List[Column], num_retained_versions: int = 10,
            extract_frames_from: Optional[str] = None, extracted_frame_col: Optional[str] = None,
            extracted_frame_idx_col: Optional[str] = None, extracted_fps: Optional[int] = None
    ) -> MutableTable:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None, expected_parent_type=Dir)
        dir = self.paths[path.parent]

        # make sure frame extraction params are either fully present or absent
        frame_extraction_param_count = int(extract_frames_from is not None) + int(extracted_frame_col is not None)\
            + int(extracted_frame_idx_col is not None) + int(extracted_fps is not None)
        if frame_extraction_param_count != 0 and frame_extraction_param_count != 4:
            raise exc.BadFormatError(
                'Frame extraction requires that all parameters (extract_frames_from, extracted_frame_col, '
                'extracted_frame_idx_col, extracted_fps) be specified')
        tbl = MutableTable.create(
            self.id, dir.id, path.name, schema, num_retained_versions, extract_frames_from, extracted_frame_col,
            extracted_frame_idx_col, extracted_fps)
        self.paths[path] = tbl
        return tbl

    def get_table(self, path_str: str) -> Table:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Table)
        obj = self.paths[path]
        assert isinstance(obj, Table)
        return obj

    def rename_table(self, path_str: str, new_name: str) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=MutableTable)
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.BadFormatError(f"Invalid table name: '{new_name}'")
        new_path = path.parent.append(new_name)
        self.paths.check_is_valid(new_path, expected=None, expected_parent_type=Dir)

        tbl = self.paths[path]
        assert isinstance(tbl, MutableTable)
        del self.paths[path]
        self.paths[new_path] = tbl
        tbl.rename(new_name)

    def move_table(self, tbl_path: str, dir_path: str) -> None:
        pass

    def list_tables(self, dir_path: str = '', recursive: bool = True) -> List[str]:
        assert dir_path is not None
        path = Path(dir_path, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=DirBase)
        return [str(p) for p in self.paths.get_children(path, child_type=Table, recursive=recursive)]

    def drop_table(self, path_str: str, force: bool = False, ignore_errors: bool = False) -> None:
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=MutableTable)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        tbl = self.paths[path]
        assert isinstance(tbl, MutableTable)
        tbl.drop()
        del self.paths[path]

    def create_snapshot(self, path_str: str, tbl_paths: List[str]) -> None:
        snapshot_dir_path = Path(path_str)
        self.paths.check_is_valid(snapshot_dir_path, expected=None, expected_parent_type=Dir)
        tbls: List[MutableTable] = []
        for tbl_path_str in tbl_paths:
            tbl_path = Path(tbl_path_str)
            self.paths.check_is_valid(tbl_path, expected=MutableTable)
            tbl = self.paths[tbl_path]
            assert isinstance(tbl, MutableTable)
            tbls.append(tbl)

        with orm.Session(Env.get().engine) as session:
            dir_record = store.Dir(db_id=self.id, path=path_str, is_snapshot=True)
            session.add(dir_record)
            session.flush()
            assert dir_record.id is not None
            self.paths[snapshot_dir_path] = Dir(dir_record.id)

            for tbl in tbls:
                snapshot_record = store.TableSnapshot(
                    db_id=self.id, dir_id=dir_record.id, name=tbl.name, tbl_id=tbl.id, tbl_version=tbl.version,
                    tbl_schema_version=tbl.schema_version)
                session.add(snapshot_record)
                session.flush()
                assert snapshot_record.id is not None
                cols = Table.load_cols(tbl.id, tbl.schema_version, session)
                snapshot = TableSnapshot(snapshot_record, cols)
                snapshot_path = snapshot_dir_path.append(tbl.name)
                self.paths[snapshot_path] = snapshot

            session.commit()

    def create_dir(self, path_str: str) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None, expected_parent_type=Dir)
        with orm.Session(Env.get().engine) as session:
            dir_record = store.Dir(db_id=self.id, path=path_str, is_snapshot=False)
            session.add(dir_record)
            session.flush()
            assert dir_record.id is not None
            self.paths[path] = Dir(dir_record.id)
            session.commit()

    def rm_dir(self, path_str: str) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Dir)

        # make sure it's empty
        if len(self.paths.get_children(path, child_type=None, recursive=True)) > 0:
            raise exc.DirectoryNotEmptyError(f'Directory {path_str}')
        # TODO: figure out how to make force=True work in the presence of snapshots
#        # delete tables
#        for tbl_path in self.paths.get_children(path, child_type=Table, recursive=True):
#            self.drop_table(str(tbl_path), force=True)
#        # rm subdirs
#        for dir_path in self.paths.get_children(path, child_type=DirBase, recursive=False):
#            self.rm_dir(str(dir_path), force=True)

        with Env.get().engine.begin() as conn:
            dir = self.paths[path]
            conn.execute(sql.delete(store.Dir.__table__).where(store.Dir.id == dir.id))
        del self.paths[path]

    def list_dirs(self, path_str: str = '', recursive: bool = True) -> List[str]:
        path = Path(path_str, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=DirBase)
        return [str(p) for p in self.paths.get_children(path, child_type=DirBase, recursive=recursive)]

    def create_function(self, path_str: str, func: Function) -> None:
        if func.is_library_function:
            raise exc.Error(f'Cannot create a named function for a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None, expected_parent_type=Dir)
        dir = self.paths[path.parent]

        FunctionRegistry.get().create_function(func, self.id, dir.id, path.name)
        self.paths[path] = NamedFunction(func.id, dir.id, path.name)

    def rename_function(self, path_str: str, new_path_str: str) -> None:
        """
        Assign a new name and/or move the function to a different directory.
        """
        path = Path(path_str)
        new_path = Path(new_path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        self.paths.check_is_valid(new_path, expected=None)
        func = self.paths[path]
        new_dir = self.paths[new_path.parent]
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Function.__table__)
                    .values({
                        store.Function.dir_id: new_dir.id,
                        store.Function.name: new_path.name,
                    })
                    .where(store.Function.id == func.id))
        del self.paths[path]
        self.paths[new_path] = func

    def update_function(self, path_str: str, new_eval_fn: Callable) -> None:
        """
        Update the Function for given path with the callable.
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        # TODO: check that function signature doesn't change if the Function is used in a computed column
        FunctionRegistry.get().update_function(named_fn.id, new_eval_fn)

    def load_function(self, path_str: str) -> Function:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        assert isinstance(named_fn, NamedFunction)
        return FunctionRegistry.get().get_function(named_fn.id)

    def drop_function(self, path_str: str, ignore_errors: bool = False) -> None:
        """
        Deletes function from db, provided that no computed columns depend on it.
        """
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=NamedFunction)
        except exc.UnknownEntityError as e:
            if ignore_errors:
                return
            else:
                raise e
        named_fn = self.paths[path]
        FunctionRegistry.get().delete_function(named_fn.id)
        del self.paths[path]

    def _load_dirs(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(Env.get().engine) as session:
            for dir_record in session.query(store.Dir).where(store.Dir.db_id == self.id).all():
                result[dir_record.path] = SnapshotDir(dir_record.id) if dir_record.is_snapshot else Dir(dir_record.id)
        return result

    def _load_tables(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(Env.get().engine) as session:
            # load all reachable (= mutable) tables
            q = session.query(store.Table, store.Dir.path) \
                .join(store.Dir)\
                .where(store.Table.db_id == self.id) \
                .where(store.Table.is_mutable == True)
            for tbl_record, dir_path in q.all():
                cols = Table.load_cols(
                    tbl_record.id, tbl_record.current_schema_version, session)
                tbl = MutableTable(tbl_record, tbl_record.current_schema_version, cols)
                tbl._load_valid_rowids()  # TODO: move this someplace more appropriate
                path = Path(dir_path, empty_is_valid=True).append(tbl_record.name)
                result[str(path)] = tbl

            # load all table snapshots
            q = session.query(store.TableSnapshot, store.Dir.path) \
                .select_from(store.TableSnapshot) \
                .join(store.Table) \
                .join(store.Dir) \
                .where(store.TableSnapshot.db_id == self.id)
            for snapshot_record, dir_path in q.all():
                cols = Table.load_cols(snapshot_record.tbl_id, snapshot_record.tbl_schema_version, session)
                snapshot = TableSnapshot(snapshot_record, cols)
                path = Path(dir_path, empty_is_valid=True).append(snapshot_record.name)
                result[str(path)] = snapshot

        return result

    def _load_function_md(self) -> Dict[str, SchemaObject]:
        """
        Loads Function metadata. Doesn't load the actual callable, which can be large and is only done on-demand by the
        FunctionRegistry.
        """
        result: Dict[str, SchemaObject] = {}
        with orm.Session(Env.get().engine) as session:
            # load all reachable (= mutable) tables
            q = session.query(store.Function.id, store.Function.dir_id, store.Function.name, store.Dir.path) \
                .join(store.Dir) \
                .where(store.Function.db_id == self.id)
            for id, dir_id, name, dir_path in q.all():
                named_fn = NamedFunction(id, dir_id, name)
                path = Path(dir_path, empty_is_valid=True).append(name)
                result[str(path)] = named_fn
        return result

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'Db(name={self.name})'

    @classmethod
    def create(cls, name: str) -> 'Db':
        db_id: int = -1
        with orm.Session(Env.get().engine) as session:
            # check for duplicate name
            is_duplicate = session.query(sql.func.count(store.Db.id)).where(store.Db.name == name).scalar() > 0
            if is_duplicate:
                raise exc.DuplicateNameError(f"Db '{name}' already exists")

            db_record = store.Db(name=name)
            session.add(db_record)
            session.flush()
            assert db_record.id is not None
            db_id = db_record.id
            # also create a top-level directory, so that every schema object has a directory
            dir_record = store.Dir(db_id=db_id, path='', is_snapshot=False)
            session.add(dir_record)
            session.flush()
            session.commit()
        assert db_id is not None
        return Db(db_id, name)

    @classmethod
    def load(cls, name: str) -> 'Db':
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.BadFormatError(f"Invalid db name: '{name}'")
        with orm.Session(Env.get().engine) as session:
            try:
                db_record = session.query(store.Db).where(store.Db.name == name).one()
                return Db(db_record.id, db_record.name)
            except sql.exc.NoResultFound:
                raise exc.UnknownEntityError(f'Db {name}')

    def delete(self) -> None:
        """
        Delete db and all associated data.
        """
        with Env.get().engine.begin() as conn:
            conn.execute(sql.delete(store.TableSnapshot.__table__).where(store.TableSnapshot.db_id == self.id))
            tbls_stmt = sql.select(store.Table.id).where(store.Table.db_id == self.id)
            conn.execute(sql.delete(store.SchemaColumn.__table__).where(store.SchemaColumn.tbl_id.in_(tbls_stmt)))
            conn.execute(sql.delete(store.StorageColumn.__table__).where(store.StorageColumn.tbl_id.in_(tbls_stmt)))
            conn.execute(
                sql.delete(store.TableSchemaVersion.__table__).where(store.TableSchemaVersion.tbl_id.in_(tbls_stmt)))
            conn.execute(sql.delete(store.Table.__table__).where(store.Table.db_id == self.id))
            conn.execute(sql.delete(store.Function.__table__).where(store.Function.db_id == self.id))
            conn.execute(sql.delete(store.Dir.__table__).where(store.Dir.db_id == self.id))
            conn.execute(sql.delete(store.Db.__table__).where(store.Db.id == self.id))
            # delete all data tables
            # TODO: also deleted generated images
            for tbl in self.paths.get(MutableTable):
                tbl.sa_md.drop_all(bind=conn)
