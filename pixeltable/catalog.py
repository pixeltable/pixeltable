from typing import Optional, List, Set, Dict, Any, Type, Union, Callable, Generator
import re
import inspect
import io
import os
import dataclasses
import pathlib

import PIL, cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.autonotebook import tqdm
import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable import store
from pixeltable.env import Env
from pixeltable import exceptions as exc
from pixeltable.type_system import ColumnType, StringType
from pixeltable.utils import clip, video
from pixeltable.index import VectorIndex
from pixeltable.function import Function, FunctionRegistry
from pixeltable.utils.video import FrameIterator
from pixeltable.utils.imgstore import ImageStore
from pixeltable.utils.filecache import FileCache


_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


class Column:
    """
    Representation of a column in the schema of a Table/DataFrame.

    Members:
    - sa_col: column in the stored table for the values of this Column

    For computed columns:
    - sa_errormsg_col: column in the stored table for the exception string produced by running self.value_expr
    - sa_errortype_col: column in the stored table for the exception type name produced by running self.value_expr
    """
    def __init__(
            self, name: str, col_type: Optional[ColumnType] = None,
            computed_with: Optional[Union['Expr', Callable]] = None,
            primary_key: bool = False, nullable: bool = True, stored: Optional[bool] = None,
            # these parameters aren't set by users
            col_id: Optional[int] = None,
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

        stored (only valid for computed cols):
        - if True: the column is present in the stored table
        - if False: the column is not present in the stored table and always recomputed during a query and never cached
        - if None: the column is not present in the stored table but column values are cached during retrieval

        indexed: only valid for image columns; if true, maintains an NN index for this column
        """
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{name}'")
        self.name = name
        if col_type is None and computed_with is None:
            raise exc.Error(f'Column {name}: col_type is required if computed_with is not specified')
        assert not(value_expr_str is not None and computed_with is not None)

        self.value_expr: Optional['Expr'] = None
        self.compute_func: Optional[Callable] = None
        from pixeltable import exprs
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
        self.value_expr_str = value_expr_str  # stored here so it's easily accessible for the Table c'tor

        if col_type is not None:
            self.col_type = col_type
        assert self.col_type is not None

        self.stored = stored
        self.dependent_cols: List[Column] = []  # cols with value_exprs that reference us
        self.id = col_id
        self.primary_key = primary_key
        # computed cols are always nullable
        self.nullable = nullable or computed_with is not None or value_expr_str is not None
        self.sa_col: Optional[sql.schema.Column] = None
        # computed cols also have storage columns for the exception string and type
        self.sa_errormsg_col: Optional[sql.schema.Column] = None
        self.sa_errortype_col: Optional[sql.schema.Column] = None
        self.tbl: Optional[Table] = None  # set by owning Table

        if indexed and not self.col_type.is_image_type():
            raise exc.Error(f'Column {name}: indexed=True requires ImageType')
        self.is_indexed = indexed
        self.idx: Optional[VectorIndex] = None

    def check_value_expr(self) -> None:
        assert self.value_expr is not None
        from pixeltable import exprs
        if self.stored == False and self.is_computed and self._has_window_fn_call():
            raise exc.Error(
                f'Column {self.name}: stored={self.stored} not supported for columns computed with window functions:'
                f'\n{self.value_expr}')

    def _has_window_fn_call(self) -> bool:
        if self.value_expr is None:
            return False
        from pixeltable import exprs
        l = list(self.value_expr.subexprs(filter=lambda e: isinstance(e, exprs.FunctionCall) and e.is_window_fn_call))
        return len(l) > 0

    @property
    def is_computed(self) -> bool:
        return self.compute_func is not None or self.value_expr is not None or self.value_expr_str is not None

    @property
    def is_stored(self) -> bool:
        """
        Returns True if column is materialized in the stored table.
        Note that the extracted frame col is effectively computed.
        """
        return self.stored == True \
            or (not self.is_computed and not self.id == self.tbl.parameters.frame_col_id) \
            or not self.col_type.is_image_type() \
            or self._has_window_fn_call()

    @property
    def is_cached(self) -> bool:
        """
        Returns True if column is not materialized in the stored table but cachable.
        """
        return not self.is_stored and self.stored is None

    def list(self) -> None:
        """
        If this is a computed col and the top-level expr is a function call, print the source, if possible.
        """
        from pixeltable import exprs
        if self.value_expr is None or not isinstance(self.value_expr, exprs.FunctionCall):
            return
        self.value_expr.fn.list()

    def create_sa_cols(self) -> None:
        """
        These need to be recreated for every new table schema version.
        """
        assert self.is_stored
        # computed cols store a NULL value when the computation has an error
        nullable = True if self.is_computed else self.nullable
        self.sa_col = sql.Column(self.storage_name(), self.col_type.to_sa_type(), nullable=nullable)
        if self.is_computed:
            self.sa_errormsg_col = sql.Column(self.errormsg_storage_name(), StringType().to_sa_type(), nullable=True)
            self.sa_errortype_col = sql.Column(self.errortype_storage_name(), StringType().to_sa_type(), nullable=True)

    def set_idx(self, idx: VectorIndex) -> None:
        self.idx = idx

    def storage_name(self) -> str:
        assert self.id is not None
        assert self.is_stored
        return f'col_{self.id}'

    def errormsg_storage_name(self) -> str:
        return f'{self.storage_name()}_errormsg'

    def errortype_storage_name(self) -> str:
        return f'{self.storage_name()}_errortype'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return False
        assert self.tbl is not None
        assert other.tbl is not None
        return self.tbl.id == other.tbl.id and self.id == other.id


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


@dataclasses.dataclass
class TableParameters:
    # garbage-collect old versions beyond this point, unless they are referenced in a snapshot
    num_retained_versions: int = 10

    # parameters for frame extraction
    frame_src_col_id: int = -1 # column id
    frame_col_id: int = -1 # column id
    frame_idx_col_id: int = -1 # column id
    extraction_fps: int = -1
    ffmpeg_filter: Dict[str, str] = dataclasses.field(default_factory=dict)


class Table(SchemaObject):
    def __init__(
            self, db_id: int, tbl_id: int, dir_id: int, name: str, version: int, params: Dict, cols: List[Column]):
        super().__init__(tbl_id)
        self.db_id = db_id
        self.dir_id = dir_id
        # TODO: this will be out-of-date after a rename()
        self.name = name
        for pos, col in enumerate(cols):
            if re.fullmatch(_ID_RE, col.name) is None:
                raise exc.BadFormatError(f"Invalid column name: '{col.name}'")
            assert col.id is not None
            col.tbl = self
        self.cols = cols
        self.cols_by_name = {col.name: col for col in cols}
        self.cols_by_id = {col.id: col for col in cols}
        self.version = version
        self.parameters = TableParameters(**params)

        # we can't call _load_valid_rowids() here because the storage table may not exist yet
        self.valid_rowids: Set[int] = set()

        # sqlalchemy-related metadata; used to insert and query the storage table
        self.sa_md = sql.MetaData()
        self._create_sa_tbl()
        self.is_dropped = False

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        for col in self.cols:
            col.tbl = self
            if col.value_expr is not None or col.value_expr_str is not None:
                self._record_value_expr(col)

    def extracts_frames(self) -> bool:
        return self.parameters.frame_col_id != -1

    def is_frame_col(self, c: Column) -> bool:
        return c.id == self.parameters.frame_col_id

    def frame_src_col(self) -> Optional[Column]:
        """
        Return the frame src col, or None if not applicable.
        """
        if self.parameters.frame_src_col_id == -1:
            return None
        return self.cols_by_id[self.parameters.frame_src_col_id]

    def frame_idx_col(self) -> Optional[Column]:
        """
        Return the frame idx col, or None if not applicable.
        """
        if self.parameters.frame_idx_col_id == -1:
            return None
        return self.cols_by_id[self.parameters.frame_idx_col_id]

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
        from pixeltable.exprs import ColumnRef, FrameColumnRef
        if self.is_frame_col(col):
            return FrameColumnRef(col)
        else:
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

    @property
    def frame_col(self) -> Optional[Column]:
        if self.parameters.frame_col_id == -1:
            return None
        return self.cols_by_id[self.parameters.frame_col_id]

    def describe(self) -> pd.DataFrame:
        pd_df = pd.DataFrame({
            'Column Name': [c.name for c in self.cols],
            'Type': [str(c.col_type) for c in self.cols],
            'Computed With':
                [c.value_expr.display_str(inline=False) if c.value_expr is not None else '' for c in self.cols],
        })
        # white-space: pre-wrap: print \n as newline
        pd_df = pd_df.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'})\
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])  # center-align headings
        return pd_df.hide(axis='index')

    def storage_name(self) -> str:
        return f'tbl_{self.id}'

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.RuntimeError('Table has been dropped')

    def _create_sa_tbl(self) -> None:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(store.Table.MAX_VERSION))

        sa_cols = [self.rowid_col, self.v_min_col, self.v_max_col]
        for col in [c for c in self.cols if c.is_stored]:
            # re-create sql.Columns for each column, regardless of whether it already has sa_col set: it was bound
            # to the last sql.Table version we created and cannot be reused
            col.create_sa_cols()
            sa_cols.append(col.sa_col)
            if col.is_computed:
                sa_cols.append(col.sa_errormsg_col)
                sa_cols.append(col.sa_errortype_col)

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
                stored=r.stored, col_id=r.col_id, value_expr_str=r.value_expr, indexed=r.is_indexed)
            for r in col_records
        ]
        for col in [col for col in cols if col.col_type.is_image_type()]:
            if col.is_indexed:
                col.set_idx(VectorIndex.load(cls._vector_idx_name(tbl_id, col), dim=512))
        return cols


class TableSnapshot(Table):
    def __init__(self, snapshot_record: store.TableSnapshot, params: Dict, cols: List[Column]):
        assert snapshot_record.db_id is not None
        assert snapshot_record.id is not None
        assert snapshot_record.dir_id is not None
        assert snapshot_record.name is not None
        assert snapshot_record.tbl_version is not None
        # the id of this SchemaObject is TableSnapshot.tbl_id, not TableSnapshot.id: we use tbl_id to construct
        # the name of the data table
        super().__init__(
            snapshot_record.db_id, snapshot_record.tbl_id, snapshot_record.dir_id, snapshot_record.name,
            snapshot_record.tbl_version, params, cols)
        self.snapshot_tbl_id = snapshot_record.id
        # it's safe to call _load_valid_rowids() here because the storage table already exists
        self._load_valid_rowids()

    def __repr__(self) -> str:
        return f'TableSnapshot(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table snapshot'

class MutableTable(Table):
    def __init__(self, tbl_record: store.Table, schema_version: int, cols: List[Column]):
        assert tbl_record.db_id is not None
        assert tbl_record.id is not None
        assert tbl_record.dir_id is not None
        assert tbl_record.name is not None
        assert tbl_record.current_version is not None
        super().__init__(
            tbl_record.db_id, tbl_record.id, tbl_record.dir_id, tbl_record.name, tbl_record.current_version,
            tbl_record.parameters, cols)
        assert tbl_record.next_col_id is not None
        self.next_col_id = tbl_record.next_col_id
        assert tbl_record.next_row_id is not None
        self.next_row_id = tbl_record.next_row_id
        self.schema_version = schema_version

    def __repr__(self) -> str:
        return f'MutableTable(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    def add_column(self, c: Column) -> str:
        self._check_is_dropped()
        if re.fullmatch(_ID_RE, c.name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{c.name}'")
        if c.name in self.cols_by_name:
            raise exc.DuplicateNameError(f'Column {c.name} already exists')
        assert self.next_col_id is not None
        c.tbl = self
        c.id = self.next_col_id
        self.next_col_id += 1

        if c.compute_func is not None:
            # create value_expr from compute_func
            self._create_value_expr(c, self.cols_by_name)
        if c.value_expr is not None:
            c.check_value_expr()
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
                sql.insert(store.ColumnHistory.__table__)
                    .values(tbl_id=self.id, col_id=c.id, schema_version_add=self.schema_version))
            self._create_col_md(conn)

            if c.is_stored:
                stmt = f'ALTER TABLE {self.storage_name()} ADD COLUMN {c.storage_name()} {c.col_type.to_sql()}'
                conn.execute(sql.text(stmt))
                if c.is_computed:
                    # we also need to create the errormsg and errortype storage cols
                    stmt = (f'ALTER TABLE {self.storage_name()} '
                            f'ADD COLUMN {c.errormsg_storage_name()} {StringType().to_sql()} DEFAULT NULL')
                    conn.execute(sql.text(stmt))
                    stmt = (f'ALTER TABLE {self.storage_name()} '
                            f'ADD COLUMN {c.errortype_storage_name()} {StringType().to_sql()} DEFAULT NULL')
                    conn.execute(sql.text(stmt))
                self._create_sa_tbl()

        row_count = self.count()
        if not c.is_computed or not c.is_stored or row_count == 0:
            return ''
        # for some reason, it's not possible to run the following updates in the same transaction as the one
        # that we just used to create the metadata (sqlalchemy hangs when exec() tries to run the query)
        from pixeltable.dataframe import DataFrame
        # use copy to avoid reusing existing execution state
        query = DataFrame(self, [c.value_expr.copy()])
        with Env.get().engine.begin() as conn:
            with tqdm(total=row_count) as progress_bar:
                try:
                    num_excs = 0
                    for result_row in query.exec(n=0, select_pk=True, ignore_errors=True):
                        # we can simply update the row, instead of creating a copy for the current version, because the
                        # added column will not be visible when querying prior versions
                        column_val, rowid, v_min = result_row

                        if isinstance(column_val, Exception):
                            num_excs += 1
                            value_exc = column_val
                            # we store a NULL value and record the exception/exc type
                            error_type = type(value_exc).__name__
                            error_msg = str(value_exc)
                            conn.execute(
                                sql.update(self.sa_tbl)
                                    .values(
                                        {c.sa_col: None, c.sa_errortype_col: error_type, c.sa_errormsg_col: error_msg})
                                    .where(self.rowid_col == rowid)
                                    .where(self.v_min_col == v_min))
                        else:
                            column_val = self._convert_to_stored(c, column_val, rowid)
                            conn.execute(
                                sql.update(self.sa_tbl)
                                    .values({c.sa_col: column_val})
                                    .where(self.rowid_col == rowid)
                                    .where(self.v_min_col == v_min))
                        progress_bar.update(1)
                    return f'Added {row_count} column values with {num_excs} error{"" if num_excs == 1 else "s"}'
                except sql.exc.DBAPIError as e:
                    self.drop_column(c.name)
                    raise exc.Error(f'Error during SQL execution:\n{e}')

    def drop_column(self, name: str) -> None:
        self._check_is_dropped()
        if name not in self.cols_by_name:
            raise exc.UnknownEntityError
        col = self.cols_by_name[name]
        if len(col.dependent_cols) > 0:
            raise exc.Error(
                f'Cannot drop column {name} because the following columns depend on it:\n',
                f'{", ".join([c.name for c in col.dependent_cols])}')
        if col.id == self.parameters.frame_col_id or col.id == self.parameters.frame_idx_col_id:
            src_col_name = self.cols_by_id[self.parameters.frame_src_col_id].name
            raise exc.Error(
                f'Cannot drop column {name} because it is used for frame extraction on column {src_col_name}')
        if col.id == self.parameters.frame_src_col_id:
            # we also need to reset the frame extraction table parameters
            self.parameters.frame_src_col_id = -1
            self.parameters.frame_col_id = -1
            self.parameters.frame_idx_col_id = -1
            self.parameters.extraction_fps = -1

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
                sql.update(store.ColumnHistory.__table__)
                    .values({store.ColumnHistory.schema_version_drop: self.schema_version})
                    .where(store.ColumnHistory.tbl_id == self.id)
                    .where(store.ColumnHistory.col_id == col.id))
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
                    value_expr=value_expr_str, stored=c.stored, is_indexed=c.is_indexed))

    def _convert_to_stored(self, col: Column, val: Any, rowid: int) -> Any:
        """
        Convert column value 'val' into a store-compatible format, if needed:
        - images are stored as files
        - arrays are stored as serialized ndarrays
        """
        if col.col_type.is_image_type():
            # replace PIL.Image.Image with file path
            img = val
            img_path = ImageStore.get_path(self.id, col.id, self.version, rowid)
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
        return self.insert_pandas(pd_df)

    def _check_data(self, data: pd.DataFrame):
        """
        Make sure 'data' conforms to schema.
        """
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = {col.name for col in self.cols if not col.nullable and col.value_expr is None}
        if self.extracts_frames():
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_col_id].name)
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_idx_col_id].name)
        given_col_names = set(data.columns)
        if not(reqd_col_names <= given_col_names):
            raise exc.InsertError(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')
        if not(given_col_names <= all_col_names):
            raise exc.InsertError(f'Unknown columns: {", ".join(given_col_names - all_col_names)}')
        computed_col_names = {col.name for col in self.cols if col.value_expr is not None}
        if self.extracts_frames():
            computed_col_names.add(self.cols_by_id[self.parameters.frame_col_id].name)
            computed_col_names.add(self.cols_by_id[self.parameters.frame_idx_col_id].name)
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
            if not col.nullable:
                # check for nulls
                nulls = data[col.name].isna()
                max_val_idx = nulls.idxmax()
                if nulls[max_val_idx]:
                    raise exc.RuntimeError(
                        f'Column {col.name}: row {max_val_idx} contains None for a non-nullable column')
                pass

            # image cols: make sure file path points to a valid image file
            if col.col_type.is_image_type():
                for _, path_str in data[col.name].items():
                    if path_str is None:
                        continue
                    try:
                        _ = Image.open(path_str)
                    except FileNotFoundError:
                        raise exc.RuntimeError(f'Column {col.name}: file does not exist: {path_str}')
                    except PIL.UnidentifiedImageError:
                        raise exc.RuntimeError(f'Column {col.name}: not a valid image file: {path_str}')

            # image cols: make sure file path points to a valid image file; build index if col is indexed
            if col.col_type.is_video_type():
                for _, path_str in data[col.name].items():
                    if path_str is None:
                        continue
                    path = pathlib.Path(path_str)
                    if not path.is_file():
                        raise exc.RuntimeError(
                            f'For frame extraction, value for column {col.name} in row {idx} requires a valid '
                            f'file path: {path}')
                    cap = cv2.VideoCapture(path_str)
                    success = cap.isOpened()
                    cap.release()
                    if not success:
                        raise exc.Error(f'Column {col.name}: could not open video file {path_str}')

            if col.col_type.is_json_type():
                for idx, d in data[col.name].items():
                    if d is not None and not isinstance(d, dict) and not isinstance(d, list):
                        raise exc.RuntimeError(
                            f'Value for column {col.name} in row {idx} requires a dictionary or list: {d} ')

    def insert_pandas(self, data: pd.DataFrame) -> str:
        """
        If self.parameters.frame_src_col_id != None:
        - each row (containing a video) is expanded into one row per extracted frame (at the rate of the fps parameter)
        - parameters.frame_col_id is the image column that receives the extracted frame
        - parameters.frame_idx_col_id is the integer column that receives the frame index (starting at 0)
        """
        self._check_is_dropped()
        self._check_data(data)

        # we're creating a new version
        self.version += 1

        if self.extracts_frames():
            video_col = self.cols_by_id[self.parameters.frame_src_col_id]
            frame_col = self.cols_by_id[self.parameters.frame_col_id]
            frame_idx_col = self.cols_by_id[self.parameters.frame_idx_col_id]
        else:
            video_col, frame_col, frame_idx_col = None, None, None

        # frame extraction from videos
        frame_iters: List[Optional[FrameIterator]] = [None] * len(data)
        est_num_rows = len(data)  # total number of rows, after frame extraction
        if self.extracts_frames():
            print('Counting frames...')
            data.sort_values([video_col.name], axis=0, inplace=True)  # we need to order by video_col, frame_idx_col
            with tqdm(total=data[video_col.name].count()) as progress_bar:
                video_paths = list(data[video_col.name])
                video_path_isnull = list(data[video_col.name].isna())
                for i in range(len(data)):
                    if video_path_isnull[i]:
                        continue
                    frame_iter = FrameIterator(
                        video_paths[i], fps=self.parameters.extraction_fps, ffmpeg_filter=self.parameters.ffmpeg_filter)
                    if frame_iter.est_num_frames is not None:
                        est_num_rows += frame_iter.est_num_frames - 1
                    else:
                        est_num_rows = None
                    frame_iters[i] = frame_iter
                    progress_bar.update(1)

        def rowid_generator(start_id: int) -> Generator[int, None, None]:
            rowid = start_id
            while True:
                yield rowid
                rowid += 1
        rowids = rowid_generator(self.next_row_id)

        # prepare state for stored computed cols
        from pixeltable import exprs
        # create copies to avoid reusing past execution state; eval ctx and evaluator need to share these copies
        stored = [[c, c.value_expr.copy()] for c in self.cols if c.is_computed and c.is_stored]
        for i in range(len(stored)):
            # substitute refs to computed columns until there aren't any
            while True:
                target = stored[i][1]
                computed_col_refs = [
                    e for e in target.subexprs() if isinstance(e, exprs.ColumnRef) and e.col.is_computed
                ]
                if len(computed_col_refs) == 0:
                    break
                for ref in computed_col_refs:
                    assert ref.col.value_expr is not None
                    stored[i][1] = stored[i][1].substitute(ref, ref.col.value_expr)

        stored_exprs = [e for _, e in stored]
        evaluator = exprs.Evaluator(stored_exprs, with_sql=False) if len(stored) > 0 else None
        stored_exprs_ctx = evaluator.get_eval_ctx([e.data_row_idx for e in stored_exprs]) if len(stored) > 0 else None
        input_col_refs = exprs.UniqueExprList([
            e for e in exprs.Expr.list_subexprs(stored_exprs)
            # we're looking for ColumnRefs to Columns that aren't themselves computed
            if isinstance(e, exprs.ColumnRef) and not e.col.is_computed])

        # if we're extracting frames, the ordering is dictated by that, and we already checked that
        # all computed cols with window functions are compatible with that
        # TODO: implement that check in add_column()
        if not self.extracts_frames():
            # determine order_by clause for window functions, if any
            window_fn_calls = [
                e for e in exprs.Expr.list_subexprs(stored_exprs)
                if isinstance(e, exprs.FunctionCall) and e.is_window_fn_call
            ]
            window_sort_exprs = window_fn_calls[0].get_window_sort_exprs() if len(window_fn_calls) > 0 else []

            if len(window_sort_exprs) > 0:
                # need to sort data in order to compute windowed agg functions
                sort_col_names = [e.col.name for e in window_sort_exprs]
                data.sort_values(sort_col_names, axis=0, inplace=True)

        # switch to storage column names in 'data'
        data = data.rename(
            {col_name: self.cols_by_name[col_name].storage_name() for col_name in data.columns}, axis='columns',
            inplace=False)

        def input_rows() -> Generator[Dict[str, Any], None, None]:
            """
            Returns rows from 'data' as dict with rowid and v_min fields.
            """
            assert video_col is None
            for input_row_idx, input_tuple in enumerate(data.itertuples(index=False)):
                output_row = {'rowid': next(rowids), 'v_min': self.version, **input_tuple._asdict()}
                yield output_row

        def expanded_input_rows() -> Generator[Dict[str, Any], None, None]:
            """
            Returns rows from 'data' as dict with rowid and v_min fields, expanded with frames:
            each input row turns into n output rows (n = number of frames in the input row's video)
            """
            assert video_col is not None
            next_output_row_idx = 0
            for input_row_idx, input_tuple in enumerate(data.itertuples(index=False)):
                with frame_iters[input_row_idx] as frame_iter:
                    for frame_idx, frame_path in frame_iter:
                        rowid = next(rowids)
                        output_row = {
                            'rowid': rowid, 'v_min': self.version, **input_tuple._asdict(),
                            frame_idx_col.storage_name(): frame_idx,
                        }
                        if frame_col.is_stored:
                            # move frame file to a permanent location
                            frame_path = str(ImageStore.add(self.id, frame_col.id, self.version, rowid, frame_path))
                            output_row[frame_col.storage_name()] = frame_path
                        output_row['frame'] = PIL.Image.open(frame_path)
                        next_output_row_idx += 1
                        yield output_row


        # we're also updating image indices; we're doing this with one idx.insert() call rather than piecemeal,
        # because the index gets rebuilt from the ground up in each such call
        indexed_cols = [c for c in self.cols if c.is_indexed]
        embeddings = {c.id: [] for c in indexed_cols}  # value: List[np.array((512))]
        cols_with_excs: Set[int] = set()  # set of ids
        num_excs = 0
        print('Inserting rows...')

        # TODO: typing for tqdm()
        def output_rows(progress_bar: Any) -> Generator[Dict[str, Any], None, None]:
            """
            Return rows used to supply values to sql.insert().
            """
            row_generator = input_rows() if video_col is None else expanded_input_rows()
            for row_idx, row in enumerate(row_generator):
                if len(stored) > 0:
                    # stored computed column values
                    data_row = evaluator.prepare([], False)

                    # copy inputs
                    for col_ref in input_col_refs:
                        if col_ref.col.id == self.parameters.frame_col_id:
                            data_row[col_ref.data_row_idx] = row['frame']
                        else:
                            data_row[col_ref.data_row_idx] = row[col_ref.col.storage_name()]
                            # load image, if this is a file path
                            if col_ref.col_type.is_image_type():
                                data_row[col_ref.data_row_idx] = PIL.Image.open(data_row[col_ref.data_row_idx])

                    evaluator.eval(data_row, stored_exprs_ctx)

                    computed_vals_dict: Dict[str, Any] = {}  # key: column's storage name
                    for col, value_expr in stored:
                        val = data_row[value_expr.data_row_idx]
                        if isinstance(val, Exception):
                            nonlocal num_excs
                            num_excs += 1
                            cols_with_excs.add(col.id)
                            computed_vals_dict[col.storage_name()] = None
                            computed_vals_dict[col.errortype_storage_name()] = type(val).__name__
                            computed_vals_dict[col.errormsg_storage_name()] = str(val)
                        else:
                            # convert data values to storage format where necessary
                            data_row[value_expr.data_row_idx] = self._convert_to_stored(col, val, row['rowid'])
                            computed_vals_dict[col.storage_name()] = data_row[value_expr.data_row_idx]
                            computed_vals_dict[col.errortype_storage_name()] = None
                            computed_vals_dict[col.errormsg_storage_name()] = None
                    row.update(computed_vals_dict)

                # compute embeddings
                for c in indexed_cols:
                    img: Optional[PIL.Image.Image] = None
                    if c.id == self.parameters.frame_col_id:
                        img = row['frame']
                    else:
                        path_str = row[c.storage_name()]
                        try:
                            img = Image.open(path_str)
                        except:
                            # this shouldn't be happening at this point
                            raise RuntimeError(f'Image column {c.name}: file does not exist or is invalid: {path_str}')
                    embeddings[c.id].append(clip.encode_image(img))

                progress_bar.update(1)
                yield row

        with tqdm(total=est_num_rows) as progress_bar:
            with Env.get().engine.begin() as conn:
                # insert 128 rows at a time
                batch_size = 128
                has_data = True
                insert_values: List[Dict[str, Any]] = []
                rows = output_rows(progress_bar)
                num_rows = 0
                while has_data:
                    try:
                        insert_values.append(next(rows))
                        num_rows += 1
                    except StopIteration:
                        has_data = False
                    if len(insert_values) == batch_size or not has_data:
                        conn.execute(sql.insert(self.sa_tbl), insert_values)
                        insert_values = []

                start_row_id = self.next_row_id
                self.next_row_id += num_rows
                conn.execute(
                    sql.update(store.Table.__table__)
                        .values({store.Table.current_version: self.version, store.Table.next_row_id: self.next_row_id})
                        .where(store.Table.id == self.id))

        # update image indices
        for c in indexed_cols:
            c.idx.insert(np.asarray(embeddings[c.id]), np.arange(start_row_id, start_row_id + num_rows))

        self.valid_rowids.update(range(start_row_id, start_row_id + num_rows))
        if num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = f'across {len(cols_with_excs)} column{"" if len(cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join([self.cols_by_id[id].name for id in cols_with_excs])})'
        return f'Inserted {num_rows} rows with {num_excs} error{"" if num_excs == 1 else "s"} {cols_with_excs_str}'

    def insert_csv(self, file_path: str) -> None:
        pass

    # TODO: update() signature?
    #def update(self, data: pd.DataFrame) -> None:

    # TODO: delete() signature?
    #def delete(self, data: DataFrame) -> None:

    def revert(self) -> None:
        self._check_is_dropped()
        if self.version == 0:
            raise exc.RuntimeError('Cannot revert version 0')
        # check if the current version is referenced by a snapshot
        with orm.Session(Env.get().engine) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(store.TableSnapshot.id)) \
                .where(store.TableSnapshot.db_id == self.db_id) \
                .where(store.TableSnapshot.tbl_id == self.id) \
                .where(store.TableSnapshot.tbl_version == self.version) \
                .scalar()
            if num_references > 0:
                raise exc.RuntimeError(
                    f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')

            conn = session.connection()
            # delete newly-added data
            ImageStore.delete(self.id, v_min=self.version)
            FileCache.get().clear(tbl_id=self.id)
            conn.execute(sql.delete(self.sa_tbl).where(self.sa_tbl.c.v_min == self.version))
            # revert new deletions
            conn.execute(
                sql.update(self.sa_tbl).values({self.sa_tbl.c.v_max: store.Table.MAX_VERSION})
                    .where(self.sa_tbl.c.v_max == self.version))

            if self.version == self.schema_version:
                # the current version involved a schema change:
                # if the schema change was to add a column, we now need to drop it
                added_col_id = session.query(store.ColumnHistory.col_id)\
                    .where(store.ColumnHistory.tbl_id == self.id)\
                    .where(store.ColumnHistory.schema_version_add == self.schema_version)\
                    .scalar()
                if added_col_id is not None:
                    # drop this newly-added column and its ColumnHistory record
                    c = self.cols_by_id[added_col_id]
                    stmt = f'ALTER TABLE {self.storage_name()} DROP COLUMN {c.storage_name()}'
                    conn.execute(sql.text(stmt))
                    conn.execute(
                        sql.delete(store.ColumnHistory.__table__)
                            .where(store.ColumnHistory.tbl_id == self.id)
                            .where(store.ColumnHistory.col_id == added_col_id))

                # if the schema change was to drop a column, we now need to undo that
                dropped_col_id = session.query(store.ColumnHistory.col_id) \
                    .where(store.ColumnHistory.tbl_id == self.id) \
                    .where(store.ColumnHistory.schema_version_drop == self.schema_version) \
                    .scalar()
                if dropped_col_id is not None:
                    # fix up the ColumnHistory record
                    conn.execute(
                        sql.update(store.ColumnHistory.__table__)
                            .values({store.ColumnHistory.schema_version_drop: None})
                            .where(store.ColumnHistory.tbl_id == self.id)
                            .where(store.ColumnHistory.col_id == dropped_col_id))

                # we need to determine the preceding schema version and reload the schema
                preceding_schema_version = session.query(store.TableSchemaVersion.preceding_schema_version) \
                    .where(store.TableSchemaVersion.tbl_id == self.id) \
                    .where(store.TableSchemaVersion.schema_version == self.schema_version) \
                    .scalar()
                self.cols = self.load_cols(self.id, preceding_schema_version, session)
                for c in self.cols:
                    c.tbl = self

                # drop all SchemaColumn records for this schema version prior to deleting from TableSchemaVersion
                # (to avoid FK violations)
                conn.execute(
                    sql.delete(store.SchemaColumn.__table__)
                        .where(store.SchemaColumn.tbl_id == self.id)
                        .where(store.SchemaColumn.schema_version == self.schema_version))
                conn.execute(
                    sql.delete(store.TableSchemaVersion.__table__)
                        .where(store.TableSchemaVersion.tbl_id == self.id)
                        .where(store.TableSchemaVersion.schema_version == self.schema_version))
                self.schema_version = preceding_schema_version

            self.version -= 1
            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version
                    })
                    .where(store.Table.id == self.id))

            session.commit()

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
        self.is_dropped = True

        with orm.Session(Env.get().engine) as session:
            # check if we have snapshots
            num_references = session.query(sql.func.count(store.TableSnapshot.id)) \
                .where(store.TableSnapshot.db_id == self.db_id) \
                .where(store.TableSnapshot.tbl_id == self.id) \
                .scalar()
            if num_references == 0:
                # we can delete this table altogether
                ImageStore.delete(self.id)
                FileCache.get().clear(self.id)
                conn = session.connection()
                conn.execute(sql.delete(store.SchemaColumn.__table__).where(store.SchemaColumn.tbl_id == self.id))
                conn.execute(sql.delete(store.ColumnHistory.__table__).where(store.ColumnHistory.tbl_id == self.id))
                conn.execute(
                    sql.delete(store.TableSchemaVersion.__table__).where(store.TableSchemaVersion.tbl_id == self.id))
                conn.execute(sql.delete(store.Table.__table__).where(store.Table.id == self.id))
                self.sa_md.drop_all(bind=conn)
                session.commit()
                return

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
        fn = Function.make_function(col.col_type, [arg.col_type for arg in args], col.compute_func)
        col.value_expr = exprs.FunctionCall(fn, args)

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
        cls, db_id: int, dir_id: int, name: str, cols: List[Column],
        num_retained_versions: int,
        extract_frames_from: Optional[str], extracted_frame_col: Optional[str], extracted_frame_idx_col: Optional[str],
        extracted_fps: Optional[int], ffmpeg_filter: Optional[Dict[str, str]]
    ) -> 'MutableTable':
        # make sure col names are unique (within the table) and assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, col in enumerate(cols):
            if col.name in cols_by_name:
                raise exc.DuplicateNameError(f'Duplicate column: {col.name}')
            col.id = pos
            cols_by_name[col.name] = col
            if col.value_expr is None and col.compute_func is not None:
                cls._create_value_expr(col, cols_by_name)
            if col.is_computed:
                col.check_value_expr()

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
            cols_by_name[extract_frames_from].id if extract_frames_from is not None else -1,
            cols_by_name[extracted_frame_col].id if extracted_frame_col is not None else -1,
            cols_by_name[extracted_frame_idx_col].id if extracted_frame_idx_col is not None else -1,
            extracted_fps,
            ffmpeg_filter)

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
                session.add(store.ColumnHistory(tbl_id=tbl_record.id, col_id=col.id, schema_version_add=0))
                session.flush()  # avoid FK violations in Postgres
                # Column.dependent_cols for existing cols is wrong at this point, but Table.init() will set it correctly
                value_expr_str = col.value_expr.serialize() if col.value_expr is not None else None
                session.add(
                    store.SchemaColumn(
                        tbl_id=tbl_record.id, schema_version=0, col_id=col.id, pos=pos, name=col.name,
                        col_type=col.col_type.serialize(), is_nullable=col.nullable, is_pk=col.primary_key,
                        value_expr=value_expr_str, stored=col.stored, is_indexed=col.is_indexed)
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
            raise exc.DuplicateNameError(f"'{path_str}' already exists")
        # check for containing directory
        parent_path = path.parent
        if str(parent_path) not in self.paths:
            raise exc.UnknownEntityError(f'Directory {str(parent_path)}')
        parent = self.paths[str(parent_path)]
        if not isinstance(parent, expected_parent_type):
            raise exc.UnknownEntityError(f'{str(parent_path)} needs to be a {expected_parent_type.display_name()}')

    def get(self, path_type: Type[SchemaObject]) -> List[SchemaObject]:
        return [obj for obj in self.paths.values() if isinstance(obj, path_type)]

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
            extracted_frame_idx_col: Optional[str] = None, extracted_fps: Optional[int] = None,
            ffmpeg_filter: Optional[Dict[str, str]] = None,
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
        if frame_extraction_param_count == 0 and ffmpeg_filter is not None:
            raise exc.BadFormatError(f'ffmpeg_filter only valid in conjunction with other frame extraction parameters')
        tbl = MutableTable.create(
            self.id, dir.id, path.name, schema, num_retained_versions, extract_frames_from, extracted_frame_col,
            extracted_frame_idx_col, extracted_fps, ffmpeg_filter)
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
                snapshot = TableSnapshot(snapshot_record, dataclasses.asdict(tbl.parameters), cols)
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
        func.md.fqn = f'{self.name}.{path}'

    def rename_function(self, path_str: str, new_path_str: str) -> None:
        """
        Assign a new name and/or move the function to a different directory.
        """
        path = Path(path_str)
        new_path = Path(new_path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        self.paths.check_is_valid(new_path, expected=None)
        named_fn = self.paths[path]
        new_dir = self.paths[new_path.parent]
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(store.Function.__table__)
                    .values({
                        store.Function.dir_id: new_dir.id,
                        store.Function.name: new_path.name,
                    })
                    .where(store.Function.id == named_fn.id))
        del self.paths[path]
        self.paths[new_path] = named_fn
        func = FunctionRegistry.get().get_function(named_fn.id)
        func.md.fqn = f'{self.name}.{new_path}'

    def update_function(self, path_str: str, new_func: Function) -> None:
        """
        Update the Function for given path with func.
        """
        if new_func.is_library_function:
            raise exc.Error(f'Cannot update a named function to a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        func = FunctionRegistry.get().get_function(named_fn.id)
        if func.md.signature != new_func.md.signature:
            raise exc.Error(
                f'The function signature cannot be changed. The existing signature is {func.md.signature}')
        if func.is_aggregate != new_func.is_aggregate:
            raise exc.Error(f'Cannot change an aggregate function into a non-aggregate function and vice versa')
        FunctionRegistry.get().update_function(named_fn.id, func)

    def get_function(self, path_str: str) -> Function:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        assert isinstance(named_fn, NamedFunction)
        func = FunctionRegistry.get().get_function(named_fn.id)
        func.md.fqn = f'{self.name}.{path}'
        return func

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
            q = session.query(store.TableSnapshot, store.Dir.path, store.Table.parameters) \
                .select_from(store.TableSnapshot) \
                .join(store.Table) \
                .join(store.Dir) \
                .where(store.TableSnapshot.db_id == self.id)
            for snapshot_record, dir_path, params in q.all():
                cols = Table.load_cols(snapshot_record.tbl_id, snapshot_record.tbl_schema_version, session)
                snapshot = TableSnapshot(snapshot_record, params, cols)
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
            conn.execute(sql.delete(store.ColumnHistory.__table__).where(store.ColumnHistory.tbl_id.in_(tbls_stmt)))
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
