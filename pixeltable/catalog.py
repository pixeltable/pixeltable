from typing import Optional, List, Set, Dict, Any, Type, Union, Callable, Generator
import re
import inspect
import io
import logging
import dataclasses
import pathlib
import copy
from uuid import UUID

import PIL, cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.autonotebook import tqdm
import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable.metadata import schema
from pixeltable.env import Env
from pixeltable import exceptions as exc
from pixeltable.type_system import ColumnType, StringType
from pixeltable.index import VectorIndex
from pixeltable.function import Function, FunctionRegistry
from pixeltable.utils.imgstore import ImageStore


_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')


class Column:
    """Representation of a column in the schema of a Table/DataFrame.
    """
    def __init__(
            self, name: str, col_type: Optional[ColumnType] = None,
            computed_with: Optional[Union['Expr', Callable]] = None,
            primary_key: bool = False, stored: Optional[bool] = None,
            indexed: bool = False,
            # these parameters aren't set by users
            col_id: Optional[int] = None,
            value_expr_str: Optional[str] = None):
        """Column constructor.

        Args:
            name: column name
            col_type: column type; can be None if the type can be derived from ``computed_with``
            computed_with: a callable or an Expr object that computes the column value
            primary_key: if True, this column is part of the primary key
            stored: determines whether a computed column is present in the stored table or recomputed on demand
            indexed: if True, this column has a nearest neighbor index (only valid for image columns)
            col_id: column ID (only used internally)
            value_expr_str: serialized form of ``computed_with`` (only used internally)

        Computed columns: those have a non-None ``computed_with`` argument

        - when constructed by the user: ``computed_with`` was constructed explicitly and is passed in;
          ``value_expr_str`` is None and col_type is None
        - when loaded from store: ``value_expr_str`` is the serialized form and col_type is set;
          ``computed_with`` is None

        ``computed_with`` is a Callable:

        - the callable's parameter names must correspond to existing columns in the table for which this Column
          is being used
        - ``col_type`` needs to be set to the callable's return type

        ``stored`` (only valid for computed image columns):

        - if True: the column is present in the stored table
        - if False: the column is not present in the stored table and recomputed during a query
        - if None: the system chooses for you (at present, this is always False, but this may change in the future)

        indexed: only valid for image columns; if true, maintains an NN index for this column
        """
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.Error(f"Invalid column name: '{name}'")
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

        # column in the stored table for the values of this Column
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
        if self.stored == False and self.is_computed and self.has_window_fn_call():
            raise exc.Error(
                f'Column {self.name}: stored={self.stored} not supported for columns computed with window functions:'
                f'\n{self.value_expr}')

    def has_window_fn_call(self) -> bool:
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
        assert self.stored is not None
        return self.stored

    def source(self) -> None:
        """
        If this is a computed col and the top-level expr is a function call, print the source, if possible.
        """
        from pixeltable import exprs
        if self.value_expr is None or not isinstance(self.value_expr, exprs.FunctionCall):
            return
        self.value_expr.fn.source()

    def create_sa_cols(self) -> None:
        """
        These need to be recreated for every new table schema version.
        """
        assert self.is_stored
        # computed cols store a NULL value when the computation has an error
        nullable = True if self.is_computed else self.col_type.nullable
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


class Dir(SchemaObject):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)

    @classmethod
    def display_name(cls) -> str:
        return 'directory'


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


class Table(SchemaObject):
    """Base class for tables."""
    def __init__(
            self, db_id: UUID, tbl_id: UUID, dir_id: UUID, name: str, version: int, params: Dict, cols: List[Column]):
        super().__init__(tbl_id)
        self.db_id = db_id
        self.dir_id = dir_id
        # TODO: this will be out-of-date after a rename()
        self.name = name
        # we create copies here because the Column objects might be owned by the caller
        self.cols = [copy.copy(col) for col in cols]
        for pos, col in enumerate(self.cols):
            if re.fullmatch(_ID_RE, col.name) is None:
                raise exc.Error(f"Invalid column name: '{col.name}'")
            assert col.id is not None
            col.tbl = self
        self.cols_by_name = {col.name: col for col in self.cols}
        self.cols_by_id = {col.id: col for col in self.cols}
        self.version = version
        self.parameters = TableParameters(**params)

        # we can't call _load_valid_rowids() here because the storage table may not exist yet
        self.valid_rowids: Set[int] = set()

        # if we're being initialized from stored metadata:
        # create Column.value_expr for computed cols now, prior to calling _create_sa_tbl()
        from pixeltable.exprs import Expr
        for col in self.cols:
            if col.value_expr_str is not None and col.value_expr is None:
                col.value_expr = Expr.deserialize(col.value_expr_str, self)

        # sqlalchemy-related metadata; used to insert and query the storage table
        self.sa_md = sql.MetaData()
        self._create_sa_tbl()
        self.is_dropped = False

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        for col in self.cols:
            col.tbl = self
            if col.value_expr is not None:
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
        """Update Column.dependent_cols for all cols referenced in col.value_expr.
        """
        from pixeltable.exprs import ColumnRef

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
        """Return a ColumnRef for the given column name.
        """
        if col_name not in self.cols_by_name:
            raise AttributeError(f'Column {col_name} unknown')
        col = self.cols_by_name[col_name]
        from pixeltable.exprs import ColumnRef, FrameColumnRef
        if self.is_frame_col(col):
            return FrameColumnRef(col)
        else:
            return ColumnRef(col)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        if isinstance(index, str):
            # basically <tbl>.<colname>
            return self.__getattr__(index)
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self)

    def show(self, *args, **kwargs) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().show(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table.
        """
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
        return f'tbl_{self.id.hex}'

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.Error('Table has been dropped')

    def _create_sa_tbl(self) -> None:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(schema.Table.MAX_VERSION))

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
    def _vector_idx_name(cls, tbl_id: UUID, col: Column) -> str:
        return f'{tbl_id.hex}_{col.id}'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def load_cols(cls, tbl_id: UUID, schema_version: int, session: orm.Session) -> List[Column]:
        """
        Returns loaded cols.
        """
        col_records = session.query(schema.SchemaColumn) \
            .where(schema.SchemaColumn.tbl_id == tbl_id) \
            .where(schema.SchemaColumn.schema_version == schema_version) \
            .order_by(schema.SchemaColumn.pos.asc()).all()
        cols = [
            Column(
                r.name, ColumnType.deserialize(r.col_type), primary_key=r.is_pk, stored=r.stored, col_id=r.col_id,
                value_expr_str=r.value_expr, indexed=r.is_indexed)
            for r in col_records
        ]
        for col in [col for col in cols if col.col_type.is_image_type()]:
            if col.is_indexed:
                col.set_idx(VectorIndex.load(cls._vector_idx_name(tbl_id, col), dim=512))
        return cols


class TableSnapshot(Table):
    def __init__(self, snapshot_record: schema.TableSnapshot, params: Dict, cols: List[Column]):
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
    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int
        num_values: int
        num_excs: int
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    """A :py:class:`Table` that can be modified.
    """
    def __init__(self, tbl_record: schema.Table, schema_version: int, cols: List[Column]):
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

    def add_column(self, col: Column, print_stats: bool = False) -> UpdateStatus:
        """Adds a column to the table.

        Args:
            col: The column to add.

        Returns:
            execution status

        Raises:
            Error: If the column name is invalid or already exists.

        Examples:
            Add an int column with ``None`` values:

            >>> tbl.add_column(Column('new_col', IntType()))

            For a table with int column ``x``, add a column that is the factorial of ``x``. Note that the names of
            the parameters of the ``computed_with`` Callable must correspond to existing column names (the column
            values are then passed as arguments to the Callable):

            >>> tbl.add_column(Column('factorial', IntType(), computed_with=lambda x: math.factorial(x)))

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees (note that in this case, the column type is inferred from the ``computed_with`` expression):

            >>> tbl.add_column(Column('rotated', computed_with=tbl.frame.rotate(90)))
            'added ...'
        """
        self._check_is_dropped()
        if re.fullmatch(_ID_RE, col.name) is None:
            raise exc.Error(f"Invalid column name: '{col.name}'")
        if col.name in self.cols_by_name:
            raise exc.Error(f'Column {col.name} already exists')
        assert self.next_col_id is not None
        col.tbl = self
        col.id = self.next_col_id
        self.next_col_id += 1

        if col.compute_func is not None:
            # create value_expr from compute_func
            self._create_value_expr(col, self.cols_by_name)
        if col.value_expr is not None:
            col.check_value_expr()
            self._record_value_expr(col)

        if col.stored is False and not (col.is_computed and col.col_type.is_image_type()):
            raise exc.Error(f'Column {col.name}: stored={col.stored} only applies to computed image columns')
        if col.stored is False and not (col.col_type.is_image_type() and not col.has_window_fn_call()):
            raise exc.Error(
                f'Column {col.name}: stored={col.stored} is not valid for image columns computed with a streaming function')
        if col.stored is None:
            col.stored = not(col.is_computed and col.col_type.is_image_type() and not col.has_window_fn_call())

        self.cols.append(col)
        self.cols_by_name[col.name] = col
        self.cols_by_id[col.id] = col

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({
                        schema.Table.current_version: self.version,
                        schema.Table.current_schema_version: self.schema_version,
                        schema.Table.next_col_id: self.next_col_id
                    })
                    .where(schema.Table.id == self.id))
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__)
                    .values(
                        tbl_id=self.id, schema_version=self.schema_version,
                        preceding_schema_version=preceding_schema_version))
            conn.execute(
                sql.insert(schema.ColumnHistory.__table__)
                    .values(tbl_id=self.id, col_id=col.id, schema_version_add=self.schema_version))
            self._create_col_md(conn)
            _logger.info(f'Added column {col.name} to table {self.name}, new version: {self.version}')

            if col.is_stored:
                stmt = f'ALTER TABLE {self.storage_name()} ADD COLUMN {col.storage_name()} {col.col_type.to_sql()}'
                conn.execute(sql.text(stmt))
                added_storage_cols = [col.storage_name()]
                if col.is_computed:
                    # we also need to create the errormsg and errortype storage cols
                    stmt = (f'ALTER TABLE {self.storage_name()} '
                            f'ADD COLUMN {col.errormsg_storage_name()} {StringType().to_sql()} DEFAULT NULL')
                    conn.execute(sql.text(stmt))
                    stmt = (f'ALTER TABLE {self.storage_name()} '
                            f'ADD COLUMN {col.errortype_storage_name()} {StringType().to_sql()} DEFAULT NULL')
                    conn.execute(sql.text(stmt))
                    added_storage_cols.extend([col.errormsg_storage_name(), col.errortype_storage_name()])
                self._create_sa_tbl()
                _logger.info(f'Added columns {added_storage_cols} to storage table {self.storage_name()}')

        if col.is_indexed:
            col.set_idx(VectorIndex.create(Table._vector_idx_name(self.id, col), 512))

        row_count = self.count()
        if row_count == 0:
            return self.UpdateStatus(0, 0, 0)
        if (not col.is_computed or not col.is_stored) and not col.is_indexed:
            return self.UpdateStatus(row_count, 0, 0)
        # compute values for the existing rows and compute embeddings, if this column is indexed;
        # for some reason, it's not possible to run the following updates in the same transaction as the one
        # that we just used to create the metadata (sqlalchemy hangs when exec() tries to run the query)
        from pixeltable.plan import Planner
        plan, value_expr_slot_idx, embedding_slot_idx = Planner.create_add_column_plan(self, col)
        plan.ctx.num_rows = row_count
        embeddings: List[np.ndarray] = []
        rowids: List[int] = []

        plan.open()
        with Env.get().engine.begin() as conn:
            try:
                num_excs = 0
                num_rows = 0
                for row_batch in plan:
                    num_rows += len(row_batch)
                    for result_row in row_batch:
                        if col.is_computed:
                            val = result_row.get_stored_val(value_expr_slot_idx)
                            if isinstance(val, Exception):
                                num_excs += 1
                                value_exc = val
                                # we store a NULL value and record the exception/exc type
                                error_type = type(value_exc).__name__
                                error_msg = str(value_exc)
                                conn.execute(
                                    sql.update(self.sa_tbl)
                                        .values({
                                            col.sa_col: None,
                                            col.sa_errortype_col: error_type,
                                            col.sa_errormsg_col: error_msg
                                        })
                                        .where(self.rowid_col == result_row.row_id)
                                        .where(self.v_min_col == result_row.v_min))
                            else:
                                conn.execute(
                                    sql.update(self.sa_tbl)
                                        .values({col.sa_col: val})
                                        .where(self.rowid_col == result_row.row_id)
                                        .where(self.v_min_col == result_row.v_min))
                        if col.is_indexed:
                            embeddings.append(result_row[embedding_slot_idx])
                            rowids.append(result_row.row_id)

                msg = f'added {row_count} column values with {num_excs} error{"" if num_excs == 1 else "s"}'
                print(msg)
                _logger.info(f'Column {col.name}: {msg}')
                if print_stats:
                    plan.ctx.profile.print(num_rows=num_rows)
                return self.UpdateStatus(row_count, row_count, num_excs, [col.name] if num_excs > 0 else [])
            except sql.exc.DBAPIError as e:
                self.drop_column(col.name)
                raise exc.Error(f'Error during SQL execution:\n{e}')
            finally:
                plan.close()

        if col.is_indexed:
            # update the index
            col.idx.add(embeddings, rowids)

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.

        Args:
            name: The name of the column to drop.

        Raises:
            Error: If the column does not exist or if it is referenced by a computed column.

        Example:
            >>> tbl.drop_column('factorial')
        """
        self._check_is_dropped()
        if name not in self.cols_by_name:
            raise exc.Error(f'Unknown column: {name}')
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
                sql.update(schema.Table.__table__)
                    .values({
                        schema.Table.parameters: dataclasses.asdict(self.parameters),
                        schema.Table.current_version: self.version,
                        schema.Table.current_schema_version: self.schema_version
                    })
                    .where(schema.Table.id == self.id))
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__)
                    .values(
                        tbl_id=self.id, schema_version=self.schema_version,
                        preceding_schema_version=preceding_schema_version))
            conn.execute(
                sql.update(schema.ColumnHistory.__table__)
                    .values({schema.ColumnHistory.schema_version_drop: self.schema_version})
                    .where(schema.ColumnHistory.tbl_id == self.id)
                    .where(schema.ColumnHistory.col_id == col.id))
            self._create_col_md(conn)
        self._create_sa_tbl()
        _logger.info(f'Dropped column {name} from table {self.name}, new version: {self.version}')

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist or if the new name is invalid or already exists.

        Example:
            >>> tbl.rename_column('factorial', 'fac')
        """
        self._check_is_dropped()
        if old_name not in self.cols_by_name:
            raise exc.Error(f'Unknown column: {old_name}')
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.Error(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise exc.Error(f'Column {new_name} already exists')
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
                sql.update(schema.Table.__table__)
                    .values({
                        schema.Table.current_version: self.version,
                        schema.Table.current_schema_version: self.schema_version
                    })
                    .where(schema.Table.id == self.id))
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__)
                    .values(tbl_id=self.id, schema_version=self.schema_version,
                            preceding_schema_version=preceding_schema_version))
            self._create_col_md(conn)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def _create_col_md(self, conn: sql.engine.base.Connection) -> None:
        for pos, c in enumerate(self.cols):
            value_expr_str = c.value_expr.serialize() if c.value_expr is not None else None
            conn.execute(
                sql.insert(schema.SchemaColumn.__table__)
                .values(
                    tbl_id=self.id, schema_version=self.version, col_id=c.id, pos=pos, name=c.name,
                    col_type=c.col_type.serialize(), is_pk=c.primary_key, value_expr=value_expr_str, stored=c.stored,
                    is_indexed=c.is_indexed))

    def _convert_to_stored(self, col: Column, val: Any, rowid: int) -> Any:
        """
        Convert column value 'val' into a store-compatible format, if needed:
        - images are stored as files
        - arrays are stored as serialized ndarrays
        """
        if col.col_type.is_image_type():
            # replace PIL.Image.Image with file path
            img = val
            img_path = ImageStore.get_path(self.id, col.id, self.version, rowid, 'jpg')
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

    def insert_rows(self, rows: List[List[Any]], columns: List[str] = [], print_stats: bool = False) -> UpdateStatus:
        """Insert rows into table.

        Args:
            rows: A list of rows to insert. Each row is a list of values, one for each column.
            columns: A list of column names that specify the columns present in ``rows``.
                If ``columns`` is empty, all columns are present in ``rows``.
            print_stats: If ``True``, print statistics about the cost of computed columns.

        Returns:
            execution status

        Raises:
            Error: If the number of columns in ``rows`` does not match the number of columns in the table or in ``columns``.

        Examples:
            Insert two rows into a table with three int columns ``a``, ``b``, and ``c``. Note that the ``columns``
            argument is required here because ``rows`` only contain two columns.

            >>> tbl.insert_rows([[1, 1], [2, 2]], columns=['a', 'b'])

            Assuming a table with columns ``video``, ``frame`` and ``frame_idx`` and set up for automatic frame extraction,
            insert a single row containing a video file path (the video contains 100 frames). The row will be expanded
            into 100 rows, one for each frame, and the ``frame`` and ``frame_idx`` columns will be populated accordingly.
            Note that the ``columns`` argument is unnecessary here because only the ``video`` column is required.

            >>> tbl.insert_rows([['/path/to/video.mp4']])

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
        return self.insert_pandas(pd_df, print_stats=print_stats)

    def _check_data(self, data: pd.DataFrame):
        """
        Make sure 'data' conforms to schema.
        """
        assert len(data) > 0
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = {col.name for col in self.cols if not col.col_type.nullable and col.value_expr is None}
        if self.extracts_frames():
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_col_id].name)
            reqd_col_names.discard(self.cols_by_id[self.parameters.frame_idx_col_id].name)
        given_col_names = set(data.columns)
        if not(reqd_col_names <= given_col_names):
            raise exc.Error(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')
        if not(given_col_names <= all_col_names):
            raise exc.Error(f'Unknown columns: {", ".join(given_col_names - all_col_names)}')
        computed_col_names = {col.name for col in self.cols if col.value_expr is not None}
        if self.extracts_frames():
            computed_col_names.add(self.cols_by_id[self.parameters.frame_col_id].name)
            computed_col_names.add(self.cols_by_id[self.parameters.frame_idx_col_id].name)
        if len(computed_col_names & given_col_names) > 0:
            raise exc.Error(
                f'Provided values for computed columns: {", ".join(computed_col_names & given_col_names)}')

        # check types
        provided_cols = [self.cols_by_name[name] for name in data.columns]
        for col in provided_cols:
            if col.col_type.is_string_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.Error(f'Column {col.name} requires string data but contains {data.dtypes[col.name]}')
            if col.col_type.is_int_type() and not pd.api.types.is_integer_dtype(data.dtypes[col.name]):
                raise exc.Error(f'Column {col.name} requires integer data but contains {data.dtypes[col.name]}')
            if col.col_type.is_float_type() and not pd.api.types.is_numeric_dtype(data.dtypes[col.name]):
                raise exc.Error(f'Column {col.name} requires numerical data but contains {data.dtypes[col.name]}')
            if col.col_type.is_bool_type() and not pd.api.types.is_bool_dtype(data.dtypes[col.name]):
                raise exc.Error(f'Column {col.name} requires boolean data but contains {data.dtypes[col.name]}')
            if col.col_type.is_timestamp_type() and not pd.api.types.is_datetime64_any_dtype(data.dtypes[col.name]):
                raise exc.Error(f'Column {col.name} requires datetime data but contains {data.dtypes[col.name]}')
            if col.col_type.is_json_type() and not pd.api.types.is_object_dtype(data.dtypes[col.name]):
                raise exc.Error(
                    f'Column {col.name} requires dictionary data but contains {data.dtypes[col.name]}')
            if col.col_type.is_array_type() and not pd.api.types.is_object_dtype(data.dtypes[col.name]):
                raise exc.Error(
                    f'Column {col.name} requires array data but contains {data.dtypes[col.name]}')
            if col.col_type.is_image_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.Error(
                    f'Column {col.name} requires local file paths but contains {data.dtypes[col.name]}')
            if col.col_type.is_video_type() and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.Error(
                    f'Column {col.name} requires local file paths but contains {data.dtypes[col.name]}')

        # check data
        data_cols = [self.cols_by_name[name] for name in data.columns]
        for col in data_cols:
            if not col.col_type.nullable:
                # check for nulls
                nulls = data[col.name].isna()
                max_val_idx = nulls.idxmax()
                if nulls[max_val_idx]:
                    raise exc.Error(
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
                        raise exc.Error(f'Column {col.name}: file does not exist: {path_str}')
                    except PIL.UnidentifiedImageError:
                        raise exc.Error(f'Column {col.name}: not a valid image file: {path_str}')

            # image cols: make sure file path points to a valid image file; build index if col is indexed
            if col.col_type.is_video_type():
                for _, path_str in data[col.name].items():
                    if path_str is None:
                        continue
                    path = pathlib.Path(path_str)
                    if not path.is_file():
                        raise exc.Error(f'Column {col.name}: file does not exist: {path_str}')
                    cap = cv2.VideoCapture(path_str)
                    success = cap.isOpened()
                    cap.release()
                    if not success:
                        raise exc.Error(f'Column {col.name}: could not open video file {path_str}')

            if col.col_type.is_json_type():
                for idx, d in data[col.name].items():
                    if d is not None and not isinstance(d, dict) and not isinstance(d, list):
                        raise exc.Error(
                            f'Value for column {col.name} in row {idx} requires a dictionary or list: {d} ')

    def insert_pandas(self, data: pd.DataFrame, print_stats: bool = False) -> UpdateStatus:
        """Insert data from pandas DataFrame into this table.

        If self.parameters.frame_src_col_id != None:

        - each row (containing a video) is expanded into one row per extracted frame (at the rate of the fps parameter)
        - parameters.frame_col_id is the image column that receives the extracted frame
        - parameters.frame_idx_col_id is the integer column that receives the frame index (starting at 0)
        """
        self._check_is_dropped()
        self._check_data(data)

        # we're creating a new version
        self.version += 1
        from pixeltable.plan import Planner
        plan, db_col_info, idx_col_info, num_values_per_row = Planner.create_insert_plan(self, data)
        plan.open()
        rows = next(plan)
        plan.close()

        # insert rows into table in batches
        start_row_id = self.next_row_id
        batch_size = 16
        progress_bar = tqdm(total=len(rows), desc='Inserting rows into table', unit='rows')
        with Env.get().engine.begin() as conn:
            num_excs = 0
            cols_with_excs: Set[int] = set()
            for batch_start_idx in range(0, len(rows), batch_size):
                # compute batch of rows and convert them into table rows
                table_rows: List[Dict[str, Any]] = []
                for row_idx in range(batch_start_idx, min(batch_start_idx + batch_size, len(rows))):
                    row = rows[row_idx]
                    table_row = {c.storage_name(): row.get_stored_val(slot_idx) for c, slot_idx in db_col_info}
                    table_row.update({'rowid': self.next_row_id, 'v_min': self.version})

                    # check for exceptions
                    for col in [c for c, _ in db_col_info if c.is_computed]:
                        val = table_row[col.storage_name()]
                        if isinstance(val, Exception):
                            # exceptions get stored in the errortype/-msg columns
                            num_excs += 1
                            cols_with_excs.add(col.id)
                            table_row[col.storage_name()] = None
                            table_row[col.errortype_storage_name()] = type(val).__name__
                            table_row[col.errormsg_storage_name()] = str(val)
                        else:
                            table_row[col.errortype_storage_name()] = None
                            table_row[col.errormsg_storage_name()] = None

                    self.next_row_id += 1
                    table_rows.append(table_row)
                    progress_bar.update(1)
                conn.execute(sql.insert(self.sa_tbl), table_rows)

            progress_bar.close()
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({schema.Table.current_version: self.version, schema.Table.next_row_id: self.next_row_id})
                    .where(schema.Table.id == self.id))

        if len(idx_col_info) > 0:
            # update image indices
            for col, slot_idx in tqdm(idx_col_info, desc='Updating image indices', unit='column'):
                embeddings = [row[slot_idx] for row in rows]
                col.idx.insert(np.asarray(embeddings), np.arange(start_row_id, self.next_row_id))

        if print_stats:
            plan.ctx.profile.print(num_rows=len(rows))
        self.valid_rowids.update(range(start_row_id, self.next_row_id))
        if num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = f'across {len(cols_with_excs)} column{"" if len(cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join([self.cols_by_id[id].name for id in cols_with_excs])})'
        msg = f'inserted {len(rows)} rows with {num_excs} error{"" if num_excs == 1 else "s"} {cols_with_excs_str}'
        print(msg)
        _logger.info(f'Table {self.name}: {msg}, new version {self.version}')
        status = self.UpdateStatus(
            len(rows), num_values_per_row * len(rows), num_excs, [self.cols_by_id[cid].name for cid in cols_with_excs])
        return status

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        if self.version == 0:
            raise exc.Error('Cannot revert version 0')
        # check if the current version is referenced by a snapshot
        with orm.Session(Env.get().engine, future=True) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.db_id == self.db_id) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .where(schema.TableSnapshot.tbl_version == self.version) \
                .scalar()
            if num_references > 0:
                raise exc.Error(
                    f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')

            conn = session.connection()
            # delete newly-added data
            ImageStore.delete(self.id, v_min=self.version)
            conn.execute(sql.delete(self.sa_tbl).where(self.sa_tbl.c.v_min == self.version))
            # revert new deletions
            conn.execute(
                sql.update(self.sa_tbl).values({self.sa_tbl.c.v_max: schema.Table.MAX_VERSION})
                    .where(self.sa_tbl.c.v_max == self.version))

            if self.version == self.schema_version:
                # the current version involved a schema change:
                # if the schema change was to add a column, we now need to drop it
                added_col_id = session.query(schema.ColumnHistory.col_id)\
                    .where(schema.ColumnHistory.tbl_id == self.id)\
                    .where(schema.ColumnHistory.schema_version_add == self.schema_version)\
                    .scalar()
                if added_col_id is not None:
                    # drop this newly-added column and its ColumnHistory record
                    c = self.cols_by_id[added_col_id]
                    stmt = f'ALTER TABLE {self.storage_name()} DROP COLUMN {c.storage_name()}'
                    conn.execute(sql.text(stmt))
                    conn.execute(
                        sql.delete(schema.ColumnHistory.__table__)
                            .where(schema.ColumnHistory.tbl_id == self.id)
                            .where(schema.ColumnHistory.col_id == added_col_id))

                # if the schema change was to drop a column, we now need to undo that
                dropped_col_id = session.query(schema.ColumnHistory.col_id) \
                    .where(schema.ColumnHistory.tbl_id == self.id) \
                    .where(schema.ColumnHistory.schema_version_drop == self.schema_version) \
                    .scalar()
                if dropped_col_id is not None:
                    # fix up the ColumnHistory record
                    conn.execute(
                        sql.update(schema.ColumnHistory.__table__)
                            .values({schema.ColumnHistory.schema_version_drop: None})
                            .where(schema.ColumnHistory.tbl_id == self.id)
                            .where(schema.ColumnHistory.col_id == dropped_col_id))

                # we need to determine the preceding schema version and reload the schema
                preceding_schema_version = session.query(schema.TableSchemaVersion.preceding_schema_version) \
                    .where(schema.TableSchemaVersion.tbl_id == self.id) \
                    .where(schema.TableSchemaVersion.schema_version == self.schema_version) \
                    .scalar()
                self.cols = self.load_cols(self.id, preceding_schema_version, session)
                for c in self.cols:
                    c.tbl = self

                # drop all SchemaColumn records for this schema version prior to deleting from TableSchemaVersion
                # (to avoid FK violations)
                conn.execute(
                    sql.delete(schema.SchemaColumn.__table__)
                        .where(schema.SchemaColumn.tbl_id == self.id)
                        .where(schema.SchemaColumn.schema_version == self.schema_version))
                conn.execute(
                    sql.delete(schema.TableSchemaVersion.__table__)
                        .where(schema.TableSchemaVersion.tbl_id == self.id)
                        .where(schema.TableSchemaVersion.schema_version == self.schema_version))
                self.schema_version = preceding_schema_version

            self.version -= 1
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({
                        schema.Table.current_version: self.version,
                        schema.Table.current_schema_version: self.schema_version
                    })
                    .where(schema.Table.id == self.id))

            session.commit()
            _logger.info(f'Table {self.name}: reverted to version {self.version}')

    # MODULE-LOCAL, NOT PUBLIC
    def rename(self, new_name: str) -> None:
        self._check_is_dropped()
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(schema.Table.__table__).values({schema.Table.name: new_name})
                    .where(schema.Table.id == self.id))

    # MODULE-LOCAL, NOT PUBLIC
    def drop(self) -> None:
        self._check_is_dropped()
        self.is_dropped = True

        with orm.Session(Env.get().engine, future=True) as session:
            # check if we have snapshots
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.db_id == self.db_id) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .scalar()
            if num_references == 0:
                # we can delete this table altogether
                ImageStore.delete(self.id)
                conn = session.connection()
                conn.execute(sql.delete(schema.SchemaColumn.__table__).where(schema.SchemaColumn.tbl_id == self.id))
                conn.execute(sql.delete(schema.ColumnHistory.__table__).where(schema.ColumnHistory.tbl_id == self.id))
                conn.execute(
                    sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == self.id))
                conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == self.id))
                self.sa_md.drop_all(bind=conn)
                session.commit()
                return

        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(schema.Table.__table__).values({schema.Table.is_mutable: False})
                    .where(schema.Table.id == self.id))

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
        cls, db_id: UUID, dir_id: UUID, name: str, cols: List[Column],
        num_retained_versions: int,
        extract_frames_from: Optional[str], extracted_frame_col: Optional[str], extracted_frame_idx_col: Optional[str],
        extracted_fps: Optional[int],
    ) -> 'MutableTable':
        # create a copy here so we can modify it
        cols = [copy.copy(c) for c in cols]
        # make sure col names are unique (within the table) and assign ids
        cols_by_name: Dict[str, Column] = {}
        for pos, col in enumerate(cols):
            if col.name in cols_by_name:
                raise exc.Error(f'Duplicate column: {col.name}')
            col.id = pos
            cols_by_name[col.name] = col
            if col.value_expr is None and col.compute_func is not None:
                cls._create_value_expr(col, cols_by_name)
            if col.is_computed:
                col.check_value_expr()
            if col.stored is True and col.name == extracted_frame_col:
                raise exc.Error(f'Column {col.name}: extracted frame column cannot be stored')
            if col.stored is False and not(col.is_computed and col.col_type.is_image_type()) \
                    and col.name != extracted_frame_col:
                raise exc.Error(f'Column {col.name}: stored={col.stored} only applies to computed image columns')
            if col.stored is None:
                if col.is_computed and col.col_type.is_image_type():
                    col.stored = False
                elif col.name == extracted_frame_col:
                    col.stored = False
                else:
                    col.stored = True

        # check frame extraction params, if present
        if extract_frames_from is not None:
            assert extracted_frame_col is not None and extracted_frame_idx_col is not None and extracted_fps is not None
            if extract_frames_from is not None and extract_frames_from not in cols_by_name:
                raise exc.Error(f'Unknown column in extract_frames_from: {extract_frames_from}')
            col_type = cols_by_name[extract_frames_from].col_type
            is_nullable = cols_by_name[extract_frames_from].col_type.nullable
            if not col_type.is_video_type():
                raise exc.Error(
                    f'extract_frames_from requires the name of a column of type video, but {extract_frames_from} has '
                    f'type {col_type}')

            if extracted_frame_col is not None and extracted_frame_col not in cols_by_name:
                raise exc.Error(f'Unknown column in extracted_frame_col: {extracted_frame_col}')
            col_type = cols_by_name[extracted_frame_col].col_type
            if not col_type.is_image_type():
                raise exc.Error(
                    f'extracted_frame_col requires the name of a column of type image, but {extracted_frame_col} has '
                    f'type {col_type}')
            # the src column determines whether the frame column is nullable
            cols_by_name[extracted_frame_col].col_type.nullable = is_nullable
            # extracted frames are never stored
            cols_by_name[extracted_frame_col].stored = False

            if extracted_frame_idx_col is not None and extracted_frame_idx_col not in cols_by_name:
                raise exc.Error(f'Unknown column in extracted_frame_idx_col: {extracted_frame_idx_col}')
            col_type = cols_by_name[extracted_frame_idx_col].col_type
            if not col_type.is_int_type():
                raise exc.Error(
                    f'extracted_frame_idx_col requires the name of a column of type int, but {extracted_frame_idx_col} '
                    f'has type {col_type}')
            # the src column determines whether the frame idx column is nullable
            cols_by_name[extracted_frame_idx_col].col_type.nullable = is_nullable

        params = TableParameters(
            num_retained_versions,
            cols_by_name[extract_frames_from].id if extract_frames_from is not None else -1,
            cols_by_name[extracted_frame_col].id if extracted_frame_col is not None else -1,
            cols_by_name[extracted_frame_idx_col].id if extracted_frame_idx_col is not None else -1,
            extracted_fps)

        with orm.Session(Env.get().engine, future=True) as session:
            tbl_record = schema.Table(
                db_id=db_id, dir_id=dir_id, name=name, parameters=dataclasses.asdict(params), current_version=0,
                current_schema_version=0, is_mutable=True, next_col_id=len(cols), next_row_id=0)
            session.add(tbl_record)
            session.flush()  # sets tbl_record.id

            tbl_version_record = schema.TableSchemaVersion(
                tbl_id=tbl_record.id, schema_version=0, preceding_schema_version=0)
            session.add(tbl_version_record)
            session.flush()  # avoid FK violations in Postgres

            cols_by_name: Dict[str, Column] = {}  # records the cols we have seen so far
            for pos, col in enumerate(cols):
                session.add(schema.ColumnHistory(tbl_id=tbl_record.id, col_id=col.id, schema_version_add=0))
                session.flush()  # avoid FK violations in Postgres
                # Column.dependent_cols for existing cols is wrong at this point, but Table.init() will set it correctly
                value_expr_str = col.value_expr.serialize() if col.value_expr is not None else None
                session.add(
                    schema.SchemaColumn(
                        tbl_id=tbl_record.id, schema_version=0, col_id=col.id, pos=pos, name=col.name,
                        col_type=col.col_type.serialize(), is_pk=col.primary_key, value_expr=value_expr_str,
                        stored=col.stored, is_indexed=col.is_indexed)
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
            _logger.info(f'created table {name}, id={tbl_record.id}')
            return tbl


class Path:
    def __init__(self, path: str, empty_is_valid: bool=False):
        if path == '' and not empty_is_valid or path != '' and re.fullmatch(_PATH_RE, path) is None:
            raise exc.Error(f"Invalid path format: '{path}'")
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
    def check_is_valid(self, path: Path, expected: Optional[Type[SchemaObject]]) -> None:
        """Check that path is valid and that the object at path has the expected type.

        Args:
            path: path to check
            expected: expected type of object at path or None if object should not exist

        Raises:
            Error if path is invalid or object at path has wrong type
        """
        path_str = str(path)
        # check for existence
        if expected is not None:
            if path_str not in self.paths:
                raise exc.Error(f'{path_str} does not exist')
            obj = self.paths[path_str]
            if not isinstance(obj, expected):
                raise exc.Error(f'{path_str} needs to be a {expected.display_name()}')
        if expected is None and path_str in self.paths:
            raise exc.Error(f"'{path_str}' already exists")
        # check for containing directory
        parent_path = path.parent
        if str(parent_path) not in self.paths:
            raise exc.Error(f'Directory {str(parent_path)} does not exist')
        parent = self.paths[str(parent_path)]
        if not isinstance(parent, Dir):
            raise exc.Error(f'{str(parent_path)} is a {type(parent).display_name()}, not a directory')

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
    """Handle to a database.

    Use this handle to create and manage tables, functions, and directories in the database.
    """
    def __init__(self, db_id: UUID, name: str):
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
    ) -> MutableTable:
        """Create a new table in the database.

        Args:
            path_str: Path to the table.
            schema: List of Columns in the table.
            num_retained_versions: Number of versions of the table to retain.
            extract_frames_from: Name of the video column from which to extract frames.
            extracted_frame_col: Name of the image column in which to store the extracted frames.
            extracted_frame_idx_col: Name of the int column in which to store the frame indices.
            extracted_fps: Frame rate at which to extract frames. 0: extract all frames.

        Returns:
            The newly created table.

        Raises:
            Error: if the path already exists or is invalid.

        Examples:
            Create a table with an int and a string column:

            >>> table = db.create_table('my_table', schema=[Column('col1', IntType()), Column('col2', StringType())])

            Create a table to store videos with automatic frame extraction. This requires a minimum of 3 columns:
            a video column, an image column to store the extracted frames, and an int column to store the frame
            indices.

            >>> table = db.create_table('my_table',
            ... schema=[Column('video', VideoType()), Column('frame', ImageType()), Column('frame_idx', IntType())],
            ... extract_frames_from='video', extracted_frame_col='frame', extracted_frame_idx_col='frame_idx',
            ... extracted_fps=1)
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None)
        dir = self.paths[path.parent]

        # make sure frame extraction params are either fully present or absent
        frame_extraction_param_count = int(extract_frames_from is not None) + int(extracted_frame_col is not None)\
            + int(extracted_frame_idx_col is not None) + int(extracted_fps is not None)
        if frame_extraction_param_count != 0 and frame_extraction_param_count != 4:
            raise exc.Error(
                'Frame extraction requires that all parameters (extract_frames_from, extracted_frame_col, '
                'extracted_frame_idx_col, extracted_fps) be specified')
        if extracted_fps is not None and extracted_fps < 0:
            raise exc.Error('extracted_fps must be >= 0')
        tbl = MutableTable.create(
            self.id, dir.id, path.name, schema, num_retained_versions, extract_frames_from, extracted_frame_col,
            extracted_frame_idx_col, extracted_fps)
        self.paths[path] = tbl
        _logger.info(f'Created table {path_str}')
        return tbl

    def get_table(self, path: str) -> Table:
        """Get a handle to a table (regular or snapshot) from the database.

        Args:
            path: Path to the table.

        Returns:
            A :py:class:`MutableTable` or :py:class:`TableSnapshot` object.

        Raises:
            Error: If the path does not exist or does not designate a table.

        Example:
            Get handle for a table in the top-level directory:

            >>> table = db.get_table('my_table')

            For a table in a subdirectory:

            >>> table = db.get_table('subdir.my_table')

            For a snapshot in the top-level directory:

            >>> table = db.get_table('my_snapshot')
        """
        p = Path(path)
        self.paths.check_is_valid(p, expected=Table)
        obj = self.paths[p]
        assert isinstance(obj, Table)
        return obj

    def rename_table(self, path_str: str, new_name: str) -> None:
        """Rename a table in the database.

        Args:
            path_str: Path to the table.
            new_name: New name for the table.

        Raises:
            Error: If the path does not exist or does not designate a table.

        Example:
            >>> db.rename_table('my_table', 'new_name_for_same_table')
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=MutableTable)
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.Error(f"Invalid table name: '{new_name}'")
        new_path = path.parent.append(new_name)
        self.paths.check_is_valid(new_path, expected=None)

        tbl = self.paths[path]
        assert isinstance(tbl, MutableTable)
        del self.paths[path]
        self.paths[new_path] = tbl
        tbl.rename(new_name)
        _logger.info(f'Renamed table {path_str} to {str(new_path)}')

    def move_table(self, tbl_path: str, dir_path: str) -> None:
        """Move a table to a new directory.

        .. warning::
            Not implemented yet.

        Args:
            tbl_path: Path to the table.
            dir_path: Path to the new directory.

        Raises:
            UnknownEntityError: If the path does not exist or does not designate a table.
        """
        pass

    def list_tables(self, dir_path: str = '', recursive: bool = True) -> List[str]:
        """List the tables in a directory.

        Args:
            dir_path: Path to the directory. Defaults to the root directory.
            recursive: Whether to list tables in subdirectories as well.

        Returns:
            A list of table paths.

        Raises:
            Error: If the path does not exist or does not designate a directory.

        Examples:
            List tables in top-level directory:

            >>> db.list_tables()
            ['my_table', ...]

            List tables in 'dir1':

            >>> db.list_tables('dir1')
            [...]
        """
        assert dir_path is not None
        path = Path(dir_path, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=Dir)
        return [str(p) for p in self.paths.get_children(path, child_type=Table, recursive=recursive)]

    def drop_table(self, path_str: str, force: bool = False, ignore_errors: bool = False) -> None:
        """Drop a table from the database.

        Args:
            path_str: Path to the table.
            force: Whether to drop the table even if it has unsaved changes.
            ignore_errors: Whether to ignore errors if the table does not exist.

        Raises:
            Error: If the path does not exist or does not designate a table and ignore_errors is False.

        Example:
            >>> db.drop_table('my_table')
        """
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
        _logger.info(f'Dropped table {path_str}')

    def create_snapshot(self, snapshot_path: str, tbl_path: str) -> None:
        """Create a snapshot of a table.

        Args:
            snapshot_path: Path to the snapshot.
            tbl_path: Path to the table.

        Raises:
            Error: If snapshot_path already exists or the parent does not exist.
        """
        snapshot_path_obj = Path(snapshot_path)
        self.paths.check_is_valid(snapshot_path_obj, expected=None)
        tbl_path_obj = Path(tbl_path)
        self.paths.check_is_valid(tbl_path_obj, expected=MutableTable)
        tbl = self.paths[tbl_path_obj]
        assert isinstance(tbl, MutableTable)

        with orm.Session(Env.get().engine, future=True) as session:
            dir = self.paths[snapshot_path_obj.parent]
            snapshot_record = schema.TableSnapshot(
                db_id=self.id, dir_id=dir.id, name=tbl.name, tbl_id=tbl.id, tbl_version=tbl.version,
                tbl_schema_version=tbl.schema_version)
            session.add(snapshot_record)
            session.flush()
            assert snapshot_record.id is not None
            cols = Table.load_cols(tbl.id, tbl.schema_version, session)
            snapshot = TableSnapshot(snapshot_record, dataclasses.asdict(tbl.parameters), cols)
            self.paths[snapshot_path_obj] = snapshot

            session.commit()

    def create_dir(self, path_str: str) -> None:
        """Create a directory.

        Args:
            path_str: Path to the directory.

        Raises:
            Error: If the path already exists or the parent is not a directory.

        Examples:
            >>> db.create_dir('my_dir')

            Create a subdirectory:

            >>> db.create_dir('my_dir.sub_dir')
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None)
        with orm.Session(Env.get().engine, future=True) as session:
            dir_record = schema.Dir(db_id=self.id, path=path_str)
            session.add(dir_record)
            session.flush()
            assert dir_record.id is not None
            self.paths[path] = Dir(dir_record.id)
            session.commit()
            _logger.info(f'Created directory {path_str}')

    def rm_dir(self, path_str: str) -> None:
        """Remove a directory.

        Args:
            path_str: Path to the directory.

        Raises:
            Error: If the path does not exist or does not designate a directory or if the directory is not empty.

        Examples:
            >>> db.rm_dir('my_dir')

            Remove a subdirectory:

            >>> db.rm_dir('my_dir.sub_dir')
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Dir)

        # make sure it's empty
        if len(self.paths.get_children(path, child_type=None, recursive=True)) > 0:
            raise exc.Error(f'Directory {path_str} is not empty')
        # TODO: figure out how to make force=True work in the presence of snapshots
#        # delete tables
#        for tbl_path in self.paths.get_children(path, child_type=Table, recursive=True):
#            self.drop_table(str(tbl_path), force=True)
#        # rm subdirs
#        for dir_path in self.paths.get_children(path, child_type=Dir, recursive=False):
#            self.rm_dir(str(dir_path), force=True)

        with Env.get().engine.begin() as conn:
            dir = self.paths[path]
            conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.id == dir.id))
        del self.paths[path]
        _logger.info(f'Removed directory {path_str}')

    def list_dirs(self, path_str: str = '', recursive: bool = True) -> List[str]:
        """List the directories in a directory.

        Args:
            path_str: Path to the directory.
            recursive: Whether to list subdirectories recursively.

        Returns:
            List of directory paths.

        Raises:
            Error: If the path does not exist or does not designate a directory.

        Example:
            >>> db.list_dirs('my_dir', recursive=True)
            ['my_dir', 'my_dir.sub_dir1']
        """
        path = Path(path_str, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=Dir)
        return [str(p) for p in self.paths.get_children(path, child_type=Dir, recursive=recursive)]

    def create_function(self, path_str: str, func: Function) -> None:
        """Create a stored function.

        Args:
            path_str: path where the function gets stored
            func: previously created Function object

        Raises:
            Error: if the path already exists or the parent is not a directory
        Examples:
            Create a function ``detect()`` that takes an image and returns a JSON object, and store it in ``my_dir``:

            >>> pt.function(param_types=[ImageType()], return_type=JsonType())
            ... def detect(img):
            ... ...
            >>> db.create_function('my_dir.detect', detect)
        """
        if func.is_library_function:
            raise exc.Error(f'Cannot create a named function for a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None)
        dir = self.paths[path.parent]

        FunctionRegistry.get().create_function(func, self.id, dir.id, path.name)
        self.paths[path] = NamedFunction(func.id, dir.id, path.name)
        func.md.fqn = f'{self.name}.{path}'
        _logger.info(f'Created function {path_str}')

    def rename_function(self, path_str: str, new_path_str: str) -> None:
        """Assign a new name and/or move the function to a different directory.

        Args:
            path_str: path to the function to be renamed
            new_path_str: new path for the function

        Raises:
            Error: if the path does not exist or new_path already exists

        Examples:
            Rename the stored function ``my_dir.detect()`` to ``detect2()``:

            >>> db.rename_function('my_dir.detect', 'my_dir.detect2')

            Move the stored function ``my_dir.detect()`` to the top-level directory:

            >>> db.rename_function('my_dir.detect', 'detect')
        """
        path = Path(path_str)
        new_path = Path(new_path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        self.paths.check_is_valid(new_path, expected=None)
        named_fn = self.paths[path]
        new_dir = self.paths[new_path.parent]
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.update(schema.Function.__table__)
                    .values({
                        schema.Function.dir_id: new_dir.id,
                        schema.Function.name: new_path.name,
                    })
                    .where(schema.Function.id == named_fn.id))
        del self.paths[path]
        self.paths[new_path] = named_fn
        func = FunctionRegistry.get().get_function(id=named_fn.id)
        func.md.fqn = f'{self.name}.{new_path}'
        _logger.info(f'Renamed function {path_str} to {new_path_str}')

    def update_function(self, path_str: str, func: Function) -> None:
        """Update the implementation of a stored function.

        Args:
            path_str: path to the function to be updated
            func: new function implementation

        Raises:
            Error: if the path does not exist or ``func`` has a different signature than the stored function.
        """
        if func.is_library_function:
            raise exc.Error(f'Cannot update a named function to a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        f = FunctionRegistry.get().get_function(id=named_fn.id)
        if f.md.signature != func.md.signature:
            raise exc.Error(
                f'The function signature cannot be changed. The existing signature is {f.md.signature}')
        if f.is_aggregate != func.is_aggregate:
            raise exc.Error(f'Cannot change an aggregate function into a non-aggregate function and vice versa')
        FunctionRegistry.get().update_function(named_fn.id, func)
        _logger.info(f'Updated function {path_str}')

    def get_function(self, path_str: str) -> Function:
        """Get a handle to a stored function.

        Args:
            path_str: path to the function

        Returns:
            Function object

        Raises:
            Error: if the path does not exist or is not a function

        Example:
            >>> detect = db.get_function('my_dir.detect')
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        assert isinstance(named_fn, NamedFunction)
        func = FunctionRegistry.get().get_function(id=named_fn.id)
        func.md.fqn = f'{self.name}.{path}'
        return func

    def drop_function(self, path_str: str, ignore_errors: bool = False) -> None:
        """Deletes stored function.

        Args:
            path_str: path to the function
            ignore_errors: if True, does not raise if the function does not exist

        Raises:
            Error: if the path does not exist or is not a function

        Example:
            >>> db.drop_function('my_dir.detect')
        """
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=NamedFunction)
        except exc.Error as e:
            if ignore_errors:
                return
            else:
                raise e
        named_fn = self.paths[path]
        FunctionRegistry.get().delete_function(named_fn.id)
        del self.paths[path]
        _logger.info(f'Dropped function {path_str}')

    def _load_dirs(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(Env.get().engine, future=True) as session:
            for dir_record in session.query(schema.Dir).where(schema.Dir.db_id == self.id).all():
                result[dir_record.path] = Dir(dir_record.id)
        return result

    def _load_tables(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(Env.get().engine, future=True) as session:
            # load all reachable (= mutable) tables
            q = session.query(schema.Table, schema.Dir.path) \
                .join(schema.Dir)\
                .where(schema.Table.db_id == self.id) \
                .where(schema.Table.is_mutable == True)
            for tbl_record, dir_path in q.all():
                cols = Table.load_cols(
                    tbl_record.id, tbl_record.current_schema_version, session)
                tbl = MutableTable(tbl_record, tbl_record.current_schema_version, cols)
                tbl._load_valid_rowids()  # TODO: move this someplace more appropriate
                path = Path(dir_path, empty_is_valid=True).append(tbl_record.name)
                result[str(path)] = tbl

            # load all table snapshots
            q = session.query(schema.TableSnapshot, schema.Dir.path, schema.Table.parameters) \
                .select_from(schema.TableSnapshot) \
                .join(schema.Table) \
                .join(schema.Dir) \
                .where(schema.TableSnapshot.db_id == self.id)
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
        with orm.Session(Env.get().engine, future=True) as session:
            # load all reachable (= mutable) tables
            q = session.query(schema.Function.id, schema.Function.dir_id, schema.Function.name, schema.Dir.path) \
                .join(schema.Dir) \
                .where(schema.Function.db_id == self.id)
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
        with orm.Session(Env.get().engine, future=True) as session:
            # check for duplicate name
            is_duplicate = session.query(sql.func.count(schema.Db.id)).where(schema.Db.name == name).scalar() > 0
            if is_duplicate:
                raise exc.Error(f"Db '{name}' already exists")

            db_record = schema.Db(name=name)
            session.add(db_record)
            session.flush()
            assert db_record.id is not None
            db_id = db_record.id
            # also create a top-level directory, so that every schema object has a directory
            dir_record = schema.Dir(db_id=db_id, path='')
            session.add(dir_record)
            session.flush()
            session.commit()
            _logger.info(f'Created db {name}')
        assert db_id is not None
        return Db(db_id, name)

    @classmethod
    def load(cls, name: str) -> 'Db':
        """Load db by name.

        Raises:
            Error: if db does not exist or the name is invalid
        """
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.Error(f"Invalid db name: '{name}'")
        with orm.Session(Env.get().engine, future=True) as session:
            try:
                db_record = session.query(schema.Db).where(schema.Db.name == name).one()
                return Db(db_record.id, db_record.name)
            except sql.exc.NoResultFound:
                raise exc.Error(f'Db {name} does not exist')

    def delete(self) -> None:
        """Delete db and all associated data.

        :meta private:
        """
        with Env.get().engine.begin() as conn:
            conn.execute(sql.delete(schema.TableSnapshot.__table__).where(schema.TableSnapshot.db_id == self.id))
            tbls_stmt = sql.select(schema.Table.id).where(schema.Table.db_id == self.id)
            conn.execute(sql.delete(schema.SchemaColumn.__table__).where(schema.SchemaColumn.tbl_id.in_(tbls_stmt)))
            conn.execute(sql.delete(schema.ColumnHistory.__table__).where(schema.ColumnHistory.tbl_id.in_(tbls_stmt)))
            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id.in_(tbls_stmt)))
            conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.db_id == self.id))
            conn.execute(sql.delete(schema.Function.__table__).where(schema.Function.db_id == self.id))
            conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.db_id == self.id))
            conn.execute(sql.delete(schema.Db.__table__).where(schema.Db.id == self.id))
            # delete all data tables
            # TODO: also deleted generated images
            for tbl in self.paths.get(MutableTable):
                tbl.sa_md.drop_all(bind=conn)
