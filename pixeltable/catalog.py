from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import pathlib
import re
from typing import Optional, List, Set, Dict, Any, Type, Union, Callable, Tuple
from uuid import UUID
from abc import abstractmethod
import datetime

import PIL
import cv2
import numpy as np
import pandas as pd
import sqlalchemy as sql
import sqlalchemy.orm as orm
from PIL import Image
from tqdm.autonotebook import tqdm

from pixeltable import exceptions as exc
from pixeltable.env import Env
from pixeltable.exec import ColumnInfo
from pixeltable.function import Function
from pixeltable.index import VectorIndex
from pixeltable.metadata import schema
from pixeltable.type_system import ColumnType, StringType
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
            col_id: Optional[int] = None):
        """Column constructor.

        Args:
            name: column name
            col_type: column type; can be None if the type can be derived from ``computed_with``
            computed_with: a callable or an Expr object that computes the column value
            primary_key: if True, this column is part of the primary key
            stored: determines whether a computed column is present in the stored table or recomputed on demand
            indexed: if True, this column has a nearest neighbor index (only valid for image columns)
            col_id: column ID (only used internally)

        Computed columns: those have a non-None ``computed_with`` argument

        - when constructed by the user: ``computed_with`` was constructed explicitly and is passed in;
          col_type is None
        - when loaded from md store: ``computed_with`` is set and col_type is set

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

        if col_type is not None:
            self.col_type = col_type
        assert self.col_type is not None

        self.stored = stored
        self.dependent_cols: List[Column] = []  # cols with value_exprs that reference us; set by Table
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

    @classmethod
    def from_md(cls, col_id: int, md: schema.SchemaColumn, tbl: Table) -> Column:
        """Construct a Column from metadata.

        Leaves out value_expr, because that requires Table.cols to be complete.
        """
        col = cls(
            md.name, col_type=ColumnType.from_dict(md.col_type), primary_key=md.is_pk,
            stored=md.stored, indexed=md.is_indexed, col_id=col_id)
        col.tbl = tbl
        return col

    def check_value_expr(self) -> None:
        assert self.value_expr is not None
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
        return self.compute_func is not None or self.value_expr is not None

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


class SchemaObject:
    """
    Base class of all addressable objects within a Db.
    Each object has an id, a name and a parent directory.
    """
    def __init__(self, obj_id: UUID, name: str, dir_id: Optional[UUID]):
        self.id = obj_id
        self.name = name
        self.dir_id = dir_id

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    @property
    def fqn(self) -> str:
        return f'{self.parent_dir().fqn}.{self.name}'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self.name = new_name
        self.dir_id = new_dir_id


class Dir(SchemaObject):
    def __init__(self, id: UUID, parent_id: UUID, name: str):
        super().__init__(id, name, parent_id)

    @classmethod
    def display_name(cls) -> str:
        return 'directory'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            dir_md = schema.DirMd(name=new_name)
            conn.execute(
                sql.update(schema.Dir.__table__)
                .values({schema.Dir.parent_id: self.dir_id, schema.Dir.md: dataclasses.asdict(dir_md)})
                .where(schema.Dir.id == self.id))


class NamedFunction(SchemaObject):
    """
    Contains references to functions that are named and have a path.
    The Function itself is stored in the FunctionRegistry.
    """
    def __init__(self, id: UUID, dir_id: UUID, name: str):
        super().__init__(id, name, dir_id)

    @classmethod
    def display_name(cls) -> str:
        return 'function'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Function.__table__} "
                f"SET {schema.Function.dir_id.name} = :new_dir_id, {schema.Function.md.name}['name'] = :new_name "
                f"WHERE {schema.Function.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})


class Table(SchemaObject):
    """Base class for tables."""
    def __init__(self, id: UUID, dir_id: UUID, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd):
        self.tbl_md = tbl_md
        super().__init__(id, self.tbl_md.name, dir_id)
        self._set_cols(schema_version_md)

        # we can't call _load_valid_rowids() here because the storage table may not exist yet
        self.valid_rowids: Set[int] = set()

        # sqlalchemy-related metadata; used to insert and query the storage table
        self.sa_md = sql.MetaData()
        self._create_sa_tbl()
        self.is_dropped = False

    def _set_cols(self, schema_version_md: schema.TableSchemaVersionMd) -> None:
        self.cols = [Column.from_md(col_id, col_md, self) for col_id, col_md in schema_version_md.columns.items()]
        self.cols_by_name = {col.name: col for col in self.cols}
        self.cols_by_id = {col.id: col for col in self.cols}

        # make sure to traverse columns ordered by position = order in which cols were created;
        # this guarantees that references always point backwards
        from pixeltable import exprs
        for col, col_md in zip(self.cols, schema_version_md.columns.values()):
            col.tbl = self
            if col_md.value_expr is not None:
                col.value_expr = exprs.Expr.from_dict(col_md.value_expr, self)
                self._record_value_expr(col)

        for col in [col for col in self.cols if col.col_type.is_image_type()]:
            if col.is_indexed:
                col.set_idx(VectorIndex.load(self._vector_idx_name(self.id, col), dim=512))

    @property
    def version(self) -> int:
        return self.tbl_md.current_version

    @property
    def schema_version(self) -> int:
        return self.tbl_md.current_schema_version

    @property
    def next_row_id(self) -> int:
        return self.tbl_md.next_row_id

    @property
    def parameters(self) -> int:
        return self.tbl_md.parameters

    def extracts_frames(self) -> bool:
        return self.tbl_md.parameters.frame_col_id != -1

    def is_frame_col(self, c: Column) -> bool:
        return c.id == self.tbl_md.parameters.frame_col_id

    def frame_src_col(self) -> Optional[Column]:
        """
        Return the frame src col, or None if not applicable.
        """
        if self.tbl_md.parameters.frame_src_col_id == -1:
            return None
        return self.cols_by_id[self.tbl_md.parameters.frame_src_col_id]

    def frame_idx_col(self) -> Optional[Column]:
        """
        Return the frame idx col, or None if not applicable.
        """
        if self.tbl_md.parameters.frame_idx_col_id == -1:
            return None
        return self.cols_by_id[self.tbl_md.parameters.frame_idx_col_id]

    def _record_value_expr(self, col: Column) -> None:
        """Update Column.dependent_cols for all cols referenced in col.value_expr.
        """
        assert col.value_expr is not None
        from pixeltable.exprs import ColumnRef
        refd_col_ids = [e.col.id for e in col.value_expr.subexprs(expr_class=ColumnRef)]
        refd_cols = [self.cols_by_id[id] for id in refd_col_ids]
        for refd_col in refd_cols:
            refd_col.dependent_cols.append(col)

    def _get_dependent_cols(self, cols: List[Column]) -> List[Column]:
        """
        Return the list of cols that transivitely depend on any of the given cols.
        """
        if len(cols) == 0:
            return []
        result: List[Column] = []
        for col in cols:
            result.extend(col.dependent_cols)
        result = [self.cols_by_id[id] for id in set([c.id for c in result])]
        return result + self._get_dependent_cols(result)

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

    def select(self, *items: 'exprs.Expr') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).select(*items)

    def where(self, pred: 'exprs.Predicate') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).where(pred)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).order_by(*items, asc=asc)

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
        if self.tbl_md.parameters.frame_col_id == -1:
            return None
        return self.cols_by_id[self.tbl_md.parameters.frame_col_id]

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


class TableSnapshot(Table):
    def __init__(self, snapshot_record: schema.TableSnapshot, schema_version_record: schema.TableSchemaVersion):
        # the id of this SchemaObject is TableSnapshot.tbl_id, not TableSnapshot.id: we use tbl_id to construct
        # the name of the data table
        snapshot_md = schema.md_from_dict(schema.TableMd, snapshot_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        super().__init__(snapshot_record.tbl_id, snapshot_record.dir_id, snapshot_md, schema_version_md)
        self.snapshot_tbl_id = snapshot_record.id
        # it's safe to call _load_valid_rowids() here because the storage table already exists
        self._load_valid_rowids()

    def __repr__(self) -> str:
        return f'TableSnapshot(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table snapshot'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.TableSnapshot.__table__} "
                f"SET {schema.TableSnapshot.dir_id.name} = :new_dir_id, "
                f"    {schema.TableSnapshot.md.name}['name'] = :new_name "
                f"WHERE {schema.TableSnapshot.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})


class MutableTable(Table):
    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int = 0
        # TODO: disambiguate what this means: # of slots computed or # of columns computed?
        num_computed_values: int = 0
        num_excs: int = 0
        updated_cols: List[str] = dataclasses.field(default_factory=list)
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    """A :py:class:`Table` that can be modified.
    """
    def __init__(self, tbl_record: schema.Table, schema_version_record: schema.TableSchemaVersion):
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        super().__init__(tbl_record.id, tbl_record.dir_id, tbl_md, schema_version_md)

    def __repr__(self) -> str:
        return f'MutableTable(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

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
        assert self.tbl_md.next_col_id is not None
        col.tbl = self
        col.id = self.tbl_md.next_col_id
        self.tbl_md.next_col_id += 1

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

        # we're creating a new schema version
        self.tbl_md.current_version += 1
        preceding_schema_version = self.tbl_md.current_schema_version
        self.tbl_md.current_schema_version = self.tbl_md.current_version

        self.cols.append(col)
        self.cols_by_name[col.name] = col
        self.cols_by_id[col.id] = col
        self.tbl_md.column_history[col.id] = schema.ColumnHistory(col.id, self.tbl_md.current_schema_version, None)

        with Env.get().engine.begin() as conn:
            self._update_md(preceding_schema_version, conn)
            _logger.info(f'Added column {col.name} to table {self.name}, new version: {self.tbl_md.current_version}')

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
            return self.UpdateStatus()
        if (not col.is_computed or not col.is_stored) and not col.is_indexed:
            return self.UpdateStatus(num_rows=row_count)
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
                            if result_row.has_exc(value_expr_slot_idx):
                                num_excs += 1
                                value_exc = result_row.get_exc(value_expr_slot_idx)
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
                                val = result_row.get_stored_val(value_expr_slot_idx)
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
                return self.UpdateStatus(
                    num_rows=row_count, num_computed_values=row_count, num_excs=num_excs,
                    cols_with_excs=[col.name] if num_excs > 0 else [])
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
        if col.id == self.tbl_md.parameters.frame_col_id or col.id == self.tbl_md.parameters.frame_idx_col_id:
            src_col_name = self.cols_by_id[self.tbl_md.parameters.frame_src_col_id].name
            raise exc.Error(
                f'Cannot drop column {name} because it is used for frame extraction on column {src_col_name}')
        if col.id == self.tbl_md.parameters.frame_src_col_id:
            # we also need to reset the frame extraction table parameters
            self.tbl_md.parameters.reset()

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

        # we're creating a new schema version
        self.tbl_md.current_version += 1
        preceding_schema_version = self.tbl_md.current_schema_version
        self.tbl_md.current_schema_version = self.version

        self.cols.remove(col)
        del self.cols_by_name[name]
        del self.cols_by_id[col.id]
        self.tbl_md.column_history[col.id].schema_version_drop = self.tbl_md.current_schema_version

        with Env.get().engine.begin() as conn:
            self._update_md(preceding_schema_version, conn)
        if col.is_stored:
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
        self.tbl_md.current_version += 1
        preceding_schema_version = self.tbl_md.current_schema_version
        self.tbl_md.current_schema_version = self.tbl_md.current_version

        with Env.get().engine.begin() as conn:
            self._update_md(preceding_schema_version, conn)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def _update_md(self, preceding_schema_version: int, conn: sql.engine.Connection) -> None:
        """Update Table.md and create a new TableSchemaVersion entry."""
        conn.execute(
            sql.update(schema.Table.__table__)
            .values({schema.Table.md: dataclasses.asdict(self.tbl_md)})
            .where(schema.Table.id == self.id))
        schema_version_md = self._create_schema_version_md(preceding_schema_version)
        conn.execute(
            sql.insert(schema.TableSchemaVersion.__table__)
            .values(
                tbl_id=self.id, schema_version=self.tbl_md.current_schema_version,
                md=dataclasses.asdict(schema_version_md)))

    def insert(self, rows: List[List[Any]], columns: List[str] = [], print_stats: bool = False) -> UpdateStatus:
        """Insert rows into table.

        Args:
            rows: A list of rows to insert. Each row is a list of values, one for each column.
            columns: A list of column names that specify the columns present in ``rows``.
                If ``columns`` is empty, all non-computed columns are present in ``rows``.
            print_stats: If ``True``, print statistics about the cost of computed columns.

        Returns:
            execution status

        Raises:
            Error: If the number of columns in ``rows`` does not match the number of columns in the table or in
            ``columns``.

        Examples:
            Insert two rows into a table with three int columns ``a``, ``b``, and ``c``. Note that the ``columns``
            argument is required here because ``rows`` only contain two columns.

            >>> tbl.insert([[1, 1], [2, 2]], columns=['a', 'b'])

            Assuming a table with columns ``video``, ``frame`` and ``frame_idx`` and set up for automatic frame extraction,
            insert a single row containing a video file path (the video contains 100 frames). The row will be expanded
            into 100 rows, one for each frame, and the ``frame`` and ``frame_idx`` columns will be populated accordingly.
            Note that the ``columns`` argument is unnecessary here because only the ``video`` column is required.

            >>> tbl.insert([['/path/to/video.mp4']])

        """
        if not isinstance(rows, list):
            raise exc.Error('rows must be a list of lists')
        if len(rows) == 0:
            raise exc.Error('rows must not be empty')
        for row in rows:
            if not isinstance(row, list):
                raise exc.Error('rows must be a list of lists')
        if not isinstance(columns, list):
            raise exc.Error('columns must be a list of column names')
        for col_name in columns:
            if not isinstance(col_name, str):
                raise exc.Error('columns must be a list of column names')

        insertable_col_names = self._get_insertable_col_names()
        if len(columns) == 0 and len(rows[0]) != len(insertable_col_names):
            raise exc.Error(
                f'Table {self.name} has {len(insertable_col_names)} user-supplied columns, but the data only contains '
                f"{len(rows[0])} columns. In this case, you need to specify the column names with the 'columns' "
                f'parameter.')

        # make sure that each row contains the same number of values
        num_col_vals = len(rows[0])
        for i in range(1, len(rows)):
            if len(rows[i]) != num_col_vals:
                raise exc.Error(
                    f'Inconsistent number of column values in rows: row 0 has {len(rows[0])}, '
                    f'row {i} has {len(rows[i])}')

        if len(columns) == 0:
            columns = insertable_col_names
        if len(rows[0]) != len(columns):
            raise exc.Error(
                f'The number of column values in rows ({len(rows[0])}) does not match the given number of column names '
                f'({", ".join(columns)})')

        self._check_input_rows(rows, columns)
        return self._insert_rows(rows, columns, print_stats=print_stats)

    def _get_insertable_col_names(self, required_only: bool = False) -> List[str]:
        """Return the names of all columns for which values can be specified."""
        names = [c.name for c in self.cols if not c.is_computed and (not required_only or not c.col_type.nullable)]
        if self.extracts_frames():
            names.remove(self.cols_by_id[self.tbl_md.parameters.frame_col_id].name)
            names.remove(self.cols_by_id[self.tbl_md.parameters.frame_idx_col_id].name)
        return names

    def _check_input_rows(self, rows: List[List[Any]], column_names: List[str]) -> None:
        """
        Make sure 'rows' conform to schema.
        """
        assert len(rows) > 0
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = set(self._get_insertable_col_names(required_only=True))
        given_col_names = set(column_names)
        if not(reqd_col_names <= given_col_names):
            raise exc.Error(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')
        if not(given_col_names <= all_col_names):
            raise exc.Error(f'Unknown columns: {", ".join(given_col_names - all_col_names)}')
        computed_col_names = {col.name for col in self.cols if col.value_expr is not None}
        if self.extracts_frames():
            computed_col_names.add(self.cols_by_id[self.tbl_md.parameters.frame_col_id].name)
            computed_col_names.add(self.cols_by_id[self.tbl_md.parameters.frame_idx_col_id].name)
        if len(computed_col_names & given_col_names) > 0:
            raise exc.Error(
                f'Provided values for computed columns: {", ".join(computed_col_names & given_col_names)}')

        # check data
        row_cols = [self.cols_by_name[name] for name in column_names]
        for col_idx, col in enumerate(row_cols):
            for row_idx, row in enumerate(rows):
                if not col.col_type.nullable and row[col_idx] is None:
                    raise exc.Error(
                        f'Column {col.name}: row {row_idx} contains None for a non-nullable column')
                val = row[col_idx]
                if val is None:
                    continue
                try:
                    col.col_type.validate_literal(val)
                except TypeError as e:
                    raise exc.Error(f'Column {col.name} in row {row_idx}: {e}')

    class TableRowBuilder:
        def __init__(self, output_spec: List[ColumnInfo]):
            """
            Args:
                output_spec: list of (Column, slot_idx)
            """
            self.output_spec = output_spec

        def create_row(
                self, input_row: 'exprs.DataRow', rowid: int, version: int, exc_col_ids: Set[int]
        ) -> Tuple[Dict[str, Any], int]:
            """Return (a dict that represents a stored row (can be passed to sql.insert()), # of exceptions)"""
            num_excs = 0
            table_row: Dict[str, Any] = {}
            for info in self.output_spec:
                if input_row.has_exc(info.slot_idx):
                    # exceptions get stored in the errortype/-msg columns
                    exc = input_row.get_exc(info.slot_idx)
                    num_excs += 1
                    exc_col_ids.add(info.col.id)
                    table_row[info.col.storage_name()] = None
                    table_row[info.col.errortype_storage_name()] = type(exc).__name__
                    table_row[info.col.errormsg_storage_name()] = str(exc)
                else:
                    val = input_row.get_stored_val(info.slot_idx)
                    table_row[info.col.storage_name()] = val
                    # we unfortunately need to set these, even if there are no errors
                    table_row[info.col.errortype_storage_name()] = None
                    table_row[info.col.errormsg_storage_name()] = None
            table_row.update({'rowid': rowid, 'v_min': version})
            return table_row, num_excs

    def _insert_rows(self, rows: List[List[Any]], column_names: List[str], print_stats: bool = False) -> UpdateStatus:
        """Insert rows into this table.

        If self.parameters.frame_src_col_id != None:

        - each row (containing a video) is expanded into one row per extracted frame (at the rate of the fps parameter)
        - parameters.frame_col_id is the image column that receives the extracted frame
        - parameters.frame_idx_col_id is the integer column that receives the frame index (starting at 0)
        """
        self._check_is_dropped()

        # we're creating a new version
        self.tbl_md.current_version += 1
        from pixeltable.plan import Planner
        plan, db_col_info, idx_col_info, num_values_per_row = Planner.create_insert_plan(self, rows, column_names)
        plan.open()
        rows = next(plan)
        plan.close()

        # insert rows into table in batches
        start_row_id = self.tbl_md.next_row_id
        batch_size = 16
        progress_bar = tqdm(total=len(rows), desc='Inserting rows into table', unit='rows')
        row_builder = self.TableRowBuilder(db_col_info)
        with Env.get().engine.begin() as conn:
            num_excs = 0
            cols_with_excs: Set[int] = set()
            for batch_start_idx in range(0, len(rows), batch_size):
                # compute batch of rows and convert them into table rows
                table_rows: List[Dict[str, Any]] = []
                for row_idx in range(batch_start_idx, min(batch_start_idx + batch_size, len(rows))):
                    row = rows[row_idx]
                    table_row, num_row_exc = row_builder.create_row(
                        row, self.tbl_md.next_row_id, self.tbl_md.current_version, cols_with_excs)
                    num_excs += num_row_exc
                    self.tbl_md.next_row_id += 1
                    table_rows.append(table_row)
                    progress_bar.update(1)
                conn.execute(sql.insert(self.sa_tbl), table_rows)

            progress_bar.close()
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({schema.Table.md: dataclasses.asdict(self.tbl_md)})
                    .where(schema.Table.id == self.id))

        if len(idx_col_info) > 0:
            # update image indices
            for info in tqdm(idx_col_info, desc='Updating image indices', unit='column'):
                embeddings = [row[info.slot_idx] for row in rows]
                info.col.idx.insert(np.asarray(embeddings), np.arange(start_row_id, self.tbl_md.next_row_id))

        if print_stats:
            plan.ctx.profile.print(num_rows=len(rows))
        self.valid_rowids.update(range(start_row_id, self.tbl_md.next_row_id))
        if num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = f'across {len(cols_with_excs)} column{"" if len(cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join([self.cols_by_id[id].name for id in cols_with_excs])})'
        msg = f'inserted {len(rows)} rows with {num_excs} error{"" if num_excs == 1 else "s"} {cols_with_excs_str}'
        print(msg)
        _logger.info(f'Table {self.name}: {msg}, new version {self.version}')
        status = self.UpdateStatus(
            num_rows=len(rows), num_computed_values=num_values_per_row * len(rows), num_excs=num_excs,
            cols_with_excs=[self.cols_by_id[cid].name for cid in cols_with_excs])
        return status

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
        """
        from  pixeltable import exprs
        update_targets: List[Tuple[Column, exprs.Expr]] = []
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise exc.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name not in self.cols_by_name:
                raise exc.Error(f'Column {col_name} unknown')
            col = self.cols_by_name[col_name]
            if col.is_computed:
                raise exc.Error(f'Column {col_name} is computed and cannot be updated')
            if col.primary_key:
                raise exc.Error(f'Column {col_name} is a primary key column and cannot be updated')
            if col.col_type.is_image_type():
                raise exc.Error(f'Column {col_name} has type image and cannot be updated')
            if col.col_type.is_video_type():
                raise exc.Error(f'Column {col_name} has type video and cannot be updated')

            # make sure that the value is compatible with the column type
            # check if this is a literal
            try:
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError:
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise exc.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.matches(value_expr.col_type):
                    raise exc.Error((
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ))
            update_targets.append((col, value_expr))

        from pixeltable.exprs import Predicate
        from pixeltable.plan import Planner
        analysis_info: Optional[Planner.AnalysisInfo] = None
        if where is not None:
            if not isinstance(where, Predicate):
                raise exc.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.get_info(self, where)
            if analysis_info.similarity_clause is not None:
                raise exc.Error('nearest() cannot be used with update()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise exc.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        # retrieve all stored cols and all target exprs
        updated_cols = [col for col, _ in update_targets]
        recomputed_cols = self._get_dependent_cols(updated_cols) if cascade else []
        copied_cols = \
            [col for col in self.cols if col.is_stored and not col in updated_cols and not col in recomputed_cols]
        select_list = [exprs.ColumnRef(col) for col in copied_cols]
        select_list.extend([expr for _, expr in update_targets])

        recomputed_exprs = [c.value_expr.copy().resolve_computed_cols(unstored_only=False) for c in recomputed_cols]
        # recomputed cols reference the new values of the updated cols
        for col, e in update_targets:
            exprs.Expr.list_substitute(recomputed_exprs, exprs.ColumnRef(col), e)
        select_list.extend(recomputed_exprs)

        plan, select_list = Planner.create_query_plan(
            self, select_list, where_clause=where, with_pk=True, ignore_errors=True)

        # we're creating a new version
        self.tbl_md.current_version += 1
        table_row_info = [
            ColumnInfo(col, select_list[i].slot_idx)
            for i, col in enumerate(copied_cols + updated_cols + recomputed_cols)  # same order as select_list
        ]
        row_builder = self.TableRowBuilder(table_row_info)
        plan.open()
        num_excs = 0
        num_rows = 0
        cols_with_excs: Set[int] = set()
        try:
            # insert new versions of updated rows
            with Env.get().engine.begin() as conn:
                for row_batch in plan:
                    num_rows += len(row_batch)
                    table_rows: List[Dict[str, Any]] = []
                    for result_row in row_batch:
                        table_row, num_row_exc = row_builder.create_row(
                            result_row, result_row.row_id, self.version, cols_with_excs)
                        num_excs += num_row_exc
                        table_rows.append(table_row)
                    conn.execute(sql.insert(self.sa_tbl), table_rows)

                # mark old versions (v_min < self.version) of updated rows as deleted
                stmt = sql.update(self.sa_tbl) \
                    .values({self.v_max_col: self.version}) \
                    .where(self.v_min_col < self.version)  \
                    .where(self.v_max_col == schema.Table.MAX_VERSION)
                if where is not None:
                    assert analysis_info is not None
                    stmt = stmt.where(analysis_info.sql_where_clause)
                conn.execute(stmt)

                # update table version
                conn.execute(
                    sql.update(schema.Table.__table__)
                    .values({schema.Table.md: dataclasses.asdict(self.tbl_md)})
                    .where(schema.Table.id == self.id))

        finally:
            plan.close()

        status = self.UpdateStatus(
            num_rows=num_rows, num_excs=num_excs, updated_cols=[c.name for c in updated_cols],
            cols_with_excs=cols_with_excs)
        return status

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        from pixeltable.exprs import Predicate
        from pixeltable.plan import Planner
        analysis_info: Optional[Planner.AnalysisInfo] = None
        if where is not None:
            if not isinstance(where, Predicate):
                raise exc.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.get_info(self, where)
            if analysis_info.similarity_clause is not None:
                raise exc.Error('nearest() cannot be used with delete()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise exc.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        # we're creating a new version
        self.tbl_md.current_version += 1
        # insert new versions of updated rows
        with Env.get().engine.begin() as conn:
            # mark rows as deleted
            stmt = sql.update(self.sa_tbl) \
                .values({self.v_max_col: self.version}) \
                .where(self.v_min_col < self.version) \
                .where(self.v_max_col == schema.Table.MAX_VERSION) \
                .returning(1)
            if where is not None:
                assert analysis_info is not None
                stmt = stmt.where(analysis_info.sql_where_clause)
            num_rows = conn.execute(stmt).rowcount

            # update table version
            conn.execute(
                sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(self.tbl_md)})
                .where(schema.Table.id == self.id))

        status = self.UpdateStatus(num_rows=num_rows)
        return status

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        if self.version == 0:
            raise exc.Error('Cannot revert version 0')

        with orm.Session(Env.get().engine, future=True) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .where(sql.text((f"({schema.TableSnapshot.__table__}.md->>'current_version')::int = {self.version}"))) \
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
                added_col_ids = [
                    col_history.col_id for col_history in self.tbl_md.column_history.values()
                    if col_history.schema_version_add == self.schema_version
                ]
                assert len(added_col_ids) <= 1
                if len(added_col_ids) == 1:
                    added_col_id = added_col_ids[0]
                    # drop this newly-added column and its ColumnHistory record
                    c = self.cols_by_id[added_col_id]
                    if c.is_stored:
                        stmt = f'ALTER TABLE {self.storage_name()} DROP COLUMN {c.storage_name()}'
                        conn.execute(sql.text(stmt))
                    del self.tbl_md.column_history[c.id]

                # we need to determine the preceding schema version and reload the schema
                schema_version_md_dict = session.query(schema.TableSchemaVersion.md) \
                    .where(schema.TableSchemaVersion.tbl_id == self.id) \
                    .where(schema.TableSchemaVersion.schema_version == self.schema_version) \
                    .scalar()
                preceding_schema_version = schema_version_md_dict['preceding_schema_version']
                preceding_schema_version_md_dict = session.query(schema.TableSchemaVersion.md) \
                    .where(schema.TableSchemaVersion.tbl_id == self.id) \
                    .where(schema.TableSchemaVersion.schema_version == preceding_schema_version) \
                    .scalar()
                preceding_schema_version_md = schema.md_from_dict(
                    schema.TableSchemaVersionMd, preceding_schema_version_md_dict)
                self._set_cols(preceding_schema_version_md)

                conn.execute(
                    sql.delete(schema.TableSchemaVersion.__table__)
                        .where(schema.TableSchemaVersion.tbl_id == self.id)
                        .where(schema.TableSchemaVersion.schema_version == self.schema_version))
                self.tbl_md.current_schema_version = preceding_schema_version

            self.tbl_md.current_version -= 1
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({schema.Table.md: dataclasses.asdict(self.tbl_md)})
                    .where(schema.Table.id == self.id))

            session.commit()
            _logger.info(f'Table {self.name}: reverted to version {self.version}')

    # MODULE-LOCAL, NOT PUBLIC
    def drop(self) -> None:
        self._check_is_dropped()
        self.is_dropped = True

        with orm.Session(Env.get().engine, future=True) as session:
            # check if we have snapshots
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .scalar()
            if num_references > 0:
                raise exc.Error((
                    f'Cannot drop table {self.name}, which has {num_references} snapshot'
                    f'{"s" if num_references > 1 else ""}'
                ))

            # we can delete this table altogether
            ImageStore.delete(self.id)
            conn = session.connection()
            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == self.id))
            conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == self.id))
            self.sa_md.drop_all(bind=conn)
            session.commit()
            return


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

    def _create_schema_version_md(self, preceding_schema_version: int) -> schema.TableSchemaVersionMd:
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(self.cols):
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)
        # preceding_schema_version to be set by the caller
        return schema.TableSchemaVersionMd(
            schema_version=self.tbl_md.current_schema_version, preceding_schema_version=preceding_schema_version,
            columns=column_md)

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
        cls, dir_id: UUID, name: str, cols: List[Column],
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

        params = schema.TableParameters(
            num_retained_versions,
            cols_by_name[extract_frames_from].id if extract_frames_from is not None else -1,
            cols_by_name[extracted_frame_col].id if extracted_frame_col is not None else -1,
            cols_by_name[extracted_frame_idx_col].id if extracted_frame_idx_col is not None else -1,
            extracted_fps)

        with orm.Session(Env.get().engine, future=True) as session:
            # create schema.Table
            column_history = {
                col.id: schema.ColumnHistory(col_id=col.id, schema_version_add=0, schema_version_drop=None)
                for col in cols
            }
            table_md = schema.TableMd(
                name=name, parameters=dataclasses.asdict(params), current_version=0, current_schema_version=0,
                next_col_id=len(cols), next_row_id=0, column_history=column_history)
            tbl_record = schema.Table(dir_id=dir_id, md=dataclasses.asdict(table_md))
            session.add(tbl_record)
            session.flush()  # sets tbl_record.id

            # create schema.TableSchemaVersion
            column_md: Dict[int, schema.SchemaColumn] = {}
            for pos, col in enumerate(cols):
                # Column.dependent_cols for existing cols is wrong at this point, but Table.init() will set it correctly
                value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
                column_md[col.id] = schema.SchemaColumn(
                    pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                    is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)

                # for image cols, add VectorIndex for kNN search
                if col.is_indexed and col.col_type.is_image_type():
                    col.set_idx(VectorIndex.create(Table._vector_idx_name(tbl_record.id, col), 512))

            schema_version_md = schema.TableSchemaVersionMd(
                schema_version=0, preceding_schema_version=None, columns=column_md)
            schema_version_record = schema.TableSchemaVersion(
                tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md))
            session.add(schema_version_record)
            session.flush()  # avoid FK violations in Postgres

            assert tbl_record.id is not None
            tbl = MutableTable(tbl_record, schema_version_record)
            tbl.sa_md.create_all(bind=session.connection())
            session.commit()
            _logger.info(f'created table {name}, id={tbl_record.id}')
            return tbl


class Path:
    def __init__(self, path: str, empty_is_valid: bool = False):
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
    def parent(self) -> Path:
        if len(self.components) == 1:
            if self.is_root:
                return self
            else:
                return Path('', empty_is_valid=True)
        else:
            return Path('.'.join(self.components[:-1]))

    def append(self, name: str) -> Path:
        if self.is_root:
            return Path(name)
        else:
            return Path(f'{str(self)}.{name}')

    def is_ancestor(self, other: Path, is_parent: bool = False) -> bool:
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
    """Keep track of all paths in a Db instance"""
    def __init__(self):
        self.dir_contents: Dict[UUID, Dict[str, SchemaObject]] = {}
        self.schema_objs: Dict[UUID, SchemaObject] = {}

        # load dirs
        with orm.Session(Env.get().engine, future=True) as session:
            _ = [dir_record for dir_record in session.query(schema.Dir).all()]
            self.schema_objs = {
                dir_record.id: Dir(dir_record.id, dir_record.parent_id, schema.DirMd(**dir_record.md).name)
                for dir_record in session.query(schema.Dir).all()
            }

        # identify root dir
        root_dirs = [dir for dir in self.schema_objs.values() if dir.dir_id is None]
        assert len(root_dirs) == 1
        self.root_dir = root_dirs[0]

        # build dir_contents
        def record_dir(dir: Dir) -> None:
            if dir.id in self.dir_contents:
                return
            else:
                self.dir_contents[dir.id] = {}
            if dir.dir_id is not None:
                record_dir(self.schema_objs[dir.dir_id])
                self.dir_contents[dir.dir_id][dir.name] = dir

        for dir in self.schema_objs.values():
            record_dir(dir)

        # load tables
        with orm.Session(Env.get().engine, future=True) as session:
            # load all reachable (= mutable) tables
            q = session.query(schema.Table, schema.TableSchemaVersion) \
                .select_from(schema.Table) \
                .join(schema.TableSchemaVersion) \
                .where(sql.text((
                    f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                    f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}")))
            for tbl_record, schema_version_record in q.all():
                tbl = MutableTable(tbl_record, schema_version_record)
                tbl._load_valid_rowids()  # TODO: move this someplace more appropriate
                self.schema_objs[tbl.id] = tbl
                assert tbl_record.dir_id is not None
                dir = self.schema_objs[tbl_record.dir_id]
                self.dir_contents[dir.id][tbl.name] = tbl

        # load table snapshots
        with orm.Session(Env.get().engine, future=True) as session:
            q = session.query(schema.TableSnapshot, schema.TableSchemaVersion) \
                .select_from(schema.TableSnapshot) \
                .join(schema.TableSchemaVersion, schema.TableSnapshot.tbl_id == schema.TableSchemaVersion.tbl_id) \
                .where(sql.text((
                    f"({schema.TableSnapshot.__table__}.md->>'current_schema_version')::int = "
                    f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}")))
            for snapshot_record, schema_version_record in q.all():
                snapshot = TableSnapshot(snapshot_record, schema_version_record)
                self.schema_objs[snapshot.id] = snapshot
                assert snapshot_record.dir_id is not None
                dir = self.schema_objs[snapshot_record.dir_id]
                self.dir_contents[dir.id][snapshot.name] = snapshot

        # load Function metadata; doesn't load the actual callable, which can be large and is only done on-demand by the
        # FunctionRegistry
        with orm.Session(Env.get().engine, future=True) as session:
            q = session.query(schema.Function.id, schema.Function.dir_id, schema.Function.md) \
                .where(sql.text(f"({schema.Function.__table__}.md->>'name')::text IS NOT NULL"))
            for id, dir_id, md in q.all():
                assert 'name' in md
                name = md['name']
                assert name is not None
                named_fn = NamedFunction(id, dir_id, name)
                self.schema_objs[id] = named_fn
                assert dir_id is not None
                dir = self.schema_objs[dir_id]
                self.dir_contents[dir.id][name] = named_fn

    def _resolve_path(self, path: Path) -> SchemaObject:
        if path.is_root:
            return self.root_dir
        dir = self.root_dir
        for i, component in enumerate(path.components):
            if component not in self.dir_contents[dir.id]:
                raise exc.Error(f'No such path: {".".join(path.components[:i+1])}')
            schema_obj = self.dir_contents[dir.id][component]
            if i < len(path.components) - 1:
                if not isinstance(schema_obj, Dir):
                    raise exc.Error(f'Not a directory: {".".join(path.components[:i+1])}')
                dir = schema_obj
        return schema_obj

    def __getitem__(self, path: Path) -> SchemaObject:
        return self._resolve_path(path)

    def get_schema_obj(self, id: UUID) -> Optional[SchemaObject]:
        return self.schema_objs.get(id)

    def __setitem__(self, path: Path, val: SchemaObject) -> None:
        parent_dir = self._resolve_path(path.parent)
        assert path.name not in self.dir_contents[parent_dir.id]
        self.schema_objs[val.id] = val
        self.dir_contents[parent_dir.id][path.name] = val
        if isinstance(val, Dir):
            self.dir_contents[val.id] = {}

    def __delitem__(self, path: Path) -> None:
        parent_dir = self._resolve_path(path.parent)
        assert path.name in self.dir_contents[parent_dir.id]
        obj = self.dir_contents[parent_dir.id][path.name]
        del self.dir_contents[parent_dir.id][path.name]
        if isinstance(obj, Dir):
            del self.dir_contents[obj.id]
        del self.schema_objs[obj.id]

    def move(self, from_path: Path, to_path: Path) -> None:
        from_dir = self._resolve_path(from_path.parent)
        assert isinstance(from_dir, Dir)
        assert from_path.name in self.dir_contents[from_dir.id]
        obj = self.dir_contents[from_dir.id][from_path.name]
        del self.dir_contents[from_dir.id][from_path.name]
        to_dir = self._resolve_path(to_path.parent)
        assert to_path.name not in self.dir_contents[to_dir.id]
        self.dir_contents[to_dir.id][to_path.name] = obj

    def check_is_valid(self, path: Path, expected: Optional[Type[SchemaObject]]) -> None:
        """Check that path is valid and that the object at path has the expected type.

        Args:
            path: path to check
            expected: expected type of object at path or None if object should not exist

        Raises:
            Error if path is invalid or object at path has wrong type
        """
        # check for existence
        if expected is not None:
            schema_obj = self._resolve_path(path)
            if not isinstance(schema_obj, expected):
                raise exc.Error(
                    f'{str(path)} needs to be a {expected.display_name()} but is a {type(schema_obj).display_name()}')
        if expected is None:
            parent_obj = self._resolve_path(path.parent)
            if not isinstance(parent_obj, Dir):
                raise exc.Error(
                    f'{str(path.parent)} is a {type(parent_obj).display_name()}, not a {Dir.display_name()}')
            if path.name in self.dir_contents[parent_obj.id]:
                obj = self.dir_contents[parent_obj.id][path.name]
                raise exc.Error(f"{type(obj).display_name()} '{str(path)}' already exists")

    def get_children(self, parent: Path, child_type: Optional[Type[SchemaObject]], recursive: bool) -> List[Path]:
        dir = self._resolve_path(parent)
        if not isinstance(dir, Dir):
            raise exc.Error(f'{str(parent)} is a {type(dir).display_name()}, not a directory')
        matches = [
            obj for obj in self.dir_contents[dir.id].values() if child_type is None or isinstance(obj, child_type)
        ]
        result = [copy.copy(parent).append(obj.name) for obj in matches]
        if recursive:
            for dir in [obj for obj in self.dir_contents[dir.id].values() if isinstance(obj, Dir)]:
                result.extend(self.get_children(copy.copy(parent).append(dir.name), child_type, recursive))
        return result

def init_catalog() -> None:
    """One-time initialization of the catalog. Idempotent."""
    with orm.Session(Env.get().engine, future=True) as session:
        if session.query(sql.func.count(schema.Dir.id)).scalar() > 0:
            return
        # create a top-level directory, so that every schema object has a directory
        dir_md = schema.DirMd(name='')
        dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        session.commit()
        _logger.info(f'Initialized catalog')
