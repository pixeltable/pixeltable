from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import re
from typing import Optional, List, Set, Dict, Any, Type, Union, Callable, Tuple
from uuid import UUID
from abc import abstractmethod
import datetime
import time

import pandas as pd
import sqlalchemy as sql
import sqlalchemy.orm as orm
from tqdm.autonotebook import tqdm
from pgvector.sqlalchemy import Vector

from pixeltable import exceptions as exc
from pixeltable.env import Env
from pixeltable.exec import ColumnInfo
from pixeltable.function import Function
from pixeltable.metadata import schema
from pixeltable.type_system import ColumnType, StringType
from pixeltable.utils.imgstore import ImageStore

_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')


class Column:
    """Representation of a column in the schema of a Table/DataFrame.

    A Column contains all the metadata necessary for executing queries and updates against a particular version of a
    table/view.
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
        # indexed columns also have a column for the embeddings
        self.sa_idx_col: Optional[sql.schema.Column] = None
        self.tbl: Optional[TableVersion] = None  # set by owning TableVersion

        if indexed and not self.col_type.is_image_type():
            raise exc.Error(f'Column {name}: indexed=True requires ImageType')
        self.is_indexed = indexed

    @classmethod
    def from_md(cls, col_id: int, md: schema.SchemaColumn, tbl: TableVersion) -> Column:
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
        if self.is_indexed:
            self.sa_idx_col = sql.Column(self.index_storage_name(), Vector(512), nullable=True)

    def storage_name(self) -> str:
        assert self.id is not None
        assert self.is_stored
        return f'col_{self.id}'

    def errormsg_storage_name(self) -> str:
        return f'{self.storage_name()}_errormsg'

    def errortype_storage_name(self) -> str:
        return f'{self.storage_name()}_errortype'

    def index_storage_name(self) -> str:
        return f'{self.storage_name()}_idx_0'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return False
        assert self.tbl is not None
        assert other.tbl is not None
        return self.tbl.id == other.tbl.id and self.id == other.id


class TableVersion:
    """
    TableVersion contains all metadata needed to execute queries and updates against a particular version of a
    table/view (ie, what is recorded in schema.Table):
    - schema information
    - for views, the full chain of base tables

    If this version is not the current version, updates are disabled.
    """

    def __init__(
            self, id: UUID, base: Optional[TableVersion], tbl_md: schema.TableMd, version: int,
            schema_version_md: schema.TableSchemaVersionMd
    ):
        self.id = id
        self.name = tbl_md.name
        self.base = base
        self.version = version
        self.schema_version = schema_version_md.schema_version
        if tbl_md.current_version == self.version:
            self.next_col_id = tbl_md.next_col_id
            self.next_rowid = tbl_md.next_row_id
        else:
            self.next_col_id = -1
            self.next_rowid = -1
        self.column_history = tbl_md.column_history
        self.parameters = tbl_md.parameters
        self._set_cols(schema_version_md)
        from pixeltable import exprs
        self.predicate = exprs.Expr.from_dict(tbl_md.predicate, self) if tbl_md.predicate is not None else None
        self.views: List[TableVersion] = []  # views that reference us
        if self.base is not None:
            self.base.views.append(self)

        from pixeltable.store import StoreTable, StoreView
        if self.is_view():
            self.store_tbl = StoreView(self)
        else:
            self.store_tbl = StoreTable(self)

    def create_snapshot_copy(self) -> TableVersion:
        """Create an immutable copy of this TableVersion for a particular snapshot"""
        result = TableVersion(
            self.id, self.base, self._create_md(), self.version,
            self._create_schema_version_md(preceding_schema_version=0))  # preceding_schema_version: dummy value
        result.next_col_id = -1
        result.next_rowid = -1
        return result

    @classmethod
    def create(
            cls, dir_id: UUID, name: str, cols: List[Column],
            base: Optional[TableVersion], predicate: Optional['exprs.Predicate'],
            num_retained_versions: int,
            extract_frames_from: Optional[str], extracted_frame_col: Optional[str],
            extracted_frame_idx_col: Optional[str], extracted_fps: Optional[int],
            session: orm.Session
    ) -> TableVersion:
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

        ts = time.time()
        # create schema.Table
        column_history = {
            col.id: schema.ColumnHistory(col_id=col.id, schema_version_add=0, schema_version_drop=None)
            for col in cols
        }
        table_md = schema.TableMd(
            name=name, parameters=params, current_version=0, current_schema_version=0,
            next_col_id=len(cols), next_row_id=0, column_history=column_history,
            predicate=predicate.as_dict() if predicate is not None else None)
        # base version: if we're referencing a live table, the base version is None
        base_version = None if base is None or base.next_rowid != -1 else base.version
        tbl_record = schema.Table(
            dir_id=dir_id, base_id=base.id if base is not None else None, base_version=base_version,
            md=dataclasses.asdict(table_md))
        session.add(tbl_record)
        session.flush()  # sets tbl_record.id
        assert tbl_record.id is not None

        # create schema.TableVersion
        table_version_md = schema.TableVersionMd(created_at=ts, version=0, schema_version=0)
        tbl_version_record = schema.TableVersion(
            tbl_id=tbl_record.id, version=0, md=dataclasses.asdict(table_version_md))
        session.add(tbl_version_record)

        # create schema.TableSchemaVersion
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(cols):
            # Column.dependent_cols for existing cols is wrong at this point, but Table.init() will set it correctly
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)

        schema_version_md = schema.TableSchemaVersionMd(
            schema_version=0, preceding_schema_version=None, columns=column_md)
        schema_version_record = schema.TableSchemaVersion(
            tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md))
        session.add(schema_version_record)

        tbl_version = cls(tbl_record.id, base, table_md, 0, schema_version_md)
        tbl_version.store_tbl.create(session.connection())
        # TODO: create pgvector indices
        return tbl_version

    def drop(self) -> None:
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
            # check if we have views
            num_references = session.query(sql.func.count(schema.Table.id)) \
                .where(schema.Table.base_id == self.id) \
                .scalar()
            if num_references > 0:
                raise exc.Error((
                    f'Cannot drop table {self.name}, which has {num_references} views'
                    f'{"s" if num_references > 1 else ""}'
                ))

            # delete this table and all associated data
            ImageStore.delete(self.id)
            conn = session.connection()
            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == self.id))
            conn.execute(
                sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == self.id))
            conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == self.id))
            self.store_tbl.drop(conn)
            session.commit()

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

    def _update_md(
            self, ts: float, preceding_schema_version: Optional[int], conn: sql.engine.Connection) -> None:
        """Update all recorded metadata in response to a data or schema change.
        Args:
            ts: timestamp of the change
            preceding_schema_version: last schema version if schema change, else None
        """
        conn.execute(
            sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(self._create_md())})
                .where(schema.Table.id == self.id))
        version_md = self._create_version_md(ts)
        conn.execute(
            sql.insert(schema.TableVersion.__table__)
                .values(tbl_id=self.id, version=self.version, md=dataclasses.asdict(version_md)))
        if preceding_schema_version is not None:
            schema_version_md = self._create_schema_version_md(preceding_schema_version)
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__)
                .values(
                    tbl_id=self.id, schema_version=self.schema_version,
                    md=dataclasses.asdict(schema_version_md)))

    def add_column(self, col: Column, print_stats: bool = False) -> Table.UpdateStatus:
        """Adds a column to the table.
        """
        assert self.next_col_id != -1
        if re.fullmatch(_ID_RE, col.name) is None:
            raise exc.Error(f"Invalid column name: '{col.name}'")
        if col.name in self.cols_by_name:
            raise exc.Error(f'Column {col.name} already exists')
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

        # we're creating a new schema version
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self.cols.append(col)
        self.cols_by_name[col.name] = col
        self.cols_by_id[col.id] = col
        self.column_history[col.id] = schema.ColumnHistory(col.id, self.schema_version, None)

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
            _logger.info(f'Added column {col.name} to table {self.name}, new version: {self.version}')
            if col.is_stored:
                self.store_tbl.add_column(col, conn)

        row_count = self.count()
        if row_count == 0:
            return Table.UpdateStatus()
        if (not col.is_computed or not col.is_stored) and not col.is_indexed:
            return Table.UpdateStatus(num_rows=row_count)
        # compute values for the existing rows and compute embeddings, if this column is indexed;
        # for some reason, it's not possible to run the following updates in the same transaction as the one
        # that we just used to create the metadata (sqlalchemy hangs when exec() tries to run the query)
        from pixeltable.plan import Planner
        plan, value_expr_slot_idx, embedding_slot_idx = Planner.create_add_column_plan(self, col)
        plan.ctx.num_rows = row_count
        # TODO: create pgvector index, if col is indexed

        plan.open()
        try:
            # TODO: do this in the same transaction as the metadata update
            with Env.get().engine.begin() as conn:
                num_excs = self.store_tbl.load_column(col, plan, value_expr_slot_idx, embedding_slot_idx, conn)
        except sql.exc.DBAPIError as e:
            self.drop_column(col.name)
            raise exc.Error(f'Error during SQL execution:\n{e}')
        finally:
            plan.close()

        msg = f'added {row_count} column values with {num_excs} error{"" if num_excs == 1 else "s"}'
        print(msg)
        _logger.info(f'Column {col.name}: {msg}')
        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)
        return Table.UpdateStatus(
            num_rows=row_count, num_computed_values=row_count, num_excs=num_excs,
            cols_with_excs=[col.name] if num_excs > 0 else [])

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.
        """
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
            self.parameters.reset()

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
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self.cols.remove(col)
        del self.cols_by_name[name]
        del self.cols_by_id[col.id]
        self.column_history[col.id].schema_version_drop = self.schema_version

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
        if col.is_stored:
            self.store_tbl.drop_column()
        _logger.info(f'Dropped column {name} from table {self.name}, new version: {self.version}')

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.
        """
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
        ts = time.time()
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with Env.get().engine.begin() as conn:
            self._update_md(ts, preceding_schema_version, conn)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def insert(self, rows: List[List[Any]], column_names: List[str], print_stats: bool = False) -> Table.UpdateStatus:
        """Insert rows into this table.

        If self.parameters.frame_src_col_id != None:

        - each row (containing a video) is expanded into one row per extracted frame (at the rate of the fps parameter)
        - parameters.frame_col_id is the image column that receives the extracted frame
        - parameters.frame_idx_col_id is the integer column that receives the frame index (starting at 0)
        """
        # we're creating a new version
        ts = time.time()
        self.version += 1
        from pixeltable.plan import Planner
        total_num_rows, total_num_excs = 0, 0
        with Env.get().engine.begin() as conn:
            plan, schema_col_info, idx_col_info, num_values_per_row = \
                Planner.create_insert_plan(self, rows, column_names)
            num_rows, num_excs, cols_with_excs = self.store_tbl.insert_rows(plan, schema_col_info, idx_col_info, conn)
            total_num_rows, total_num_excs = total_num_rows + num_rows, total_num_excs + num_excs
            self._update_md(ts, None, conn)

            # update views
            for view in self.views:
                plan, schema_col_info, idx_col_info, num_values_per_row = Planner.create_view_load_plan(
                    view, base_version=self.version)
                num_rows, num_excs, cols_with_excs = \
                    view.store_tbl.insert_rows(plan, schema_col_info, idx_col_info, conn)
                total_num_rows, total_num_excs = total_num_rows + num_rows, total_num_excs + num_excs

        if print_stats:
            plan.ctx.profile.print(num_rows=len(rows))
        if num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = f'across {len(cols_with_excs)} column{"" if len(cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join([self.cols_by_id[id].name for id in cols_with_excs])})'
        msg = f'inserted {len(rows)} rows with {num_excs} error{"" if num_excs == 1 else "s"} {cols_with_excs_str}'
        print(msg)
        _logger.info(f'Table {self.name}: {msg}, new version {self.version}')
        status = Table.UpdateStatus(
            num_rows=total_num_rows, num_computed_values=num_values_per_row * len(rows), num_excs=total_num_excs,
            cols_with_excs=[self.cols_by_id[cid].name for cid in cols_with_excs])
        return status

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> Table.UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
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
        recomputed_view_cols = [col for col in recomputed_cols if col.tbl != self]
        recomputed_cols = [col for col in recomputed_cols if col.tbl == self]
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
        ts = time.time()
        self.version += 1
        table_row_info = [
            ColumnInfo(col, select_list[i].slot_idx)
            for i, col in enumerate(copied_cols + updated_cols + recomputed_cols)  # same order as select_list
        ]
        total_num_rows, total_num_excs = 0, 0
        with Env.get().engine.begin() as conn:
            plan.open()
            try:
                total_num_rows, total_num_excs, cols_with_excs = self.store_tbl.update_rows(
                    plan, table_row_info, analysis_info.sql_where_clause if analysis_info is not None else None, conn)
            finally:
                plan.close()
            self._update_md(ts, None, conn)

            # update views
            for view in self.views:
                plan, schema_col_info, idx_col_info, num_values_per_row = Planner.create_view_load_plan(
                    view, base_version=self.version)
                num_rows, num_excs, cols_with_excs = \
                    view.store_tbl.insert_rows(plan, schema_col_info, idx_col_info, conn)
                total_num_rows, total_num_excs = total_num_rows + num_rows, total_num_excs + num_excs
                view.store_tbl.mark_deleted(self.version, conn)

        status = Table.UpdateStatus(
            num_rows=total_num_rows, num_excs=total_num_excs, updated_cols=[c.name for c in updated_cols],
            cols_with_excs=cols_with_excs)
        return status

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> Table.UpdateStatus:
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
        ts = time.time()
        self.version += 1
        # insert new versions of updated rows
        with Env.get().engine.begin() as conn:
            # mark rows as deleted
            stmt = sql.update(self.store_tbl.sa_tbl) \
                .values({self.store_tbl.v_max_col: self.version}) \
                .where(self.store_tbl.v_min_col < self.version) \
                .where(self.store_tbl.v_max_col == schema.Table.MAX_VERSION) \
                .returning(1)
            if where is not None:
                assert analysis_info is not None
                stmt = stmt.where(analysis_info.sql_where_clause)
            num_rows = conn.execute(stmt).rowcount
            self._update_md(ts, None, conn)

        status = Table.UpdateStatus(num_rows=num_rows)
        return status

    def revert(self) -> None:
        """Reverts the table to the previous version.
        """
        if self.version == 0:
            raise exc.Error('Cannot revert version 0')

        with orm.Session(Env.get().engine, future=True) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(schema.TableSnapshot.id)) \
                .where(schema.TableSnapshot.tbl_id == self.id) \
                .where(schema.TableSnapshot.tbl_version == self.version) \
                .scalar()
            if num_references > 0:
                raise exc.Error(
                    f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')
            # TODO: check for views referencing this version

            conn = session.connection()
            # delete newly-added data
            ImageStore.delete(self.id, version=self.version)
            conn.execute(sql.delete(self.store_tbl.sa_tbl).where(self.store_tbl.sa_tbl.c.v_min == self.version))
            # revert new deletions
            conn.execute(
                sql.update(self.store_tbl.sa_tbl) \
                    .values({self.store_tbl.sa_tbl.c.v_max: schema.Table.MAX_VERSION})
                    .where(self.store_tbl.sa_tbl.c.v_max == self.version))

            if self.version == self.schema_version:
                # the current version involved a schema change:
                # if the schema change was to add a column, we now need to drop it
                added_col_ids = [
                    col_history.col_id for col_history in self.column_history.values()
                    if col_history.schema_version_add == self.schema_version
                ]
                assert len(added_col_ids) <= 1
                added_col: Optional[Column] = None
                if len(added_col_ids) == 1:
                    added_col_id = added_col_ids[0]
                    # drop this newly-added column and its ColumnHistory record
                    c = self.cols_by_id[added_col_id]
                    if c.is_stored:
                        added_col = c
                    del self.column_history[c.id]

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

                # physically drop the column, but only after we have re-created the schema
                if added_col is not None:
                    self.store_tbl.drop_column(added_col, conn)

                conn.execute(
                    sql.delete(schema.TableSchemaVersion.__table__)
                        .where(schema.TableSchemaVersion.tbl_id == self.id)
                        .where(schema.TableSchemaVersion.schema_version == self.schema_version))
                self.schema_version = preceding_schema_version

            conn.execute(
                sql.delete(schema.TableVersion.__table__)
                    .where(schema.TableVersion.tbl_id == self.id)
                    .where(schema.TableVersion.version == self.version)
            )
            self.version -= 1
            conn.execute(
                sql.update(schema.Table.__table__)
                    .values({schema.Table.md: dataclasses.asdict(self._create_md())})
                    .where(schema.Table.id == self.id))

            session.commit()
            _logger.info(f'Table {self.name}: reverted to version {self.version}')

    def is_view(self) -> bool:
        return self.base is not None

    def is_insertable(self) -> bool:
        """Returns True if this is a live non-view table (a table that allows insertion)"""
        return self.next_rowid != -1 and not self.is_view()

    def is_mutable(self) -> bool:
        """Returns True if this is a live table (a table or view that allows updates)"""
        return self.next_rowid != -1 and not self.is_view()

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

    def get_insertable_col_names(self, required_only: bool = False) -> List[str]:
        """Return the names of all columns for which values can be specified."""
        assert not self.is_view()
        names = [c.name for c in self.cols if not c.is_computed and (not required_only or not c.col_type.nullable)]
        if self.extracts_frames():
            names.remove(self.cols_by_id[self.parameters.frame_col_id].name)
            names.remove(self.cols_by_id[self.parameters.frame_idx_col_id].name)
        return names

    def check_input_rows(self, rows: List[List[Any]], column_names: List[str]) -> None:
        """
        Make sure 'rows' conform to schema.
        """
        assert len(rows) > 0
        all_col_names = {col.name for col in self.cols}
        reqd_col_names = set(self.get_insertable_col_names(required_only=True))
        given_col_names = set(column_names)
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
        col.value_expr = fn(*args)

    def _record_value_expr(self, col: Column) -> None:
        """Update Column.dependent_cols for all cols referenced in col.value_expr.
        """
        assert col.value_expr is not None
        from pixeltable.exprs import ColumnRef
        refd_cols = [e.col for e in col.value_expr.subexprs(expr_class=ColumnRef)]
        for refd_col in refd_cols:
            refd_col.dependent_cols.append(col)

    def _get_dependent_cols(self, cols: List[Column]) -> List[Column]:
        """
        Return the list of cols that transivitely depend on any of the given cols.
        """
        if len(cols) == 0:
            return []
        all_dependent_cols: List[Column] = []
        for col in cols:
            all_dependent_cols.extend(col.dependent_cols)
        # remove duplicates
        unique_cols: Dict[int, Column] = {}  # key: id()
        [unique_cols.setdefault(id(col), col) for col in all_dependent_cols]
        result = list(unique_cols.values())
        return result + self._get_dependent_cols(result)

    def _create_md(self) -> schema.TableMd:
        return schema.TableMd(
            name=self.name, current_version=self.version, current_schema_version=self.schema_version,
            next_col_id=self.next_col_id, next_row_id=self.next_rowid, column_history=self.column_history,
            parameters=self.parameters, predicate=self.predicate.as_dict() if self.predicate is not None else None)

    def _create_version_md(self, ts: float) -> schema.TableVersionMd:
        return schema.TableVersionMd(created_at=ts, version=self.version, schema_version=self.schema_version)

    def _create_schema_version_md(self, preceding_schema_version: int) -> schema.TableSchemaVersionMd:
        column_md: Dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(self.cols):
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.SchemaColumn(
                pos=pos, name=col.name, col_type=col.col_type.as_dict(),
                is_pk=col.primary_key, value_expr=value_expr_dict, stored=col.stored, is_indexed=col.is_indexed)
        # preceding_schema_version to be set by the caller
        return schema.TableSchemaVersionMd(
            schema_version=self.schema_version, preceding_schema_version=preceding_schema_version,
            columns=column_md)

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name.
        """
        if col_name not in self.cols_by_name:
            if self.base is None:
                raise AttributeError(f'Column {col_name} unknown')
            return getattr(self.base, col_name)
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

    def show(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().show(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table.
        """
        return self.df().count()

    def columns(self) -> List[Column]:
        """Return all columns visible in this table, including columns from bases"""
        result = self.cols.copy()
        if self.base is not None:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend([c for c in base_cols if c.name not in self.cols_by_name])
        return result

    @property
    def frame_col(self) -> Optional[Column]:
        if self.parameters.frame_col_id == -1:
            return None
        return self.cols_by_id[self.parameters.frame_col_id]


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


class TableBase(SchemaObject):
    """Base class for all schema objects that can be queried."""
    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version: TableVersion):
        super().__init__(id, name, dir_id)
        self.is_dropped = False
        self.tbl_version = tbl_version

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.Error(f'{self.display_name()} {self.name} has been dropped')

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name.
        """
        return getattr(self.tbl_version, col_name)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        return self.tbl_version.__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version)

    def select(self, *items: 'exprs.Expr') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).select(*items)

    def where(self, pred: 'exprs.Predicate') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).where(pred)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version).order_by(*items, asc=asc)

    def show(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().show(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table.
        """
        return self.df().count()

    def describe(self) -> pd.DataFrame:
        pd_df = pd.DataFrame({
            'Column Name': [c.name for c in self.cols],
            'Type': [str(c.col_type) for c in self.cols],
            'Computed With':
                [c.value_expr.display_str(inline=False) if c.value_expr is not None else '' for c in self.cols],
        })
        # white-space: pre-wrap: print \n as newline
        pd_df = pd_df.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])  # center-align headings
        return pd_df.hide(axis='index')

    def drop(self) -> None:
        self._check_is_dropped()
        self.is_dropped = True
        self.tbl_version.drop()


class TableSnapshot(TableBase):
    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version: TableVersion):
        super().__init__(id, dir_id, name, tbl_version)

    @classmethod
    def create(cls, dir_id: UUID, name: str, tbl_version: TableVersion) -> TableSnapshot:
        with orm.Session(Env.get().engine, future=True) as session:
            snapshot_md = schema.TableSnapshotMd(name=name, created_at=time.time())
            snapshot_record = schema.TableSnapshot(
                dir_id=dir_id, tbl_id=tbl_version.id, tbl_version=tbl_version.version,
                md=dataclasses.asdict(snapshot_md))
            session.add(snapshot_record)
            session.flush()
            assert snapshot_record.id is not None
            snapshot = TableSnapshot(snapshot_record.id, dir_id, name, tbl_version)
            session.commit()
            return snapshot

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


class Table(TableBase):
    """Base class for MutableTable and View"""

    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int = 0
        # TODO: disambiguate what this means: # of slots computed or # of columns computed?
        num_computed_values: int = 0
        num_excs: int = 0
        updated_cols: List[str] = dataclasses.field(default_factory=list)
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    def __init__(self, id: UUID, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(id, dir_id, tbl_version.name, tbl_version)


    def add_column(self, col: Column, print_stats: bool = False) -> Table.UpdateStatus:
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
        return self.tbl_version.add_column(col, print_stats=print_stats)

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
        self.tbl_version.drop_column(name)

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
        self.tbl_version.rename_column(old_name, new_name)

    def insert(self, rows: List[List[Any]], columns: List[str] = [], print_stats: bool = False) -> Table.UpdateStatus:
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

        insertable_col_names = self.tbl_version.get_insertable_col_names()
        if len(columns) == 0 and len(rows[0]) != len(insertable_col_names):
                if len(rows[0]) < len(insertable_col_names):
                    raise exc.Error((
                        f'Table {self.name} has {len(insertable_col_names)} user-supplied columns, but the data only '
                        f'contains {len(rows[0])} columns. In this case, you need to specify the column names with the '
                        f"'columns' parameter."))
                else:
                    raise exc.Error((
                        f'Table {self.name} has {len(insertable_col_names)} user-supplied columns, but the data '
                        f'contains {len(rows[0])} columns. '))

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

        self.tbl_version.check_input_rows(rows, columns)
        return self.tbl_version.insert(rows, columns, print_stats=print_stats)

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> Table.UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
        """
        return self.tbl_version.update(value_spec, where, cascade)

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> Table.UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        return self.tbl_version.delete(where)

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        self.tbl_version.revert()


class MutableTable(Table):
    """A :py:class:`Table` that can be modified.
    """
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
            cls, dir_id: UUID, name: str, cols: List[Column],
            num_retained_versions: int,
            extract_frames_from: Optional[str], extracted_frame_col: Optional[str],
            extracted_frame_idx_col: Optional[str], extracted_fps: Optional[int],
    ) -> MutableTable:
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, None, None, num_retained_versions, extract_frames_from, extracted_frame_col,
                extracted_frame_idx_col, extracted_fps, session)
            tbl = cls(dir_id, tbl_version)
            session.commit()
            _logger.info(f'created table {name}, id={tbl_version.id}')
            return tbl


class View(Table):
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'view'

    @classmethod
    def create(
            cls, dir_id: UUID, name: str, base: TableVersion, cols: List[Column], predicate: 'exprs.Predicate',
            num_retained_versions: int) -> View:
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, base, predicate, num_retained_versions, None, None, None, None, session)
            view = cls(dir_id, tbl_version)

            from pixeltable.plan import Planner
            plan, schema_col_info, idx_col_info, num_values_per_row = Planner.create_view_load_plan(tbl_version)
            num_rows, num_excs, cols_with_excs = \
                tbl_version.store_tbl.insert_rows(plan, schema_col_info, idx_col_info, session.connection())
            session.commit()
            _logger.info(f'created view {name}, id={tbl_version.id}')
            msg = f'created view {name} with {num_rows} rows, {num_excs} exceptions'
            print(msg)
            return view


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

    def add_schema_obj(self, dir_id: UUID, name: str, val: SchemaObject) -> None:
        self.dir_contents[dir_id][name] = val
        self.schema_objs[val.id] = val

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
