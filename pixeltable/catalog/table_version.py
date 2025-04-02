from __future__ import annotations

import dataclasses
import importlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional
from uuid import UUID

import jsonschema.exceptions
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

from ..func.globals import resolve_symbol
from .column import Column
from .globals import _POS_COLUMN_NAME, _ROWID_COLUMN_NAME, MediaValidation, UpdateStatus, is_valid_identifier

if TYPE_CHECKING:
    from pixeltable import exec, store

    from .table_version_handle import TableVersionHandle

_logger = logging.getLogger('pixeltable')


class TableVersion:
    """
    TableVersion represents a particular version of a table/view along with its physical representation:
    - the physical representation is a store table with indices
    - the version can be mutable or a snapshot
    - tables and their recursive views form a tree, and a mutable TableVersion also records its own
      mutable views in order to propagate updates
    - each view TableVersion records its base:
      * the base is correct only for mutable views (snapshot versions form a DAG, not a tree)
      * the base is useful for getting access to the StoreTable and the base id
      * TODO: create a separate hierarchy of objects that records the version-independent tree of tables/views, and
        have TableVersions reference those
    - mutable TableVersions record their TableVersionPath, which is needed for expr evaluation in updates

    Instances of TableVersion should not be stored as member variables (ie, used across transaction boundaries).
    Use a TableVersionHandle instead.
    """

    id: UUID
    name: str
    effective_version: Optional[int]
    version: int
    comment: str
    media_validation: MediaValidation
    num_retained_versions: int
    schema_version: int
    view_md: Optional[schema.ViewMd]
    path: Optional[pxt.catalog.TableVersionPath]  # only set for live tables; needed to resolve computed cols
    base: Optional[TableVersionHandle]  # only set for views
    next_col_id: int
    next_idx_id: int
    next_rowid: int
    predicate: Optional[exprs.Expr]
    mutable_views: list[TableVersionHandle]  # target for data operation propagation (only set for live tables)
    iterator_cls: Optional[type[ComponentIterator]]
    iterator_args: Optional[exprs.InlineDict]
    num_iterator_cols: int

    # contains complete history of columns, incl dropped ones
    cols: list[Column]
    # contains only user-facing (named) columns visible in this version
    cols_by_name: dict[str, Column]
    # contains only columns visible in this version, both system and user
    cols_by_id: dict[int, Column]
    # needed for _create_tbl_md()
    idx_md: dict[int, schema.IndexMd]
    # contains only actively maintained indices
    idxs_by_name: dict[str, TableVersion.IndexInfo]

    external_stores: dict[str, pxt.io.ExternalStore]
    store_tbl: 'store.StoreBase'

    @dataclasses.dataclass
    class IndexInfo:
        id: int
        name: str
        idx: index.IndexBase
        col: Column
        val_col: Column
        undo_col: Column

    def __init__(
        self,
        id: UUID,
        tbl_md: schema.TableMd,
        effective_version: Optional[int],
        schema_version_md: schema.TableSchemaVersionMd,
        mutable_views: list[TableVersionHandle],
        base_path: Optional[pxt.catalog.TableVersionPath] = None,
        base: Optional[TableVersionHandle] = None,
        # base_store_tbl: Optional['store.StoreBase'] = None,
    ):
        self.id = id
        self.name = tbl_md.name
        self.effective_version = effective_version
        self.version = tbl_md.current_version if effective_version is None else effective_version
        self.comment = schema_version_md.comment
        self.num_retained_versions = schema_version_md.num_retained_versions
        self.schema_version = schema_version_md.schema_version
        self.view_md = tbl_md.view_md  # save this as-is, it's needed for _create_md()
        self.media_validation = MediaValidation[schema_version_md.media_validation.upper()]
        assert not (self.is_view and base is None)
        self.base = base

        # mutable tables need their TableVersionPath for expr eval during updates
        from .table_version_handle import TableVersionHandle
        from .table_version_path import TableVersionPath

        if self.is_snapshot:
            self.path = None
        else:
            self_handle = TableVersionHandle(id, self.effective_version)
            if self.is_view:
                assert base_path is not None
            self.path = TableVersionPath(self_handle, base=base_path)

        if self.is_snapshot:
            self.next_col_id = -1
            self.next_idx_id = -1  # TODO: can snapshots have separate indices?
            self.next_rowid = -1
        else:
            assert tbl_md.current_version == self.version
            self.next_col_id = tbl_md.next_col_id
            self.next_idx_id = tbl_md.next_idx_id
            self.next_rowid = tbl_md.next_row_id

        # view-specific initialization
        from pixeltable import exprs

        predicate_dict = None if self.view_md is None or self.view_md.predicate is None else self.view_md.predicate
        self.predicate = exprs.Expr.from_dict(predicate_dict) if predicate_dict is not None else None
        self.mutable_views = mutable_views

        # component view-specific initialization
        self.iterator_cls = None
        self.iterator_args = None
        self.num_iterator_cols = 0
        if self.view_md is not None and self.view_md.iterator_class_fqn is not None:
            module_name, class_name = tbl_md.view_md.iterator_class_fqn.rsplit('.', 1)
            module = importlib.import_module(module_name)
            self.iterator_cls = getattr(module, class_name)
            self.iterator_args = exprs.InlineDict.from_dict(tbl_md.view_md.iterator_args)
            output_schema, _ = self.iterator_cls.output_schema(**self.iterator_args.to_kwargs())
            self.num_iterator_cols = len(output_schema)
            assert tbl_md.view_md.iterator_args is not None

        # register this table version now so that it's available when we're re-creating value exprs
        cat = pxt.catalog.Catalog.get()
        cat.add_tbl_version(self)

        # init schema after we determined whether we're a component view, and before we create the store table
        self.cols = []
        self.cols_by_name = {}
        self.cols_by_id = {}
        self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        self.external_stores = {}

        self._init_schema(tbl_md, schema_version_md)

        # Init external stores (this needs to happen after the schema is created)
        self._init_external_stores(tbl_md)

    def __hash__(self) -> int:
        return hash(self.id)

    def create_snapshot_copy(self) -> TableVersion:
        """Create a snapshot copy of this TableVersion"""
        assert not self.is_snapshot
        base = self.path.base.tbl_version if self.is_view else None
        return TableVersion(
            self.id,
            self._create_tbl_md(),
            self.version,
            self._create_schema_version_md(preceding_schema_version=0),  # preceding_schema_version: dummy value
            mutable_views=[],
            base=base,
        )

    def create_handle(self) -> TableVersionHandle:
        from .table_version_handle import TableVersionHandle

        return TableVersionHandle(self.id, self.effective_version, tbl_version=self)

    @classmethod
    def create(
        cls,
        dir_id: UUID,
        name: str,
        cols: list[Column],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        # base_path: Optional[pxt.catalog.TableVersionPath] = None,
        view_md: Optional[schema.ViewMd] = None,
    ) -> tuple[UUID, Optional[TableVersion]]:
        session = Env.get().session

        # assign ids
        cols_by_name: dict[str, Column] = {}
        for pos, col in enumerate(cols):
            col.id = pos
            col.schema_version_add = 0
            cols_by_name[col.name] = col
            if col.is_computed:
                col.check_value_expr()

        timestamp = time.time()
        # create schema.Table
        # Column.dependent_cols for existing cols is wrong at this point, but init() will set it correctly
        column_md = cls._create_column_md(cols)
        tbl_id = uuid.uuid4()
        table_md = schema.TableMd(
            tbl_id=str(tbl_id),
            name=name,
            user=None,
            current_version=0,
            current_schema_version=0,
            next_col_id=len(cols),
            next_idx_id=0,
            next_row_id=0,
            column_md=column_md,
            index_md={},
            external_stores=[],
            view_md=view_md,
            additional_md={},
        )
        # create a schema.Table here, we need it to call our c'tor;
        # don't add it to the session yet, we might add index metadata
        tbl_record = schema.Table(id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(table_md))

        # create schema.TableVersion
        table_version_md = schema.TableVersionMd(
            tbl_id=str(tbl_record.id), created_at=timestamp, version=0, schema_version=0, additional_md={}
        )
        tbl_version_record = schema.TableVersion(
            tbl_id=tbl_record.id, version=0, md=dataclasses.asdict(table_version_md)
        )

        # create schema.TableSchemaVersion
        schema_col_md: dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(cols):
            md = schema.SchemaColumn(
                pos=pos,
                name=col.name,
                media_validation=col._media_validation.name.lower() if col._media_validation is not None else None,
            )
            schema_col_md[col.id] = md

        schema_version_md = schema.TableSchemaVersionMd(
            tbl_id=str(tbl_record.id),
            schema_version=0,
            preceding_schema_version=None,
            columns=schema_col_md,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation.name.lower(),
            additional_md={},
        )
        schema_version_record = schema.TableSchemaVersion(
            tbl_id=tbl_record.id, schema_version=0, md=dataclasses.asdict(schema_version_md)
        )

        # if this is purely a snapshot (it doesn't require any additional storage for columns and it doesn't have a
        # predicate to apply at runtime), we don't create a physical table and simply use the base's table version path
        if view_md is not None and view_md.is_snapshot and view_md.predicate is None and len(cols) == 0:
            session.add(tbl_record)
            session.add(tbl_version_record)
            session.add(schema_version_record)
            return tbl_record.id, None

        # assert (base_path is not None) == (view_md is not None)
        is_snapshot = view_md is not None and view_md.is_snapshot
        effective_version = 0 if is_snapshot else None
        base_path = pxt.catalog.TableVersionPath.from_md(view_md.base_versions) if view_md is not None else None
        base = base_path.tbl_version if base_path is not None else None
        tbl_version = cls(
            tbl_record.id, table_md, effective_version, schema_version_md, [], base_path=base_path, base=base
        )

        tbl_version.store_tbl.create()
        if view_md is None or not view_md.is_snapshot:
            # add default indices, after creating the store table
            for col in tbl_version.cols_by_name.values():
                status = tbl_version._add_default_index(col)
                assert status is None or status.num_excs == 0

        # we re-create the tbl_record here, now that we have new index metadata
        tbl_record = schema.Table(id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(tbl_version._create_tbl_md()))
        session.add(tbl_record)
        session.add(tbl_version_record)
        session.add(schema_version_record)
        return tbl_record.id, tbl_version

    @classmethod
    def delete_md(cls, tbl_id: UUID) -> None:
        conn = Env.get().conn
        conn.execute(sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == tbl_id))

    def drop(self) -> None:
        # delete this table and all associated data
        MediaStore.delete(self.id)
        FileCache.get().clear(tbl_id=self.id)
        self.delete_md(self.id)
        self.store_tbl.drop()

        # de-register table version from catalog
        from .catalog import Catalog

        cat = Catalog.get()
        cat.remove_tbl_version(self)

    def _init_schema(self, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd) -> None:
        # create columns first, so the indices can reference them
        self._init_cols(tbl_md, schema_version_md)
        if not self.is_snapshot:
            self._init_idxs(tbl_md)
        # create the sa schema only after creating the columns and indices
        self._init_sa_schema()

    def _init_cols(self, tbl_md: schema.TableMd, schema_version_md: schema.TableSchemaVersionMd) -> None:
        """Initialize self.cols with the columns visible in our effective version"""
        self.cols = []
        self.cols_by_name = {}
        self.cols_by_id = {}
        for col_md in tbl_md.column_md.values():
            schema_col_md = schema_version_md.columns[col_md.id] if col_md.id in schema_version_md.columns else None
            col_name = schema_col_md.name if schema_col_md is not None else None
            media_val = (
                MediaValidation[schema_col_md.media_validation.upper()]
                if schema_col_md is not None and schema_col_md.media_validation is not None
                else None
            )
            col = Column(
                col_id=col_md.id,
                name=col_name,
                col_type=ts.ColumnType.from_dict(col_md.col_type),
                is_pk=col_md.is_pk,
                stored=col_md.stored,
                media_validation=media_val,
                schema_version_add=col_md.schema_version_add,
                schema_version_drop=col_md.schema_version_drop,
                value_expr_dict=col_md.value_expr,
            )
            col.tbl = self.create_handle()
            self.cols.append(col)

            # populate the lookup structures before Expr.from_dict()
            if col_md.schema_version_add > self.schema_version:
                # column was added after this version
                continue
            if col_md.schema_version_drop is not None and col_md.schema_version_drop <= self.schema_version:
                # column was dropped
                continue
            if col.name is not None:
                self.cols_by_name[col.name] = col
            self.cols_by_id[col.id] = col

            # make sure to traverse columns ordered by position = order in which cols were created;
            # this guarantees that references always point backwards
            if col_md.value_expr is not None:
                self._record_refd_columns(col)

    def _init_idxs(self, tbl_md: schema.TableMd) -> None:
        self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        import pixeltable.index as index_module

        for md in tbl_md.index_md.values():
            if (
                md.schema_version_add > self.schema_version
                or md.schema_version_drop is not None
                and md.schema_version_drop <= self.schema_version
            ):
                # index not visible in this schema version
                continue

            # instantiate index object
            cls_name = md.class_fqn.rsplit('.', 1)[-1]
            cls = getattr(index_module, cls_name)
            idx_col = self.path.get_column_by_id(UUID(md.indexed_col_tbl_id), md.indexed_col_id)
            idx = cls.from_dict(idx_col, md.init_args)

            # fix up the sa column type of the index value and undo columns
            val_col = self.cols_by_id[md.index_val_col_id]
            val_col.sa_col_type = idx.index_sa_type()
            val_col._records_errors = False
            undo_col = self.cols_by_id[md.index_val_undo_col_id]
            undo_col.sa_col_type = idx.index_sa_type()
            undo_col._records_errors = False
            idx_info = self.IndexInfo(id=md.id, name=md.name, idx=idx, col=idx_col, val_col=val_col, undo_col=undo_col)
            self.idxs_by_name[md.name] = idx_info

    def _init_sa_schema(self) -> None:
        # create the sqlalchemy schema; do this after instantiating columns, in order to determine whether they
        # need to record errors
        from pixeltable.store import StoreComponentView, StoreTable, StoreView

        if self.is_component_view:
            self.store_tbl = StoreComponentView(self)
        elif self.is_view:
            self.store_tbl = StoreView(self)
        else:
            self.store_tbl = StoreTable(self)

    def _update_md(
        self, timestamp: float, update_tbl_version: bool = True, preceding_schema_version: Optional[int] = None
    ) -> None:
        """Writes table metadata to the database.

        Args:
            timestamp: timestamp of the change
            conn: database connection to use
            update_tbl_version: if `True`, will also write `TableVersion` metadata
            preceding_schema_version: if specified, will also write `TableSchemaVersion` metadata, recording the
                specified preceding schema version
        """
        assert update_tbl_version or preceding_schema_version is None

        conn = Env.get().conn
        conn.execute(
            sql.update(schema.Table.__table__)
            .values({schema.Table.md: dataclasses.asdict(self._create_tbl_md())})
            .where(schema.Table.id == self.id)
        )

        if update_tbl_version:
            version_md = self._create_version_md(timestamp)
            conn.execute(
                sql.insert(schema.TableVersion.__table__).values(
                    tbl_id=self.id, version=self.version, md=dataclasses.asdict(version_md)
                )
            )

        if preceding_schema_version is not None:
            schema_version_md = self._create_schema_version_md(preceding_schema_version)
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__).values(
                    tbl_id=self.id, schema_version=self.schema_version, md=dataclasses.asdict(schema_version_md)
                )
            )

    def ensure_md_loaded(self) -> None:
        """Ensure that table metadata is loaded."""
        for col in self.cols_by_id.values():
            _ = col.value_expr

    def _store_idx_name(self, idx_id: int) -> str:
        """Return name of index in the store, which needs to be globally unique"""
        return f'idx_{self.id.hex}_{idx_id}'

    def add_index(self, col: Column, idx_name: Optional[str], idx: index.IndexBase) -> UpdateStatus:
        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        status = self._add_index(col, idx_name, idx)
        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'Added index {idx_name} on column {col.name} to table {self.name}')
        return status

    def _add_default_index(self, col: Column) -> Optional[UpdateStatus]:
        """Add a B-tree index on this column if it has a compatible type"""
        if not col.stored:
            # if the column is intentionally not stored, we want to avoid the overhead of an index
            return None
        # Skip index for stored media columns produced by an iterator
        if col.col_type.is_media_type() and self.is_iterator_column(col):
            return None
        if not col.col_type.is_scalar_type() and not (col.col_type.is_media_type() and not col.is_computed):
            # wrong type for a B-tree
            return None
        if col.col_type.is_bool_type():
            # B-trees on bools aren't useful
            return None
        status = self._add_index(col, idx_name=None, idx=index.BtreeIndex(col))
        return status

    def _add_index(self, col: Column, idx_name: Optional[str], idx: index.IndexBase) -> UpdateStatus:
        assert not self.is_snapshot
        idx_id = self.next_idx_id
        self.next_idx_id += 1
        if idx_name is None:
            idx_name = f'idx{idx_id}'
        else:
            assert is_valid_identifier(idx_name)
            assert idx_name not in [i.name for i in self.idx_md.values()]

        # add the index value and undo columns (which need to be nullable)
        val_col = Column(
            col_id=self.next_col_id,
            name=None,
            computed_with=idx.index_value_expr(),
            sa_col_type=idx.index_sa_type(),
            stored=True,
            schema_version_add=self.schema_version,
            schema_version_drop=None,
            records_errors=idx.records_value_errors(),
        )
        val_col.tbl = self.create_handle()
        val_col.col_type = val_col.col_type.copy(nullable=True)
        self.next_col_id += 1

        undo_col = Column(
            col_id=self.next_col_id,
            name=None,
            col_type=val_col.col_type,
            sa_col_type=val_col.sa_col_type,
            stored=True,
            schema_version_add=self.schema_version,
            schema_version_drop=None,
            records_errors=False,
        )
        undo_col.tbl = self.create_handle()
        undo_col.col_type = undo_col.col_type.copy(nullable=True)
        self.next_col_id += 1

        # create and register the index metadata
        idx_cls = type(idx)
        idx_md = schema.IndexMd(
            id=idx_id,
            name=idx_name,
            indexed_col_id=col.id,
            indexed_col_tbl_id=str(col.tbl.id),
            index_val_col_id=val_col.id,
            index_val_undo_col_id=undo_col.id,
            schema_version_add=self.schema_version,
            schema_version_drop=None,
            class_fqn=idx_cls.__module__ + '.' + idx_cls.__name__,
            init_args=idx.as_dict(),
        )
        idx_info = self.IndexInfo(id=idx_id, name=idx_name, idx=idx, col=col, val_col=val_col, undo_col=undo_col)
        self.idx_md[idx_id] = idx_md
        self.idxs_by_name[idx_name] = idx_info

        # add the columns and update the metadata
        # TODO support on_error='abort' for indices; it's tricky because of the way metadata changes are entangled
        # with the database operations
        status = self._add_columns([val_col, undo_col], print_stats=False, on_error='ignore')
        # now create the index structure
        idx.create_index(self._store_idx_name(idx_id), val_col)

        return status

    def drop_index(self, idx_id: int) -> None:
        assert not self.is_snapshot
        assert idx_id in self.idx_md

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        idx_md = self.idx_md[idx_id]
        idx_md.schema_version_drop = self.schema_version
        assert idx_md.name in self.idxs_by_name
        idx_info = self.idxs_by_name[idx_md.name]
        # remove this index entry from the active indexes (in memory)
        # and the index metadata (in persistent table metadata)
        del self.idxs_by_name[idx_md.name]
        del self.idx_md[idx_id]

        self._drop_columns([idx_info.val_col, idx_info.undo_col])
        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'Dropped index {idx_md.name} on table {self.name}')

    def add_columns(
        self, cols: Iterable[Column], print_stats: bool, on_error: Literal['abort', 'ignore']
    ) -> UpdateStatus:
        """Adds a column to the table."""
        assert not self.is_snapshot
        assert all(is_valid_identifier(col.name) for col in cols)
        assert all(col.stored is not None for col in cols)
        assert all(col.name not in self.cols_by_name for col in cols)
        for col in cols:
            col.tbl = self.create_handle()
            col.id = self.next_col_id
            self.next_col_id += 1

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        status = self._add_columns(cols, print_stats=print_stats, on_error=on_error)
        for col in cols:
            _ = self._add_default_index(col)
        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'Added columns {[col.name for col in cols]} to table {self.name}, new version: {self.version}')

        msg = (
            f'Added {status.num_rows} column value{"" if status.num_rows == 1 else "s"} '
            f'with {status.num_excs} error{"" if status.num_excs == 1 else "s"}.'
        )
        Env.get().console_logger.info(msg)
        _logger.info(f'Columns {[col.name for col in cols]}: {msg}')
        return status

    def _add_columns(
        self, cols: Iterable[Column], print_stats: bool, on_error: Literal['abort', 'ignore']
    ) -> UpdateStatus:
        """Add and populate columns within the current transaction"""
        cols = list(cols)
        row_count = self.store_tbl.count()
        for col in cols:
            if not col.col_type.nullable and not col.is_computed:
                if row_count > 0:
                    raise excs.Error(
                        f'Cannot add non-nullable column "{col.name}" to table {self.name} with existing rows'
                    )

        num_excs = 0
        cols_with_excs: list[Column] = []
        for col in cols:
            col.schema_version_add = self.schema_version
            # add the column to the lookup structures now, rather than after the store changes executed successfully,
            # because it might be referenced by the next column's value_expr
            self.cols.append(col)
            if col.name is not None:
                self.cols_by_name[col.name] = col
            self.cols_by_id[col.id] = col
            if col.value_expr is not None:
                col.check_value_expr()
                self._record_refd_columns(col)

            if col.is_stored:
                self.store_tbl.add_column(col)

            if not col.is_computed or not col.is_stored or row_count == 0:
                continue

            # populate the column
            from pixeltable.plan import Planner

            plan, value_expr_slot_idx = Planner.create_add_column_plan(self.path, col)
            plan.ctx.num_rows = row_count

            try:
                plan.open()
                try:
                    num_excs = self.store_tbl.load_column(col, plan, value_expr_slot_idx, on_error)
                except sql.exc.DBAPIError as exc:
                    # Wrap the DBAPIError in an excs.Error to unify processing in the subsequent except block
                    raise excs.Error(f'SQL error during execution of computed column `{col.name}`:\n{exc}') from exc
                if num_excs > 0:
                    cols_with_excs.append(col)
            except excs.Error as exc:
                self.cols.pop()
                for col in cols:
                    # remove columns that we already added
                    if col.id not in self.cols_by_id:
                        continue
                    if col.name is not None:
                        del self.cols_by_name[col.name]
                    del self.cols_by_id[col.id]
                # we need to re-initialize the sqlalchemy schema
                self.store_tbl.create_sa_tbl()
                raise exc
            finally:
                plan.close()

        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)
        # TODO(mkornacker): what to do about system columns with exceptions?
        return UpdateStatus(
            num_rows=row_count,
            num_computed_values=row_count,
            num_excs=num_excs,
            cols_with_excs=[f'{col.tbl.get().name}.{col.name}' for col in cols_with_excs if col.name is not None],
        )

    def drop_column(self, col: Column) -> None:
        """Drop a column from the table."""

        assert not self.is_snapshot

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        # drop this column and all dependent index columns and indices
        dropped_cols = [col]
        dropped_idx_names: list[str] = []
        for idx_info in self.idxs_by_name.values():
            if idx_info.col != col:
                continue
            dropped_cols.extend([idx_info.val_col, idx_info.undo_col])
            idx_md = self.idx_md[idx_info.id]
            idx_md.schema_version_drop = self.schema_version
            assert idx_md.name in self.idxs_by_name
            dropped_idx_names.append(idx_md.name)
        # update idxs_by_name
        for idx_name in dropped_idx_names:
            del self.idxs_by_name[idx_name]
        self._drop_columns(dropped_cols)
        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'Dropped column {col.name} from table {self.name}, new version: {self.version}')

    def _drop_columns(self, cols: Iterable[Column]) -> None:
        """Mark columns as dropped"""
        assert not self.is_snapshot

        for col in cols:
            if col.value_expr is not None:
                # update Column.dependent_cols
                for c in self.cols:
                    if c == col:
                        break
                    c.dependent_cols.discard(col)

            col.schema_version_drop = self.schema_version
            if col.name is not None:
                assert col.name in self.cols_by_name
                del self.cols_by_name[col.name]
            assert col.id in self.cols_by_id
            del self.cols_by_id[col.id]

        self.store_tbl.create_sa_tbl()

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column."""
        assert not self.is_snapshot
        if old_name not in self.cols_by_name:
            raise excs.Error(f'Unknown column: {old_name}')
        if not is_valid_identifier(new_name):
            raise excs.Error(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise excs.Error(f'Column {new_name} already exists')
        col = self.cols_by_name[old_name]
        del self.cols_by_name[old_name]
        col.name = new_name
        self.cols_by_name[new_name] = col

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def set_comment(self, new_comment: Optional[str]):
        _logger.info(f'[{self.name}] Updating comment: {new_comment}')
        self.comment = new_comment
        self._create_schema_version()

    def set_num_retained_versions(self, new_num_retained_versions: int):
        _logger.info(
            f'[{self.name}] Updating num_retained_versions: {new_num_retained_versions} (was {self.num_retained_versions})'
        )
        self.num_retained_versions = new_num_retained_versions
        self._create_schema_version()

    def _create_schema_version(self):
        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version
        self._update_md(time.time(), preceding_schema_version=preceding_schema_version)
        _logger.info(f'[{self.name}] Updating table schema to version: {self.version}')

    def insert(
        self,
        rows: Optional[list[dict[str, Any]]],
        df: Optional[pxt.DataFrame],
        print_stats: bool = False,
        fail_on_exception: bool = True,
    ) -> UpdateStatus:
        """
        Insert rows into this table, either from an explicit list of dicts or from a `DataFrame`.
        """
        from pixeltable.plan import Planner

        assert self.is_insertable()
        assert (rows is None) != (df is None)  # Exactly one must be specified
        if rows is not None:
            plan = Planner.create_insert_plan(self, rows, ignore_errors=not fail_on_exception)
        else:
            plan = Planner.create_df_insert_plan(self, df, ignore_errors=not fail_on_exception)

        # this is a base table; we generate rowids during the insert
        def rowids() -> Iterator[int]:
            while True:
                rowid = self.next_rowid
                self.next_rowid += 1
                yield rowid

        return self._insert(plan, time.time(), print_stats=print_stats, rowids=rowids(), abort_on_exc=fail_on_exception)

    def _insert(
        self,
        exec_plan: 'exec.ExecNode',
        timestamp: float,
        *,
        rowids: Optional[Iterator[int]] = None,
        print_stats: bool = False,
        abort_on_exc: bool = False,
    ) -> UpdateStatus:
        """Insert rows produced by exec_plan and propagate to views"""
        # we're creating a new version
        self.version += 1
        result = UpdateStatus()
        num_rows, num_excs, cols_with_excs = self.store_tbl.insert_rows(
            exec_plan, v_min=self.version, rowids=rowids, abort_on_exc=abort_on_exc
        )
        result.num_rows = num_rows
        result.num_excs = num_excs
        result.num_computed_values += exec_plan.ctx.num_computed_exprs * num_rows
        result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
        self._update_md(timestamp)

        # update views
        for view in self.mutable_views:
            from pixeltable.plan import Planner

            plan, _ = Planner.create_view_load_plan(view.get().path, propagates_insert=True)
            status = view.get()._insert(plan, timestamp, print_stats=print_stats)
            result.num_rows += status.num_rows
            result.num_excs += status.num_excs
            result.num_computed_values += status.num_computed_values
            result.cols_with_excs += status.cols_with_excs

        result.cols_with_excs = list(dict.fromkeys(result.cols_with_excs).keys())  # remove duplicates
        if print_stats:
            plan.ctx.profile.print(num_rows=num_rows)
        _logger.info(f'TableVersion {self.name}: new version {self.version}')
        return result

    def update(
        self, value_spec: dict[str, Any], where: Optional[exprs.Expr] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this TableVersionPath.
        Args:
            value_spec: a list of (column, value) pairs specifying the columns to update and their new values.
            where: a predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
        """
        if self.is_snapshot:
            raise excs.Error('Cannot update a snapshot')

        from pixeltable.plan import Planner

        update_spec = self._validate_update_spec(value_spec, allow_pk=False, allow_exprs=True)
        if where is not None:
            if not isinstance(where, exprs.Expr):
                raise excs.Error(f"'where' argument must be a predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.path, where)
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        plan, updated_cols, recomputed_cols = Planner.create_update_plan(self.path, update_spec, [], where, cascade)
        from pixeltable.exprs import SqlElementCache

        result = self.propagate_update(
            plan,
            where.sql_expr(SqlElementCache()) if where is not None else None,
            recomputed_cols,
            base_versions=[],
            timestamp=time.time(),
            cascade=cascade,
            show_progress=True,
        )
        result.updated_cols = updated_cols
        return result

    def batch_update(
        self,
        batch: list[dict[Column, exprs.Expr]],
        rowids: list[tuple[int, ...]],
        insert_if_not_exists: bool,
        error_if_not_exists: bool,
        cascade: bool = True,
    ) -> UpdateStatus:
        """Update rows in batch.
        Args:
            batch: one dict per row, each mapping Columns to LiteralExprs representing the new values
            rowids: if not empty, one tuple per row, each containing the rowid values for the corresponding row in batch
        """
        # if we do lookups of rowids, we must have one for each row in the batch
        assert len(rowids) == 0 or len(rowids) == len(batch)
        cols_with_excs: set[str] = set()

        from pixeltable.plan import Planner

        plan, row_update_node, delete_where_clause, updated_cols, recomputed_cols = Planner.create_batch_update_plan(
            self.path, batch, rowids, cascade=cascade
        )
        result = self.propagate_update(
            plan, delete_where_clause, recomputed_cols, base_versions=[], timestamp=time.time(), cascade=cascade
        )
        result.updated_cols = [c.qualified_name for c in updated_cols]

        unmatched_rows = row_update_node.unmatched_rows()
        if len(unmatched_rows) > 0:
            if error_if_not_exists:
                raise excs.Error(f'batch_update(): {len(unmatched_rows)} row(s) not found')
            if insert_if_not_exists:
                insert_status = self.insert(unmatched_rows, None, print_stats=False, fail_on_exception=False)
                result += insert_status
        return result

    def _validate_update_spec(
        self, value_spec: dict[str, Any], allow_pk: bool, allow_exprs: bool
    ) -> dict[Column, exprs.Expr]:
        update_targets: dict[Column, exprs.Expr] = {}
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name == _ROWID_COLUMN_NAME:
                # a valid rowid is a list of ints, one per rowid column
                assert len(val) == len(self.store_tbl.rowid_columns())
                for el in val:
                    assert isinstance(el, int)
                continue
            col = self.path.get_column(col_name, include_bases=False)
            if col is None:
                # TODO: return more informative error if this is trying to update a base column
                raise excs.Error(f'Column {col_name} unknown')
            if col.is_computed:
                raise excs.Error(f'Column {col_name} is computed and cannot be updated')
            if col.is_pk and not allow_pk:
                raise excs.Error(f'Column {col_name} is a primary key column and cannot be updated')

            # make sure that the value is compatible with the column type
            value_expr: exprs.Expr
            try:
                # check if this is a literal
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except (TypeError, jsonschema.exceptions.ValidationError):
                if not allow_exprs:
                    raise excs.Error(
                        f'Column {col_name}: value {val!r} is not a valid literal for this column '
                        f'(expected {col.col_type})'
                    )
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.is_supertype_of(value_expr.col_type, ignore_nullable=True):
                    raise excs.Error(
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    )
            update_targets[col] = value_expr

        return update_targets

    def propagate_update(
        self,
        plan: Optional[exec.ExecNode],
        where_clause: Optional[sql.ColumnElement],
        recomputed_view_cols: list[Column],
        base_versions: list[Optional[int]],
        timestamp: float,
        cascade: bool,
        show_progress: bool = True,
    ) -> UpdateStatus:
        result = UpdateStatus()
        if plan is not None:
            # we're creating a new version
            self.version += 1
            result.num_rows, result.num_excs, cols_with_excs = self.store_tbl.insert_rows(
                plan, v_min=self.version, show_progress=show_progress
            )
            result.cols_with_excs = [f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs]
            self.store_tbl.delete_rows(
                self.version, base_versions=base_versions, match_on_vmin=True, where_clause=where_clause
            )
            self._update_md(timestamp)

        if cascade:
            base_versions = [None if plan is None else self.version] + base_versions  # don't update in place
            # propagate to views
            for view in self.mutable_views:
                recomputed_cols = [col for col in recomputed_view_cols if col.tbl == view]
                plan = None
                if len(recomputed_cols) > 0:
                    from pixeltable.plan import Planner

                    plan = Planner.create_view_update_plan(view.get().path, recompute_targets=recomputed_cols)
                status = view.get().propagate_update(
                    plan, None, recomputed_view_cols, base_versions=base_versions, timestamp=timestamp, cascade=True
                )
                result.num_rows += status.num_rows
                result.num_excs += status.num_excs
                result.cols_with_excs += status.cols_with_excs

        result.cols_with_excs = list(dict.fromkeys(result.cols_with_excs).keys())  # remove duplicates
        return result

    def delete(self, where: Optional[exprs.Expr] = None) -> UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a predicate to filter rows to delete.
        """
        assert self.is_insertable()
        from pixeltable.exprs import Expr
        from pixeltable.plan import Planner

        sql_where_clause: Optional[Expr] = None
        if where is not None:
            if not isinstance(where, Expr):
                raise excs.Error(f"'where' argument must be a predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.path, where)
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')
            sql_where_clause = analysis_info.sql_where_clause

        num_rows = self.propagate_delete(sql_where_clause, base_versions=[], timestamp=time.time())

        status = UpdateStatus(num_rows=num_rows)
        return status

    def propagate_delete(
        self, where: Optional[exprs.Expr], base_versions: list[Optional[int]], timestamp: float
    ) -> int:
        """Delete rows in this table and propagate to views.
        Args:
            where: a predicate to filter rows to delete.
        Returns:
            number of deleted rows
        """
        sql_where_clause = where.sql_expr(exprs.SqlElementCache()) if where is not None else None
        num_rows = self.store_tbl.delete_rows(
            self.version + 1, base_versions=base_versions, match_on_vmin=False, where_clause=sql_where_clause
        )
        if num_rows > 0:
            # we're creating a new version
            self.version += 1
            self._update_md(timestamp)
        else:
            pass
        for view in self.mutable_views:
            num_rows += view.get().propagate_delete(
                where=None, base_versions=[self.version] + base_versions, timestamp=timestamp
            )
        return num_rows

    def revert(self) -> None:
        """Reverts the table to the previous version."""
        assert not self.is_snapshot
        if self.version == 0:
            raise excs.Error('Cannot revert version 0')
        self._revert()

    def _delete_column(self, col: Column) -> None:
        """Physically remove the column from the schema and the store table"""
        if col.is_stored:
            self.store_tbl.drop_column(col)
        self.cols.remove(col)
        if col.name is not None:
            del self.cols_by_name[col.name]
        del self.cols_by_id[col.id]

    def _revert(self) -> None:
        """Reverts this table version and propagates to views"""
        conn = Env.get().conn
        # make sure we don't have a snapshot referencing this version
        # (unclear how to express this with sqlalchemy)
        query = (
            f"select ts.dir_id, ts.md->'name' "
            f'from {schema.Table.__tablename__} ts '
            f"cross join lateral jsonb_path_query(md, '$.view_md.base_versions[*]') as tbl_version "
            f"where tbl_version->>0 = '{self.id.hex}' and (tbl_version->>1)::int = {self.version}"
        )
        result = list(conn.execute(sql.text(query)))
        if len(result) > 0:
            names = [row[1] for row in result]
            raise excs.Error(
                (
                    f'Current version is needed for {len(result)} snapshot{"s" if len(result) > 1 else ""} '
                    f'({", ".join(names)})'
                )
            )

        # delete newly-added data
        MediaStore.delete(self.id, version=self.version)
        conn.execute(sql.delete(self.store_tbl.sa_tbl).where(self.store_tbl.sa_tbl.c.v_min == self.version))

        # revert new deletions
        set_clause: dict[sql.Column, Any] = {self.store_tbl.sa_tbl.c.v_max: schema.Table.MAX_VERSION}
        for index_info in self.idxs_by_name.values():
            # copy the index value back from the undo column and reset the undo column to NULL
            set_clause[index_info.val_col.sa_col] = index_info.undo_col.sa_col
            set_clause[index_info.undo_col.sa_col] = None
        stmt = sql.update(self.store_tbl.sa_tbl).values(set_clause).where(self.store_tbl.sa_tbl.c.v_max == self.version)
        conn.execute(stmt)

        # revert schema changes
        if self.version == self.schema_version:
            # delete newly-added columns
            added_cols = [col for col in self.cols if col.schema_version_add == self.schema_version]
            if len(added_cols) > 0:
                next_col_id = min(col.id for col in added_cols)
                for col in added_cols:
                    self._delete_column(col)
                self.next_col_id = next_col_id

            # remove newly-added indices from the lookup structures
            # (the value and undo columns got removed in the preceding step)
            added_idx_md = [md for md in self.idx_md.values() if md.schema_version_add == self.schema_version]
            if len(added_idx_md) > 0:
                next_idx_id = min(md.id for md in added_idx_md)
                for md in added_idx_md:
                    del self.idx_md[md.id]
                    del self.idxs_by_name[md.name]
                self.next_idx_id = next_idx_id

            # make newly-dropped columns visible again
            dropped_cols = [col for col in self.cols if col.schema_version_drop == self.schema_version]
            for col in dropped_cols:
                col.schema_version_drop = None

            # make newly-dropped indices visible again
            dropped_idx_md = [md for md in self.idx_md.values() if md.schema_version_drop == self.schema_version]
            for md in dropped_idx_md:
                md.schema_version_drop = None

            session = Env.get().session
            # we need to determine the preceding schema version and reload the schema
            schema_version_md_dict = (
                session.query(schema.TableSchemaVersion.md)
                .where(schema.TableSchemaVersion.tbl_id == self.id)
                .where(schema.TableSchemaVersion.schema_version == self.schema_version)
                .scalar()
            )
            preceding_schema_version = schema_version_md_dict['preceding_schema_version']
            preceding_schema_version_md_dict = (
                session.query(schema.TableSchemaVersion.md)
                .where(schema.TableSchemaVersion.tbl_id == self.id)
                .where(schema.TableSchemaVersion.schema_version == preceding_schema_version)
                .scalar()
            )
            preceding_schema_version_md = schema.md_from_dict(
                schema.TableSchemaVersionMd, preceding_schema_version_md_dict
            )
            tbl_md = self._create_tbl_md()
            self._init_schema(tbl_md, preceding_schema_version_md)

            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__)
                .where(schema.TableSchemaVersion.tbl_id == self.id)
                .where(schema.TableSchemaVersion.schema_version == self.schema_version)
            )
            self.schema_version = preceding_schema_version
            self.comment = preceding_schema_version_md.comment
            self.num_retained_versions = preceding_schema_version_md.num_retained_versions

        conn.execute(
            sql.delete(schema.TableVersion.__table__)
            .where(schema.TableVersion.tbl_id == self.id)
            .where(schema.TableVersion.version == self.version)
        )
        self.version -= 1
        conn.execute(
            sql.update(schema.Table.__table__)
            .values({schema.Table.md: dataclasses.asdict(self._create_tbl_md())})
            .where(schema.Table.id == self.id)
        )

        # propagate to views
        for view in self.mutable_views:
            view.get()._revert()
        _logger.info(f'TableVersion {self.name}: reverted to version {self.version}')

    def _init_external_stores(self, tbl_md: schema.TableMd) -> None:
        for store_md in tbl_md.external_stores:
            store_cls = resolve_symbol(store_md['class'])
            assert isinstance(store_cls, type) and issubclass(store_cls, pxt.io.ExternalStore)
            store = store_cls.from_dict(store_md['md'])
            self.external_stores[store.name] = store

    def link_external_store(self, store: pxt.io.ExternalStore) -> None:
        store.link(self)  # May result in additional metadata changes
        self.external_stores[store.name] = store
        self._update_md(time.time(), update_tbl_version=False)

    def unlink_external_store(self, store_name: str, delete_external_data: bool) -> None:
        assert store_name in self.external_stores
        store = self.external_stores[store_name]
        store.unlink(self)  # May result in additional metadata changes
        del self.external_stores[store_name]
        self._update_md(time.time(), update_tbl_version=False)

        if delete_external_data and isinstance(store, pxt.io.external_store.Project):
            store.delete()

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    @property
    def is_view(self) -> bool:
        return self.view_md is not None

    @property
    def include_base_columns(self) -> bool:
        return self.view_md is not None and self.view_md.include_base_columns

    @property
    def is_component_view(self) -> bool:
        return self.iterator_cls is not None

    def is_insertable(self) -> bool:
        """Returns True if this corresponds to an InsertableTable"""
        return not self.is_snapshot and not self.is_view

    def is_iterator_column(self, col: Column) -> bool:
        """Returns True if col is produced by an iterator"""
        # the iterator columns directly follow the pos column
        return self.is_component_view and col.id > 0 and col.id < self.num_iterator_cols + 1

    def is_system_column(self, col: Column) -> bool:
        """Return True if column was created by Pixeltable"""
        if col.name == _POS_COLUMN_NAME and self.is_component_view:
            return True
        return False

    def user_columns(self) -> list[Column]:
        """Return all non-system columns"""
        return [c for c in self.cols if not self.is_system_column(c)]

    def primary_key_columns(self) -> list[Column]:
        """Return all non-system columns"""
        return [c for c in self.cols if c.is_pk]

    @property
    def primary_key(self) -> list[str]:
        """Return the names of the primary key columns"""
        return [c.name for c in self.cols if c.is_pk]

    def get_required_col_names(self) -> list[str]:
        """Return the names of all columns for which values must be specified in insert()"""
        assert not self.is_view
        names = [c.name for c in self.cols_by_name.values() if not c.is_computed and not c.col_type.nullable]
        return names

    def get_computed_col_names(self) -> list[str]:
        """Return the names of all computed columns"""
        names = [c.name for c in self.cols_by_name.values() if c.is_computed]
        return names

    def _record_refd_columns(self, col: Column) -> None:
        """Update Column.dependent_cols for all cols referenced in col.value_expr."""
        import pixeltable.exprs as exprs

        if col.value_expr_dict is not None:
            # if we have a value_expr_dict, use that instead of instantiating the value_expr
            refd_cols = exprs.Expr.get_refd_columns(col.value_expr_dict)
        else:
            refd_cols = [e.col for e in col.value_expr.subexprs(expr_class=exprs.ColumnRef)]
        for refd_col in refd_cols:
            refd_col.dependent_cols.add(col)

    def get_idx_val_columns(self, cols: Iterable[Column]) -> set[Column]:
        result = {info.val_col for col in cols for info in col.get_idx_info().values()}
        return result

    def get_dependent_columns(self, cols: Iterable[Column]) -> set[Column]:
        """
        Return the set of columns that transitively depend on any of the given ones.
        """
        result = {dependent_col for col in cols for dependent_col in col.dependent_cols}
        if len(result) > 0:
            result.update(self.get_dependent_columns(result))
        return result

    def num_rowid_columns(self) -> int:
        """Return the number of columns of the rowids, without accessing store_tbl"""
        if self.is_component_view:
            return 1 + self.base.get().num_rowid_columns()
        return 1

    @classmethod
    def _create_column_md(cls, cols: list[Column]) -> dict[int, schema.ColumnMd]:
        column_md: dict[int, schema.ColumnMd] = {}
        for col in cols:
            value_expr_dict = col.value_expr.as_dict() if col.value_expr is not None else None
            column_md[col.id] = schema.ColumnMd(
                id=col.id,
                col_type=col.col_type.as_dict(),
                is_pk=col.is_pk,
                schema_version_add=col.schema_version_add,
                schema_version_drop=col.schema_version_drop,
                value_expr=value_expr_dict,
                stored=col.stored,
            )
        return column_md

    @classmethod
    def _create_stores_md(cls, stores: Iterable[pxt.io.ExternalStore]) -> list[dict[str, Any]]:
        return [
            {'class': f'{type(store).__module__}.{type(store).__qualname__}', 'md': store.as_dict()} for store in stores
        ]

    def _create_tbl_md(self) -> schema.TableMd:
        return schema.TableMd(
            tbl_id=str(self.id),
            name=self.name,
            user=None,
            current_version=self.version,
            current_schema_version=self.schema_version,
            next_col_id=self.next_col_id,
            next_idx_id=self.next_idx_id,
            next_row_id=self.next_rowid,
            column_md=self._create_column_md(self.cols),
            index_md=self.idx_md,
            external_stores=self._create_stores_md(self.external_stores.values()),
            view_md=self.view_md,
            additional_md={},
        )

    def _create_version_md(self, timestamp: float) -> schema.TableVersionMd:
        return schema.TableVersionMd(
            tbl_id=str(self.id),
            created_at=timestamp,
            version=self.version,
            schema_version=self.schema_version,
            additional_md={},
        )

    def _create_schema_version_md(self, preceding_schema_version: int) -> schema.TableSchemaVersionMd:
        column_md: dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(self.cols_by_name.values()):
            column_md[col.id] = schema.SchemaColumn(
                pos=pos,
                name=col.name,
                media_validation=col._media_validation.name.lower() if col._media_validation is not None else None,
            )
        # preceding_schema_version to be set by the caller
        return schema.TableSchemaVersionMd(
            tbl_id=str(self.id),
            schema_version=self.schema_version,
            preceding_schema_version=preceding_schema_version,
            columns=column_md,
            num_retained_versions=self.num_retained_versions,
            comment=self.comment,
            media_validation=self.media_validation.name.lower(),
            additional_md={},
        )

    def as_dict(self) -> dict:
        return {'id': str(self.id), 'effective_version': self.effective_version}

    @classmethod
    def from_dict(cls, d: dict) -> TableVersion:
        import pixeltable.catalog as catalog

        id = UUID(d['id'])
        effective_version = d['effective_version']
        return catalog.Catalog.get().get_tbl_version(id, effective_version)
