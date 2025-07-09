from __future__ import annotations

import copy
import dataclasses
import importlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional, Tuple
from uuid import UUID

import jsonschema.exceptions
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import exprs, index
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.exception_handler import run_cleanup_on_exception
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

if TYPE_CHECKING:
    from pixeltable.plan import SampleClause


from ..func.globals import resolve_symbol
from .column import Column
from .globals import _POS_COLUMN_NAME, _ROWID_COLUMN_NAME, MediaValidation, is_valid_identifier
from .update_status import RowCountStats, UpdateStatus

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

    Only TableVersion and Catalog interact directly with stored metadata. Everything else needs to go through these
    two classes.
    """

    id: UUID

    # record metadata stored in catalog
    _tbl_md: schema.TableMd
    _schema_version_md: schema.TableSchemaVersionMd

    effective_version: Optional[int]
    path: Optional[pxt.catalog.TableVersionPath]  # only set for live tables; needed to resolve computed cols
    base: Optional[TableVersionHandle]  # only set for views
    predicate: Optional[exprs.Expr]
    sample_clause: Optional['SampleClause']

    iterator_cls: Optional[type[ComponentIterator]]
    iterator_args: Optional[exprs.InlineDict]
    num_iterator_cols: int

    # target for data operation propagation (only set for non-snapshots, and only records non-snapshot views)
    mutable_views: set[TableVersionHandle]

    # contains complete history of columns, incl dropped ones
    cols: list[Column]
    # contains only user-facing (named) columns visible in this version
    cols_by_name: dict[str, Column]
    # contains only columns visible in this version, both system and user
    cols_by_id: dict[int, Column]
    # contains only actively maintained indices
    idxs_by_name: dict[str, TableVersion.IndexInfo]

    external_stores: dict[str, pxt.io.ExternalStore]
    store_tbl: Optional['store.StoreBase']

    # used by Catalog to invalidate cached instances at the end of a transaction;
    # True if this instance reflects the state of stored metadata in the context of this transaction and
    # it is the instance cached in Catalog
    is_validated: bool

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
    ):
        self.is_validated = True  # a freshly constructed instance is always valid
        self.id = id
        self._tbl_md = copy.deepcopy(tbl_md)
        self._schema_version_md = copy.deepcopy(schema_version_md)
        self.effective_version = effective_version
        assert not (self.is_view and base is None)
        self.base = base
        self.store_tbl = None

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

        # view-specific initialization
        from pixeltable import exprs
        from pixeltable.plan import SampleClause

        predicate_dict = None if self.view_md is None or self.view_md.predicate is None else self.view_md.predicate
        self.predicate = exprs.Expr.from_dict(predicate_dict) if predicate_dict is not None else None
        sample_dict = None if self.view_md is None or self.view_md.sample_clause is None else self.view_md.sample_clause
        self.sample_clause = SampleClause.from_dict(sample_dict) if sample_dict is not None else None

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

        self.mutable_views = set(mutable_views)
        assert self.is_mutable or len(self.mutable_views) == 0

        self.cols = []
        self.cols_by_name = {}
        self.cols_by_id = {}
        self.idxs_by_name = {}
        self.external_stores = {}

    def __hash__(self) -> int:
        return hash(self.id)

    def create_snapshot_copy(self) -> TableVersion:
        """Create a snapshot copy of this TableVersion"""
        assert not self.is_snapshot
        base = self.path.base.tbl_version if self.is_view else None
        return TableVersion(self.id, self.tbl_md, self.version, self.schema_version_md, mutable_views=[], base=base)

    @property
    def versioned_name(self) -> str:
        if self.effective_version is None:
            return self.name
        else:
            return f'{self.name}:{self.effective_version}'

    @property
    def handle(self) -> 'TableVersionHandle':
        from .table_version_handle import TableVersionHandle

        return TableVersionHandle(self.id, self.effective_version, self)

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
        user = Env.get().user

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
        tbl_id_str = str(tbl_id)
        table_md = schema.TableMd(
            tbl_id=tbl_id_str,
            name=name,
            user=user,
            is_replica=False,
            current_version=0,
            current_schema_version=0,
            next_col_id=len(cols),
            next_idx_id=0,
            next_row_id=0,
            view_sn=0,
            column_md=column_md,
            index_md={},
            external_stores=[],
            view_md=view_md,
            additional_md={},
        )

        # create schema.TableVersion of the initial version
        table_version_md = schema.TableVersionMd(
            tbl_id=tbl_id_str,
            created_at=timestamp,
            version=0,
            schema_version=0,
            user=user,
            update_status=None,
            additional_md={},
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
            tbl_id=tbl_id_str,
            schema_version=0,
            preceding_schema_version=None,
            columns=schema_col_md,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation.name.lower(),
            additional_md={},
        )

        cat = pxt.catalog.Catalog.get()

        # if this is purely a snapshot (it doesn't require any additional storage for columns and it doesn't have a
        # predicate to apply at runtime), we don't create a physical table and simply use the base's table version path
        if (
            view_md is not None
            and view_md.is_snapshot
            and view_md.predicate is None
            and view_md.sample_clause is None
            and len(cols) == 0
        ):
            cat.store_tbl_md(
                tbl_id=tbl_id,
                dir_id=dir_id,
                tbl_md=table_md,
                version_md=table_version_md,
                schema_version_md=schema_version_md,
            )
            return tbl_id, None

        # assert (base_path is not None) == (view_md is not None)
        is_snapshot = view_md is not None and view_md.is_snapshot
        effective_version = 0 if is_snapshot else None
        base_path = pxt.catalog.TableVersionPath.from_md(view_md.base_versions) if view_md is not None else None
        base = base_path.tbl_version if base_path is not None else None
        tbl_version = cls(tbl_id, table_md, effective_version, schema_version_md, [], base_path=base_path, base=base)
        # TODO: break this up, so that Catalog.create_table() registers tbl_version
        cat._tbl_versions[tbl_id, effective_version] = tbl_version
        tbl_version.init()
        tbl_version.store_tbl.create()
        is_mutable = not is_snapshot and not table_md.is_replica
        if base is not None and base.get().is_mutable and is_mutable:
            from .table_version_handle import TableVersionHandle

            handle = TableVersionHandle(tbl_version.id, effective_version)
            assert handle not in base.get().mutable_views
            base.get().mutable_views.add(handle)

        if view_md is None or not view_md.is_snapshot:
            # add default indices, after creating the store table
            for col in tbl_version.cols_by_name.values():
                status = tbl_version._add_default_index(col)
                assert status is None or status.num_excs == 0

        cat.store_tbl_md(
            tbl_id=tbl_id,
            dir_id=dir_id,
            tbl_md=tbl_version.tbl_md,
            version_md=table_version_md,
            schema_version_md=schema_version_md,
        )
        return tbl_id, tbl_version

    @classmethod
    def create_replica(cls, md: schema.FullTableMd) -> TableVersion:
        assert Env.get().in_xact
        tbl_id = UUID(md.tbl_md.tbl_id)
        _logger.info(f'Creating replica table version {tbl_id}:{md.version_md.version}.')
        view_md = md.tbl_md.view_md
        base_path = pxt.catalog.TableVersionPath.from_md(view_md.base_versions) if view_md is not None else None
        base = base_path.tbl_version if base_path is not None else None
        tbl_version = cls(
            tbl_id, md.tbl_md, md.version_md.version, md.schema_version_md, [], base_path=base_path, base=base
        )
        cat = pxt.catalog.Catalog.get()
        # We're creating a new TableVersion replica, so we should never have seen this particular
        # TableVersion instance before.
        assert tbl_version.effective_version is not None
        assert (tbl_version.id, tbl_version.effective_version) not in cat._tbl_versions
        cat._tbl_versions[tbl_version.id, tbl_version.effective_version] = tbl_version
        tbl_version.init()
        tbl_version.store_tbl.create()
        tbl_version.store_tbl.ensure_columns_exist(col for col in tbl_version.cols if col.is_stored)
        return tbl_version

    def drop(self) -> None:
        if self.is_view and self.is_mutable:
            # update mutable_views
            # TODO: invalidate base to force reload
            from .table_version_handle import TableVersionHandle

            assert self.base is not None
            if self.base.get().is_mutable:
                self.base.get().mutable_views.remove(TableVersionHandle.create(self))

        # cat = Catalog.get()
        # delete this table and all associated data
        MediaStore.delete(self.id)
        FileCache.get().clear(tbl_id=self.id)
        # cat.delete_tbl_md(self.id)
        self.store_tbl.drop()
        # de-register table version from catalog
        # cat.remove_tbl_version(self)

    def init(self) -> None:
        """
        Initialize schema-related in-memory metadata separately, now that this TableVersion instance is visible
        in Catalog.
        """
        from .catalog import Catalog

        cat = Catalog.get()
        assert (self.id, self.effective_version) in cat._tbl_versions
        self._init_schema()
        if not self.is_snapshot:
            cat.record_column_dependencies(self)

        # init external stores; this needs to happen after the schema is created
        self._init_external_stores()

    def _init_schema(self) -> None:
        # create columns first, so the indices can reference them
        self._init_cols()
        if not self.is_snapshot:
            self._init_idxs()
        # create the sa schema only after creating the columns and indices
        self._init_sa_schema()

        # created value_exprs after everything else has been initialized
        for col in self.cols_by_id.values():
            col.init_value_expr()

    def _init_cols(self) -> None:
        """Initialize self.cols with the columns visible in our effective version"""
        self.cols = []
        self.cols_by_name = {}
        self.cols_by_id = {}
        # Sort columns in column_md by the position specified in col_md.id to guarantee that all references
        # point backward.
        sorted_column_md = sorted(self.tbl_md.column_md.values(), key=lambda item: item.id)
        for col_md in sorted_column_md:
            schema_col_md = self.schema_version_md.columns.get(col_md.id)
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
                tbl=self,
            )
            col.tbl = self
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

            # # make sure to traverse columns ordered by position = order in which cols were created;
            # # this guarantees that references always point backwards
            # if not self.is_snapshot and col_md.value_expr is not None:
            #     self._record_refd_columns(col)

    def _init_idxs(self) -> None:
        # self.idx_md = tbl_md.index_md
        self.idxs_by_name = {}
        import pixeltable.index as index_module

        for md in self.tbl_md.index_md.values():
            if md.schema_version_add > self.schema_version or (
                md.schema_version_drop is not None and md.schema_version_drop <= self.schema_version
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
            val_col._stores_cellmd = False
            undo_col = self.cols_by_id[md.index_val_undo_col_id]
            undo_col.sa_col_type = idx.index_sa_type()
            undo_col._stores_cellmd = False
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

    def _write_md(
        self,
        new_version: bool,
        new_version_ts: float,
        new_schema_version: bool,
        update_status: Optional[UpdateStatus] = None,
    ) -> None:
        """Writes table metadata to the database.

        Args:
            timestamp: timestamp of the change
            update_tbl_version: if `True`, will also write `TableVersion` metadata
            preceding_schema_version: if specified, will also write `TableSchemaVersion` metadata, recording the
                specified preceding schema version
        """
        from pixeltable.catalog import Catalog

        version_md = self._create_version_md(new_version_ts, update_status=update_status) if new_version else None

        Catalog.get().store_tbl_md(
            self.id, None, self._tbl_md, version_md, self._schema_version_md if new_schema_version else None
        )

    def _write_md_update_status(self, new_version_ts: float, update_status: UpdateStatus) -> None:
        """Writes a new update_status in the table version metadata in the database.

        Args:
            timestamp: timestamp of the change
            update_status: UpdateStatus to be updated in the database
        """
        from pixeltable.catalog import Catalog

        Catalog.get().update_tbl_version_md(self._create_version_md(new_version_ts, update_status))

    def _store_idx_name(self, idx_id: int) -> str:
        """Return name of index in the store, which needs to be globally unique"""
        return f'idx_{self.id.hex}_{idx_id}'

    def add_index(self, col: Column, idx_name: Optional[str], idx: index.IndexBase) -> UpdateStatus:
        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version
        status = self._add_index(col, idx_name, idx)
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)
        _logger.info(f'Added index {idx_name} on column {col.name} to table {self.name}')
        return status

    def _is_btree_indexable(self, col: Column) -> bool:
        if not col.stored:
            # if the column is intentionally not stored, we want to avoid the overhead of an index
            return False
        # Skip index for stored media columns produced by an iterator
        if col.col_type.is_media_type() and self.is_iterator_column(col):
            return False
        if not col.col_type.is_scalar_type() and not (col.col_type.is_media_type() and not col.is_computed):
            # wrong type for a B-tree
            return False
        if col.col_type.is_bool_type():  # noqa : SIM103 Supress `Return the negated condition directly` check
            # B-trees on bools aren't useful
            return False
        return True

    def _add_default_index(self, col: Column) -> Optional[UpdateStatus]:
        """Add a B-tree index on this column if it has a compatible type"""
        if not self._is_btree_indexable(col):
            return None
        status = self._add_index(col, idx_name=None, idx=index.BtreeIndex(col))
        return status

    def _create_index_columns(self, idx: index.IndexBase) -> Tuple[Column, Column]:
        """Create value and undo columns for the given index.
        Args:
            idx:  index for which columns will be created.
        Returns:
            A tuple containing the value column and the undo column.
        """
        assert not self.is_snapshot
        # add the index value and undo columns (which need to be nullable)
        val_col = Column(
            col_id=self.next_col_id,
            name=None,
            computed_with=idx.index_value_expr(),
            sa_col_type=idx.index_sa_type(),
            stored=True,
            schema_version_add=self.schema_version,
            schema_version_drop=None,
            stores_cellmd=idx.records_value_errors(),
        )
        val_col.tbl = self
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
            stores_cellmd=False,
        )
        undo_col.tbl = self
        undo_col.col_type = undo_col.col_type.copy(nullable=True)
        self.next_col_id += 1
        return val_col, undo_col

    def _create_index(
        self, col: Column, val_col: Column, undo_col: Column, idx_name: Optional[str], idx: index.IndexBase
    ) -> None:
        """Create the given index along with index md"""
        idx_id = self.next_idx_id
        self.next_idx_id += 1
        if idx_name is None:
            idx_name = f'idx{idx_id}'
        else:
            assert is_valid_identifier(idx_name)
            assert idx_name not in [i.name for i in self._tbl_md.index_md.values()]
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
        self._tbl_md.index_md[idx_id] = idx_md
        self.idxs_by_name[idx_name] = idx_info
        try:
            idx.create_index(self._store_idx_name(idx_id), val_col)
        finally:

            def cleanup_index() -> None:
                """Delete the newly added in-memory index structure"""
                del self.idxs_by_name[idx_name]
                del self._tbl_md.index_md[idx_id]
                self.next_idx_id = idx_id

            # Run cleanup only if there has been an exception; otherwise, skip cleanup.
            run_cleanup_on_exception(cleanup_index)

    def _add_index(self, col: Column, idx_name: Optional[str], idx: index.IndexBase) -> UpdateStatus:
        val_col, undo_vol = self._create_index_columns(idx)
        # add the columns and update the metadata
        # TODO support on_error='abort' for indices; it's tricky because of the way metadata changes are entangled
        # with the database operations
        status = self._add_columns([val_col, undo_vol], print_stats=False, on_error='ignore')
        # now create the index structure
        self._create_index(col, val_col, undo_vol, idx_name, idx)
        return status

    def drop_index(self, idx_id: int) -> None:
        assert not self.is_snapshot
        assert idx_id in self._tbl_md.index_md

        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version
        idx_md = self._tbl_md.index_md[idx_id]
        idx_md.schema_version_drop = self.schema_version
        assert idx_md.name in self.idxs_by_name
        idx_info = self.idxs_by_name[idx_md.name]
        # remove this index entry from the active indexes (in memory)
        # and the index metadata (in persistent table metadata)
        # TODO: this is wrong, it breaks revert()
        del self.idxs_by_name[idx_md.name]
        del self._tbl_md.index_md[idx_id]

        self._drop_columns([idx_info.val_col, idx_info.undo_col])
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)
        _logger.info(f'Dropped index {idx_md.name} on table {self.name}')

    def add_columns(
        self, cols: Iterable[Column], print_stats: bool, on_error: Literal['abort', 'ignore']
    ) -> UpdateStatus:
        """Adds columns to the table."""
        assert not self.is_snapshot
        assert all(is_valid_identifier(col.name) for col in cols if col.name is not None)
        assert all(col.stored is not None for col in cols)
        assert all(col.name not in self.cols_by_name for col in cols if col.name is not None)
        for col in cols:
            col.tbl = self
            col.id = self.next_col_id
            self.next_col_id += 1

        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version
        index_cols: dict[Column, tuple[index.BtreeIndex, Column, Column]] = {}
        all_cols: list[Column] = []
        for col in cols:
            all_cols.append(col)
            if col.name is not None and self._is_btree_indexable(col):
                idx = index.BtreeIndex(col)
                val_col, undo_col = self._create_index_columns(idx)
                index_cols[col] = (idx, val_col, undo_col)
                all_cols.append(val_col)
                all_cols.append(undo_col)
        # Add all columns
        status = self._add_columns(all_cols, print_stats=print_stats, on_error=on_error)
        # Create indices and their md records
        for col, (idx, val_col, undo_col) in index_cols.items():
            self._create_index(col, val_col, undo_col, idx_name=None, idx=idx)
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True, update_status=status)
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
        cols_to_add = list(cols)
        row_count = self.store_tbl.count()
        for col in cols_to_add:
            if not col.col_type.nullable and not col.is_computed and row_count > 0:
                raise excs.Error(
                    f'Cannot add non-nullable column {col.name!r} to table {self.name!r} with existing rows'
                )

        computed_values = 0
        num_excs = 0
        cols_with_excs: list[Column] = []
        for col in cols_to_add:
            excs_per_col = 0
            col.schema_version_add = self.schema_version
            # add the column to the lookup structures now, rather than after the store changes executed successfully,
            # because it might be referenced by the next column's value_expr
            self.cols.append(col)
            if col.name is not None:
                self.cols_by_name[col.name] = col
            self.cols_by_id[col.id] = col

            # also add to stored md
            self._tbl_md.column_md[col.id] = schema.ColumnMd(
                id=col.id,
                col_type=col.col_type.as_dict(),
                is_pk=col.is_pk,
                schema_version_add=col.schema_version_add,
                schema_version_drop=col.schema_version_drop,
                value_expr=col.value_expr.as_dict() if col.value_expr is not None else None,
                stored=col.stored,
            )
            if col.name is not None:
                self._schema_version_md.columns[col.id] = schema.SchemaColumn(
                    name=col.name,
                    pos=len(self.cols_by_name),
                    media_validation=col._media_validation.name.lower() if col._media_validation is not None else None,
                )

            if col.is_stored:
                self.store_tbl.add_column(col)

            if not col.is_computed or not col.is_stored or row_count == 0:
                continue

            # populate the column
            from pixeltable.plan import Planner

            plan = Planner.create_add_column_plan(self.path, col)
            plan.ctx.num_rows = row_count
            try:
                plan.open()
                try:
                    excs_per_col = self.store_tbl.load_column(col, plan, on_error == 'abort')
                except sql.exc.DBAPIError as exc:
                    # Wrap the DBAPIError in an excs.Error to unify processing in the subsequent except block
                    raise excs.Error(f'SQL error during execution of computed column `{col.name}`:\n{exc}') from exc
                if excs_per_col > 0:
                    cols_with_excs.append(col)
                    num_excs += excs_per_col
                computed_values += plan.ctx.num_computed_exprs * row_count
            finally:
                # Ensure cleanup occurs if an exception or keyboard interruption happens during `load_column()`.
                def cleanup_on_error() -> None:
                    """Delete columns that are added as part of current add_columns operation and re-initialize
                    the sqlalchemy schema"""
                    self.cols = [col for col in self.cols if col not in cols_to_add]
                    for col in cols_to_add:
                        # remove columns that we already added
                        if col.id in self.cols_by_id:
                            del self.cols_by_id[col.id]
                        if col.name is not None and col.name in self.cols_by_name:
                            del self.cols_by_name[col.name]
                    self.store_tbl.create_sa_tbl()

                # Run cleanup only if there has been an exception; otherwise, skip cleanup.
                run_cleanup_on_exception(cleanup_on_error)
                plan.close()

        pxt.catalog.Catalog.get().record_column_dependencies(self)

        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)

        # TODO: what to do about system columns with exceptions?
        row_counts = RowCountStats(
            upd_rows=row_count, num_excs=num_excs, computed_values=computed_values
        )  # add_columns
        return UpdateStatus(
            cols_with_excs=[f'{col.tbl.name}.{col.name}' for col in cols_with_excs if col.name is not None],
            row_count_stats=row_counts,
        )

    def drop_column(self, col: Column) -> None:
        """Drop a column from the table."""

        assert not self.is_snapshot

        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version

        # drop this column and all dependent index columns and indices
        dropped_cols = [col]
        dropped_idx_names: list[str] = []
        for idx_info in self.idxs_by_name.values():
            if idx_info.col != col:
                continue
            dropped_cols.extend([idx_info.val_col, idx_info.undo_col])
            idx_md = self._tbl_md.index_md[idx_info.id]
            idx_md.schema_version_drop = self.schema_version
            assert idx_md.name in self.idxs_by_name
            dropped_idx_names.append(idx_md.name)

        # update idxs_by_name
        for idx_name in dropped_idx_names:
            del self.idxs_by_name[idx_name]

        self._drop_columns(dropped_cols)
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)
        _logger.info(f'Dropped column {col.name} from table {self.name}, new version: {self.version}')

    def _drop_columns(self, cols: Iterable[Column]) -> None:
        """Mark columns as dropped"""
        assert not self.is_snapshot

        for col in cols:
            col.schema_version_drop = self.schema_version
            if col.name is not None:
                assert col.name in self.cols_by_name
                del self.cols_by_name[col.name]
            assert col.id in self.cols_by_id
            del self.cols_by_id[col.id]
            # update stored md
            self._tbl_md.column_md[col.id].schema_version_drop = col.schema_version_drop
            if col.name is not None:
                del self._schema_version_md.columns[col.id]

        # update positions
        for pos, schema_col in enumerate(self._schema_version_md.columns.values()):
            schema_col.pos = pos

        self.store_tbl.create_sa_tbl()
        pxt.catalog.Catalog.get().record_column_dependencies(self)

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
        self._schema_version_md.columns[col.id].name = new_name

        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def set_comment(self, new_comment: Optional[str]) -> None:
        _logger.info(f'[{self.name}] Updating comment: {new_comment}')
        self.comment = new_comment
        self._create_schema_version()

    def set_num_retained_versions(self, new_num_retained_versions: int) -> None:
        _logger.info(
            f'[{self.name}] Updating num_retained_versions: {new_num_retained_versions} '
            f'(was {self.num_retained_versions})'
        )
        self.num_retained_versions = new_num_retained_versions
        self._create_schema_version()

    def _create_schema_version(self) -> None:
        # we're creating a new schema version
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)
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

        assert self.is_insertable
        assert (rows is None) != (df is None)  # Exactly one must be specified
        if rows is not None:
            plan = Planner.create_insert_plan(self, rows, ignore_errors=not fail_on_exception)

        else:
            plan = Planner.create_df_insert_plan(self, df, ignore_errors=not fail_on_exception)

        # this is a base table; we generate rowids during the insert
        def rowids() -> Iterator[int]:
            while True:
                rowid = self.next_row_id
                self.next_row_id += 1
                yield rowid

        result = self._insert(
            plan, time.time(), print_stats=print_stats, rowids=rowids(), abort_on_exc=fail_on_exception
        )
        return result

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
        cols_with_excs, row_counts = self.store_tbl.insert_rows(
            exec_plan, v_min=self.version, rowids=rowids, abort_on_exc=abort_on_exc
        )
        result = UpdateStatus(
            cols_with_excs=[f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs],
            row_count_stats=row_counts,
        )

        # update views
        for view in self.mutable_views:
            from pixeltable.plan import Planner

            plan2, _ = Planner.create_view_load_plan(view.get().path, propagates_insert=True)
            status = view.get()._insert(plan2, timestamp, print_stats=print_stats)
            result += status.to_cascade()

        # Use the net status after all propagations
        self._write_md(new_version=True, new_version_ts=timestamp, new_schema_version=False, update_status=result)
        if print_stats:
            exec_plan.ctx.profile.print(num_rows=result.num_rows)
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

        update_spec = self._validate_update_spec(value_spec, allow_pk=False, allow_exprs=True, allow_media=True)
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
        result += UpdateStatus(updated_cols=updated_cols)
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

        from pixeltable.plan import Planner

        plan, row_update_node, delete_where_clause, updated_cols, recomputed_cols = Planner.create_batch_update_plan(
            self.path, batch, rowids, cascade=cascade
        )
        result = self.propagate_update(
            plan, delete_where_clause, recomputed_cols, base_versions=[], timestamp=time.time(), cascade=cascade
        )
        result += UpdateStatus(updated_cols=[c.qualified_name for c in updated_cols])

        unmatched_rows = row_update_node.unmatched_rows()
        if len(unmatched_rows) > 0:
            if error_if_not_exists:
                raise excs.Error(f'batch_update(): {len(unmatched_rows)} row(s) not found')
            if insert_if_not_exists:
                insert_status = self.insert(unmatched_rows, None, print_stats=False, fail_on_exception=False)
                result += insert_status.to_cascade()
        return result

    def _validate_update_spec(
        self, value_spec: dict[str, Any], allow_pk: bool, allow_exprs: bool, allow_media: bool
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
            if col.col_type.is_media_type() and not allow_media:
                raise excs.Error(f'Column {col_name} is a media column and cannot be updated')

            # make sure that the value is compatible with the column type
            value_expr: exprs.Expr
            try:
                # check if this is a literal
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except (TypeError, jsonschema.exceptions.ValidationError) as exc:
                if not allow_exprs:
                    raise excs.Error(
                        f'Column {col_name}: value {val!r} is not a valid literal for this column '
                        f'(expected {col.col_type})'
                    ) from exc
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.Error(
                        f'Column {col_name}: value {val!r} is not a recognized literal or expression'
                    ) from exc
                if not col.col_type.is_supertype_of(value_expr.col_type, ignore_nullable=True):
                    raise excs.Error(
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ) from exc
            update_targets[col] = value_expr

        return update_targets

    def recompute_columns(self, col_names: list[str], errors_only: bool = False, cascade: bool = True) -> UpdateStatus:
        assert not self.is_snapshot
        assert all(name in self.cols_by_name for name in col_names)
        assert len(col_names) > 0
        assert len(col_names) == 1 or not errors_only

        from pixeltable.plan import Planner

        target_columns = [self.cols_by_name[name] for name in col_names]
        where_clause: Optional[exprs.Expr] = None
        if errors_only:
            where_clause = (
                exprs.ColumnPropertyRef(exprs.ColumnRef(target_columns[0]), exprs.ColumnPropertyRef.Property.ERRORTYPE)
                != None
            )
        plan, updated_cols, recomputed_cols = Planner.create_update_plan(
            self.path, update_targets={}, recompute_targets=target_columns, where_clause=where_clause, cascade=cascade
        )
        from pixeltable.exprs import SqlElementCache

        result = self.propagate_update(
            plan,
            where_clause.sql_expr(SqlElementCache()) if where_clause is not None else None,
            recomputed_cols,
            base_versions=[],
            timestamp=time.time(),
            cascade=cascade,
            show_progress=True,
        )
        result += UpdateStatus(updated_cols=updated_cols)
        return result

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
        create_new_table_version = plan is not None
        if create_new_table_version:
            self.version += 1
            cols_with_excs, row_counts = self.store_tbl.insert_rows(
                plan, v_min=self.version, show_progress=show_progress
            )
            result += UpdateStatus(
                row_count_stats=row_counts.insert_to_update(),
                cols_with_excs=[f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs],
            )
            self.store_tbl.delete_rows(
                self.version, base_versions=base_versions, match_on_vmin=True, where_clause=where_clause
            )

        if cascade:
            base_versions = [None if plan is None else self.version, *base_versions]  # don't update in place
            # propagate to views
            for view in self.mutable_views:
                recomputed_cols = [col for col in recomputed_view_cols if col.tbl.id == view.id]
                plan = None
                if len(recomputed_cols) > 0:
                    from pixeltable.plan import Planner

                    plan = Planner.create_view_update_plan(view.get().path, recompute_targets=recomputed_cols)
                status = view.get().propagate_update(
                    plan, None, recomputed_view_cols, base_versions=base_versions, timestamp=timestamp, cascade=True
                )
                result += status.to_cascade()
        if create_new_table_version:
            self._write_md(new_version=True, new_version_ts=timestamp, new_schema_version=False, update_status=result)
        return result

    def delete(self, where: Optional[exprs.Expr] = None) -> UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a predicate to filter rows to delete.

        Returns:
            UpdateStatus: an object containing the number of deleted rows and other statistics.
        """
        assert self.is_insertable
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

        status = self.propagate_delete(sql_where_clause, base_versions=[], timestamp=time.time())
        return status

    def propagate_delete(
        self, where: Optional[exprs.Expr], base_versions: list[Optional[int]], timestamp: float
    ) -> UpdateStatus:
        """Delete rows in this table and propagate to views.
        Args:
            where: a predicate to filter rows to delete.
        Returns:
            number of deleted rows
        """
        # print(f'calling sql_expr()')
        sql_where_clause = where.sql_expr(exprs.SqlElementCache()) if where is not None else None
        # #print(f'sql_where_clause={str(sql_where_clause) if sql_where_clause is not None else None}')
        # sql_cols: list[sql.Column] = []
        # def collect_cols(col) -> None:
        #     sql_cols.append(col)
        # sql.sql.visitors.traverse(sql_where_clause, {}, {'column': collect_cols})
        # x = [f'{str(c)}:{hash(c)}:{id(c.table)}' for c in sql_cols]
        # print(f'where_clause cols: {x}')
        del_rows = self.store_tbl.delete_rows(
            self.version + 1, base_versions=base_versions, match_on_vmin=False, where_clause=sql_where_clause
        )
        row_counts = RowCountStats(del_rows=del_rows)  # delete
        result = UpdateStatus(row_count_stats=row_counts)
        if del_rows > 0:
            # we're creating a new version
            self.version += 1
        for view in self.mutable_views:
            status = view.get().propagate_delete(
                where=None, base_versions=[self.version, *base_versions], timestamp=timestamp
            )
            result += status.to_cascade()
        if del_rows > 0:
            self._write_md(new_version=True, new_version_ts=timestamp, new_schema_version=False, update_status=result)
        return result

    def revert(self) -> None:
        """Reverts the table to the previous version."""
        assert not self.is_snapshot
        if self.version == 0:
            raise excs.Error('Cannot revert version 0')
        self._revert()

    def _revert(self) -> None:
        """
        Reverts the stored metadata for this table version and propagates to views.

        Doesn't attempt to revert the in-memory metadata, but instead invalidates this TableVersion instance
        and relies on Catalog to reload it
        """
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

        # revert schema changes:
        # - undo changes to self._tbl_md and write that back
        # - delete newly-added TableVersion/TableSchemaVersion records
        if self.version == self.schema_version:
            # physically delete newly-added columns and remove them from the stored md
            added_cols = [col for col in self.cols if col.schema_version_add == self.schema_version]
            if len(added_cols) > 0:
                self._tbl_md.next_col_id = min(col.id for col in added_cols)
                for col in added_cols:
                    if col.is_stored:
                        self.store_tbl.drop_column(col)
                    del self._tbl_md.column_md[col.id]

            # remove newly-added indices from the lookup structures
            # (the value and undo columns got removed in the preceding step)
            added_idx_md = [md for md in self._tbl_md.index_md.values() if md.schema_version_add == self.schema_version]
            if len(added_idx_md) > 0:
                self._tbl_md.next_idx_id = min(md.id for md in added_idx_md)
                for md in added_idx_md:
                    # TODO: drop the index
                    del self._tbl_md.index_md[md.id]

            # make newly-dropped columns visible again
            dropped_col_md = [
                md for md in self._tbl_md.column_md.values() if md.schema_version_drop == self.schema_version
            ]
            for col_md in dropped_col_md:
                col_md.schema_version_drop = None

            # make newly-dropped indices visible again
            dropped_idx_md = [
                md for md in self._tbl_md.index_md.values() if md.schema_version_drop == self.schema_version
            ]
            for idx_md in dropped_idx_md:
                idx_md.schema_version_drop = None

            conn.execute(
                sql.delete(schema.TableSchemaVersion.__table__)
                .where(schema.TableSchemaVersion.tbl_id == self.id)
                .where(schema.TableSchemaVersion.schema_version == self.schema_version)
            )
            self._tbl_md.current_schema_version = self._schema_version_md.preceding_schema_version

        conn.execute(
            sql.delete(schema.TableVersion.__table__)
            .where(schema.TableVersion.tbl_id == self.id)
            .where(schema.TableVersion.version == self.version)
        )

        self.version -= 1
        self._write_md(new_version=False, new_version_ts=0, new_schema_version=False)

        # propagate to views
        views_str = ', '.join([str(v.id) for v in self.mutable_views])
        print(f'revert(): mutable_views={views_str}')
        for view in self.mutable_views:
            view.get()._revert()

        # force reload on next operation
        self.is_validated = False
        pxt.catalog.Catalog.get().remove_tbl_version(self)
        _logger.info(f'TableVersion {self.name}: reverted to version {self.version}')

    def _init_external_stores(self) -> None:
        for store_md in self.tbl_md.external_stores:
            store_cls = resolve_symbol(store_md['class'])
            assert isinstance(store_cls, type) and issubclass(store_cls, pxt.io.ExternalStore)
            store = store_cls.from_dict(store_md['md'])
            self.external_stores[store.name] = store

    def link_external_store(self, store: pxt.io.ExternalStore) -> None:
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version

        self.external_stores[store.name] = store
        self._tbl_md.external_stores.append(
            {'class': f'{type(store).__module__}.{type(store).__qualname__}', 'md': store.as_dict()}
        )
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)

    def unlink_external_store(self, store: pxt.io.ExternalStore) -> None:
        del self.external_stores[store.name]
        self.version += 1
        self.preceding_schema_version = self.schema_version
        self.schema_version = self.version
        idx = next(i for i, store_md in enumerate(self._tbl_md.external_stores) if store_md['md']['name'] == store.name)
        self._tbl_md.external_stores.pop(idx)
        self._write_md(new_version=True, new_version_ts=time.time(), new_schema_version=True)

    @property
    def tbl_md(self) -> schema.TableMd:
        return self._tbl_md

    @property
    def schema_version_md(self) -> schema.TableSchemaVersionMd:
        return self._schema_version_md

    @property
    def view_md(self) -> Optional[schema.ViewMd]:
        return self._tbl_md.view_md

    @property
    def name(self) -> str:
        return self._tbl_md.name

    @property
    def user(self) -> Optional[str]:
        return self._tbl_md.user

    @property
    def is_replica(self) -> bool:
        return self._tbl_md.is_replica

    @property
    def comment(self) -> str:
        return self._schema_version_md.comment

    @comment.setter
    def comment(self, c: str) -> None:
        assert self.effective_version is None
        self._schema_version_md.comment = c

    @property
    def num_retained_versions(self) -> int:
        return self._schema_version_md.num_retained_versions

    @num_retained_versions.setter
    def num_retained_versions(self, n: int) -> None:
        assert self.effective_version is None
        self._schema_version_md.num_retained_versions = n

    @property
    def version(self) -> int:
        # if this is a snapshot instance, we need to ignore current_version
        return self._tbl_md.current_version if self.effective_version is None else self.effective_version

    @version.setter
    def version(self, version: int) -> None:
        assert self.effective_version is None
        self._tbl_md.current_version = version

    @property
    def schema_version(self) -> int:
        return self._schema_version_md.schema_version

    @schema_version.setter
    def schema_version(self, version: int) -> None:
        assert self.effective_version is None
        self._tbl_md.current_schema_version = version
        self._schema_version_md.schema_version = version

    @property
    def preceding_schema_version(self) -> int:
        return self._schema_version_md.preceding_schema_version

    @preceding_schema_version.setter
    def preceding_schema_version(self, v: int) -> None:
        assert self.effective_version is None
        self._schema_version_md.preceding_schema_version = v

    @property
    def media_validation(self) -> MediaValidation:
        return MediaValidation[self._schema_version_md.media_validation.upper()]

    @property
    def next_col_id(self) -> int:
        return self._tbl_md.next_col_id

    @next_col_id.setter
    def next_col_id(self, id: int) -> None:
        assert self.effective_version is None
        self._tbl_md.next_col_id = id

    @property
    def next_idx_id(self) -> int:
        return self._tbl_md.next_idx_id

    @next_idx_id.setter
    def next_idx_id(self, id: int) -> None:
        assert self.effective_version is None
        self._tbl_md.next_idx_id = id

    @property
    def next_row_id(self) -> int:
        return self._tbl_md.next_row_id

    @next_row_id.setter
    def next_row_id(self, id: int) -> None:
        assert self.effective_version is None
        self._tbl_md.next_row_id = id

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    @property
    def is_mutable(self) -> bool:
        return not self.is_snapshot and not self.is_replica

    @property
    def is_view(self) -> bool:
        return self.view_md is not None

    @property
    def include_base_columns(self) -> bool:
        return self.view_md is not None and self.view_md.include_base_columns

    @property
    def is_component_view(self) -> bool:
        return self.iterator_cls is not None

    @property
    def is_insertable(self) -> bool:
        """Returns True if this corresponds to an InsertableTable"""
        return not self.is_snapshot and not self.is_view

    def is_iterator_column(self, col: Column) -> bool:
        """Returns True if col is produced by an iterator"""
        # the iterator columns directly follow the pos column
        return self.is_component_view and col.id > 0 and col.id < self.num_iterator_cols + 1

    def is_system_column(self, col: Column) -> bool:
        """Return True if column was created by Pixeltable"""
        return col.name == _POS_COLUMN_NAME and self.is_component_view

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

    def get_idx_val_columns(self, cols: Iterable[Column]) -> set[Column]:
        result = {info.val_col for col in cols for info in col.get_idx_info().values()}
        return result

    def get_dependent_columns(self, cols: Iterable[Column]) -> set[Column]:
        """
        Return the set of columns that transitively depend on any of the given ones.
        """
        cat = pxt.catalog.Catalog.get()
        result = set().union(*[cat.get_column_dependents(col.tbl.id, col.id) for col in cols])
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
            assert col.is_pk is not None
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

    def _create_version_md(self, timestamp: float, update_status: Optional[UpdateStatus]) -> schema.TableVersionMd:
        return schema.TableVersionMd(
            tbl_id=str(self.id),
            created_at=timestamp,
            version=self.version,
            schema_version=self.schema_version,
            user=Env.get().user,
            update_status=update_status,
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
        from pixeltable import catalog

        id = UUID(d['id'])
        effective_version = d['effective_version']
        return catalog.Catalog.get().get_tbl_version(id, effective_version)
