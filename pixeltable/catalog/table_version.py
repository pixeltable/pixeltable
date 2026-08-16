from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, cast
from uuid import UUID

import sqlalchemy as sql
from sqlalchemy import exc as sql_exc

import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.type_system as ts
from pixeltable import telemetry
from pixeltable.env import Env
from pixeltable.exprs.inline_expr import InlineDict
from pixeltable.func.iterator import GeneratingFunctionCall
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.utils.object_stores import ObjectOps

from .column import Column
from .globals import _ROWID_COLUMN_NAME, IndexSpec, MediaValidation, QColumnId, TableVersionMd, is_valid_identifier
from .tbl_ops import (
    CreateColumnMdOp,
    CreateStoreColumnsOp,
    CreateStoreIdxsOp,
    CreateTableVersionOp,
    DeleteTableMdOp,
    DeleteTableMediaFilesOp,
    DropStoreTableOp,
    TableOp,
    TableOpsBuilder,
)
from .update_status import RowCountStats, UpdateStatus

if TYPE_CHECKING:
    from pixeltable import exec, store
    from pixeltable._query import Query
    from pixeltable.catalog.table_version_handle import TableVersionHandle
    from pixeltable.io.data_sources import SqlDataSource
    from pixeltable.plan import SampleClause

    from .table_path import TableVersionPath

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class TableVersionKey:
    tbl_id: UUID
    effective_version: int | None

    # Allow unpacking as a tuple
    def __iter__(self) -> Iterator[Any]:
        return iter((self.tbl_id, self.effective_version))

    def as_dict(self) -> dict:
        return {'id': str(self.tbl_id), 'effective_version': self.effective_version}

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionKey:
        tbl_id = UUID(d['id'])
        effective_version = d['effective_version']
        return cls(tbl_id, effective_version)


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

    Effective version distinguishes between live tables and snapshots:
    - None for a live table (that can receive updates)
    - VersionMd.version for a snapshot

    Instances of TableVersion should not be stored as member variables (ie, used across transaction boundaries).
    Use a TableVersionHandle instead.

    Only TableVersion and Catalog interact directly with stored metadata. Everything else needs to go through these
    two classes.
    """

    key: TableVersionKey

    # record metadata stored in catalog
    _tbl_md: schema.TableMd
    _version_md: schema.VersionMd
    # User and system columns that are visible in this schema version.
    _schema_version_md: schema.SchemaVersionMd

    path: 'TableVersionPath' | None  # only set for non-snapshots; needed to resolve computed cols
    base: TableVersionHandle | None  # only set for views
    predicate: exprs.Expr | None
    sample_clause: 'SampleClause' | None

    iterator_call: GeneratingFunctionCall | None
    num_iterator_cols: int

    # target for data operation propagation (only set for non-snapshots, and only records non-snapshot views)
    mutable_views: frozenset[TableVersionHandle]

    # User and system columns that are live in this schema version, ordered by column id (asc). The order matters for
    # iterator_columns().
    cols_by_id: dict[int, Column]
    # contains only user-facing (named) columns visible in this version
    cols_by_name: dict[str, Column]

    # True if this TableVersion instance can have indices:
    # - live version of a mutable table
    supports_idxs: bool

    # only populated with indices visible in this TableVersion instance
    idxs: dict[int, TableVersion.IndexInfo]  # key: index id
    idxs_by_name: dict[str, TableVersion.IndexInfo]
    idxs_by_col: dict[QColumnId, list[TableVersion.IndexInfo]]

    store_tbl: 'store.StoreBase' | None

    is_initialized: bool  # True if init() has been called

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
        # Three valid combinations of value and undo columns:
        # 1. val_col and undo_col are both None: the index is backed by a store index created directly on the
        #    indexed column. Example: any B-tree index on an operational table.
        # 2. val_col is present, undo_col is None: no row is ever soft-deleted, so there is no index value to
        #    preserve. Example: an embedding index on an operational table.
        # 3. val_col and undo_col are both present: undo_col preserves the index value of an expired row.
        #    Example: any index on a data-versioned table.
        val_col: Column | None
        undo_col: Column | None

        @property
        def columns(self) -> list[Column]:
            """This index's value and undo columns, if it has them."""
            return [c for c in (self.val_col, self.undo_col) if c is not None]

        @property
        def indexed_sa_col(self) -> sql.Column:
            """The store column that the store index is created on."""
            if self.idx.uses_value_col:
                return self.val_col.sa_col
            return self.col.sa_col

    def __init__(
        self,
        key: TableVersionKey,
        tbl_md: schema.TableMd,
        version_md: schema.VersionMd,
        schema_version_md: schema.SchemaVersionMd,
        mutable_views: list[TableVersionHandle],
        base_path: 'TableVersionPath' | None = None,
        base: TableVersionHandle | None = None,
    ):
        self.is_validated = True  # a freshly constructed instance is always valid
        self.is_initialized = False
        self.key = key
        self._tbl_md = copy.deepcopy(tbl_md)
        self._version_md = copy.deepcopy(version_md)
        self._schema_version_md = copy.deepcopy(schema_version_md)
        assert self.is_data_versioned or self._version_md.version == self._schema_version_md.schema_version, (
            self._version_md,
            self._schema_version_md,
        )
        assert not (self.is_view and base is None)
        self.base = base
        self.store_tbl = None

        # mutable tables need their TableVersionPath for expr eval during updates
        from .table_path import TableVersionPath
        from .table_version_handle import TableVersionHandle

        if self.is_snapshot:
            self.path = None
        else:
            self_handle = TableVersionHandle(key)
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
        self.iterator_call = None
        self.num_iterator_cols = 0
        if self.view_md is not None and self.view_md.iterator_call is not None:
            self.iterator_call = GeneratingFunctionCall.from_dict(self.view_md.iterator_call)
            # outputs is None: this catches views created before iterator outputs were persisted;
            # for those, num_iterator_cols stays 0, which keeps the view loadable and its stored data readable
            if self.iterator_call.outputs is not None:
                self.num_iterator_cols = len(self.iterator_call.outputs)

        self.mutable_views = frozenset(mutable_views)
        assert self.is_mutable or len(self.mutable_views) == 0

        self.cols_by_name = {}
        self.cols_by_id = {}
        self.idxs = {}
        self.idxs_by_name = {}
        self.idxs_by_col = {}
        self.supports_idxs = self.effective_version is None

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def versioned_name(self) -> str:
        if self.effective_version is None:
            return self.name
        else:
            return f'{self.name}:{self.effective_version}'

    def __repr__(self) -> str:
        version_info = ''
        if self.is_data_versioned:
            version_info = f', effective_version={self.effective_version}, version={self.version}'
        return (
            f'TableVersion(id={self.id!r}, name={self.name!r}, '
            f'is_data_versioned={self.is_data_versioned}{version_info})'
        )

    @property
    def handle(self) -> 'TableVersionHandle':
        from .table_version_handle import TableVersionHandle

        return TableVersionHandle(self.key)

    @classmethod
    def create_initial_md(
        cls,
        tbl_id: UUID,
        name: str,
        cols: list[Column],
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        has_default_idxs: bool,
        view_md: schema.ViewMd | None,
        is_data_versioned: bool,
        additional_idxs: list[IndexSpec],
    ) -> TableVersionMd:
        from .table_version_handle import TableVersionHandle

        user = Env.get().user
        timestamp = time.time()

        tbl_id_str = str(tbl_id)
        tbl_handle = TableVersionHandle(TableVersionKey(tbl_id, None))
        column_ids = itertools.count()
        index_ids = itertools.count()

        # assign ids
        for col in cols:
            col.tbl_handle = tbl_handle
            col.id = next(column_ids)
            col.schema_version_add = 0

        # resolve ColumnRefByName's to ColumnRefs in computed columns
        subst = exprs.ExprDict[exprs.Expr](
            (
                exprs.ColumnRefByName(col.name),
                exprs.ColumnRef(
                    col.column_version_md(),
                    perform_validation=(col._media_validation or media_validation) == MediaValidation.ON_READ,
                ),
            )
            for col in cols
            if col.name is not None
        )

        # create metadata
        column_md: dict[int, schema.ColumnMd] = {}
        schema_col_md: dict[int, schema.SchemaColumn] = {}
        for pos, col in enumerate(cols):
            value_expr = col.value_expr
            if value_expr is not None:
                col.set_value_expr(value_expr.substitute(subst))
            if col.is_computed:
                col.check_value_expr()
            col_md, col_schema_md = col.to_md(pos)
            column_md[col.id] = col_md
            schema_col_md[col.id] = col_schema_md

        cls._validate_idxs(tbl_id, additional_idxs, has_default_idxs)

        # Merge default indexes and additional indexes into a manifest of indexes to create.
        index_md: dict[int, schema.IndexMd] = {}
        idxs_to_create: list[IndexSpec] = []
        if has_default_idxs and (view_md is None or not view_md.is_snapshot):
            idxs_to_create.extend(
                IndexSpec(col, None, index.BtreeIndex(uses_value_col=is_data_versioned))
                for col in cols
                if index.BtreeIndex.can_index(col)
            )

        # an index on a column of this table must reference the instance in cols, which is the one that got an id
        # above; an index on a base column references that column directly
        own_cols = {id(col) for col in cols}
        assert all(
            id(spec.indexed_column) in own_cols
            for spec in additional_idxs
            if cast(Column, spec.indexed_column).tbl_handle.id == tbl_id
        )
        idxs_to_create.extend(additional_idxs)

        taken_idx_names = {spec.idx_name for spec in idxs_to_create if spec.idx_name is not None}

        index_cols: list[Column] = []
        for idx_col, idx_name, idx in idxs_to_create:
            assert isinstance(idx_col, Column)
            val_col, undo_col = Column.create_index_columns(
                tbl_handle,
                idx_col,
                idx,
                schema_version=0,
                is_data_versioned=is_data_versioned,
                next_col_id=lambda: next(column_ids),
            )
            index_cols.extend(c for c in (val_col, undo_col) if c is not None)

            idx_id = next(index_ids)
            resolved_idx_name: str
            if idx_name is not None:
                resolved_idx_name = idx_name
            else:
                resolved_idx_name = cls._generate_idx_name(taken_idx_names)
                taken_idx_names.add(resolved_idx_name)
            idx_cls = type(idx)
            md = schema.IndexMd(
                id=idx_id,
                name=resolved_idx_name,
                indexed_col_id=idx_col.id,
                indexed_col_tbl_id=str(idx_col.tbl_handle.id),
                index_val_col_id=None if val_col is None else val_col.id,
                index_val_undo_col_id=None if undo_col is None else undo_col.id,
                schema_version_add=0,
                schema_version_drop=None,
                class_fqn=idx_cls.__module__ + '.' + idx_cls.__name__,
                init_args=idx.as_dict(),
            )
            index_md[idx_id] = md

        for col in index_cols:
            col_md, col_schema_md = col.to_md(pos=None)
            column_md[col.id] = col_md
            schema_col_md[col.id] = col_schema_md

        assert all(column_md[col_id].id == col_id for col_id in column_md)
        assert all(index_md[idx_id].id == idx_id for idx_id in index_md)

        tbl_md = schema.TableMd(
            tbl_id=tbl_id_str,
            name=name,
            user=user,
            current_version=0,
            current_schema_version=0,
            next_col_id=next(column_ids),
            next_idx_id=next(index_ids),
            next_row_id=0,
            view_sn=0,
            column_md=column_md,
            index_md=index_md,
            view_md=view_md,
            additional_md={},
            is_data_versioned=is_data_versioned,
            has_default_idxs=has_default_idxs,
        )

        table_version_md = schema.VersionMd(
            tbl_id=tbl_id_str,
            created_at=timestamp,
            version=0,
            schema_version=0,
            user=user,
            update_status=None,
            additional_md={},
        )

        schema_version_md = schema.SchemaVersionMd(
            tbl_id=tbl_id_str,
            schema_version=0,
            preceding_schema_version=None,
            columns=schema_col_md,
            comment=comment,
            custom_metadata=custom_metadata,
            media_validation=media_validation.name.lower(),
            additional_md={},
        )
        return TableVersionMd(tbl_md, table_version_md, schema_version_md)

    def delete_media(self, tbl_version: int | None = None) -> None:
        # Assemble a set of column destinations and delete objects from all of them
        # None is a valid column destination which refers to the default object location
        destinations = {col.destination for col in self.cols_by_id.values() if col.is_stored}
        for dest in destinations:
            ObjectOps.delete(dest, self.id, tbl_version=tbl_version)

    def drop_ops(self) -> tuple[list[TableOp], bool]:
        """Returns a tuple of drop table ops, and a boolean that indicates whether a new table and schema
        versions were created."""
        new_version = self.is_mutable and self.is_data_versioned
        if new_version:
            self.bump_version(bump_schema_version=True)
        mutable_base_tbl_id: str | None = None
        # if this is a mutable view of a mutable base, advance the base's view_sn in the end
        if self.is_view and self.is_mutable and self.path.base.is_mutable():
            mutable_base_tbl_id = str(self.path.base.tbl_id)
        id_str = str(self.id)
        ops = (
            TableOpsBuilder(id_str, tbl_version=self._tbl_md.current_version)
            .add(DeleteTableMediaFilesOp)
            .add(DropStoreTableOp, is_view=self.is_view)
            .add(DeleteTableMdOp, mutable_base_tbl_id=mutable_base_tbl_id)
            .build()
        )
        return ops, new_version

    def init(self) -> None:
        """
        Initialize schema-related in-memory metadata separately, now that this TableVersion instance is visible
        in Catalog.
        """
        cat = get_runtime().catalog
        assert self.key in cat._tbl_versions
        self._init_schema()
        if self.is_mutable:
            cat.record_column_dependencies(self)

        self.is_initialized = True

    def _init_schema(self) -> None:
        from pixeltable.store import StoreComponentView, StoreTable, StoreView

        assert len(self.cols_by_id) == 0
        assert len(self.idxs) == 0

        # Construct a target table versions dict for column retargeting, if needed
        tvp: TableVersionPath | None = None
        if self.effective_version is not None:
            # for snapshot TableVersion instances, we need to retarget the column value_exprs to the snapshot;
            # otherwise they'll incorrectly refer to the live table. So, construct a full TableVersionPath to
            # use for retargeting.
            tvp = get_runtime().catalog.construct_tvp(
                self.id, self.effective_version, self.tbl_md.ancestors, self.version_md.created_at
            )
        target_tbl_versions = {tvh.id: tvh.get() for tvh in tvp.get_tbl_versions()} if tvp is not None else None

        # Reconstruct Column and Index objects from metadata, populating all internal lookup structures.
        # Indexes are initialized in lock-step, immediately after the last column they reference is initialized.
        idxs_by_last_col_id = self._build_idxs_by_last_col_id()
        # Indexes that do not depend on any columns of this table can be initialized right away
        self._init_idxs(idxs_by_last_col_id.get(None, []))

        # Sort columns in column_md by the position specified in col_md.id to guarantee that all references
        # point backward.
        sorted_column_md = sorted(self.tbl_md.column_md.values(), key=lambda item: item.id)
        for col_md in sorted_column_md:
            if not col_md.is_visible_in_version(self.schema_version):
                continue
            schema_col_md = self._schema_version_md.columns[col_md.id]
            col_type = ts.ColumnType.from_dict(schema_col_md.col_type)
            media_val = (
                MediaValidation[schema_col_md.media_validation.upper()]
                if schema_col_md.media_validation is not None
                else None
            )

            is_iterator_col = self.is_component_view and col_md.id < self.num_iterator_cols
            value_expr = self._init_col_value_expr_from_md(schema_col_md, target_tbl_versions)

            col = Column(
                col_id=col_md.id,
                name=schema_col_md.name,
                col_type=col_type,
                is_pk=schema_col_md.is_pk,
                is_iterator_col=is_iterator_col,
                stored=col_md.stored,
                media_validation=media_val,
                sa_col_type=ts.sa_type_from_dict(col_md.sa_col_type) if col_md.sa_col_type is not None else None,
                schema_version_add=col_md.schema_version_add,
                schema_version_drop=col_md.schema_version_drop,
                stores_cellmd=col_md.stores_cellmd,
                computed_with=value_expr,
                value_expr_dict=schema_col_md.value_expr,
                tbl_handle=self.handle,
                destination=col_md.destination,
                custom_metadata=schema_col_md.custom_metadata if schema_col_md is not None else None,
                comment=schema_col_md.comment,
            )

            self.cols_by_id[col.id] = col
            if not col.is_system_col:
                self.cols_by_name[col.name] = col

            # Initialize the indexes for which this is the last column they reference. All columns required for these
            # indexes have now been initialized. These indexes cannot be initialized later because some of the upcoming
            # columns can depend on them.
            self._init_idxs(idxs_by_last_col_id.get(col.id, []))

        # create the sqlalchemy schema, after instantiating all Columns
        if self.is_component_view:
            self.store_tbl = StoreComponentView(self)
        elif self.is_view:
            self.store_tbl = StoreView(self)
        else:
            self.store_tbl = StoreTable(self)

    def _init_col_value_expr_from_md(
        self, schema_col_md: schema.SchemaColumn, target_tbl_versions: dict[UUID, 'TableVersion'] | None
    ) -> exprs.Expr | None:
        if schema_col_md.value_expr is None:
            return None
        value_expr = exprs.Expr.from_dict(schema_col_md.value_expr, target_tbl_versions)
        value_expr.bind_rel_paths()
        if not value_expr.is_valid:
            col_name = schema_col_md.name if schema_col_md is not None else '<unnamed>'
            message = '\n'.join(
                [
                    f'The computed column {col_name!r} in table {self.name!r} is no longer valid.',
                    value_expr.validation_error,
                    'You can continue to query existing data from this column, but evaluating it on new data will raise an error.',  # noqa: E501
                ]
            )
            warnings.warn(message, category=excs.PixeltableWarning)  # noqa: B028
        return value_expr

    def _build_idxs_by_last_col_id(self) -> dict[int | None, list[tuple[index.IndexBase, schema.IndexMd]]]:
        """Group the indexes by the last column of this table that each one references.

        An index's last column is the highest id among the columns of this table it references: its value and undo
        columns, if it has them, plus the indexed column itself if that belongs to this table. Indexes that reference
        no column of this table (e.g. an index on a base column) are keyed by None."""
        if not self.supports_idxs:
            return {}

        idxs_by_last_col_id: dict[int | None, list[tuple[index.IndexBase, schema.IndexMd]]] = {}
        for md in self.tbl_md.index_md.values():
            cls_name = md.class_fqn.rsplit('.', 1)[-1]
            cls = getattr(index, cls_name)
            idx = cls.from_dict(md.init_args)
            col_ids = [col_id for col_id in (md.index_val_col_id, md.index_val_undo_col_id) if col_id is not None]
            if UUID(md.indexed_col_tbl_id) == self.id:
                col_ids.append(md.indexed_col_id)
            idxs_by_last_col_id.setdefault(max(col_ids, default=None), []).append((idx, md))

        return idxs_by_last_col_id

    def _init_idxs(self, idxs: list[tuple[index.IndexBase, schema.IndexMd]]) -> None:
        """Initialize those of idxs that are visible in the current schema version."""
        if len(idxs) == 0:
            return
        assert self.supports_idxs
        for idx, idx_md in idxs:
            if idx_md.is_visible_in_version(self.schema_version):
                self._init_idx(idx, idx_md)

    def _init_idx(self, idx: index.IndexBase, md: schema.IndexMd) -> None:
        indexed_col_id = QColumnId(UUID(md.indexed_col_tbl_id), md.indexed_col_id)
        idx_col = self.lookup_column(indexed_col_id)
        assert idx_col is not None
        assert (md.index_val_col_id is not None) == idx.uses_value_col, md
        assert md.index_val_undo_col_id is None or md.index_val_col_id is not None, md
        info = self.IndexInfo(
            id=md.id,
            name=md.name,
            idx=idx,
            col=idx_col,
            val_col=None if md.index_val_col_id is None else self.cols_by_id[md.index_val_col_id],
            undo_col=None if md.index_val_undo_col_id is None else self.cols_by_id[md.index_val_undo_col_id],
        )
        self.idxs[md.id] = info
        self.idxs_by_name[md.name] = info
        self.idxs_by_col.setdefault(indexed_col_id, []).append(info)

    def lookup_column(self, qid: QColumnId) -> Column | None:
        """
        Look up the column with the given table id and column id, searching through the ancestors of this TableVersion
        to find it. We avoid referencing TableVersionPath in order to work properly with snapshots as well.
        """
        if qid.tbl_id == self.id:
            return self.cols_by_id.get(qid.col_id)
        elif self.base is not None:
            return self.base.get().lookup_column(qid)
        else:
            return None

    def _write_md(self, new_version: bool, new_schema_version: bool) -> None:
        get_runtime().catalog.write_tbl_md(
            self.id,
            None,
            self._tbl_md,
            self._version_md if new_version else None,
            self._schema_version_md if new_schema_version else None,
        )

    def _store_idx_name(self, idx_id: int) -> str:
        """Return name of index in the store, which needs to be globally unique"""
        return f'idx_{self.id.hex}_{idx_id}'

    def add_index(self, col: Column, idx_name: str | None, idx: index.IndexBase) -> UpdateStatus:
        self._validate_idxs(self.id, [IndexSpec(col, idx_name, idx)], self.has_default_idxs, self.idxs.values())
        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)
        status = self._add_index(col, idx_name, idx)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Added index {idx_name} on column {col.name} to table {self.name}')
        return status

    @classmethod
    def _generate_idx_name(cls, taken_names: set[str]) -> str:
        """Generates an index name that is not in `taken_names`."""
        i = 0
        while True:
            name = f'idx{i}'
            if name not in taken_names:
                return name
            i += 1

    def _create_index_md(
        self, col: Column, val_col: Column | None, undo_col: Column | None, idx_name: str | None, idx: index.IndexBase
    ) -> int:
        """Create md for given index and update self._tbl_md. Returns index id."""
        existing_names = {i.name for i in self._tbl_md.index_md.values()}
        if idx_name is None:
            idx_name = self._generate_idx_name(existing_names)
        else:
            assert is_valid_identifier(idx_name)
            assert idx_name not in existing_names
        idx_id = self.next_idx_id
        self.next_idx_id += 1

        # create and register the index metadata
        idx_cls = type(idx)
        idx_md = schema.IndexMd(
            id=idx_id,
            name=idx_name,
            indexed_col_id=col.id,
            indexed_col_tbl_id=str(col.get_tbl().id),
            index_val_col_id=None if val_col is None else val_col.id,
            index_val_undo_col_id=None if undo_col is None else undo_col.id,
            schema_version_add=self.schema_version,
            schema_version_drop=None,
            class_fqn=idx_cls.__module__ + '.' + idx_cls.__name__,
            init_args=idx.as_dict(),
        )
        self._tbl_md.index_md[idx_id] = idx_md
        return idx_id

    def _create_index(
        self, col: Column, val_col: Column | None, undo_col: Column | None, idx_name: str | None, idx: index.IndexBase
    ) -> None:
        """Create the given index along with index md"""
        assert (val_col is not None) == idx.uses_value_col
        assert undo_col is None or val_col is not None
        idx_id = self._create_index_md(col, val_col, undo_col, idx_name, idx)
        idx_name = self.tbl_md.index_md[idx_id].name
        idx_info = self.IndexInfo(id=idx_id, name=idx_name, idx=idx, col=col, val_col=val_col, undo_col=undo_col)
        self.idxs[idx_id] = idx_info
        self.idxs_by_name[idx_name] = idx_info
        self.idxs_by_col.setdefault(col.qid, []).append(idx_info)
        self.store_tbl.create_index(idx_id)

    def _create_index_columns(self, col: Column, idx: index.IndexBase) -> tuple[Column | None, Column | None]:
        """Create the columns that idx needs in order to index col of this table."""
        return Column.create_index_columns(
            self.handle,
            col,
            idx,
            schema_version=self.schema_version,
            is_data_versioned=self.is_data_versioned,
            next_col_id=self.next_col_id,
        )

    def _add_index(self, col: Column, idx_name: str | None, idx: index.IndexBase) -> UpdateStatus:
        val_col, undo_col = self._create_index_columns(col, idx)
        status = UpdateStatus()
        new_cols = [c for c in (val_col, undo_col) if c is not None]
        if len(new_cols) > 0:
            # add the columns and update the metadata
            # TODO support on_error='abort' for indices; it's tricky because of the way metadata changes are entangled
            # with the database operations
            status = self._add_columns(new_cols, print_stats=False, on_error='ignore')
        # now create the index structure
        self._create_index(col, val_col, undo_col, idx_name, idx)
        return status

    @classmethod
    def _validate_idxs(
        cls,
        tbl_id: UUID,
        idxs: Iterable[IndexSpec],
        has_default_idxs: bool,
        existing_idxs: Iterable[TableVersion.IndexInfo] = (),
    ) -> None:
        """Validate the indexes in idxs, which are about to be created on the table with id tbl_id.

        idxs: resolved specs, ie. every indexed_column is a Column, not a column name.
        existing_idxs: the table's live indexes; a new index must not collide with one of those.
        """
        existing_by_name = {info.name: info for info in existing_idxs}
        # names of the columns that already have a B-tree index; a view's base columns are excluded, because the
        # validation below rejects them as targets anyway
        btree_col_names = {
            info.col.name
            for info in existing_idxs
            if isinstance(info.idx, index.BtreeIndex) and info.col.tbl_handle.id == tbl_id
        }
        new_names: set[str] = set()
        new_btree_col_names: set[str] = set()

        for idx_col, idx_name, idx in idxs:
            assert isinstance(idx_col, Column)
            if isinstance(idx, index.BtreeIndex):
                assert idx_col.name is not None, repr(idx_col)
                if has_default_idxs:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        'Cannot create an explicit B-tree index on a table with has_default_idxs=True; '
                        'its eligible columns are indexed automatically.',
                    )
                index.BtreeIndex.validate_column(idx_col)
                if idx_col.name in new_btree_col_names or idx_col.name in btree_col_names:
                    raise excs.AlreadyExistsError(
                        excs.ErrorCode.INDEX_ALREADY_EXISTS,
                        f'A B-tree index already exists on column {idx_col.name!r}.',
                    )
                new_btree_col_names.add(idx_col.name)
                if idx_col.tbl_handle.id != tbl_id:
                    # PXT-1260 Allow views to create a b-tree index on a base column
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Cannot create a B-tree index on column {idx_col.name!r}: it belongs to a base table. '
                        'Add the index to the base table instead.',
                    )
            if idx_name is not None:
                assert idx_name not in new_names, idx_name
                existing_info = existing_by_name.get(idx_name)
                if existing_info is not None:
                    raise excs.AlreadyExistsError(
                        excs.ErrorCode.INDEX_ALREADY_EXISTS,
                        f'Index {idx_name!r} already exists on column {existing_info.col.name!r}.',
                    )
                new_names.add(idx_name)

    def _validate_idx_drops(self, idx_ids: Iterable[int]) -> None:
        """Reject the removal of a default B-tree index."""
        if not self.has_default_idxs:
            return
        for idx_id in idx_ids:
            info = self.idxs[idx_id]
            if isinstance(info.idx, index.BtreeIndex):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot drop B-tree index {info.name!r} from a table with has_default_idxs=True',
                )

    def drop_index(self, idx_id: int) -> None:
        assert self.is_mutable
        assert idx_id in self._tbl_md.index_md
        self._validate_idx_drops([idx_id])

        idx_name = self._tbl_md.index_md[idx_id].name
        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)
        cols_to_drop = self._cascade_drop_index(idx_id)
        if len(cols_to_drop) > 0:
            self._drop_columns(cols_to_drop)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Dropped index {idx_name} on table {self.name}')

    def _cascade_drop_index(self, idx_id: int) -> list[Column]:
        """
        Mark an index as dropped, drop it from the store, and remove it from the in-memory and persistent lookup
        structures. Returns its value and undo columns.
        """
        assert idx_id in self._tbl_md.index_md
        idx_md = self._tbl_md.index_md[idx_id]
        idx_md.schema_version_drop = self.schema_version
        assert idx_md.name in self.idxs_by_name
        idx_info = self.idxs[idx_id]

        # Drop the physical index from the store
        self.store_tbl.drop_index(idx_id)

        # remove this index entry from the active indexes (in memory)
        # and the index metadata (in persistent table metadata)
        # TODO(PXT-1254): this is wrong, it breaks revert()
        del self.idxs[idx_id]
        del self.idxs_by_name[idx_md.name]
        if idx_info.col.qid in self.idxs_by_col:
            self.idxs_by_col[idx_info.col.qid].remove(idx_info)
        del self._tbl_md.index_md[idx_id]

        return idx_info.columns

    def add_columns_ops(self, cols: Iterable[Column]) -> tuple[TableVersionMd, list[TableOp]]:
        """Applies the column-addition metadata changes and builds the TableOps to execute them in the store."""
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        assert self.is_mutable
        assert all(is_valid_identifier(col.name) for col in cols if col.name is not None)
        assert all(col.stored is not None for col in cols)
        assert all(col.name not in self.cols_by_name for col in cols if col.name is not None)
        # row count computed on demand
        row_count: int | None = None
        for col in cols:
            if col.is_pk:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot add primary key column {col.name!r} after table creation',
                )
            # TODO: check this elsewhere?
            if not col.col_type.nullable and not col.is_computed:
                if row_count is None:
                    row_count = self.store_tbl.count()
                if row_count > 0:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Cannot add non-nullable column {col.name!r} to table {self.name!r} with existing rows',
                    )
            col.tbl_handle = self.handle
            col.id = self.next_col_id()

        # collect all columns we need to add, plus indices
        index_cols: dict[Column, tuple[index.BtreeIndex, Column | None, Column | None]] = {}
        all_cols: list[Column] = []
        for col in cols:
            all_cols.append(col)
            if self.has_default_idxs and col.name is not None and index.BtreeIndex.can_index(col):
                idx = index.BtreeIndex(uses_value_col=self.is_data_versioned)

                val_col, undo_col = self._create_index_columns(col, idx)
                index_cols[col] = (idx, val_col, undo_col)
                all_cols.extend(c for c in (val_col, undo_col) if c is not None)

        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)

        # create column md
        next_pos = self._next_col_pos()
        for col in all_cols:
            assert col.id is not None
            col.schema_version_add = self.schema_version
            assert col.schema_version_drop is None
            pos = next(next_pos) if not col.is_system_col else None
            col_md, sch_col_md = col.to_md(pos=pos)
            assert col.id not in self._tbl_md.column_md
            self._tbl_md.column_md[col.id] = col_md
            assert col.id not in self._schema_version_md.columns
            self._schema_version_md.columns[col.id] = sch_col_md

        # Create index md
        idx_ids: list[int] = []
        for col, (idx, val_col, undo_col) in index_cols.items():
            idx_ids.append(self._create_index_md(col, val_col, undo_col, idx_name=None, idx=idx))

        id_str = str(self.id)
        col_ids = [col.id for col in all_cols]
        tbl_ops = (
            TableOpsBuilder(id_str, tbl_version=self._tbl_md.current_version)
            .add(CreateTableVersionOp)
            .add(CreateColumnMdOp, column_ids=col_ids)
            .add(CreateStoreColumnsOp, column_ids=col_ids)
            .add(CreateStoreIdxsOp, idx_ids=idx_ids)
            .build()
        )
        return TableVersionMd(self._tbl_md, self._version_md, self._schema_version_md), tbl_ops

    def add_columns(self, cols: list[Column], print_stats: bool, on_error: Literal['abort', 'ignore']) -> UpdateStatus:
        """Adds columns to the table."""
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        assert self.is_mutable

        # we're creating a new schema version
        start_ts = time.perf_counter()
        self.bump_version(bump_schema_version=True)
        status = self._add_columns_in_version(cols, print_stats=print_stats, on_error=on_error)
        self.set_version_update_status(status)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Added columns {[col.name for col in cols]} to table {self.name}, new version: {self.version}')

        duration = time.perf_counter() - start_ts
        rate_str = f' ({status.num_rows / duration:.2f} rows/s)' if duration > 0 and status.num_rows > 0 else ''
        msg = (
            f'Added {status.num_rows} column value{"" if status.num_rows == 1 else "s"} '
            f'with {status.num_excs} error{"" if status.num_excs == 1 else "s"} '
            f'in {duration:.2f} s{rate_str}'
        )
        Env.get().console_logger.info(msg)
        _logger.info(f'Columns {[col.name for col in cols]}: {msg}')
        return status

    def _add_columns(
        self, cols: Iterable[Column], print_stats: bool, on_error: Literal['abort', 'ignore']
    ) -> UpdateStatus:
        """Add and populate columns within the current transaction"""
        from pixeltable.plan import Planner

        cols_to_add = list(cols)

        row_count = self.store_tbl.count()
        for col in cols_to_add:
            assert col.tbl_handle.id == self.id
            if not col.col_type.nullable and not col.is_computed and row_count > 0:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot add non-nullable column {col.name!r} to table {self.name!r} with existing rows',
                )

        num_excs = 0
        cols_with_excs: list[Column] = []
        next_pos = self._next_col_pos()
        for col in cols_to_add:
            assert col.id is not None
            col.schema_version_add = self.schema_version
            # add the column to the lookup structures now, rather than after the store changes executed successfully,
            # because it might be referenced by the next column's value_expr
            self.cols_by_id[col.id] = col
            if not col.is_system_col:
                self.cols_by_name[col.name] = col
            pos = next(next_pos) if not col.is_system_col else None
            col_md, col_schema_md = col.to_md(pos=pos)
            assert col.id not in self._tbl_md.column_md
            self._tbl_md.column_md[col.id] = col_md
            assert col.id not in self._schema_version_md.columns
            self._schema_version_md.columns[col.id] = col_schema_md

            if col.is_stored:
                self.store_tbl.add_column(col, if_not_exists=False)

            # cols_by_id was just mutated in-place; invalidate the TVP's cached CVMD so the next
            # create_add_column_plan() call (e.g. for a btree index column) sees the new column.
            self.path.clear_cached_md()

            if not col.is_computed or not col.is_stored or row_count == 0:
                continue

            # populate the column
            plan = Planner.create_add_column_plan(self.path, col)
            excs_per_col = 0
            with get_runtime().report_progress():
                try:
                    plan.ctx.title = self.display_str()
                    excs_per_col = self.store_tbl.write_column(col, plan, on_error == 'abort')
                except sql_exc.DBAPIError as exc:
                    get_runtime().catalog.convert_sql_exc(exc, self.id, self.handle, convert_db_excs=True)
                    # If it wasn't converted, re-raise as a generic Pixeltable error
                    # (this means it's not a known concurrency error; it's something else)
                    raise excs.Error(
                        excs.ErrorCode.INTERNAL_ERROR,
                        f'Unexpected SQL error during execution of computed column {col.name!r}:\n{exc}',
                    ) from exc
            if excs_per_col > 0:
                cols_with_excs.append(col)
                num_excs += excs_per_col

        get_runtime().catalog.record_column_dependencies(self)

        if print_stats:
            plan.ctx.profile.print(num_rows=row_count)

        # TODO: what to do about system columns with exceptions?
        row_counts = RowCountStats(upd_rows=row_count, num_excs=num_excs, computed_values=0)  # add_columns
        return UpdateStatus(
            cols_with_excs=[f'{col.get_tbl().name}.{col.name}' for col in cols_with_excs if col.name is not None],
            row_count_stats=row_counts,
        )

    def _next_col_pos(self) -> itertools.count:
        """Returns a counter starting at the next available column position."""
        highest_pos = max((c.pos for c in self._schema_version_md.columns.values() if c.pos is not None), default=-1)
        return itertools.count(start=highest_pos + 1)

    def drop_column(self, col: Column) -> None:
        """Drop a column from the table."""

        assert self.is_mutable
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'

        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)
        cols_to_drop = self._cascade_drop_column(col)
        self._drop_columns(cols_to_drop)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Dropped column {col.name} from table {self.name}, new version: {self.version}')

    def _cascade_drop_column(self, col: Column) -> list[Column]:
        """
        Marks any indices on the column as dropped and returns the column together with the value and undo columns of
        those indices.
        Also fixes up idxs/idxs_by_name/idxs_by_col.
        """
        if col.is_pk:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot drop primary key column {col.name!r}'
            )

        # drop this column and all dependent index columns and indices
        to_drop = [col]
        dropped_idx_info: list[TableVersion.IndexInfo] = []
        for idx_info in self.idxs_by_name.values():
            if idx_info.col != col:
                continue
            to_drop.extend(idx_info.columns)
            idx_md = self._tbl_md.index_md[idx_info.id]
            idx_md.schema_version_drop = self.schema_version
            assert idx_md.name in self.idxs_by_name
            dropped_idx_info.append(idx_info)

        # update index lookup structures
        for info in dropped_idx_info:
            del self.idxs[info.id]
            del self.idxs_by_name[info.name]
        if col.qid in self.idxs_by_col:
            del self.idxs_by_col[col.qid]
        return to_drop

    def _drop_columns(self, cols: Iterable[Column]) -> None:
        """Mark columns as dropped on a data-versioned table. On an operational table, remove them outright"""
        assert self.is_mutable

        for col in cols:
            if col.name is not None:
                assert col.name in self.cols_by_name
                del self.cols_by_name[col.name]
            assert col.id in self.cols_by_id
            del self.cols_by_id[col.id]
            # update stored md
            if self.is_data_versioned:
                col.schema_version_drop = self.schema_version
                self._tbl_md.column_md[col.id].schema_version_drop = col.schema_version_drop
            else:
                if col.is_stored:
                    self.store_tbl.drop_column(col, if_exists=False)
                del self._tbl_md.column_md[col.id]
            del self._schema_version_md.columns[col.id]

        # Update positions of user columns
        pos = itertools.count()
        for schema_col in self._schema_version_md.columns.values():
            if not schema_col.is_system_column:
                schema_col.pos = next(pos)

        self.store_tbl.create_sa_tbl()
        get_runtime().catalog.record_column_dependencies(self)

    def apply_schema_change(
        self,
        expected_schema_version: int,
        added_cols: list[Column],
        dropped_cols: list[Column],
        added_idxs: list[IndexSpec],
        dropped_idx_ids: list[int],
    ) -> UpdateStatus:
        """Apply multiple column and index add/drop operations as a single new schema version.

        - added_cols are in declaration order; their value expressions may still contain exprs.ColumnRefByName
          referencing other columns in added_cols, which are resolved here once ids are assigned
        - Drops precede adds, and index drops precede column drops, so an index that is both explicitly dropped and
          attached to a dropped column is processed only once.
        """
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        assert self.is_mutable

        if self.schema_version != expected_schema_version:
            raise excs.ConcurrencyError(
                excs.ErrorCode.CONCURRENT_MODIFICATION,
                f'Table {self.name!r} was modified since update_all() computed its changes; re-run update_all().',
            )

        self._validate_idx_drops(dropped_idx_ids)

        self.bump_version(bump_schema_version=True)

        cols_to_drop: list[Column] = []
        for idx_id in dropped_idx_ids:
            cols_to_drop.extend(self._cascade_drop_index(idx_id))
        for col in dropped_cols:
            cols_to_drop.extend(self._cascade_drop_column(col))
        if len(cols_to_drop) > 0:
            self._drop_columns(cols_to_drop)

        # Validate the new indexes against the post-drop state, so that dropping an index and adding another one with
        # the same name, or on the same column, in a single change set is allowed.
        self._validate_idxs(self.id, added_idxs, self.has_default_idxs, self.idxs.values())

        status = UpdateStatus()
        if len(added_cols) > 0:
            status += self._add_columns_in_version(added_cols, print_stats=False, on_error='abort')
        for col, idx_name, idx in added_idxs:
            assert isinstance(col, Column)
            status += self._add_index(col, idx_name, idx)

        self.set_version_update_status(status)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Applied model updates to table {self.name}, new version: {self.version}')
        return status

    def _add_columns_in_version(
        self, cols: list[Column], print_stats: bool, on_error: Literal['abort', 'ignore']
    ) -> UpdateStatus:
        """Add cols within the current schema version, each with a default btree index if the table enables those.

        - the caller is responsible for recording the schema version change
        - value expressions that carry ColumnRefByName placeholders are resolved against cols, which need to be in
          declaration order, so a computed column only refers back to columns preceding it
        - an expression without placeholders is left as it is.
        """
        assert all(is_valid_identifier(col.name) for col in cols if col.name is not None)
        assert all(col.stored is not None for col in cols)
        assert all(col.name not in self.cols_by_name for col in cols if col.name is not None)

        # assign column ids
        for col in cols:
            if col.is_pk:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot add primary key column {col.name!r} after table creation',
                )
            col.tbl_handle = self.handle
            col.id = self.next_col_id()

        # the ids exist now, so a placeholder can be resolved to the column it names
        subst = exprs.ExprDict[exprs.Expr](
            (
                exprs.ColumnRefByName(col.name),
                exprs.ColumnRef(
                    col.column_version_md(),
                    perform_validation=((col.media_validation or self.media_validation) == MediaValidation.ON_READ),
                ),
            )
            for col in cols
        )
        for col in cols:
            value_expr = col.value_expr
            if value_expr is not None:
                col.set_value_expr(value_expr.substitute(subst))
            if col.is_computed:
                col.check_value_expr()

        index_cols: dict[Column, tuple[index.BtreeIndex, Column | None, Column | None]] = {}
        all_cols: list[Column] = []
        for col in cols:
            all_cols.append(col)
            if self.has_default_idxs and col.name is not None and index.BtreeIndex.can_index(col):
                idx = index.BtreeIndex(uses_value_col=self.is_data_versioned)
                val_col, undo_col = self._create_index_columns(col, idx)
                index_cols[col] = (idx, val_col, undo_col)
                all_cols.extend(c for c in (val_col, undo_col) if c is not None)

        status = self._add_columns(all_cols, print_stats=print_stats, on_error=on_error)
        # create the indices and their md records only once the columns they index exist
        for col, (idx, val_col, undo_col) in index_cols.items():
            self._create_index(col, val_col, undo_col, idx_name=None, idx=idx)
        return status

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column."""
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        if not self.is_mutable:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot rename column for immutable table {self.name!r}'
            )
        col = self.path.get_column(old_name)
        if col is None:
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {old_name}')
        if col.get_tbl().id != self.id:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot rename base table column {col.name!r}'
            )
        if not is_valid_identifier(new_name):
            raise excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME, f'Invalid column name: {new_name}')
        if new_name in self.cols_by_name:
            raise excs.AlreadyExistsError(excs.ErrorCode.COLUMN_ALREADY_EXISTS, f'Column {new_name!r} already exists')
        del self.cols_by_name[old_name]
        col.name = new_name
        self.cols_by_name[new_name] = col
        self._schema_version_md.columns[col.id].name = new_name

        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)

        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'Renamed column {old_name} to {new_name} in table {self.name}, new version: {self.version}')

    def alter_column(self, col: Column, type_: ts.ColumnType) -> None:
        """Alter the type of a column. Currently only supports widening a value column to nullable."""
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        assert self.is_mutable
        assert not col.is_computed
        assert not col.is_pk
        if type_ == col.col_type:
            # no-op
            return

        valid_change = type_.matches(col.col_type) and not col.col_type.nullable and type_.nullable
        if not valid_change:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Column {col.name!r} type cannot be changed from {col.col_type} to {type_}',
            )

        # alter column type from required to nullable
        old_type = col.col_type
        col.col_type = type_
        self._schema_version_md.columns[col.id].col_type = type_.as_dict()
        self.bump_version(bump_schema_version=True)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(
            f'Altered column {col.name!r} type from {old_type} to {type_} in table {self.name}, new version: '
            f'{self.version}'
        )

    def set_comment(self, new_comment: str | None) -> None:
        _logger.info(f'[{self.name}] Updating comment: {new_comment}')
        self.comment = new_comment
        self._create_schema_version()

    def _create_schema_version(self) -> None:
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        # we're creating a new schema version
        self.bump_version(bump_schema_version=True)
        self._write_md(new_version=True, new_schema_version=True)
        _logger.info(f'[{self.name}] Updating table schema to version: {self.version}')

    def insert(
        self,
        source: list[dict[str, Any]] | SqlDataSource | None,
        query: Query | None,
        print_stats: bool = False,
        fail_on_exception: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """
        Insert rows into this table from an explicit list of dicts, a Query, or a SqlDataSource.
        """
        from pixeltable.plan import Planner

        assert self.is_insertable
        # Exactly one of source / query must be specified
        assert (source is None) != (query is None)
        with telemetry.span('pixeltable.plan.create', level=telemetry.DEBUG):
            if query is not None:
                plan = Planner.create_query_insert_plan(self, query, ignore_errors=not fail_on_exception)
            else:
                plan = Planner.create_insert_plan(self, source, ignore_errors=not fail_on_exception)

        rowid_gen: Iterator[int] | None = None
        # For data-versioned tables, generate rowids from the table's sequence.
        # For operational tables, row ids are generated by StoreBase itself.
        if self.is_data_versioned:

            def rowids() -> Iterator[int]:
                while True:
                    rowid = self.next_row_id
                    self.next_row_id += 1
                    yield rowid

            rowid_gen = rowids()

        with get_runtime().report_progress():
            result = self._insert(
                plan,
                time.time(),
                print_stats=print_stats,
                rowids=rowid_gen,
                abort_on_exc=fail_on_exception,
                return_rows=return_rows,
            )
            return result

    def _insert(
        self,
        exec_plan: 'exec.ExecNode',
        timestamp: float,
        *,
        rowids: Iterator[int] | None = None,
        print_stats: bool = False,
        abort_on_exc: bool = False,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Insert rows produced by exec_plan and propagate to views"""
        if self.is_data_versioned:
            # we're creating a new version
            self.bump_version(timestamp, bump_schema_version=False)
        exec_plan.ctx.title = self.display_str()
        cols_with_excs, row_counts, rows = self.store_tbl.insert_rows(
            exec_plan, rowids=rowids, abort_on_exc=abort_on_exc, return_rows=return_rows
        )
        result = UpdateStatus(
            cols_with_excs=[f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs],
            rows=rows,
            row_count_stats=row_counts,
        )

        # update views
        for view in self.mutable_views:
            from pixeltable.plan import Planner

            # Make this span current only when it is nested under an operation span.
            with telemetry.span(
                'pixeltable.view_load', set_current=telemetry.current_span() is not None, view=view.get().name
            ):
                view_plan = Planner.create_view_load_plan(view.get().path, propagates_insert=True)
                status = view.get()._insert(view_plan, timestamp, print_stats=print_stats)
            result += status.to_cascade()

        # Use the net status after all propagations
        if self.is_data_versioned:
            self.set_version_update_status(result)
            self._write_md(new_version=True, new_schema_version=False)
            _logger.info(f'TableVersion {self.name}: new version {self.version}')
        if print_stats:
            exec_plan.ctx.profile.print(num_rows=result.num_rows)
        return result

    def update(
        self,
        value_spec: dict[str, Any],
        where: exprs.Expr | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update rows in this TableVersionPath.
        Args:
            value_spec: a list of (column, value) pairs specifying the columns to update and their new values.
            where: a predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns,
                including within views.
            return_rows: if True, capture the post-update row state in UpdateStatus.rows.
        """
        from pixeltable.exprs import SqlElementCache
        from pixeltable.plan import Planner

        assert self.is_mutable
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'

        update_spec = self._validate_update_spec(value_spec, allow_pk=False, allow_exprs=True, allow_media=True)
        if where is not None:
            if not isinstance(where, exprs.Expr):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_EXPRESSION,
                    f'`where` argument must be a valid Pixeltable expression; got `{type(where)}`',
                )
            analysis_info = Planner.analyze(self.path, where)
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Filter not expressible in SQL: {analysis_info.filter}'
                )

        plan, updated_cols, recomputed_cols = Planner.create_update_plan(self.path, update_spec, [], cascade)

        result = self.propagate_update(
            [plan],
            where.sql_expr(SqlElementCache()) if where is not None else None,
            recomputed_cols,
            modified_cols=list(update_spec.keys()),
            base_versions=[],
            timestamp=time.time(),
            cascade=cascade,
            return_rows=return_rows,
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
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update rows in batch.
        Args:
            batch: one dict per row, each mapping Columns to LiteralExprs representing the new values
            rowids: if not empty, one tuple per row, each containing the rowid values for the corresponding row in batch
            return_rows: if True, returns all newly-inserted/updated rows in UpdateStatus.rows.
        """
        from pixeltable.plan import Planner

        # if we do lookups of rowids, we must have one for each row in the batch
        assert len(rowids) == 0 or len(rowids) == len(batch)
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'

        plan, row_update_node, delete_where_clause, updated_cols, recomputed_cols = Planner.create_batch_update_plan(
            self.path, batch, rowids, cascade=cascade
        )
        result = self.propagate_update(
            [plan],
            delete_where_clause,
            recomputed_cols,
            modified_cols=updated_cols,
            base_versions=[],
            timestamp=time.time(),
            cascade=cascade,
            return_rows=return_rows,
        )
        result += UpdateStatus(updated_cols=[c.qualified_name for c in updated_cols])

        unmatched_rows = row_update_node.unmatched_rows()
        if len(unmatched_rows) > 0:
            if error_if_not_exists:
                raise excs.NotFoundError(
                    excs.ErrorCode.ROW_NOT_FOUND, f'batch_update(): {len(unmatched_rows)} row(s) not found'
                )
            if insert_if_not_exists:
                insert_status = self.insert(
                    unmatched_rows, None, print_stats=False, fail_on_exception=False, return_rows=return_rows
                )
                result += insert_status.to_cascade()
        return result

    def _validate_update_spec(
        self, value_spec: dict[str, Any], allow_pk: bool, allow_exprs: bool, allow_media: bool
    ) -> dict[Column, exprs.Expr]:
        update_targets: dict[Column, exprs.Expr] = {}
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'Update specification: dict key must be column name; got {col_name!r}',
                )
            if col_name == _ROWID_COLUMN_NAME:
                # a valid rowid is a list of ints, one per rowid column
                num_rowid_cols = len(self.store_tbl.rowid_columns())
                if len(val) != num_rowid_cols:
                    raise excs.Error(
                        excs.ErrorCode.INTERNAL_ERROR,
                        f'Malformed _rowid: expected {num_rowid_cols} components, got {len(val)}',
                    )
                if not all(isinstance(el, int) for el in val):
                    raise excs.Error(
                        excs.ErrorCode.INTERNAL_ERROR, f'Malformed _rowid: all components must be int, got {val!r}'
                    )
                continue
            col = self.path.get_column(col_name)
            if col is None:
                raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {col_name}')
            if col.get_tbl().id != self.id:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Column {col.name!r} is a base table column and cannot be updated',
                )
            if col.is_computed:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Column {col_name!r} is computed and cannot be updated'
                )
            if col.is_pk and not allow_pk:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Column {col_name!r} is a primary key column and cannot be updated',
                )
            if col.col_type.is_media_type() and not allow_media:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Column {col_name!r} is a media column and cannot be updated'
                )

            # make sure that the value is compatible with the column type
            value_expr: exprs.Expr
            try:
                # check if this is a literal
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError as exc:
                if not allow_exprs:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Column {col_name!r}: value is not a valid literal for this column '
                        f'(expected `{col.col_type}`): {val!r}',
                    ) from exc
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Column {col_name!r}: value is not a recognized literal or expression: {val!r}',
                    ) from exc
                if not col.col_type.is_supertype_of(value_expr.col_type, ignore_nullable=True):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'The literal value {val!r} is not compatible with the type '
                        f'`{col.col_type}` of column {col_name!r}',
                    ) from exc
            update_targets[col] = value_expr

        return update_targets

    def recompute_columns(
        self, col_names: list[str], where: exprs.Expr | None = None, errors_only: bool = False, cascade: bool = True
    ) -> UpdateStatus:
        from pixeltable.exprs import CompoundPredicate, SqlElementCache
        from pixeltable.plan import Planner

        assert self.is_mutable
        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        assert all(name in self.cols_by_name for name in col_names)
        assert len(col_names) > 0
        assert len(col_names) == 1 or not errors_only

        target_columns = [self.cols_by_name[name] for name in col_names]
        where_clause: exprs.Expr | None = None
        if where is not None:
            self._validate_where_clause(where, error_prefix='`where` argument')
            where_clause = where
        if errors_only:
            errortype_pred = (
                exprs.ColumnPropertyRef(
                    exprs.ColumnRef(self.path.get_column_md(target_columns[0].qid)),
                    exprs.ColumnPropertyRef.Property.ERRORTYPE,
                )
                != None
            )
            where_clause = CompoundPredicate.make_conjunction([where_clause, errortype_pred])
        plan, updated_cols, recomputed_cols = Planner.create_update_plan(
            self.path, update_targets={}, recompute_targets=target_columns, cascade=cascade
        )

        result = self.propagate_update(
            [plan],
            where_clause.sql_expr(SqlElementCache()) if where_clause is not None else None,
            recomputed_cols,
            modified_cols=target_columns,
            base_versions=[],
            timestamp=time.time(),
            cascade=cascade,
        )
        result += UpdateStatus(updated_cols=updated_cols)
        return result

    def propagate_update(
        self,
        plans: list[exec.ExecNode],
        where_clause: sql.ColumnElement | None,
        recomputed_view_cols: list[Column],
        modified_cols: list[Column],
        base_versions: list[int | None],
        timestamp: float,
        cascade: bool,
        base_membership_change: bool = False,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """
        base_membership_change: True if the update may have added rows to this table's base; a row appearing in the
            base is a row this view needs to add, whether or not it has a predicate of its own
        """
        from pixeltable.plan import Planner

        assert self.is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
        get_runtime().catalog.mark_modified_tv(self.handle)
        result = UpdateStatus()
        create_new_table_version = len(plans) > 0
        if create_new_table_version:
            self.bump_version(timestamp, bump_schema_version=False)
            # soft delete must be done before insert, otherwise we would have duplicate primary key values
            # upon insert since the rows would be duplicated until the soft delete occurs
            self.store_tbl.soft_delete_rows(
                self.version, base_versions=base_versions, match_on_vmin=True, where_clause=where_clause
            )

            if self.base is not None and self.base.get().is_view:
                # a view base can retire a row without replacing it:
                # - a component view's iterator can yield fewer rows
                # - a row can stop satisfying a filter
                # Our rows under such a base row join to a base row created at
                # an earlier version, so the match above doesn't reach them; match on the retired base rows instead.
                # An insertable-table base always replaces the rows it retires, so nothing is left behind there.
                self.store_tbl.soft_delete_rows(
                    self.version, base_versions=base_versions, match_on_vmin=False, where_clause=where_clause
                )

        for p in plans:
            cols_with_excs, row_counts, rows = self.store_tbl.insert_rows(p, return_rows=return_rows)
            result += UpdateStatus(
                row_count_stats=row_counts.insert_to_update(),
                cols_with_excs=[f'{self.name}.{self.cols_by_id[cid].name}' for cid in cols_with_excs],
                rows=rows,
            )

        if cascade:
            base_versions = [None if len(plans) == 0 else self.version, *base_versions]  # don't update in place
            changed_qids = {c.qid for c in modified_cols} | {c.qid for c in recomputed_view_cols}

            # propagate to views
            for view in self.mutable_views:
                view_tv = view.get()
                recomputed_cols = [col for col in recomputed_view_cols if col.get_tbl().id == view.id]
                # any and all are equivalent since all iterator columns share the same input columns meaning
                # if one is recomputed they all are. However any is more efficient since it can short circuit.
                needs_iterator_reload = view_tv.is_component_view and any(
                    view_tv.is_iterator_column(col) for col in recomputed_cols
                )

                membership_change = base_membership_change or (
                    # an update can move a row in or out of a filtered view, but only if it changed one of the columns
                    # the filter is evaluated over; a view without a predicate still gains the rows its base gained
                    view_tv.predicate is not None
                    and len(exprs.Expr.get_refd_column_ids(view_tv.predicate.as_dict()) & changed_qids) > 0
                )

                view_plans: list[exec.ExecNode] = []
                if needs_iterator_reload or (membership_change and view_tv.is_component_view):
                    view_plans.append(Planner.create_view_load_plan(view_tv.path, propagates_insert=True))
                elif membership_change:
                    # rows already in the view that still satisfy the predicate; their stored columns that aren't
                    # recomputed carry over from the row that was just soft-deleted
                    view_plans.append(Planner.create_view_update_plan(view_tv.path, recompute_targets=recomputed_cols))
                    # rows that satisfy the predicate now and weren't in the view; nothing to carry over
                    view_plans.append(
                        Planner.create_view_load_plan(view_tv.path, propagates_insert=True, exclude_existing_rows=True)
                    )
                elif len(recomputed_cols) > 0:
                    view_plans.append(Planner.create_view_update_plan(view_tv.path, recompute_targets=recomputed_cols))

                status = view_tv.propagate_update(
                    view_plans,
                    None,
                    recomputed_view_cols,
                    modified_cols,
                    base_versions=base_versions,
                    timestamp=timestamp,
                    cascade=True,
                    base_membership_change=membership_change,
                )
                result += status.to_cascade()

        if create_new_table_version:
            self.set_version_update_status(result)
            self._write_md(new_version=True, new_schema_version=False)
        return result

    def _validate_where_clause(self, pred: exprs.Expr, error_prefix: str) -> None:
        """Validates that pred can be expressed as a SQL Where clause"""
        assert self.is_mutable
        from pixeltable.exprs import Expr
        from pixeltable.plan import Planner

        if not isinstance(pred, Expr):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_EXPRESSION,
                f'{error_prefix} must be a valid Pixeltable expression; got `{type(pred)}`',
            )
        analysis_info = Planner.analyze(self.path, pred)
        # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
        if analysis_info.filter is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Filter not expressible in SQL: {analysis_info.filter}'
            )

    def delete(self, where: exprs.Expr | None = None) -> UpdateStatus:
        assert self.is_insertable
        if where is not None:
            self._validate_where_clause(where, error_prefix='`where` argument')
        status = self.propagate_delete(where, base_versions=[], timestamp=time.time())
        return status

    def propagate_delete(
        self, where: exprs.Expr | None, base_versions: list[int | None], timestamp: float
    ) -> UpdateStatus:
        """Delete rows in this table and propagate to views"""
        get_runtime().catalog.mark_modified_tv(self.handle)

        sql_where_clause = where.sql_expr(exprs.SqlElementCache()) if where is not None else None
        if self.is_data_versioned:
            del_rows = self.store_tbl.soft_delete_rows(
                self.version + 1, base_versions=base_versions, match_on_vmin=False, where_clause=sql_where_clause
            )
        else:
            del_rows = self.store_tbl.delete_rows(where_clause=sql_where_clause)
        row_counts = RowCountStats(del_rows=del_rows)  # delete
        result = UpdateStatus(row_count_stats=row_counts)
        if del_rows > 0 and self.is_data_versioned:
            # we're creating a new version
            self.bump_version(timestamp, bump_schema_version=False)
        for view in self.mutable_views:
            status = view.get().propagate_delete(
                where=None, base_versions=[self.version, *base_versions], timestamp=timestamp
            )
            result += status.to_cascade()

        if del_rows > 0 and self.is_data_versioned:
            self.set_version_update_status(result)
            self._write_md(new_version=True, new_schema_version=False)
        return result

    def revert(self) -> None:
        """Reverts the table to the previous version."""
        assert self.is_mutable
        assert self.is_data_versioned
        if self.version == 0:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot revert version 0')
        self._revert()

    def _revert(self) -> None:
        """
        Reverts the stored metadata for this table version and propagates to views.

        Doesn't attempt to revert the in-memory metadata, but instead invalidates this TableVersion instance
        and relies on Catalog to reload it
        """
        conn = get_runtime().conn
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
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Current version is needed for {len(result)} snapshot{"s" if len(result) > 1 else ""}: '
                f'({", ".join(names)})',
            )

        conn.execute(sql.delete(self.store_tbl.sa_tbl).where(self.store_tbl.sa_tbl.c.v_min == self.version))

        # revert new deletions
        set_clause: dict[sql.Column, Any] = {self.store_tbl.sa_tbl.c.v_max: schema.Table.MAX_VERSION}
        for index_info in self.idxs.values():
            assert index_info.val_col is not None
            assert index_info.undo_col is not None
            # copy the index value back from the undo column and reset the undo column to NULL
            set_clause[index_info.val_col.sa_col] = index_info.undo_col.sa_col
            set_clause[index_info.undo_col.sa_col] = None
        stmt = sql.update(self.store_tbl.sa_tbl).values(set_clause).where(self.store_tbl.sa_tbl.c.v_max == self.version)
        conn.execute(stmt)

        # revert schema changes:
        # - undo changes to self._tbl_md and write that back
        # - delete newly-added TableVersion/TableSchemaVersion records
        get_runtime().catalog.mark_modified_tv(self.handle)
        old_version = self.version
        if self.version == self.schema_version:
            # physically delete newly-added columns and remove them from the stored md
            added_cols = [col for col in self.cols_by_id.values() if col.schema_version_add == self.schema_version]
            if len(added_cols) > 0:
                self._tbl_md.next_col_id = min(col.id for col in added_cols)
                for col in added_cols:
                    if col.is_stored:
                        self.store_tbl.drop_column(col, if_exists=False)
                    del self._tbl_md.column_md[col.id]
                    del self._schema_version_md.columns[col.id]

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

        self._tbl_md.current_version = self._version_md.version = self.version - 1

        self._write_md(new_version=False, new_schema_version=False)

        # propagate to views
        for view in self.mutable_views:
            view.get()._revert()

        # force reload on next operation
        self.is_validated = False
        get_runtime().catalog.remove_tbl_version(self.key)

        # delete newly-added data
        # Do this at the end, after all DB operations have completed.
        # TODO: The transaction could still fail. Really this should be done via PendingTableOps.
        self.delete_media(tbl_version=old_version)
        _logger.info(f'TableVersion {self.name!r}: reverted to version {self.version}')

    @property
    def id(self) -> UUID:
        return self.key.tbl_id

    @property
    def effective_version(self) -> int | None:
        if not self.is_data_versioned:
            assert self.key.effective_version is None
        return self.key.effective_version

    @property
    def tbl_md(self) -> schema.TableMd:
        return self._tbl_md

    @property
    def version_md(self) -> schema.VersionMd:
        return self._version_md

    @property
    def schema_version_md(self) -> schema.SchemaVersionMd:
        return self._schema_version_md

    @property
    def view_md(self) -> schema.ViewMd | None:
        return self._tbl_md.view_md

    @property
    def name(self) -> str:
        return self._tbl_md.name

    @property
    def user(self) -> str | None:
        return self._tbl_md.user

    @property
    def comment(self) -> str | None:
        return self._schema_version_md.comment

    @comment.setter
    def comment(self, c: str | None) -> None:
        assert self.effective_version is None
        self._schema_version_md.comment = c

    @property
    def custom_metadata(self) -> Any:
        return self._schema_version_md.custom_metadata

    @property
    def version(self) -> int:
        return self._version_md.version

    @property
    def created_at(self) -> float:
        return self._version_md.created_at

    @property
    def schema_version(self) -> int:
        return self._schema_version_md.schema_version

    @property
    def is_data_versioned(self) -> bool:
        return self._tbl_md.is_data_versioned

    @property
    def has_default_idxs(self) -> bool:
        """Whether eligible columns of this table get a default B-tree index.

        This is fixed at creation time and is the sole determinant for columns added later.

        For tables created before this property was recorded in the metadata, fall back to True except for snapshots
        (that are not allowed to have indexes).
        """
        if self._tbl_md.has_default_idxs is None:
            return not self._tbl_md.is_snapshot
        return self._tbl_md.has_default_idxs

    def bump_version(self, timestamp: float | None = None, *, bump_schema_version: bool) -> None:
        """
        Increments the table version and adjusts all associated metadata. This will *not* trigger a database action;
        _write_md() must be called separately to persist the changes.

        Args:
            timestamp: the creation time for the new version. Can be used to synchronize multiple metadata changes
                to the same timestamp. If `None`, then defaults to `time.time()`.
            bump_schema_version: if True, also adjusts the schema version (setting it equal to the new version)
                and associated metadata.
        """
        # on operational tables, only a schema change can trigger a version bump
        assert self.is_data_versioned or bump_schema_version
        assert self.effective_version is None

        if timestamp is None:
            timestamp = time.time()

        get_runtime().catalog.mark_modified_tv(self.handle)

        old_version = self._tbl_md.current_version
        assert self._version_md.version == old_version
        new_version = old_version + 1
        self._tbl_md.current_version = new_version
        self._version_md.version = new_version
        self._version_md.created_at = timestamp

        if bump_schema_version:
            old_schema_version = self._tbl_md.current_schema_version
            assert self._version_md.schema_version == old_schema_version
            assert self._schema_version_md.schema_version == old_schema_version
            self._tbl_md.current_schema_version = new_version
            self._version_md.schema_version = new_version
            self._schema_version_md.preceding_schema_version = old_schema_version
            self._schema_version_md.schema_version = new_version

    @property
    def preceding_schema_version(self) -> int | None:
        return self._schema_version_md.preceding_schema_version

    def set_version_update_status(self, status: UpdateStatus) -> None:
        """Record status as the UpdateStatus of the change that created the current version."""
        assert self.is_data_versioned
        assert self.effective_version is None
        # we need to strip out UpdateStatus.rows, if set
        if status.rows is not None:
            status = dataclasses.replace(status, rows=None)
        self._version_md.update_status = status

    @property
    def media_validation(self) -> MediaValidation:
        return MediaValidation[self._schema_version_md.media_validation.upper()]

    def next_col_id(self) -> int:
        val = self._tbl_md.next_col_id
        self._tbl_md.next_col_id += 1
        return val

    @property
    def next_idx_id(self) -> int:
        return self._tbl_md.next_idx_id

    @next_idx_id.setter
    def next_idx_id(self, idx_id: int) -> None:
        assert self.effective_version is None
        self._tbl_md.next_idx_id = idx_id

    @property
    def next_row_id(self) -> int:
        assert self.is_data_versioned
        return self._tbl_md.next_row_id

    @next_row_id.setter
    def next_row_id(self, row_id: int) -> None:
        assert self.is_data_versioned
        assert self.effective_version is None
        self._tbl_md.next_row_id = row_id

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    @property
    def is_mutable(self) -> bool:
        return not self.is_snapshot

    @property
    def is_view(self) -> bool:
        return self.view_md is not None

    @property
    def include_base_columns(self) -> bool:
        return self.view_md is not None and self.view_md.include_base_columns

    @property
    def is_component_view(self) -> bool:
        return self.iterator_call is not None

    @property
    def is_insertable(self) -> bool:
        """Returns True if this corresponds to an InsertableTable"""
        return self.is_mutable and not self.is_view

    def display_str(self) -> str:
        return f'{"Table" if self.is_insertable else "View"} {self.name!r}'

    def is_iterator_column(self, col: Column) -> bool:
        """Returns True if col is produced by an iterator (including the pos column)"""
        return col.is_iterator_col

    def iterator_columns(self) -> list[Column]:
        """Return all iterator-produced columns (including the pos column)"""
        # This is correct because self.cols_by_id is ordered by id (asc).
        return list(self.cols_by_id.values())[: self.num_iterator_cols]

    def iterator_args_expr(self) -> InlineDict | None:
        if self.is_component_view:
            return InlineDict(self.iterator_call.bound_args).copy()
        return None

    def primary_key_columns(self) -> list[Column]:
        """Return all non-system columns"""
        return [c for c in self.cols_by_id.values() if c.is_pk]

    @property
    def primary_key(self) -> list[str]:
        """Return the names of the primary key columns"""
        return [c.name for c in self.cols_by_id.values() if c.is_pk]

    def get_required_col_names(self) -> list[str]:
        """Return the names of all columns for which values must be specified in insert()"""
        assert not self.is_view
        names = [c.name for c in self.cols_by_name.values() if not c.is_computed and not c.col_type.nullable]
        return names

    def get_computed_col_names(self) -> list[str]:
        """Return the names of all computed columns"""
        names = [c.name for c in self.cols_by_name.values() if c.is_computed]
        return names

    @property
    def idx_val_cols(self) -> set[Column]:
        """All index value columns of all indexes in this version."""
        return {info.val_col for info in self.idxs.values() if info.val_col is not None}

    @property
    def idx_undo_cols(self) -> set[Column]:
        return {info.undo_col for info in self.idxs.values() if info.undo_col is not None}

    def get_idx_val_columns(self, cols: Iterable[Column]) -> set[Column]:
        # assumes that the indexed columns are all in this table
        assert all(col.get_tbl().id == self.id for col in cols)
        col_ids = {col.id for col in cols}
        return {info.val_col for info in self.idxs.values() if info.col.id in col_ids and info.val_col is not None}

    def get_idx(self, col: Column, idx_name: str | None, idx_cls: type[index.IndexBase]) -> TableVersion.IndexInfo:
        if not self.supports_idxs:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Snapshot does not support indices')
        candidates = [info for info in self.idxs_by_col.get(col.qid, []) if isinstance(info.idx, idx_cls)]
        if len(candidates) == 0:
            raise excs.NotFoundError(
                excs.ErrorCode.INDEX_NOT_FOUND, f'No {idx_cls.display_name()} index found for column {col.name!r}'
            )
        if len(candidates) > 1 and idx_name is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Column {col.name!r} has multiple {idx_cls.display_name()} indices; specify `idx_name` instead',
            )
        if idx_name is not None and idx_name not in [info.name for info in candidates]:
            raise excs.NotFoundError(
                excs.ErrorCode.INDEX_NOT_FOUND, f'Index {idx_name!r} not found for column {col.name!r}'
            )
        return candidates[0] if idx_name is None else next(info for info in candidates if info.name == idx_name)

    def find_btree_index(self, col: Column) -> TableVersion.IndexInfo | None:
        """Return the B-tree index on col, or None if it doesn't have one."""
        assert col.tbl_handle.id == self.id
        infos = [info for info in self.idxs_by_col.get(col.qid, []) if isinstance(info.idx, index.BtreeIndex)]
        assert len(infos) <= 1, repr(col)  # at most one B-tree index per column
        return infos[0] if len(infos) > 0 else None

    def get_dependent_columns(self, cols: Iterable[Column]) -> set[Column]:
        """
        Return the set of columns that transitively depend on any of the given ones.
        """
        cat = get_runtime().catalog
        result = set().union(*[cat.get_column_dependents(col.get_tbl().id, col.id) for col in cols])
        if len(result) > 0:
            result.update(self.get_dependent_columns(result))
        return result

    def num_rowid_columns(self) -> int:
        """Return the number of columns of the rowids, without accessing store_tbl"""
        if self.is_component_view:
            return 1 + self.base.get().num_rowid_columns()
        return 1

    def as_dict(self) -> dict:
        return self.key.as_dict()

    @classmethod
    def from_dict(cls, d: dict) -> TableVersion:
        key = TableVersionKey.from_dict(d)
        return get_runtime().catalog.get_tbl_version(key)
