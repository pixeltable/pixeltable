from __future__ import annotations

import abc
import copy
import dataclasses
import threading
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pixeltable.exceptions as excs
from pixeltable.func.globals import resolve_symbol
from pixeltable.index import IndexBase
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime

from .column import Column
from .globals import ColumnVersionMd, MediaValidation, QColumnId, TableVersionMd
from .table_version import TableVersion, TableVersionKey
from .table_version_handle import TableVersionHandle

if TYPE_CHECKING:
    from .catalog import Catalog


@dataclasses.dataclass(frozen=True, slots=True)
class TablePathKey:
    """A table path's version identity: a sequence of TableVersionKey, leaf first then bases."""

    keys: tuple[TableVersionKey, ...]

    @property
    def leaf(self) -> TableVersionKey:
        return self.keys[0]

    def as_dict(self) -> dict:
        # recursive {tbl_version, base} shape, byte-identical to TableVersionPath.as_dict()
        base = TablePathKey(self.keys[1:]).as_dict() if len(self.keys) > 1 else None
        return {'tbl_version': self.keys[0].as_dict(), 'base': base}

    @classmethod
    def from_dict(cls, d: dict) -> TablePathKey:
        keys = [TableVersionKey.from_dict(d['tbl_version'])]
        if d['base'] is not None:
            keys.extend(cls.from_dict(d['base']).keys)
        return cls(tuple(keys))


class TablePath(abc.ABC):
    """
    TablePath represents an abstract interface to the complete metadata of a table or view
    - completeness here means all the metadata needed to execute queries and updates against a particular version of a
      table/view.
    - for a view, it also includes metadata for all of its bases
    - multiple snapshots can reference the same TableVersion, but with different bases, which means that the
      graph of TableVersions is a DAG, not a tree (which is why we cannot embed the DAG into TableVersion directly)


    All subclasses need to be thread-safe.
    """

    base: TablePath | None

    @property
    @abc.abstractmethod
    def tbl_id(self) -> UUID: ...

    @property
    def tbl_ids(self) -> list[UUID]:
        if self.base is not None:
            return [self.tbl_id, *self.base.tbl_ids]
        else:
            return [self.tbl_id]

    @abc.abstractmethod
    def tbl_name(self) -> str: ...

    @abc.abstractmethod
    def version(self) -> int | None: ...

    @abc.abstractmethod
    def effective_version(self) -> int | None: ...

    @abc.abstractmethod
    def is_snapshot(self) -> bool: ...

    @abc.abstractmethod
    def is_view(self) -> bool: ...

    @abc.abstractmethod
    def is_mutable(self) -> bool: ...

    @abc.abstractmethod
    def is_versioned(self) -> bool: ...

    @property
    def path_len(self) -> int:
        return 1 if self.base is None else 1 + self.base.path_len

    @abc.abstractmethod
    def get_column_md(self, qcolid: QColumnId) -> ColumnVersionMd: ...

    @abc.abstractmethod
    def get_column_md_by_name(self, name: str) -> ColumnVersionMd | None: ...

    @abc.abstractmethod
    def column_md(self) -> list[ColumnVersionMd]: ...

    @abc.abstractmethod
    def has_column(self, qcolid: QColumnId) -> bool:
        """Return True if this table path contains the given column."""

    @abc.abstractmethod
    def get_idx_md(self, qcolid: QColumnId, name: str | None, idx_class: type[IndexBase]) -> schema.IndexMd:
        """Return the index metadata for an index on the given column."""

    def key(self) -> TablePathKey:
        """The path's effective-version identity (None for live elements, version for snapshots)."""
        keys = [TableVersionKey(self.tbl_id, self.effective_version())]
        if self.base is not None:
            keys.extend(self.base.key().keys)
        return TablePathKey(tuple(keys))

    def snapshot_key(self) -> TablePathKey:
        """The path's snapshot-version identity: the concrete version of each element."""
        keys = [TableVersionKey(self.tbl_id, self.version())]
        if self.base is not None:
            keys.extend(self.base.snapshot_key().keys)
        return TablePathKey(tuple(keys))

    def as_schema_path(self) -> schema.TableVersionPath:
        """Serialize to the persisted base_versions form: a flat (tbl_id_hex, effective_version) list."""
        return [(k.tbl_id.hex, k.effective_version) for k in self.key().keys]


class TableVersionPath(TablePath):
    """
    A TableVersionPath represents the sequence of TableVersions from a base table to a particular view:
    - for a base table: only includes that TableVersion
    - for a view: includes the TableVersion for the view and all its bases
    - multiple snapshots can reference the same TableVersion, but with different bases, which means that the
      graph of TableVersions is a DAG, not a tree (which is why we cannot embed the DAG into TableVersion directly)

    TableVersionPath contains all metadata needed to execute queries and updates against a particular version of a
    table/view.

    TableVersionPath supplies metadata needed for query construction (eg, column names), for which it uses a
    cached TableVersion instance.
    - when running inside a transaction, this instance is guaranteed to be validated
    - when running outside a transaction, we use an unvalidated instance in order to avoid repeated validation
      on every metadata-related method call (the instance won't stay validated, because TableVersionHandle.get()
      runs a local transaction, at the end of which the instance is again invalidated)
    - supplying metadata from an unvalidated instance is okay, because it needs to get revalidated anyway when a
      query actually runs (at which point there is a transaction context) - there is no guarantee that in between
      constructing a Query and executing it, the underlying table schema hasn't changed (eg, a concurrent process
      could have dropped a column referenced in the query).

    Thread-safe: all mutable state is in _local
    """

    tbl_version: TableVersionHandle
    base: TableVersionPath | None

    # Per-thread cache: dict containing
    # - cached_tbl_version: TableVersion
    # - origin_catalog: Catalog
    # - column_version_md: dict[QColumnId, ColumnVersionMd]
    _local: threading.local

    def __init__(self, tbl_version: TableVersionHandle, base: TableVersionPath | None = None):
        assert tbl_version is not None
        self.tbl_version = tbl_version
        self.base = base
        self._local = threading.local()
        self._local.cached_tbl_version = None
        self._local.origin_catalog = None
        self._local.column_version_md = None

    def __deepcopy__(self, memo: dict[int, object]) -> TableVersionPath:
        # Thread-safe and structurally immutable: callers can share a single instance.
        return self

    @classmethod
    def from_key(cls, key: TablePathKey) -> TableVersionPath:
        result: TableVersionPath | None = None
        for k in key.keys[::-1]:  # base first, so each new node wraps the previous as its base
            result = TableVersionPath(TableVersionHandle(k), base=result)
        assert result is not None
        return result

    @classmethod
    def from_schema_path(cls, path: schema.TableVersionPath) -> TableVersionPath:
        """Reconstruct from the persisted base_versions form (a flat (tbl_id_hex, effective_version) list)."""
        return cls.from_key(TablePathKey(tuple(TableVersionKey(UUID(s), v) for s, v in path)))

    def _cached_tv(self) -> TableVersion:
        """Return the validated cached TableVersion for the calling thread."""
        cat = get_runtime().catalog
        # getattr(), not attribute access: threads other than the originating one will have an empty _local
        cached: TableVersion | None = getattr(self._local, 'cached_tbl_version', None)
        origin_catalog: Catalog | None = getattr(self._local, 'origin_catalog', None)
        if origin_catalog is cat and cached is not None and (not get_runtime().in_xact or cached.is_validated):
            return cached

        with get_runtime().catalog.begin_xact(for_write=False, read_tbl_ids=[self.tbl_version.id]):
            new_tv = self.tbl_version.get()
        self._local.cached_tbl_version = new_tv
        self._local.origin_catalog = cat
        self._local.column_version_md = None

        return self._local.cached_tbl_version

    def clear_cached_md(self) -> None:
        self._local.cached_tbl_version = None
        self._local.origin_catalog = None
        self._local.column_version_md = None
        if self.base is not None:
            self.base.clear_cached_md()

    def _create_column_version_md(self, tv: TableVersion) -> dict[QColumnId, ColumnVersionMd]:
        # all physically reachable columns, keyed by qcolid
        effective_version = self.tbl_version.effective_version
        column_version_md: dict[QColumnId, ColumnVersionMd] = {}
        # own columns (all, incl. system) first, so they shadow same-named base columns in iteration order
        for col in tv.cols_by_id.values():
            col_md_obj = tv.tbl_md.column_md[col.id]
            schema_col = tv._schema_version_md.columns[col.id]
            qcolid = QColumnId(self.tbl_id, col.id)
            column_version_md[qcolid] = ColumnVersionMd(
                tbl_id=self.tbl_id,
                effective_version=effective_version,
                qcolid=qcolid,
                col_effective_version=effective_version,
                col_md=col_md_obj,
                schema_col=schema_col,
                is_iterator_col=col.is_iterator_col,
            )
        if self.base is not None:
            for base_col_md in self.base._cached_column_version_md().values():
                column_version_md[base_col_md.qcolid] = base_col_md.with_context(self.tbl_version.id, effective_version)
        return column_version_md

    def _cached_column_version_md(self) -> dict[QColumnId, ColumnVersionMd]:
        col_md: dict[QColumnId, ColumnVersionMd] | None = getattr(self._local, 'column_version_md', None)
        if col_md is None:
            self._local.column_version_md = self._create_column_version_md(self._cached_tv())
        return self._local.column_version_md

    @property
    def tbl_id(self) -> UUID:
        return self.tbl_version.id

    def version(self) -> int | None:
        if not self.is_versioned():
            return None
        return self._cached_tv().version

    def effective_version(self) -> int | None:
        return self.tbl_version.effective_version

    def schema_version(self) -> int:
        return self._cached_tv().schema_version

    def is_versioned(self) -> bool:
        return self._cached_tv().is_versioned

    def tbl_name(self) -> str:
        return self._cached_tv().name

    def is_snapshot(self) -> bool:
        return self.tbl_version.is_snapshot

    def is_view(self) -> bool:
        return self._cached_tv().is_view

    def is_component_view(self) -> bool:
        return self._cached_tv().is_component_view

    def is_mutable(self) -> bool:
        return self._cached_tv().is_mutable

    def is_insertable(self) -> bool:
        return self._cached_tv().is_insertable

    def comment(self) -> str:
        return self._cached_tv().comment

    def custom_metadata(self) -> Any:
        return copy.deepcopy(self._cached_tv().custom_metadata)

    def media_validation(self) -> MediaValidation:
        return self._cached_tv().media_validation

    def get_tbl_versions(self) -> list[TableVersionHandle]:
        """Return all tbl versions"""
        if self.base is None:
            return [self.tbl_version]
        return [self.tbl_version, *self.base.get_tbl_versions()]

    def tbl_versions(self) -> dict[UUID, TableVersion]:
        """The TableVersion instances along this path, keyed by table id."""
        return {h.id: h.get() for h in self.get_tbl_versions()}

    def get_bases(self) -> list[TableVersionHandle]:
        """Return all tbl versions"""
        if self.base is None:
            return []
        return self.base.get_tbl_versions()

    def find_tbl_version(self, id: UUID) -> TableVersionHandle | None:
        """Return the matching TableVersion in the chain of TableVersions, starting with this one"""
        if self.tbl_version.id == id:
            return self.tbl_version
        if self.base is None:
            return None
        return self.base.find_tbl_version(id)

    def columns(self) -> list[Column]:
        """Return all user columns visible in this tbl version path, including columns from bases"""
        tv = self._cached_tv()
        result = list(tv.cols_by_name.values())
        if self.base is not None and tv.include_base_columns:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend(c for c in base_cols if c.name not in tv.cols_by_name)
        return result

    def get_column_by_id(self, qcol_id: QColumnId) -> Column | None:
        tv = self._cached_tv()
        if qcol_id.tbl_id == self.tbl_version.id:
            return tv.cols_by_id.get(qcol_id.col_id)
        if self.base is not None:
            return self.base.get_column_by_id(qcol_id)
        return None

    def get_column(self, name: str) -> Column | None:
        """Return the column with the given name, or None if not found"""
        tv = self._cached_tv()
        col = tv.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None and tv.include_base_columns:
            return self.base.get_column(name)
        else:
            return None

    def has_column(self, qcolid: QColumnId) -> bool:
        return qcolid in self._cached_column_version_md()

    def column_md(self) -> list[ColumnVersionMd]:
        """Return metadata for all user columns visible by name in this path.

        Visibility honors include_base_columns at each level and name shadowing (a view's own column
        hides a same-named base column).
        """
        tv = self._cached_tv()
        # own user columns
        result = [
            col_md
            for col_md in self._cached_column_version_md().values()
            if col_md.qcolid.tbl_id == self.tbl_version.id and not col_md.is_system_col
        ]
        if self.base is not None and tv.include_base_columns:
            own_names = {col_md.name for col_md in result}
            ev = self.tbl_version.effective_version
            result.extend(
                base_col_md.with_context(self.tbl_version.id, ev)
                for base_col_md in self.base.column_md()
                if base_col_md.name not in own_names
            )
        return result

    def get_column_md(self, qcolid: QColumnId) -> ColumnVersionMd:
        """Return metadata for the column with the given qualified id (any physically reachable column)."""
        col_md = self._cached_column_version_md().get(qcolid)
        if col_md is None:
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Column {qcolid!r} not found')
        return col_md

    def get_column_md_by_name(self, name: str) -> ColumnVersionMd | None:
        """Return metadata for the user column visible under the given name, or None if not found."""
        return next((col_md for col_md in self.column_md() if col_md.name == name), None)

    def get_idx_md(self, qcolid: QColumnId, name: str | None, idx_class: type[IndexBase]) -> schema.IndexMd:
        tv = self._cached_tv()
        col = tv.cols_by_id.get(qcolid.col_id) if qcolid.tbl_id == self.tbl_version.id else None
        if col is None:
            if self.base is not None:
                return self.base.get_idx_md(qcolid, name, idx_class)
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Column {qcolid!r} not found')
        idx_info = tv.get_idx(col, name, idx_class)
        return tv._tbl_md.index_md[idx_info.id]

    def is_validate_on_read(self, col_md: ColumnVersionMd) -> bool:
        """Return whether a ColumnRef for this column should perform ON_READ media validation.

        Uses the column-level setting when present; falls back to this path's table-level default.
        """
        if not col_md.col_type.is_media_type():
            return False
        effective_mv = col_md.media_validation
        if effective_mv is None:
            effective_mv = self._cached_tv().media_validation
        return effective_mv == MediaValidation.ON_READ

    def as_dict(self) -> dict:
        return {
            'tbl_version': self.tbl_version.as_dict(),
            'base': self.base.as_dict() if self.base is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionPath:
        tbl_version = TableVersionHandle.from_dict(d['tbl_version'])
        base = TableVersionPath.from_dict(d['base']) if d['base'] is not None else None
        return cls(tbl_version, base)


class TableMdPath(TablePath):
    """
    TablePath backed by TableVersionMd. Does not provide access to catalog objects (Table/TableVersion/Column).

    Immutable.
    """

    md: TableVersionMd
    base: TableMdPath | None
    catalog_uri: str  # the hosted catalog this path belongs to ('' if not proxied); used to route proxy queries

    # All physically reachable columns, keyed by qcolid: own columns (incl. system) plus every base column,
    # regardless of include_base_columns or name shadowing. Name-based visibility is applied by column_md()/
    # get_column_md_by_name(); has_column()/get_column_md() resolve against this dict.
    _column_version_md: dict[QColumnId, ColumnVersionMd]

    def __init__(self, path: list[TableVersionMd], effective_versions: list[int | None], catalog_uri: str = ''):
        # effective_versions carries the version each element is pinned at (None = live), supplied by the
        # caller alongside the metadata; this mirrors TableVersionPath, where each node's TableVersionHandle
        # carries its own effective version. Use from_md() to derive it from an exported metadata list.
        assert len(path) > 0
        assert len(effective_versions) == len(path), (len(effective_versions), len(path))
        self.md = path[0]
        self.catalog_uri = catalog_uri
        self._effective_version = effective_versions[0]
        self.base = TableMdPath(path[1:], effective_versions[1:], catalog_uri) if len(path) > 1 else None
        self._column_version_md = {}

        num_iter_cols = (
            len((self.md.tbl_md.view_md.iterator_call or {}).get('outputs') or {})
            if self.md.tbl_md.view_md is not None
            else 0
        )
        tbl_id = UUID(self.md.tbl_md.tbl_id)
        effective_version = self._effective_version

        # live columns at this schema version (incl. system) first, so they shadow same-named base columns
        # in iteration order; dropped columns are absent from schema_version_md.columns, hence excluded
        for col_id, schema_col in self.md.schema_version_md.columns.items():
            qcolid = QColumnId(tbl_id, col_id)
            col_md = self.md.tbl_md.column_md[col_id]
            cvmd = ColumnVersionMd(
                tbl_id=tbl_id,
                effective_version=effective_version,
                qcolid=qcolid,
                col_effective_version=effective_version,
                col_md=col_md,
                schema_col=schema_col,
                is_iterator_col=col_id < num_iter_cols,
            )
            self._column_version_md[qcolid] = cvmd

        if self.base is not None:
            self._column_version_md.update(
                {
                    base_col_md.qcolid: base_col_md.with_context(tbl_id, effective_version)
                    for base_col_md in self.base._column_version_md.values()
                }
            )

    @classmethod
    def from_md(cls, md: list[TableVersionMd], catalog_uri: str = '') -> TableMdPath:
        """Build from an exported metadata list (leaf first).

        Per-element effective versions are read from the leaf's base_versions (the persisted ancestor
        pinning), mirroring Catalog._get_table: a snapshot is pinned at version 0, a live table/view is
        unpinned (None), and each ancestor takes the version recorded in base_versions.
        """
        leaf_view_md = md[0].tbl_md.view_md
        leaf_version = 0 if leaf_view_md is not None and leaf_view_md.is_snapshot else None
        effective_versions: list[int | None] = [leaf_version]
        if leaf_view_md is not None:
            effective_versions.extend(version for _, version in leaf_view_md.base_versions)
        return cls(md, effective_versions, catalog_uri)

    def __deepcopy__(self, memo: dict[int, object]) -> TableMdPath:
        return self

    @property
    def tbl_id(self) -> UUID:
        return UUID(self.md.tbl_md.tbl_id)

    def tbl_name(self) -> str:
        return self.md.tbl_md.name

    def version(self) -> int | None:
        return self.md.version_md.version if self.md.tbl_md.is_versioned else None

    def effective_version(self) -> int | None:
        return self._effective_version

    def is_snapshot(self) -> bool:
        return self.md.tbl_md.is_snapshot

    def is_pure_snapshot(self) -> bool:
        return self.md.tbl_md.is_pure_snapshot

    def is_view(self) -> bool:
        return self.md.tbl_md.view_md is not None

    def is_mutable(self) -> bool:
        return self.md.tbl_md.is_mutable

    def is_versioned(self) -> bool:
        return self.md.tbl_md.is_versioned

    def has_column(self, qcolid: QColumnId) -> bool:
        return qcolid in self._column_version_md

    def column_md(self) -> list[ColumnVersionMd]:
        """Return metadata for all user columns visible by name in this path.

        Visibility honors include_base_columns at each level and name shadowing (a view's own column
        hides a same-named base column).
        """
        tbl_id = self.tbl_id
        # own user columns
        result = [
            col_md
            for col_md in self._column_version_md.values()
            if col_md.qcolid.tbl_id == tbl_id and not col_md.is_system_col
        ]
        include_base = (
            self.base is not None and self.md.tbl_md.view_md is not None and self.md.tbl_md.view_md.include_base_columns
        )
        if include_base:
            own_names = {col_md.name for col_md in result}
            effective_version = self.effective_version()
            result.extend(
                base_col_md.with_context(tbl_id, effective_version)
                for base_col_md in self.base.column_md()
                if base_col_md.name not in own_names
            )
        return result

    def get_column_md(self, qcolid: QColumnId) -> ColumnVersionMd:
        """Return metadata for the column with the given qualified id (any physically reachable column)."""
        result = self._column_version_md.get(qcolid)
        if result is None:
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Column {qcolid!r} not found')
        return result

    def get_column_md_by_name(self, name: str) -> ColumnVersionMd | None:
        """Return metadata for the user column visible under the given name, or None if not found."""
        return next((col_md for col_md in self.column_md() if col_md.name == name), None)

    def get_idx_md(self, qcolid: QColumnId, name: str | None, idx_class: type[IndexBase]) -> schema.IndexMd:
        schema_version = self.md.schema_version_md.schema_version
        candidates = [
            idx_md
            for idx_md in self.md.tbl_md.index_md.values()
            if (
                idx_md.indexed_col_id == qcolid.col_id
                and UUID(idx_md.indexed_col_tbl_id) == qcolid.tbl_id
                and idx_md.schema_version_add <= schema_version
                and (idx_md.schema_version_drop is None or idx_md.schema_version_drop > schema_version)
                and issubclass(resolve_symbol(idx_md.class_fqn), idx_class)  # type: ignore[arg-type]
            )
        ]
        if name is not None:
            candidates = [idx_md for idx_md in candidates if idx_md.name == name]
        if len(candidates) == 1 or (len(candidates) > 1 and name is not None):
            return candidates[0]
        if len(candidates) > 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Column {qcolid!r} has multiple {idx_class.display_name()} indices; specify idx_name instead',
            )
        if self.base is not None:
            return self.base.get_idx_md(qcolid, name, idx_class)
        raise excs.NotFoundError(
            excs.ErrorCode.INDEX_NOT_FOUND, f'No {idx_class.display_name()} index found for column {qcolid!r}'
        )
