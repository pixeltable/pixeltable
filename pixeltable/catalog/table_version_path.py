from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.metadata import schema

from .column import Column
from .globals import MediaValidation, QColumnId
from .table_version import TableVersion, TableVersionKey
from .table_version_handle import TableVersionHandle

if TYPE_CHECKING:
    from .catalog import Catalog


class TableVersionPath:
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
    """

    tbl_version: TableVersionHandle
    base: TableVersionPath | None
    # Per-(thread, xact) cache of the resolved TableVersion. Reset on __deepcopy__ when the
    # TVP crosses a thread boundary (e.g., a Query template captured at module-import time
    # being executed in a worker thread, or a Query held across xacts in the face of a
    # schema change from another process). Re-validation happens on the next
    # tbl_version.get() through the catalog's is_validated machinery.
    _cached_tbl_version: TableVersion | None
    # The thread that constructed (or most recently deepcopied) this TVP. Tables hold a TVP
    # via Table._tbl_version_path; cross-thread access through a Table is rejected at
    # refresh_cached_md() with a clear error pointing the user at pxt.get_table(). Queries,
    # by contrast, defensively deepcopy at collect-time, so the per-Query TVPs end up tagged
    # to the executing thread.
    _origin_thread_id: int
    # The Catalog instance this path was last resolved against. Used for the same-thread
    # cross-Catalog case (reset_runtime in tests): we self-heal by dropping the cache and
    # re-resolving via the new Catalog. See the corresponding comment on
    # TableVersionHandle._origin_catalog.
    _origin_catalog: Catalog

    def __init__(self, tbl_version: TableVersionHandle, base: TableVersionPath | None = None):
        from pixeltable.runtime import get_runtime

        assert tbl_version is not None
        self.tbl_version = tbl_version
        self.base = base
        self._cached_tbl_version = None
        self._origin_thread_id = threading.get_ident()
        self._origin_catalog = get_runtime().catalog

        if self.base is not None and tbl_version.anchor_tbl_id is not None:
            self.base = self.base.anchor_to(tbl_version.anchor_tbl_id)

    def __deepcopy__(self, memo: dict[int, object]) -> TableVersionPath:
        # Reset _cached_tbl_version and recurse into tbl_version (TVH) and base (TVP). Both
        # have their own __deepcopy__ that reset their respective per-(thread, xact) caches.
        # See the comment on _cached_tbl_version above for the invariant.
        result = TableVersionPath(
            tbl_version=copy.deepcopy(self.tbl_version, memo),
            base=copy.deepcopy(self.base, memo) if self.base is not None else None,
        )
        memo[id(self)] = result
        return result

    @classmethod
    def from_md(cls, path: schema.TableVersionPath) -> TableVersionPath:
        assert len(path) > 0
        result: TableVersionPath | None = None
        for tbl_id_str, effective_version in path[::-1]:
            tbl_id = UUID(tbl_id_str)
            key = TableVersionKey(tbl_id, effective_version, None)
            result = TableVersionPath(TableVersionHandle(key), base=result)
        return result

    def as_md(self) -> schema.TableVersionPath:
        result = [(self.tbl_version.id.hex, self.tbl_version.effective_version)]
        if self.base is not None:
            result.extend(self.base.as_md())
        return result

    def refresh_cached_md(self) -> None:
        from pixeltable.runtime import get_runtime

        # Reject cross-thread access. Tables hold a TVP and are not thread-safe: their
        # cached metadata is a per-instance, per-thread cache. Concurrent threads sharing a
        # Table would race on this cache, and even if races are made benign via atomic
        # replace, the cache pings between catalogs and forces redundant metadata loads.
        # Cross-thread sharing is supposed to go through Query.collect() (which deepcopies
        # and produces a thread-local TVP), or through a per-thread pxt.get_table().
        if self._origin_thread_id != threading.get_ident():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                f'Table {self.tbl_version.id} was accessed from a thread other than the one that constructed it. '
                f'Tables are not thread-safe; call pxt.get_table(...) on this thread to obtain a thread-local '
                f'instance.',
            )
        # Same thread but different Catalog (e.g., after reset_runtime): drop the cache and
        # re-resolve. When we're inside a transaction, also reload if our cached TV's
        # is_validated flipped (mixing stale and current metadata produces multiple
        # sqlalchemy Table instances for the same underlying table and corrupts From
        # clauses). Snapshot the cache, then atomically replace - never set to None
        # mid-method.
        cat = get_runtime().catalog
        cached = self._cached_tbl_version
        needs_refresh = (
            self._origin_catalog is not cat or cached is None or (get_runtime().in_xact and not cached.is_validated)
        )
        if not needs_refresh:
            return

        with get_runtime().catalog.begin_xact(for_write=False, read_tbl_ids=[self.tbl_version.id]):
            new_tv = self.tbl_version.get()
        self._cached_tbl_version = new_tv
        self._origin_catalog = cat

    def anchor_to(self, anchor_tbl_id: UUID | None) -> TableVersionPath:
        """
        Return a new TableVersionPath with all of its non-snapshot TableVersions pointing to the given anchor_tbl_id.
        (This will clear the existing anchor_tbl_id in the case anchor_tbl_id=None.)
        """
        if self.tbl_version.effective_version is not None:
            return self

        return TableVersionPath(
            TableVersionHandle(TableVersionKey(self.tbl_version.id, None, anchor_tbl_id)),
            base=self.base.anchor_to(anchor_tbl_id) if self.base is not None else None,
        )

    def clear_cached_md(self) -> None:
        self._cached_tbl_version = None
        if self.base is not None:
            self.base.clear_cached_md()

    @property
    def tbl_id(self) -> UUID:
        """Return the id of the table/view that this path represents"""
        return self.tbl_version.id

    def version(self) -> int | None:
        """Return the version of the table/view that this path represents"""
        if not self.is_versioned():
            return None
        self.refresh_cached_md()
        return self._cached_tbl_version.version

    def schema_version(self) -> int:
        """Return the version of the table/view that this path represents"""
        self.refresh_cached_md()
        return self._cached_tbl_version.schema_version

    def is_versioned(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_versioned

    def tbl_name(self) -> str:
        """Return the name of the table/view that this path represents"""
        self.refresh_cached_md()
        return self._cached_tbl_version.name

    def path_len(self) -> int:
        """Return the length of the path"""
        return 1 if self.base is None else 1 + self.base.path_len()

    def is_snapshot(self) -> bool:
        """Return True if this is a path of snapshot versions"""
        return self.tbl_version.is_snapshot

    def is_view(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_view

    def is_component_view(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_component_view

    def is_replica(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_replica

    def is_mutable(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_mutable

    def is_insertable(self) -> bool:
        self.refresh_cached_md()
        return self._cached_tbl_version.is_insertable

    def comment(self) -> str:
        self.refresh_cached_md()
        return self._cached_tbl_version.comment

    def custom_metadata(self) -> Any:
        self.refresh_cached_md()
        return copy.deepcopy(self._cached_tbl_version.custom_metadata)

    def media_validation(self) -> MediaValidation:
        self.refresh_cached_md()
        return self._cached_tbl_version.media_validation

    def get_tbl_versions(self) -> list[TableVersionHandle]:
        """Return all tbl versions"""
        if self.base is None:
            return [self.tbl_version]
        return [self.tbl_version, *self.base.get_tbl_versions()]

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
        self.refresh_cached_md()
        result = list(self._cached_tbl_version.cols_by_name.values())
        if self.base is not None and self._cached_tbl_version.include_base_columns:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend(c for c in base_cols if c.name not in self._cached_tbl_version.cols_by_name)
        return result

    def get_column_by_qid(self, qcol_id: QColumnId) -> Column | None:
        return next(
            (col for col in self.columns() if col.id == qcol_id.col_id and col.tbl_handle.id == qcol_id.tbl_id), None
        )

    def get_column(self, name: str) -> Column | None:
        """Return the column with the given name, or None if not found"""
        self.refresh_cached_md()
        col = self._cached_tbl_version.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None and self._cached_tbl_version.include_base_columns:
            return self.base.get_column(name)
        else:
            return None

    def has_column(self, col: Column) -> bool:
        """Return True if this table has the given column."""
        assert col.get_tbl() is not None
        self.refresh_cached_md()

        if (
            col.get_tbl().id == self.tbl_version.id
            and col.get_tbl().effective_version == self.tbl_version.effective_version
            and col.id in self._cached_tbl_version.cols_by_id
        ):
            # the column is visible in this table version
            return True
        elif self.base is not None:
            return self.base.has_column(col)
        else:
            return False

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
