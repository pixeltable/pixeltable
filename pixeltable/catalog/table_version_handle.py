from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.runtime import get_runtime

from .table_version import TableVersion, TableVersionKey

if TYPE_CHECKING:
    from pixeltable.catalog import Column

    from .catalog import Catalog

_logger = logging.getLogger('pixeltable')


class TableVersionHandle:
    """
    Indirection mechanism for TableVersion instances, which get resolved against the (thread-local) catalog at runtime.
    Each get() call needs to resolve against the current thread's catalog in order to avoid picking up catalog
    elements (eg, Column.sa_column) created in a different thread.

    See the TableVersion docstring for details on the semantics of effective_version and anchor_tbl_id.
    """

    key: TableVersionKey
    # Per-(thread, xact) cache of the resolved TableVersion. Reset on __deepcopy__ when the
    # handle crosses a thread/transaction boundary (e.g., a Query template captured at
    # module-import time being executed in a worker thread, or a Query held across xacts in
    # the face of a schema change from another process). Re-validation happens on the next
    # get() through the catalog's is_validated machinery.
    _tbl_version: TableVersion | None
    # The Catalog instance this handle was last resolved against. Set in __init__ from
    # `get_runtime().catalog`, re-set by __deepcopy__, and updated by get() whenever it
    # detects a mismatch with the calling thread's catalog (which can happen across worker
    # threads or after reset_runtime). On mismatch get() drops the cached TableVersion and
    # re-resolves through the current catalog, so consumers always observe metadata from the
    # catalog they're running against.
    _origin_catalog: Catalog

    def __init__(self, key: TableVersionKey, *, tbl_version: TableVersion | None = None):
        self.key = key
        self._tbl_version = tbl_version
        self._origin_catalog = get_runtime().catalog

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> TableVersionHandle:
        # Returning a fresh handle resets _tbl_version (so the next get() re-resolves via the
        # current thread's catalog) and re-tags _origin_catalog to the calling thread's
        # catalog via __init__.
        return TableVersionHandle(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableVersionHandle):
            return False
        return self.id == other.id and self.effective_version == other.effective_version

    def __hash__(self) -> int:
        return hash((self.id, self.effective_version))

    def __repr__(self) -> str:
        return (
            f'TableVersionHandle(id={self.id!r}, effective_version={self.effective_version}, '
            f'anchor_tbl_id={self.anchor_tbl_id})'
        )

    @property
    def id(self) -> UUID:
        return self.key.tbl_id

    @property
    def effective_version(self) -> int | None:
        return self.key.effective_version

    @property
    def anchor_tbl_id(self) -> UUID | None:
        return self.key.anchor_tbl_id

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    def get(self) -> TableVersion:
        # Snapshot the cache once, decide whether to refresh, and atomically replace at the
        # end — never reset to None mid-method, so that concurrent readers (this handle can
        # be shared across threads via Table._tbl_version_path or Column.tbl_handle) always
        # observe a complete TableVersion. The mismatch case happens when the handle is used
        # in a different catalog context than it was built against (worker thread with its
        # own Catalog, or after reset_runtime); we re-resolve so sa_tbl etc. come from the
        # current catalog's metadata.
        cat = get_runtime().catalog
        cached = self._tbl_version
        needs_refresh = self._origin_catalog is not cat or cached is None or not cached.is_validated
        if not needs_refresh:
            return cached

        if self.effective_version is not None and cached is not None:
            # Snapshot version; refer to the instance cached in Catalog, in order to avoid
            # mixing sa_tbl instances in the same transaction (which would generate
            # duplicates in the From clause produced by SqlNode.create_from_clause()).
            assert self.key in cat._tbl_versions
            new_tv = cat._tbl_versions[self.key]
            new_tv.is_validated = True
        else:
            new_tv = cat.get_tbl_version(self.key)
            assert new_tv.key == self.key
        self._tbl_version = new_tv
        self._origin_catalog = cat
        return new_tv

    def as_dict(self) -> dict:
        return self.key.as_dict()

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionHandle:
        return cls(TableVersionKey.from_dict(d))


@dataclass(frozen=True)
class ColumnHandle:
    tbl_version: TableVersionHandle
    col_id: int

    def get(self) -> 'Column':
        if self.col_id not in self.tbl_version.get().cols_by_id:
            schema_version_drop = self.tbl_version.get()._tbl_md.column_md[self.col_id].schema_version_drop
            raise excs.NotFoundError(
                excs.ErrorCode.COLUMN_NOT_FOUND,
                f'Column was dropped (no record for column ID {self.col_id} in table '
                f'{self.tbl_version.get().versioned_name!r}; it was dropped in table version {schema_version_drop})',
            )
        return self.tbl_version.get().cols_by_id[self.col_id]

    def as_dict(self) -> dict:
        return {'tbl_version': self.tbl_version.as_dict(), 'col_id': self.col_id}

    @classmethod
    def from_dict(cls, d: dict) -> ColumnHandle:
        return cls(tbl_version=TableVersionHandle.from_dict(d['tbl_version']), col_id=d['col_id'])
