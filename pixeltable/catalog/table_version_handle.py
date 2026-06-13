from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.runtime import get_runtime

from .table_version import TableVersion, TableVersionKey

if TYPE_CHECKING:
    from pixeltable.catalog import Catalog, Column


class TableVersionHandle:
    """
    Indirection mechanism for TableVersion instances, which get resolved against the catalog at runtime.

    Thread-safe: all mutable state is stored behind _local
    """

    key: TableVersionKey

    # Per-thread cache of the resolved TableVersion plus the Catalog instance it came from.
    # threading.local gives each thread its own attribute namespace; reads on the hot path are
    # plain attribute lookups.
    _local: threading.local

    def __init__(self, key: TableVersionKey, *, tbl_version: TableVersion | None = None):
        self.key = key
        self._local = threading.local()
        self._local.tbl_version = tbl_version
        self._local.origin_catalog = get_runtime().catalog

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> TableVersionHandle:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableVersionHandle):
            return False
        return self.id == other.id and self.effective_version == other.effective_version

    def __hash__(self) -> int:
        return hash((self.id, self.effective_version))

    def __repr__(self) -> str:
        return f'TableVersionHandle(id={self.id!r}, effective_version={self.effective_version})'

    @property
    def id(self) -> UUID:
        return self.key.tbl_id

    @property
    def effective_version(self) -> int | None:
        return self.key.effective_version

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    def get(self) -> TableVersion:
        cat = get_runtime().catalog
        # getattr(), not attribute access: threads other than the originating one will have an empty _local
        cached: TableVersion | None = getattr(self._local, 'tbl_version', None)
        origin_catalog: Catalog | None = getattr(self._local, 'origin_catalog', None)
        if origin_catalog is cat and cached is not None and cached.is_validated:
            return cached

        if self.effective_version is not None and cached is not None and origin_catalog is cat:
            # Snapshot version, same catalog as before: reuse the instance cached in Catalog to avoid mixing sa_tbl
            # instances in the same transaction (which would produce duplicates in the From clause).
            assert self.key in cat._tbl_versions
            new_tv = cat._tbl_versions[self.key]
            new_tv.is_validated = True
        else:
            # either no cached instance on this thread, or the catalog changed
            new_tv = cat.get_tbl_version(self.key)
            assert new_tv.key == self.key
        self._local.tbl_version = new_tv
        self._local.origin_catalog = cat
        return new_tv

    def as_dict(self) -> dict:
        return self.key.as_dict()

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionHandle:
        return cls(TableVersionKey.from_dict(d))


@dataclass(frozen=True)
class ColumnHandle:
    """
    Indirection mechanism for Column instances, which get resolved against the catalog at runtime.

    Thread-safe: stateless beyond the underlying TableVersionHandle.
    """

    tbl_version: TableVersionHandle
    col_id: int

    def get(self) -> 'Column':
        tv = self.tbl_version.get()
        if self.col_id not in tv.cols_by_id:
            schema_version_drop = tv._tbl_md.column_md[self.col_id].schema_version_drop
            raise excs.NotFoundError(
                excs.ErrorCode.COLUMN_NOT_FOUND,
                f'Column was dropped (no record for column ID {self.col_id} in table '
                f'{tv.versioned_name!r}; it was dropped in table version {schema_version_drop})',
            )
        return tv.cols_by_id[self.col_id]

    def as_dict(self) -> dict:
        return {'tbl_version': self.tbl_version.as_dict(), 'col_id': self.col_id}

    @classmethod
    def from_dict(cls, d: dict) -> ColumnHandle:
        return cls(tbl_version=TableVersionHandle.from_dict(d['tbl_version']), col_id=d['col_id'])
