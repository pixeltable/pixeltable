from __future__ import annotations

import logging
import threading
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
    Indirection mechanism for TableVersion instances, which get resolved against the catalog at runtime.

    Not thread-safe.
    """

    key: TableVersionKey

    # cache of the resolved TableVersion; needs to be cleared whenever we cross a transaction
    # or Catalog instance boundary
    _tbl_version: TableVersion | None

    # id of the constructing thread; used to guard against cross-thread access
    _origin_thread_id: int

    # Catalog instance against which this handle was last resolved
    _origin_catalog: Catalog

    def __init__(self, key: TableVersionKey, *, tbl_version: TableVersion | None = None):
        self.key = key
        self._tbl_version = tbl_version
        self._origin_thread_id = threading.get_ident()
        self._origin_catalog = get_runtime().catalog

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> TableVersionHandle:
        # reset _tbl_version, we might end up in a different thread
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
        # guard against incorrect cross-thread access; inherited-context threads are allowed
        assert self._origin_thread_id == threading.get_ident() or get_runtime().context_inherited

        cat = get_runtime().catalog
        cached = self._tbl_version
        needs_refresh = self._origin_catalog is not cat or cached is None or not cached.is_validated
        if not needs_refresh:
            return cached

        if self.effective_version is not None and cached is not None:
            # This is a snapshot version. We need to reuse the instance cached in Catalog, to avoid mixing sa_tbl
            # instances in the same transaction (which would produce duplicates in the From clause).
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
    """
    Indirection mechanism for Column instances, which get resolved against the catalog at runtime.

    Not thread-safe.
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
