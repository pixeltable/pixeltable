from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.runtime import get_runtime

from .table_version import TableVersion, TableVersionKey

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class TableVersionHandle:
    """
    Indirection mechanism for TableVersion instances, which get resolved against the catalog at runtime.

    Thread-safe: no mutable state.
    """

    key: TableVersionKey

    def __init__(self, key: TableVersionKey):
        self.key = key

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
        """
        Resolve this handle to the TableVersion instance from the catalog.

        Inside a transaction, always defers to the catalog, which returns the version consistent with the
        transaction's view. Outside a transaction, a snapshot version can be served from the catalog's cache without
        starting a transaction; a live version (effective_version is None) is always fetched from the database.
        """
        cat = get_runtime().catalog
        if get_runtime().in_xact:
            return cat.get_tbl_version(self.key)

        if self.effective_version is not None:
            cached_tv = cat._tbl_versions.get(self.key)
            if cached_tv is not None:
                return cached_tv

        return cat.get_tbl_version(self.key)

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
