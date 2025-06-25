from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from pixeltable import exceptions as excs

from .table_version import TableVersion

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class TableVersionHandle:
    """
    Indirection mechanism for TableVersion instances, which get resolved against the catalog at runtime.
    """

    id: UUID
    effective_version: Optional[int]
    _tbl_version: Optional[TableVersion]

    def __init__(self, tbl_id: UUID, effective_version: Optional[int], tbl_version: Optional[TableVersion] = None):
        self.id = tbl_id
        self.effective_version = effective_version
        self._tbl_version = tbl_version

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableVersionHandle):
            return False
        return self.id == other.id and self.effective_version == other.effective_version

    def __hash__(self) -> int:
        return hash((self.id, self.effective_version))

    @property
    def is_snapshot(self) -> bool:
        return self.effective_version is not None

    @classmethod
    def create(cls, tbl_version: TableVersion) -> TableVersionHandle:
        return cls(tbl_version.id, tbl_version.effective_version, tbl_version)

    def get(self) -> TableVersion:
        from .catalog import Catalog

        cat = Catalog.get()
        if self._tbl_version is None or not self._tbl_version.is_validated:
            if self.effective_version is not None and self._tbl_version is not None:
                # this is a snapshot version; we need to make sure we refer to the instance cached
                # in Catalog, in order to avoid mixing sa_tbl instances in the same transaction
                # (which will lead to duplicates in the From clause generated in SqlNode.create_from_clause())
                assert (self.id, self.effective_version) in cat._tbl_versions
                self._tbl_version = cat._tbl_versions[self.id, self.effective_version]
                self._tbl_version.is_validated = True
            else:
                self._tbl_version = Catalog.get().get_tbl_version(self.id, self.effective_version)
        if self.effective_version is None:
            tvs = list(Catalog.get()._tbl_versions.values())
            assert self._tbl_version in tvs
        return self._tbl_version

    def as_dict(self) -> dict:
        return {'id': str(self.id), 'effective_version': self.effective_version}

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionHandle:
        return cls(UUID(d['id']), d['effective_version'])


@dataclass(frozen=True)
class ColumnHandle:
    tbl_version: TableVersionHandle
    col_id: int

    def get(self) -> 'Column':
        if self.col_id not in self.tbl_version.get().cols_by_id:
            schema_version_drop = self.tbl_version.get()._tbl_md.column_md[self.col_id].schema_version_drop
            raise excs.Error(
                f'Column has been dropped (no record for column ID {self.col_id} in table '
                f'{self.tbl_version.get().versioned_name!r}; it was dropped in table version {schema_version_drop})'
            )
        return self.tbl_version.get().cols_by_id[self.col_id]

    def as_dict(self) -> dict:
        return {'tbl_version': self.tbl_version.as_dict(), 'col_id': self.col_id}

    @classmethod
    def from_dict(cls, d: dict) -> ColumnHandle:
        return cls(tbl_version=TableVersionHandle.from_dict(d['tbl_version']), col_id=d['col_id'])
