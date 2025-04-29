from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from .table_version import TableVersion

if TYPE_CHECKING:
    pass

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

    @classmethod
    def create(cls, tbl_version: TableVersion) -> TableVersionHandle:
        return cls(tbl_version.id, tbl_version.effective_version, tbl_version)

    def get(self) -> TableVersion:
        from .catalog import Catalog

        if self._tbl_version is None or not self._tbl_version.is_validated:
            self._tbl_version = Catalog.get().get_tbl_version(self.id, self.effective_version)
            #print(f'self.tbl_version.isvalidated={self._tbl_version.is_validated}')
        return self._tbl_version

    def as_dict(self) -> dict:
        return {'id': str(self.id), 'effective_version': self.effective_version}

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionHandle:
        return cls(UUID(d['id']), d['effective_version'])
