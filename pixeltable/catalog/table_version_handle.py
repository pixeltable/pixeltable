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
