from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pixeltable import exceptions as excs

from .table_version import TableVersion, TableVersionKey

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class TableVersionHandle:
    """
    Indirection mechanism for TableVersion instances, which get resolved against the catalog at runtime.

    See the TableVersion docstring for details on the semantics of `effective_version` and `anchor_tbl_id`.
    """

    key: TableVersionKey
    _tbl_version: TableVersion | None

    def __init__(self, key: TableVersionKey, *, tbl_version: TableVersion | None = None):
        self.key = key
        self._tbl_version = tbl_version

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
        from .catalog import Catalog

        cat = Catalog.get()
        if self._tbl_version is None or not self._tbl_version.is_validated:
            if self.effective_version is not None and self._tbl_version is not None:
                # this is a snapshot version; we need to make sure we refer to the instance cached
                # in Catalog, in order to avoid mixing sa_tbl instances in the same transaction
                # (which will lead to duplicates in the From clause generated in SqlNode.create_from_clause())
                assert self.key in cat._tbl_versions
                self._tbl_version = cat._tbl_versions[self.key]
                self._tbl_version.is_validated = True
            else:
                self._tbl_version = Catalog.get().get_tbl_version(self.key)
                assert self._tbl_version.key == self.key
        if self.effective_version is None:
            tvs = list(Catalog.get()._tbl_versions.values())
            assert self._tbl_version in tvs, self._tbl_version
        return self._tbl_version

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
            raise excs.Error(
                f'Column was dropped (no record for column ID {self.col_id} in table '
                f'{self.tbl_version.get().versioned_name!r}; it was dropped in table version {schema_version_drop})'
            )
        return self.tbl_version.get().cols_by_id[self.col_id]

    def as_dict(self) -> dict:
        return {'tbl_version': self.tbl_version.as_dict(), 'col_id': self.col_id}

    @classmethod
    def from_dict(cls, d: dict) -> ColumnHandle:
        return cls(tbl_version=TableVersionHandle.from_dict(d['tbl_version']), col_id=d['col_id'])
