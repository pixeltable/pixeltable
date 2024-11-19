from __future__ import annotations

import logging
from typing import Optional, Union
from uuid import UUID

import pixeltable as pxt
from pixeltable import exprs

from .column import Column
from .table_version import TableVersion

_logger = logging.getLogger('pixeltable')


class TableVersionPath:
    """
    A TableVersionPath represents the sequence of TableVersions from a base table to a particular view:
    - for a base table: only includes that TableVersion
    - for a view: includes the TableVersion for the view and all its bases
    - multiple snapshots can reference the same TableVersion, but with different bases, which means that the
      graph of TableVersions is a DAG, not a tree (which is why we cannot embed the DAG into TableVersion directly)

    TableVersionPath contains all metadata needed to execute queries and updates against a particular version of a
    table/view.
    """

    def __init__(self, tbl_version: TableVersion, base: Optional[TableVersionPath] = None):
        assert tbl_version is not None
        self.tbl_version = tbl_version
        self.base = base

    def tbl_id(self) -> UUID:
        """Return the id of the table/view that this path represents"""
        return self.tbl_version.id

    def version(self) -> int:
        """Return the version of the table/view that this path represents"""
        return self.tbl_version.version

    def tbl_name(self) -> str:
        """Return the name of the table/view that this path represents"""
        return self.tbl_version.name

    def path_len(self) -> int:
        """Return the length of the path"""
        return 1 if self.base is None else 1 + self.base.path_len()

    def is_snapshot(self) -> bool:
        """Return True if this is a path of snapshot versions"""
        if not self.tbl_version.is_snapshot:
            return False
        return self.base.is_snapshot() if self.base is not None else True

    def is_view(self) -> bool:
        return self.tbl_version.is_view()

    def is_component_view(self) -> bool:
        return self.tbl_version.is_component_view()

    def is_insertable(self) -> bool:
        return self.tbl_version.is_insertable()

    def get_tbl_versions(self) -> list[TableVersion]:
        """Return all tbl versions"""
        if self.base is None:
            return [self.tbl_version]
        return [self.tbl_version] + self.base.get_tbl_versions()

    def get_bases(self) -> list[TableVersion]:
        """Return all tbl versions"""
        if self.base is None:
            return []
        return self.base.get_tbl_versions()

    def find_tbl_version(self, id: UUID) -> Optional[TableVersion]:
        """Return the matching TableVersion in the chain of TableVersions, starting with this one"""
        if self.tbl_version.id == id:
            return self.tbl_version
        if self.base is None:
            return None
        return self.base.find_tbl_version(id)

    def get_column_ref(self, col_name: str) -> exprs.ColumnRef:
        """Return a ColumnRef for the given column name."""
        from pixeltable.exprs import ColumnRef
        if col_name not in self.tbl_version.cols_by_name:
            if self.base is None:
                raise AttributeError(f'Column {col_name} unknown')
            return self.base.get_column_ref(col_name)
        col = self.tbl_version.cols_by_name[col_name]
        return ColumnRef(col)

    def columns(self) -> list[Column]:
        """Return all user columns visible in this tbl version path, including columns from bases"""
        result = list(self.tbl_version.cols_by_name.values())
        if self.base is not None:
            base_cols = self.base.columns()
            # we only include base columns that don't conflict with one of our column names
            result.extend(c for c in base_cols if c.name not in self.tbl_version.cols_by_name)
        return result

    def cols_by_name(self) -> dict[str, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.name: col for col in cols}

    def cols_by_id(self) -> dict[int, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.id: col for col in cols}

    def get_column(self, name: str, include_bases: bool = True) -> Optional[Column]:
        """Return the column with the given name, or None if not found"""
        col = self.tbl_version.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None and include_bases:
            return self.base.get_column(name)
        else:
            return None

    def has_column(self, col: Column, include_bases: bool = True) -> bool:
        """Return True if this table has the given column.
        """
        assert col.tbl is not None
        if col.tbl.id == self.tbl_version.id and col.tbl.effective_version == self.tbl_version.effective_version \
                and col.id in self.tbl_version.cols_by_id:
            # the column is visible in this table version
            return True
        elif self.base is not None and include_bases:
            return self.base.has_column(col)
        else:
            return False

    def as_dict(self) -> dict:
        return {
            'tbl_version': self.tbl_version.as_dict(),
            'base': self.base.as_dict() if self.base is not None else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionPath:
        tbl_version = TableVersion.from_dict(d['tbl_version'])
        base = TableVersionPath.from_dict(d['base']) if d['base'] is not None else None
        return cls(tbl_version, base)
