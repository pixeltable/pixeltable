from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from pixeltable.env import Env
from pixeltable.metadata import schema

from .column import Column
from .globals import MediaValidation
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle

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

    TableVersionPath supplies metadata needed for query construction (eg, column names), for which it uses a
    cached TableVersion instance.
    - when running inside a transaction, this instance is guaranteed to be validated
    - when running outside a transaction, we use an unvalidated instance in order to avoid repeated validation
      on every metadata-related method call (the instance won't stay validated, because TableVersionHandle.get()
      runs a local transaction, at the end of which the instance is again invalidated)
    - supplying metadata from an unvalidated instance is okay, because it needs to get revalidated anyway when a
      query actually runs (at which point there is a transaction context) - there is no guarantee that in between
      constructing a DataFrame and executing it, the underlying table schema hasn't changed (eg, a concurrent process
      could have dropped a column referenced in the query).
    """

    tbl_version: TableVersionHandle
    base: Optional[TableVersionPath]
    _cached_tbl_version: Optional[TableVersion]

    def __init__(self, tbl_version: TableVersionHandle, base: Optional[TableVersionPath] = None):
        assert tbl_version is not None
        self.tbl_version = tbl_version
        self.base = base
        self._cached_tbl_version = None

    @classmethod
    def from_md(cls, path: schema.TableVersionPath) -> TableVersionPath:
        assert len(path) > 0
        result: Optional[TableVersionPath] = None
        for tbl_id_str, effective_version in path[::-1]:
            tbl_id = UUID(tbl_id_str)
            result = TableVersionPath(TableVersionHandle(tbl_id, effective_version), base=result)
        return result

    def as_md(self) -> schema.TableVersionPath:
        result = [(self.tbl_version.id.hex, self.tbl_version.effective_version)]
        if self.base is not None:
            result.extend(self.base.as_md())
        return result

    def refresh_cached_md(self) -> None:
        from pixeltable.catalog import Catalog

        if Env.get().in_xact:
            # when we're running inside a transaction, we need to make sure to supply current metadata;
            # mixing stale metadata with current metadata leads to query construction failures
            # (multiple sqlalchemy Table instances for the same underlying table create corrupted From clauses)
            if self._cached_tbl_version is not None and self._cached_tbl_version.is_validated:
                # nothing to refresh
                return
        elif self._cached_tbl_version is not None:
            return

        with Catalog.get().begin_xact(for_write=False):
            self._cached_tbl_version = self.tbl_version.get()

    def clear_cached_md(self) -> None:
        self._cached_tbl_version = None
        if self.base is not None:
            self.base.clear_cached_md()

    @property
    def tbl_id(self) -> UUID:
        """Return the id of the table/view that this path represents"""
        return self.tbl_version.id

    def version(self) -> int:
        """Return the version of the table/view that this path represents"""
        self.refresh_cached_md()
        return self._cached_tbl_version.version

    def schema_version(self) -> int:
        """Return the version of the table/view that this path represents"""
        self.refresh_cached_md()
        return self._cached_tbl_version.schema_version

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

    def num_retained_versions(self) -> int:
        self.refresh_cached_md()
        return self._cached_tbl_version.num_retained_versions

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

    def find_tbl_version(self, id: UUID) -> Optional[TableVersionHandle]:
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

    def cols_by_name(self) -> dict[str, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.name: col for col in cols}

    def cols_by_id(self) -> dict[int, Column]:
        """Return a dict of all user columns visible in this tbl version path, including columns from bases"""
        cols = self.columns()
        return {col.id: col for col in cols}

    def get_column(self, name: str, include_bases: Optional[bool] = None) -> Optional[Column]:
        """Return the column with the given name, or None if not found"""
        self.refresh_cached_md()
        col = self._cached_tbl_version.cols_by_name.get(name)
        if col is not None:
            return col
        elif self.base is not None and (include_bases or self._cached_tbl_version.include_base_columns):
            return self.base.get_column(name)
        else:
            return None

    def get_column_by_id(self, tbl_id: UUID, col_id: int) -> Optional[Column]:
        """Return the column for the given tbl/col id"""
        self.refresh_cached_md()
        if self.tbl_version.id == tbl_id:
            assert col_id in self._cached_tbl_version.cols_by_id
            return self._cached_tbl_version.cols_by_id[col_id]
        elif self.base is not None:
            return self.base.get_column_by_id(tbl_id, col_id)
        else:
            return None

    def has_column(self, col: Column, include_bases: bool = True) -> bool:
        """Return True if this table has the given column."""
        self.refresh_cached_md()
        assert col.tbl is not None
        if (
            col.tbl.id == self.tbl_version.id
            and col.tbl.effective_version == self.tbl_version.effective_version
            and col.id in self._cached_tbl_version.cols_by_id
        ):
            # the column is visible in this table version
            return True
        elif self.base is not None and include_bases:
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
