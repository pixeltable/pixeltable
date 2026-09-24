from __future__ import annotations

import copy
from typing import Any, cast
from uuid import UUID

import sqlalchemy as sql

from pixeltable import catalog, type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class RowidRef(Expr):
    """A reference to a part of a table rowid

    This is used internally to support grouping by a base table and for references to the 'pos' column.
    Construction either walks the given TableVersion down to the base that owns the component, or takes that
    base's key directly, when no TableVersion is available.
    """

    tbl: catalog.TableVersionHandle
    normalized_base: catalog.TableVersionHandle
    rowid_component_idx: int

    def __init__(
        self,
        tbl: catalog.TableVersionHandle | None,
        idx: int,
        tbl_id: UUID | None = None,
        effective_version: int | None = None,
        normalized_base_id: UUID | None = None,
        normalized_base_effective_version: int | None = None,
    ):
        super().__init__(ts.IntType(nullable=False))
        if tbl is not None:
            self.tbl = tbl
            # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
            # (which has the same values as all its descendent views)
            normalized_base = tbl
            # don't try to reference tbl.store_tbl here
            while normalized_base.get().base is not None and normalized_base.get().base.get().num_rowid_columns() > idx:
                normalized_base = normalized_base.get().base
            self.normalized_base = normalized_base
        else:
            assert tbl_id is not None and normalized_base_id is not None
            # constructing these handles does not cause catalog loads
            self.tbl = catalog.TableVersionHandle(catalog.TableVersionKey(tbl_id, effective_version))
            self.normalized_base = catalog.TableVersionHandle(
                catalog.TableVersionKey(normalized_base_id, normalized_base_effective_version)
            )
        self.rowid_component_idx = idx
        self.id = self._create_id()

    @property
    def tbl_id(self) -> UUID:
        return self.tbl.id

    @property
    def normalized_base_id(self) -> UUID:
        return self.normalized_base.id

    def default_column_name(self) -> str | None:
        return str(self)

    def _equals(self, other: RowidRef) -> bool:
        return self.normalized_base == other.normalized_base and self.rowid_component_idx == other.rowid_component_idx

    def _id_attrs(self) -> list[tuple[str, Any]]:
        # must mirror the fields in _equals()
        return [
            *super()._id_attrs(),
            ('normalized_base_id', self.normalized_base_id),
            ('normalized_base_effective_version', self.normalized_base.effective_version),
            ('idx', self.rowid_component_idx),
        ]

    # override
    def copy(self) -> RowidRef:
        # deepcopy(tvh) is needed to create a copy for the local thread/catalog
        result = super().copy()
        result.tbl = copy.deepcopy(self.tbl)
        result.normalized_base = copy.deepcopy(self.normalized_base)
        return result

    def __repr__(self) -> str:
        # check if this is the pos column of a component view
        from pixeltable import store

        tbl = self.tbl.get()
        if (
            tbl.is_component_view
            and self.rowid_component_idx == cast(store.StoreComponentView, tbl.store_tbl).pos_col_idx
        ):
            return catalog.globals._POS_COLUMN_NAME
        return ''

    def is_bound_by(self, tbls: list[catalog.TablePath], siblings: list[catalog.Column] | None = None) -> bool:
        # base impl checks ColumnRef subexprs and trivially returns True for RowidRef (which has none);
        # match against our tbl_id instead so rowid refs aren't pulled into unrelated table scans in joins
        return any(self.tbl_id in tbl.tbl_ids for tbl in tbls)

    def set_tbl(self, tbl: catalog.TableVersionPath) -> None:
        """Change the table that is being referenced.
        This can be necessary during query planning, because at that stage we try to minimize the total number of
        tables that are referenced/need to be joined.
        We can only change to a view of the original table (which shares the base's rowid columns).
        """
        if self.tbl_id == tbl.tbl_version.id:
            return
        base_ids = [tbl_version.id for tbl_version in tbl.get_tbl_versions()]
        assert self.tbl_id in base_ids  # our current TableVersion is a base of the new TableVersion
        self.tbl = tbl.tbl_version

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        tbl = self.tbl.get()
        assert tbl.is_validated
        rowid_cols = tbl.store_tbl.rowid_columns()
        assert self.rowid_component_idx <= len(rowid_cols), (
            f'{self.rowid_component_idx} not consistent with {rowid_cols}'
        )
        return rowid_cols[self.rowid_component_idx]

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row.pk[self.rowid_component_idx]

    def _as_dict(self) -> dict:
        return {
            'tbl_id': str(self.tbl_id),
            'effective_version': self.tbl.effective_version,
            'normalized_base_id': str(self.normalized_base_id),
            'normalized_base_effective_version': self.normalized_base.effective_version,
            'idx': self.rowid_component_idx,
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr], tbl_versions: Any = None) -> RowidRef:
        # a dict written before the versions were recorded pins nothing, which is the live version
        return cls(
            tbl=None,
            idx=d['idx'],
            tbl_id=UUID(d['tbl_id']),
            effective_version=d.get('effective_version'),
            normalized_base_id=UUID(d['normalized_base_id']),
            normalized_base_effective_version=d.get('normalized_base_effective_version'),
        )
