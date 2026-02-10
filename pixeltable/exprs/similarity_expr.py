from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion, TableVersionKey



class SimilarityExpr(Expr):
    """
    A similarity expression against an embedding index.
    """

    idx_id: int
    _tbl_id: UUID
    _tbl_version: int | None

    def __init__(self, col_ref: ColumnRef, item: Expr, idx_name: str | None = None):
        from pixeltable.index import EmbeddingIndex

        super().__init__(ts.FloatType())

        self.components = [item]

        # determine index to use
        idx_info = col_ref.tbl.get().get_idx(col_ref.col, idx_name, EmbeddingIndex)
        self.idx_id = idx_info.id
        self._tbl_id = idx_info.col.tbl_handle.id
        self._tbl_version = idx_info.col.tbl_handle.effective_version
        idx = idx_info.idx
        assert isinstance(idx, EmbeddingIndex)

        if item.col_type._type not in idx.embeddings:
            type_str = item.col_type._type.name.lower()
            article = 'an' if type_str[0] in 'aeiou' else 'a'
            raise excs.Error(
                f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have {article} '
                f'{type_str} embedding and does not support {type_str} queries'
            )
        self.id = self._create_id()

    def __repr__(self) -> str:
        idx_info = self._resolve_idx()
        return f'{idx_info.name}.similarity({self.components[0]})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('idx_id', self.idx_id)]

    def default_column_name(self) -> str:
        return 'similarity'

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[0], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.similarity_clause(idx_info.val_col, self.components[0])

    def as_order_by_clause(self, is_asc: bool) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[0], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.order_by_clause(idx_info.val_col, self.components[0], is_asc)

    def _resolve_idx(self) -> 'TableVersion.IndexInfo':
        from pixeltable.catalog.table_version import TableVersionKey
        from pixeltable.index import EmbeddingIndex

        key = TableVersionKey(self._tbl_id, self._tbl_version, None)
        tbl_version = catalog.Catalog.get().get_tbl_version(key)
        if tbl_version is None or self.idx_id not in tbl_version.idxs:
            raise excs.Error(f'Embedding index with id {self.idx_id} not found')
        idx_info = tbl_version.idxs[self.idx_id]
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {
            'idx_id': self.idx_id,
            'tbl_id': str(self._tbl_id),
            'tbl_version': self._tbl_version,
            'components': [c.as_dict() for c in self.components],
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        assert len(components) == 1, f'SimilarityExpr expects 1 component (item), got {len(components)}'

        # Legacy format: full ColumnRef
        if 'indexed_col' in d:
            col_ref = ColumnRef._from_dict(d['indexed_col'], [])
            return cls(col_ref, components[0], idx_name=d.get('idx_name'))
        else:
            tbl_version = d['tbl_version']
            tbl_id = d['tbl_id']
            idx_id = d['idx_id']
            return cls._from_index_ids(
                item=components[0],
                idx_id=idx_id,
                tbl_id=tbl_id,
                tbl_version=tbl_version,
            )

    @classmethod
    def _from_index_ids(
        cls, *, item: Expr, idx_id: int, tbl_id: UUID, tbl_version: int | None
    ) -> 'SimilarityExpr':
        """Internal constructor used for deserialization when we only have ids."""
        obj = object.__new__(cls)
        Expr.__init__(obj, ts.FloatType())
        obj.components = [item]
        obj.idx_id = idx_id
        obj._tbl_id = tbl_id
        obj._tbl_version = tbl_version
        obj.id = obj._create_id()
        return obj
