from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.catalog.table_version import TableVersionKey

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion


class SimilarityExpr(Expr):
    """
    A similarity expression against an embedding index.
    Resolved by index name so that drop + recreate under the same name still works.
    """

    tbl_version_key: TableVersionKey
    idx_name: str

    def _init_common(self, item: Expr, idx_name: str, tbl_version_key: TableVersionKey) -> None:
        super().__init__(ts.FloatType())
        self.components = [item]
        self.idx_name = idx_name
        self.tbl_version_key = tbl_version_key
        self.id = self._create_id()

    def __init__(self, col_ref: ColumnRef, item: Expr, idx_name: str | None = None):
        from pixeltable.index import EmbeddingIndex

        # determine index to use
        idx_info = col_ref.tbl.get().get_idx(col_ref.col, idx_name, EmbeddingIndex)
        idx = idx_info.idx
        assert isinstance(idx, EmbeddingIndex)

        if item.col_type._type not in idx.embeddings:
            type_str = item.col_type._type.name.lower()
            article = 'an' if type_str[0] in 'aeiou' else 'a'
            raise excs.Error(
                f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have {article} '
                f'{type_str} embedding and does not support {type_str} queries'
            )
        self._init_common(item, idx_info.name, col_ref.tbl.get().key)

    def __repr__(self) -> str:
        idx_info = self._resolve_idx()
        return f'{idx_info.name}.similarity({self.components[0]})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('idx_name', self.idx_name)]

    def tbl_ids(self) -> set[UUID]:
        return {self.tbl_version_key.tbl_id} | super().tbl_ids()

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
        from pixeltable.index import EmbeddingIndex

        tbl_version = catalog.Catalog.get().get_tbl_version(self.tbl_version_key, validate_initialized=True)
        if tbl_version is None or self.idx_name not in tbl_version.idxs_by_name:
            raise excs.Error(f'Embedding index {self.idx_name!r} not found')
        idx_info = tbl_version.idxs_by_name[self.idx_name]
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {
            'idx_name': self.idx_name,
            'tbl_version_key': self.tbl_version_key.as_dict(),
            'components': [c.as_dict() for c in self.components],
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        assert len(components) == 1, f'SimilarityExpr expects 1 component (item), got {len(components)}'
        # TODO remove
        #if 'tbl_version_key' not in d:
        #    assert len(components) == 2
        #    assert isinstance(components[0], ColumnRef)
        #    return cls(components[0], components[1], idx_name=idx_name)
        tbl_version_key = TableVersionKey.from_dict(d['tbl_version_key'])
        idx_name = d.get('idx_name')
        return cls._from_index_name(item=components[0], idx_name=idx_name, tbl_version_key=tbl_version_key)

    @classmethod
    def _from_index_name(cls, *, item: Expr, idx_name: str, tbl_version_key: TableVersionKey) -> 'SimilarityExpr':
        obj = cls.__new__(cls)
        obj._init_common(item, idx_name, tbl_version_key)
        return obj
