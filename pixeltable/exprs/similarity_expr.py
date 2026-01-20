from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

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
    """

    idx_id: int
    idx_name: str

    def __init__(self, col_ref: ColumnRef, item: Expr, idx_name: str | None = None):
        from pixeltable.index import EmbeddingIndex

        super().__init__(ts.FloatType())

        self.components = [col_ref, item]

        # determine index to use
        idx_info = col_ref.tbl.get().get_idx(col_ref.col, idx_name, EmbeddingIndex)
        self.idx_id = idx_info.id
        self.idx_name = idx_info.name
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
        return f'{self.components[0]}.similarity({self.components[1]})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('idx_id', self.idx_id)]

    def default_column_name(self) -> str:
        return 'similarity'

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex
        import logging
        _logger = logging.getLogger('pixeltable')

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        
        # Debug logging
        _logger.debug(f'SimilarityExpr.sql_expr:')
        _logger.debug(f'  components[0] (col_ref): {self.components[0]}')
        _logger.debug(f'    col.name: {self.components[0].col.name}')
        _logger.debug(f'    col.id: {self.components[0].col.id}')
        _logger.debug(f'    col.col_type: {self.components[0].col.col_type}')
        _logger.debug(f'    col.has_sa_vector_type(): {self.components[0].col.has_sa_vector_type()}')
        _logger.debug(f'  idx_info.col (original column): name={idx_info.col.name}, id={idx_info.col.id}, type={idx_info.col.col_type}')
        _logger.debug(f'  idx_info.val_col (index value column): name={idx_info.val_col.name}, id={idx_info.val_col.id}, type={idx_info.val_col.col_type}')
        _logger.debug(f'    val_col.has_sa_vector_type(): {idx_info.val_col.has_sa_vector_type()}')
        _logger.debug(f'    val_col.sa_col_type: {type(idx_info.val_col.sa_col_type)}')
        
        # val_col should be the index value column (vector column), not the original array column
        assert idx_info.val_col.has_sa_vector_type(), \
            f'val_col should be a vector column, but got column {idx_info.val_col.name} (id={idx_info.val_col.id}) with type {type(idx_info.val_col.sa_col_type)}'
        assert idx_info.val_col.id != idx_info.col.id, \
            f'val_col (id={idx_info.val_col.id}) should be different from the original column (id={idx_info.col.id})'
        
        similarity_clause = idx_info.idx.similarity_clause(idx_info.val_col, self.components[1])
        _logger.debug(f'  similarity_clause: {similarity_clause}')
        return similarity_clause

    def as_order_by_clause(self, is_asc: bool) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a value, not an expression')
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.order_by_clause(idx_info.val_col, self.components[1], is_asc)

    def _resolve_idx(self) -> 'TableVersion.IndexInfo':
        from pixeltable.index import EmbeddingIndex

        # resolve idx_id
        col_ref = self.components[0]
        if self.idx_id not in col_ref.tbl.get().idxs:
            raise excs.Error(f'Index {self.idx_name!r} not found')
        idx_info = col_ref.tbl.get().idxs[self.idx_id]
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {'idx_name': self.idx_name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        idx_name = d.get('idx_name')
        assert len(components) == 2
        assert isinstance(components[0], ColumnRef)
        return cls(components[0], components[1], idx_name=idx_name)
