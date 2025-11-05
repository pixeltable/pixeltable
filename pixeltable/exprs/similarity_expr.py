from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    def __init__(self, col_ref: ColumnRef, item: Any, idx_name: str | None = None):
        from pixeltable.index import EmbeddingIndex

        super().__init__(ts.FloatType())
        item_expr = Expr.from_object(item)
        if item_expr is None or not (item_expr.col_type.is_string_type() or item_expr.col_type.is_image_type()):
            raise excs.Error(f'similarity(): requires a string or a PIL.Image.Image object, not a {type(item)}')

        self.components = [col_ref, item_expr]

        # determine index to use
        idx_info = col_ref.tbl.get().get_idx(col_ref.col, idx_name, EmbeddingIndex)
        self.idx_id = idx_info.id
        self.idx_name = idx_info.name
        idx = idx_info.idx
        assert isinstance(idx, EmbeddingIndex)

        if item_expr.col_type.is_string_type() and idx.string_embed is None:
            raise excs.Error(
                f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have a '
                f'string embedding and does not support string queries'
            )
        if item_expr.col_type.is_image_type() and idx.image_embed is None:
            raise excs.Error(
                f'Embedding index {idx_info.name!r} on column {idx_info.col.name!r} does not have an '
                f'image embedding and does not support image queries'
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

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a string or a PIL.Image.Image object, not an expression')
        item = self.components[1].val
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.similarity_clause(idx_info.val_col, item)

    def as_order_by_clause(self, is_asc: bool) -> sql.ColumnElement | None:
        from pixeltable.index import EmbeddingIndex

        # check for a literal here, instead of the c'tor: needed for ExprTemplateFunctions
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a string or a PIL.Image.Image object, not an expression')
        item = self.components[1].val
        idx_info = self._resolve_idx()
        assert isinstance(idx_info.idx, EmbeddingIndex)
        return idx_info.idx.order_by_clause(idx_info.val_col, item, is_asc)

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
