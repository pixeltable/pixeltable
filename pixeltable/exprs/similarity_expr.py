from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class SimilarityExpr(Expr):
    def __init__(self, col_ref: ColumnRef, item: Any, idx_name: Optional[str] = None):
        super().__init__(ts.FloatType())
        item_expr = Expr.from_object(item)
        if item_expr is None or not (item_expr.col_type.is_string_type() or item_expr.col_type.is_image_type()):
            raise excs.Error(f'similarity(): requires a string or a PIL.Image.Image object, not a {type(item)}')
        assert item_expr.col_type.is_string_type() or item_expr.col_type.is_image_type()

        self.components = [col_ref, item_expr]

        from pixeltable import index

        # determine index to use
        idx_dict = col_ref.find_embedding_index(idx_name, 'similarity')
        assert len(idx_dict) == 1
        self.idx_info = next(iter(idx_dict.values()))
        idx = self.idx_info.idx
        assert isinstance(idx, index.EmbeddingIndex)

        if item_expr.col_type.is_string_type() and idx.string_embed is None:
            raise excs.Error(
                f'Embedding index {self.idx_info.name!r} on column {self.idx_info.col.name!r} does not have a '
                f'string embedding and does not support string queries'
            )
        if item_expr.col_type.is_image_type() and idx.image_embed is None:
            raise excs.Error(
                f'Embedding index {self.idx_info.name!r} on column {self.idx_info.col.name!r} does not have an '
                f'image embedding and does not support image queries'
            )
        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'{self.components[0]}.similarity({self.components[1]})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('idx_name', self.idx_info.name)]

    def default_column_name(self) -> str:
        return 'similarity'

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        # TODO: validate that the index still exists
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a string or a PIL.Image.Image object, not an expression')
        item = self.components[1].val
        from pixeltable import index

        assert isinstance(self.idx_info.idx, index.EmbeddingIndex)
        return self.idx_info.idx.similarity_clause(self.idx_info.val_col, item)

    def as_order_by_clause(self, is_asc: bool) -> Optional[sql.ColumnElement]:
        if not isinstance(self.components[1], Literal):
            raise excs.Error('similarity(): requires a string or a PIL.Image.Image object, not an expression')
        item = self.components[1].val
        from pixeltable import index

        assert isinstance(self.idx_info.idx, index.EmbeddingIndex)
        return self.idx_info.idx.order_by_clause(self.idx_info.val_col, item, is_asc)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise excs.Error('similarity(): cannot be used in a computed column')

    def _as_dict(self) -> dict:
        return {'idx_name': self.idx_info.name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'SimilarityExpr':
        iname = d.get('idx_name')
        assert len(components) == 2
        assert isinstance(components[0], ColumnRef)
        return cls(components[0], components[1], idx_name=iname)
