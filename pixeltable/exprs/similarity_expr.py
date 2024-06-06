from typing import Optional, List

import sqlalchemy as sql
import PIL.Image

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder


class SimilarityExpr(Expr):

    def __init__(self, col_ref: ColumnRef, item: Expr):
        super().__init__(ts.FloatType())
        self.components = [col_ref, item]
        self.id = self._create_id()
        assert isinstance(item, Literal)
        assert item.col_type.is_string_type() or item.col_type.is_image_type()

        # determine index to use
        idx_info = col_ref.col.get_idx_info()
        import pixeltable.index as index
        embedding_idx_info = [info for info in idx_info.values() if isinstance(info.idx, index.EmbeddingIndex)]
        if len(embedding_idx_info) == 0:
            raise excs.Error(f'No index found for column {col_ref.col}')
        if len(embedding_idx_info) > 1:
            raise excs.Error(
                f'Column {col_ref.col.name} has multiple indices; use the index name to disambiguate, '
                f'e.g., `{col_ref.col.name}.<index-name>.similarity(...)`')
        self.idx_info = embedding_idx_info[0]
        idx = self.idx_info.idx

        if item.col_type.is_string_type() and idx.txt_embed is None:
            raise excs.Error(
                f'Embedding index {self.idx_info.name} on column {self.idx_info.col.name} was created without the '
                f'text_embed parameter and does not support text queries')
        if item.col_type.is_image_type() and idx.img_embed is None:
            raise excs.Error(
                f'Embedding index {self.idx_info.name} on column {self.idx_info.col.name} was created without the '
                f'img_embed parameter and does not support image queries')

    def __str__(self) -> str:
        return f'{self.components[0]}.similarity({self.components[1]})'

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        assert isinstance(self.components[1], Literal)
        item = self.components[1].val
        return self.idx_info.idx.similarity_clause(self.idx_info.val_col, item)

    def as_order_by_clause(self, is_asc: bool) -> Optional[sql.ClauseElement]:
        assert isinstance(self.components[1], Literal)
        item = self.components[1].val
        return self.idx_info.idx.order_by_clause(self.idx_info.val_col, item, is_asc)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this should never get called
        assert False

    @classmethod
    def _from_dict(cls, d: dict, components: List[Expr]) -> Expr:
        assert len(components) == 2
        assert isinstance(components[0], ColumnRef)
        return cls(components[0], components[1])
