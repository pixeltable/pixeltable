from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple

import sqlalchemy as sql
import PIL
import numpy as np

from .expr import Expr
from .predicate import Predicate
from .column_ref import ColumnRef
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.catalog as catalog
import pixeltable.utils.clip as clip

class ImageSimilarityPredicate(Predicate):
    def __init__(self, img_col_ref: ColumnRef, img: Optional[PIL.Image.Image] = None, text: Optional[str] = None):
        assert (img is None) != (text is None)
        super().__init__()
        self.img_col_ref = img_col_ref
        self.components = [img_col_ref]
        self.img = img
        self.text = text
        self.id = self._create_id()

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return clip.embed_text(self.text)
        else:
            return clip.embed_image(self.img)

    def __str__(self) -> str:
        return f'{str(self.img_col_ref)}.nearest({"<img>" if self.img is not None else self.text})'

    def _equals(self, other: ImageSimilarityPredicate) -> bool:
        return False

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('img', id(self.img)), ('text', self.text)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        assert False

    def _as_dict(self) -> Dict:
        assert False, 'not implemented'
        # TODO: convert self.img into a serializable string
        return {'img': self.img, 'text': self.text, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'img' in d
        assert 'text' in d
        assert len(components) == 1
        return cls(components[0], d['img'], d['text'])

