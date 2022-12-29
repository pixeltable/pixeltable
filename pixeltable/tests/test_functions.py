import sqlalchemy as sql
import pytest

from pixeltable import catalog
from pixeltable.functions.pil.image import blend


class TestFunctions:
    def test_pil(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        _ = t[t.img, t.img.rotate(90), blend(t.img, t.img.rotate(90), 0.5)].show()
