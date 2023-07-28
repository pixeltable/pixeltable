from typing import List
import sqlalchemy as sql
import pytest

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import StringType, BoolType, IntType, ImageType, ArrayType, ColumnType, FloatType, VideoType
from pixeltable.exprs import Expr, CompoundPredicate, FunctionCall, Literal, InlineDict, InlineArray, ColumnRef
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.functions import udf_call, dict_map, cast, sum, count
from pixeltable.functions.pil.image import blend
from pixeltable import exceptions as exc
from pixeltable import exprs
from pixeltable.function import FunctionRegistry

class TestDataFrame:
    def test_select_where(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res1 = t[t.c1, t.c2, t.c3].show(0)
        res2 = t.select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        res1 = t[t.c2 < 10][t.c1, t.c2, t.c3].show(0)
        res2 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

    def test_count(self, test_tbl: catalog.Table, indexed_img_tbl: catalog.Table) -> None:
        t = test_tbl
        cnt = t.count()
        assert cnt == 100

        cnt = t.where(t.c2 < 10).count()
        assert cnt == 10

        # count() doesn't work with similarity search
        t = indexed_img_tbl
        probe = t.select(t.img).show(1)
        img = probe[0, 0]
        with pytest.raises(exc.Error):
            _ = t.where(t.img.nearest(img)).count()
        with pytest.raises(exc.Error):
            _ = t.where(t.img.matches('car')).count()

        # for now, count() doesn't work with non-SQL Where clauses
        with pytest.raises(exc.Error):
            _ = t.where(t.img.width > 100).count()
