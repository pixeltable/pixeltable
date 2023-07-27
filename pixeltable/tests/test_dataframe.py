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
    def test_basic(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res1 = t[t.c1, t.c2, t.c3].show(0)
        res2 = t.select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        res1 = t[t.c2 < 10][t.c1, t.c2, t.c3].show(0)
        res2 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2
