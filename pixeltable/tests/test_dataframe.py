import datetime
import pytest

from pixeltable import catalog
from pixeltable import exceptions as exc

class TestDataFrame:
    def test_select_where(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res1 = t[t.c1, t.c2, t.c3].show(0)
        res2 = t.select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        res1 = t[t.c2 < 10][t.c1, t.c2, t.c3].show(0)
        res2 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).show(0)
        assert res1 == res2

        # duplicate select list
        with pytest.raises(exc.Error) as exc_info:
            _ = t.select(t.c1).select(t.c2).show(0)
        assert 'already specified' in str(exc_info.value)

        # invalid expr in select list: Callable is not a valid literal
        with pytest.raises(TypeError) as exc_info:
            _ = t.select(datetime.datetime.now).show(0)
        assert 'Not a valid literal' in str(exc_info.value)

    def test_order_by(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(t.c4, t.c2).order_by(t.c4).order_by(t.c2, asc=False).show(0)

        # invalid expr in order_by()
        with pytest.raises(exc.Error) as exc_info:
            _ = t.order_by(datetime.datetime.now()).show(0)
        assert 'Invalid expression' in str(exc_info.value)

    def test_head(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        assert t.head() == t.show()
        assert t.head(10) == t.show(10)
        assert t.head(10) == t.df().limit(10).collect()

    def test_describe(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        df = t.select(t.c1).where(t.c2 < 10).limit(10)
        df.describe()

        # TODO: how to you check the output of these?
        _ = df.__repr__()
        _ = df._repr_html_()

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
            _ = t.where(t.img.nearest('car')).count()

        # for now, count() doesn't work with non-SQL Where clauses
        with pytest.raises(exc.Error):
            _ = t.where(t.img.width > 100).count()

    def test_select_literal(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(1.0).where(t.c2 < 10).show(0)
        assert res.rows == [[1.0]] * 10
