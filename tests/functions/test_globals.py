import numpy as np

import pixeltable as pxt
import pixeltable.functions as pxtf


class TestGlobals:
    def test_aggregators(self, test_tbl: pxt.Table) -> None:
        t = test_tbl

        # count() over various datatypes
        res = t.select(
            c1=pxtf.count(t.c1),
            c2=pxtf.count(t.c2),
            c3=pxtf.count(t.c3),
            c4=pxtf.count(t.c4),
            c5=pxtf.count(t.c5),
            c6=pxtf.count(t.c6),
            c7=pxtf.count(t.c7),
        ).collect()
        size = t.count()
        assert res[0] == {f'c{i}': size for i in range(1, 8)}

        # count() with nulls
        res = t.select(c1n=pxtf.count(t.c1n)).collect()
        res_direct = t.select(t.c1n).collect()
        assert res[0] == {'c1n': sum(s is not None for s in res_direct['c1n'])}

        # min() and max()
        res = t.select(
            c1min=pxtf.min(t.c1),
            c2min=pxtf.min(t.c2),
            c3min=pxtf.min(t.c3),
            c4min=pxtf.min(t.c4),
            c5min=pxtf.min(t.c5),
            c1max=pxtf.max(t.c1),
            c2max=pxtf.max(t.c2),
            c3max=pxtf.max(t.c3),
            c4max=pxtf.max(t.c4),
            c5max=pxtf.max(t.c5),
        ).collect()
        res_direct = t.select(t.c1, t.c2, t.c3, t.c4, t.c5).collect()
        assert res[0] == {f'c{i}{op.__name__}': op(res_direct[f'c{i}']) for i in range(1, 6) for op in [min, max]}

        # sum() and mean()
        res = t.select(
            c2sum=pxtf.sum(t.c2), c3sum=pxtf.sum(t.c3), c2mean=pxtf.mean(t.c2), c3mean=pxtf.mean(t.c3)
        ).collect()
        res_direct = t.select(t.c2, t.c3).collect()
        assert res[0] == {f'c{i}{op.__name__}': op(res_direct[f'c{i}']) for i in (2, 3) for op in (sum, np.mean)}
