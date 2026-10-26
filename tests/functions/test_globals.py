import numpy as np

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import SAMPLE_IMAGE_URL, DatabaseRoot, assert_type_eq, pxt_raises


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

    def test_coalesce(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_tbl'),
            {
                'reviewed': pxt.String | None,
                'predicted': pxt.String | None,
                'n': pxt.Int | None,
                'x': pxt.Float,
                'img': pxt.Image | None,
                'thumbnail': pxt.Image | None,
                'meta': pxt.Json | None,
                'emb': pxt.Array[(2,), np.float32] | None,
                'raw': pxt.Binary | None,
            },
        )
        t.insert(
            [
                {
                    'reviewed': None,
                    'predicted': 'cat',
                    'n': None,
                    'x': 1.5,
                    'img': SAMPLE_IMAGE_URL,
                    'meta': {'scores': np.array([0.5, 1.5])},
                    'emb': np.array([1.0, 2.0], dtype=np.float32),
                    'raw': b'abc',
                },
                {'reviewed': 'dog', 'predicted': 'cat', 'n': 3, 'x': 2.5},
                {'reviewed': None, 'predicted': None, 'n': None, 'x': 3.5},
            ]
        )
        t.add_computed_column(label=pxtf.coalesce(t.reviewed, t.predicted, 'unknown'))
        assert_type_eq(t.label.col_type, pxt.String)
        query = t.order_by(t.x).select(
            t.label, either=pxtf.coalesce(t.reviewed, t.predicted), num=pxtf.coalesce(t.n, t.x)
        )
        assert_type_eq(query.schema['either'], pxt.String | None)
        assert_type_eq(query.schema['num'], pxt.Float)
        res = query.collect()
        assert res['label'] == ['cat', 'dog', 'unknown']
        assert res['either'] == ['cat', 'dog', None]
        assert res['num'] == [1.5, 3, 3.5]
        assert len(t.where(pxtf.coalesce(t.reviewed, t.predicted) == 'cat').collect()) == 1

        res = (
            t.order_by(t.x)
            .select(
                img=pxtf.coalesce(t.thumbnail, t.img),
                meta=pxtf.coalesce(t.meta, {}),
                emb=pxtf.coalesce(t.emb, np.zeros(2, dtype=np.float32)),
                raw=pxtf.coalesce(t.raw, b''),
            )
            .collect()
        )
        assert res['img'][0].size == (640, 480) and res['img'][1:] == [None, None]
        assert np.array_equal(res['meta'][0]['scores'], [0.5, 1.5]) and res['meta'][1:] == [{}, {}]
        assert np.array_equal(res['emb'][0], [1.0, 2.0]) and all(np.array_equal(e, [0.0, 0.0]) for e in res['emb'][1:])
        assert res['raw'] == [b'abc', b'', b'']

        assert_type_eq(pxtf.coalesce(None, 2).col_type, pxt.Int)
        assert pxt.eval_udf(pxtf.coalesce, None, 2) == 2

        with pxt_raises(pxt.ErrorCode.INVALID_TYPE, match=r'no common supertype: String \| None, Image \| None'):
            pxtf.coalesce(t.reviewed, t.img)
