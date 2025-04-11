import hashlib
import random
from typing import Union

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
import pixeltable.type_system as ts
from pixeltable import catalog, exceptions as excs

from .utils import ReloadTester


class TestSampling:
    def test_sample_errors(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        # ------- Test that sample is not preceded by anything unexpected
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).sample(n=10)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().group_by(t.c1).sample(n=10)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().order_by(t.c1).sample(n=10)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().limit(5).sample(n=10)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().join(t, on=t.c1).sample(n=10)

        # ------- Test that sample is not followed by anything unexpected
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).show()
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).head()
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).tail()
        with pytest.raises(excs.Error, match='cannot be used after'):
            _ = t.select().sample(n=10).where(t.c1 > 10)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).group_by(t.c1)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).order_by(t.c1)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).group_by(t.c1)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).limit(5)
        with pytest.raises(excs.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).join(t, on=t.c1)

        # ------- Test sample parameter correctness
        with pytest.raises(excs.Error, match='At least one of '):
            _ = t.select().sample()
        # This is now a feature (sample size = min(fraction * count, n)
        # with pytest.raises(excs.Error, match='Exactly one'):
        #     _ = t.select().sample(fraction=0.10, n=10)
        with pytest.raises(excs.Error, match='must be of type int'):
            _ = t.select().sample(n=0.01)  # type: ignore[arg-type]
        with pytest.raises(excs.Error, match='must be of type float'):
            _ = t.select().sample(fraction=24)
        with pytest.raises(excs.Error, match='fraction parameter must be between'):
            _ = t.select().sample(fraction=-0.5)
        with pytest.raises(excs.Error, match='fraction parameter must be between'):
            _ = t.select().sample(fraction=12.9)
        with pytest.raises(excs.Error, match='must be of type int'):
            _ = t.select().sample(n=10, seed=-123.456)  # type: ignore[arg-type]

        # test stratify_by list
        with pytest.raises(excs.Error, match='must be composed of expressions'):
            _ = t.select().sample(n=10, stratify_by=47)  # type: ignore[arg-type]
        with pytest.raises(excs.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[None])
        with pytest.raises(excs.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[123])
        with pytest.raises(excs.Error, match='Invalid type'):
            _ = t.select().sample(n=10, stratify_by=[t.c6])

        # String, Int, and Bool types
        _ = t.select().sample(n=10, fraction=0.123, seed=27, stratify_by=[t.c1, t.c2, t.c4])

        # Preceding where clauses must be suitable for direct sql translation
        with pytest.raises(excs.Error, match='not expressible in SQL'):
            _ = t.select().where(t.c2.apply(str) == '11').sample(n=10)

    def test_sample_display(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        df = t.select().sample(n=10, fraction=0.123, seed=27, stratify_by=[t.c1, t.c2, t.c4])
        s = repr(df)
        print(s)
        assert 'sample_1(10, 0.123, 27, [c1,c2,c4])' in s

        s = df._repr_html_()
        print(s)
        assert 'sample_1(10, 0.123, 27, [c1,c2,c4])' in s

    def test_sample_md5_fraction(self) -> None:
        from pixeltable.utils.sample import SampleClause

        fract = 0.422

        threshold_hex = SampleClause.fraction_to_md5_hex(fract)
        print(threshold_hex)

        for count in (100, 1000, 10000, 100000, 1000000, 10000000):
            k = 1
            for i in range(count):
                b = hashlib.md5(str(random.randint(0, 1000000000)).encode()).hexdigest() < threshold_hex
                if b:
                    #                print(i, b)
                    k += 1
            print(fract, count, k, k / count)
            self._check_sample_count(fract * count, k)

    @classmethod
    def _check_sample_count(cls, expected: Union[int, float], actual: int) -> None:
        assert abs(expected - actual) / actual < 0.25

    @classmethod
    def _check_sample(cls, df: pxt.DataFrame, expected: Union[int, float]) -> None:
        r = df.collect()
        print(r)
        cls._check_sample_count(expected, len(r))

    @classmethod
    def create_table(cls, t_rows: int, cat_count: int, with_null: bool) -> pxt.Table:
        schema = {
            'id': pxt.IntType(nullable=False),
            'cat1': pxt.IntType(nullable=with_null),
            'cat2': pxt.IntType(nullable=with_null),
        }
        rows = []
        for i in range(t_rows):
            cat1 = i % cat_count
            if with_null and cat1 == cat_count - 1:
                cat1 = None
            cat2 = (i // cat_count) % cat_count
            if with_null and cat2 == cat_count - 1:
                cat2 = None
            rows.append({'id': i, 'cat1': cat1, 'cat2': cat2})
        return pxt.create_table('s_t', source=rows, schema_overrides=schema)

    @classmethod
    def create_sample_data_2(cls, row_mult: int, cat_count: int, with_null: bool) -> pxt.Table:
        schema = {
            'id': ts.IntType(nullable=False),
            'cat1': ts.IntType(nullable=with_null),
            'cat2': ts.IntType(nullable=with_null),
        }
        rows = []
        rowid = 0
        for cat1 in range(cat_count):
            for cat2 in range(cat_count):
                cat1v = cat1 if not with_null or cat1 != cat_count - 1 else None
                cat2v = cat2 if not with_null or cat2 != cat_count - 1 else None
                for i in range(row_mult * (cat1 + 1) * (cat2 + 1)):
                    rows.append({'id': rowid, 'cat1': cat1v, 'cat2': cat2v})
                    rowid += 1
        return pxt.create_table('scm_t', source=rows, schema_overrides=schema)

    def test_sample_basic(self, test_tbl: catalog.Table) -> None:
        t_rows = 360
        t = self.create_table(t_rows, 6, False)

        df = t.select(t.id).sample(fraction=0.10, seed=12345)
        self._check_sample(df, t_rows * 0.10)

        df = t.select().sample(n=20)
        self._check_sample(df, 20)

        df = t.select().sample(fraction=0.123, seed=42)
        self._check_sample(df, t_rows * 0.123)

        df = t.select().sample(n=10, fraction=0.5)
        self._check_sample(df, 10)

        df = t.select().where(t.id < 200).sample(fraction=0.5)
        self._check_sample(df, 200 * 0.5)

    def test_sample_view(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t_rows = 360
        t = self.create_table(t_rows, 6, False)

        df = t.select(t.cat1).sample(fraction=0.3, seed=51)
        v = pxt.create_view('view1', df)

        results = reload_tester.run_query(v.select())
        print(results)
        reload_tester.run_reload_test()

    def test_sample_stratified(self, test_tbl: catalog.Table) -> None:
        t = self.create_sample_data_2(4, 6, True)

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n=2, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        print(r)

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        print(r)
#        assert False

    def test_sample_stratified_nulls(self, test_tbl: catalog.Table) -> None:
        t_rows = 360
        t = self.create_table(t_rows, 6, True)

        df = t.select(t.cat1, t.cat2, count=pxtf.count(t.cat1)).group_by(t.cat1, t.cat2).order_by(t.cat1, t.cat2)
        r = df.collect()
        print(r)
