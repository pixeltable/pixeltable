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
    @classmethod
    def create_sample_data(cls, row_mult: int, cat_count: int, with_null: bool) -> pxt.Table:
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
                for _ in range(row_mult * (cat1 + 1) * (cat2 + 1)):
                    rows.append({'id': rowid, 'cat1': cat1v, 'cat2': cat2v})
                    rowid += 1
        return pxt.create_table('scm_t', source=rows, schema_overrides=schema)

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
        with pytest.raises(excs.Error, match='must be of type Int'):
            _ = t.select().sample(n=0.01)  # type: ignore[arg-type]
        with pytest.raises(excs.Error, match='must be >'):
            _ = t.select().sample(n=-1)
        with pytest.raises(excs.Error, match='must be of type Int'):
            _ = t.select().sample(n_per_stratum='abc', stratify_by=t.c1)  # type: ignore[arg-type]
        with pytest.raises(excs.Error, match='must be >'):
            _ = t.select().sample(n_per_stratum=0, stratify_by=t.c1)
        with pytest.raises(excs.Error, match='must be of type Float'):
            _ = t.select().sample(fraction=24)
        with pytest.raises(excs.Error, match='parameter must be >'):
            _ = t.select().sample(fraction=-0.5)
        with pytest.raises(excs.Error, match='parameter must be <'):
            _ = t.select().sample(fraction=12.9)
        with pytest.raises(excs.Error, match='must be of type Int'):
            _ = t.select().sample(n=10, seed=-123.456)  # type: ignore[arg-type]

        # Test invalid sample parameter combinations
        with pytest.raises(excs.Error, match='At least one of '):
            _ = t.select().sample()
        with pytest.raises(excs.Error, match='Must specify'):
            _ = t.select().sample(n_per_stratum=5)
        with pytest.raises(excs.Error, match='Cannot specify both'):
            _ = t.select().sample(n=10, n_per_stratum=5, stratify_by=t.c1)
        with pytest.raises(excs.Error, match='Cannot specify both'):
            _ = t.select().sample(n=10, fraction=0.10)
        with pytest.raises(excs.Error, match='Cannot specify both'):
            _ = t.select().sample(n_per_stratum=10, fraction=0.10, stratify_by=t.c1)

        # test valid parameter combinations
        _ = t.select().sample(n=10)
        _ = t.select().sample(fraction=0.123)
        _ = t.select().sample(n=10, stratify_by=t.c1)
        _ = t.select().sample(fraction=0.123, stratify_by=t.c1)
        _ = t.select().sample(n_per_stratum=5, stratify_by=t.c1)

        # test stratify_by list
        with pytest.raises(excs.Error, match='must be a list of scalar expressions'):
            _ = t.select().sample(n=10, stratify_by=47)
        with pytest.raises(excs.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[None])
        with pytest.raises(excs.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[123])
        with pytest.raises(excs.Error, match='Invalid type'):
            _ = t.select().sample(n=10, stratify_by=[t.c6])

        # String, Int, Float, Bool, Timestamp types
        _ = t.select().sample(n=10, seed=27, stratify_by=[t.c1, t.c2, t.c3, t.c4, t.c5])

        # Preceding where clauses must be suitable for direct sql translation
        with pytest.raises(excs.Error, match='not expressible in SQL'):
            _ = t.select().where(t.c2.apply(str) == '11').sample(n=10)

    def test_sample_display(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        df = t.select(t.c1).sample(n=10, seed=27, stratify_by=[t.c1, t.c2, t.c4])
        s = repr(df)
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

        s = df._repr_html_()
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

    def test_sample_md5_fraction(self) -> None:
        from pixeltable.utils.sample import SampleClause

        fract = 0.422

        threshold_hex = SampleClause.fraction_to_md5_hex(fract)
        print(threshold_hex)

        for count in (100, 1000, 10000, 100000, 1000000):
            k = 1
            for _i in range(count):
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

    def test_sample_basic(self, test_tbl: catalog.Table) -> None:
        t = self.create_sample_data(4, 6, False)
        t_rows = t.count()

        df = t.select(t.id).sample(fraction=0.10, seed=12345)
        self._check_sample(df, t_rows * 0.10)

        df = t.select().sample(n=20)
        self._check_sample(df, 20)

        df = t.select().sample(fraction=0.123, seed=42)
        self._check_sample(df, t_rows * 0.123)

        df = t.select().where(t.id < 200).sample(fraction=0.5)
        self._check_sample(df, 200 * 0.5)

    def test_sample_view_reload(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select(t.cat1).sample(fraction=0.3, seed=51)
        v = pxt.create_view('view1', df)

        results = reload_tester.run_query(v.select())
        print(results)
        reload_tester.run_reload_test()

    def summarize_sample(self, df: pxt.DataFrame) -> pxt.DataFrame:
        ss = pxt.create_snapshot('sampled', df, if_exists='replace_force')
        return ss.select(ss.cat1, ss.cat2, count1=pxtf.count(1)).group_by(ss.cat1, ss.cat2).order_by(ss.cat1, ss.cat2)

    def test_sample_stratified(self, test_tbl: catalog.Table) -> None:
        t = self.create_sample_data(4, 6, True)

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n_per_stratum=2, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        print(r)
        assert len(r) == 2 * 5 * 6

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n=70, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        p = r.to_pandas().sort_values(by=['cat1', 'cat2']).to_string()
        print(p)
        assert len(r) == 70

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        print(r)

        df = t.select(t.cat1, t.cat2, t.id).sample(n_per_stratum=1, stratify_by=[t.cat1 % 3, t.cat2], seed=12345)
        r = df.collect()
        print(r)
        assert len(r) == 4 * 6

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat2 == 0).sample(fraction=0.5, stratify_by=[t.cat1 % 2], seed=1)
        r = df.collect()
        print(r)
        print('summary:\n', self.summarize_sample(df).collect())
