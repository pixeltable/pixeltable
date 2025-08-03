import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

from .utils import SAMPLE_IMAGE_URL, ReloadTester


class TestSample:
    @classmethod
    def create_sample_data(cls, row_mult: int, cat_count: int, with_null: bool) -> pxt.Table:
        schema = {
            'id': pxt.Required[pxt.Int],
            'cat1': pxt.Int if with_null else pxt.Required[pxt.Int],
            'cat2': pxt.Int if with_null else pxt.Required[pxt.Int],
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

    def test_sample_errors(self, test_tbl: pxt.Table) -> None:
        t = test_tbl

        # ------- Test that sample is not preceded by anything unexpected
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).sample(n=10)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().group_by(t.c1).sample(n=10)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().order_by(t.c1).sample(n=10)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().limit(5).sample(n=10)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().join(t, on=t.c1).sample(n=10)

        # ------- Test that sample is not followed by anything unexpected
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).show()
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).head()
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).tail()
        with pytest.raises(pxt.Error, match='cannot be used after'):
            _ = t.select().sample(n=10).where(t.c1 > 10)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).group_by(t.c1)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).order_by(t.c1)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).group_by(t.c1)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).limit(5)
        with pytest.raises(pxt.Error, match='cannot be used with'):
            _ = t.select().sample(n=10).join(t, on=t.c1)

        # ------- Test sample parameter correctness
        with pytest.raises(pxt.Error, match='must be of type Int'):
            _ = t.select().sample(n=0.01)  # type: ignore[arg-type]
        with pytest.raises(pxt.Error, match='must be >'):
            _ = t.select().sample(n=-1)
        with pytest.raises(pxt.Error, match='must be of type Int'):
            _ = t.select().sample(n_per_stratum='abc', stratify_by=t.c1)  # type: ignore[arg-type]
        with pytest.raises(pxt.Error, match='must be >'):
            _ = t.select().sample(n_per_stratum=0, stratify_by=t.c1)
        with pytest.raises(pxt.Error, match='must be of type Float'):
            _ = t.select().sample(fraction=24)
        with pytest.raises(pxt.Error, match='parameter must be >'):
            _ = t.select().sample(fraction=-0.5)
        with pytest.raises(pxt.Error, match='parameter must be <'):
            _ = t.select().sample(fraction=12.9)
        with pytest.raises(pxt.Error, match='must be of type Int'):
            _ = t.select().sample(n=10, seed=-123.456)  # type: ignore[arg-type]

        # Test invalid sample parameter combinations
        with pytest.raises(pxt.Error, match='Exactly one of '):
            _ = t.select().sample()
        with pytest.raises(pxt.Error, match='Exactly one of '):
            _ = t.select().sample(n=10, n_per_stratum=5, stratify_by=t.c1)
        with pytest.raises(pxt.Error, match='Exactly one of '):
            _ = t.select().sample(n=10, fraction=0.10)
        with pytest.raises(pxt.Error, match='Exactly one of '):
            _ = t.select().sample(n_per_stratum=10, fraction=0.10, stratify_by=t.c1)
        with pytest.raises(pxt.Error, match='Must specify'):
            _ = t.select().sample(n_per_stratum=5)

        # test valid parameter combinations
        _ = t.select().sample(n=10)
        _ = t.select().sample(fraction=0.123)
        _ = t.select().sample(n=10, stratify_by=t.c1)
        _ = t.select().sample(fraction=0.123, stratify_by=t.c1)
        _ = t.select().sample(n_per_stratum=5, stratify_by=t.c1)

        # test stratify_by list
        with pytest.raises(pxt.Error, match='must be a list of scalar expressions'):
            _ = t.select().sample(n=10, stratify_by=47)
        with pytest.raises(pxt.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[None])
        with pytest.raises(pxt.Error, match='Invalid expression'):
            _ = t.select().sample(n=10, stratify_by=[123])
        with pytest.raises(pxt.Error, match='Invalid type'):
            _ = t.select().sample(n=10, stratify_by=[t.c6])

        # String, Int, Float, Bool, Timestamp types
        _ = t.select().sample(n=10, seed=27, stratify_by=[t.c1, t.c2, t.c3, t.c4, t.c5])

        # Preceding where clauses must be suitable for direct sql translation
        with pytest.raises(pxt.Error, match='not expressible in SQL'):
            t.select().where(t.c2.apply(str) == '11').sample(n=10).collect()

    def test_sample_display(self, test_tbl: pxt.Table) -> None:
        t = test_tbl

        df = t.select(t.c1).sample(n=10, seed=27, stratify_by=[t.c1, t.c2, t.c4])
        s = repr(df)
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

        s = df._repr_html_()
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

    @classmethod
    def _check_sample_count(cls, expected: int | float, actual: int) -> None:
        assert abs(expected - actual) / actual < 0.25

    @classmethod
    def _check_sample(cls, df: pxt.DataFrame, expected: int | float) -> None:
        r = df.collect()
        print(r)
        cls._check_sample_count(expected, len(r))

    def test_sample_basic_n(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select().sample(n=20)
        self._check_sample(df, 20)

        df = t.select().where(t.id < 200).sample(n=20)
        self._check_sample(df, 20)

    def test_sample_basic_f(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)
        t_rows = t.count()

        df = t.select(t.id).sample(fraction=0.10, seed=12345)
        self._check_sample(df, t_rows * 0.10)

        df = t.select().sample(fraction=0.123, seed=42)
        self._check_sample(df, t_rows * 0.123)

        df = t.select().where(t.id < 200).sample(fraction=0.5)
        self._check_sample(df, 200 * 0.5)

    def test_sample_snapshot_reload(self, test_tbl: pxt.Table, reload_tester: ReloadTester) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select(t.cat1).sample(fraction=0.3, seed=51, stratify_by=[t.cat1])
        v = pxt.create_snapshot('sn_1', df)

        results = reload_tester.run_query(v.select())
        print(results)
        reload_tester.run_reload_test()

    def test_sample_stratified_n(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, True)

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n_per_stratum=2, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        assert len(r) == 2 * 5 * 6

        df = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n=10, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        p = r.to_pandas().sort_values(by=['cat1', 'cat2']).to_string()
        print(p)
        assert len(r) == 10

    def test_sample_stratified_f(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()

        df = t.select(t.cat1, t.cat2, t.id).sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        r = df.collect()
        self._check_sample_count(0.1 * t_rows, len(r))
        print(r)

    def validate_snapshot(self, df: pxt.DataFrame, t_rows: int) -> None:
        r = df.collect()
        print(f'collected: {len(r)} of {t_rows} rows\n', r)
        pr = r.to_pandas().sort_values(by=['id']).reset_index(drop=True)

        ss = pxt.create_snapshot('sampled', df, if_exists='replace_force')
        assert ss.count() == len(r)
        rs = ss.select(ss.id, ss.cat1, ss.cat2).collect()
        print(f'snapshot: count: {ss.count()}, result: {len(rs)} of {t_rows} rows\n', rs)
        prs = rs.to_pandas().sort_values(by=['id']).reset_index(drop=True)
        assert pr.equals(prs)
        rsum = (
            ss.select(ss.cat1, ss.cat2, count1=pxtf.count(1))
            .group_by(ss.cat1, ss.cat2)
            .order_by(ss.cat1, ss.cat2)
            .collect()
        )
        print('summary:\n', rsum)

    def test_sample_snapshot(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()
        df = t.select().sample(n=10)
        self.validate_snapshot(df, t_rows)

        df = t.select().sample(fraction=0.1)
        self.validate_snapshot(df, t_rows)

    def test_sample_snapshot_stratified(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()
        df = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2])
        self.validate_snapshot(df, t_rows)

        df = t.select().sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        self.validate_snapshot(df, t_rows)

    def check_create_insert(self, t: pxt.Table, df: pxt.DataFrame, n_sample: int) -> None:
        r = df.collect()
        print(r)
        assert len(r) == n_sample

        # Create a new table from the sample
        new_table = pxt.create_table('new_table', source=df, if_exists='replace_force')
        assert new_table.count() == n_sample
        assert new_table._get_schema() == t._get_schema()
        assert new_table._get_schema() == df.schema
        r2 = new_table.collect()
        assert r2 == r

        new_table.insert(df)
        assert new_table.count() == 2 * n_sample

    def test_sample_create_insert_table(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2])
        self.check_create_insert(t, df, 6 * 6)

        df = t.select().sample(n=20)
        self.check_create_insert(t, df, 20)

        df = t.select().sample(fraction=0.1)
        n_sample = len(df.collect())
        self.check_create_insert(t, df, n_sample)

        t = test_tbl
        df = t.sample(n=20)
        self.check_create_insert(t, df, 20)
        print(df.collect())

    def test_reproducible_sample(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2])
        r0 = df.collect()
        r1 = df.collect()
        assert r0 == r1
        r2 = df.collect()
        assert r0 == r2
        r3 = df.collect()
        assert r0 == r3
        r4 = df.collect()
        assert r0 == r4

    def test_sample_view(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)

        df = t.select().sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        with pytest.raises(pxt.Error, match='cannot be created with'):
            _ = pxt.create_view('v1', df)

        df = t.select().sample(n=20)
        with pytest.raises(pxt.Error, match='cannot be created with'):
            _ = pxt.create_view('v1', df)

        df = t.select().sample(fraction=0.01)
        n = len(df.collect())
        v = pxt.create_view('v1', df)
        assert v.count() == n

        t.insert(t.select())
        n = len(t.select().sample(fraction=0.01).collect())
        assert v.count() == n

    def test_sample_iterator(self, test_tbl: pxt.Table) -> None:
        print('\n\nCREATE TABLE WITH ONE IMAGE COLUMN\n')
        t = pxt.create_table('test_tile_tbl', {'image': pxt.Image})

        print('\n\nINSERT ONE IMAGE\n')
        t.insert(image=SAMPLE_IMAGE_URL)

        print('\n\nSAMPLE IMAGE FROM TABLE\n')
        df = t.select().sample(fraction=0.001)
        r = df.collect()
        print(f'total rows: {t.count()}, sample rows: {len(r)}')
        assert t.count() > len(r)

        print('\n\nCREATE ITERATOR VIEW\n')
        v = pxt.create_view(
            'test_view',
            t,
            iterator=pxt.iterators.TileIterator.create(image=t.image, tile_size=(100, 100), overlap=(10, 10)),
        )
        v_rows = v.count()
        print(f'total rows: {v_rows}')
        print(v._get_schema)

        print('\n\nSELECT SAMPLE OF ITERATOR VIEW\n')
        df = v.select().sample(fraction=0.1)
        r = df.collect()
        print(f'total rows: {v_rows}, sample rows: {len(r)}')
        assert v_rows > len(r)
        print(r)

        print('\n\nCREATE VIEW OF FRACTIONAL SAMPLE OF ITERATOR VIEW\n')
        df = v.select().sample(fraction=0.1)
        r = df.collect()
        vs = pxt.create_view('test_view_sample', df)
        vs_rows = vs.count()
        print(f'total rows: {vs_rows}, sample rows: {len(r)}')
        print(r)
        assert vs_rows == len(r)

        print('\n\nSELECT STRATIFIED SAMPLES OF ITERATOR VIEW\n')
        df = v.select().sample(fraction=0.01, stratify_by=[v.pos % 10])
        assert len(df.collect()) == 10
        df = v.select().sample(n_per_stratum=1, stratify_by=[v.pos % 10])
        assert len(df.collect()) == 10
        df = v.select().sample(n=10, stratify_by=[v.pos % 10])
        assert len(df.collect()) == 10

        print('\n\nRENAME tile COLUMN in ITERATOR VIEW\n')
        v.rename_column('tile', 'tile_renamed')
        v_rows = v.count()
        print(f'total rows: {v_rows}')
