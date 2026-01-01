import datetime

import pytest

import pixeltable as pxt

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
        with pytest.raises(pxt.Error, match=r'Multiple sample\(\) clauses not allowed'):
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
        with pytest.raises(pxt.Error, match='must be of type `Int`; got `Float`'):
            _ = t.select().sample(n=0.01)  # type: ignore[arg-type]
        with pytest.raises(pxt.Error, match='must be >'):
            _ = t.select().sample(n=-1)
        with pytest.raises(pxt.Error, match='must be of type `Int`; got `String`'):
            _ = t.select().sample(n_per_stratum='abc', stratify_by=t.c1)  # type: ignore[arg-type]
        with pytest.raises(pxt.Error, match='must be >'):
            _ = t.select().sample(n_per_stratum=0, stratify_by=t.c1)
        with pytest.raises(pxt.Error, match='must be of type `Float`; got `Int`'):
            _ = t.select().sample(fraction=24)
        with pytest.raises(pxt.Error, match='parameter must be >'):
            _ = t.select().sample(fraction=-0.5)
        with pytest.raises(pxt.Error, match='parameter must be <'):
            _ = t.select().sample(fraction=12.9)
        with pytest.raises(pxt.Error, match='must be of type `Int`; got `Float`'):
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

        query = t.select(t.c1).sample(n=10, seed=27, stratify_by=[t.c1, t.c2, t.c4])
        s = repr(query)
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

        s = query._repr_html_()
        print(s)
        assert 'sample_1(n=10, n_per_stratum=None, fraction=' in s

    @classmethod
    def _check_sample_count(cls, expected: int | float, actual: int) -> None:
        assert abs(expected - actual) / actual < 0.25

    @classmethod
    def _check_sample(cls, query: pxt.Query, expected: int | float) -> None:
        r = query.collect()
        print(r)
        cls._check_sample_count(expected, len(r))

    def test_sample_basic_n(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, False)

        query = t.select().sample(n=20)
        self._check_sample(query, 20)

        query = t.select().where(t.id < 200).sample(n=20)
        self._check_sample(query, 20)

    def test_sample_basic_f(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, False)
        t_rows = t.count()

        query = t.select(t.id).sample(fraction=0.10, seed=12345)
        self._check_sample(query, t_rows * 0.10)

        query = t.select().sample(fraction=0.123, seed=42)
        self._check_sample(query, t_rows * 0.123)

        query = t.select().where(t.id < 200).sample(fraction=0.5, seed=876)
        self._check_sample(query, 200 * 0.5)

    def test_sample_snapshot_reload(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = self.create_sample_data(4, 6, False)

        query = t.select(t.cat1).sample(fraction=0.3, seed=51, stratify_by=[t.cat1])
        v = pxt.create_snapshot('sn_1', query)

        results = reload_tester.run_query(v.select())
        print(results)
        reload_tester.run_reload_test()

    def test_sample_stratified_n(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, True)

        query = (
            t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n_per_stratum=2, stratify_by=[t.cat1, t.cat2])
        )
        r = query.collect()
        assert len(r) == 2 * 5 * 6

        query = t.select(t.cat1, t.cat2, t.id).where(t.cat1 != None).sample(n=10, stratify_by=[t.cat1, t.cat2])
        r = query.collect()
        p = r.to_pandas().sort_values(by=['cat1', 'cat2']).to_string()
        print(p)
        assert len(r) == 10

    def test_sample_stratified_f(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()

        query = t.select(t.cat1, t.cat2, t.id).sample(fraction=0.1, stratify_by=[t.cat1, t.cat2])
        r = query.collect()
        self._check_sample_count(0.1 * t_rows, len(r))
        print(r)

    def validate_snapshot(
        self, query: pxt.Query, t_rows: int, allow_mutable_view: bool = False, seeded: bool = False
    ) -> None:
        snap = pxt.create_snapshot('sampled_snap', query, if_exists='replace')

        # Subsequent calls to the same snapshot should return the same results.
        snap_results_1 = snap.collect().to_pandas().sort_values(by=['id'])
        snap_results_2 = snap.collect().to_pandas().sort_values(by=['id'])
        assert snap_results_1.equals(snap_results_2)

        # If it's a seeded sample, the results should match a collect() on the query.
        if seeded:
            query_results = query.collect().to_pandas().sort_values(by=['id'])
            query_results.equals(snap_results_1)

        if allow_mutable_view:
            # Try with a mutable view too.
            view = pxt.create_view('sampled_view', query, if_exists='replace')
            view_results_1 = view.collect().to_pandas().sort_values(by=['id'])
            view_results_2 = view.collect().to_pandas().sort_values(by=['id'])
            assert view_results_1.equals(view_results_2)
            if seeded:
                assert view_results_1.equals(snap_results_1)

    @pytest.mark.parametrize('seed', [None, 4171780])
    def test_sample_snapshot(self, reset_db: None, seed: int) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()
        query = t.select().sample(n=10, seed=seed)
        self.validate_snapshot(query, t_rows, seeded=(seed is not None))

        query = t.select().sample(fraction=0.1, seed=seed)
        self.validate_snapshot(query, t_rows, allow_mutable_view=True, seeded=(seed is not None))

    @pytest.mark.parametrize('seed', [None, 4171780])
    def test_sample_snapshot_stratified(self, reset_db: None, seed: int) -> None:
        t = self.create_sample_data(4, 6, True)
        t_rows = t.count()
        query = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2], seed=seed)
        self.validate_snapshot(query, t_rows, seeded=(seed is not None))

        query = t.select().sample(fraction=0.1, stratify_by=[t.cat1, t.cat2], seed=seed)
        self.validate_snapshot(query, t_rows, seeded=(seed is not None))

    def check_create_insert(self, t: pxt.Table, query: pxt.Query, n_sample: int) -> None:
        r = query.collect()
        print(r)
        assert len(r) == n_sample

        # Create a new table from the sample
        new_table = pxt.create_table('new_table', source=query, if_exists='replace_force')
        assert new_table.count() == n_sample
        assert new_table._get_schema() == t._get_schema()
        assert new_table._get_schema() == query.schema
        r2 = new_table.collect()
        assert r2 == r

        new_table.insert(query)
        assert new_table.count() == 2 * n_sample

    def test_sample_create_insert_table(self, test_tbl: pxt.Table) -> None:
        t = self.create_sample_data(4, 6, False)

        query = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2], seed=4171780)
        self.check_create_insert(t, query, 6 * 6)

        query = t.select().sample(n=20, seed=4171780)
        self.check_create_insert(t, query, 20)

        query = t.select().sample(fraction=0.1, seed=4171780)
        n_sample = len(query.collect())
        self.check_create_insert(t, query, n_sample)

        t = test_tbl
        query = t.sample(n=20)
        _ = query.collect()
        query = t.sample(n=20, seed=4171780)
        self.check_create_insert(t, query, 20)

    def test_randomized_sample(self, reset_db: None) -> None:
        """Test that subsequent calls to a non-seeded sample return different results."""
        t = self.create_sample_data(4, 6, False)

        query = t.select().sample(n=10)
        r0 = query.collect().to_pandas().sort_values(by=['id'])
        r1 = query.collect().to_pandas().sort_values(by=['id'])
        # In theory this will fail with probability 2^-63. In practice, it is vanishingly less likely than other
        # potential causes of test failure.
        assert not r0.equals(r1)

    def test_reproducible_sample(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, False)

        query = t.select().sample(n_per_stratum=1, stratify_by=[t.cat1, t.cat2], seed=4141480)
        r0 = query.collect()
        r1 = query.collect()
        assert r0 == r1
        r2 = query.collect()
        assert r0 == r2
        r3 = query.collect()
        assert r0 == r3
        r4 = query.collect()
        assert r0 == r4

    def test_sample_view(self, reset_db: None) -> None:
        t = self.create_sample_data(4, 6, False)

        query = t.select().sample(fraction=0.1, stratify_by=[t.cat1, t.cat2], seed=0)
        with pytest.raises(pxt.Error, match='cannot be created with'):
            _ = pxt.create_view('v1', query)

        query = t.select().sample(n=20, seed=0)
        with pytest.raises(pxt.Error, match='cannot be created with'):
            _ = pxt.create_view('v1', query)

        query = t.select().sample(fraction=0.01, seed=0)
        n = len(query.collect())
        v = pxt.create_view('v1', query)
        assert v.count() == n

        t.insert(t.select())
        n = len(t.select().sample(fraction=0.01, seed=0).collect())
        assert v.count() == n

    def test_sample_iterator(self, reset_db: None) -> None:
        print('\n\nCREATE TABLE WITH ONE IMAGE COLUMN\n')
        t = pxt.create_table('test_tile_tbl', {'image': pxt.Image})

        print('\n\nINSERT ONE IMAGE\n')
        t.insert(image=SAMPLE_IMAGE_URL)

        print('\n\nSAMPLE IMAGE FROM TABLE\n')
        query = t.select().sample(fraction=0.001, seed=4171780)
        r = query.collect()
        print(f'total rows: {t.count()}, sample rows: {len(r)}')
        assert t.count() > len(r)

        print('\n\nCREATE ITERATOR VIEW\n')
        v = pxt.create_view(
            'test_view', t, iterator=pxt.functions.image.tile_iterator(t.image, tile_size=(100, 100), overlap=(10, 10))
        )
        v_rows = v.count()
        print(f'total rows: {v_rows}')
        print(v._get_schema)

        print('\n\nSELECT SAMPLE OF ITERATOR VIEW\n')
        query = v.select().sample(fraction=0.1, seed=4171780)
        r = query.collect()
        print(f'total rows: {v_rows}, sample rows: {len(r)}')
        assert v_rows > len(r)
        print(r)

        print('\n\nCREATE VIEW OF FRACTIONAL SAMPLE OF ITERATOR VIEW\n')
        query = v.select().sample(fraction=0.1, seed=4171780)
        r = query.collect()
        vs = pxt.create_view('test_view_sample', query)
        vs_rows = vs.count()
        print(f'total rows: {vs_rows}, sample rows: {len(r)}')
        print(r)
        assert vs_rows == len(r)

        print('\n\nSELECT STRATIFIED SAMPLES OF ITERATOR VIEW\n')
        query = v.select().sample(fraction=0.01, stratify_by=[v.pos % 10])
        assert len(query.collect()) == 10
        query = v.select().sample(n_per_stratum=1, stratify_by=[v.pos % 10])
        assert len(query.collect()) == 10
        query = v.select().sample(n=10, stratify_by=[v.pos % 10])
        assert len(query.collect()) == 10

        print('\n\nRENAME tile COLUMN in ITERATOR VIEW\n')
        v.rename_column('tile', 'tile_renamed')
        v_rows = v.count()
        print(f'total rows: {v_rows}')

    def test_count(self, test_tbl: pxt.Table) -> None:
        """Test that count() correctly returns the number of sampled rows."""
        t = test_tbl
        # Add duplicate c1 values for stratified sampling tests
        existing_c1_values = [row['c1'] for row in t.select(t.c1).distinct().collect()]

        # Add 4 more rows per distinct c1 value to get 5 total per value (total 500 rows)
        additional_rows = []
        max_c2 = t.select(t.c2).order_by(t.c2, asc=False).limit(1).collect()[0, 'c2']
        for i, c1_val in enumerate(existing_c1_values):
            for j in range(4):
                additional_rows.append(
                    {
                        'c1': c1_val,
                        'c1n': c1_val if (i * 4 + j) % 10 != 0 else None,
                        'c2': max_c2 + 1 + i * 4 + j,
                        'c3': float(max_c2 + 1 + i * 4 + j),
                        'c4': bool((max_c2 + 1 + i * 4 + j) % 2),
                        'c5': datetime.datetime(2024, 7, 1) + datetime.timedelta(hours=max_c2 + 1 + i * 4 + j),
                        'c6': {'f1': f'test string {max_c2 + 1 + i * 4 + j}', 'f2': max_c2 + 1 + i * 4 + j},
                        'c7': [{'f1': 'test string 1', 'f2': 1}],
                    }
                )
        t.insert(additional_rows)

        total_rows = t.count()
        assert total_rows == 100 + len(additional_rows)  # Original 100 + additional rows

        # Test count() with sample(n=...)
        cnt = t.sample(n=10).count()
        assert cnt == 10

        # Sample more rows than total rows in the table
        cnt = t.sample(n=1000).count()
        assert cnt == total_rows

        # Test count() with sample with fraction
        cnt = t.sample(fraction=0.5).count()
        # Should be 50% of total_rows, with tolerance of +/- 25%
        expected_min = total_rows * 0.5 * 0.75
        expected_max = total_rows * 0.5 * 1.25
        assert expected_min <= cnt <= expected_max

        # Test count() with sample() and where clause
        cnt = t.where(t.c2 < 10).sample(n=5).count()
        assert cnt == 5

        # Test with sample larger than filtered rows
        cnt = t.where(t.c2 < 10).sample(n=20).count()
        assert cnt == 10

        # Test with fraction on filtered data
        cnt = t.where(t.c2 < 80).sample(fraction=0.8).count()
        filtered_rows = t.where(t.c2 < 80).count()
        # Should be 50% of filtered_rows, with tolerance of +/- 25%
        expected_min = filtered_rows * 0.8 * 0.75
        expected_max = filtered_rows * 0.8 * 1.25
        assert expected_min <= cnt <= expected_max

        # Test with stratified sampling
        cnt = t.sample(n=10, stratify_by=t.c1).count()
        assert cnt == 10  # Should return 10 sampled rows

        # Test with stratified sampling and where clause
        cnt = t.where(t.c2 < 10).sample(n=5, stratify_by=t.c1).count()
        assert cnt == 5  # Should return 5 sampled rows

        # Test with n_per_stratum
        cnt = t.sample(n_per_stratum=2, stratify_by=t.c1).count()
        # Should return 2 times the distinct c1 values
        distinct_c1 = len(t.select(t.c1).distinct().collect())
        expected = 2 * distinct_c1  # 2 per stratum * number of distinct c1 values
        assert cnt == expected

        # Test with stratified fraction sampling
        cnt = t.sample(fraction=0.1, stratify_by=t.c1).count()
        # Stratified fraction sampling: ceil(0.1 * 5) = 1 per stratum
        # With 100 distinct c1 values, each appearing 5 times, we get exactly 1 * 100 = 100 rows
        distinct_c1 = len(t.select(t.c1).distinct().collect())
        expected = distinct_c1  # 1 per stratum * number of distinct c1 values = 100
        assert cnt == expected  # Should be exactly 100
