import datetime
from typing import Any, Callable, Literal

import pytest

import pixeltable as pxt
from tests.utils import assert_version_metadata_eq


class TestHistory:
    def pr_us(self, us: pxt.UpdateStatus, op: str = '') -> None:
        """Print contents of UpdateStatus"""
        print(f'=========================== pr_us =========================== op: {op}')
        print(f'num_rows: {us.num_rows}')
        print(f'num_computed_values: {us.num_computed_values}')
        print(f'num_excs: {us.num_excs}')
        print(f'updated_cols: {us.updated_cols}')
        print(f'cols_with_excs: {us.cols_with_excs}')
        print(us.row_count_stats)
        print(us.cascade_row_count_stats)
        print('============================================================')

    @pytest.mark.parametrize('variant', ['get_versions', 'history'])
    def test_history(self, variant: Literal['get_versions', 'history'], reset_db: None) -> None:
        fn: Callable[[pxt.Table, int | None], Any]
        if variant == 'get_versions':
            fn = pxt.Table.get_versions
        else:
            fn = pxt.Table.history
        t = pxt.create_table(
            'test',
            source=[{'c1': 1, 'c2': 'a'}, {'c1': 2, 'c2': 'b'}],
            schema_overrides={'c1': pxt.Required[pxt.Int], 'c2': pxt.String},
            comment='some random table comment',
            primary_key=['c1'],
        )
        s = t.insert([{'c1': 0, 'c2': 'c'}])
        self.pr_us(s, 'i1')
        r = fn(t)
        print(r)
        assert s.num_rows == 1  # 1 row inserted

        s = t.add_computed_column(c3=t.c1 + 10)
        self.pr_us(s, 'acc1')
        s = t.add_computed_column(c4=t.c2.upper())
        self.pr_us(s, 'acc2')
        v = pxt.create_view('view_of_test', t, comment='view of test table')
        r = fn(v)
        print(r)
        view_created_at = (
            r[0]['created_at'] if variant == 'get_versions' else r['created_at'][0]  # type: ignore[call-overload]
        )
        # created_at should be recent
        assert view_created_at > datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=30)
        assert view_created_at < datetime.datetime.now(tz=datetime.timezone.utc)
        inserts = r[0]['inserts'] if variant == 'get_versions' else r['inserts'][0]  # type: ignore[call-overload]
        assert inserts > 0
        assert len(r) == 1

        s = t.add_computed_column(c5=t.c1 + 20)
        self.pr_us(s, 'acc3')
        s = t.add_columns({'c6': pxt.String, 'c7': pxt.Int, 'c8': pxt.Float})
        self.pr_us(s, 'ac1')
        s = t.insert(
            [
                {'c1': 3, 'c2': 'c', 'c6': 'yyy', 'c7': 100, 'c8': 1.0},
                {'c1': 4, 'c2': 'd', 'c6': 'xxx', 'c7': 200, 'c8': 2.0},
            ]
        )
        self.pr_us(s, 'i2 - insert 2 rows into table')
        assert s.num_rows == 4  # inserted: 2 rows table, 2 rows view
        # 6 -- [str_filter(c6), c2.upper(), c1 + 20, c1 + 10, str_filter(c2), str_filter(c2.upper())]

        s = t.insert([{'c1': 5, 'c2': 'e'}, {'c1': 6, 'c2': 'e'}])
        self.pr_us(s, 'i3')

        s = v.add_computed_column(v1=v.c2 + '_view')
        print(v.history())
        self.pr_us(s, 'vacc')
        assert s.num_rows == 7

        s = v.recompute_columns('v1')
        print(v.history())
        self.pr_us(s, 'vrc1')
        assert s.num_rows == 7

        s = t.insert([{'c1': 7, 'c2': 'a'}, {'c1': 8, 'c2': 'b'}])
        self.pr_us(s, 'i4')
        s = t.delete((t.c1 >= 4) & (t.c1 <= 5))
        self.pr_us(s, 'd')

        s = t.batch_update([{'c1': 2, 'c2': 'xxx'}])
        self.pr_us(s, 'u')
        assert s.num_rows == 1 + 1  # One in table, one in view

        t.rename_column('c2', 'c2_renamed')
        t.drop_column('c4')
        r = fn(t)
        print(r)

        with pytest.raises(pxt.Error, match='Invalid value for'):
            fn(t, n=0)
        with pytest.raises(pxt.Error, match='Invalid value for'):
            fn(t, n=1.5)  # type: ignore[arg-type]

        r = fn(t, n=3)
        print(r)
        assert len(r) == 3

        r = fn(t)
        print(r)
        assert len(r) == 14

        r = fn(v)
        print(r)
        assert len(r) == 8

        s = pxt.create_snapshot('snapshot_of_test_view', t, comment='snapshot of view of test table')
        r = fn(s)
        print(r)
        assert len(r) == 1

        expected = [
            {
                'version': 13,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 2,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Deleted: c4',
            },
            {
                'version': 12,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 2,
                'deletes': 0,
                'errors': 0,
                'schema_change': "Renamed: 'c2' to 'c2_renamed'",
            },
            {
                'version': 11,
                'user': None,
                'change_type': 'data',
                'inserts': 0,
                'updates': 2,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 10,
                'user': None,
                'change_type': 'data',
                'inserts': 0,
                'updates': 0,
                'deletes': 4,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 9,
                'user': None,
                'change_type': 'data',
                'inserts': 4,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 8,
                'user': None,
                'change_type': 'data',
                'inserts': 4,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 7,
                'user': None,
                'change_type': 'data',
                'inserts': 4,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 6,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 3,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Added: c6, c7, c8',
            },
            {
                'version': 5,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 3,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Added: c5',
            },
            {
                'version': 4,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 3,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Added: c4',
            },
            {
                'version': 3,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 3,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Added: c3',
            },
            {
                'version': 2,
                'user': None,
                'change_type': 'data',
                'inserts': 1,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 1,
                'user': None,
                'change_type': 'data',
                'inserts': 2,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': None,
            },
            {
                'version': 0,
                'user': None,
                'change_type': 'schema',
                'inserts': 0,
                'updates': 0,
                'deletes': 0,
                'errors': 0,
                'schema_change': 'Initial Version',
            },
        ]

        actual = t.get_versions()
        assert len(actual) == len(expected)
        for i in range(len(expected)):
            assert_version_metadata_eq(expected[i], actual[i])
