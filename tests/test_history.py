import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs


class TestHistory:
    def pr_us(self, us: pxt.UpdateStatus, op: str = '') -> None:
        """Print contents of UpdateStatus"""
        print(f'=========================== pr_us =========================== op: {op}')
        print(f'op_note: {us.op_note}')
        print(f'num_rows: {us.num_rows}')
        print(f'num_computed_values: {us.num_computed_values}')
        print(f'num_excs: {us.num_excs}')
        print(f'updated_cols: {us.updated_cols}')
        print(f'cols_with_excs: {us.cols_with_excs}')
        print(us.row_count_stats)
        print(us.cascade_row_count_stats)
        print('============================================================')

    def test_history(self, reset_db: None) -> None:
        t = pxt.create_table(
            'test',
            source=[{'c1': 1, 'c2': 'a'}, {'c1': 2, 'c2': 'b'}],
            schema_overrides={'c1': pxt.Required[pxt.Int], 'c2': pxt.String},
            comment='some random table comment',
            primary_key=['c1'],
        )
        s = t.insert([{'c1': 0, 'c2': 'c'}])
        self.pr_us(s, 'i1')
        r = t.history()
        print(r)
        assert s.num_rows == 1  # 1 row inserted
        assert s.num_computed_values == 1  # 1 computed value: str_filter(c2)

        s = t.add_computed_column(c3=t.c1 + 10)
        self.pr_us(s, 'acc1')
        s = t.add_computed_column(c4=t.c2.upper())
        self.pr_us(s, 'acc2')
        v = pxt.create_view('view_of_test', t, comment='view of test table')
        r = v.history()
        print(r)
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
        assert s.num_computed_values == 2 * 6

        s = t.insert([{'c1': 5, 'c2': 'e'}, {'c1': 6, 'c2': 'e'}])
        self.pr_us(s, 'i3')

        s = v.add_computed_column(v1=v.c2 + '_view')
        print(v.history())
        self.pr_us(s, 'vacc')
        assert s.num_rows == 7
        assert s.num_computed_values == 7 * 2  # computed values: c2 + '_view', str_filter(v1)

        s = v.recompute_columns('v1')
        print(v.history())
        self.pr_us(s, 'vrc1')
        assert s.num_rows == 7
        assert s.num_computed_values == 7 * 2  # missing the str_filter() recompute v1, str_filter(v1)

        s = t.insert([{'c1': 7, 'c2': 'a'}, {'c1': 8, 'c2': 'b'}])
        self.pr_us(s, 'i4')
        s = t.delete((t.c1 >= 4) & (t.c1 <= 5))
        self.pr_us(s, 'd')

        s = t.batch_update([{'c1': 2, 'c2': 'xxx'}])
        self.pr_us(s, 'u')
        assert s.num_rows == 1 + 1  # One in table, one in view
        assert s.num_computed_values == 3 + 1  # missing the str_filter() computation for the index column

        t.rename_column('c2', 'c2_renamed')
        t.drop_column('c4')
        r = t.history()
        print(r.schema)
        print(r)

        with pytest.raises(excs.Error, match='Invalid value for'):
            t.history(n=0)
        with pytest.raises(excs.Error, match='Invalid value for'):
            t.history(n=1.5)  # type: ignore[arg-type]

        r = t.history(n=3)
        print(r)
        assert len(r) == 3

        r = t.history()
        print(r)
        assert len(r) == 14

        r = v.history()
        print(r)
        assert len(r) == 8

        s = pxt.create_snapshot('snapshot_of_test_view', t, comment='snapshot of view of test table')
        r = s.history()
        print(r)
        assert len(r) == 1
