import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs


class TestHistory:
    def test_history(self, reset_db: None) -> None:
        t = pxt.create_table(
            'test',
            {'c1': pxt.Required[pxt.Int], 'c2': pxt.String},
            comment='some random table comment',
            primary_key=['c1'],
        )
        t.insert([{'c1': 1, 'c2': 'a'}, {'c1': 2, 'c2': 'b'}])
        t.add_computed_column(c3=t.c1 + 10)
        t.add_computed_column(c4=t.c2.upper())
        t.add_computed_column(c5=t.c1 + 10)
        t.add_columns({'c6': pxt.String, 'c7': pxt.Int})
        t.insert([{'c1': 3, 'c2': 'c'}, {'c1': 4, 'c2': 'd'}])
        t.insert([{'c1': 5, 'c2': 'e'}, {'c1': 6, 'c2': 'e'}])
        t.insert([{'c1': 7, 'c2': 'a'}, {'c1': 8, 'c2': 'b'}])
        t.delete((t.c1 >= 4) & (t.c1 <= 5))
        t.batch_update([{'c1': 2, 'c2': 'xxx'}])
        t.rename_column('c2', 'c2_renamed')
        t.drop_column('c4')
        r = t.history()
        print(r.schema)
        print(r)
        p = r.to_pandas()
        print(p)

        with pytest.raises(excs.Error) as exc_info:
            t.history(max_versions=0)
        assert 'invalid value for' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t.history(max_versions=1.5)  # type: ignore[arg-type]
        assert 'invalid value for' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t.history(summarize_data_changes='invalid')  # type: ignore[arg-type]
        assert 'invalid value for' in str(exc_info.value).lower()

        r = t.history(max_versions=3)
        print(r)
        assert len(r) == 3

        r = t.history(summarize_data_changes=True)
        print(r)
        assert len(r) == 13

        v = pxt.create_view('view_of_test', t, comment='view of test table')
        r = v.history()
        print(r)
        assert len(r) == 1

        s = pxt.create_snapshot('snapshot_of_test_view', t, comment='snapshot of view of test table')
        r = s.history()
        print(r)
        assert len(r) == 1
