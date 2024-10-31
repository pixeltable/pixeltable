from typing import Any, Union

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

from .utils import assert_resultset_eq, clip_img_embed, create_img_tbl, create_test_tbl, reload_catalog


class TestSnapshot:
    def run_basic_test(
        self,
        tbl: pxt.Table,
        orig_query: Union[pxt.Table, pxt.DataFrame],
        snap: pxt.Table,
        extra_items: dict[str, Any],
        reload_md: bool
    ) -> None:
        tbl_path, snap_path = tbl._path, snap._path
        # run the initial query against the base table here, before reloading, otherwise the filter breaks
        tbl_select_list = [tbl[col_name] for col_name in tbl._schema.keys()]
        tbl_select_list.extend([value_expr for _, value_expr in extra_items.items()])
        orig_resultset = orig_query.select(*tbl_select_list).order_by(tbl.c2).collect()

        if reload_md:
            # reload md
            reload_catalog()
            tbl = pxt.get_table(tbl_path)
            snap = pxt.get_table(snap_path)

        # view select list: base cols followed by view cols
        column_names = list(snap._schema.keys())
        snap_select_list = [snap[col_name] for col_name in column_names[len(extra_items):]]
        snap_select_list.extend([snap[col_name] for col_name in extra_items.keys()])
        snap_query = snap.select(*snap_select_list).order_by(snap.c2)
        r1 = list(orig_resultset)
        r2 = list(snap_query.collect())
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # adding data to a base table doesn't change the snapshot
        rows = list(tbl.select(tbl.c1, tbl.c1n, tbl.c2, tbl.c3, tbl.c4, tbl.c5, tbl.c6, tbl.c7).collect())
        status = tbl.insert(rows)
        assert status.num_rows == len(rows)
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # update() doesn't affect the view
        status = tbl.update({'c3': tbl.c3 + 1.0})
        assert status.num_rows == tbl.count()
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # delete() doesn't affect the view
        num_tbl_rows = tbl.count()
        status = tbl.delete()
        assert status.num_rows == num_tbl_rows
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        tbl.revert()  # undo delete()
        tbl.revert()  # undo update()
        tbl.revert()  # undo insert()
        # can't revert a version referenced by a snapshot
        with pytest.raises(excs.Error) as excinfo:
            tbl.revert()
        assert 'version is needed' in str(excinfo.value)

        # can't drop a table with snapshots
        with pytest.raises(excs.Error) as excinfo:
            pxt.drop_table(tbl_path)
        assert snap_path in str(excinfo.value)

        pxt.drop_table(snap_path)
        pxt.drop_table(tbl_path)

    def test_basic(self, reset_db) -> None:
        pxt.create_dir('main')
        pxt.create_dir('snap')
        tbl_path = 'main.tbl1'
        snap_path = 'snap.snap1'

        for reload_md in [False, True]:
            for has_filter in [False, True]:
                for has_cols in [False, True]:
                    reload_catalog()
                    tbl = create_test_tbl(name=tbl_path)
                    schema = {
                        'v1': tbl.c3 * 2.0,
                        # include a lambda to make sure that is handled correctly
                        'v2': tbl.c3.apply(lambda x: x * 2.0, col_type=pxt.Float)
                    } if has_cols else {}
                    extra_items = {'v1': tbl.c3 * 2.0, 'v2': tbl.c3 * 2.0} if has_cols else {}
                    query = tbl.where(tbl.c2 < 10) if has_filter else tbl
                    snap = pxt.create_snapshot(snap_path, query, additional_columns=schema)
                    self.run_basic_test(tbl, query, snap, extra_items=extra_items, reload_md=reload_md)

    def test_errors(self, reset_db) -> None:
        tbl = create_test_tbl()
        snap = pxt.create_snapshot('snap', tbl)

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.insert([{'c3': 1.0}])
        assert 'cannot insert into view' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.insert(c3=1.0)
        assert 'cannot insert into view' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.delete()
        assert 'cannot delete from view' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.update({'c3': snap.c3 + 1.0})
        assert 'cannot update a snapshot' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.batch_update([{'c3': 1.0, 'c2': 1}])
        assert 'cannot update a snapshot' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            _ = snap.revert()
        assert 'cannot revert a snapshot' in str(excinfo.value).lower()

        with pytest.raises(pxt.Error) as excinfo:
            img_tbl = create_img_tbl()
            snap = pxt.create_snapshot('img_snap', img_tbl)
            snap.add_embedding_index('img', image_embed=clip_img_embed)
        assert 'cannot add an index to a snapshot' in str(excinfo.value).lower()

    def test_views_of_snapshots(self, reset_db) -> None:
        t = pxt.create_table('tbl', {'a': pxt.Int})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        s1 = pxt.create_snapshot('s1', t)
        v1 = pxt.create_view('v1', s1)
        s2 = pxt.create_snapshot('s2', v1)
        v2 = pxt.create_view('v2', s2)

        def verify(s1: pxt.Table, s2: pxt.Table, v1: pxt.Table, v2: pxt.Table) -> None:
            assert s1.count() == len(rows)
            assert v1.count() == len(rows)
            assert s2.count() == len(rows)
            assert v2.count() == len(rows)

        verify(s1, s2, v1, v2)

        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        verify(s1, s2, v1, v2)

        reload_catalog()
        s1 = pxt.get_table('s1')
        s2 = pxt.get_table('s2')
        v1 = pxt.get_table('v1')
        v2 = pxt.get_table('v2')
        verify(s1, s2, v1, v2)

    def test_snapshot_of_view_chain(self, reset_db) -> None:
        t = pxt.create_table('tbl', {'a': pxt.Int})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', v1)
        s = pxt.create_snapshot('s', v2)

        def verify(v1: pxt.Table, v2: pxt.Table, s: pxt.Table) -> None:
            assert v1.count() == t.count()
            assert v2.count() == t.count()
            assert s.count() == len(rows)

        verify(v1, v2, s)

        status = t.insert(rows)
        assert status.num_rows == len(rows) * 3  # we also updated 2 views
        assert status.num_excs == 0
        verify(v1, v2, s)

        reload_catalog()
        v1 = pxt.get_table('v1')
        v2 = pxt.get_table('v2')
        s = pxt.get_table('s')
        verify(v1, v2, s)

    def test_multiple_snapshot_paths(self, reset_db) -> None:
        t = create_test_tbl()
        c4 = t.select(t.c4).order_by(t.c2).collect().to_pandas()['c4']
        orig_c3 = t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3']
        v = pxt.create_view('v', base=t, additional_columns={'v1': t.c3 + 1})
        s1 = pxt.create_snapshot('s1', v)
        t.drop_column('c4')
        # s2 references the same view version as s1, but a different version of t (due to a schema change)
        s2 = pxt.create_snapshot('s2', v)
        t.update({'c6': {'a': 17}})
        # s3 references the same view version as s2, but a different version of t (due to a data change)
        s3 = pxt.create_snapshot('s3', v)
        t.update({'c3': t.c3 + 1})
        # s4 references different versions of t and v
        s4 = pxt.create_snapshot('s4', v)

        def validate(t: pxt.Table, v: pxt.Table, s1: pxt.Table, s2: pxt.Table, s3: pxt.Table, s4: pxt.Table) -> None:
            # c4 is only visible in s1
            assert np.all(s1.select(s1.c4).order_by(s1.c2).collect().to_pandas()['c4'] == c4)
            with pytest.raises(AttributeError):
                _ = t.select(t.c4).collect()
            with pytest.raises(AttributeError):
                _ = v.select(v.c4).collect()
            with pytest.raises(AttributeError):
                _ = s2.select(s2.c4).collect()
            with pytest.raises(AttributeError):
                _ = s3.select(s3.c4).collect()
            with pytest.raises(AttributeError):
                _ = s4.select(s4.c4).collect()

            # c3
            assert np.all(t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] == orig_c3 + 1)
            assert np.all(s1.select(s1.c3).order_by(s1.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s2.select(s2.c3).order_by(s2.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s3.select(s3.c3).order_by(s3.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s4.select(s4.c3).order_by(s4.c2).collect().to_pandas()['c3'] == orig_c3 + 1)

            # v1
            assert np.all(
                v.select(v.v1).order_by(v.c2).collect().to_pandas()['v1'] == \
                t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1)
            assert np.all(s1.select(s1.v1).order_by(s1.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s2.select(s2.v1).order_by(s2.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s3.select(s3.v1).order_by(s3.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(
                s4.select(s4.v1).order_by(s4.c2).collect().to_pandas()['v1'] == \
                t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1)

        validate(t, v, s1, s2, s3, s4)

        # make sure it works after metadata reload
        reload_catalog()
        t, v = pxt.get_table('test_tbl'), pxt.get_table('v')
        s1, s2, s3, s4 = pxt.get_table('s1'), pxt.get_table('s2'), pxt.get_table('s3'), pxt.get_table('s4')
        validate(t, v, s1, s2, s3, s4)
