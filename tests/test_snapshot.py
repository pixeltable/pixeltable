from typing import Any

import numpy as np
import pytest

import pixeltable as pxt

from .utils import (
    ReloadTester,
    assert_resultset_eq,
    create_img_tbl,
    create_test_tbl,
    reload_catalog,
    validate_update_status,
)


class TestSnapshot:
    def run_basic_test(
        self,
        tbl: pxt.Table,
        orig_query: pxt.Table | pxt.Query,
        snap: pxt.Table,
        extra_items: dict[str, Any],
        reload_md: bool,
    ) -> None:
        tbl_path, snap_path = tbl._path(), snap._path()
        # run the initial query against the base table here, before reloading, otherwise the filter breaks
        tbl_select_list = [tbl[col_name] for col_name in tbl._get_schema()]
        tbl_select_list.extend([value_expr for _, value_expr in extra_items.items()])
        orig_resultset = orig_query.select(*tbl_select_list).order_by(tbl.c2).collect()

        if reload_md:
            # reload md
            reload_catalog()
            tbl = pxt.get_table(tbl_path)
            snap = pxt.get_table(snap_path)

        # view select list: base cols followed by view cols
        column_names = list(snap._get_schema().keys())
        snap_select_list = [snap[col_name] for col_name in column_names[len(extra_items) :]]
        snap_select_list.extend(snap[col_name] for col_name in extra_items)
        snap_query = snap.select(*snap_select_list).order_by(snap.c2)
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
        with pytest.raises(pxt.Error) as excinfo:
            tbl.revert()
        assert 'version is needed' in str(excinfo.value)

        # can't drop a table with snapshots
        with pytest.raises(pxt.Error, match='has dependents'):
            pxt.drop_table(tbl_path)

        pxt.drop_table(snap_path)
        pxt.drop_table(tbl_path)

    def test_basic(self, uses_db: None) -> None:
        pxt.create_dir('main')
        pxt.create_dir('snap')
        tbl_path = 'main.tbl1'
        snap_path = 'snap.snap1'

        for reload_md in [False, True]:
            for has_filter in [False, True]:
                for has_cols in [False, True]:
                    reload_catalog()
                    tbl = create_test_tbl(name=tbl_path)
                    schema = (
                        {
                            'v1': tbl.c3 * 2.0,
                            # include a lambda to make sure that is handled correctly
                            'v2': tbl.c3.apply(lambda x: x * 2.0, col_type=pxt.Float),
                        }
                        if has_cols
                        else {}
                    )
                    extra_items = {'v1': tbl.c3 * 2.0, 'v2': tbl.c3 * 2.0} if has_cols else {}
                    query: pxt.Table | pxt.Query = tbl.where(tbl.c2 < 10) if has_filter else tbl
                    snap = pxt.create_snapshot(snap_path, query, additional_columns=schema)
                    self.run_basic_test(tbl, query, snap, extra_items=extra_items, reload_md=reload_md)

        # adding column with same name as a base table column at
        # the time of creating a snapshot will raise an error now.
        tbl = create_test_tbl(name=tbl_path)
        assert 'c1' in tbl.columns()
        with pytest.raises(pxt.Error, match="Column 'c1' already exists in the base table"):
            pxt.create_snapshot('snap2', tbl, additional_columns={'c1': pxt.Int})

    def __test_create_if_exists(self, sname: str, t: pxt.Table, s: pxt.Table) -> None:
        """Helper function for testing if_exists parameter while creating a snaphot.

        Args:
            sname: name of an existing snapshot
            t: base table or view of the snapshot
            s: handle to the existing snapshot
        """
        id_before = s._id
        # invalid if_exists value is rejected
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_snapshot(sname, t, if_exists='invalid')  # type: ignore[arg-type]
        assert (
            "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        )

        # scenario 1: a snapshot exists at the path already
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_snapshot(sname, t)
        # if_exists='ignore' should return the existing snapshot
        s12 = pxt.create_snapshot(sname, t, if_exists='ignore')
        assert s12 == s
        assert s12._id == id_before
        # if_exists='replace' should drop the existing snapshot and create a new one
        s12 = pxt.create_snapshot(sname, t, additional_columns={'s1': pxt.Int}, if_exists='replace')
        assert s12 != s
        assert s12._id != id_before
        id_before = s12._id

        # scenario 2: a snapshot exists at the path, but has dependency
        # Note that when a view is created on a snapshot, the view is
        # dependent of the snapshot iff the snapshot has additional columns
        # not present in the base table/view of that snapshot.
        _v_on_s1 = pxt.create_view('test_view_on_snapshot1', s12)
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_snapshot(sname, t)
        # if_exists='ignore' should return the existing snapshot
        s13 = pxt.create_snapshot(sname, t, if_exists='ignore')
        assert s13 == s12
        assert s13._id == id_before
        assert 'test_view_on_snapshot1' in pxt.list_tables()
        # if_exists='replace' cannot drop a snapshot with a dependent view.
        # it should raise an error and recommend using 'replace_force'
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_snapshot(sname, t, if_exists='replace')
        err_msg = str(exc_info.value).lower()
        assert 'already exists' in err_msg and 'has dependents' in err_msg and 'replace_force' in err_msg
        assert 'test_view_on_snapshot1' in pxt.list_tables()
        # if_exists='replace_force' should drop the existing snapshot and
        # its dependent views and create a new one
        s13 = pxt.create_snapshot(sname, t, if_exists='replace_force')
        assert s13 != s12
        assert s13._id != id_before
        assert 'test_view_on_snapshot1' not in pxt.list_tables()

        # scenario 3: path exists but is not a snapshot
        _ = pxt.create_table('not_snapshot', {'c1': pxt.String}, if_exists='replace')
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_snapshot('not_snapshot', t)
        # if_exists='ignore' should error when existing object is not a snapshot
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_snapshot('not_snapshot', t, if_exists='ignore')
        err_msg = str(exc_info.value).lower()
        assert 'already exists' in err_msg and 'is not a snapshot' in err_msg
        assert 'not_snapshot' in pxt.list_tables()
        # if_exists='replace' and 'replace_force' should replace the table with a snapshot
        snap = pxt.create_snapshot('not_snapshot', t, if_exists='replace')
        assert 'not_snapshot' in pxt.list_tables()
        assert snap._tbl_version_path.is_snapshot()

    def test_create_if_exists(self, uses_db: None, reload_tester: ReloadTester) -> None:
        """Test the if_exists parameter while creating a snapshot."""
        t = create_test_tbl()
        v = pxt.create_view('test_view', t)
        s1 = pxt.create_snapshot('test_snap_t', t)
        s2 = pxt.create_snapshot('test_snap_v', v)
        id_before = {'test_snap_t': s1._id, 'test_snap_v': s2._id}
        self.__test_create_if_exists('test_snap_t', t, s1)
        self.__test_create_if_exists('test_snap_v', v, s2)
        # sanity check persistence
        _ = reload_tester.run_query(t.select())
        _ = reload_tester.run_query(v.select())
        # get the snapshot handles again, they would be replaced at the end of __test_create_if_exists
        s1 = pxt.get_table('test_snap_t')
        s2 = pxt.get_table('test_snap_v')
        id_before = {'test_snap_t': s1._id, 'test_snap_v': s2._id}
        _ = reload_tester.run_query(s1.select())
        _ = reload_tester.run_query(s2.select())
        reload_tester.run_reload_test()
        # get the snapshot handles again after reload
        s1 = pxt.get_table('test_snap_t')
        s2 = pxt.get_table('test_snap_v')
        assert s1._id == id_before['test_snap_t']
        assert s2._id == id_before['test_snap_v']

    def test_errors(self, test_tbl: pxt.Table, clip_embed: pxt.Function) -> None:
        tbl = test_tbl
        snap = pxt.create_snapshot('snap', tbl)
        display_str = "snapshot 'snap'"

        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot insert into a snapshot.'):
            _ = snap.insert([{'c3': 1.0}])

        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot insert into a snapshot.'):
            _ = snap.insert(c3=1.0)

        # adding column is not supported for snapshots
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a snapshot.'):
            snap.add_column(non_existing_col1=pxt.String)
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a snapshot.'):
            snap.add_computed_column(on_existing_col1=tbl.c2 + tbl.c3)
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a snapshot.'):
            snap.add_columns({'non_existing_col1': pxt.String, 'non_existing_col2': pxt.String})
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot delete from a snapshot.'):
            _ = snap.delete()
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot update a snapshot.'):
            _ = snap.update({'c3': snap.c3 + 1.0})
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot update a snapshot.'):
            _ = snap.batch_update([{'c3': 1.0, 'c2': 1}])
        with pytest.raises(pxt.Error, match=f'{display_str}: Cannot revert a snapshot.'):
            snap.revert()

        with pytest.raises(pxt.Error, match=r"snapshot 'img_snap': Cannot add an index to a snapshot."):
            img_tbl = create_img_tbl()
            snap = pxt.create_snapshot('img_snap', img_tbl)
            snap.add_embedding_index('img', image_embed=clip_embed)

        with pytest.raises(pxt.Error, match='Cannot create default indexes on a snapshot'):
            _ = pxt.create_view('default_snap', tbl, is_snapshot=True, create_default_idxs=True)

    @pytest.mark.parametrize('anonymous', [True, False])
    def test_views_of_snapshots(self, anonymous: bool, uses_db: None) -> None:
        t = pxt.create_table('tbl', {'a': pxt.Int})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        validate_update_status(t.insert(rows), expected_rows=len(rows))
        assert t._get_version() == 1
        s1 = pxt.get_table('tbl:1') if anonymous else pxt.create_snapshot('s1', t)
        v1 = pxt.create_view('v1', s1)
        s2 = pxt.get_table('v1:0') if anonymous else pxt.create_snapshot('s2', v1)
        v2 = pxt.create_view('v2', s2)

        def verify(s1: pxt.Table, s2: pxt.Table, v1: pxt.Table, v2: pxt.Table) -> None:
            assert s1.count() == len(rows)
            assert v1.count() == len(rows)
            assert s2.count() == len(rows)
            assert v2.count() == len(rows)

        verify(s1, s2, v1, v2)

        validate_update_status(t.insert(rows), expected_rows=len(rows))
        verify(s1, s2, v1, v2)

        reload_catalog()
        s1 = pxt.get_table('tbl:1') if anonymous else pxt.get_table('s1')
        s2 = pxt.get_table('v1:0') if anonymous else pxt.get_table('s2')
        v1 = pxt.get_table('v1')
        v2 = pxt.get_table('v2')
        verify(s1, s2, v1, v2)

    def test_snapshot_of_view_chain(self, uses_db: None) -> None:
        t = pxt.create_table('tbl', {'a': pxt.Int})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        validate_update_status(t.insert(rows), expected_rows=len(rows))
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', v1)
        s = pxt.create_snapshot('s', v2)

        def verify(v1: pxt.Table, v2: pxt.Table, s: pxt.Table) -> None:
            assert v1.count() == t.count()
            assert v2.count() == t.count()
            assert s.count() == len(rows)

        verify(v1, v2, s)

        validate_update_status(t.insert(rows), expected_rows=(len(rows) * 3))  # we also updated 2 views
        verify(v1, v2, s)

        reload_catalog()
        v1 = pxt.get_table('v1')
        v2 = pxt.get_table('v2')
        s = pxt.get_table('s')
        verify(v1, v2, s)

    def test_multiple_snapshot_paths(self, uses_db: None) -> None:
        t = create_test_tbl()
        c4 = t.select(t.c4).order_by(t.c2).collect().to_pandas()['c4']
        orig_c3 = t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3']
        v = pxt.create_view('v', base=t, additional_columns={'v1': t.c3 + 1})
        s1 = pxt.create_snapshot('s1', v)
        t.drop_column('c4')
        # s2 references the same view version as s1, but a different version of t (due to a schema change)
        s2 = pxt.create_view('s2', v, is_snapshot=True)  # Test alternate syntax; equiv. pxt.create_snapshot('s2', v)
        t.update({'c6': {'a': 17}})
        # s3 references the same view version as s2, but a different version of t (due to a data change)
        s3 = pxt.create_snapshot('s3', v)
        t.update({'c3': t.c3 + 1})
        # s4 references different versions of t and v
        s4 = pxt.create_snapshot('s4', v)

        def validate(t: pxt.Table, v: pxt.Table, s1: pxt.Table, s2: pxt.Table, s3: pxt.Table, s4: pxt.Table) -> None:
            # c4 is only visible in s1
            _ = s1.c4
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
                v.select(v.v1).order_by(v.c2).collect().to_pandas()['v1']
                == t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1
            )
            assert np.all(s1.select(s1.v1).order_by(s1.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s2.select(s2.v1).order_by(s2.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s3.select(s3.v1).order_by(s3.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(
                s4.select(s4.v1).order_by(s4.c2).collect().to_pandas()['v1']
                == t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1
            )

        validate(t, v, s1, s2, s3, s4)

        # make sure it works after metadata reload
        reload_catalog()
        t, v = pxt.get_table('test_tbl'), pxt.get_table('v')
        s1, s2, s3, s4 = pxt.get_table('s1'), pxt.get_table('s2'), pxt.get_table('s3'), pxt.get_table('s4')
        validate(t, v, s1, s2, s3, s4)

    def test_drop_column_in_view_predicate(self, uses_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('tbl', {'c1': pxt.Int, 'c2': pxt.Int})
        _ = pxt.create_snapshot('base_snap', t, additional_columns={'s1': pxt.Int})
        v1 = pxt.create_view('view1', t.where(t.c1 % 2 == 0), additional_columns={'vc1': pxt.Int})  # uses c1
        v1s = pxt.create_snapshot('v1_snap', v1, additional_columns={'v1s1': v1.c2 + v1.vc1})  # snapshot uses c2
        v2 = pxt.create_view(
            'view2', v1.where((v1.c2 + v1.vc1) % 2 == 0), additional_columns={'vc2': pxt.Int}
        )  # uses c2
        v2s = pxt.create_snapshot('v2_snap', v2, additional_columns={'v2s1': v2.c1 + v2.vc2})  # snapshot uses c1

        # Create view on snapshot
        _ = pxt.create_view('view_snap1', v1s.where(v1s.c1 % 4 == 0))
        _ = pxt.create_view('view_snap2', v2s.where(v2s.c2 % 4 == 0))

        # Delete first column, only mutable tables will show up in error
        with pytest.raises(pxt.Error, match="Cannot drop column 'c1' because the following views depend on it") as e:
            t.drop_column('c1')
        assert 'view: view1, predicate: c1 % 2 == 0' in str(e.value).lower()
        assert 'v2_snap' not in str(e.value).lower()  # v2_snap uses c1
        assert 'view_snap1' not in str(e.value).lower()

        # Delete 2nd column
        with pytest.raises(pxt.Error, match="Cannot drop column 'c2' because the following views depend on it") as e:
            t.drop_column('c2')
        assert 'view: view2, predicate: (c2 + vc1) % 2 == 0' in str(e.value).lower()
        assert 'v1_snap' not in str(e.value).lower()  # v1_snap uses c2
        assert 'view_snap2' not in str(e.value).lower()

        # Delete view's column
        with pytest.raises(pxt.Error, match="Cannot drop column 'vc1' because the following views depend on it") as e:
            v1.drop_column('vc1')
        assert 'view: view2, predicate: (c2 + vc1) % 2 == 0' in str(e.value).lower()
        assert 'v2_snap' not in str(e.value).lower()
        assert 'view_snap1' not in str(e.value).lower()
        assert 'view_snap2' not in str(e.value).lower()

    def test_unstored_snapshot(self, uses_db: None, reload_tester: ReloadTester) -> None:
        """Tests that a snapshot of a table with unstored columns is queryable."""
        t = pxt.create_table('tbl', {'c1': pxt.Int})
        t.add_computed_column(c2=(t.c1 + 1), stored=False)
        t.insert({'c1': i} for i in range(100))
        snap = pxt.create_snapshot('snap', t)
        reload_tester.run_query(snap.order_by(t.c1))
        reload_tester.run_reload_test()

    def test_rename_column(self, uses_db: None) -> None:
        t = pxt.create_table('tbl', {'c1': pxt.Int, 'c2': pxt.Int})

        s1 = pxt.create_snapshot('base_snap', t, additional_columns={'s1': pxt.Int})
        v1 = pxt.create_view('view_snap', s1, additional_columns={'v1': pxt.Int})

        v2 = pxt.create_view('view', t, additional_columns={'v2': pxt.Int})
        s2 = pxt.create_snapshot('snap_view', v2, additional_columns={'s2': pxt.Int})

        with pytest.raises(pxt.Error, match=r"Cannot rename column for immutable table 'base_snap'"):
            s1.rename_column('s1', 'new_s1')

        with pytest.raises(pxt.Error, match=r"Cannot rename column for immutable table 'snap_view'"):
            s2.rename_column('v2', 'new_v2')

        with pytest.raises(pxt.Error, match=r"Cannot rename base table column 'c1'"):
            v1.rename_column('c1', 'new_c1')

        with pytest.raises(pxt.Error, match=r"Cannot rename base table column 's1'"):
            v1.rename_column('s1', 'new_s1')

        # should work
        v1.rename_column('v1', 'new_v1')

    # TODO: Currently, comments and custom_metadata are not persisted for pure snapshots.
    # Should we consider snapshots as non-pure when these are provided?
    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_snapshot_comment(self, uses_db: None, do_reload_catalog: bool) -> None:
        t = pxt.create_table('tbl', {'c': pxt.Int})
        s1 = pxt.create_snapshot(
            'tbl_snapshot', t, additional_columns={'d': pxt.Int}, comment='This is a test snapshot.'
        )
        assert s1.get_metadata()['comment'] == 'This is a test snapshot.'

        reload_catalog(do_reload_catalog)
        s1 = pxt.get_table('tbl_snapshot')
        assert s1.get_metadata()['comment'] == 'This is a test snapshot.'

        # check that raw object JSON comments are rejected
        with pytest.raises(pxt.Error, match='`comment` must be a string'):
            pxt.create_snapshot(
                'tbl_snapshot_invalid',
                t,
                additional_columns={'d': pxt.Int},
                comment={'comment': 'This is a test snapshot.'},  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_snapshot_custom_metadata(self, uses_db: None, do_reload_catalog: bool) -> None:
        custom_metadata = {'key1': 'value1', 'key2': 2, 'key3': [1, 2, 3]}
        t = pxt.create_table('tbl', {'c': pxt.Int})
        s1 = pxt.create_snapshot('tbl_snapshot', t, additional_columns={'d': pxt.Int}, custom_metadata=custom_metadata)
        assert s1.get_metadata()['custom_metadata'] == custom_metadata

        reload_catalog(do_reload_catalog)
        s1 = pxt.get_table('tbl_snapshot')
        assert s1.get_metadata()['custom_metadata'] == custom_metadata

        # check that invalid JSON user metadata are rejected
        with pytest.raises(pxt.Error):
            pxt.create_snapshot(
                'tbl_snapshot_invalid', t, additional_columns={'d': pxt.Int}, custom_metadata={'key': set}
            )
