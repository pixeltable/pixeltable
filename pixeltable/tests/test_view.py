import pytest
import math
import numpy as np
import pandas as pd
import datetime

import PIL

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import create_test_tbl, assert_resultset_eq, get_video_files
from pixeltable.iterators import FrameIterator


class TestView:
    """
    TODO:
    - test tree of views
    - test consecutive component views

    """
    def create_tbl(self, cl: pt.Client) -> catalog.InsertableTable:
        """Create table with computed columns"""
        t = create_test_tbl(cl)
        t.add_column(catalog.Column('d1', computed_with=t.c3 - 1))
        # add column that can be updated
        t.add_column(catalog.Column('c10', FloatType()))
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_column(catalog.Column('d2', computed_with=t.c3 - t.c10))
        return t

    def test_basic(self, test_client: pt.Client) -> None:
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        cols = [
            catalog.Column('v1', computed_with=t.c3 * 2.0),
            catalog.Column('v2', computed_with=t.c6.f5)
        ]
        v = cl.create_view('test_view', t, schema=cols, filter=t.c2 < 10)
        # TODO: test repr more thoroughly
        _ = v.__repr__()
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).collect(),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).collect())
        # view-only query; returns the same result
        assert_resultset_eq(
            v.select(v.v1).order_by(v.v1).collect(),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).collect())
        # computed columns that don't reference the base table
        v.add_column(catalog.Column('v3', computed_with=v.v1 * 2.0))
        v.add_column(catalog.Column('v4', computed_with=v.v2[0]))
        assert v.count() == t.where(t.c2 < 10).count()
        _ = v.where(v.c2 > 4).show(0)

        # check view md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        v_old = v
        v = cl.get_table('test_view')
        assert v.predicate == v_old.predicate
        assert set(v_old.cols_by_name.keys()) == set(v.cols_by_name.keys())
        assert v.cols_by_name['v1'].value_expr == v.c3 * 2.0

        view_query = v.select(v.v1).order_by(v.c2)
        base_query = t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2)

        # insert data: of 20 new rows, only 10 are reflected in the view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        status = t.insert([list(r.values()) for r in rows])
        assert status.num_rows == 30
        assert t.count() == 120
        assert_resultset_eq(view_query.collect(), base_query.collect())

        # update data: cascade to view
        status = t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert status.num_rows == 10 * 2  # *2: rows affected in both base table and view
        assert t.count() == 120
        assert_resultset_eq(view_query.collect(), base_query.collect())

        # base table delete is reflected in view
        status = t.delete(where=t.c2 < 5)
        status.num_rows == 10 * 2  # *2: rows affected in both base table and view
        assert t.count() == 110
        assert_resultset_eq(view_query.collect(), base_query.collect())

        # test delete view
        cl.drop_table('test_view')
        cl = pt.Client()

        with pytest.raises(exc.Error) as exc_info:
            _ = cl.get_table('test_view')
        assert 'No such path:' in str(exc_info.value)

    def test_computed_cols(self, test_client: pt.Client) -> None:
        cl = test_client
        t = self.create_tbl(cl)

        # create view with computed columns
        cols = [
            catalog.Column('v1', computed_with=t.c3 * 2.0),
            catalog.Column('v2', computed_with=t.c6.f5)
        ]
        v = cl.create_view('test_view', t, schema=cols)
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).order_by(t.c2).show(0))
        # computed columns that don't reference the base table
        v.add_column(catalog.Column('v3', computed_with=v.v1 * 2.0))
        v.add_column(catalog.Column('v4', computed_with=v.v2[0]))

        # use view md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')

        # insert data
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).collect())
        t.insert([list(r.values()) for r in rows])
        assert t.count() == 200
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).order_by(t.c2).show(0))

        # update data: cascade to view
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 200
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).order_by(t.c2).show(0))

        # base table delete is reflected in view
        t.delete(where=t.c2 < 5)
        assert t.count() == 190
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).order_by(t.c2).show(0))

    def test_filter(self, test_client: pt.Client) -> None:
        cl = test_client
        t = create_test_tbl(cl)

        # create view with filter
        v = cl.create_view('test_view', t, filter=t.c2 < 10)
        assert_resultset_eq(
            v.order_by(v.c2).show(0),
            t.where(t.c2 < 10).order_by(t.c2).show(0))

        # use view md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')

        # insert data: of 20 new rows, only 10 are reflected in the view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7).where(t.c2 < 20).collect())
        t.insert([list(r.values()) for r in rows])
        assert t.count() == 120
        assert_resultset_eq(
            v.order_by(v.c2).show(0),
            t.where(t.c2 < 10).order_by(t.c2).show(0))

        # update data
        t.update({'c4': True, 'c3': t.c3 + 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        assert_resultset_eq(
            v.order_by(v.c2).show(0),
            t.where(t.c2 < 10).order_by(t.c2).show(0))

        # base table delete is reflected in view
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        assert_resultset_eq(
            v.order_by(v.c2).show(0),
            t.where(t.c2 < 10).order_by(t.c2).show(0))

    @pytest.mark.skip(reason='revise view snapshots')
    def test_snapshot_view(self, test_client: pt.Client) -> None:
        """Test view over a snapshot"""
        cl = test_client
        t = self.create_tbl(cl)
        snap = cl.create_snapshot('test_snap', 'test_tbl')

        # create view with filter and computed columns
        cols = [
            catalog.Column('v1', computed_with=snap.c3 * 2.0),
            catalog.Column('v2', computed_with=snap.c6.f5)
        ]
        v = cl.create_view('test_view', snap, schema=cols, filter=snap.c2 < 10)
        res = v.select(v.v1).order_by(v.c2).show(0)
        assert_resultset_eq(
            res,
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).show(0))
        # computed columns that don't reference the base table
        v.add_column(catalog.Column('v3', computed_with=v.v1 * 2.0))
        v.add_column(catalog.Column('v4', computed_with=v.v2[0]))
        assert v.count() == t.where(t.c2 < 10).count()
        _ = v.where(v.c2 > 4).show(0)

        # use view md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')

        # insert data: no changes to view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        t.insert([list(r.values()) for r in rows])
        assert t.count() == 120
        assert_resultset_eq(v.select(v.v1).order_by(v.c2).show(0), res)

        # update data: no changes to view
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        assert_resultset_eq(v.select(v.v1).order_by(v.c2).show(0), res)

        # base table delete: no change to view
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        assert_resultset_eq(v.select(v.v1).order_by(v.c2).show(0), res)

    @pytest.mark.skip(reason='revise view snapshots')
    def test_snapshots(self, test_client: pt.Client) -> None:
        """Test snapshot of a view of a snapshot"""
        cl = test_client
        t = self.create_tbl(cl)
        s = cl.create_snapshot('test_snap', 'test_tbl')

        # create view with filter and computed columns
        cols = [
            catalog.Column('v1', computed_with=t.c3 * 2.0),
            catalog.Column('v2', computed_with=t.c6.f5)
        ]
        v = cl.create_view('test_view', s, schema=cols, filter=t.c2 < 10)
        view_s = cl.create_snapshot('test_view_snap', 'test_view')
        snapshot_query = view_s.order_by(view_s.c2)
        table_snapshot_query = \
            s.select(s.c3 * 2.0, s.c6.f5, s.c1, s.c1n, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7, s.c8, s.d1, s.c10, s.d2) \
                .where(s.c2 < 10).order_by(s.c2)
        assert_resultset_eq(snapshot_query.show(0), table_snapshot_query.show(0))
        # add more columns
        v.add_column(catalog.Column('v3', computed_with=v.v1 * 2.0))
        v.add_column(catalog.Column('v4', computed_with=v.v2[0]))
        assert_resultset_eq(snapshot_query.show(0), table_snapshot_query.show(0))

        # check md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        view_s_old = view_s
        view_s = cl.get_table('test_view_snap')
        assert view_s.tbl_version.predicate == view_s_old.tbl_version.predicate
        assert set(view_s_old.tbl_version.cols_by_name.keys()) == set(view_s.tbl_version.cols_by_name.keys())
        assert view_s.tbl_version.cols_by_name['v1'].value_expr == t.c3 * 2.0

        # insert data: no changes to snapshot
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        t.insert([list(r.values()) for r in rows])
        assert t.count() == 120
        assert_resultset_eq(snapshot_query.show(0), table_snapshot_query.show(0))

        # update data: no changes to snapshot
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        assert_resultset_eq(snapshot_query.show(0), table_snapshot_query.show(0))

        # base table delete: no changes to snapshot
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        assert_resultset_eq(snapshot_query.show(0), table_snapshot_query.show(0))
