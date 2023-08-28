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
from pixeltable.tests.utils import create_test_tbl, assert_resultset_eq


class TestView:
    def create_tbl(self, cl: pt.Client) -> catalog.MutableTable:
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
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).show(0))
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

        # insert data: of 20 new rows, only 10 are reflected in the view
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).show(0).rows
        t.insert(rows)
        assert t.count() == 120
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).show(0))

        # update data: cascade to view
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).show(0))

        # base table delete is reflected in view
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).show(0))

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
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).show(0).rows
        t.insert(rows)
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
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7).where(t.c2 < 20).show(0).rows
        t.insert(rows)
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
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).show(0).rows
        t.insert(rows)
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

    def test_view_snapshot(self, test_client: pt.Client) -> None:
        """Test snapshot of a view"""
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        cols = [
            catalog.Column('v1', computed_with=t.c3 * 2.0),
            catalog.Column('v2', computed_with=t.c6.f5)
        ]
        v = cl.create_view('test_view', t, schema=cols, filter=t.c2 < 10)
        s = cl.create_snapshot('test_snap', 'test_view')
        snapshot_query = s.order_by(s.c2)
        table_query = \
            t.select(t.c3 * 2.0, t.c6.f5, t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c8, t.d1, t.c10, t.d2) \
                .where(t.c2 < 10).order_by(t.c2)
        assert_resultset_eq(snapshot_query.show(0), table_query.show(0))
        # add more columns
        v.add_column(catalog.Column('v3', computed_with=v.v1 * 2.0))
        v.add_column(catalog.Column('v4', computed_with=v.v2[0]))
        assert_resultset_eq(snapshot_query.show(0), table_query.show(0))

        # check md after reload
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        s_old = s
        s = cl.get_table('test_snap')
        assert s.tbl_version.predicate == s_old.tbl_version.predicate
        assert set(s_old.tbl_version.cols_by_name.keys()) == set(s.tbl_version.cols_by_name.keys())
        assert s.tbl_version.cols_by_name['v1'].value_expr == t.c3 * 2.0

        # insert data: snapshot sees new rows
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).show(0).rows
        t.insert(rows)
        assert t.count() == 120
        assert_resultset_eq(snapshot_query.show(0), table_query.show(0))

        # update data: snapshot sees changes
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        assert_resultset_eq(snapshot_query.show(0), table_query.show(0))

        # base table delete: snapshot sees changes
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        assert_resultset_eq(snapshot_query.show(0), table_query.show(0))

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
        rows = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).show(0).rows
        t.insert(rows)
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

