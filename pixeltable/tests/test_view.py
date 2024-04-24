import datetime
import logging

import PIL
import pytest

import pixeltable as pxt
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable.tests.utils import create_test_tbl, assert_resultset_eq
from pixeltable.type_system import IntType, FloatType, ImageType

logger = logging.getLogger('pixeltable')

class TestView:
    """
    TODO:
    - test tree of views
    - test consecutive component views

    """
    def create_tbl(self, cl: pxt.Client) -> catalog.InsertableTable:
        """Create table with computed columns"""
        t = create_test_tbl(cl)
        t.add_column(d1=t.c3 - 1)
        # add column that can be updated
        t.add_column(c10=FloatType(nullable=True))
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_column(d2=t.c3 - t.c10)
        return t

    def test_basic(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        schema = {
            'v1': t.c3 * 2.0,
            'v2': t.c6.f5,
        }
        v = cl.create_view('test_view', t, schema=schema, filter=t.c2 < 10)
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
        v.add_column(v3=v.v1 * 2.0)
        v.add_column(v4=v.v2[0])

        def check_view(t: pxt.Table, v: pxt.Table) -> None:
            assert v.count() == t.where(t.c2 < 10).count()
            assert_resultset_eq(
                v.select(v.v1).order_by(v.c2).collect(),
                t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2).collect())
            assert_resultset_eq(
                v.select(v.v3).order_by(v.c2).collect(),
                t.select(t.c3 * 4.0).where(t.c2 < 10).order_by(t.c2).collect())
            assert_resultset_eq(
                v.select(v.v4).order_by(v.c2).collect(),
                t.select(t.c6.f5[0]).where(t.c2 < 10).order_by(t.c2).collect())
        check_view(t, v)

        # check view md after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')
        check_view(t, v)

        view_query = v.select(v.v1).order_by(v.c2)
        base_query = t.select(t.c3 * 2.0).where(t.c2 < 10).order_by(t.c2)

        # insert data: of 20 new rows, only 10 are reflected in the view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        status = t.insert(rows)
        assert status.num_rows == 30
        assert t.count() == 120
        check_view(t, v)

        # update data: cascade to view
        status = t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert status.num_rows == 10 * 2  # *2: rows affected in both base table and view
        assert t.count() == 120
        check_view(t, v)

        # base table delete is reflected in view
        status = t.delete(where=t.c2 < 5)
        status.num_rows == 10 * 2  # *2: rows affected in both base table and view
        assert t.count() == 110
        check_view(t, v)

        # test delete view
        cl.drop_table('test_view')
        with pytest.raises(excs.Error) as exc_info:
            _ = cl.get_table('test_view')
        assert 'No such path:' in str(exc_info.value)
        cl = pxt.Client(reload=True)
        # still true after reload
        with pytest.raises(excs.Error) as exc_info:
            _ = cl.get_table('test_view')
        assert 'No such path:' in str(exc_info.value)

        t = cl.get_table('test_tbl')
        with pytest.raises(excs.Error) as exc_info:
            _ = cl.create_view('lambda_view', t, schema={'v1': lambda c3: c3 * 2.0})
        assert 'computed with a callable' in str(exc_info.value).lower()

    def test_parallel_views(self, test_client: pxt.Client) -> None:
        """Two views over the same base table, with non-overlapping filters"""
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        v1 = cl.create_view('v1', t, schema={'v1': t.c3 * 2}, filter=t.c2 < 10)
        # create another view with a non-overlapping filter and computed columns
        v2 = cl.create_view('v2', t, schema={'v1': t.c3 * 3}, filter=(t.c2 < 20) & (t.c2 >= 10))

        # sanity checks
        v1_query = v1.select(v1.v1).order_by(v1.c2)
        v2_query = v2.select(v2.v1).order_by(v2.c2)
        b1_query = t.select(t.c3 * 2).where(t.c2 < 10).order_by(t.c2)
        b2_query = t.select(t.c3 * 3).where((t.c2 >= 10) & (t.c2 < 20)).order_by(t.c2)
        assert_resultset_eq(v1_query.collect(), b1_query.collect())
        assert_resultset_eq(v2_query.collect(), b2_query.collect())

        # insert data: of 20 new rows, only 10 show up in each view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        status = t.insert(rows)
        assert status.num_rows == 40
        assert t.count() == 120
        assert v1.count() == 20
        assert v2.count() == 20
        assert_resultset_eq(v1_query.collect(), b1_query.collect())
        assert_resultset_eq(v2_query.collect(), b2_query.collect())

        # update data: cascade to views
        status = t.update(
            {'c4': True, 'c3': t.c3 + 1, 'c10': t.c10 - 1.0}, where=(t.c2 >= 5) & (t.c2 < 15), cascade=True)
        assert status.num_rows == 20 * 2  # *2: rows affected in both base table and view
        assert t.count() == 120
        assert v1.count() == 20
        assert v2.count() == 20
        assert_resultset_eq(v1_query.collect(), b1_query.collect())
        assert_resultset_eq(v2_query.collect(), b2_query.collect())


        # base table delete is reflected in view
        status = t.delete(where=(t.c2 >= 5) & (t.c2 < 15))
        status.num_rows == 10 * 2  # *2: rows affected in both base table and view
        assert t.count() == 100
        assert v1.count() == 10
        assert v2.count() == 10
        assert_resultset_eq(v1_query.collect(), b1_query.collect())
        assert_resultset_eq(v2_query.collect(), b2_query.collect())

    def test_chained_views(self, test_client: pxt.Client) -> None:
        """Two views, the second one is a view over the first one"""
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        v1 = cl.create_view('v1', t, schema={'col1': t.c3 * 2}, filter=t.c2 < 10)
        # create a view on top of v1
        v2_schema = {
            'col2': t.c3 * 3,  # only base
            'col3': v1.col1 / 2,  # only v1
            'col4': t.c10 + v1.col1,  # both base and v1
        }
        v2 = cl.create_view('v2', v1, schema=v2_schema, filter=t.c2 < 5)

        def check_views():
            assert_resultset_eq(
                v1.select(v1.col1).order_by(v1.c2).collect(),
                t.select(t.c3 * 2).where(t.c2 < 10).order_by(t.c2).collect())
            assert_resultset_eq(
                v2.select(v2.col1).order_by(v2.c2).collect(),
                v1.select(v1.col1).where(v1.c2 < 5).order_by(v1.c2).collect())
            assert_resultset_eq(
                v2.select(v2.col2).order_by(v2.c2).collect(),
                t.select(t.c3 * 3).where(t.c2 < 5).order_by(t.c2).collect())
            assert_resultset_eq(
                v2.select(v2.col3).order_by(v2.c2).collect(),
                v1.select(v1.col1 / 2).where(v1.c2 < 5).order_by(v2.c2).collect())
            assert_resultset_eq(
                v2.select(v2.col4).order_by(v2.c2).collect(),
                v1.select(v1.c10 + v1.col1).where(v1.c2 < 5).order_by(v1.c2).collect())
                #t.select(t.c10 * 2).where(t.c2 < 5).order_by(t.c2).collect())
        check_views()

        # insert data: of 20 new rows; 10 show up in v1, 5 in v2
        base_version, v1_version, v2_version = t.version(), v1.version(), v2.version()
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        status = t.insert(rows)
        assert status.num_rows == 20 + 10 + 5
        assert t.count() == 120
        assert v1.count() == 20
        assert v2.count() == 10
        # all versions were incremented
        assert t.version() == base_version + 1
        assert v1.version() == v1_version + 1
        assert v2.version() == v2_version + 1
        check_views()

        # update data: cascade to both views
        base_version, v1_version, v2_version = t.version(), v1.version(), v2.version()
        status = t.update({'c4': True, 'c3': t.c3 + 1}, where=t.c2 < 15, cascade=True)
        assert status.num_rows == 30 + 20 + 10
        assert t.count() == 120
        # all versions were incremented
        assert t.version() == base_version + 1
        assert v1.version() == v1_version + 1
        assert v2.version() == v2_version + 1
        check_views()

        # update data: cascade only to v2
        base_version, v1_version, v2_version = t.version(), v1.version(), v2.version()
        status = t.update({'c10': t.c10 - 1.0}, where=t.c2 < 15, cascade=True)
        assert status.num_rows == 30 + 10
        assert t.count() == 120
        # v1 did not get updated
        assert t.version() == base_version + 1
        assert v1.version() == v1_version
        assert v2.version() == v2_version + 1
        check_views()

        # base table delete is reflected in both views
        base_version, v1_version, v2_version = t.version(), v1.version(), v2.version()
        status = t.delete(where=t.c2 == 0)
        status.num_rows == 1 + 1 + 1
        assert t.count() == 118
        assert v1.count() == 18
        assert v2.count() == 8
        # all versions were incremented
        assert t.version() == base_version + 1
        assert v1.version() == v1_version + 1
        assert v2.version() == v2_version + 1
        check_views()

        # base table delete is reflected only in v1
        base_version, v1_version, v2_version = t.version(), v1.version(), v2.version()
        status = t.delete(where=t.c2 == 5)
        status.num_rows == 1 + 1
        assert t.count() == 116
        assert v1.count() == 16
        assert v2.count() == 8
        # v2 was not updated
        assert t.version() == base_version + 1
        assert v1.version() == v1_version + 1
        assert v2.version() == v2_version
        check_views()

    def test_unstored_columns(self, test_client: pxt.Client) -> None:
        """Test chained views with unstored columns"""
        # create table with image column and two updateable int columns
        cl = test_client
        schema = {
            'img': ImageType(),
            'int1': IntType(nullable=False),
            'int2': IntType(nullable=False),
        }
        t = cl.create_table('test_tbl', schema)
        # populate table with images of a defined size
        width, height = 100, 100
        rows = [
            {
                'img': PIL.Image.new('RGB', (width, height), color=(0, 0, 0)).tobytes('jpeg', 'RGB'),
                'int1': i,
                'int2': i,
            }
            for i in range(100)
        ]
        t.insert(rows)

        # view with unstored column that depends on int1 and a manually updated column (int4)
        v1_schema = {
            'img2': {
                'value': t.img.crop([t.int1, t.int1, width, height]),
                'stored': False,
            },
            'int3': t.int1 * 2,
            'int4': IntType(nullable=True),  # TODO: add default
        }
        logger.debug('******************* CREATE V1')
        v1 = cl.create_view('v1', t, schema=v1_schema)
        v1.update({'int4': 1})
        _ = v1.select(v1.img2.width, v1.img2.height).collect()

        # view with stored column that depends on t and view1
        v2_schema = {
            'img3': {
                # use the actual width and height of the image (not 100, which will pad the image)
                'value': v1.img2.crop([t.int1 + t.int2, v1.int3 + v1.int4, v1.img2.width, v1.img2.height]),
                'stored': True,
              },
        }
        logger.debug('******************* CREATE V2')
        v2 = cl.create_view('v2', v1, schema=v2_schema, filter=v1.int1 < 10)

        def check_views() -> None:
            assert_resultset_eq(
                v1.select(v1.img2.width, v1.img2.height).order_by(v1.int1).collect(),
                t.select(t.img.width - t.int1, t.img.height - t.int1).order_by(t.int1).collect())
            assert_resultset_eq(
                v2.select(v2.img3.width, v2.img3.height).order_by(v2.int1).collect(),
                v1.select(v1.img2.width - v1.int1 - v1.int2, v1.img2.height - v1.int3 - v1.int4)\
                    .where(v1.int1 < 10).order_by(v1.int1).collect())
        check_views()

        logger.debug('******************* INSERT')
        status = t.insert(rows, fail_on_exception=False)
        v1.update({'int4': 1}, where=v1.int4 == None)
        logger.debug('******************* POST INSERT')
        check_views()

        # update int1:
        # - cascades to v1 and v2
        # - removes a row from v2 (only 9 rows in t now qualify)
        logger.debug('******************* UPDATE INT1')
        t.update({'int1': t.int1 + 1})
        logger.debug('******************* POST UPDATE INT1')
        check_views()

        # update int2:
        # - cascades only to v2
        # - but requires join against v1 to access int4
        # TODO: but requires join against v1 to access int3 and int4
        logger.debug('******************* UPDATE INT2')
        t.update({'int2': t.int2 + 1})
        logger.debug('******************* POST UPDATE INT2')
        check_views()

    def test_computed_cols(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = self.create_tbl(cl)

        # create view with computed columns
        schema = {
            'v1': t.c3 * 2.0,
            'v2': t.c6.f5,
        }
        v = cl.create_view('test_view', t, schema=schema)
        assert_resultset_eq(
            v.select(v.v1).order_by(v.c2).show(0),
            t.select(t.c3 * 2.0).order_by(t.c2).show(0))
        # computed columns that don't reference the base table
        v.add_column(v3=v.v1 * 2.0)
        v.add_column(v4=v.v2[0])

        # use view md after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')

        # insert data
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).collect())
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

    def test_filter(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = create_test_tbl(cl)

        # create view with filter
        v = cl.create_view('test_view', t, filter=t.c2 < 10)
        assert_resultset_eq(
            v.order_by(v.c2).show(0),
            t.where(t.c2 < 10).order_by(t.c2).show(0))

        # use view md after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        v = cl.get_table('test_view')

        # insert data: of 20 new rows, only 10 are reflected in the view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7).where(t.c2 < 20).collect())
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

        # create views with filters containing date and datetime
        _ = cl.create_view('test_view_2', t, filter=t.c5 >= datetime.date.today())
        _ = cl.create_view('test_view_3', t, filter=t.c5 < datetime.datetime.now())

    def test_view_of_snapshot(self, test_client: pxt.Client) -> None:
        """Test view over a snapshot"""
        cl = test_client
        t = self.create_tbl(cl)
        snap = cl.create_view('test_snap', t, is_snapshot=True)

        # create view with filter and computed columns
        schema = {
            'v1': snap.c3 * 2.0,
            'v2': snap.c6.f5,
        }
        v = cl.create_view('test_view', snap, schema=schema, filter=snap.c2 < 10)

        def check_view(s: pxt.Table, v: pxt.Table) -> None:
            assert v.count() == s.where(s.c2 < 10).count()
            assert_resultset_eq(
                v.select(v.v1).order_by(v.c2).collect(),
                s.select(s.c3 * 2.0).where(s.c2 < 10).order_by(s.c2).collect())
            assert_resultset_eq(
                v.select(v.v2).order_by(v.c2).collect(),
                s.select(s.c6.f5).where(s.c2 < 10).order_by(s.c2).collect())

        check_view(snap, v)
        # computed columns that don't reference the base table
        v.add_column(v3=v.v1 * 2.0)
        v.add_column(v4=v.v2[0])
        assert v.count() == t.where(t.c2 < 10).count()

        # use view md after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        snap = cl.get_table('test_snap')
        v = cl.get_table('test_view')

        # insert data: no changes to view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        t.insert(rows)
        assert t.count() == 120
        check_view(snap, v)

        # update data: no changes to view
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        check_view(snap, v)

        # base table delete: no change to view
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        check_view(snap, v)

    def test_snapshots(self, test_client: pxt.Client) -> None:
        """Test snapshot of a view of a snapshot"""
        cl = test_client
        t = self.create_tbl(cl)
        s = cl.create_view('test_snap', t, is_snapshot=True)
        assert s.select(s.c2).order_by(s.c2).collect()['c2'] == t.select(t.c2).order_by(t.c2).collect()['c2']

        with pytest.raises(excs.Error) as exc_info:
            v = cl.create_view('test_view', s, schema={'v1': t.c3 * 2.0})
        assert 'value expression cannot be computed in the context of the base test_snap' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            v = cl.create_view('test_view', s, filter=t.c2 < 10)
        assert 'filter cannot be computed in the context of the base test_snap' in str(exc_info.value).lower()

        # create view with filter and computed columns
        schema = {
            'v1': s.c3 * 2.0,
            'v2': s.c6.f5,
        }
        v = cl.create_view('test_view', s, schema=schema, filter=s.c2 < 10)
        orig_view_cols = v.column_names()
        view_s = cl.create_view('test_view_snap', v, is_snapshot=True)
        assert set(view_s.column_names()) == set(orig_view_cols)

        def check(s1: pxt.Table, v: pxt.Table, s2: pxt.Table) -> None:
            assert s1.where(s1.c2 < 10).count() == v.count()
            assert v.count() == s2.count()
            assert_resultset_eq(
                s1.select(s1.c3 * 2.0, s1.c6.f5).where(s1.c2 < 10).order_by(s1.c2).collect(),
                v.select(v.v1, v.v2).order_by(v.c2).collect())
            assert_resultset_eq(
                v.select(v.c3, v.c6, v.v1, v.v2).order_by(v.c2).collect(),
                s2.select(s2.c3, s2.c6, s2.v1, s2.v2).order_by(s2.c2).collect())
        check(s, v, view_s)

        # add more columns
        v.add_column(v3=v.v1 * 2.0)
        v.add_column(v4=v.v2[0])
        check(s, v, view_s)
        assert set(view_s.column_names()) == set(orig_view_cols)

        # check md after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        view_s = cl.get_table('test_view_snap')
        check(s, v, view_s)
        assert set(view_s.column_names()) == set(orig_view_cols)

        # insert data: no changes to snapshot
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        t.insert(rows)
        assert t.count() == 120
        check(s, v, view_s)

        # update data: no changes to snapshot
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        assert t.count() == 120
        check(s, v, view_s)

        # base table delete: no changes to snapshot
        t.delete(where=t.c2 < 5)
        assert t.count() == 110
        check(s, v, view_s)
