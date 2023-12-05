import pytest
import logging

import PIL

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import create_test_tbl, assert_resultset_eq, get_video_files
from pixeltable.iterators import FrameIterator


logger = logging.getLogger('pixeltable')

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

    def test_parallel_views(self, test_client: pt.Client) -> None:
        """Two views over the same base table, with non-overlapping filters"""
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        v1 = cl.create_view('v1', t, schema=[catalog.Column('v1', computed_with=t.c3 * 2)], filter=t.c2 < 10)
        # create another view with a non-overlapping filter and computed columns
        v2 = cl.create_view(
            'v2', t, schema=[catalog.Column('v1', computed_with=t.c3 * 3)], filter=(t.c2 < 20) & (t.c2 >= 10))

        # sanity checks
        v1_query = v1.select(v1.v1).order_by(v1.c2)
        v2_query = v2.select(v2.v1).order_by(v2.c2)
        b1_query = t.select(t.c3 * 2).where(t.c2 < 10).order_by(t.c2)
        b2_query = t.select(t.c3 * 3).where((t.c2 >= 10) & (t.c2 < 20)).order_by(t.c2)
        assert_resultset_eq(v1_query.collect(), b1_query.collect())
        assert_resultset_eq(v2_query.collect(), b2_query.collect())

        # insert data: of 20 new rows, only 10 show up in each view
        rows = list(t.select(t.c1, t.c1n, t.c2, t.c3, t.c4, t.c5, t.c6, t.c7, t.c10).where(t.c2 < 20).collect())
        status = t.insert([list(r.values()) for r in rows])
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

    def test_chained_views(self, test_client: pt.Client) -> None:
        """Two views, the second one is a view over the first one"""
        cl = test_client
        t = self.create_tbl(cl)

        # create view with filter and computed columns
        v1 = cl.create_view('v1', t, schema=[catalog.Column('col1', computed_with=t.c3 * 2)], filter=t.c2 < 10)
        # create a view on top of v1
        v2_schema = [
            catalog.Column('col2', computed_with=t.c3 * 3),  # only base
            catalog.Column('col3', computed_with=v1.col1 / 2),  # only v1
            catalog.Column('col4', computed_with=t.c10 + v1.col1),  # both base and v1
        ]
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
        status = t.insert([list(r.values()) for r in rows])
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

    def test_unstored_columns(self, test_client: pt.Client) -> None:
        """Test chained views with unstored columns"""
        # create table with image column and two updateable int columns
        cl = test_client
        cols = [
            catalog.Column('img', ImageType()),
            catalog.Column('int1', IntType(nullable=False)),
            catalog.Column('int2', IntType(nullable=False))
        ]
        t = cl.create_table('test_tbl', cols)
        # populate table with images of a defined size
        width, height = 100, 100
        rows = [
            [PIL.Image.new('RGB', (width, height), color=(0, 0, 0)).tobytes('jpeg', 'RGB') , i, i] for i in range(100)
        ]
        t.insert(rows)

        # view with unstored column that depends on int1 and a manually updated column (int4)
        v1_cols = [
            catalog.Column('img2', computed_with=t.img.crop([t.int1, t.int1, width, height]), stored=False),
            catalog.Column('int3', computed_with=t.int1 * 2),
            catalog.Column('int4', IntType(nullable=True)),  # TODO: add default
        ]
        logger.debug('******************* CREATE V1')
        v1 = cl.create_view('v1', t, schema=v1_cols)
        v1.update({'int4': 1})
        _ = v1.select(v1.img2.width, v1.img2.height).collect()

        # view with stored column that depends on t and view1
        v2_cols = [
            catalog.Column(
                'img3',
                # use the actual width and height of the image (not 100, which will pad the image)
                computed_with=v1.img2.crop([t.int1 + t.int2, v1.int3 + v1.int4, v1.img2.width, v1.img2.height]),
                stored=True),
        ]
        logger.debug('******************* CREATE V2')
        v2 = cl.create_view('v2', v1, schema=v2_cols, filter=v1.int1 < 10)

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
        t.insert(rows)
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
