import datetime
import filecmp
import io
import json
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pyarrow.parquet as pq
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exprs, metadata, type_system as ts
from pixeltable.dataframe import DataFrameResultSet
from pixeltable.env import Env
from pixeltable.share.packager import TablePackager, TableRestorer
from tests.conftest import clean_db

from ..utils import (
    SAMPLE_IMAGE_URL,
    assert_resultset_eq,
    create_table_data,
    get_image_files,
    get_video_files,
    reload_catalog,
)


class TestPackager:
    def test_packager(self, test_tbl: pxt.Table) -> None:
        packager = TablePackager(test_tbl)
        bundle_path = packager.package()

        # Reinstantiate a catalog to test reads from scratch
        dest = self.__extract_bundle(bundle_path)
        metadata = json.loads((dest / 'metadata.json').read_text())
        self.__validate_metadata(metadata, test_tbl)
        self.__check_parquet_tbl(test_tbl, dest)

    def test_packager_with_views(self, test_tbl: pxt.Table) -> None:
        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir.subdir')
        view = pxt.create_view('test_dir.subdir.test_view', test_tbl)
        view.add_computed_column(vc2=(view.c2 + 1))
        subview = pxt.create_view('test_dir.subdir.test_subview', view.where(view.c2 % 5 == 0))
        subview.add_computed_column(vvc2=(subview.vc2 + 1))
        packager = TablePackager(subview)
        bundle_path = packager.package()

        dest = self.__extract_bundle(bundle_path)
        metadata = json.loads((dest / 'metadata.json').read_text())
        self.__validate_metadata(metadata, subview)
        self.__check_parquet_tbl(test_tbl, dest, scope_tbl=subview)
        self.__check_parquet_tbl(view, dest, scope_tbl=subview)
        self.__check_parquet_tbl(subview, dest, scope_tbl=subview)

    def test_media_packager(self, reset_db: None) -> None:
        t = pxt.create_table('media_tbl', {'image': pxt.Image, 'video': pxt.Video})
        images = get_image_files()[:10]
        videos = get_video_files()[:2]
        t.insert({'image': image} for image in images)
        t.insert({'video': video} for video in videos)
        t.insert(image=SAMPLE_IMAGE_URL)  # Test an image from a remote URL
        # Test a bad image that generates an errormsg
        t.insert(image=get_image_files(include_bad_image=True)[0], on_error='ignore')
        t.add_computed_column(rot=t.image.rotate(90))  # Add a stored computed column to test from media store
        t.add_computed_column(rot2=t.image.rotate(-90), stored=False)  # Add an unstored column
        print(repr(t.select(t.image.fileurl, t.rot.fileurl).collect()))
        print(repr(t.select(t.video.fileurl).collect()))

        packager = TablePackager(t)
        bundle_path = packager.package()

        dest = self.__extract_bundle(bundle_path)
        metadata = json.loads((dest / 'metadata.json').read_text())
        self.__validate_metadata(metadata, t)

        self.__check_parquet_tbl(t, dest, media_dir=(dest / 'media'), expected_cols=13)

    def __extract_bundle(self, bundle_path: Path) -> Path:
        tmp_dir = Path(Env.get().create_tmp_path())
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(tmp_dir)
        return tmp_dir

    def __validate_metadata(self, md: dict, tbl: pxt.Table) -> None:
        assert md['pxt_version'] == pxt.__version__
        assert md['pxt_md_version'] == metadata.VERSION
        assert len(md['md']['tables']) == len(tbl._get_base_tables()) + 1
        for t_md, t in zip(md['md']['tables'], (tbl, *tbl._get_base_tables())):
            assert t_md['table_id'] == str(t._tbl_version.id)

    def __check_parquet_tbl(
        self,
        t: pxt.Table,
        bundle_path: Path,
        media_dir: Optional[Path] = None,
        scope_tbl: Optional[pxt.Table] = None,  # If specified, use instead of `tbl` to select rows
        expected_cols: Optional[int] = None,
    ) -> None:
        parquet_dir = bundle_path / 'tables' / f'tbl_{t._id.hex}'
        parquet_table = pq.read_table(str(parquet_dir))
        parquet_data = parquet_table.to_pandas()

        if expected_cols is not None:
            assert len(parquet_data.columns) == expected_cols

        # Only check columns defined in the table (not ancestors)
        select_exprs: dict[str, exprs.Expr] = {}
        actual_col_types: list[ts.ColumnType] = []
        for col_name, col in t._tbl_version.get().cols_by_name.items():
            if not col.is_stored:
                continue
            if col.col_type.is_media_type():
                select_exprs[col.store_name()] = t[col_name].fileurl
            else:
                select_exprs[col.store_name()] = t[col_name]
            actual_col_types.append(col.col_type)
            if col.stores_cellmd:
                from pixeltable.exprs.column_property_ref import ColumnPropertyRef

                # This is not available in the user-facing API, but we use it for testing.
                select_exprs[col.cellmd_store_name()] = exprs.ColumnPropertyRef(
                    t[col_name], ColumnPropertyRef.Property.CELLMD
                )
                actual_col_types.append(col.cellmd_type())

        scope_tbl = scope_tbl or t
        pxt_data = scope_tbl.select(**select_exprs).collect()
        for col, col_type in zip(select_exprs.keys(), actual_col_types):
            print(f'Checking column: {col}')
            pxt_values: list = pxt_data[col]
            parquet_values = list(parquet_data[col])
            if col_type.is_array_type():
                parquet_values = [np.load(io.BytesIO(val)) for val in parquet_values]
                for pxt_val, parquet_val in zip(pxt_values, parquet_values):
                    assert np.array_equal(pxt_val, parquet_val)
            elif col_type.is_json_type():
                # JSON columns were exported as strings; check that they parse properly
                assert pxt_values == [json.loads(val) if val is not None else None for val in parquet_values]
            elif col_type.is_media_type():
                assert media_dir is not None
                self.__check_media(pxt_values, parquet_values, media_dir)
            else:
                assert pxt_values == parquet_values

    def __check_media(self, pxt_values: list, parquet_values: list, media_dir: Path) -> None:
        for pxt_val, parquet_val in zip(pxt_values, parquet_values):
            if pxt_val is None:
                assert parquet_val is None
                continue
            assert isinstance(pxt_val, str)
            assert isinstance(parquet_val, str)
            parsed_url = urllib.parse.urlparse(pxt_val)
            if parsed_url.scheme == 'file':
                assert parquet_val.startswith('pxtmedia://')
                path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_url.path)))
                bundled_path = media_dir / parquet_val.removeprefix('pxtmedia://')
                assert bundled_path.exists(), bundled_path
                assert filecmp.cmp(path, bundled_path)
            else:
                assert pxt_val == parquet_val

    class BundleInfo(NamedTuple):
        """
        Saved information about a bundle and its associated table. This is used for testing to track various
        information about the expected behavior of the bundle after it is restored.
        """

        bundle_path: Path  # Path of the bundle on disk
        depth: int  # Depth of the table in the table hierarchy (= length of the table's TableVersionPath)
        schema: dict[str, ts.ColumnType]  # Schema of the table
        result_set: DataFrameResultSet  # Resultset corresponding to the query `tbl.head(n=5000)`

    def __package_table(self, tbl: pxt.Table) -> BundleInfo:
        """
        Runs the query `tbl.head(n=5000)`, packages the table into a bundle, and returns a BundleInfo.
        """
        schema = tbl._get_schema()
        depth = tbl._tbl_version_path.path_len()
        result_set = tbl.head(n=5000)

        # Package the snapshot into a tarball
        packager = TablePackager(tbl)
        bundle_path = packager.package()

        return TestPackager.BundleInfo(bundle_path, depth, schema, result_set)

    def __restore_and_check_table(self, bundle_info: 'TestPackager.BundleInfo', tbl_name: str) -> None:
        """
        Restores the table that was packaged in `bundle_info` and validates its contents against the tracked data.
        """
        restorer = TableRestorer(tbl_name)
        restorer.restore(bundle_info.bundle_path)
        self.__check_table(bundle_info, tbl_name)

    def __check_table(self, bundle_info: 'TestPackager.BundleInfo', tbl_name: str) -> None:
        t = pxt.get_table(tbl_name)
        assert bundle_info.schema == t._get_schema()
        assert bundle_info.depth == t._tbl_version_path.path_len()
        reconstituted_data = t.head(n=5000)
        assert_resultset_eq(bundle_info.result_set, reconstituted_data)

    def __do_round_trip(self, tbl: pxt.Table) -> None:
        bundle = self.__package_table(tbl)
        clean_db()
        reload_catalog()
        self.__restore_and_check_table(bundle, 'new_replica')

    def test_round_trip(self, test_tbl: pxt.Table) -> None:
        """package() / restore() round trip for a single snapshot"""
        # Add some additional columns to test various additional datatypes
        t = test_tbl
        t.add_column(dt=pxt.Date)
        t.update({'dt': pxtf.date.add_days(datetime.date(2025, 1, 1), t.c2)})
        t.add_column(arr1=pxt.Array[pxt.Float, (1, 3)])  # type: ignore[misc]
        t.update({'arr1': pxt.array([[1.7, 2.32, t.c3]])})
        t.add_column(arr2=pxt.Array[pxt.String])  # type: ignore[misc]
        t.update({'arr2': pxt.array(['xyz', t.c1])})

        snapshot = pxt.create_snapshot('snapshot', t)
        self.__do_round_trip(snapshot)

    def test_non_snapshot_round_trip(self, reset_db: None) -> None:
        """package() / restore() round trip for multiple versions of a table that is not a snapshot"""
        t = pxt.create_table('tbl', {'int_col': pxt.Int})
        t.insert({'int_col': i} for i in range(200))

        bundle1 = self.__package_table(t)

        t.add_column(str_col=pxt.String)
        t.insert({'int_col': i} for i in range(200, 400))
        t.where(t.int_col % 2 == 0).update({'str_col': pxtf.string.format('string {0}', t.int_col)})

        bundle2 = self.__package_table(t)

        clean_db()
        reload_catalog()

        self.__restore_and_check_table(bundle1, 'replica')
        self.__restore_and_check_table(bundle2, 'replica')

    def test_media_round_trip(self, img_tbl: pxt.Table) -> None:
        snapshot = pxt.create_snapshot('snapshot', img_tbl)
        self.__do_round_trip(snapshot)

    def test_views_round_trip(self, test_tbl: pxt.Table) -> None:
        v1 = pxt.create_view('v1', test_tbl, additional_columns={'x1': pxt.Int})
        v1.update({'x1': test_tbl.c2 * 10})
        v2 = pxt.create_view('v2', v1.where(v1.c2 % 3 == 0), additional_columns={'x2': pxt.Int})
        v2.update({'x2': v1.x1 + 8})
        snapshot = pxt.create_snapshot('snapshot', v2)
        self.__do_round_trip(snapshot)

    def test_iterator_view_round_trip(self, reset_db: None) -> None:
        t = pxt.create_table('base_tbl', {'video': pxt.Video})
        t.insert({'video': video} for video in get_video_files()[:2])

        v = pxt.create_view('frames_view', t, iterator=pxt.iterators.FrameIterator.create(video=t.video, fps=1))
        # Add a stored computed column that will generate a bunch of media files in the view.
        v.add_computed_column(rot_frame=v.frame.rotate(180))
        snapshot = pxt.create_snapshot('snapshot', v)
        snapshot_row_count = snapshot.count()

        self.__do_round_trip(snapshot)

        # Double-check that the iterator view and its base table have the correct number of rows
        snapshot_replica = pxt.get_table('new_replica')
        assert snapshot_replica._snapshot_only
        assert snapshot_replica.count() == snapshot_row_count
        v_replica = snapshot_replica.get_base_table()
        assert v_replica.count() == snapshot_row_count
        t_replica = v_replica.get_base_table()
        assert t_replica.count() == 2

    def test_multi_view_round_trip_1(self, reset_db: None) -> None:
        """
        Simplest multi-view test: two snapshots that are exported at the same time.
        (All v_min/v_max values are consistent in the bundles.)
        """

        t = pxt.create_table('base_tbl', {'int_col': pxt.Int})
        t.insert({'int_col': i} for i in range(200))

        snap1 = pxt.create_snapshot('snap1', t.where(t.int_col % 5 == 0))
        t.add_column(str_col=pxt.String)
        t.insert({'int_col': i} for i in range(200, 400))
        t.where(t.int_col % 3 == 0).update({'str_col': pxtf.string.format('string {0}', t.int_col)})

        snap2 = pxt.create_snapshot('snap2', t.where(t.int_col % 7 == 0))

        bundle1 = self.__package_table(snap1)
        bundle2 = self.__package_table(snap2)

        clean_db()
        reload_catalog()

        self.__restore_and_check_table(bundle1, 'replica1')
        self.__restore_and_check_table(bundle2, 'replica2')

    def test_multi_view_round_trip_2(self, reset_db: None) -> None:
        """
        Two snapshots that are exported at different times, requiring rectification of the v_max values.
        """
        t = pxt.create_table('base_tbl', {'int_col': pxt.Int})
        t.insert({'int_col': i} for i in range(200))

        snap1 = pxt.create_snapshot('snap1', t.where(t.int_col % 3 == 0))
        bundle1 = self.__package_table(snap1)

        t.add_column(str_col=pxt.String)
        t.insert({'int_col': i} for i in range(200, 400))
        t.where(t.int_col % 2 == 0).update({'str_col': pxtf.string.format('string {0}', t.int_col)})

        snap2 = pxt.create_snapshot('snap2', t.where(t.int_col % 5 == 0))
        bundle2 = self.__package_table(snap2)

        clean_db()
        reload_catalog()

        self.__restore_and_check_table(bundle1, 'replica1')
        self.__restore_and_check_table(bundle2, 'replica2')

    @pytest.mark.parametrize('pure_snapshots', [False, True])
    def test_multi_view_round_trip_3(self, reset_db: None, pure_snapshots: bool) -> None:
        """
        Two snapshots that are exported at different times, involving column operations.
        """
        t = pxt.create_table('base_tbl', {'int_col': pxt.Int})
        t.insert({'int_col': i} for i in range(100))

        snap1 = pxt.create_snapshot('snap1', t if pure_snapshots else t.where(t.int_col % 3 == 0))
        bundle1 = self.__package_table(snap1)

        t.add_computed_column(int_col_2=(t.int_col + 5))
        t.insert({'int_col': i} for i in range(100, 200))

        snap2 = pxt.create_snapshot('snap2', t if pure_snapshots else t.where(t.int_col % 5 == 0))
        bundle2 = self.__package_table(snap2)

        clean_db()
        reload_catalog()

        self.__restore_and_check_table(bundle1, 'replica1')
        self.__restore_and_check_table(bundle2, 'replica2')

    def test_multi_view_round_trip_4(self, all_datatypes_tbl: pxt.Table) -> None:
        """
        Snapshots that involve all the different column types.
        """
        t = all_datatypes_tbl
        snap1 = pxt.create_snapshot('snap1', t.where(t.row_id % 2 != 0))
        bundle1 = self.__package_table(snap1)

        more_data = create_table_data(t, num_rows=22)
        t.insert(more_data[11:])

        snap2 = pxt.create_snapshot('snap2', t.where(t.row_id % 3 != 0))
        bundle2 = self.__package_table(snap2)

        clean_db()
        reload_catalog()

        self.__restore_and_check_table(bundle1, 'replica1')
        self.__restore_and_check_table(bundle2, 'replica2')

    def test_multi_view_round_trip_5(self, reset_db: None) -> None:
        """
        A much more sophisticated multi-view test. Here we create 11 snapshots, each one modifying a
        different subset of the rows in the table. The snapshots are then reconstituted in an arbitrary
        order.
        """
        bundles: list[TestPackager.BundleInfo] = []

        t = pxt.create_table('base_tbl', {'row_number': pxt.Int, 'value': pxt.Int})
        t.insert({'row_number': i} for i in range(1024))
        bundles.append(self.__package_table(pxt.create_snapshot('snap', t)))

        for n in range(10):
            t.where(t.row_number.bitwise_and(2**n) != 0).update({'value': n})
            bundles.append(self.__package_table(pxt.create_snapshot(f'snap_{n}', t)))

        clean_db()
        reload_catalog()

        for n in (7, 3, 0, 9, 4, 10, 1, 5, 8):
            # Snapshots 2 and 6 are intentionally never restored.
            self.__restore_and_check_table(bundles[n], f'replica_{n}')

        # Check all the tables again to verify that everything is consistent.
        for n in (0, 1, 3, 4, 5, 7, 8, 9, 10):
            self.__check_table(bundles[n], f'replica_{n}')

    def test_multi_view_round_trip_6(self, reset_db: None) -> None:
        """
        Another test with many snapshots, involving row and column additions and deletions.
        """
        bundles: list[TestPackager.BundleInfo] = []

        t = pxt.create_table('base_tbl', {'row_number': pxt.Int, 'value': pxt.Int})
        t.insert({'row_number': i} for i in range(1024))
        bundles.append(self.__package_table(pxt.create_snapshot('snap', t)))

        for n in range(10):
            t.insert({'row_number': i} for i in range(1024 + 64 * n, 1024 + 64 * (n + 1)))
            t.add_computed_column(**{f'new_col_{n}': t.value * n})
            t.where(t.row_number.bitwise_and(2**n) != 0).update({'value': n})
            t.where(t.row_number < 32 * n).delete()
            if n >= 5:
                t.drop_column(f'new_col_{n - 5}')
            bundles.append(self.__package_table(pxt.create_snapshot(f'snap_{n}', t)))

        clean_db()
        reload_catalog()

        for n in (7, 3, 0, 9, 4, 10, 1, 5, 8):
            # Snapshots 2 and 6 are intentionally never restored.
            self.__restore_and_check_table(bundles[n], f'replica_{n}')

        # Check all the tables again to verify that everything is consistent.
        for n in (0, 1, 3, 4, 5, 7, 8, 9, 10):
            self.__check_table(bundles[n], f'replica_{n}')

    def test_multi_view_non_snapshot_round_trip(self, reset_db: None) -> None:
        """
        A similar test, this one involving multiple versions of a table that is not a snapshot,
        intermixed with various snapshots.
        """
        bundles: list[TestPackager.BundleInfo] = []

        t = pxt.create_table('base_tbl', {'row_number': pxt.Int, 'value': pxt.Int})
        t.insert({'row_number': i} for i in range(1024))
        bundles.append(self.__package_table(pxt.create_snapshot('snap', t)))

        for n in range(1, 11):
            t.insert({'row_number': i} for i in range(1024 + 64 * n, 1024 + 64 * (n + 1)))
            t.add_computed_column(**{f'new_col_{n}': t.value * n})
            t.where(t.row_number.bitwise_and(2**n) != 0).update({'value': n})
            t.where(t.row_number < 32 * n).delete()
            if n >= 6:
                t.drop_column(f'new_col_{n - 5}')
            if n % 2 == 0:
                # Even-numbered iterations create a snapshot
                bundles.append(self.__package_table(pxt.create_snapshot(f'snap_{n}', t)))
            else:
                # Odd-numbered iterations just package the table directly
                bundles.append(self.__package_table(t))

        clean_db()
        reload_catalog()

        # Restore the odd-numbered bundles (directly packaged table versions). This needs to be done in order
        # currently, because we don't have a way to get a handle to an older version of a table.
        # TODO: Randomize the order once we have such a feature.
        for n in (1, 3, 7, 9):
            self.__restore_and_check_table(bundles[n], 'replica')

        for n in (4, 0, 8, 10, 6):
            self.__restore_and_check_table(bundles[n], f'replica_{n}')
