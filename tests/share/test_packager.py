import datetime
import filecmp
import io
import json
import platform
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exprs, metadata, type_system as ts
from pixeltable.catalog import Catalog
from pixeltable.catalog.table_version import TableVersionMd
from pixeltable.env import Env
from pixeltable.index.embedding_index import EmbeddingIndex
from pixeltable.metadata import schema
from pixeltable.plan import FromClause
from pixeltable.share.packager import TablePackager, TableRestorer
from pixeltable.utils.local_store import LocalStore, TempStore
from tests.conftest import clean_db

from ..utils import (
    SAMPLE_IMAGE_URL,
    assert_resultset_eq,
    create_table_data,
    get_audio_files,
    get_image_files,
    get_video_files,
    reload_catalog,
    skip_test_if_not_installed,
)


# Bug(PXT-943): non-latest row versions have non-NULL index column values
@pytest.mark.corrupts_db
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
        t = pxt.create_table('media_tbl', {'image': pxt.Image, 'audio': pxt.Audio, 'video': pxt.Video})
        images = get_image_files()[:10]
        audio = get_audio_files()[:5]
        videos = get_video_files()[:2]
        t.insert({'video': video} for video in videos)
        t.insert({'audio': audio} for audio in audio)
        t.insert({'image': image} for image in images)
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

        self.__check_parquet_tbl(t, dest, media_dir=(dest / 'media'), expected_cols=17)

    def __extract_bundle(self, bundle_path: Path) -> Path:
        tmp_dir = TempStore.create_path()
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(tmp_dir)
        return tmp_dir

    def __validate_metadata(self, md: dict, tbl: pxt.Table) -> None:
        assert md['pxt_version'] == pxt.__version__
        assert md['pxt_md_version'] == metadata.VERSION
        assert len(md['md']) == len(tbl._get_base_tables()) + 1
        for t_md, t in zip(md['md'], (tbl, *tbl._get_base_tables())):
            assert schema.md_from_dict(TableVersionMd, t_md).version_md.tbl_id == str(t._tbl_version.id)

    def __check_parquet_tbl(
        self,
        t: pxt.Table,
        bundle_path: Path,
        media_dir: Path | None = None,
        scope_tbl: pxt.Table | None = None,  # If specified, use instead of `tbl` to select rows
        expected_cols: int | None = None,
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
        metadata: pxt.TableMetadata  # User-facing metadata of the table
        store_col_schema: set[tuple[str, str]]  # Set of (column_name, data_type) for the store table's columns
        store_idx_schema: set[tuple[str, str]]  # Set of (indexname, indexdef) for the store table's indices
        result_set: pxt.ResultSet  # Resultset corresponding to the query `tbl.head(n=5000)`

    def __package_table(self, tbl: pxt.Table) -> BundleInfo:
        """
        Runs the query `tbl.head(n=5000)`, packages the table into a bundle, and returns a BundleInfo.
        """
        schema = tbl._get_schema()
        metadata = tbl.get_metadata()
        depth = tbl._tbl_version_path.path_len()
        result_set = tbl.head(n=5000)
        store_col_schema = self.__extract_store_col_schema(tbl)
        store_idx_schema = self.__extract_store_idx_schema(tbl)

        # Package the snapshot into a tarball
        packager = TablePackager(tbl)
        bundle_path = packager.package()

        return TestPackager.BundleInfo(
            bundle_path, depth, schema, metadata, store_col_schema, store_idx_schema, result_set
        )

    def __restore_and_check_table(self, bundle_info: BundleInfo, tbl_name: str, version: int | None = None) -> None:
        """
        Restores the table that was packaged in `bundle_info` and validates its contents against the tracked data.
        """
        restorer = TableRestorer(tbl_name)
        restorer.restore(bundle_info.bundle_path)
        versioned_name = tbl_name if version is None else f'{tbl_name}:{version}'
        self.__check_table(bundle_info, versioned_name)

    def __check_table(self, bundle_info: BundleInfo, tbl_name: str) -> None:
        t = pxt.get_table(tbl_name)

        # Ensure repr() works.
        _ = repr(t)

        assert bundle_info.schema == t._get_schema()
        assert bundle_info.depth == t._tbl_version_path.path_len()

        # Certain metadata properties must be identical.
        metadata = t.get_metadata()
        for property in ('indices', 'version', 'version_created', 'schema_version', 'comment', 'media_validation'):
            assert metadata[property] == bundle_info.metadata[property]

        # Verify that the postgres schema subsumes the original.
        # (There may be additional columns in the restored table depending on the order in which different versions are
        # restored. But the columns present in the original table must be present and identical in the restored table.
        # Regarding indices: the restored table omits indices that were dropped on the Table before the bundle was
        # created.)
        assert bundle_info.store_col_schema.issubset(self.__extract_store_col_schema(t))
        t._tbl_version_path.tbl_version.get().store_tbl.validate()

        reconstituted_data = t.head(n=5000)
        assert_resultset_eq(bundle_info.result_set, reconstituted_data)

    def __extract_store_col_schema(self, tbl: pxt.Table) -> set[tuple[str, str]]:
        with Env.get().begin_xact():
            store_tbl_name = tbl._tbl_version_path.tbl_version.get().store_tbl._storage_name()
            sql_text = (
                f'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = {store_tbl_name!r}'
            )
            result = Env.get().conn.execute(sql.text(sql_text)).fetchall()
            return {(col_name, data_type) for col_name, data_type in result}

    def __extract_store_idx_schema(self, tbl: pxt.Table) -> set[tuple[str, str]]:
        with Env.get().begin_xact():
            store_tbl_name = tbl._tbl_version_path.tbl_version.get().store_tbl._storage_name()
            sql_text = f'SELECT indexname, indexdef FROM pg_indexes WHERE tablename = {store_tbl_name!r}'
            result = Env.get().conn.execute(sql.text(sql_text)).fetchall()
            return {(indexname, indexdef) for indexname, indexdef in result}

    def __purge_db(self) -> None:
        clean_db()
        # Delete any locally stored media files (so that if any stale references to them inadvertently remain after
        # packaging, then those stale references will be invalid).
        # We need to skip this step on Windows; it's flaky due to the way Windows handles file locks.
        # (But testing without media purge on Windows, and with it on other systems, should provide suitable coverage.)
        if platform.system() != 'Windows':
            LocalStore(Env.get().media_dir).clear()
        reload_catalog()

    def __do_round_trip(self, tbl: pxt.Table) -> None:
        bundle = self.__package_table(tbl)
        self.__purge_db()
        self.__restore_and_check_table(bundle, 'new_replica')

    def __validate_index_data(
        self, tbl: pxt.Table, expected_vals: int | None = None, expected_undos: int | None = None
    ) -> None:
        """
        Sanity checks that the val and undo columns are properly configured in the given Table's indices.
        It is important to do this check at a lower level, because improperly categorized val/undo columns may have
        performance implications that are not user visible.
        """
        tv = tbl._tbl_version_path.tbl_version.get()
        with Env.get().begin_xact():
            head_version = Catalog.get()._collect_tbl_history(tbl._id, n=1)[0].version_md.version
            for idx_info in tv.idxs_by_name.values():
                if isinstance(idx_info.idx, EmbeddingIndex):
                    q = sql.select(
                        tv.store_tbl.v_min_col,
                        tv.store_tbl.v_max_col,
                        idx_info.val_col.sa_col,
                        idx_info.undo_col.sa_col,
                    ).order_by(*tv.store_tbl._pk_cols)
                    val_count = 0
                    undo_count = 0
                    for result in Env.get().conn.execute(q).fetchall():
                        v_min, v_max, val, undo = result
                        if v_min <= head_version and v_max > head_version:
                            assert val is None or isinstance(val, (np.ndarray, pgvector.sqlalchemy.HalfVector))
                            assert undo is None
                        else:
                            assert val is None
                            assert undo is None or isinstance(undo, (np.ndarray, pgvector.sqlalchemy.HalfVector))
                        if val is not None:
                            val_count += 1
                        if undo is not None:
                            undo_count += 1
                    if expected_vals is not None:
                        assert val_count == expected_vals
                    if expected_undos is not None:
                        assert undo_count == expected_undos

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

        self.__purge_db()

        self.__restore_and_check_table(bundle1, 'replica')
        self.__restore_and_check_table(bundle2, 'replica')

    def test_media_round_trip(self, img_tbl: pxt.Table) -> None:
        self.__do_round_trip(img_tbl)

    def test_array_round_trip(self, reset_db: None) -> None:
        t = pxt.create_table('tbl', {'arr1': pxt.Array[pxt.Int, (200, 200)], 'arr2': pxt.Array[pxt.Bool]})  # type: ignore[misc]
        t.insert(
            {'arr1': np.ones((200, 200), dtype=np.int64) * i, 'arr2': np.array([j % 19 == 0 for j in range(10000 + i)])}
            for i in range(5)
        )
        self.__do_round_trip(t)

    def test_json_round_trip(self, reset_db: None) -> None:
        images = get_image_files()
        t = pxt.create_table('tbl', {'jcol': pxt.Json})
        t.insert(
            [
                {'jcol': {'this': 'is', 'a': 'test', 'img1': images[0], 'img2': images[22]}},
                {'jcol': {'this': 'is', 'a': 'test', 'img': images[34], 'arr': np.ones((200, 200), dtype=np.int64)}},
            ]
        )
        self.__do_round_trip(t)

    def test_views_round_trip(self, test_tbl: pxt.Table) -> None:
        v1 = pxt.create_view('v1', test_tbl, additional_columns={'x1': pxt.Int})
        v1.update({'x1': test_tbl.c2 * 10})
        v2 = pxt.create_view('v2', v1.where(v1.c2 % 3 == 0), additional_columns={'x2': pxt.Int})
        v2.update({'x2': v1.x1 + 8})
        snapshot = pxt.create_snapshot('snapshot', v2)
        self.__do_round_trip(snapshot)

    def test_restricted_view_round_trip(self, reset_db: None) -> None:
        """Tests a view that only selects a subset of the columns from its base table."""
        t = pxt.create_table('base_tbl', {'icol': pxt.Int, 'scol': pxt.String})
        t.insert({'icol': i, 'scol': f'string {i}'} for i in range(100))
        v = pxt.create_view('view', t.select(t.icol))

        self.__do_round_trip(v)

    def test_iterator_view_round_trip(self, reset_db: None) -> None:
        t = pxt.create_table('base_tbl', {'video': pxt.Video})
        t.insert({'video': video} for video in get_video_files()[:2])

        v = pxt.create_view('frames_view', t, iterator=pxt.functions.video.frame_iterator(t.video, fps=1))
        # Add a stored computed column that will generate a bunch of media files in the view.
        v.add_computed_column(rot_frame=v.frame.rotate(180))
        snapshot = pxt.create_snapshot('snapshot', v)
        snapshot_row_count = snapshot.count()

        self.__do_round_trip(snapshot)

        # Double-check that the snapshot and its base table have the correct number of rows
        snapshot_replica = pxt.get_table('new_replica')
        assert snapshot_replica._snapshot_only
        assert snapshot_replica.count() == snapshot_row_count
        # We can't query the base table directly via snapshot_replica.get_base_table(), because it doesn't exist as a
        # visible catalog object (it's hidden in _system). But we can manually construct the Query and check that.
        t_replica_query = pxt.Query(FromClause(tbls=[snapshot_replica._tbl_version_path.base]))
        assert t_replica_query.count() == 2

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

        self.__purge_db()

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

        self.__purge_db()

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

        self.__purge_db()

        self.__restore_and_check_table(bundle1, 'replica1')
        self.__restore_and_check_table(bundle2, 'replica2')

    @pytest.mark.parametrize('different_versions', [False, True])
    def test_multi_view_round_trip_4(self, different_versions: bool, all_datatypes_tbl: pxt.Table) -> None:
        """
        Snapshots that involve all the different column types. Two snapshots of the same base table will be created;
        they will snapshot either the same or different versions of the table, depending on `different_versions`.
        """
        t = all_datatypes_tbl
        snap1 = pxt.create_snapshot('snap1', t.where(t.row_id % 2 != 0))
        bundle1 = self.__package_table(snap1)

        if different_versions:
            more_data = create_table_data(t, num_rows=22)
            t.insert(more_data[11:])

        snap2 = pxt.create_snapshot('snap2', t.where(t.row_id % 3 != 0))
        bundle2 = self.__package_table(snap2)

        self.__purge_db()

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

        self.__purge_db()

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
            snap = pxt.create_snapshot(f'snap_{n}', t)
            bundles.append(self.__package_table(snap))

        self.__purge_db()

        for n in (7, 3, 0, 9, 4, 10, 1, 5, 8):
            # Snapshots 2 and 6 are intentionally never restored.
            self.__restore_and_check_table(bundles[n], f'replica_{n}')

        # Check all the tables again to verify that everything is consistent.
        for n in (0, 1, 3, 4, 5, 7, 8, 9, 10):
            self.__check_table(bundles[n], f'replica_{n}')

    def test_interleaved_non_snapshots(self, reset_db: None) -> None:
        """
        Test the case where two versions of a non-snapshot table are packaged out of order.
        """
        t = pxt.create_table('tbl', {'int_col': pxt.Int})
        t.insert({'int_col': i} for i in range(512))

        t_bundle = self.__package_table(t)

        v = pxt.create_view('view', t.where(t.int_col % 3 == 0))
        t.insert({'int_col': i} for i in range(512, 1024))

        v_bundle = self.__package_table(v)

        # v_bundle is missing some of the rows that were present in t_bundle, but has some new ones as well.

        self.__purge_db()

        self.__restore_and_check_table(v_bundle, 'view_replica')
        self.__restore_and_check_table(t_bundle, 'tbl_replica')

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

        self.__purge_db()

        for n in (4, 1, 0, 3, 8, 10, 7, 9, 6):
            # The non-snapshot bundles all refer to the same table UUID, so we use the consistent name 'replica' for
            # all of them. The snapshot bundles refer to distinct schema objects, so we use distinct names for them.
            # The non-snapshot bundles are traversed in temporal order; the snapshot bundles are randomized.
            name = 'replica' if n % 2 != 0 else f'replica_{n}'
            self.__restore_and_check_table(bundles[n], name)

    def test_replica_ops(self, reset_db: None, clip_embed: pxt.Function) -> None:
        t = pxt.create_table('test_tbl', {'icol': pxt.Int, 'scol': pxt.String})
        t.insert({'icol': i, 'scol': f'string {i}'} for i in range(10))
        v = pxt.create_view('test_view', t)
        v.add_computed_column(iccol=(v.icol + 1))

        t_bundle = self.__package_table(t)
        v_bundle = self.__package_table(v)

        self.__purge_db()

        self.__restore_and_check_table(v_bundle, 'view_replica')
        # Check that test_tbl was instantiated as a system table
        assert pxt.list_tables() == ['view_replica']
        system_contents = pxt.globals._list_tables('_system', allow_system_paths=True)
        assert len(system_contents) == 1 and system_contents[0].startswith('_system.replica_')

        self.__restore_and_check_table(t_bundle, 'tbl_replica')
        # Check that test_tbl has been renamed to a user table
        assert pxt.list_tables() == ['view_replica', 'tbl_replica']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0

        t = pxt.get_table('tbl_replica')
        v = pxt.get_table('view_replica')

        for s, name in ((t, 'tbl_replica'), (v, 'view_replica')):
            display_str = f'replica {name!r}'
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot insert into a replica.'):
                s.insert({'icol': 10, 'scol': 'string 10'})
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot delete from a replica.'):
                s.delete()
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a replica.'):
                s.add_column(new_col=pxt.Bool)
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a replica.'):
                s.add_columns({'new_col': pxt.Bool})
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add columns to a replica.'):
                s.add_computed_column(new_col=(t.icol + 1))
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot drop columns from a replica.'):
                s.drop_column('scol')
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot add an index to a replica.'):
                s.add_embedding_index('icol', embedding=clip_embed)
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot drop an index from a replica.'):
                s.drop_embedding_index(column='icol')
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot update a replica.'):
                s.update({'icol': 11})
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot recompute columns of a replica.'):
                s.recompute_columns('icol')
            with pytest.raises(pxt.Error, match=f'{display_str}: Cannot revert a replica.'):
                s.revert()

            # TODO: Align these Query error messages with Table error messages
            with pytest.raises(pxt.Error, match=r'Cannot use `update` on a replica.'):
                s.where(s.icol < 5).update({'icol': 100})
            with pytest.raises(pxt.Error, match=r'Cannot use `delete` on a replica.'):
                s.where(s.icol < 5).delete()

            with pytest.raises(pxt.Error, match='Cannot create a view or snapshot on top of a replica'):
                _ = pxt.create_view(f'subview_of_{name}', s)

    def test_drop_replica(self, reset_db: None) -> None:
        """
        Test dropping a replica table.
        """
        t = pxt.create_table('base_tbl', {'c1': pxt.Int})
        t.insert([{'c1': i} for i in range(100)])
        t_bundle = self.__package_table(t)

        v = pxt.create_view('view', t)
        v_bundle = self.__package_table(v)

        self.__purge_db()

        self.__restore_and_check_table(t_bundle, 'replica_tbl')
        assert pxt.list_tables() == ['replica_tbl']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0
        pxt.drop_table('replica_tbl')
        assert pxt.list_tables() == []
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0

        self.__purge_db()

        # Now try with both a table and a view

        # Restoring the view should materialize the base table as a hidden table
        self.__restore_and_check_table(v_bundle, 'replica_view')
        assert pxt.list_tables() == ['replica_view']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 1

        # Restoring the table should rename it to a visible table
        self.__restore_and_check_table(t_bundle, 'replica_tbl')
        assert sorted(pxt.list_tables()) == ['replica_tbl', 'replica_view']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0
        self.__check_table(v_bundle, 'replica_view')  # Check the view again

        # Now drop the table; this should revert it to a hidden table
        pxt.drop_table('replica_tbl')
        assert pxt.list_tables() == ['replica_view']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 1

        # Now drop the view; this should delete it
        pxt.drop_table('replica_view')
        assert pxt.list_tables() == []
        # this should also have resulted in the base table being deleted
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0

        # Try again, this time deleting the view first.
        # We do all the checks again with a full restore cycle, so that we also check that multiple
        # create/delete roundtrips on the same replicas succeed without leaving cruft around.
        self.__restore_and_check_table(v_bundle, 'replica_view')
        assert pxt.list_tables() == ['replica_view']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 1
        self.__restore_and_check_table(t_bundle, 'replica_tbl')
        assert sorted(pxt.list_tables()) == ['replica_tbl', 'replica_view']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0
        self.__check_table(v_bundle, 'replica_view')  # Check the view again

        # Now drop the view; this should delete it
        pxt.drop_table('replica_view')
        # this should NOT have resulted in the base table being deleted, since it's now a visible table
        assert pxt.list_tables() == ['replica_tbl']
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0

        # Now drop the table; this should delete it
        pxt.drop_table('replica_tbl')
        assert pxt.list_tables() == []
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 0

    def test_deep_view_hierarchy(self, reset_db: None) -> None:
        """
        Test dropping various replica tables.
        """
        t = pxt.create_table('base_tbl', {'c1': pxt.Int})
        tbls: list[pxt.Table] = [t]
        bundles: list[TestPackager.BundleInfo] = [self.__package_table(t)]
        for i in range(10):
            t.insert([{'c1': i} for i in range(i * 10, (i + 1) * 10)])
            v = pxt.create_view(f'view_{i}', tbls[i])
            tbls.append(v)
            bundles.append(self.__package_table(v))

        assert len(tbls) == 11
        assert len(bundles) == 11

        self.__purge_db()

        # Restore a few intermediate views
        for i in (7, 5, 2, 10):
            self.__restore_and_check_table(bundles[i], f'replica_{i}')

        assert pxt.list_tables() == [f'replica_{i}' for i in (2, 5, 7, 10)]  # 4 visible tables
        _x = pxt.globals._list_tables('_system', allow_system_paths=True)
        assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == 7  # 7 hidden tables

        # Now drop the visible tables one by one.
        tables = [7, 5, 2, 10]
        for i in (5, 10, 2, 7):
            print(f'Dropping replica_{i}')
            pxt.drop_table(f'replica_{i}')
            tables.remove(i)
            assert sorted(pxt.list_tables()) == sorted(f'replica_{j}' for j in tables)
            # The total number of tables remaining should be equal to 1 + max(tables); any tables beyond this no longer
            # have dependents and should have been purged. Of these, len(tables) are visible, so the rest are hidden.
            expected_hidden_tables = 0 if len(tables) == 0 else 1 + max(tables) - len(tables)
            _ = pxt.globals._list_tables('_system', allow_system_paths=True)
            assert len(pxt.globals._list_tables('_system', allow_system_paths=True)) == expected_hidden_tables
            for j in tables:
                # Re-check all tables that are still present
                self.__check_table(bundles[j], f'replica_{j}')

    def test_older_versions_round_trip(self, reset_db: None) -> None:
        t = pxt.create_table('tbl', {'int_col': pxt.Int})
        for i in range(50):
            t.insert([{'int_col': i}])
        assert len(t.get_versions()) == 51

        versions = (36, 11, 23, 42, 5, 46)
        snapshots = tuple(pxt.get_table(f'tbl:{i}') for i in versions)
        bundles = tuple(self.__package_table(snap) for snap in snapshots)

        clean_db()
        reload_catalog()

        for i, bundle in zip(versions, bundles, strict=True):
            self.__restore_and_check_table(bundle, 'replica', version=i)

    def test_view_over_snapshot_round_trip(self, reset_db: None) -> None:
        pxt.create_dir('dir')
        t = pxt.create_table('dir.test_tbl', {'c1': pxt.Int})

        views: list[pxt.Table] = []
        bundles: list[TestPackager.BundleInfo] = []

        # Create 5 snapshots with views on top of them, modifying the base table in between.
        for i in range(5):
            t.insert(c1=i)
            t.add_computed_column(**{f'x{i}': t.c1 + i * 10})
            snap = pxt.create_snapshot(f'dir.test_snap_{i}', t)
            view = pxt.create_view(f'dir.test_view_{i}', snap)
            views.append(view)

        # Now modify each of the views.
        for i in range(5):
            views[i].add_computed_column(**{f'y{i}': views[i].c1 + i * 100})

        # Package the views.
        for i in range(5):
            bundles.append(self.__package_table(views[i]))

        self.__purge_db()

        # Now restore each of the views, ensuring that each view properly publishes and restores according
        # to its underlying snapshot state.
        for i in (3, 0, 1, 4, 2):
            self.__restore_and_check_table(bundles[i], f'replica_view_{i}')

    @pytest.mark.parametrize('embedding_precision', ['fp16', 'fp32'])
    def test_embedding_index(
        self, reset_db: None, clip_embed: pxt.Function, embedding_precision: Literal['fp16', 'fp32']
    ) -> None:
        skip_test_if_not_installed('transformers')  # needed for CLIP

        t = pxt.create_table('tbl', {'image': pxt.Image})
        images = get_image_files()[:10]
        t.insert({'image': image} for image in images)
        t.add_embedding_index('image', embedding=clip_embed, precision=embedding_precision)

        self.__do_round_trip(t)

    @pytest.mark.parametrize('embedding_precision', ['fp16', 'fp32'])
    def test_multi_version_embedding_index(
        self, reset_db: None, clip_embed: pxt.Function, embedding_precision: Literal['fp16', 'fp32']
    ) -> None:
        skip_test_if_not_installed('transformers')  # needed for CLIP

        t = pxt.create_table('tbl', {'id': pxt.Int, 'image': pxt.Image})
        images = get_image_files()
        t.insert({'id': i, 'image': image} for i, image in enumerate(images[:10]))
        t.add_embedding_index('image', embedding=clip_embed, precision=embedding_precision)
        bundle1 = self.__package_table(t)
        sim_1 = t.image.similarity(string=images[25])
        sim_results_1 = t.select(t.id, sim_1).order_by(sim_1, asc=False).limit(5).collect()

        t.delete(t.id < 5)
        t.insert({'id': i, 'image': image} for i, image in enumerate(images[10:20], start=10))
        bundle2 = self.__package_table(t)
        sim_2 = t.image.similarity(string=images[25])
        sim_results_2 = t.select(t.id, sim_2).order_by(sim_2, asc=False).limit(5).collect()

        self.__purge_db()

        self.__restore_and_check_table(bundle1, 'replica')
        t = pxt.get_table('replica')
        sim_1_replica = t.image.similarity(string=images[25])
        sim_results_1_replica = t.select(t.id, sim_1_replica).order_by(sim_1_replica, asc=False).limit(5).collect()
        assert_resultset_eq(sim_results_1, sim_results_1_replica)

        self.__restore_and_check_table(bundle2, 'replica')
        t = pxt.get_table('replica')
        sim_2_replica = t.image.similarity(string=images[25])
        sim_results_2_replica = t.select(t.id, sim_2_replica).order_by(sim_2_replica, asc=False).limit(5).collect()
        assert_resultset_eq(sim_results_2, sim_results_2_replica)

        self.__validate_index_data(t, 15, 5)

    def test_replicating_view_with_existing_base_tbl(self, reset_db: None) -> None:
        """
        Test restoring a view when its base table already exists in the catalog as a non-replica table.
        """
        t = pxt.create_table('base_tbl', {'c1': pxt.Int})
        t.insert([{'c1': i} for i in range(100)])
        v = pxt.create_view('view', t.where(t.c1 % 2 == 0))

        v_bundle = self.__package_table(v)
        # Drop just `v` without purging the DB
        pxt.drop_table(v)

        with pytest.raises(
            pxt.Error,
            match=(
                r'(?s)An attempt was made to replicate a view whose base table already exists'
                r".*pxt.drop_table\('base_tbl'\)"
            ),
        ):
            self.__restore_and_check_table(v_bundle, 'replica_view')
