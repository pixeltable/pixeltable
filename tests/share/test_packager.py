import filecmp
import io
import json
import tarfile
import urllib.parse
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import exprs, metadata
from pixeltable.env import Env
from pixeltable.share.packager import TablePackager, TableRestorer
from tests.conftest import clean_db

from ..utils import SAMPLE_IMAGE_URL, assert_resultset_eq, get_image_files, get_video_files, reload_catalog


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

        expected_cols = 1 + 3 * 3  # pk plus three stored media/computed columns with error columns
        self.__check_parquet_tbl(t, dest, media_dir=(dest / 'media'), expected_cols=expected_cols)

    def __extract_bundle(self, bundle_path: Path) -> Path:
        tmp_dir = Path(Env.get().create_tmp_path())
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(tmp_dir)
        return tmp_dir

    def __validate_metadata(self, md: dict, tbl: pxt.Table) -> None:
        assert md['pxt_version'] == pxt.__version__
        assert md['pxt_md_version'] == metadata.VERSION
        assert len(md['md']['tables']) == len(tbl._base_tables) + 1
        for t_md, t in zip(md['md']['tables'], (tbl, *tbl._base_tables)):
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
                select_exprs[f'val_{col_name}'] = t[col_name].fileurl
            else:
                select_exprs[f'val_{col_name}'] = t[col_name]
            actual_col_types.append(col.col_type)
            if col.records_errors:
                select_exprs[f'errortype_{col_name}'] = t[col_name].errortype
                actual_col_types.append(ts.StringType())
                select_exprs[f'errormsg_{col_name}'] = t[col_name].errormsg
                actual_col_types.append(ts.StringType())

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
                assert pxt_values == [json.loads(val) for val in parquet_values]
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

    def __do_round_trip(self, snapshot: pxt.Table) -> None:
        assert snapshot._tbl_version.get().is_snapshot

        schema = snapshot._schema
        depth = len(snapshot._tbl_version_path.ancestors)
        data = snapshot.head(n=500)

        # Package the snapshot into a tarball
        packager = TablePackager(snapshot)
        bundle_path = packager.package()

        # Clear out the db
        clean_db()
        reload_catalog()

        # Restore the snapshot from the tarball
        restorer = TableRestorer('new_replica')
        restorer.restore(bundle_path)
        t = pxt.get_table('new_replica')
        assert t._schema == schema
        assert len(snapshot._tbl_version_path.ancestors) == depth
        reconstituted_data = t.head(n=500)

        assert_resultset_eq(data, reconstituted_data)

    def test_round_trip(self, test_tbl: pxt.Table) -> None:
        """package() / unpackage() round trip"""
        snapshot = pxt.create_snapshot('snapshot', test_tbl)
        self.__do_round_trip(snapshot)

    def test_media_round_trip(self, img_tbl: pxt.Table) -> None:
        snapshot = pxt.create_snapshot('snapshot', img_tbl)
        self.__do_round_trip(snapshot)

    def test_views_round_trip(self, test_tbl: pxt.Table) -> None:
        v1 = pxt.create_view('v1', test_tbl, additional_columns={'x1': pxt.Int})
        v1.update({'x1': test_tbl.c2 * 10})
        v2 = pxt.create_view('v2', v1, additional_columns={'x2': pxt.Int})
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
        v_replica = snapshot_replica.base_table
        assert v_replica.count() == snapshot_row_count
        t_replica = v_replica.base_table
        assert t_replica.count() == 2
