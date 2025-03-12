import filecmp
import io
import json
import tarfile
import urllib.parse
from pathlib import Path
from typing import Optional

import numpy as np
from pyiceberg.table import Table as IcebergTable

import pixeltable as pxt
from pixeltable import exprs, metadata
from pixeltable.env import Env
from pixeltable.share.packager import TablePackager
from pixeltable.utils.iceberg import sqlite_catalog

from ..utils import SAMPLE_IMAGE_URL, get_image_files, get_video_files


class TestPackager:
    def test_packager(self, test_tbl: pxt.Table):
        packager = TablePackager(test_tbl)
        bundle_path = packager.package()

        # Reinstantiate a catalog to test reads from scratch
        dest = self.__extract_bundle(bundle_path)
        metadata = json.loads((dest / 'metadata.json').read_text())
        self.__validate_metadata(metadata, test_tbl)
        catalog = sqlite_catalog(dest / 'warehouse')
        assert catalog.list_tables('pxt') == [('pxt', 'test_tbl')]
        iceberg_tbl = catalog.load_table('pxt.test_tbl')
        self.__check_iceberg_tbl(test_tbl, iceberg_tbl)

    def test_packager_with_views(self, test_tbl: pxt.Table):
        pxt.create_dir('iceberg_dir')
        pxt.create_dir('iceberg_dir.subdir')
        view = pxt.create_view('iceberg_dir.subdir.test_view', test_tbl)
        view.add_computed_column(vc2=(view.c2 + 1))
        subview = pxt.create_view('iceberg_dir.subdir.test_subview', view.where(view.c2 % 5 == 0))
        subview.add_computed_column(vvc2=(subview.vc2 + 1))
        packager = TablePackager(subview)
        bundle_path = packager.package()

        dest = self.__extract_bundle(bundle_path)
        metadata = json.loads((dest / 'metadata.json').read_text())
        self.__validate_metadata(metadata, subview)
        catalog = sqlite_catalog(dest / 'warehouse')
        assert catalog.list_tables('pxt') == [('pxt', 'test_tbl')]
        assert set(catalog.list_tables('pxt.iceberg_dir.subdir')) == {
            ('pxt', 'iceberg_dir', 'subdir', 'test_view'),
            ('pxt', 'iceberg_dir', 'subdir', 'test_subview'),
        }
        self.__check_iceberg_tbl(test_tbl, catalog.load_table('pxt.test_tbl'), scope_tbl=subview)
        self.__check_iceberg_tbl(view, catalog.load_table('pxt.iceberg_dir.subdir.test_view'), scope_tbl=subview)
        self.__check_iceberg_tbl(subview, catalog.load_table('pxt.iceberg_dir.subdir.test_subview'))

    def test_media_packager(self, reset_db):
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

        catalog = sqlite_catalog(dest / 'warehouse')

        expected_cols = 2 + 3 * 3  # rowid, v_min, plus three stored media/computed columns with error columns
        self.__check_iceberg_tbl(
            t, catalog.load_table('pxt.media_tbl'), media_dir=(dest / 'media'), expected_cols=expected_cols
        )

    def __extract_bundle(self, bundle_path: Path) -> Path:
        tmp_dir = Path(Env.get().create_tmp_path())
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(tmp_dir)
        return tmp_dir

    def __validate_metadata(self, md: dict, tbl: pxt.Table) -> None:
        assert md['pxt_version'] == pxt.__version__
        assert md['pxt_md_version'] == metadata.VERSION
        assert len(md['md']['tables']) == len(tbl._bases) + 1
        for t_md, t in zip(md['md']['tables'], (tbl, *tbl._bases)):
            assert t_md['table_id'] == str(t._tbl_version.id)

    def __check_iceberg_tbl(
        self,
        t: pxt.Table,
        iceberg_tbl: IcebergTable,
        media_dir: Optional[Path] = None,
        scope_tbl: Optional[pxt.Table] = None,  # If specified, use instead of `tbl` to select rows
        expected_cols: Optional[int] = None,
    ) -> None:
        iceberg_data = iceberg_tbl.scan().to_pandas()

        if expected_cols is not None:
            assert len(iceberg_data.columns) == expected_cols

        # Only check columns defined in the table (not ancestors)
        select_exprs: dict[str, exprs.Expr] = {}
        actual_col_types: list[pxt.ColumnType] = []
        for col_name, col in t._tbl_version.cols_by_name.items():
            if not col.is_stored:
                continue
            if col.col_type.is_media_type():
                select_exprs[col_name] = t[col_name].fileurl
            else:
                select_exprs[col_name] = t[col_name]
            actual_col_types.append(col.col_type)
            if col.records_errors:
                select_exprs[f'{col_name}_errortype'] = t[col_name].errortype
                actual_col_types.append(pxt.StringType())
                select_exprs[f'{col_name}_errormsg'] = t[col_name].errormsg
                actual_col_types.append(pxt.StringType())

        scope_tbl = scope_tbl or t
        pxt_data = scope_tbl.select(**select_exprs).collect()
        for col, col_type in zip(select_exprs.keys(), actual_col_types):
            print(f'Checking column: {col}')
            pxt_values: list = pxt_data[col]
            iceberg_values = list(iceberg_data[col])
            if col_type.is_array_type():
                iceberg_values = [np.load(io.BytesIO(val)) for val in iceberg_values]
                for pxt_val, iceberg_val in zip(pxt_values, iceberg_values):
                    assert np.array_equal(pxt_val, iceberg_val)
            elif col_type.is_json_type():
                # JSON columns were exported as strings; check that they parse properly
                assert pxt_values == [json.loads(val) for val in iceberg_values]
            elif col_type.is_media_type():
                assert media_dir is not None
                self.__check_media(pxt_values, iceberg_values, media_dir)
            else:
                assert pxt_values == iceberg_values

    def __check_media(self, pxt_values: list, iceberg_values: list, media_dir: Path) -> None:
        for pxt_val, iceberg_val in zip(pxt_values, iceberg_values):
            if pxt_val is None:
                assert iceberg_val is None
                continue
            assert isinstance(pxt_val, str)
            assert isinstance(iceberg_val, str)
            parsed_url = urllib.parse.urlparse(pxt_val)
            if parsed_url.scheme == 'file':
                assert iceberg_val.startswith('pxtmedia://')
                path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_url.path)))
                bundled_path = media_dir / iceberg_val.removeprefix('pxtmedia://')
                assert bundled_path.exists(), bundled_path
                assert filecmp.cmp(path, bundled_path)
            else:
                assert pxt_val == iceberg_val
