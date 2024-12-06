import datetime
import random
import math
import os
import random
from typing import Union, _GenericAlias

import av  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import PIL
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable.io.external_store import MockProject
from pixeltable.iterators import FrameIterator
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

from .utils import (assert_resultset_eq, create_table_data, get_audio_files, get_documents, get_image_files,
                    get_video_files, make_tbl, read_data_file, reload_catalog, skip_test_if_not_installed, strip_lines,
                    validate_update_status, get_multimedia_commons_video_uris, ReloadTester)


class TestTable:
    # exc for a % 10 == 0
    @pxt.udf
    def f1(a: int) -> float:
        return a / (a % 10)

    # exception for a == None; this should not get triggered
    @pxt.udf
    def f2(a: float) -> float:
        return a + 1

    @pxt.expr_udf
    def add1(a: int) -> int:
        return a + 1

    @pxt.uda(update_types=[pxt.IntType()], value_type=pxt.IntType(), requires_order_by=True, allows_window=True)
    class window_fn:
        def __init__(self):
            pass

        def update(self, i: int) -> None:
            pass

        def value(self) -> int:
            return 1

    @pxt.expr_udf
    def add1(a: int) -> int:
        return a + 1

    def test_create(self, reset_db: None) -> None:
        pxt.create_dir('dir1')
        schema = {
            'c1': pxt.String,
            'c2': pxt.Int,
            'c3': pxt.Float,
            'c4': pxt.Timestamp,
        }
        tbl = pxt.create_table('test', schema)
        _ = pxt.create_table('dir1.test', schema)

        with pytest.raises(excs.Error):
            _ = pxt.create_table('1test', schema)
        with pytest.raises(excs.Error):
            _ = pxt.create_table('bad name', {'c1': pxt.String})
        with pytest.raises(excs.Error):
            _ = pxt.create_table('test', schema)
        with pytest.raises(excs.Error):
            _ = pxt.create_table('dir2.test2', schema)

        _ = pxt.list_tables()
        _ = pxt.list_tables('dir1')

        with pytest.raises(excs.Error):
            _ = pxt.list_tables('1dir')
        with pytest.raises(excs.Error):
            _ = pxt.list_tables('dir2')

        # test loading with new client
        reload_catalog()

        tbl = pxt.get_table('test')
        assert isinstance(tbl, catalog.InsertableTable)
        tbl.add_column(c5=pxt.Int)
        tbl.drop_column('c1')
        tbl.rename_column('c2', 'c17')

        pxt.move('test', 'test2')

        pxt.drop_table('test2')
        pxt.drop_table('dir1.test')

        with pytest.raises(excs.Error):
            pxt.drop_table('test')
        with pytest.raises(excs.Error):
            pxt.drop_table('dir1.test2')
        with pytest.raises(excs.Error):
            pxt.drop_table('.test2')

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('bad_col_name', {'pos': pxt.Int})
        assert "'pos' is a reserved name in pixeltable" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('test', {'add_column': pxt.Int})
        assert "'add_column' is a reserved name in pixeltable" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('test', {'insert': pxt.Int})
        assert "'insert' is a reserved name in pixeltable" in str(exc_info.value).lower()

    def test_columns(self, reset_db: None) -> None:  # noqa: PLR6301
        schema = {
            'c1': pxt.String,
            'c2': pxt.Int,
            'c3': pxt.Float,
            'c4': pxt.Timestamp,
        }
        t = pxt.create_table('test', schema)
        assert t.columns == ['c1', 'c2', 'c3', 'c4']

    def test_names(self, reset_db: None) -> None:
        pxt.create_dir('dir')
        pxt.create_dir('dir.subdir')
        for tbl_path, media_val in [('test', 'on_read'), ('dir.test', 'on_write'), ('dir.subdir.test', 'on_read')]:
            tbl = pxt.create_table(tbl_path, {'col': pxt.String}, media_validation=media_val)
            view = pxt.create_view(f'{tbl_path}_view', tbl, media_validation=media_val)
            snap = pxt.create_snapshot(f'{tbl_path}_snap', tbl, media_validation=media_val)
            assert tbl._path == tbl_path
            assert tbl._name == tbl_path.split('.')[-1]
            assert tbl._parent._path == '.'.join(tbl_path.split('.')[:-1])
            for t in (tbl, view, snap):
                assert t.get_metadata() == {
                    'base': None if t._base is None else t._base._path,
                    'comment': t._comment,
                    'is_view': isinstance(t, catalog.View),
                    'is_snapshot': t._tbl_version.is_snapshot,
                    'name': t._name,
                    'num_retained_versions': t._num_retained_versions,
                    'media_validation': media_val,
                    'parent': t._parent._path,
                    'path': t._path,
                    'schema': t._schema,
                    'schema_version': t._tbl_version.schema_version,
                    'version': t._version,
                }

    def test_media_validation(self, reset_db: None) -> None:
        tbl_schema = {
            'img': {'type': pxt.Image, 'media_validation': 'on_write'},
            'video': pxt.Video
        }
        t = pxt.create_table('test', tbl_schema, media_validation='on_read')
        assert t.get_metadata()['media_validation'] == 'on_read'
        assert t.img.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        # table default applies
        assert t.video.col.media_validation == pxt.catalog.MediaValidation.ON_READ

        v_schema = {
            'doc': {'type': pxt.Document, 'media_validation': 'on_read'},
            'audio': pxt.Audio
        }
        v = pxt.create_view('test_view', t, additional_columns=v_schema, media_validation='on_write')
        assert v.get_metadata()['media_validation'] == 'on_write'
        assert v.doc.col.media_validation == pxt.catalog.MediaValidation.ON_READ
        # view default applies
        assert v.audio.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        # flags for base still apply
        assert v.img.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        assert v.video.col.media_validation == pxt.catalog.MediaValidation.ON_READ

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table(
                'validation_error', {'img': pxt.Image}, media_validation='wrong_value')
        assert "media_validation must be one of: ['on_read', 'on_write']" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table(
                'validation_error', {'img': {'type': pxt.Image, 'media_validation': 'wrong_value'}})
        assert "media_validation must be one of: ['on_read', 'on_write']" in str(exc_info.value)

    def test_validate_on_read(self, reset_db: None) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for f in files]
        schema = {'media': pxt.Video, 'is_bad_media': pxt.Bool}

        on_read_tbl = pxt.create_table('read_validated', schema, media_validation='on_read')
        validate_update_status(on_read_tbl.insert(rows), len(rows))
        on_read_res = (
            on_read_tbl.select(
                on_read_tbl.media, on_read_tbl.media.localpath, on_read_tbl.media.errortype, on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media
            ).collect()
        )

        on_write_tbl = pxt.create_table('write_validated', schema, media_validation='on_write')
        status = on_write_tbl.insert(rows, on_error='ignore')
        assert status.num_excs == 2  # 1 row with exceptions in the media col and the index col
        on_write_res = (
            on_write_tbl.select(
                on_write_tbl.media, on_write_tbl.media.localpath, on_write_tbl.media.errortype,
                on_write_tbl.media.errormsg, on_write_tbl.is_bad_media
            ).collect()
        )
        assert_resultset_eq(on_read_res, on_write_res)

        reload_catalog()
        on_read_tbl = pxt.get_table('read_validated')
        on_read_res = (
            on_read_tbl.select(
                on_read_tbl.media, on_read_tbl.media.localpath, on_read_tbl.media.errortype, on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media
            ).collect()
        )
        assert_resultset_eq(on_read_res, on_write_res)

    def test_validate_on_read_with_computed_col(self, reset_db: None) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for f in files]
        schema = {'media': pxt.Video, 'is_bad_media': pxt.Bool, 'stage': pxt.Required[pxt.Int]}

        # we are testing a nonsensical scenario: a computed column that references a read-validated media column,
        # which forces validation
        on_read_tbl = pxt.create_table('read_validated', schema, media_validation='on_read')
        on_read_tbl.add_column(md=on_read_tbl.media.get_metadata())
        status = on_read_tbl.insert(({**r, 'stage': 0} for r in rows), on_error='ignore')
        assert status.num_excs == 1
        on_read_res_1 = (
            on_read_tbl
            .select(
                on_read_tbl.media, on_read_tbl.media.localpath, on_read_tbl.media.errortype, on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media, on_read_tbl.md
            )
            .order_by(on_read_tbl.media)
            .collect()
        )

        reload_catalog()
        on_read_tbl = pxt.get_table('read_validated')
        # we can still insert into the table after a catalog reload, and the result is the same
        status = on_read_tbl.insert(({**r, 'stage': 1} for r in rows), on_error='ignore')
        assert status.num_excs == 1
        on_read_res_2 = (
            on_read_tbl
            .where(on_read_tbl.stage == 1)
            .select(
                on_read_tbl.media, on_read_tbl.media.localpath, on_read_tbl.media.errortype, on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media, on_read_tbl.md
            )
            .order_by(on_read_tbl.media)
            .collect()
        )
        assert_resultset_eq(on_read_res_1, on_read_res_2)

    def test_create_from_df(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        df1 = t.where(t.c2 >= 50).order_by(t.c2, asc=False).select(t.c2, t.c3, t.c7, t.c2 + 26, t.c1.contains('19'))
        t1 = pxt.create_table('test1', df1)
        assert t1._schema == df1.schema
        assert t1.collect() == df1.collect()

        from pixeltable.functions import sum
        t.add_column(c2mod=t.c2 % 5)
        df2 = t.group_by(t.c2mod).select(t.c2mod, sum(t.c2))
        t2 = pxt.create_table('test2', df2)
        assert t2._schema == df2.schema
        assert t2.collect() == df2.collect()

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('test3', ['I am a string.'])
        assert '`schema_or_df` must be either a schema dictionary or a Pixeltable DataFrame' in str(exc_info.value)

    # Test the various combinations of type hints available in schema definitions and validate that they map to the
    # correct ColumnType instances.
    def test_schema_types(self, reset_db: None) -> None:
        test_columns: dict[str, Union[type, _GenericAlias]] = {
            'str_col': pxt.String,
            'req_str_col': pxt.Required[pxt.String],
            'int_col': pxt.Int,
            'req_int_col': pxt.Required[pxt.Int],
            'float_col': pxt.Float,
            'req_float_col': pxt.Required[pxt.Float],
            'bool_col': pxt.Bool,
            'req_bool_col': pxt.Required[pxt.Bool],
            'ts_col': pxt.Timestamp,
            'req_ts_col': pxt.Required[pxt.Timestamp],
            'json_col': pxt.Json,
            'req_json_col': pxt.Required[pxt.Json],
            'array_col': pxt.Array[(5, None, 3), pxt.Int],
            'req_array_col': pxt.Required[pxt.Array[(5, None, 3), pxt.Int]],
            'img_col': pxt.Image,
            'req_img_col': pxt.Required[pxt.Image],
            'spec_img_col': pxt.Image[(300, 300), 'RGB'],
            'req_spec_img_col': pxt.Required[pxt.Image[(300, 300), 'RGB']],
            'video_col': pxt.Video,
            'req_video_col': pxt.Required[pxt.Video],
            'audio_col': pxt.Audio,
            'req_audio_col': pxt.Required[pxt.Audio],
            'doc_col': pxt.Document,
            'req_doc_col': pxt.Required[pxt.Document],
        }

        t = pxt.create_table('test', test_columns)

        # Test all the types with add_column as well
        for col_name, col_type in test_columns.items():
            t.add_column(**{f'added_{col_name}': col_type})

        expected_schema = {
            'str_col': pxt.StringType(nullable=True),
            'req_str_col': pxt.StringType(nullable=False),
            'int_col': pxt.IntType(nullable=True),
            'req_int_col': pxt.IntType(nullable=False),
            'float_col': pxt.FloatType(nullable=True),
            'req_float_col': pxt.FloatType(nullable=False),
            'bool_col': pxt.BoolType(nullable=True),
            'req_bool_col': pxt.BoolType(nullable=False),
            'ts_col': pxt.TimestampType(nullable=True),
            'req_ts_col': pxt.TimestampType(nullable=False),
            'json_col': pxt.JsonType(nullable=True),
            'req_json_col': pxt.JsonType(nullable=False),
            'array_col': pxt.ArrayType((5, None, 3), dtype=pxt.IntType(), nullable=True),
            'req_array_col': pxt.ArrayType((5, None, 3), dtype=pxt.IntType(), nullable=False),
            'img_col': pxt.ImageType(nullable=True),
            'req_img_col': pxt.ImageType(nullable=False),
            'spec_img_col': pxt.ImageType(width=300, height=300, mode='RGB', nullable=True),
            'req_spec_img_col': pxt.ImageType(width=300, height=300, mode='RGB', nullable=False),
            'video_col': pxt.VideoType(nullable=True),
            'req_video_col': pxt.VideoType(nullable=False),
            'audio_col': pxt.AudioType(nullable=True),
            'req_audio_col': pxt.AudioType(nullable=False),
            'doc_col': pxt.DocumentType(nullable=True),
            'req_doc_col': pxt.DocumentType(nullable=False),
        }
        expected_schema.update({
            f'added_{col_name}': col_type for col_name, col_type in expected_schema.items()
        })

        assert t._schema == expected_schema

        expected_strings = [
            'String',
            'Required[String]',
            'Int',
            'Required[Int]',
            'Float',
            'Required[Float]',
            'Bool',
            'Required[Bool]',
            'Timestamp',
            'Required[Timestamp]',
            'Json',
            'Required[Json]',
            'Array[(5, None, 3), Int]',
            'Required[Array[(5, None, 3), Int]]',
            'Image',
            'Required[Image]',
            "Image[(300, 300), 'RGB']",
            "Required[Image[(300, 300), 'RGB']]",
            'Video',
            'Required[Video]',
            'Audio',
            'Required[Audio]',
            'Document',
            'Required[Document]',
        ]
        df = t._col_descriptor()
        assert list(df['Type']) == expected_strings + expected_strings

    def test_empty_table(self, reset_db: None) -> None:
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('empty_table', {})
        assert 'Table schema is empty' in str(exc_info.value)

    def test_drop_table(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        pxt.drop_table('test_tbl')
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.get_table('test_tbl')
        assert 'no such path: test_tbl' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.show(1)
        assert 'table test_tbl has been dropped' in str(exc_info.value).lower()

    def test_drop_table_via_handle(self, test_tbl: pxt.Table) -> None:
        t = pxt.create_table('test1', {'c1': pxt.String})
        pxt.drop_table(t)
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.get_table('test1')
        assert 'no such path: test1' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.show(1)
        assert 'table test1 has been dropped' in str(exc_info.value).lower()
        t = pxt.create_table('test2', {'c1': pxt.String})
        t = pxt.get_table('test2')
        pxt.drop_table(t)
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.get_table('test2')
        assert 'no such path: test2' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.show(1)
        assert 'table test2 has been dropped' in str(exc_info.value).lower()
        t = pxt.create_table('test3', {'c1': pxt.String})
        v = pxt.create_view('view3', t)
        pxt.drop_table(v)
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.get_table('view3')
        assert 'no such path: view3' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = v.show(1)
        assert 'view view3 has been dropped' in str(exc_info.value).lower()
        _ = pxt.get_table('test3')
        v = pxt.create_view('view4', t)
        v = pxt.get_table('view4')
        pxt.drop_table(v)
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.get_table('view4')
        assert 'no such path: view4' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = v.show(1)
        assert 'view view4 has been dropped' in str(exc_info.value).lower()
        _ = pxt.get_table('test3')
        pxt.drop_table(t)

    def test_drop_table_force(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', t)
        v3 = pxt.create_view('v3', v1)
        v4 = pxt.create_view('v4', v2)
        v5 = pxt.create_view('v5', t)
        assert len(pxt.list_tables()) == 6
        pxt.drop_table('v2', force=True)  # Drops v2 and v4, but not the others
        assert len(pxt.list_tables()) == 4
        pxt.drop_table('test_tbl', force=True)  # Drops everything else
        assert len(pxt.list_tables()) == 0

    def test_drop_table_force_via_handle(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', t)
        v3 = pxt.create_view('v3', v1)
        v4 = pxt.create_view('v4', v2)
        v5 = pxt.create_view('v5', t)
        assert len(pxt.list_tables()) == 6
        pxt.drop_table(v2, force=True)  # Drops v2 and v4, but not the others
        assert len(pxt.list_tables()) == 4
        assert 'v2' not in pxt.list_tables()
        assert 'v4' not in pxt.list_tables()
        pxt.drop_table(t, force=True)  # Drops everything else
        assert len(pxt.list_tables()) == 0


    @pytest.mark.skip(reason='Skip until we figure out the right API for altering table attributes')
    def test_table_attrs(self, reset_db: None) -> None:
        schema = {'c': pxt.String}
        num_retained_versions = 20
        comment = 'This is a table.'
        tbl = pxt.create_table('test_table_attrs', schema, num_retained_versions=num_retained_versions, comment=comment)
        assert tbl._num_retained_versions == num_retained_versions
        assert tbl._comment == comment
        new_num_retained_versions = 30
        new_comment = 'This is an updated table.'
        tbl._num_retained_versions = new_num_retained_versions
        assert tbl._num_retained_versions == new_num_retained_versions
        tbl._comment = new_comment
        assert tbl._comment == new_comment
        tbl.revert()
        assert tbl._comment == comment
        tbl.revert()
        assert tbl._num_retained_versions == num_retained_versions

    def test_image_table(self, reset_db: None) -> None:
        n_sample_rows = 20
        schema = {
            'img': pxt.Image,
            'category': pxt.String,
            'split': pxt.String,
            'img_literal': pxt.Image,
        }
        tbl = pxt.create_table('test', schema)
        assert MediaStore.count(tbl._id) == 0

        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        sample_rows = random.sample(rows, n_sample_rows)

        # add literal image data and column
        for r in rows:
            with open(r['img'], 'rb') as f:
                r['img_literal'] = f.read()

        tbl.insert(sample_rows)
        assert MediaStore.count(tbl._id) == n_sample_rows

        # compare img and img_literal
        # TODO: make tbl.select(tbl.img == tbl.img_literal) work
        tdf = tbl.select(tbl.img, tbl.img_literal).show()
        pdf = tdf.to_pandas()
        for tup in pdf.itertuples():
            assert tup.img == tup.img_literal

        # Test adding stored image transformation
        tbl.add_column(rotated=tbl.img.rotate(30), stored=True)
        assert MediaStore.count(tbl._id) == 2 * n_sample_rows

        # Test MediaStore.stats()
        stats = list(filter(lambda x: x[0] == tbl._id, MediaStore.stats()))
        assert len(stats) == 2  # Two columns
        assert stats[0][2] == n_sample_rows  # Each column has n_sample_rows associated images
        assert stats[1][2] == n_sample_rows

        # Test that version-specific images are cleared when table is reverted
        tbl.revert()
        assert MediaStore.count(tbl._id) == n_sample_rows

        # Test that all stored images are cleared when table is dropped
        pxt.drop_table('test')
        assert MediaStore.count(tbl._id) == 0

    def test_schema_spec(self, reset_db: None) -> None:
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c 1': pxt.Int})
        assert 'invalid column name' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {}})
        assert "'type' or 'value' must be specified" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'xyz': pxt.Int}})
        assert "invalid key 'xyz'" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'stored': True}})
        assert "'type' or 'value' must be specified" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'type': 'string'}})
        assert 'must be a type or ColumnType' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'value': 1, 'type': pxt.String}})
        assert "'type' is redundant" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'value': pytest}})
        assert 'value must be a Pixeltable expression' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:

            def f() -> float:
                return 1.0

            pxt.create_table('test', {'c1': {'value': f}})
        assert 'value must be a Pixeltable expression' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': {'type': pxt.String, 'stored': 'true'}})
        assert '"stored" must be a bool' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': pxt.Required[pxt.String]}, primary_key='c2')
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': pxt.Required[pxt.String]}, primary_key=['c1', 'c2'])
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': pxt.Required[pxt.String]}, primary_key=['c2'])
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': pxt.Required[pxt.String]}, primary_key=0)
        assert 'primary_key must be a' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', {'c1': pxt.String}, primary_key='c1')
        assert 'cannot be nullable' in str(exc_info.value).lower()

        for badtype, name, suggestion in [
            (str, 'str', 'pxt.String'),
            (int, 'int', 'pxt.Int'),
            (float, 'float', 'pxt.Float'),
            (bool, 'bool', 'pxt.Bool'),
            (datetime.datetime, 'datetime.datetime', 'pxt.Timestamp'),
            (list, 'list', 'pxt.Json'),
            (dict, 'dict', 'pxt.Json'),
            (PIL.Image.Image, 'PIL.Image.Image', 'pxt.Image'),
        ]:
            with pytest.raises(excs.Error) as exc_info:
                pxt.create_table('test', {'c1': badtype})
            assert f'Standard Python type `{name}` cannot be used here; use `{suggestion}` instead' in str(exc_info.value)

    def check_bad_media(
        self, rows: list[tuple[str, bool]], col_type: type, validate_local_path: bool = True
    ) -> None:
        schema = {
            'media': col_type,
            'is_bad_media': pxt.Bool,
        }
        tbl = pxt.create_table('test', schema)

        assert len(rows) > 0
        total_bad_rows = sum([int(row['is_bad_media']) for row in rows])
        assert total_bad_rows > 0

        # Mode 1: Validation error on bad input (default)
        # we ignore the exact error here, because it depends on the media type
        with pytest.raises(excs.Error):
            tbl.insert(rows, on_error='abort')

        # Mode 2: ignore_errors=True, store error information in table
        status = tbl.insert(rows, on_error='ignore')
        _ = tbl.select(tbl.media, tbl.media.errormsg).show()
        assert status.num_rows == len(rows)
        assert status.num_excs >= total_bad_rows

        # check that we have the right number of bad and good rows
        assert tbl.where(tbl.is_bad_media == True).count() == total_bad_rows
        assert tbl.where(tbl.is_bad_media == False).count() == len(rows) - total_bad_rows

        # check error type is set correctly
        assert tbl.where((tbl.is_bad_media == True) & (tbl.media.errortype == None)).count() == 0
        assert (
            tbl.where((tbl.is_bad_media == False) & (tbl.media.errortype == None)).count() == len(rows) - total_bad_rows
        )

        # check fileurl is set for valid images, and check no file url is set for bad images
        assert tbl.where((tbl.is_bad_media == False) & (tbl.media.fileurl == None)).count() == 0
        assert tbl.where((tbl.is_bad_media == True) & (tbl.media.fileurl != None)).count() == 0

        if validate_local_path:
            # check that tbl.media is a valid local path
            paths = tbl.where(tbl.media != None).select(output=tbl.media).collect()['output']
            for path in paths:
                assert os.path.exists(path) and os.path.isfile(path)

    def test_validate_image(self, reset_db: None) -> None:
        rows = read_data_file('imagenette2-160', 'manifest_bad.csv', ['img'])
        rows = [{'media': r['img'], 'is_bad_media': r['is_bad_image']} for r in rows]
        self.check_bad_media(rows, pxt.Image, validate_local_path=False)

    def test_validate_video(self, reset_db: None) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for f in files]
        self.check_bad_media(rows, pxt.Video)

    def test_validate_audio(self, reset_db: None) -> None:
        files = get_audio_files(include_bad_audio=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_audio.mp3')} for f in files]
        self.check_bad_media(rows, pxt.Audio)

    def test_validate_docs(self, reset_db: None) -> None:
        skip_test_if_not_installed('mistune')
        valid_doc_paths = get_documents()
        invalid_doc_paths = [get_video_files()[0], get_audio_files()[0], get_image_files()[0]]
        doc_paths = valid_doc_paths + invalid_doc_paths
        is_valid = [True] * len(valid_doc_paths) + [False] * len(invalid_doc_paths)
        rows = [{'media': f, 'is_bad_media': not is_valid} for f, is_valid in zip(doc_paths, is_valid)]
        self.check_bad_media(rows, pxt.Document)

    def test_validate_external_url(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        rows = [
            {'media': 's3://open-images-dataset/validation/doesnotexist.jpg', 'is_bad_media': True},
            {'media': 'https://archive.random.org/download?file=2024-01-28.bin', 'is_bad_media': True},  # 403 error
            {'media': 's3://open-images-dataset/validation/3c02ca9ec9b2b77b.jpg', 'is_bad_media': True},  # wrong media
            # test s3 url
            {
                'media': 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4',
                'is_bad_media': False,
            },
            # test http url
            {
                'media': 'https://raw.githubusercontent.com/pixeltable/pixeltable/d8b91c5/tests/data/videos/bangkok_half_res.mp4',
                'is_bad_media': False,
            },
        ]
        self.check_bad_media(rows, pxt.Video)

    def test_create_s3_image_table(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        tbl = pxt.create_table('test', {'img': pxt.Image})
        # this is needed because reload_db() doesn't call TableVersion.drop(), which would
        # clear the file cache
        # TODO: change reset_catalog() to drop tables
        FileCache.get().clear()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 0, f'{str(cache_stats)} tbl_id={tbl._id}'
        # add computed column to make sure that external files are cached locally during insert
        tbl.add_column(rotated=tbl.img.rotate(30), stored=True)
        urls = [
            's3://open-images-dataset/validation/3c02ca9ec9b2b77b.jpg',
            's3://open-images-dataset/validation/3c13e0015b6c3bcf.jpg',
            's3://open-images-dataset/validation/3ba5380490084697.jpg',
            's3://open-images-dataset/validation/3afeb4b34f90c0cf.jpg',
            's3://open-images-dataset/validation/3b07a2c0d5c0c789.jpg',
        ]

        validate_update_status(tbl.insert({'img': url} for url in urls), expected_rows=len(urls))
        # check that we populated the cache
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls), f'{str(cache_stats)} tbl_id={tbl._id}'
        assert cache_stats.num_hits == 0
        assert FileCache.get().num_files() == len(urls)
        assert FileCache.get().num_files(tbl._id) == len(urls)
        assert FileCache.get().avg_file_size() > 0

        # query: we read from the cache
        _ = tbl.collect()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 2 * len(urls)
        assert cache_stats.num_hits == len(urls)

        # after clearing the cache, we need to re-fetch the files
        FileCache.get().clear()
        _ = tbl.collect()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls)
        assert cache_stats.num_hits == 0

        # start with fresh client and FileCache instance to test FileCache initialization with pre-existing files
        reload_catalog()
        FileCache.init()
        t = pxt.get_table('test')
        _ = t.collect()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls)
        assert cache_stats.num_hits == len(urls)

        # dropping the table also clears the file cache
        pxt.drop_table('test')
        cache_stats = FileCache.get().stats()
        assert cache_stats.total_size == 0

    def test_video_url(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        schema = {
            'payload': pxt.Int,
            'video': pxt.Video,
        }
        tbl = pxt.create_table('test', schema)
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        tbl.insert(payload=1, video=url)
        row = tbl.select(tbl.video.fileurl, tbl.video.localpath).collect()[0]
        assert row['video_fileurl'] == url
        # row[1] contains valid path to an mp4 file
        local_path = row['video_localpath']
        assert os.path.exists(local_path) and os.path.isfile(local_path)
        with av.open(local_path) as container:
             assert container.streams.video[0].codec_context.name == 'h264'

    def test_create_video_table(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        tbl = pxt.create_table('test_tbl', {'payload': pxt.Int, 'video': pxt.Video})
        args = {'video': tbl.video, 'fps': 0}
        view = pxt.create_view('test_view', tbl, iterator=FrameIterator.create(video=tbl.video, fps=0))
        view.add_column(c1=view.frame.rotate(30), stored=True)
        view.add_column(c2=view.c1.rotate(40), stored=False)
        view.add_column(c3=view.c2.rotate(50), stored=True)
        # a non-materialized column that refers to another non-materialized column
        view.add_column(c4=view.c2.rotate(60), stored=False)

        # cols computed with window functions are stored by default
        view.add_column(c5=self.window_fn(view.frame_idx, 1, group_by=view.video))

        # reload to make sure that metadata gets restored correctly
        reload_catalog()
        tbl = pxt.get_table('test_tbl')
        view = pxt.get_table('test_view')
        # we're inserting only a single row and the video column is not in position 0
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        status = tbl.insert(payload=1, video=url)
        assert status.num_excs == 0
        # * 2: we have 2 stored img cols
        assert MediaStore.count(view._id) == view.count() * 2
        # also insert a local file
        tbl.insert(payload=1, video=get_video_files()[0])
        assert MediaStore.count(view._id) == view.count() * 2

        # TODO: test inserting Nulls
        # status = tbl.insert(payload=1, video=None)
        # assert status.num_excs == 0

        # revert() clears stored images
        tbl.revert()
        tbl.revert()
        assert MediaStore.count(view._id) == 0

        with pytest.raises(excs.Error):
            # can't drop frame col
            view.drop_column('frame')
        with pytest.raises(excs.Error):
            # can't drop frame_idx col
            view.drop_column('frame_idx')

        # drop() clears stored images and the cache
        tbl.insert(payload=1, video=get_video_files()[0])
        with pytest.raises(excs.Error) as exc_info:
            pxt.drop_table('test_tbl')
        assert 'has dependents: test_view' in str(exc_info.value)
        pxt.drop_table('test_view')
        pxt.drop_table('test_tbl')
        assert MediaStore.count(view._id) == 0

    def test_video_urls(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        tbl = pxt.create_table('test', {'video': pxt.Video})

        # create a list of uris with duplicates, to test the duplicate-handling logic of CachePrefetchNode
        uris = get_multimedia_commons_video_uris(n=pxt.exec.CachePrefetchNode.BATCH_SIZE * 2)
        uris = [uri for uri in uris for _ in range(10)]
        random.seed(0)
        random.shuffle(uris)

        # clearing the file cache here makes the tests fail on Windows
        # TODO: investigate why
        #FileCache.get().clear()  # make sure we need to download the files
        validate_update_status(tbl.insert({'video': uri} for uri in uris), expected_rows=len(uris))
        row = tbl.select(tbl.video.fileurl, tbl.video.localpath).head(1)[0]
        assert row['video_fileurl'] == uris[0]
        # tbl.video.localpath contains valid path to an mp4 file
        local_path = row['video_localpath']
        assert os.path.exists(local_path) and os.path.isfile(local_path)
        with av.open(local_path) as container:
            assert container.streams.video[0].codec_context.name == 'h264'

    def test_insert_nulls(self, reset_db: None) -> None:
        schema = {
            'c1': pxt.String,
            'c2': pxt.Int,
            'c3': pxt.Float,
            'c4': pxt.Bool,
            'c5': pxt.Array[(2, 3), pxt.Int],
            'c6': pxt.Json,
            'c7': pxt.Image,
            'c8': pxt.Video,
        }
        t = pxt.create_table('test1', schema)
        status = t.insert(c1='abc')
        assert status.num_rows == 1
        assert status.num_excs == 0

    def test_insert(self, reset_db: None) -> None:
        schema = {
            'c1': pxt.Required[pxt.String],
            'c2': pxt.Required[pxt.Int],
            'c3': pxt.Required[pxt.Float],
            'c4': pxt.Required[pxt.Bool],
            'c5': pxt.Required[pxt.Array[(2, 3), pxt.Int]],
            'c6': pxt.Required[pxt.Json],
            'c7': pxt.Required[pxt.Image],
            'c8': pxt.Required[pxt.Video],
        }
        tbl_name = 'test1'
        t = pxt.create_table(tbl_name, schema)
        rows = create_table_data(t)
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        # alternate (kwargs) insert syntax
        status = t.insert(
            c1='string',
            c2=91,
            c3=1.0,
            c4=True,
            c5=np.ones((2, 3), dtype=np.dtype(np.int64)),
            c6={'key': 'val'},
            c7=get_image_files()[0],
            c8=get_video_files()[0],
        )
        assert status.num_rows == 1
        assert status.num_excs == 0

        # drop column, then add it back; insert still works
        t.drop_column('c4')
        t.add_column(c4=pxt.Bool)
        reload_catalog()
        t = pxt.get_table(tbl_name)
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        # empty input
        with pytest.raises(excs.Error) as exc_info:
            t.insert([])
        assert 'empty' in str(exc_info.value)

        # missing column
        with pytest.raises(excs.Error) as exc_info:
            # drop first column
            col_names = list(rows[0].keys())[1:]
            new_rows = [{col_name: row[col_name] for col_name in col_names} for row in rows]
            t.insert(new_rows)
        assert 'Missing' in str(exc_info.value)

        # incompatible schema
        for (col_name, col_type), value_col_name in zip(
            schema.items(), ['c2', 'c3', 'c5', 'c5', 'c6', 'c7', 'c2', 'c2']
        ):
            pxt.drop_table(tbl_name, ignore_errors=True)
            t = pxt.create_table(tbl_name, {col_name: col_type})
            with pytest.raises(excs.Error) as exc_info:
                t.insert({col_name: r[value_col_name]} for r in rows)
            assert 'expected' in str(exc_info.value).lower()

        # rows not list of dicts
        pxt.drop_table(tbl_name, ignore_errors=True)
        t = pxt.create_table(tbl_name, {'c1': pxt.String})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(['1'])
        assert 'list of dictionaries' in str(exc_info.value)

        # bad null value
        pxt.drop_table(tbl_name, ignore_errors=True)
        t = pxt.create_table(tbl_name, {'c1': pxt.Required[pxt.String]})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(c1=None)
        assert 'expected non-None' in str(exc_info.value)

        # bad array literal
        pxt.drop_table(tbl_name, ignore_errors=True)
        t = pxt.create_table(tbl_name, {'c5': pxt.Array[(2, 3), pxt.Int]})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(c5=np.ndarray((3, 2)))
        assert 'expected ndarray((2, 3)' in str(exc_info.value)

        # test that insert skips expression evaluation for
        # any columns that are not part of the current schema.
        @pxt.udf(_force_stored=True)
        def bad_udf(x: str) -> str:
            assert False
        t = pxt.create_table('test', {'str_col': pxt.String})
        t.add_column(bad=bad_udf(t.str_col))  # Succeeds because the table has no data
        t.drop_column('bad')
        t.insert(str_col='Hello there.') # Succeeds because column 'bad' is dropped
        pxt.drop_table('test')

    def test_insert_string_with_null(self, reset_db) -> None:
        t = pxt.create_table('test', {'c1': pxt.String})

        t.insert([{'c1': 'this is a python\x00string'}])
        assert t.count() == 1
        for tup in t.collect():
            assert tup['c1'] == 'this is a python string'

    def test_query(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        col_names = ['c1', 'c2', 'c3', 'c4', 'c5']
        t = make_tbl('test', col_names)
        rows = create_table_data(t)
        t.insert(rows)
        _ = t.show(n=0)

        # test querying existing table
        reload_catalog()
        t2 = pxt.get_table('test')
        _ = t2.show(n=0)

    def test_batch_update(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        num_rows = t.count()
        # update existing rows
        validate_update_status(t.batch_update([{'c1': '1', 'c2': 1}, {'c1': '2', 'c2': 2}]), expected_rows=2)
        assert t.count() == num_rows  # make sure we didn't lose any rows
        assert t.where(t.c2 == 1).collect()[0]['c1'] == '1'
        assert t.where(t.c2 == 2).collect()[0]['c1'] == '2'
        # the same, but with _rowid
        validate_update_status(
            t.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two', '_rowid': (2,)}]), expected_rows=2
        )
        assert t.count() == num_rows  # make sure we didn't lose any rows
        assert t.where(t.c2 == 1).collect()[0]['c1'] == 'one'
        assert t.where(t.c2 == 2).collect()[0]['c1'] == 'two'

        # unknown primary key: raise error
        with pytest.raises(excs.Error) as exc_info:
            _ = t.batch_update([{'c1': 'eins', 'c2': 1}, {'c1': 'zweihundert', 'c2': 200}], if_not_exists='error')
        assert '1 row(s) not found' in str(exc_info.value).lower()

        # unknown primary key: ignore
        validate_update_status(
            t.batch_update([{'c1': 'eins', 'c2': 1}, {'c1': 'zweihundert', 'c2': 200}], if_not_exists='ignore'),
            expected_rows=1)
        assert t.count() == num_rows  # make sure we didn't lose any rows
        assert t.where(t.c2 == 1).collect()[0]['c1'] == 'eins'
        assert t.where(t.c2 == 200).count() == 0

        # unknown primary key: insert
        validate_update_status(
            t.batch_update([{'c1': 'zwei', 'c2': 2}, {'c1': 'zweihundert', 'c2': 200}], if_not_exists='insert'),
            expected_rows=2)
        assert t.count() == num_rows + 1
        assert t.where(t.c2 == 2).collect()[0]['c1'] == 'zwei'
        assert t.where(t.c2 == 200).collect()[0]['c1'] == 'zweihundert'

        # test composite primary key
        schema = {'c1': pxt.Required[pxt.String], 'c2': pxt.Required[pxt.Int], 'c3': pxt.Float}
        t = pxt.create_table('composite', schema, primary_key=['c1', 'c2'])
        rows = [{'c1': str(i), 'c2': i, 'c3': float(i)} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        validate_update_status(
            t.batch_update([{'c1': '1', 'c2': 1, 'c3': 2.0}, {'c1': '2', 'c2': 2, 'c3': 3.0}]), expected_rows=2
        )
        assert t.count() == len(rows)
        assert t.where(t.c2 == 1).collect()[0]['c3'] == 2.0
        assert t.where(t.c2 == 2).collect()[0]['c3'] == 3.0

        with pytest.raises(excs.Error) as exc_info:
            # can't mix _rowid with primary key
            _ = t.batch_update([{'c1': '1', 'c2': 1, 'c3': 2.0, '_rowid': (1,)}])
        assert 'c1 is a primary key column' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # bad literal
            _ = t.batch_update([{'c2': 1, 'c3': 'a'}])
        assert "'a' is not a valid literal" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # missing primary key column
            t.batch_update([{'c1': '1', 'c3': 2.0}])
        assert 'primary key columns (c2) missing' in str(exc_info.value).lower()

        # table without primary key
        t2 = pxt.create_table('no_pk', schema)
        validate_update_status(t2.insert(rows), expected_rows=10)
        with pytest.raises(excs.Error) as exc_info:
            _ = t2.batch_update([{'c1': '1', 'c2': 1, 'c3': 2.0}])
        assert 'must have primary key for batch update' in str(exc_info.value).lower()

        # updating with _rowid still works
        validate_update_status(
            t2.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two', '_rowid': (2,)}]), expected_rows=2
        )
        assert t2.count() == len(rows)
        assert t2.where(t2.c2 == 1).collect()[0]['c1'] == 'one'
        assert t2.where(t2.c2 == 2).collect()[0]['c1'] == 'two'
        with pytest.raises(AssertionError):
            # some rows are missing rowids
            _ = t2.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two'}])

    def test_update(self, test_tbl: pxt.Table, small_img_tbl: pxt.Table) -> None:
        t = test_tbl
        # update every type with a literal
        test_cases = [
            ('c1', 'new string'),
            # TODO: ('c1n', None),
            ('c3', -1.0),
            ('c4', True),
            ('c5', datetime.datetime.now()),
            ('c6', [{'x': 1, 'y': 2}]),
        ]
        count = t.count()
        for col_name, literal in test_cases:
            status = t.update({col_name: literal}, where=t.c3 < 10.0, cascade=False)
            assert status.num_rows == 10
            assert status.updated_cols == [f'{t._name}.{col_name}']
            assert t.count() == count
            t.revert()

        # exchange two columns
        t.add_column(float_col=pxt.Float)
        t.update({'float_col': 1.0})
        float_col_vals = t.order_by(t.c2).select(t.float_col).collect().to_pandas()['float_col']
        c3_vals = t.order_by(t.c2).select(t.c3).collect().to_pandas()['c3']
        assert np.all(float_col_vals == pd.Series([1.0] * t.count()))
        t.update({'c3': t.float_col, 'float_col': t.c3})
        assert np.all(t.order_by(t.c2).select(t.c3).collect().to_pandas()['c3'] == float_col_vals)
        assert np.all(t.order_by(t.c2).select(t.float_col).collect().to_pandas()['float_col'] == c3_vals)
        t.revert()

        # update column that is used in computed cols
        t.add_column(computed1=t.c3 + 1)
        t.add_column(computed2=t.computed1 + 1)
        t.add_column(computed3=t.c3 + 3)

        # cascade=False
        computed1 = t.order_by(t.computed1).collect().to_pandas()['computed1']
        computed2 = t.order_by(t.computed2).collect().to_pandas()['computed2']
        computed3 = t.order_by(t.computed3).collect().to_pandas()['computed3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1
        # update to a value that also satisfies the where clause
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=False)
        assert status.num_rows == 10
        assert status.updated_cols == ['test_tbl.c3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        # computed cols are not updated
        assert np.all(t.order_by(t.computed1).collect().to_pandas()['computed1'] == computed1)
        assert np.all(t.order_by(t.computed2).collect().to_pandas()['computed2'] == computed2)
        assert np.all(t.order_by(t.computed3).collect().to_pandas()['computed3'] == computed3)

        # revert, then verify that we're back to where we started
        reload_catalog()
        t = pxt.get_table(t._name)
        t.revert()
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1

        # cascade=True
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=True)
        assert status.num_rows == 10
        assert set(status.updated_cols) == set(
            ['test_tbl.c3', 'test_tbl.computed1', 'test_tbl.computed2', 'test_tbl.computed3']
        )
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        assert np.all(t.order_by(t.computed1).collect().to_pandas()['computed1'][:10] == pd.Series([1.0] * 10))
        assert np.all(t.order_by(t.computed2).collect().to_pandas()['computed2'][:10] == pd.Series([2.0] * 10))
        assert np.all(t.order_by(t.computed3).collect().to_pandas()['computed3'][:10] == pd.Series([3.0] * 10))

        # bad update spec
        with pytest.raises(excs.Error) as excinfo:
            t.update({1: 1})
        assert 'dict key' in str(excinfo.value)

        # unknown column
        with pytest.raises(excs.Error) as excinfo:
            t.update({'unknown': 1})
        assert 'unknown unknown' in str(excinfo.value)

        # incompatible type
        with pytest.raises(excs.Error) as excinfo:
            t.update({'c1': 1})
        assert 'not compatible' in str(excinfo.value)

        # can't update primary key
        with pytest.raises(excs.Error) as excinfo:
            t.update({'c2': 1})
        assert 'primary key' in str(excinfo.value)

        # can't update computed column
        with pytest.raises(excs.Error) as excinfo:
            t.update({'computed1': 1})
        assert 'is computed' in str(excinfo.value)

        # non-expr
        with pytest.raises(excs.Error) as excinfo:
            t.update({'c3': lambda c3: math.sqrt(c3)})
        assert 'not a recognized' in str(excinfo.value)

        # non-Predicate filter
        with pytest.raises(excs.Error) as excinfo:
            t.update({'c3': 1.0}, where=lambda c2: c2 == 10)
        assert 'predicate' in str(excinfo.value)

        img_t = small_img_tbl

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_cascading_update(self, test_tbl: pxt.InsertableTable) -> None:
        t = test_tbl
        t.add_column(d1=t.c3 - 1)
        # add column that can be updated
        t.add_column(c10=pxt.Float)
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_column(d2=t.c3 - t.c10)
        r1 = t.where(t.c2 < 5).select(t.c3 + 1.0, t.c10 - 1.0, t.c3, 2.0).order_by(t.c2).collect()
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        r2 = t.where(t.c2 < 5).select(t.c3, t.c10, t.d1, t.d2).order_by(t.c2).collect()
        assert_resultset_eq(r1, r2)

    def test_delete(self, test_tbl: pxt.Table, small_img_tbl: pxt.Table) -> None:
        t = test_tbl

        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 10
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1
        status = t.delete(where=t.c3 < 10.0)
        assert status.num_rows == 10
        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 0
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1

        # revert, then verify that we're back where we started
        reload_catalog()
        t = pxt.get_table(t._name)
        t.revert()
        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 10
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1

        # non-Predicate filter
        with pytest.raises(excs.Error) as excinfo:
            t.delete(where=lambda c2: c2 == 10)
        assert 'predicate' in str(excinfo.value)

        img_t = small_img_tbl

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.delete(where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_computed_cols(self, reset_db: None) -> None:
        schema = {'c1': pxt.Int, 'c2': pxt.Float, 'c3': pxt.Json}
        t: pxt.InsertableTable = pxt.create_table('test', schema)
        status = t.add_column(c4=t.c1 + 1)
        assert status.num_excs == 0
        status = t.add_column(c5=t.c4 + 1)
        assert status.num_excs == 0
        status = t.add_column(c6=t.c1 / t.c2)
        assert status.num_excs == 0
        status = t.add_column(c7=t.c6 * t.c2)
        assert status.num_excs == 0
        status = t.add_column(c8=t.c3.detections['*'].bounding_box)
        assert status.num_excs == 0
        status = t.add_column(c9=t.c2.apply(math.sqrt, col_type=pxt.Float))
        assert status.num_excs == 0

        # unstored cols that compute window functions aren't currently supported
        with pytest.raises(excs.Error):
            t.add_column(c10=pxtf.sum(t.c1, group_by=t.c1), stored=False)

        # Column.dependent_cols are computed correctly
        assert len(t.c1.col.dependent_cols) == 3
        assert len(t.c2.col.dependent_cols) == 4
        assert len(t.c3.col.dependent_cols) == 1
        assert len(t.c4.col.dependent_cols) == 2
        assert len(t.c5.col.dependent_cols) == 1
        assert len(t.c6.col.dependent_cols) == 2
        assert len(t.c7.col.dependent_cols) == 1
        assert len(t.c8.col.dependent_cols) == 0

        rows = create_table_data(t, ['c1', 'c2', 'c3'], num_rows=10)
        t.insert(rows)
        _ = t.show()

        # not allowed to pass values for computed cols
        with pytest.raises(excs.Error):
            rows2 = create_table_data(t, ['c1', 'c2', 'c3', 'c4'], num_rows=10)
            t.insert(rows2)

        # test loading from store
        reload_catalog()
        t2 = pxt.get_table('test')
        t2_columns = t2._tbl_version_path.columns()
        assert len(t2_columns) == len(t2.columns)
        t_columns = t._tbl_version_path.columns()
        assert len(t_columns) == len(t2_columns)
        for i in range(len(t_columns)):
            if t_columns[i].value_expr is not None:
                assert t_columns[i].value_expr.equals(t2_columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        status = t.insert(rows)
        assert status.num_excs == 0
        res = t.collect()
        tbl_df = t.collect().to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(excs.Error):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    def test_expr_udf_computed_cols(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int})
        rows = [{'c1': i} for i in range(100)]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        status = t.add_column(c2=t.c1 + 1)
        assert status.num_excs == 0
        # call with positional arg
        status = t.add_column(c3=self.add1(t.c1))
        assert status.num_excs == 0
        # call with keyword arg
        status = t.add_column(c4=self.add1(a=t.c1))
        assert status.num_excs == 0

        # TODO: how to verify the output?
        describe_output = repr(t)
        # 'add1' didn't get swallowed/the expr udf is still visible in the column definition
        assert 'add1' in describe_output

        def check(t: pxt.Table) -> None:
            assert_resultset_eq(t.select(t.c1 + 1).order_by(t.c1).collect(), t.select(t.c2).order_by(t.c1).collect())
            assert_resultset_eq(t.select(t.c1 + 1).order_by(t.c1).collect(), t.select(t.c3).order_by(t.c1).collect())

        check(t)
        # test loading from store
        reload_catalog()
        t = pxt.get_table('test')
        check(t)

        # make sure we can still insert data and that computed cols are still set correctly
        status = t.insert(rows)
        assert status.num_excs == 0
        check(t)

    def test_computed_col_exceptions(self, reset_db: None, test_tbl: catalog.Table) -> None:
        # exception during insert()
        schema = {'c2': pxt.Int}
        rows = list(test_tbl.select(test_tbl.c2).collect())
        t = pxt.create_table('test_insert', schema)
        status = t.add_column(add1=self.f2(self.f1(t.c2)))
        assert status.num_excs == 0
        status = t.insert(rows, on_error='ignore')
        assert status.num_excs >= 10
        assert 'test_insert.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10

        # exception during add_column()
        t = pxt.create_table('test_add_column', schema)
        status = t.insert(rows)
        assert status.num_rows == 100
        assert status.num_excs == 0

        with pytest.raises(excs.Error) as exc:
            t.add_column(add1=self.f2(self.f1(t.c2)))
        assert 'division by zero' in str(exc.value)

        # on_error='abort' is the default
        with pytest.raises(excs.Error) as exc:
            t.add_column(add1=self.f2(self.f1(t.c2)), on_error='abort')
        assert 'division by zero' in str(exc.value)

        # on_error='ignore' stores the exception in errortype/errormsg
        status = t.add_column(add1=self.f2(self.f1(t.c2)), on_error='ignore')
        assert status.num_excs == 10
        assert 'test_add_column.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10
        msgs = t.select(msg=t.add1.errormsg).collect()['msg']
        assert sum('division by zero' in msg for msg in msgs if msg is not None) == 10

    def _test_computed_img_cols(self, t: catalog.Table, stores_img_col: bool) -> None:
        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        rows = [{'img': r['img']} for r in rows[:20]]
        status = t.insert(rows)
        assert status.num_rows == 20
        _ = t.count()
        _ = t.show()
        assert MediaStore.count(t._id) == t.count() * stores_img_col

        # test loading from store
        reload_catalog()
        t2 = pxt.get_table(t._name)
        assert len(t.columns) == len(t2.columns)
        t_columns = t._tbl_version_path.columns()
        t2_columns = t2._tbl_version_path.columns()
        assert len(t_columns) == len(t2_columns)
        for i in range(len(t_columns)):
            if t_columns[i].value_expr is not None:
                assert t_columns[i].value_expr.equals(t2_columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert(rows)
        assert MediaStore.count(t2._id) == t2.count() * stores_img_col
        res = t2.collect()
        tbl_df = t2.collect().to_pandas()

        # revert also removes computed images
        t2.revert()
        assert MediaStore.count(t2._id) == t2.count() * stores_img_col

    @pxt.udf
    def img_fn_with_exc(img: PIL.Image.Image) -> PIL.Image.Image:
        raise RuntimeError

    def test_computed_img_cols(self, reset_db: None) -> None:
        schema = {'img': pxt.Image}
        t = pxt.create_table('test', schema)
        t.add_column(c2=t.img.width)
        # c3 is not stored by default
        t.add_column(c3=t.img.rotate(90), stored=False)
        self._test_computed_img_cols(t, stores_img_col=False)

        t = pxt.create_table('test2', schema)
        # c3 is now stored
        t.add_column(c3=t.img.rotate(90))
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t[t.c3.errortype].collect()

        # computed img col with exceptions
        t = pxt.create_table('test3', schema)
        t.add_column(c3=self.img_fn_with_exc(t.img))
        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        rows = [{'img': r['img']} for r in rows[:20]]
        t.insert(rows, on_error='ignore')
        _ = t[t.c3.errortype].collect()

    def test_computed_window_fn(self, reset_db: None, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # backfill
        t.add_column(c9=pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3))

        schema = {'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Bool}
        new_t = pxt.create_table('insert_test', schema)
        new_t.add_column(c5=t.c2.apply(lambda x: x * x, col_type=pxt.Int))
        new_t.add_column(c6=pxtf.sum(new_t.c5, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.collect()

    def test_revert(self, reset_db: None) -> None:
        t1 = make_tbl('test1', ['c1', 'c2'])
        assert t1._version == 0
        rows1 = create_table_data(t1)
        t1.insert(rows1)
        assert t1.count() == len(rows1)
        assert t1._version == 1
        rows2 = create_table_data(t1)
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1._version == 2
        t1.revert()
        assert t1.count() == len(rows1)
        assert t1._version == 1
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1._version == 2

        # can't revert past version 0
        t1.revert()
        t1.revert()
        with pytest.raises(excs.Error) as excinfo:
            t1.revert()
        assert 'version 0' in str(excinfo.value)

    def test_add_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t.add_column(add1=pxt.Int)
        # Make sure that `name` and `id` are allowed, i.e., not reserved as system names
        t.add_column(name=pxt.String)
        t.add_column(id=pxt.String)
        assert len(t.columns) == num_orig_cols + 3

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.Required[pxt.Int])
        assert 'cannot add non-nullable' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.Int, add3=pxt.String)
        assert 'requires exactly one keyword argument' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(pos=pxt.String)
        assert "'pos' is a reserved name in pixeltable" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as excs_info:
            _ = t.add_column(add_column=pxt.Int)
        assert "'add_column' is a reserved name in pixeltable" in str(excs_info.value).lower()

        with pytest.raises(excs.Error) as excs_info:
            _ = t.add_column(insert=pxt.Int)
        assert "'insert' is a reserved name in pixeltable" in str(excs_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.Int, stored=False)
        assert 'stored=false only applies' in str(exc_info.value).lower()

        # duplicate name
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(c1=pxt.Int)
        assert 'duplicate column name' in str(exc_info.value).lower()

        # 'stored' kwarg only applies to computed image columns
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=pxt.Int, stored=False)
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=pxt.Image, stored=False)
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=(t.c2 + t.c3), stored=False)

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols + 3

        # revert() works
        t.revert()
        t.revert()
        t.revert()
        assert len(t.columns) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols

    def test_add_column_setitem(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t['add1'] = pxt.Int
        assert len(t.columns) == num_orig_cols + 1
        t['computed1'] = t.c2 + 1
        assert len(t.columns) == num_orig_cols + 2

        with pytest.raises(excs.Error) as exc_info:
            _ = t['pos'] = pxt.String
        assert "'pos' is a reserved name in pixeltable" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add_column'] = pxt.String
        assert "'add_column' is a reserved name in pixeltable" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t[2] = pxt.String
        assert 'must be a string' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add 2'] = pxt.String
        assert 'invalid column name' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add2'] = {'value': t.c2 + 1, 'type': pxt.String}
        assert 'column spec must be a columntype, expr, or type' in str(exc_info.value).lower()

        # duplicate name
        with pytest.raises(excs.Error) as exc_info:
            _ = t['c1'] = pxt.Int
        assert 'duplicate column name' in str(exc_info.value).lower()

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols + 2

        # revert() works
        t.revert()
        t.revert()
        assert len(t.columns) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols

    def test_drop_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t.drop_column('c1')
        assert len(t.columns) == num_orig_cols - 1
        assert 'c1' not in t.columns
        with pytest.raises(AttributeError) as exc_info:
            _ = t.c1
        assert 'column c1 unknown' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.drop_column('c1')
        assert "column 'c1' unknown" in str(exc_info.value).lower()

        assert 'unknown' not in t.columns
        with pytest.raises(excs.Error) as exc_info:
            t.drop_column('unknown')
        assert "column 'unknown' unknown" in str(exc_info.value).lower()
        with pytest.raises(AttributeError) as exc_info:
            t.drop_column(t.unknown)
        assert 'column unknown unknown' in str(exc_info.value).lower()

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols - 1

        # revert() works
        t.revert()
        assert len(t.columns) == num_orig_cols
        assert 'c1' in t.columns
        _ = t.c1

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns) == num_orig_cols
        assert 'c1' in t.columns
        _ = t.c1

    def test_drop_column_via_reference(self, reset_db) -> None:
        t1 = pxt.create_table('test1', {'c1': pxt.String, 'c2': pxt.String})
        t1.insert([{'c1': 'a1', 'c2': 'b1'}, {'c1': 'a2', 'c2': 'b2'}])
        t2 = pxt.create_table('test2', {'c1': pxt.String, 'c2': pxt.String})

        # cannot pass another table's column reference
        with pytest.raises(excs.Error) as exc_info:
            t1.drop_column(t2.c2)
        assert 'unknown column: test2.c2' in str(exc_info.value).lower()
        assert 'c2' in t1.columns
        assert 'c2' in t2.columns
        _ = t1.c2
        _ = t2.c2

        t1.drop_column(t1.c2)
        assert 'c2' not in t1.columns
        with pytest.raises(AttributeError) as exc_info:
            _ = t1.c2
        assert 'column c2 unknown' in str(exc_info.value).lower()
        with pytest.raises(AttributeError) as exc_info:
            t1.drop_column(t1.c2)
        assert 'column c2 unknown' in str(exc_info.value).lower()
        assert 'c2' in t2.columns
        pxt.drop_table(t1)
        pxt.drop_table(t2)

    def test_rename_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t.rename_column('c1', 'c1_renamed')
        assert len(t.columns) == num_orig_cols

        def check_rename(t: pxt.Table, known: str, unknown: str) -> None:
            with pytest.raises(AttributeError) as exc_info:
                _ = t.select(t[unknown]).collect()
            assert 'unknown' in str(exc_info.value).lower()
            _ = t.select(t[known]).collect()

        check_rename(t, 'c1_renamed', 'c1')

        # unknown column
        with pytest.raises(excs.Error):
            t.rename_column('unknown', 'unknown_renamed')
        # bad name
        with pytest.raises(excs.Error):
            t.rename_column('c2', 'bad name')
        # existing name
        with pytest.raises(excs.Error):
            t.rename_column('c2', 'c3')

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        check_rename(t, 'c1_renamed', 'c1')

        # revert() works
        _ = t.select(t.c1_renamed).collect()
        t.revert()
        _ = t.select(t.c1).collect()
        # check_rename(t, 'c1', 'c1_renamed')

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        check_rename(t, 'c1', 'c1_renamed')

    def test_add_computed_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        status = t.add_column(add1=t.c2 + 10)
        assert status.num_excs == 0
        _ = t.show()

        # TODO(aaron-siegel): This has to be commented out. See explanation in test_exprs.py.
        # with pytest.raises(excs.Error):
        #     t.add_column(add2=(t.c2 - 10) / (t.c3 - 10))

        # with exception in Python for c6.f2 == 10
        status = t.add_column(add2=(t.c6.f2 - 10) / (t.c6.f2 - 10), on_error='ignore')
        assert status.num_excs == 1
        result = t.where(t.add2.errortype != None).select(t.c6.f2, t.add2, t.add2.errortype, t.add2.errormsg).show()
        assert len(result) == 1

        # test case: exceptions in dependencies prevent execution of dependent exprs
        status = t.add_column(add3=self.f2(self.f1(t.c2)), on_error='ignore')
        assert status.num_excs == 10
        result = t.where(t.add3.errortype != None).select(t.c2, t.add3, t.add3.errortype, t.add3.errormsg).show()
        assert len(result) == 10

    def test_add_embedding_as_computed_column(self, reload_tester: ReloadTester) -> None:
        skip_test_if_not_installed('transformers')
        t = pxt.create_table('t', {'s': pxt.String})
        t.add_computed_column(s2=pxt.functions.huggingface.sentence_transformer(t.s, model_id='all-mpnet-base-v2'))
        df = t.select()
        _ = reload_tester.run_query(df)
        _ = reload_tester.run_reload_test(df)

        v = pxt.create_view('v', t)
        v.add_computed_column(s3=pxt.functions.huggingface.sentence_transformer(t.s, model_id='all-mpnet-base-v2'))
        df = v.select()
        _ = reload_tester.run_query(df)
        _ = reload_tester.run_reload_test(df)

    def test_computed_column_types(self, reset_db: None) -> None:
        t = pxt.create_table(
            'test',
            {
                'c1': pxt.Int,
                'c1_r': pxt.Required[pxt.Int],
                'c2': pxt.String,
                'c2_r': pxt.Required[pxt.String],
            }
        )

        # Ensure that arithmetic and (non-nullable) function call expressions inherit nullability from their arguments
        t.add_column(arith=t.c1 + 1)
        t.add_column(arith_r=t.c1_r + 1)
        t.add_column(func=t.c2.upper())
        t.add_column(func_r=t.c2_r.upper())

        assert t.get_metadata()['schema'] == {
            'c1': pxt.IntType(nullable=True),
            'c1_r': pxt.IntType(nullable=False),
            'c2': pxt.StringType(nullable=True),
            'c2_r': pxt.StringType(nullable=False),
            'arith': pxt.IntType(nullable=True),
            'arith_r': pxt.IntType(nullable=False),
            'func': pxt.StringType(nullable=True),
            'func_r': pxt.StringType(nullable=False),
        }

    def test_repr(self, test_tbl: catalog.Table) -> None:
        skip_test_if_not_installed('sentence_transformers')

        v = pxt.create_view('test_view', test_tbl)
        pxt.create_dir('test_dir')
        v2 = pxt.create_view('test_subview', v, comment='This is an intriguing table comment.')
        fn = lambda x: np.full((3, 4), x)
        v2.add_column(computed1=v2.c2.apply(fn, col_type=pxt.Array[(3, 4), pxt.Int]))
        v2.add_embedding_index(
            'c1',
            string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='all-mpnet-base-v2')
        )
        v2._link_external_store(MockProject.create(v2, 'project', {}, {}))
        v2.describe()

        r = repr(v2)
        assert strip_lines(r) == strip_lines(
            '''View
            'test_subview'
            (of 'test_view', 'test_tbl')

            Column Name                          Type           Computed With
              computed1  Required[Array[(3, 4), Int]]            <lambda>(c2)
                     c1              Required[String]
                    c1n                        String
                     c2                 Required[Int]
                     c3               Required[Float]
                     c4                Required[Bool]
                     c5           Required[Timestamp]
                     c6                Required[Json]
                     c7                Required[Json]
                     c8  Required[Array[(2, 3), Int]]  [[1, 2, 3], [4, 5, 6]]

            Index Name Column  Metric                                          Embedding
                  idx0     c1  cosine  sentence_transformer(sentence, model_id='all-m...

            External Store         Type
                   project  MockProject

            COMMENT: This is an intriguing table comment.'''
        )
        _ = v2._repr_html_()  # TODO: Is there a good way to test this output?

        c = repr(v2.c1)
        assert strip_lines(c) == strip_lines(
            '''Column
            'c1'
            (of table 'test_tbl')

            Column Name              Type Computed With
                     c1  Required[String]'''
        )
        _ = v2.c1._repr_html_()

    def test_common_col_names(self, reset_db: None) -> None:
        """Make sure that commonly used column names don't collide with Table member vars"""
        names = ['id', 'name', 'version', 'comment']
        schema = {name: pxt.Int for name in names}
        tbl = pxt.create_table('test', schema)
        status = tbl.insert({name: id for name in names} for id in range(10))
        assert status.num_rows == 10
        assert status.num_excs == 0
        assert tbl.count() == 10
        # we can create references to those column via __getattr__
        _ = tbl.select(tbl.id, tbl._name).collect()

    def test_table_api_on_dropped_table(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int, 'c2': pxt.String})
        pxt.drop_table('test')

        # confirm the _check_is_dropped() method raises the expected exception
        with pytest.raises(excs.Error) as exc_info:
            t._check_is_dropped()
        assert 'table test has been dropped' in str(exc_info.value).lower()
        expected_err_msg = 'table test has been dropped'

        # verify that all the user facing APIs acting on a table handle
        # of a dropped table, raised the above exception gracefully
        # before SQL execution.

        # verify basic table properties/methods.
        # A _check_is_dropped() call in these helps to catch the error
        # for many other user facing APIs that go via them to SQL execution.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.columns
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t._df()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t._schema
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t._tbl_version
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t._version
        assert expected_err_msg in str(exc_info.value).lower()
        # earlier this returned the column reference object
        with pytest.raises(excs.Error) as exc_info:
            _ = t.c1
        assert expected_err_msg in str(exc_info.value).lower()

        # verify DML APIs. These were failing with error during
        # SQL execution before.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.delete(t.c1 > 3)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.insert([{'c1': 1, 'c2': 'abc'}])
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.update({'c1': 2})
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.batch_update([{'c1': 2, 'c2': 'f'}])
        assert expected_err_msg in str(exc_info.value).lower()

        # verify DDL APIs. Most of these already had the check.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(c2=pxt.Int)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_columns({'c2': pxt.Int})
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_computed_column(c3=t.c1 +10)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_embedding_index('c2', string_embed=str.split)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.drop_embedding_index(column='c2')
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.drop_index(column='c2')
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.drop_column('c1')
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.rename_column('c1', 'c1_renamed')
        assert expected_err_msg in str(exc_info.value).lower()

        # verify df/query APIs. Most of these won't fail until
        # materialized via collect/show/count before, and
        # were failing with error during SQL execution.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.group_by(t.c1)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.where(t.c1 > 3)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.order_by(t.c1)
        assert expected_err_msg in str(exc_info.value).lower()
        # RESOLVE: the t.queries and t.query() APIs dont seem to work.
        # hits an assrtion failure in the code.
        #t.query('select c1 from test')
        #t.queries(['select c1 from test', 'select c2 from test'])

        with pytest.raises(excs.Error) as exc_info:
            _ = t.collect()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.count()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.head()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.limit(1)
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.tail()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.show()
        assert expected_err_msg in str(exc_info.value).lower()

        # verify metadata-ish APIs. Many of these would return
        # results and not error out before. Some of these were
        # failing with error during SQL execution.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.describe()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.get_metadata()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.list_views()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.__repr__()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t._repr_html_()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.external_stores()
        assert expected_err_msg in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.unlink_external_stores()
        assert expected_err_msg in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.sync()
        assert expected_err_msg in str(exc_info.value).lower()

        # verify dataset APIs. These were failing with error during
        # SQL execution before.
        with pytest.raises(excs.Error) as exc_info:
            _ = t.to_coco_dataset()
        assert expected_err_msg in str(exc_info.value).lower()
        # Earlier raised a psycopg.errors.UndefinedTable exception
        with pytest.raises(excs.Error) as exc_info:
            _ = t.to_pytorch_dataset()
        assert expected_err_msg in str(exc_info.value).lower()

        # verify transaction APIs. We cannot revert a drop table operation.
        with pytest.raises(excs.Error) as exc_info:
            t.revert()
        assert expected_err_msg in str(exc_info.value).lower()

