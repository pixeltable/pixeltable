import datetime
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Optional, Union, _GenericAlias  # type: ignore[attr-defined]

import av
import numpy as np
import pandas as pd
import PIL
import pytest
from jsonschema.exceptions import ValidationError

import pixeltable as pxt
import pixeltable.functions as pxtf
import pixeltable.type_system as ts
from pixeltable import catalog, exceptions as excs, func
from pixeltable.exprs import ColumnRef
from pixeltable.func import Batch
from pixeltable.io.external_store import MockProject
from pixeltable.iterators import FrameIterator
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

from .utils import (
    TESTS_DIR,
    ReloadTester,
    assert_resultset_eq,
    create_table_data,
    get_audio_files,
    get_documents,
    get_image_files,
    get_multimedia_commons_video_uris,
    get_video_files,
    make_tbl,
    read_data_file,
    reload_catalog,
    skip_test_if_not_installed,
    strip_lines,
    validate_update_status,
)

test_unstored_table_base_val: int = 0


@pxt.udf(batch_size=20)
def add_unstored_table_base_val(vals: Batch[int]) -> Batch[int]:
    results = []
    for val in vals:
        results.append(val + test_unstored_table_base_val)
    return results


class TestTable:
    # exc for a % 10 == 0
    @staticmethod
    @pxt.udf
    def f1(a: int) -> float:
        return a / (a % 10)

    # exception for a == None; this should not get triggered
    @staticmethod
    @pxt.udf
    def f2(a: float) -> float:
        return a + 1

    @staticmethod
    @pxt.expr_udf
    def add1(a: int) -> int:
        return a + 1

    @staticmethod
    @pxt.udf
    def function_with_error(a: int, b: int) -> int:
        if a == b:
            raise KeyboardInterrupt
        return a + 1

    @pxt.uda(requires_order_by=True, allows_window=True)
    class window_fn(pxt.Aggregator):
        def __init__(self) -> None:
            pass

        def update(self, i: int) -> None:
            pass

        def value(self) -> int:
            return 1

    def test_create(self, reset_db: None, reload_tester: ReloadTester) -> None:
        pxt.create_dir('dir1')
        schema = {'c1': pxt.String, 'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Timestamp}
        tbl = pxt.create_table('test', schema)
        _ = pxt.create_table('dir1.test', schema)

        with pytest.raises(excs.Error, match='Invalid path format'):
            pxt.create_table('1test', schema)
        with pytest.raises(excs.Error, match='Invalid path format'):
            pxt.create_table('bad name', {'c1': pxt.String})
        with pytest.raises(excs.Error, match='is an existing table'):
            pxt.create_table('test', schema)
        with pytest.raises(excs.Error, match='does not exist'):
            pxt.create_table('dir2.test2', schema)

        _ = pxt.list_tables()
        _ = pxt.list_tables('dir1')

        with pytest.raises(excs.Error, match='Invalid path format'):
            pxt.list_tables('1dir')
        with pytest.raises(excs.Error, match='does not exist'):
            pxt.list_tables('dir2')

        # test loading with new client
        _ = tbl.select().collect()
        _ = reload_tester.run_query(tbl.select())
        reload_tester.run_reload_test()

        tbl = pxt.get_table('test')
        assert isinstance(tbl, catalog.InsertableTable)
        tbl.add_column(c5=pxt.Int)
        tbl.drop_column('c1')
        tbl.rename_column('c2', 'c17')

        pxt.move('test', 'test2')

        pxt.drop_table('test2')
        pxt.drop_table('dir1.test')

        with pytest.raises(excs.Error, match="Path 'test' does not exist"):
            pxt.drop_table('test')
        with pytest.raises(excs.Error, match=r"Path 'dir1.test2' does not exist"):
            pxt.drop_table('dir1.test2')
        with pytest.raises(excs.Error, match='Invalid path format'):
            pxt.drop_table('.test2')

        with pytest.raises(excs.Error, match="'pos' is a reserved name in Pixeltable"):
            pxt.create_table('bad_col_name', {'pos': pxt.Int})

        with pytest.raises(excs.Error, match="'add_column' is a reserved name in Pixeltable"):
            pxt.create_table('test', {'add_column': pxt.Int})

        with pytest.raises(excs.Error, match="'insert' is a reserved name in Pixeltable"):
            pxt.create_table('test', {'insert': pxt.Int})

    def test_create_if_exists(self, reset_db: None, reload_tester: ReloadTester) -> None:
        """Test the if_exists parameter of create_table API"""
        schema = {'c1': pxt.String, 'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Timestamp}
        tbl = pxt.create_table('test', schema)
        tbl.insert(create_table_data(tbl, num_rows=5))
        id_before = tbl._id
        res_before = tbl.select().collect()
        assert len(res_before) == 5

        # invalid if_exists value is rejected
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', schema, if_exists='invalid')  # type: ignore[arg-type]
        assert (
            "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        )

        # scenario 1: a table exists at the path already
        with pytest.raises(excs.Error, match='is an existing'):
            pxt.create_table('test', schema)
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('test', schema)
        assert 'is an existing' in str(exc_info.value)
        assert len(tbl.select().collect()) == 5
        # if_exists='ignore' should return the existing table
        tbl2 = pxt.create_table('test', schema, if_exists='ignore')
        assert tbl2 == tbl
        assert tbl2._id == id_before
        res_after = tbl2.select().collect()
        assert_resultset_eq(res_before, res_after)
        # if_exists='replace' should drop the existing table
        tbl3 = pxt.create_table('test', schema, if_exists='replace')
        assert tbl3 != tbl
        assert tbl3._id != id_before
        res_after = tbl3.select().collect()
        assert len(res_after) == 0
        id_before = tbl3._id

        # sanity check persistence
        _ = reload_tester.run_query(tbl3.select())
        reload_tester.run_reload_test()

        tbl = pxt.get_table('test')
        assert tbl._id == id_before

        tbl.insert(create_table_data(tbl, num_rows=3))
        assert len(tbl.select().collect()) == 3
        view = pxt.create_view('test_view', tbl)
        assert len(view.select().collect()) == 3

        # scenario 2: a table exists at the path, but has dependency
        with pytest.raises(excs.Error, match='is an existing'):
            pxt.create_table('test', schema)
        assert len(tbl.select().collect()) == 3
        # if_exists='ignore' should return the existing table
        tbl2 = pxt.create_table('test', schema, if_exists='ignore')
        assert tbl2 == tbl
        assert tbl2._id == id_before
        assert len(tbl2.select().collect()) == 3
        # if_exists='replace' cannot drop a table with a dependent view.
        # it should raise an error and recommend using 'replace_force'
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_table('test', schema, if_exists='replace')
        err_msg = str(exc_info.value).lower()
        assert 'already exists' in err_msg and 'has dependents' in err_msg and 'replace_force' in err_msg
        # if_exists='replace_force' should drop the existing table
        # and its dependent view.
        tbl = pxt.create_table('test', schema, if_exists='replace_force')
        assert tbl._id != id_before
        id_before = tbl._id
        assert len(tbl.select().collect()) == 0
        assert 'test_view' not in pxt.list_tables()

        tbl.insert(create_table_data(tbl, num_rows=1))

        pxt.create_dir('dir1')
        # scenario 3: path exists but is not a table
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('dir1', schema)
        assert 'is an existing' in str(exc_info.value)
        assert len(tbl.select().collect()) == 1
        for _ie in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(excs.Error) as exc_info:
                pxt.create_table('dir1', schema, if_exists=_ie)  # type: ignore[arg-type]
            err_msg = str(exc_info.value).lower()
            assert 'already exists' in err_msg and 'is not a table' in err_msg
            assert len(tbl.select().collect()) == 1, f'with if_exists={_ie}'
            assert 'dir1' in pxt.list_dirs(), f'with if_exists={_ie}'

        # sanity check persistence
        _ = reload_tester.run_query(tbl.select())
        reload_tester.run_reload_test()

        tbl = pxt.get_table('test')
        assert tbl._id == id_before

    def test_columns(self, reset_db: None) -> None:
        schema = {'c1': pxt.String, 'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Timestamp}
        t = pxt.create_table('test', schema)
        assert t.columns() == ['c1', 'c2', 'c3', 'c4']

    def test_names(self, reset_db: None) -> None:
        pxt.create_dir('dir')
        pxt.create_dir('dir.subdir')
        for tbl_path, media_val in (('test', 'on_read'), ('dir.test', 'on_write'), ('dir.subdir.test', 'on_read')):
            tbl = pxt.create_table(tbl_path, {'col': pxt.String}, media_validation=media_val)  # type: ignore[arg-type]
            view_path = f'{tbl_path}_view'
            view = pxt.create_view(view_path, tbl, media_validation=media_val)  # type: ignore[arg-type]
            puresnap_path = f'{tbl_path}_puresnap'
            puresnap = pxt.create_snapshot(puresnap_path, tbl, media_validation=media_val)  # type: ignore[arg-type]
            snap_path = f'{tbl_path}_snap'
            snap = pxt.create_snapshot(
                snap_path,
                tbl,
                media_validation=media_val,  # type: ignore[arg-type]
                additional_columns={'col2': tbl.col + 'x'},
            )
            assert tbl._path() == tbl_path
            assert tbl._name == tbl_path.split('.')[-1]
            assert tbl._parent()._path() == '.'.join(tbl_path.split('.')[:-1])

            assert tbl.get_metadata() == {
                'base': None,
                'comment': '',
                'is_view': False,
                'is_snapshot': False,
                'is_replica': False,
                'name': 'test',
                'num_retained_versions': 10,
                'media_validation': media_val,
                'path': tbl_path,
                'schema': tbl._get_schema(),
                'schema_version': 0,
                'version': 0,
            }

            assert view.get_metadata() == {
                'base': tbl_path,
                'comment': '',
                'is_view': True,
                'is_snapshot': False,
                'is_replica': False,
                'name': 'test_view',
                'num_retained_versions': 10,
                'media_validation': media_val,
                'path': view_path,
                'schema': view._get_schema(),
                'schema_version': 0,
                'version': 0,
            }

            assert puresnap.get_metadata() == {
                'base': f'{tbl_path}:0',
                'comment': '',
                'is_view': True,
                'is_snapshot': True,
                'is_replica': False,
                'name': 'test_puresnap',
                'num_retained_versions': 10,
                'media_validation': media_val,
                'path': puresnap_path,
                'schema': puresnap._get_schema(),
                'schema_version': 0,
                'version': 0,
            }

            assert snap.get_metadata() == {
                'base': f'{tbl_path}:0',
                'comment': '',
                'is_view': True,
                'is_snapshot': True,
                'is_replica': False,
                'name': 'test_snap',
                'num_retained_versions': 10,
                'media_validation': media_val,
                'path': snap_path,
                'schema': snap._get_schema(),
                'schema_version': 0,
                'version': 0,
            }

    def test_media_validation(self, reset_db: None) -> None:
        tbl_schema = {'img': {'type': pxt.Image, 'media_validation': 'on_write'}, 'video': pxt.Video}
        t = pxt.create_table('test', tbl_schema, media_validation='on_read')
        assert t.get_metadata()['media_validation'] == 'on_read'
        assert t.img.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        # table default applies
        assert t.video.col.media_validation == pxt.catalog.MediaValidation.ON_READ

        v_schema = {'doc': {'type': pxt.Document, 'media_validation': 'on_read'}, 'audio': pxt.Audio}
        v = pxt.create_view('test_view', t, additional_columns=v_schema, media_validation='on_write')
        assert v.get_metadata()['media_validation'] == 'on_write'
        assert v.doc.col.media_validation == pxt.catalog.MediaValidation.ON_READ
        # view default applies
        assert v.audio.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        # flags for base still apply
        assert v.img.col.media_validation == pxt.catalog.MediaValidation.ON_WRITE
        assert v.video.col.media_validation == pxt.catalog.MediaValidation.ON_READ

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('validation_error', {'img': pxt.Image}, media_validation='wrong_value')  # type: ignore[arg-type]
        assert "media_validation must be one of: ['on_read', 'on_write']" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_table('validation_error', {'img': {'type': pxt.Image, 'media_validation': 'wrong_value'}})
        assert "media_validation must be one of: ['on_read', 'on_write']" in str(exc_info.value)

    def test_validate_on_read(self, reset_db: None, reload_tester: ReloadTester) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'id': i, 'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for i, f in enumerate(files)]
        schema = {'id': pxt.Int, 'media': pxt.Video, 'is_bad_media': pxt.Bool}

        on_read_tbl = pxt.create_table('read_validated', schema, media_validation='on_read')
        validate_update_status(on_read_tbl.insert(rows), len(rows))
        on_read_res = reload_tester.run_query(
            on_read_tbl.select(
                on_read_tbl.media,
                on_read_tbl.media.localpath,
                on_read_tbl.media.errortype,
                on_read_tbl.media.errormsg,
                on_read_tbl.media.cellmd,
                on_read_tbl.is_bad_media,
            ).order_by(on_read_tbl.id)
        )

        on_write_tbl = pxt.create_table('write_validated', schema, media_validation='on_write')
        status = on_write_tbl.insert(rows, on_error='ignore')
        assert status.num_excs == 2  # 1 row with exceptions in the media col and the index col
        on_write_res = reload_tester.run_query(
            on_write_tbl.select(
                on_write_tbl.media,
                on_write_tbl.media.localpath,
                on_write_tbl.media.errortype,
                on_write_tbl.media.errormsg,
                on_write_tbl.media.cellmd,
                on_write_tbl.is_bad_media,
            ).order_by(on_write_tbl.id)
        )
        assert_resultset_eq(on_read_res, on_write_res)

        reload_tester.run_reload_test()

    def test_validate_on_read_with_computed_col(self, reset_db: None) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for f in files]
        schema = {'media': pxt.Video, 'is_bad_media': pxt.Bool, 'stage': pxt.Required[pxt.Int]}

        # we are testing a nonsensical scenario: a computed column that references a read-validated media column,
        # which forces validation
        on_read_tbl = pxt.create_table('read_validated', schema, media_validation='on_read')
        on_read_tbl.add_computed_column(md=on_read_tbl.media.get_metadata())
        status = on_read_tbl.insert(({**r, 'stage': 0} for r in rows), on_error='ignore')
        assert status.num_excs == 1
        on_read_res_1 = (
            on_read_tbl.select(
                on_read_tbl.media,
                on_read_tbl.media.localpath,
                on_read_tbl.media.errortype,
                on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media,
                on_read_tbl.md,
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
            on_read_tbl.where(on_read_tbl.stage == 1)
            .select(
                on_read_tbl.media,
                on_read_tbl.media.localpath,
                on_read_tbl.media.errortype,
                on_read_tbl.media.errormsg,
                on_read_tbl.is_bad_media,
                on_read_tbl.md,
            )
            .order_by(on_read_tbl.media)
            .collect()
        )
        assert_resultset_eq(on_read_res_1, on_read_res_2)

    def test_create_from_df(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        df1 = t.where(t.c2 >= 50).order_by(t.c2, asc=False).select(t.c2, t.c3, t.c7, t.c2 + 26, t.c1.contains('19'))
        t1 = pxt.create_table('test1', source=df1)
        assert t1._get_schema() == df1.schema
        assert t1.collect() == df1.collect()

        from pixeltable.functions import sum

        t.add_computed_column(c2mod=t.c2 % 5)
        df2 = t.group_by(t.c2mod).select(t.c2mod, sum(t.c2))
        t2 = pxt.create_table('test2', source=df2)
        assert t2._get_schema() == df2.schema
        assert t2.collect() == df2.collect()

        with pytest.raises(excs.Error, match='must be a non-empty dictionary'):
            _ = pxt.create_table('test3', ['I am a string.'])  # type: ignore[arg-type]

    def test_insert_df(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        df1 = t.where(t.c2 >= 50).order_by(t.c2, asc=False).select(t.c2, t.c3, t.c7, t.c2 + 26, t.c1.contains('19'))
        t1 = pxt.create_table('test1', source=df1)
        assert t1._get_schema() == df1.schema
        assert t1.collect() == df1.collect()

        t1.insert(df1)
        assert len(t1.collect()) == 2 * len(df1.collect())

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
            'date_col': pxt.Date,
            'req_date_col': pxt.Required[pxt.Date],
            'json_col': pxt.Json,
            'req_json_col': pxt.Required[pxt.Json],
            'array_col': pxt.Array[(5, None, 3), pxt.Int],  # type: ignore[misc]
            'req_array_col': pxt.Required[pxt.Array[(5, None, 3), pxt.Int]],  # type: ignore[misc]
            'gen_array_col': pxt.Array[pxt.Float],  # type: ignore[misc]
            'req_gen_array_col': pxt.Required[pxt.Array[pxt.Float]],
            'full_gen_array_col': pxt.Array,
            'req_full_gen_array_col': pxt.Required[pxt.Array],
            'img_col': pxt.Image,
            'req_img_col': pxt.Required[pxt.Image],
            'spec_img_col': pxt.Image[(300, 300), 'RGB'],  # type: ignore[misc]
            'req_spec_img_col': pxt.Required[pxt.Image[(300, 300), 'RGB']],  # type: ignore[misc]
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
            'str_col': ts.StringType(nullable=True),
            'req_str_col': ts.StringType(nullable=False),
            'int_col': ts.IntType(nullable=True),
            'req_int_col': ts.IntType(nullable=False),
            'float_col': ts.FloatType(nullable=True),
            'req_float_col': ts.FloatType(nullable=False),
            'bool_col': ts.BoolType(nullable=True),
            'req_bool_col': ts.BoolType(nullable=False),
            'ts_col': ts.TimestampType(nullable=True),
            'req_ts_col': ts.TimestampType(nullable=False),
            'date_col': ts.DateType(nullable=True),
            'req_date_col': ts.DateType(nullable=False),
            'json_col': ts.JsonType(nullable=True),
            'req_json_col': ts.JsonType(nullable=False),
            'array_col': ts.ArrayType((5, None, 3), dtype=ts.IntType(), nullable=True),
            'req_array_col': ts.ArrayType((5, None, 3), dtype=ts.IntType(), nullable=False),
            'gen_array_col': ts.ArrayType(dtype=ts.FloatType(), nullable=True),
            'req_gen_array_col': ts.ArrayType(dtype=ts.FloatType(), nullable=False),
            'full_gen_array_col': ts.ArrayType(nullable=True),
            'req_full_gen_array_col': ts.ArrayType(nullable=False),
            'img_col': ts.ImageType(nullable=True),
            'req_img_col': ts.ImageType(nullable=False),
            'spec_img_col': ts.ImageType(width=300, height=300, mode='RGB', nullable=True),
            'req_spec_img_col': ts.ImageType(width=300, height=300, mode='RGB', nullable=False),
            'video_col': ts.VideoType(nullable=True),
            'req_video_col': ts.VideoType(nullable=False),
            'audio_col': ts.AudioType(nullable=True),
            'req_audio_col': ts.AudioType(nullable=False),
            'doc_col': ts.DocumentType(nullable=True),
            'req_doc_col': ts.DocumentType(nullable=False),
        }
        expected_schema.update({f'added_{col_name}': col_type for col_name, col_type in expected_schema.items()})

        assert t._get_schema() == expected_schema

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
            'Date',
            'Required[Date]',
            'Json',
            'Required[Json]',
            'Array[(5, None, 3), Int]',
            'Required[Array[(5, None, 3), Int]]',
            'Array[Float]',
            'Required[Array[Float]]',
            'Array',
            'Required[Array]',
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
        with pytest.raises(excs.Error, match='must be a non-empty dictionary'):
            pxt.create_table('empty_table', {})

    def test_drop_table(self, test_tbl: pxt.Table) -> None:
        pxt.drop_table('test_tbl')
        with pytest.raises(excs.Error, match='does not exist'):
            _ = pxt.get_table('test_tbl')
        # TODO: deal with concurrent drop_table() in another process
        # with pytest.raises(excs.Error, match='has been dropped') as exc_info:
        #     _ = t.show(1)

    def test_drop_table_via_handle(self, test_tbl: pxt.Table) -> None:
        t = pxt.create_table('test1', {'c1': pxt.String})
        pxt.drop_table(t)
        with pytest.raises(excs.Error, match='does not exist'):
            _ = pxt.get_table('test1')
        # with pytest.raises(excs.Error) as exc_info:
        #     _ = t.show(1)
        # assert 'table test1 has been dropped' in str(exc_info.value).lower()
        t = pxt.create_table('test2', {'c1': pxt.String})
        t = pxt.get_table('test2')
        pxt.drop_table(t)
        with pytest.raises(excs.Error, match='does not exist'):
            _ = pxt.get_table('test2')
        # with pytest.raises(excs.Error) as exc_info:
        #     _ = t.show(1)
        # assert 'table test2 has been dropped' in str(exc_info.value).lower()
        t = pxt.create_table('test3', {'c1': pxt.String})
        v = pxt.create_view('view3', t)
        pxt.drop_table(v)
        with pytest.raises(excs.Error, match='does not exist'):
            _ = pxt.get_table('view3')
        # with pytest.raises(excs.Error) as exc_info:
        #     _ = v.show(1)
        # assert 'view view3 has been dropped' in str(exc_info.value).lower()
        _ = pxt.get_table('test3')
        v = pxt.create_view('view4', t)
        v = pxt.get_table('view4')
        pxt.drop_table(v)
        with pytest.raises(excs.Error, match='does not exist'):
            _ = pxt.get_table('view4')
        # with pytest.raises(excs.Error) as exc_info:
        #     _ = v.show(1)
        # assert 'view view4 has been dropped' in str(exc_info.value).lower()
        _ = pxt.get_table('test3')
        pxt.drop_table(t)

    def test_drop_table_force(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', t)
        _v3 = pxt.create_view('v3', v1)
        _v4 = pxt.create_view('v4', v2)
        _v5 = pxt.create_view('v5', t)
        assert len(pxt.list_tables()) == 6
        pxt.drop_table('v2', force=True)  # Drops v2 and v4, but not the others
        assert len(pxt.list_tables()) == 4
        pxt.drop_table('test_tbl', force=True)  # Drops everything else
        assert len(pxt.list_tables()) == 0

    def test_drop_table_force_via_handle(self, test_tbl: pxt.Table) -> None:
        t = pxt.get_table('test_tbl')
        v1 = pxt.create_view('v1', t)
        v2 = pxt.create_view('v2', t)
        _v3 = pxt.create_view('v3', v1)
        _v4 = pxt.create_view('v4', v2)
        _v5 = pxt.create_view('v5', t)
        assert len(pxt.list_tables()) == 6
        pxt.drop_table(v2, force=True)  # Drops v2 and v4, but not the others
        assert len(pxt.list_tables()) == 4
        assert 'v2' not in pxt.list_tables()
        assert 'v4' not in pxt.list_tables()
        pxt.drop_table(t, force=True)  # Drops everything else
        assert len(pxt.list_tables()) == 0

    def test_drop_table_if_not_exists(self, reset_db: None) -> None:
        """Test the if_not_exists parameter of drop_table API"""
        non_existing_t = 'non_existing_table'
        table_list = pxt.list_tables()
        assert non_existing_t not in table_list
        # invalid if_not_exists value is rejected
        with pytest.raises(excs.Error) as exc_info:
            pxt.drop_table(non_existing_t, if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

        # if_not_exists='error' should raise an error if the table exists
        with pytest.raises(excs.Error, match='does not exist'):
            pxt.drop_table(non_existing_t, if_not_exists='error')
        # default behavior is to raise an error if the table does not exist
        with pytest.raises(excs.Error, match='does not exist'):
            pxt.drop_table(non_existing_t)
        # if_not_exists='ignore' should not raise an error
        pxt.drop_table(non_existing_t, if_not_exists='ignore')
        # force=True should not raise an error, irrespective of if_not_exists value
        pxt.drop_table(non_existing_t, force=True)
        assert table_list == pxt.list_tables()

    def test_image_table(self, reset_db: None) -> None:
        n_sample_rows = 20
        schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String, 'img_literal': pxt.Image}
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
        tbl.add_computed_column(rotated=tbl.img.rotate(30), stored=True)
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
            pxt.create_table('test', {'c1': pxt.Required[pxt.String]}, primary_key=0)  # type: ignore[arg-type]
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
            assert f'Standard Python type `{name}` cannot be used here; use `{suggestion}` instead' in str(
                exc_info.value
            )

    def check_bad_media(self, rows: list[dict[str, Any]], col_type: type, validate_local_path: bool = True) -> None:
        schema = {'media': col_type, 'is_bad_media': pxt.Bool}
        tbl = pxt.create_table('test', schema)

        assert len(rows) > 0
        total_bad_rows = sum(int(row['is_bad_media']) for row in rows)
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

    def test_validate_json(self, reset_db: None) -> None:
        json_schema = {
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
                'c': {'type': 'number'},
                'd': {'type': 'boolean'},
            },
            'required': ['a', 'b'],
        }

        t = pxt.create_table(
            'test',
            {
                'json_col': pxt.Json[json_schema]  # type: ignore[misc]
            },
        )
        t.insert(json_col={'a': 'coconuts', 'b': 1, 'c': 3.0, 'd': True})
        t.update({'json_col': {'a': 'mangoes', 'b': 2}})  # Omit optional properties

        with pytest.raises(ValidationError) as exc_info:
            t.insert(json_col={'a': 'apples', 'b': 'elephant'})  # Wrong type
        assert "'elephant' is not of type 'integer'" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            t.insert(json_col={'a': 'apples'})  # Missing required field
        assert "'b' is a required property" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            t.update({'json_col': {'a': 'apples'}})  # Validation error on update
        assert 'is not compatible with the type of column json_col' in str(exc_info.value)

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

    def test_file_paths(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test', {'img': pxt.Image})

        # File path contains unusual characters such as '#'
        path = TESTS_DIR / 'data/images/#_strange_file name!@$.jpg'
        validate_update_status(t.insert(img=str(path)), 1)
        # Run a query that selects both the image and its path, to ensure it's loadable
        res = reload_tester.run_query(t.select(t.img, path=t.img.localpath))
        assert res[0]['path'] == str(path)

        # File path is a relative path
        validate_update_status(t.insert(img='tests/data/images/#_strange_file name!@$.jpg'), 1)
        res = reload_tester.run_query(t.select(t.img, path=t.img.localpath))
        assert res[1]['path'] == str(Path('tests/data/images/#_strange_file name!@$.jpg').absolute())

    def test_create_s3_image_table(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        tbl = pxt.create_table('test', {'img': pxt.Image})
        # this is needed because reload_db() doesn't call TableVersion.drop(), which would
        # clear the file cache
        # TODO: change reset_catalog() to drop tables
        FileCache.get().clear()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 0, f'{cache_stats} tbl_id={tbl._id}'
        # add computed column to make sure that external files are cached locally during insert
        tbl.add_computed_column(rotated=tbl.img.rotate(30), stored=True)
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
        assert cache_stats.num_requests == len(urls), f'{cache_stats} tbl_id={tbl._id}'
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

    def test_image_formats(self, reset_db: None) -> None:
        tbl = pxt.create_table('test', {'img': pxt.Image})
        files = [
            'sewing-threads.heic'  # HEIC format
        ]
        tbl.insert({'img': f'{TESTS_DIR}/data/images/{file}'} for file in files)

    def test_video_url(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        schema = {'payload': pxt.Int, 'video': pxt.Video}
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
        view = pxt.create_view('test_view', tbl, iterator=FrameIterator.create(video=tbl.video, fps=0))
        view.add_computed_column(c1=view.frame.rotate(30), stored=True)
        view.add_computed_column(c2=view.c1.rotate(40), stored=False)
        view.add_computed_column(c3=view.c2.rotate(50), stored=True)
        # a non-materialized column that refers to another non-materialized column
        view.add_computed_column(c4=view.c2.rotate(60), stored=False)

        # cols computed with window functions are stored by default
        view.add_computed_column(c5=self.window_fn(view.frame_idx, 1, group_by=view.video))

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

        with pytest.raises(excs.Error, match=r'because the following columns depend on it:\nc1'):
            view.drop_column('frame')
        with pytest.raises(excs.Error, match=r'because the following columns depend on it:\nc5'):
            view.drop_column('frame_idx')

        # drop() clears stored images and the cache
        tbl.insert(payload=1, video=get_video_files()[0])
        with pytest.raises(excs.Error, match='has dependents'):
            pxt.drop_table('test_tbl')
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
        # FileCache.get().clear()  # make sure we need to download the files
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
            'c5': pxt.Array[(2, 3), pxt.Int],  # type: ignore[misc]
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
            'c5': pxt.Required[pxt.Array[(2, 3), pxt.Int]],  # type: ignore[misc]
            'c6': pxt.Required[pxt.Json],
            'c7': pxt.Required[pxt.Image],
            'c8': pxt.Required[pxt.Video],
        }
        tbl_name = 'test1'
        t = pxt.create_table(tbl_name, schema)
        rows = create_table_data(t)
        status = t.insert(rows)
        assert t.count() == len(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        # alternate (**kwargs: Any) insert syntax
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
        assert 'Unsupported data source type' in str(exc_info.value)

        # missing column
        with pytest.raises(excs.Error) as exc_info:
            # drop first column
            col_names = list(rows[0].keys())[1:]
            new_rows = [{col_name: row[col_name] for col_name in col_names} for row in rows]
            t.insert(new_rows)
        assert 'Missing' in str(exc_info.value)

        # incompatible schema
        for (col_name, col_type), value_col_name in zip(
            schema.items(), ['c2', 'c3', 'c5', 'c5', 'c6', 'c5', 'c2', 'c2']
        ):
            pxt.drop_table(tbl_name, if_not_exists='ignore')
            t = pxt.create_table(tbl_name, {col_name: col_type})
            with pytest.raises(excs.Error, match=r'expected|not a valid Pixeltable JSON object') as exc_info:
                t.insert({col_name: r[value_col_name]} for r in rows)

        # rows not list of dicts
        pxt.drop_table(tbl_name, if_not_exists='ignore')
        t = pxt.create_table(tbl_name, {'c1': pxt.String})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(['1'])  # xtype: ignore[list-item]
        assert 'Unsupported data source type' in str(exc_info.value)

        # bad null value
        pxt.drop_table(tbl_name, if_not_exists='ignore')
        t = pxt.create_table(tbl_name, {'c1': pxt.Required[pxt.String]})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(c1=None)
        assert 'expected non-None' in str(exc_info.value)

        # bad array literal
        pxt.drop_table(tbl_name, if_not_exists='ignore')
        t = pxt.create_table(tbl_name, {'c5': pxt.Array[(2, 3), pxt.Int]})  # type: ignore[misc]
        with pytest.raises(excs.Error, match=r'expected numpy.ndarray\(\(2, 3\)'):
            t.insert(c5=np.ndarray((3, 2)))

        # bad array literal
        pxt.drop_table(tbl_name, if_not_exists='ignore')
        t = pxt.create_table(tbl_name, {'c5': pxt.Array[pxt.Int]})  # type: ignore[misc]
        with pytest.raises(excs.Error, match=r'expected numpy.ndarray of dtype int64'):
            t.insert(c5=np.ndarray((3, 2), dtype=np.float32))

        # bad array literal
        pxt.drop_table(tbl_name, if_not_exists='ignore')
        t = pxt.create_table(tbl_name, {'c5': pxt.Array})
        with pytest.raises(excs.Error, match=r'expected numpy.ndarray, got'):
            t.insert(c5=8)
        with pytest.raises(excs.Error, match='unsupported dtype'):
            t.insert(c5=np.ndarray((3, 2), dtype=np.complex128))  # unsupported dtype

        # test that insert skips expression evaluation for
        # any columns that are not part of the current schema.
        @pxt.udf(_force_stored=True)
        def bad_udf(x: str) -> str:
            raise AssertionError()

        t = pxt.create_table('test', {'str_col': pxt.String})
        t.add_computed_column(bad=bad_udf(t.str_col))  # Succeeds because the table has no data
        t.drop_column('bad')
        t.insert(str_col='Hello there.')  # Succeeds because column 'bad' is dropped
        pxt.drop_table('test')

    def test_insert_string_with_null(self, reset_db: None) -> None:
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
            expected_rows=1,
        )
        assert t.count() == num_rows  # make sure we didn't lose any rows
        assert t.where(t.c2 == 1).collect()[0]['c1'] == 'eins'
        assert t.where(t.c2 == 200).count() == 0

        # unknown primary key: insert
        validate_update_status(
            t.batch_update([{'c1': 'zwei', 'c2': 2}, {'c1': 'zweihundert', 'c2': 200}], if_not_exists='insert'),
            expected_rows=2,
        )
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
        t.add_computed_column(computed1=t.c3 + 1)
        t.add_computed_column(computed2=t.computed1 + 1)
        t.add_computed_column(computed3=t.c3 + 3)

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
        assert set(status.updated_cols) == {
            'test_tbl.c3',
            'test_tbl.computed1',
            'test_tbl.computed2',
            'test_tbl.computed3',
        }
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        assert np.all(t.order_by(t.computed1).collect().to_pandas()['computed1'][:10] == pd.Series([1.0] * 10))
        assert np.all(t.order_by(t.computed2).collect().to_pandas()['computed2'][:10] == pd.Series([2.0] * 10))
        assert np.all(t.order_by(t.computed3).collect().to_pandas()['computed3'][:10] == pd.Series([3.0] * 10))

        # bad update spec
        with pytest.raises(excs.Error) as excinfo:
            t.update({1: 1})  # type: ignore[dict-item]
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
            t.update({'c3': 1.0}, where=lambda c2: c2 == 10)  # type: ignore[arg-type]
        assert 'predicate' in str(excinfo.value)

        img_t = small_img_tbl

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_cascading_update(self, test_tbl: pxt.InsertableTable) -> None:
        t = test_tbl
        t.add_computed_column(d1=t.c3 - 1)
        # add column that can be updated
        t.add_column(c10=pxt.Float)
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_computed_column(d2=t.c3 - t.c10)
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
            t.delete(where=lambda c2: c2 == 10)  # type: ignore[arg-type]
        assert 'predicate' in str(excinfo.value)

        img_t = small_img_tbl

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.delete(where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_computed_cols(self, reset_db: None) -> None:
        schema = {'c1': pxt.Int, 'c2': pxt.Float, 'c3': pxt.Json}
        t: pxt.Table = pxt.create_table('test', schema)
        status = t.add_computed_column(c4=t.c1 + 1)
        assert status.num_excs == 0
        status = t.add_computed_column(c5=t.c4 + 1)
        assert status.num_excs == 0
        status = t.add_computed_column(c6=t.c1 / t.c2)
        assert status.num_excs == 0
        status = t.add_computed_column(c7=t.c6 * t.c2)
        assert status.num_excs == 0
        status = t.add_computed_column(c8=t.c3.detections['*'].bounding_box)
        assert status.num_excs == 0
        status = t.add_computed_column(c9=t.c2.apply(math.sqrt, col_type=pxt.Float))
        assert status.num_excs == 0

        # unstored cols that compute window functions aren't currently supported
        with pytest.raises(excs.Error):
            t.add_computed_column(c10=pxtf.sum(t.c1, group_by=t.c1), stored=False)

        # # Column.dependent_cols are computed correctly
        # assert len(t.c1.col.dependent_cols) == 3
        # assert len(t.c2.col.dependent_cols) == 4
        # assert len(t.c3.col.dependent_cols) == 1
        # assert len(t.c4.col.dependent_cols) == 2
        # assert len(t.c5.col.dependent_cols) == 1
        # assert len(t.c6.col.dependent_cols) == 2
        # assert len(t.c7.col.dependent_cols) == 1
        # assert len(t.c8.col.dependent_cols) == 0

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
        assert len(t2_columns) == len(t2.columns())
        t_columns = t._tbl_version_path.columns()
        assert len(t_columns) == len(t2_columns)
        for i in range(len(t_columns)):
            if t_columns[i].value_expr is not None:
                assert t_columns[i].value_expr.equals(t2_columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        status = t.insert(rows)
        assert status.num_excs == 0
        _ = t.collect()
        _ = t.collect().to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(excs.Error):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    def test_unstored_computed_cols(self, reset_db: None) -> None:
        schema = {'c1': pxt.Int, 'c2': pxt.Float}
        t = pxt.create_table('test', schema)

        status = t.add_computed_column(c3=add_unstored_table_base_val(t.c1), stored=True)
        assert status.num_excs == 0
        status = t.add_computed_column(c4=add_unstored_table_base_val(t.c1), stored=False)
        assert status.num_excs == 0

        rows = create_table_data(t, ['c1', 'c2'], num_rows=10)
        global test_unstored_table_base_val  # noqa: PLW0603
        test_unstored_table_base_val = 1000
        t.insert(rows)
        _ = t.show()

        test_unstored_table_base_val = 2000
        t_res = t.select(t.c3, t.c4).collect()
        print(t_res)
        for row in t_res:
            assert row['c3'] + 1000 == row['c4']

    def test_expr_udf_computed_cols(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int})
        rows = [{'c1': i} for i in range(100)]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        status = t.add_computed_column(c2=t.c1 + 1)
        assert status.num_excs == 0
        # call with positional arg
        status = t.add_computed_column(c3=self.add1(t.c1))
        assert status.num_excs == 0
        # call with keyword arg
        status = t.add_computed_column(c4=self.add1(a=t.c1))
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
        status = t.add_computed_column(add1=self.f2(self.f1(t.c2)))
        assert status.num_excs == 0
        status = t.insert(rows, on_error='ignore')
        assert status.num_excs >= 10
        assert 'test_insert.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10

        # exception during add_computed_column()
        t = pxt.create_table('test_add_column', schema)
        status = t.insert(rows)
        assert status.num_rows == 100
        assert status.num_excs == 0

        with pytest.raises(excs.Error) as exc:
            t.add_computed_column(add1=self.f2(self.f1(t.c2)))
        assert 'division by zero' in str(exc.value)

        # on_error='abort' is the default
        with pytest.raises(excs.Error) as exc:
            t.add_computed_column(add1=self.f2(self.f1(t.c2)), on_error='abort')
        assert 'division by zero' in str(exc.value)

        # on_error='ignore' stores the exception in errortype/errormsg
        status = t.add_computed_column(add1=self.f2(self.f1(t.c2)), on_error='ignore')
        assert status.num_excs == 10
        assert 'test_add_column.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10
        msgs = t.select(msg=t.add1.errormsg).collect()['msg']
        assert sum('division by zero' in msg for msg in msgs if msg is not None) == 10

    @pytest.mark.skip('Crashes pytest')
    def test_computed_col_with_interrupts(self, reset_db: None) -> None:
        schema = {'c1': pxt.Int}
        t = pxt.create_table('test_interrupt', schema)
        t.insert(({'c1': i} for i in range(0, 1000)))
        with pytest.raises(KeyboardInterrupt):
            _ = t.add_computed_column(cc1=self.function_with_error(t.c1, 245), on_error='ignore')
        results = t.head(1)
        assert results[0] == {'c1': 0}
        results = t.tail(1)
        assert results[0] == {'c1': 999}
        assert len(results.schema) == 1
        assert results.schema.get('cc1') is None

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
        assert len(t.columns()) == len(t2.columns())
        t_columns = t._tbl_version_path.columns()
        t2_columns = t2._tbl_version_path.columns()
        assert len(t_columns) == len(t2_columns)
        for i in range(len(t_columns)):
            if t_columns[i].value_expr is not None:
                assert t_columns[i].value_expr.equals(t2_columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert(rows)
        assert MediaStore.count(t2._id) == t2.count() * stores_img_col
        _ = t2.collect()
        _ = t2.collect().to_pandas()

        # revert also removes computed images
        t2.revert()
        assert MediaStore.count(t2._id) == t2.count() * stores_img_col

    @staticmethod
    @pxt.udf
    def img_fn_with_exc(img: PIL.Image.Image) -> PIL.Image.Image:
        raise RuntimeError

    def test_computed_img_cols(self, reset_db: None) -> None:
        schema = {'img': pxt.Image}
        t = pxt.create_table('test', schema)
        t.add_computed_column(c2=t.img.width)
        # c3 is not stored by default
        t.add_computed_column(c3=t.img.rotate(90), stored=False)
        self._test_computed_img_cols(t, stores_img_col=False)

        t = pxt.create_table('test2', schema)
        # c3 is now stored
        t.add_computed_column(c3=t.img.rotate(90))
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t.select(t.c3).collect()
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t.select(t.c3.errortype).collect()

        # computed img col with exceptions
        t = pxt.create_table('test3', schema)
        t.add_computed_column(c3=self.img_fn_with_exc(t.img))
        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        rows = [{'img': r['img']} for r in rows[:20]]
        t.insert(rows, on_error='ignore')
        _ = t.select(t.c3.errortype).collect()

    def test_computed_window_fn(self, reset_db: None, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # backfill
        t.add_computed_column(c9=pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3))

        schema = {'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Bool}
        new_t = pxt.create_table('insert_test', schema)
        new_t.add_computed_column(c5=t.c2.apply(lambda x: x * x, col_type=pxt.Int))
        new_t.add_computed_column(c6=pxtf.sum(new_t.c5, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.collect()

    def test_revert(self, reset_db: None) -> None:
        t1 = make_tbl('test1', ['c1', 'c2'])
        assert t1._get_version() == 0
        rows1 = create_table_data(t1)
        t1.insert(rows1)
        assert t1.count() == len(rows1)
        assert t1._get_version() == 1
        rows2 = create_table_data(t1)
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1._get_version() == 2
        t1.revert()
        assert t1.count() == len(rows1)
        assert t1._get_version() == 1
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1._get_version() == 2

        # can't revert past version 0
        t1.revert()
        t1.revert()
        with pytest.raises(excs.Error) as excinfo:
            t1.revert()
        assert 'version 0' in str(excinfo.value)

    def test_add_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.add_column(add1=pxt.Int)
        # Make sure that `name` and `id` are allowed, i.e., not reserved as system names
        t.add_column(name=pxt.String)
        t.add_column(id=pxt.String)
        assert len(t.columns()) == num_orig_cols + 3

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
            _ = t.add_computed_column(c5=(t.c2 + t.c3), stored=False)

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns()) == num_orig_cols + 3

        # revert() works
        t.revert()
        t.revert()
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns()) == num_orig_cols

    def test_bool_column(self, reset_db: None, reload_tester: ReloadTester) -> None:
        # test adding a bool column with constant value
        t1 = pxt.create_table('test1', {'c1': pxt.Int})
        t1.insert([{'c1': 1}, {'c1': 2}])
        assert t1.count() == 2
        t1.add_computed_column(bool_const=False)
        assert t1.where(~t1.bool_const).count() == 2
        res = t1.collect()
        assert res['bool_const'] == [False, False]
        t1.insert([{'c1': 3}, {'c1': 4}])
        assert t1.where(~t1.bool_const).count() == 4
        res = t1.collect()
        assert res['bool_const'] == [False, False, False, False]

        # test adding a bool column with constant value to a view
        t2 = pxt.create_table('test2', {'c1': pxt.Int})
        validate_update_status(t2.insert([{'c1': 1}, {'c1': 2}]), expected_rows=2)
        v = pxt.create_view('test_view', t2)
        assert v.count() == 2
        v.add_computed_column(bool_const=True)
        assert v.where(v.bool_const).count() == 2
        res = v.collect()
        assert res['bool_const'] == [True, True]
        t2.insert([{'c1': 3}, {'c1': 4}])
        assert v.where(v.bool_const).count() == 4
        res = v.collect()
        assert res['bool_const'] == [True, True, True, True]

        # test using the bool column in a conditional expression
        res = v.select((v.c1 > 1) & v.bool_const).collect()
        assert len(res) == 4
        assert res['col_0'] == [False, True, True, True]
        # reversing the condition order should not affect the result
        res = v.select(v.bool_const & (v.c1 > 1)).collect()
        assert len(res) == 4
        assert res['col_0'] == [False, True, True, True]

        # test adding a bool column with a computed value
        t1.add_computed_column(bool_computed=t1.c1 > 1)
        res = t1.collect()
        assert res['bool_computed'] == [False, True, True, True]
        res = t1.where(t1.bool_computed).collect()
        assert res['c1'] == [2, 3, 4]
        res = t1.where(~t1.bool_computed).collect()
        assert res['c1'] == [1]

        t3 = pxt.create_table('test3', {'c1': pxt.Int, 'c2': pxt.Bool})
        t3.insert([{'c1': 1, 'c2': True}, {'c1': 2, 'c2': False}])
        assert t3.count() == 2

        # bool columns accept int values that can be cast to bool.
        t3.insert(c2=3)
        res = t3.select(t3.c2).collect()
        assert res['c2'] == [True, False, True]
        t3.insert(c2=0)
        res = t3.select(t3.c2).collect()
        assert res['c2'] == [True, False, True, False]
        t3.insert(c2=-1)
        res = t3.select(t3.c2).collect()
        assert res['c2'] == [True, False, True, False, True]

        # bool columns do not accept other types.
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c2='T')
        assert 'error in column c2: expected bool, got str' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c2=4.5)
        assert 'error in column c2: expected bool, got float' in str(exc_info.value).lower()

        # test that int columns only accept int values, not bool
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c1=True)
        assert 'error in column c1: expected int, got bool' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c1=False)
        assert 'error in column c1: expected int, got bool' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c1='T')
        assert 'error in column c1: expected int, got str' in str(exc_info.value).lower()
        with pytest.raises(excs.Error) as exc_info:
            t3.insert(c1=4.5)
        assert 'error in column c1: expected int, got float' in str(exc_info.value).lower()

        # sanity test persistence
        _ = reload_tester.run_query(t1.select())
        _ = reload_tester.run_query(t2.select())
        _ = reload_tester.run_query(t3.select())
        _ = reload_tester.run_query(v.select())
        _ = reload_tester.run_query(v.select((v.c1 > 1) & v.bool_const))
        _ = reload_tester.run_query(v.select(v.bool_const & (v.c1 > 1)))
        _ = reload_tester.run_query(t1.where(t1.bool_computed).select())
        _ = reload_tester.run_query(t1.where(~t1.bool_computed).select())

        reload_tester.run_reload_test()

    def test_add_column_if_exists(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        """Test the if_exists parameter of add_column."""
        t = test_tbl
        orig_cnames = t.columns()
        orig_res = t.select(t.c1).order_by(t.c1).collect()

        # invalid if_exists is rejected
        expected_err_str = "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']"
        with pytest.raises(excs.Error, match=re.escape(expected_err_str)):
            t.add_column(non_existing_col1=pxt.Int, if_exists='invalid')
        with pytest.raises(excs.Error, match=re.escape(expected_err_str)):
            t.add_computed_column(non_existing_col1=t.c2 + t.c3, if_exists='invalid')
        with pytest.raises(excs.Error, match=re.escape(expected_err_str)):
            t.add_columns({'non_existing_col1': pxt.Int, 'non_existing_col2': pxt.String}, if_exists='invalid')  # type: ignore[arg-type]
        assert orig_cnames == t.columns()

        # if_exists='error' raises an error if the column already exists
        # by default, if_exists='error'
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_column(c1=pxt.Int)
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_computed_column(c1=t.c2 + t.c3)
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_columns({'c1': pxt.Int, 'non_existing_col1': pxt.String})
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_column(c1=pxt.Int, if_exists='error')
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_computed_column(c1=t.c2 + t.c3, if_exists='error')
        with pytest.raises(excs.Error, match="Duplicate column name: 'c1'"):
            t.add_columns({'c1': pxt.Int, 'non_existing_col1': pxt.String}, if_exists='error')
        assert orig_cnames == t.columns()
        assert_resultset_eq(t.select(t.c1).order_by(t.c1).collect(), orig_res, True)

        # if_exists='ignore' does nothing if the column already exists
        t.add_column(c1=pxt.Int, if_exists='ignore')
        assert orig_cnames == t.columns()
        assert_resultset_eq(t.select(t.c1).order_by(t.c1).collect(), orig_res, True)

        t.add_computed_column(c1=t.c2 + 1, if_exists='ignore')
        assert orig_cnames == t.columns()
        assert_resultset_eq(t.select(t.c1).order_by(t.c1).collect(), orig_res, True)

        t.add_columns({'c1': pxt.Int, 'non_existing_col1': pxt.String}, if_exists='ignore')
        assert 'c1' in t.columns()
        assert_resultset_eq(t.select(t.c1).order_by(t.c1).collect(), orig_res, True)
        assert 'non_existing_col1' in t.columns()

        # if_exists='replace' replaces the column if it has no dependents
        t.add_columns({'c1': pxt.Int, 'non_existing_col2': pxt.String}, if_exists='replace')
        assert 'c1' in t.columns()
        assert t.select(t.c1).collect()[0] == {'c1': None}
        assert 'non_existing_col2' in t.columns()
        before_cnames = t.columns()

        t.add_computed_column(c1=10, if_exists='replace')
        assert set(t.columns()) == set(before_cnames)
        assert 'c1' in t.columns()
        assert t.select(t.c1).collect()[0] != orig_res[0]
        assert t.select(t.c1).collect()[0] == {'c1': 10}

        # revert restores the state back wrt the underlying persistence.
        # so it will revert the add_column operation and drop the column
        # and not restore the original column it replaced.
        t.revert()
        assert 'c1' not in t.columns()

        t.add_computed_column(c1=10)
        assert set(t.columns()) == set(before_cnames)
        assert 'c1' in t.columns()
        assert t.select(t.c1).collect()[0] == {'c1': 10}

        t.add_computed_column(c1=t.c2 + t.c3, if_exists='replace')
        assert set(t.columns()) == set(before_cnames)
        assert 'c1' in t.columns()
        assert t.select(t.c1).collect()[0] != {'c1': 10}
        assert (
            t.select().order_by(t.c1).collect()[0]['c1']
            == t.select().order_by(t.c1).collect()[0]['c2'] + t.select().order_by(t.c1).collect()[0]['c3']
        )

        # replace will raise an error if the column has dependents
        t.add_computed_column(non_existing_col3=t.c1 + 10)
        with pytest.raises(excs.Error) as exc_info:
            t.add_column(c1=pxt.Int, if_exists='replace')
        error_msg = str(exc_info.value).lower()
        assert 'already exists' in error_msg and 'has dependents' in error_msg
        assert 'c1' in t.columns()
        assert t.select(t.c1).collect()[0] != {'c1': 10}
        assert (
            t.select().order_by(t.c1).collect()[0]['c1']
            == t.select().order_by(t.c1).collect()[0]['c2'] + t.select().order_by(t.c1).collect()[0]['c3']
        )

        _ = reload_tester.run_query(t.select(t.c1))

        reload_tester.run_reload_test()

    recompute_udf_increment = 0
    recompute_udf_error_val: Optional[int] = None

    @staticmethod
    @pxt.udf
    def recompute_int_udf(i: int) -> int:
        if TestTable.recompute_udf_error_val is not None and i % TestTable.recompute_udf_error_val == 0:
            raise RuntimeError(f'Error in recompute_udf for value {i}')
        return i + TestTable.recompute_udf_increment

    @staticmethod
    @pxt.udf
    def recompute_str_udf(s: str) -> str:
        i = int(s)
        if TestTable.recompute_udf_error_val is not None and i % TestTable.recompute_udf_error_val == 0:
            raise RuntimeError(f'Error in recompute_udf for value {i}')
        return str(i + TestTable.recompute_udf_increment)

    def test_recompute_column(self, reset_db: None) -> None:
        t = pxt.create_table('recompute_test', schema={'i': pxt.Int, 's': pxt.String})
        status = t.add_computed_column(i1=self.recompute_int_udf(t.i))
        assert status.num_excs == 0
        status = t.add_computed_column(s1=self.recompute_str_udf(t.s))
        assert status.num_excs == 0
        status = t.add_computed_column(i2=t.i1 * 2)
        assert status.num_excs == 0
        v = pxt.create_view('recompute_view', base=t.where(t.i < 20), additional_columns={'i3': t.i2 + 1})
        validate_update_status(t.insert({'i': i, 's': str(i)} for i in range(100)), expected_rows=100 + 20)

        # recompute without propagation
        TestTable.recompute_udf_increment = 1
        status = t.recompute_columns(t.i1, cascade=False)
        assert status.num_rows == 100
        assert set(status.updated_cols) == {'recompute_test.i1'}
        result = t.select(t.i1, t.i2).order_by(t.i).collect()
        assert result['i1'] == [i + 1 for i in range(100)]
        assert result['i2'] == [2 * i for i in range(100)]
        result = v.select(v.i3).order_by(v.i).collect()
        assert result['i3'] == [2 * i + 1 for i in range(20)]

        # recompute with propagation, via a ColumnRef
        TestTable.recompute_udf_increment = 1
        status = t.i1.recompute()
        assert status.num_rows == 100 + 20
        assert set(status.updated_cols) == {'recompute_test.i1', 'recompute_test.i2', 'recompute_view.i3'}
        result = t.select(t.i1, t.i2).order_by(t.i).collect()
        assert result['i1'] == [i + 1 for i in range(100)]
        assert result['i2'] == [2 * (i + 1) for i in range(100)]
        result = v.select(v.i3).order_by(v.i).collect()
        assert result['i3'] == [2 * (i + 1) + 1 for i in range(20)]

        # recompute multiple columns
        status = t.recompute_columns('i1', t.s1)
        assert status.num_rows == 100 + 20
        assert set(status.updated_cols) == {
            'recompute_test.i1',
            'recompute_test.i2',
            'recompute_test.s1',
            'recompute_view.i3',
        }
        result = t.select(t.i1, t.i2).order_by(t.i).collect()
        assert result['i1'] == [i + 1 for i in range(100)]
        assert result['i2'] == [2 * (i + 1) for i in range(100)]
        result = v.select(v.i3).order_by(v.i).collect()
        assert result['i3'] == [2 * (i + 1) + 1 for i in range(20)]

        # add some errors
        TestTable.recompute_udf_increment = 0
        TestTable.recompute_udf_error_val = 10
        status = t.recompute_columns('i1')
        assert status.num_rows == 100 + 20
        assert status.num_excs == 4 * 10  # c1 and c2 plus their index value cols
        assert set(status.updated_cols) == {'recompute_test.i1', 'recompute_test.i2', 'recompute_view.i3'}
        _ = t.select(t.i2.errormsg).where(t.i2.errormsg != None).collect()
        assert t.where(t.i1.errortype != None).count() == 10
        assert t.where(t.i1.errormsg.startswith('Error in recompute_udf') != None).count() == 10
        assert t.where(t.i2.errortype != None).count() == 10
        assert t.where(t.i2.errormsg.startswith('Error in recompute_udf') != None).count() == 10
        assert t.where(t.i1.errortype == None).count() == 90
        assert t.where(t.i2.errortype == None).count() == 90

        # recompute errors
        TestTable.recompute_udf_error_val = None
        status = t.recompute_columns(t.i1, errors_only=True)
        assert status.num_rows == 10 + 2
        assert status.num_excs == 0
        assert set(status.updated_cols) == {'recompute_test.i1', 'recompute_test.i2', 'recompute_view.i3'}

        with pytest.raises(excs.Error, match='Unknown column'):
            t.recompute_columns('h')
        with pytest.raises(excs.Error, match='is not a computed column'):
            t.recompute_columns(t.i)
        with pytest.raises(excs.Error, match='of a snapshot'):
            s = pxt.create_snapshot('recompute_snap', t, additional_columns={'i4': t.i2 + 1})
            s.recompute_columns(s.i4)
        with pytest.raises(excs.Error, match='At least one column must be specified'):
            t.recompute_columns()
        with pytest.raises(excs.Error, match='Cannot use errors_only=True with multiple columns'):
            t.recompute_columns(t.i1, t.s1, errors_only=True)
        with pytest.raises(excs.Error, match='Cannot recompute column of a base'):
            v.recompute_columns(v.i1)
        with pytest.raises(excs.Error, match='Cannot recompute column of a base'):
            v.i1.recompute()

    def __test_drop_column_if_not_exists(self, t: catalog.Table, non_existing_col: Union[str, ColumnRef]) -> None:
        """Test the if_not_exists parameter of drop_column API"""
        # invalid if_not_exists parameter is rejected
        with pytest.raises(excs.Error) as exc_info:
            t.drop_column(non_existing_col, if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

        # if_not_exists='error' raises an error if the column does not exist
        with pytest.raises(excs.Error) as exc_info:
            t.drop_column(non_existing_col, if_not_exists='error')
        err_msg = str(exc_info.value).lower()
        if isinstance(non_existing_col, str):
            assert f"column '{non_existing_col}' unknown" in err_msg
        else:
            assert f'unknown column: {non_existing_col.col.qualified_name}' in err_msg
        # if_not_exists='ignore' does nothing if the column does not exist
        t.drop_column(non_existing_col, if_not_exists='ignore')

    def test_drop_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        dummy_t = pxt.create_table('dummy', {'dummy_col': pxt.Int})
        num_orig_cols = len(t.columns())
        t.drop_column('c1')
        assert len(t.columns()) == num_orig_cols - 1
        assert 'c1' not in t.columns()
        with pytest.raises(AttributeError, match="Column 'c1' unknown"):
            _ = t.c1
        with pytest.raises(excs.Error, match="Column 'c1' unknown"):
            t.drop_column('c1')
        # non-existing column by name - column was already dropped
        self.__test_drop_column_if_not_exists(t, 'c1')
        # non-existing column by reference - valid column reference
        # but of a different table
        self.__test_drop_column_if_not_exists(t, dummy_t.dummy_col)
        assert 'unknown' not in t.columns()
        with pytest.raises(excs.Error, match="Column 'unknown' unknown"):
            t.drop_column('unknown')
        with pytest.raises(AttributeError, match="Column 'unknown' unknown"):
            t.drop_column(t.unknown)
        # non-existing column by name - column was never created
        self.__test_drop_column_if_not_exists(t, 'unknown')

        # make sure this is still true after reloading the metadata
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns()) == num_orig_cols - 1

        # revert() works
        t.revert()
        assert len(t.columns()) == num_orig_cols
        assert 'c1' in t.columns()
        _ = t.c1

        # make sure this is still true after reloading the metadata once more
        reload_catalog()
        t = pxt.get_table(t._name)
        assert len(t.columns()) == num_orig_cols
        assert 'c1' in t.columns()
        _ = t.c1

        # Test drop_column for a view
        t.drop_column('c1')
        assert len(t.columns()) == num_orig_cols - 1
        assert 'c1' not in t.columns()
        v = pxt.create_view('v', base=t, additional_columns={'v1': t.c3 + 1})
        assert 'c1' not in v.columns()
        assert 'v1' in v.columns()
        # non-existing column by name - column was already dropped, base table column
        self.__test_drop_column_if_not_exists(v, 'c1')
        v.drop_column('v1')
        assert 'v1' not in v.columns()
        # non-existing column by name - column was already dropped, view column
        self.__test_drop_column_if_not_exists(v, 'v1')
        # non-existing column by name - column was never created
        self.__test_drop_column_if_not_exists(t, 'unknown')
        # non-existing column by reference - valid column reference of a different table
        self.__test_drop_column_if_not_exists(v, dummy_t.dummy_col)

        # drop_column is not allowed on a snapshot
        s1 = pxt.create_snapshot('s1', t, additional_columns={'s1': t.c3 + 1})
        assert 'c1' not in s1.columns()
        with pytest.raises(excs.Error, match='Cannot drop column from a snapshot'):
            s1.drop_column('c1')
        assert 's1' in s1.columns()
        with pytest.raises(excs.Error, match='Cannot drop column from a snapshot'):
            s1.drop_column('s1')
        assert 's1' in s1.columns()

    def test_drop_column_via_reference(self, reset_db: None) -> None:
        t1 = pxt.create_table('test1', {'c1': pxt.String, 'c2': pxt.String})
        t1.insert([{'c1': 'a1', 'c2': 'b1'}, {'c1': 'a2', 'c2': 'b2'}])
        t2 = pxt.create_table('test2', {'c1': pxt.String, 'c2': pxt.String})

        # cannot pass another table's column reference
        with pytest.raises(excs.Error) as exc_info:
            t1.drop_column(t2.c2)
        assert 'unknown column: test2.c2' in str(exc_info.value).lower()
        assert 'c2' in t1.columns()
        assert 'c2' in t2.columns()
        _ = t1.c2
        _ = t2.c2

        t1.drop_column(t1.c2)
        assert 'c2' not in t1.columns()
        with pytest.raises(AttributeError, match="Column 'c2' unknown") as exc_info:
            _ = t1.c2
        with pytest.raises(AttributeError, match="Column 'c2' unknown") as exc_info:
            t1.drop_column(t1.c2)
        assert 'c2' in t2.columns()
        pxt.drop_table(t1)
        pxt.drop_table(t2)

    def test_rename_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.rename_column('c1', 'c1_renamed')
        assert len(t.columns()) == num_orig_cols

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

    def test_add_computed_column(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t = test_tbl
        status = t.add_computed_column(add1=t.c2 + 10)
        assert status.num_excs == 0
        _ = t.show()

        # TODO(aaron-siegel): This has to be commented out. See explanation in test_exprs.py.
        # with pytest.raises(excs.Error):
        #     t.add_computed_column(add2=(t.c2 - 10) / (t.c3 - 10))

        # with exception in Python for c6.f2 == 10
        status = t.add_computed_column(add2=(t.c6.f2 - 10) / (t.c6.f2 - 10), on_error='ignore')
        assert status.num_excs == 1
        result = t.where(t.add2.errortype != None).select(t.c6.f2, t.add2, t.add2.errortype, t.add2.errormsg).show()
        assert len(result) == 1

        # test case: exceptions in dependencies prevent execution of dependent exprs
        status = t.add_computed_column(add3=self.f2(self.f1(t.c2)), on_error='ignore')
        assert status.num_excs == 10
        result = t.where(t.add3.errortype != None).select(t.c2, t.add3, t.add3.errortype, t.add3.errormsg).show()
        assert len(result) == 10

        # test case: add computed column on a view that refers to a base table column
        v = pxt.create_view('test_view', t)
        v.add_computed_column(add4=v.c2 + 10)
        v.add_computed_column(add5=t.c2 + 10)
        _ = v.show()

        # sanity check persistence
        _ = reload_tester.run_query(t.select())
        _ = reload_tester.run_query(v.select())

        reload_tester.run_reload_test()

    def test_computed_column_types(self, reset_db: None) -> None:
        t = pxt.create_table(
            'test', {'c1': pxt.Int, 'c1_r': pxt.Required[pxt.Int], 'c2': pxt.String, 'c2_r': pxt.Required[pxt.String]}
        )

        # Ensure that arithmetic and (non-nullable) function call expressions inherit nullability from their arguments
        t.add_computed_column(arith=t.c1 + 1)
        t.add_computed_column(arith_r=t.c1_r + 1)
        t.add_computed_column(func=t.c2.upper())
        t.add_computed_column(func_r=t.c2_r.upper())

        assert t.get_metadata()['schema'] == {
            'c1': ts.IntType(nullable=True),
            'c1_r': ts.IntType(nullable=False),
            'c2': ts.StringType(nullable=True),
            'c2_r': ts.StringType(nullable=False),
            'arith': ts.IntType(nullable=True),
            'arith_r': ts.IntType(nullable=False),
            'func': ts.StringType(nullable=True),
            'func_r': ts.StringType(nullable=False),
        }

    def test_repr(self, test_tbl: catalog.Table, all_mpnet_embed: func.Function) -> None:
        skip_test_if_not_installed('sentence_transformers')

        v = pxt.create_view('test_view', test_tbl)
        v2 = pxt.create_view('test_subview', v.where(v.c1 != None), comment='This is an intriguing table comment.')

        v2.add_computed_column(computed1=v2.c2.apply(lambda x: np.full((3, 4), x), col_type=pxt.Array[(3, 4), pxt.Int]))  # type: ignore[misc]
        v2.add_embedding_index('c1', string_embed=all_mpnet_embed)
        v2._link_external_store(MockProject.create(v2, 'project', {}, {}))
        v2.describe()

        # test case: view with additional columns
        r = repr(v2)
        assert strip_lines(r) == strip_lines(
            """View 'test_subview' (of 'test_view', 'test_tbl')
            Where: ~(c1 == None)

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
                  idx0     c1  cosine  sentence_transformer(sentence, normalize_embed...

            External Store         Type
                   project  MockProject

            COMMENT: This is an intriguing table comment."""
        )
        _ = v2._repr_html_()  # TODO: Is there a good way to test this output?

        # test case: snapshot of view
        s1 = pxt.create_snapshot('test_snap1', v2)
        r = repr(s1)
        assert strip_lines(r) == strip_lines(
            """Snapshot 'test_snap1' (of 'test_subview:3', 'test_view:0', 'test_tbl:2')
            Where: ~(c1 == None)

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

            External Store         Type
                   project  MockProject

            COMMENT: This is an intriguing table comment."""
        )

        # test case: snapshot of base table
        s2 = pxt.create_snapshot('test_snap2', test_tbl)
        r = repr(s2)
        assert strip_lines(r) == strip_lines(
            """Snapshot 'test_snap2' (of 'test_tbl:2')

            Column Name                          Type           Computed With
                     c1              Required[String]
                    c1n                        String
                     c2                 Required[Int]
                     c3               Required[Float]
                     c4                Required[Bool]
                     c5           Required[Timestamp]
                     c6                Required[Json]
                     c7                Required[Json]
                     c8  Required[Array[(2, 3), Int]]  [[1, 2, 3], [4, 5, 6]]"""
        )

        # test case: snapshot with additional columns
        s3 = pxt.create_snapshot('test_snap3', test_tbl, additional_columns={'computed1': test_tbl.c2 + test_tbl.c3})
        r = repr(s3)
        assert strip_lines(r) == strip_lines(
            """View 'test_snap3' (of 'test_tbl:2')

            Column Name                          Type           Computed With
              computed1               Required[Float]                 c2 + c3
                     c1              Required[String]
                    c1n                        String
                     c2                 Required[Int]
                     c3               Required[Float]
                     c4                Required[Bool]
                     c5           Required[Timestamp]
                     c6                Required[Json]
                     c7                Required[Json]
                     c8  Required[Array[(2, 3), Int]]  [[1, 2, 3], [4, 5, 6]]"""
        )

        c = repr(v2.c1)
        assert strip_lines(c) == strip_lines(
            """Column
            'c1'
            (of table 'test_tbl')

            Column Name              Type Computed With
                     c1  Required[String]"""
        )
        _ = v2.c1._repr_html_()

    def test_common_col_names(self, reset_db: None) -> None:
        """Make sure that commonly used column names don't collide with Table member vars"""
        names = ['id', 'name', 'version', 'comment']
        schema = dict.fromkeys(names, pxt.Int)
        tbl = pxt.create_table('test', schema)
        status = tbl.insert(dict.fromkeys(names, id) for id in range(10))
        assert status.num_rows == 10
        assert status.num_excs == 0
        assert tbl.count() == 10
        # we can create references to those column via __getattr__
        _ = tbl.select(tbl.id, tbl._name).collect()

    def test_table_api_on_dropped_table(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int, 'c2': pxt.String})
        pxt.drop_table('test')
        unknown_tbl_msg = 'Table was dropped'

        # verify that queries and data changes fail with unkown_tbl_msg
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.delete(t.c1 > 3)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.insert([{'c1': 1, 'c2': 'abc'}])
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.update({'c1': 2})
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.batch_update([{'c1': 2, 'c2': 'f'}])

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.add_column(c2=pxt.Int)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.add_columns({'c2': pxt.Int})
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.add_computed_column(c3=t.c1 + 10)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.add_embedding_index('c2', string_embed=str.split)  # type: ignore[arg-type]
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.drop_embedding_index(column='c2', if_not_exists='ignore')
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.drop_index(column='c2')
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.drop_column('c1')
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.rename_column('c1', 'c1_renamed')

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.group_by(t.c1)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.select(t.c1)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.where(t.c1 > 3)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.order_by(t.c1)

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.collect()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.count()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.head()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.limit(1)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.tail()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.show()

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.describe()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.get_metadata()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.list_views()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = repr(t)
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t._repr_html_()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.external_stores()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.unlink_external_stores()

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.sync()

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.to_coco_dataset()
        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            _ = t.to_pytorch_dataset()

        with pytest.raises(excs.Error, match=unknown_tbl_msg):
            t.revert()

    def test_array_columns(self, reset_db: None, reload_tester: ReloadTester) -> None:
        schema = {
            'fixed_shape': pxt.Array[(3, None, 5), pxt.Int],  # type: ignore[misc]
            'gen_shape': pxt.Array[pxt.Float],  # type: ignore[misc]
            'gen': pxt.Array,
        }
        t = pxt.create_table('array_tbl', schema)
        rows = [
            {
                'fixed_shape': np.ones((3, 2, 5), dtype=np.int64),
                'gen_shape': np.ones((1, 2, 3, 4), dtype=np.float32),
                'gen': np.array(['a', 'b', 'c']),
            },
            {
                'fixed_shape': np.zeros((3, 7, 5), dtype=np.int64),
                'gen_shape': np.zeros((2, 6), dtype=np.float32),
                'gen': np.array([[1, 7, 3], [2, 4, 5]], dtype=np.int64),
            },
        ]
        t.insert(rows)
        results = reload_tester.run_query(t.select())
        for row, result in zip(rows, results):
            for key in row:
                a1 = row[key]
                a2 = result[key]
                assert isinstance(a1, np.ndarray)
                assert isinstance(a2, np.ndarray)
                assert np.array_equal(a1, a2)

        reload_tester.run_reload_test()
