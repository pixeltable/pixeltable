import datetime
import math
import os
import random
from typing import List, Tuple

import PIL
import cv2
import numpy as np
import pandas as pd
import pathlib
import pytest

import pixeltable as pxt
import pixeltable.functions as ptf
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable.iterators import FrameIterator
from pixeltable.tests.utils import \
    make_tbl, create_table_data, read_data_file, get_video_files, get_audio_files, get_image_files, get_documents, \
    assert_resultset_eq, assert_hf_dataset_equal, make_test_arrow_table, validate_update_status
from pixeltable.tests.utils import skip_test_if_not_installed
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType, AudioType, \
    DocumentType
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

class TestTable:
    # exc for a % 10 == 0
    @pxt.udf(return_type=FloatType(), param_types=[IntType()])
    def f1(a: int) -> float:
        return a / (a % 10)

    # exception for a == None; this should not get triggered
    @pxt.udf(return_type=FloatType(), param_types=[FloatType()])
    def f2(a: float) -> float:
        return a + 1

    @pxt.expr_udf(param_types=[IntType(nullable=False)])
    def add1(a: int) -> int:
        return a + 1

    @pxt.uda(
        update_types=[IntType()], value_type=IntType(), requires_order_by=True,
        allows_window=True)
    class window_fn:
        def __init__(self):
            pass
        def update(self, i: int) -> None:
            pass
        def value(self) -> int:
            return 1

    @pxt.expr_udf(param_types=[IntType(nullable=False)])
    def add1(a: int) -> int:
        return a + 1

    def test_create(self, test_client: pxt.Client) -> None:
        cl = test_client
        cl.create_dir('dir1')
        schema = {
            'c1': StringType(nullable=False),
            'c2': IntType(nullable=False),
            'c3': FloatType(nullable=False),
            'c4': TimestampType(nullable=False),
        }
        tbl = cl.create_table('test', schema)
        _ = cl.create_table('dir1.test', schema)

        with pytest.raises(excs.Error):
            _ = cl.create_table('1test', schema)
        with pytest.raises(excs.Error):
            _ = cl.create_table('bad name', schema={'c1': StringType()})
        with pytest.raises(excs.Error):
            _ = cl.create_table('test', schema)
        with pytest.raises(excs.Error):
            _ = cl.create_table('dir2.test2', schema)

        _ = cl.list_tables()
        _ = cl.list_tables('dir1')

        with pytest.raises(excs.Error):
            _ = cl.list_tables('1dir')
        with pytest.raises(excs.Error):
            _ = cl.list_tables('dir2')

        # test loading with new client
        cl = pxt.Client(reload=True)

        tbl = cl.get_table('test')
        assert isinstance(tbl, catalog.InsertableTable)
        tbl.add_column(c5=IntType())
        tbl.drop_column('c1')
        tbl.rename_column('c2', 'c17')

        cl.move('test', 'test2')

        cl.drop_table('test2')
        cl.drop_table('dir1.test')

        with pytest.raises(excs.Error):
            cl.drop_table('test')
        with pytest.raises(excs.Error):
            cl.drop_table('dir1.test2')
        with pytest.raises(excs.Error):
            cl.drop_table('.test2')

    def test_empty_table(self, test_client: pxt.Client) -> None:
        cl = test_client
        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('empty_table', {})
        assert 'Table schema is empty' in str(exc_info.value)

    def test_table_attrs(self, test_client: pxt.Client) -> None:
        cl = test_client
        schema = {'c': StringType(nullable=False)}
        num_retained_versions = 20
        comment = "This is a table."
        tbl = cl.create_table('test_table_attrs', schema, num_retained_versions=num_retained_versions, comment=comment)
        assert tbl.num_retained_versions == num_retained_versions
        assert tbl.comment == comment
        new_num_retained_versions = 30
        new_comment = "This is an updated table."
        tbl.num_retained_versions = new_num_retained_versions
        assert tbl.num_retained_versions == new_num_retained_versions
        tbl.comment = new_comment
        assert tbl.comment == new_comment
        tbl.revert()
        assert tbl.comment == comment
        tbl.revert()
        assert tbl.num_retained_versions == num_retained_versions

    def test_import_parquet(self, test_client: pxt.Client, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pixeltable.utils.arrow import iter_tuples

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        make_test_arrow_table(parquet_dir)

        tab = test_client.import_parquet('test_parquet', parquet_path=str(parquet_dir))
        assert 'test_parquet' in test_client.list_tables()
        assert tab is not None
        num_elts = tab.count()
        arrow_tab: pa.Table = pa.parquet.read_table(str(parquet_dir))
        assert num_elts == arrow_tab.num_rows
        assert set(tab.column_names()) == set(arrow_tab.column_names)

        result_set = tab.order_by(tab.c_id).collect()
        column_types = tab.column_types()

        for tup, arrow_tup in zip(result_set, iter_tuples(arrow_tab)):
            assert tup['c_id'] == arrow_tup['c_id']
            for col, val in tup.items():
                if val is None:
                    assert arrow_tup[col] is None
                    continue

                if column_types[col].is_array_type():
                    assert (val == arrow_tup[col]).all()
                else:
                    assert val == arrow_tup[col]

    def test_import_huggingface_dataset(self, test_client: pxt.Client, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        test_cases = [
            # { # includes a timestamp. 20MB for specific slice
            # Disbled this test case because download is failing, and its not critical.
            #     'dataset_name': 'c4',
            #     # see https://huggingface.co/datasets/allenai/c4/blob/main/realnewslike/c4-train.00000-of-00512.json.gz
            #     'dataset': datasets.load_dataset(
            #         "allenai/c4",
            #         data_dir="realnewslike",
            #         data_files="c4-train.00000-of-00512.json.gz",
            #         split='train[:1000]',
            #         cache_dir=tmp_path
            #     ),
            # },
            {  # includes an embedding (array type), common in a few RAG datasets.
                'dataset_name': 'cohere_wikipedia',
                'dataset': datasets.load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3",
                                                 data_dir='cr').select_columns(['url', 'title', 'text', 'emb']),
                # column with name `_id`` is not currently allowed by pixeltable rules,
                # so filter out that column.
                # cr subdir has a small number of rows, avoid running out of space in CI runner
                # see https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/tree/main/cr
                'schema_override': {'emb': ArrayType((1024,), dtype=FloatType(), nullable=False)}
            },
            # example of dataset dictionary with multiple splits
            {
                'dataset_name': 'rotten_tomatoes',
                'dataset': datasets.load_dataset("rotten_tomatoes"),
            },
        ]

        # test a column name for splits other than the default of 'split'
        split_column_name = 'my_split_col'
        for rec in test_cases:
            dataset_name = rec['dataset_name']
            hf_dataset = rec['dataset']

            tab = test_client.import_huggingface_dataset(
                dataset_name,
                hf_dataset,
                column_name_for_split=split_column_name,
                schema_override=rec.get('schema_override', None),
            )
            if isinstance(hf_dataset, datasets.Dataset):
                assert_hf_dataset_equal(hf_dataset, tab.df(), split_column_name)
            elif isinstance(hf_dataset, datasets.DatasetDict):
                assert tab.count() == sum(hf_dataset.num_rows.values())
                assert split_column_name in tab.column_names()

                for dataset_name in hf_dataset:
                    df = tab.where(tab.my_split_col == dataset_name)
                    assert_hf_dataset_equal(hf_dataset[dataset_name], df, split_column_name)
            else:
                assert False

        with pytest.raises(excs.Error) as exc_info:
            test_client.import_huggingface_dataset('test', {})
        assert 'type(dataset)' in str(exc_info.value)

    def test_image_table(self, test_client: pxt.Client) -> None:
        n_sample_rows = 20
        cl = test_client
        schema = {
            'img': ImageType(nullable=False),
            'category': StringType(nullable=False),
            'split': StringType(nullable=False),
            'img_literal': ImageType(nullable=False),
        }
        tbl = cl.create_table('test', schema)
        assert(MediaStore.count(tbl.get_id()) == 0)

        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        sample_rows = random.sample(rows, n_sample_rows)

        # add literal image data and column
        for r in rows:
            with open(r['img'], 'rb') as f:
                r['img_literal'] = f.read()

        tbl.insert(sample_rows)
        assert(MediaStore.count(tbl.get_id()) == n_sample_rows)

        # compare img and img_literal
        # TODO: make tbl.select(tbl.img == tbl.img_literal) work
        tdf = tbl.select(tbl.img, tbl.img_literal).show()
        pdf = tdf.to_pandas()
        for tup in pdf.itertuples():
            assert tup.img == tup.img_literal

        # Test adding stored image transformation
        tbl.add_column(rotated=tbl.img.rotate(30), stored=True)
        assert(MediaStore.count(tbl.get_id()) == 2 * n_sample_rows)

        # Test MediaStore.stats()
        stats = list(filter(lambda x: x[0] == tbl.get_id(), MediaStore.stats()))
        assert len(stats) == 2                 # Two columns
        assert stats[0][2] == n_sample_rows    # Each column has n_sample_rows associated images
        assert stats[1][2] == n_sample_rows

        # Test that version-specific images are cleared when table is reverted
        tbl.revert()
        assert(MediaStore.count(tbl.get_id()) == n_sample_rows)

        # Test that all stored images are cleared when table is dropped
        cl.drop_table('test')
        assert(MediaStore.count(tbl.get_id()) == 0)

    def test_schema_spec(self, test_client: pxt.Client) -> None:
        cl = test_client

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c 1': IntType()})
        assert 'invalid column name' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {}})
        assert '"type" is required' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'xyz': IntType()}})
        assert "invalid key 'xyz'" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'stored': True}})
        assert '"type" is required' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'type': 'string'}})
        assert 'must be a ColumnType' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'value': 1, 'type': StringType()}})
        assert '"type" is redundant' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'value': pytest}})
        assert 'value needs to be either' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            def f() -> float:
                return 1.0
            cl.create_table('test', {'c1': {'value': f}})
        assert '"type" is required' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': {'type': StringType(), 'stored': 'true'}})
        assert '"stored" must be a bool' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': StringType()}, primary_key='c2')
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': StringType()}, primary_key=['c1', 'c2'])
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': StringType()}, primary_key=['c2'])
        assert 'primary key column c2 not found' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': StringType()}, primary_key=0)
        assert 'primary_key must be a' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            cl.create_table('test', {'c1': StringType(nullable=True)}, primary_key='c1')
        assert 'cannot be nullable' in str(exc_info.value).lower()

    def check_bad_media(
            self, test_client: pxt.Client, rows: List[Tuple[str, bool]], col_type: pxt.ColumnType,
            validate_local_path: bool = True
    ) -> None:
        schema = {
            'media': col_type,
            'is_bad_media': BoolType(nullable=False),
        }
        tbl = test_client.create_table('test', schema)

        assert len(rows) > 0
        total_bad_rows = sum([int(row['is_bad_media']) for row in rows])
        assert total_bad_rows > 0

        # Mode 1: Validation error on bad input (default)
        # we ignore the exact error here, because it depends on the media type
        with pytest.raises(excs.Error):
            tbl.insert(rows, fail_on_exception=True)

        # Mode 2: ignore_errors=True, store error information in table
        status = tbl.insert(rows, fail_on_exception=False)
        _ = tbl.select(tbl.media, tbl.media.errormsg).show()
        assert status.num_rows == len(rows)
        assert status.num_excs == total_bad_rows

        # check that we have the right number of bad and good rows
        assert tbl.where(tbl.is_bad_media == True).count() == total_bad_rows
        assert tbl.where(tbl.is_bad_media == False).count() == len(rows) - total_bad_rows

        # check error type is set correctly
        assert tbl.where((tbl.is_bad_media == True) & (tbl.media.errortype == None)).count() == 0
        assert tbl.where((tbl.is_bad_media == False) & (tbl.media.errortype == None)).count() \
            == len(rows) - total_bad_rows

        # check fileurl is set for valid images, and check no file url is set for bad images
        assert tbl.where((tbl.is_bad_media == False) & (tbl.media.fileurl == None)).count() == 0
        assert tbl.where((tbl.is_bad_media == True) & (tbl.media.fileurl != None)).count() == 0

        if validate_local_path:
            # check that tbl.media is a valid local path
            paths = tbl.where(tbl.media != None).select(output=tbl.media).collect()['output']
            for path in paths:
                assert os.path.exists(path) and os.path.isfile(path)

    def test_validate_image(self, test_client: pxt.Client) -> None:
        rows = read_data_file('imagenette2-160', 'manifest_bad.csv', ['img'])
        rows = [{'media': r['img'], 'is_bad_media': r['is_bad_image']} for r in rows]
        self.check_bad_media(test_client, rows, ImageType(nullable=True), validate_local_path=False)

    def test_validate_video(self, test_client: pxt.Client) -> None:
        files = get_video_files(include_bad_video=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_video.mp4')} for f in files]
        self.check_bad_media(test_client, rows, VideoType(nullable=True))

    def test_validate_audio(self, test_client: pxt.Client) -> None:
        files = get_audio_files(include_bad_audio=True)
        rows = [{'media': f, 'is_bad_media': f.endswith('bad_audio.mp3')} for f in files]
        self.check_bad_media(test_client, rows, AudioType(nullable=True))

    def test_validate_docs(self, test_client: pxt.Client) -> None:
        valid_doc_paths = get_documents()
        invalid_doc_paths = [get_video_files()[0], get_audio_files()[0], get_image_files()[0]]
        doc_paths = valid_doc_paths + invalid_doc_paths
        is_valid = [True] * len(valid_doc_paths) + [False] * len(invalid_doc_paths)
        rows = [{'media': f, 'is_bad_media': not is_valid} for f, is_valid in zip(doc_paths, is_valid)]
        self.check_bad_media(test_client, rows, DocumentType(nullable=True))

    def test_validate_external_url(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('boto3')
        rows = [
            {'media': 's3://open-images-dataset/validation/doesnotexist.jpg', 'is_bad_media': True},
            {'media': 'https://archive.random.org/download?file=2024-01-28.bin', 'is_bad_media': True},  # 403 error
            {'media': 's3://open-images-dataset/validation/3c02ca9ec9b2b77b.jpg', 'is_bad_media': True},  # wrong media
            # test s3 url
            {
                'media': 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4',
                'is_bad_media': False
            },
            # test http url
            {
                'media': 'https://github.com/pixeltable/pixeltable/raw/master/pixeltable/tests/data/videos/bangkok.mp4',
                'is_bad_media': False
            },

        ]
        self.check_bad_media(test_client, rows, VideoType(nullable=True))

    def test_create_s3_image_table(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('boto3')
        cl = test_client
        tbl = cl.create_table('test', {'img': ImageType(nullable=False)})
        # this is needed because Client.reset_catalog() doesn't call TableVersion.drop(), which would
        # clear the file cache
        # TODO: change reset_catalog() to drop tables
        FileCache.get().clear()
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 0, f'{str(cache_stats)} tbl_id={tbl.get_id()}'
        # add computed column to make sure that external files are cached locally during insert
        tbl.add_column(rotated=tbl.img.rotate(30), stored=True)
        urls = [
            's3://open-images-dataset/validation/3c02ca9ec9b2b77b.jpg',
            's3://open-images-dataset/validation/3c13e0015b6c3bcf.jpg',
            's3://open-images-dataset/validation/3ba5380490084697.jpg',
            's3://open-images-dataset/validation/3afeb4b34f90c0cf.jpg',
            's3://open-images-dataset/validation/3b07a2c0d5c0c789.jpg',
        ]

        tbl.insert({'img': url} for url in urls)
        # check that we populated the cache
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls), f'{str(cache_stats)} tbl_id={tbl.get_id()}'
        assert cache_stats.num_hits == 0
        assert FileCache.get().num_files() == len(urls)
        assert FileCache.get().num_files(tbl.get_id()) == len(urls)
        assert FileCache.get().avg_file_size() > 0

        # query: we read from the cache
        _ = tbl.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 2 * len(urls)
        assert cache_stats.num_hits == len(urls)

        # after clearing the cache, we need to re-fetch the files
        FileCache.get().clear()
        _ = tbl.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls)
        assert cache_stats.num_hits == 0

        # start with fresh client and FileCache instance to test FileCache initialization with pre-existing files
        cl = pxt.Client(reload=True)
        # is there a better way to do this?
        FileCache._instance = None
        t = cl.get_table('test')
        _ = t.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == len(urls)
        assert cache_stats.num_hits == len(urls)

        # dropping the table also clears the file cache
        cl.drop_table('test')
        cache_stats = FileCache.get().stats()
        assert cache_stats.total_size == 0

    def test_video_url(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('boto3')
        cl = test_client
        schema = {
            'payload': IntType(nullable=False),
            'video': VideoType(nullable=False),
        }
        tbl = cl.create_table('test', schema)
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        tbl.insert(payload=1, video=url)
        row = tbl.select(tbl.video.fileurl, tbl.video.localpath).collect()[0]
        assert row['video_fileurl'] == url
        # row[1] contains valid path to an mp4 file
        local_path = row['video_localpath']
        assert os.path.exists(local_path) and os.path.isfile(local_path)
        cap = cv2.VideoCapture(local_path)
        # TODO: this isn't sufficient to determine that this is actually a video, rather than an image
        assert cap.isOpened()
        cap.release()

    def test_create_video_table(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('boto3')
        cl = test_client
        tbl = cl.create_table(
            'test_tbl',
            {'payload': IntType(nullable=False), 'video': VideoType(nullable=True)})
        args = {'video': tbl.video, 'fps': 0}
        view = cl.create_view('test_view', tbl, iterator_class=FrameIterator, iterator_args=args)
        view.add_column(c1=view.frame.rotate(30), stored=True)
        view.add_column(c2=view.c1.rotate(40), stored=False)
        view.add_column(c3=view.c2.rotate(50), stored=True)
        # a non-materialized column that refers to another non-materialized column
        view.add_column(c4=view.c2.rotate(60), stored=False)

        # cols computed with window functions are stored by default
        view.add_column(c5=self.window_fn(view.frame_idx, 1, group_by=view.video))

        # reload to make sure that metadata gets restored correctly
        cl = pxt.Client(reload=True)
        tbl = cl.get_table('test_tbl')
        view = cl.get_table('test_view')
        # we're inserting only a single row and the video column is not in position 0
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        status = tbl.insert(payload=1, video=url)
        assert status.num_excs == 0
        # * 2: we have 2 stored img cols
        assert MediaStore.count(view.get_id()) == view.count() * 2
        # also insert a local file
        tbl.insert(payload=1, video=get_video_files()[0])
        assert MediaStore.count(view.get_id()) == view.count() * 2

        # TODO: test inserting Nulls
        #status = tbl.insert(payload=1, video=None)
        #assert status.num_excs == 0

        # revert() clears stored images
        tbl.revert()
        tbl.revert()
        assert MediaStore.count(view.get_id()) == 0

        with pytest.raises(excs.Error):
            # can't drop frame col
            view.drop_column('frame')
        with pytest.raises(excs.Error):
            # can't drop frame_idx col
            view.drop_column('frame_idx')

        # drop() clears stored images and the cache
        tbl.insert(payload=1, video=get_video_files()[0])
        with pytest.raises(excs.Error) as exc_info:
            cl.drop_table('test_tbl')
        assert 'has dependents: test_view' in str(exc_info.value)
        cl.drop_table('test_view')
        cl.drop_table('test_tbl')
        assert MediaStore.count(view.get_id()) == 0

    def test_insert_nulls(self, test_client: pxt.Client) -> None:
        cl = test_client
        schema = {
            'c1': StringType(nullable=True),
            'c2': IntType(nullable=True),
            'c3': FloatType(nullable=True),
            'c4': BoolType(nullable=True),
            'c5': ArrayType((2, 3), dtype=IntType(), nullable=True),
            'c6': JsonType(nullable=True),
            'c7': ImageType(nullable=True),
            'c8': VideoType(nullable=True),
        }
        t = cl.create_table('test1', schema)
        status = t.insert(c1='abc')
        assert status.num_rows == 1
        assert status.num_excs == 0

    def test_insert(self, test_client: pxt.Client) -> None:
        cl = test_client
        schema = {
            'c1': StringType(nullable=False),
            'c2': IntType(nullable=False),
            'c3': FloatType(nullable=False),
            'c4': BoolType(nullable=False),
            'c5': ArrayType((2, 3), dtype=IntType(), nullable=False),
            'c6': JsonType(nullable=False),
            'c7': ImageType(nullable=False),
            'c8': VideoType(nullable=False),
        }
        t = cl.create_table('test1', schema)
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
            c8=get_video_files()[0]
        )
        assert status.num_rows == 1
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
        for (col_name, col_type), value_col_name in zip(schema.items(), ['c2', 'c3', 'c5', 'c5', 'c6', 'c7', 'c2', 'c2']):
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', {col_name: col_type})
            with pytest.raises(excs.Error) as exc_info:
                t.insert({col_name: r[value_col_name]} for r in rows)
            assert 'expected' in str(exc_info.value).lower()

        # rows not list of dicts
        cl.drop_table('test1', ignore_errors=True)
        t = cl.create_table('test1', {'c1': StringType()})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(['1'])
        assert 'list of dictionaries' in str(exc_info.value)

        # bad null value
        cl.drop_table('test1', ignore_errors=True)
        t = cl.create_table('test1', {'c1': StringType(nullable=False)})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(c1=None)
        assert 'expected non-None' in str(exc_info.value)

        # bad array literal
        cl.drop_table('test1', ignore_errors=True)
        t = cl.create_table('test1', {'c5': ArrayType((2, 3), dtype=IntType(), nullable=False)})
        with pytest.raises(excs.Error) as exc_info:
            t.insert(c5=np.ndarray((3, 2)))
        assert 'expected ndarray((2, 3)' in str(exc_info.value)

    def test_insert_string_with_null(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table('test', {'c1': StringType()})

        t.insert([{'c1': 'this is a python\x00string'}])
        assert t.count() == 1
        for tup in t.df().collect():
            assert tup['c1'] == 'this is a python string'

    def test_query(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('boto3')
        cl = test_client
        col_names = ['c1', 'c2', 'c3', 'c4', 'c5']
        t = make_tbl(cl, 'test', col_names)
        rows = create_table_data(t)
        t.insert(rows)
        _ = t.show(n=0)

        # test querying existing table
        cl = pxt.Client(reload=True)
        t2 = cl.get_table('test')
        _  = t2.show(n=0)

    def test_batch_update(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        validate_update_status(
            t.batch_update([{'c1': '1', 'c2': 1}, {'c1': '2', 'c2': 2}]),
            expected_rows=2)
        assert t.where(t.c2 == 1).collect()[0]['c1'] == '1'
        assert t.where(t.c2 == 2).collect()[0]['c1'] == '2'
        validate_update_status(
            t.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two', '_rowid': (2,)}]),
            expected_rows=2)
        assert t.where(t.c2 == 1).collect()[0]['c1'] == 'one'
        assert t.where(t.c2 == 2).collect()[0]['c1'] == 'two'

        cl = pxt.Client()
        # test composite primary key
        schema = {'c1': StringType(), 'c2': IntType(), 'c3': FloatType()}
        t = cl.create_table('composite', schema=schema, primary_key=['c1', 'c2'])
        rows = [{'c1': str(i), 'c2': i, 'c3': float(i)} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        validate_update_status(
            t.batch_update([{'c1': '1', 'c2': 1, 'c3': 2.0}, {'c1': '2', 'c2': 2, 'c3': 3.0}]),
            expected_rows=2)

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
        t2 = cl.create_table('no_pk', schema=schema)
        validate_update_status(t2.insert(rows), expected_rows=10)
        with pytest.raises(excs.Error) as exc_info:
            _ = t2.batch_update([{'c1': '1', 'c2': 1, 'c3': 2.0}])
        assert 'must have primary key for batch update' in str(exc_info.value).lower()

        # updating with _rowid still works
        validate_update_status(
            t2.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two', '_rowid': (2,)}]),
            expected_rows=2)
        assert t2.where(t2.c2 == 1).collect()[0]['c1'] == 'one'
        assert t2.where(t2.c2 == 2).collect()[0]['c1'] == 'two'
        with pytest.raises(AssertionError):
            # some rows are missing rowids
            _ = t2.batch_update([{'c1': 'one', '_rowid': (1,)}, {'c1': 'two'}])

    def test_update(self, test_tbl: pxt.Table, small_img_tbl) -> None:
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
            assert status.updated_cols == [f'{t.get_name()}.{col_name}']
            assert t.count() == count
            t.revert()

        # exchange two columns
        t.add_column(float_col=FloatType(nullable=True))
        t.update({'float_col': 1.0})
        float_col_vals = t.select(t.float_col).collect().to_pandas()['float_col']
        c3_vals = t.select(t.c3).collect().to_pandas()['c3']
        assert np.all(float_col_vals == pd.Series([1.0] * t.count()))
        t.update({'c3': t.float_col, 'float_col': t.c3})
        assert np.all(t.select(t.c3).collect().to_pandas()['c3'] == float_col_vals)
        assert np.all(t.select(t.float_col).collect().to_pandas()['float_col'] == c3_vals)
        t.revert()

        # update column that is used in computed cols
        t.add_column(computed1=t.c3 + 1)
        t.add_column(computed2=t.computed1 + 1)
        t.add_column(computed3=t.c3 + 3)

        # cascade=False
        computed1 = t.order_by(t.computed1).show(0).to_pandas()['computed1']
        computed2 = t.order_by(t.computed2).show(0).to_pandas()['computed2']
        computed3 = t.order_by(t.computed3).show(0).to_pandas()['computed3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1
        # update to a value that also satisfies the where clause
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=False)
        assert status.num_rows == 10
        assert status.updated_cols == ['test_tbl.c3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        # computed cols are not updated
        assert np.all(t.order_by(t.computed1).show(0).to_pandas()['computed1'] == computed1)
        assert np.all(t.order_by(t.computed2).show(0).to_pandas()['computed2'] == computed2)
        assert np.all(t.order_by(t.computed3).show(0).to_pandas()['computed3'] == computed3)

        # revert, then verify that we're back to where we started
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        t.revert()
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1

        # cascade=True
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=True)
        assert status.num_rows == 10
        assert set(status.updated_cols) == \
               set(['test_tbl.c3', 'test_tbl.computed1', 'test_tbl.computed2', 'test_tbl.computed3'])
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        assert np.all(t.order_by(t.computed1).show(0).to_pandas()['computed1'][:10] == pd.Series([1.0] * 10))
        assert np.all(t.order_by(t.computed2).show(0).to_pandas()['computed2'][:10] == pd.Series([2.0] * 10))
        assert np.all(t.order_by(t.computed3).show(0).to_pandas()['computed3'][:10] == pd.Series([3.0] * 10))

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
        assert 'Predicate' in str(excinfo.value)

        img_t = small_img_tbl

        # can't update image col
        with pytest.raises(excs.Error) as excinfo:
            img_t.update({'img': 17}, where=img_t.img.nearest('car'))
        assert 'has type image' in str(excinfo.value)

        # similarity search is not supported
        with pytest.raises(excs.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.nearest('car'))
        assert 'nearest()' in str(excinfo.value)

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_cascading_update(self, test_tbl: pxt.InsertableTable) -> None:
        t = test_tbl
        t.add_column(d1=t.c3 - 1)
        # add column that can be updated
        t.add_column(c10=FloatType(nullable=True))
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_column(d2=t.c3 - t.c10)
        r1 = t.where(t.c2 < 5).select(t.c3 + 1.0, t.c10 - 1.0, t.c3, 2.0).order_by(t.c2).show(0)
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        r2 = t.where(t.c2 < 5).select(t.c3, t.c10, t.d1, t.d2).order_by(t.c2).show(0)
        assert_resultset_eq(r1, r2)

    def test_delete(self, test_tbl: pxt.Table, small_img_tbl) -> None:
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
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        t.revert()
        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 10
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1

        # non-Predicate filter
        with pytest.raises(excs.Error) as excinfo:
            t.delete(where=lambda c2: c2 == 10)
        assert 'Predicate' in str(excinfo.value)

        img_t = small_img_tbl
        # similarity search is not supported
        with pytest.raises(excs.Error) as excinfo:
            img_t.delete(where=img_t.img.nearest('car'))
        assert 'nearest()' in str(excinfo.value)

        # filter not expressible in SQL
        with pytest.raises(excs.Error) as excinfo:
            img_t.delete(where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_computed_cols(self, test_client: pxt.client) -> None:
        cl = test_client
        schema = {
            'c1': IntType(nullable=False),
            'c2': FloatType(nullable=False),
            'c3': JsonType(nullable=False),
        }
        t : pxt.InsertableTable = cl.create_table('test', schema)
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
        status = t.add_column(c9=lambda c2: math.sqrt(c2), type=FloatType())
        assert status.num_excs == 0

        # unstored cols that compute window functions aren't currently supported
        with pytest.raises((excs.Error)):
            t.add_column(c10=ptf.sum(t.c1, group_by=t.c1), stored=False)

        # Column.dependent_cols are computed correctly
        assert len(t.c1.col.dependent_cols) == 2
        assert len(t.c2.col.dependent_cols) == 3
        assert len(t.c3.col.dependent_cols) == 1
        assert len(t.c4.col.dependent_cols) == 1
        assert len(t.c5.col.dependent_cols) == 0
        assert len(t.c6.col.dependent_cols) == 1
        assert len(t.c7.col.dependent_cols) == 0
        assert len(t.c8.col.dependent_cols) == 0

        rows = create_table_data(t, ['c1', 'c2', 'c3'], num_rows=10)
        t.insert(rows)
        _ = t.show()

        # not allowed to pass values for computed cols
        with pytest.raises(excs.Error):
            rows2 = create_table_data(t, ['c1', 'c2', 'c3', 'c4'], num_rows=10)
            t.insert(rows2)

        # test loading from store
        cl = pxt.Client(reload=True)
        t = cl.get_table('test')
        assert len(t.columns()) == len(t.columns())
        for i in range(len(t.columns())):
            if t.columns()[i].value_expr is not None:
                assert t.columns()[i].value_expr.equals(t.columns()[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        status = t.insert(rows)
        assert status.num_excs == 0
        res = t.show(0)
        tbl_df = t.show(0).to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(excs.Error):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    def test_expr_udf_computed_cols(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table('test', {'c1': IntType(nullable=False)})
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
        describe_output = t.__repr__()
        # 'add1' didn't get swallowed/the expr udf is still visible in the column definition
        assert 'add1' in describe_output

        def check(t: pxt.Table) -> None:
            assert_resultset_eq(
                t.select(t.c1 + 1).order_by(t.c1).collect(),
                t.select(t.c2).order_by(t.c1).collect())
            assert_resultset_eq(
                t.select(t.c1 + 1).order_by(t.c1).collect(),
                t.select(t.c3).order_by(t.c1).collect())

        check(t)
        # test loading from store
        cl = pxt.Client(reload=True)
        t = cl.get_table('test')
        check(t)

        # make sure we can still insert data and that computed cols are still set correctly
        status = t.insert(rows)
        assert status.num_excs == 0
        check(t)

    def test_computed_col_exceptions(self, test_client: pxt.Client, test_tbl: catalog.Table) -> None:
        cl = test_client

        # exception during insert()
        schema = {'c2': IntType(nullable=False)}
        rows = list(test_tbl.select(test_tbl.c2).collect())
        t = cl.create_table('test_insert', schema)
        status = t.add_column(add1=self.f2(self.f1(t.c2)))
        assert status.num_excs == 0
        status = t.insert(rows, fail_on_exception=False)
        assert status.num_excs == 10
        assert 'test_insert.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10

        # exception during add_column()
        t = cl.create_table('test_add_column', schema)
        status = t.insert(rows)
        assert status.num_rows == 100
        assert status.num_excs == 0
        status = t.add_column(add1=self.f2(self.f1(t.c2)))
        assert status.num_excs == 10
        assert 'test_add_column.add1' in status.cols_with_excs
        assert t.where(t.add1.errortype != None).count() == 10

    def _test_computed_img_cols(self, t: catalog.Table, stores_img_col: bool) -> None:
        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        rows = [{'img': r['img']} for r in rows[:20]]
        status = t.insert(rows)
        assert status.num_rows == 20
        _ = t.count()
        _ = t.show()
        assert MediaStore.count(t.get_id()) == t.count() * stores_img_col

        # test loading from store
        cl = pxt.Client(reload=True)
        t2 = cl.get_table(t.get_name())
        assert len(t.columns()) == len(t2.columns())
        for i in range(len(t.columns())):
            if t.columns()[i].value_expr is not None:
                assert t.columns()[i].value_expr.equals(t2.columns()[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert(rows)
        assert MediaStore.count(t2.get_id()) == t2.count() * stores_img_col
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()

        # revert also removes computed images
        t2.revert()
        assert MediaStore.count(t2.get_id()) == t2.count() * stores_img_col

    @pxt.udf(return_type=ImageType(), param_types=[ImageType()])
    def img_fn_with_exc(img: PIL.Image.Image) -> PIL.Image.Image:
        raise RuntimeError

    def test_computed_img_cols(self, test_client: pxt.Client) -> None:
        cl = test_client
        schema = {'img': ImageType(nullable=False)}
        t = cl.create_table('test', schema)
        t.add_column(c2=t.img.width)
        # c3 is not stored by default
        t.add_column(c3=t.img.rotate(90))
        self._test_computed_img_cols(t, stores_img_col=False)

        t = cl.create_table('test2', schema)
        # c3 is now stored
        t.add_column(c3=t.img.rotate(90), stored=True)
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t[t.c3.errortype].show(0)

        # computed img col with exceptions
        t = cl.create_table('test3', schema)
        t.add_column(c3=self.img_fn_with_exc(t.img), stored=True)
        rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        rows = [{'img': r['img']} for r in rows[:20]]
        t.insert(rows, fail_on_exception=False)
        _ = t[t.c3.errortype].show(0)

    def test_computed_window_fn(self, test_client: pxt.Client, test_tbl: catalog.Table) -> None:
        cl = test_client
        t = test_tbl
        # backfill
        t.add_column(c9=ptf.sum(t.c2, group_by=t.c4, order_by=t.c3))

        schema = {
            'c2': IntType(nullable=False),
            'c3': FloatType(nullable=False),
            'c4': BoolType(nullable=False),
        }
        new_t = cl.create_table('insert_test', schema)
        new_t.add_column(c5=lambda c2: c2 * c2, type=IntType())
        new_t.add_column(c6=ptf.sum(new_t.c5, group_by=new_t.c4, order_by=new_t.c3))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert(rows)
        _ = new_t.show(0)

    def test_revert(self, test_client: pxt.Client) -> None:
        cl = test_client
        t1 = make_tbl(cl, 'test1', ['c1', 'c2'])
        assert t1.version() == 0
        rows1 = create_table_data(t1)
        t1.insert(rows1)
        assert t1.count() == len(rows1)
        assert t1.version() == 1
        rows2 = create_table_data(t1)
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1.version() == 2
        t1.revert()
        assert t1.count() == len(rows1)
        assert t1.version() == 1
        t1.insert(rows2)
        assert t1.count() == len(rows1) + len(rows2)
        assert t1.version() == 2

        # can't revert past version 0
        t1.revert()
        t1.revert()
        with pytest.raises(excs.Error) as excinfo:
            t1.revert()
        assert 'version 0' in str(excinfo.value)

    def test_add_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.add_column(add1=pxt.IntType(nullable=True))
        assert len(t.columns()) == num_orig_cols + 1

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.IntType(nullable=False))
        assert 'cannot add non-nullable' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.IntType(nullable=False), add3=pxt.StringType())
        assert 'requires exactly one keyword argument' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(pos=pxt.StringType(nullable=True))
        assert 'is reserved' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.IntType(nullable=False), type=pxt.StringType())
        assert '"type" is redundant' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=[[1.0, 2.0], [3.0, 4.0]], type=pxt.StringType())
        assert '"type" is redundant' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(add2=pxt.IntType(nullable=False), stored=False)
        assert 'stored=false only applies' in str(exc_info.value).lower()

        # duplicate name
        with pytest.raises(excs.Error) as exc_info:
            _ = t.add_column(c1=pxt.IntType())
        assert 'duplicate column name' in str(exc_info.value).lower()

        # 'stored' kwarg only applies to computed image columns
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=IntType(), stored=False)
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=ImageType(), stored=False)
        with pytest.raises(excs.Error):
            _ = t.add_column(c5=(t.c2 + t.c3), stored=False)

        # make sure this is still true after reloading the metadata
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols + 1

        # revert() works
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols

    def test_add_column_setitem(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t['add1'] = pxt.IntType(nullable=True)
        assert len(t.columns()) == num_orig_cols + 1
        t['computed1'] = t.c2 + 1
        assert len(t.columns()) == num_orig_cols + 2

        with pytest.raises(excs.Error) as exc_info:
            _ = t['pos'] = pxt.StringType()
        assert 'is reserved' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t[2] = pxt.StringType()
        assert 'must be a string' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add 2'] = pxt.StringType()
        assert 'invalid column name' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add2'] = {'value': t.c2 + 1, 'type': pxt.StringType()}
        assert '"type" is redundant' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add2'] = {'value': pxt.IntType()}
        assert 'value needs to be either' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            _ = t['add2'] = {'value': t.c2 + 1, 'stored': False}
        assert 'stored=false only applies' in str(exc_info.value).lower()

        # duplicate name
        with pytest.raises(excs.Error) as exc_info:
            _ = t['c1'] = pxt.IntType()
        assert 'duplicate column name' in str(exc_info.value).lower()

        # make sure this is still true after reloading the metadata
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols + 2

        # revert() works
        t.revert()
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols

    def test_drop_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.drop_column('c1')
        assert len(t.columns()) == num_orig_cols - 1

        with pytest.raises(excs.Error):
            t.drop_column('unknown')

        # make sure this is still true after reloading the metadata
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols - 1

        # revert() works
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        assert len(t.columns()) == num_orig_cols

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
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        check_rename(t, 'c1_renamed', 'c1')

        # revert() works
        _ = t.select(t.c1_renamed).collect()
        t.revert()
        _ = t.select(t.c1).collect()
        #check_rename(t, 'c1', 'c1_renamed')

        # make sure this is still true after reloading the metadata once more
        cl = pxt.Client(reload=True)
        t = cl.get_table(t.get_name())
        check_rename(t, 'c1', 'c1_renamed')

    def test_add_computed_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        status = t.add_column(add1=t.c2 + 10)
        assert status.num_excs == 0
        _ = t.show()

        # with exception in SQL
        with pytest.raises(excs.Error):
            t.add_column(add2=(t.c2 - 10) / (t.c3 - 10))

        # with exception in Python for c6.f2 == 10
        status = t.add_column(add2=(t.c6.f2 - 10) / (t.c6.f2 - 10))
        assert status.num_excs == 1
        result = t[t.add2.errortype != None][t.c6.f2, t.add2, t.add2.errortype, t.add2.errormsg].show()
        assert len(result) == 1

        # test case: exceptions in dependencies prevent execution of dependent exprs
        status = t.add_column(add3=self.f2(self.f1(t.c2)))
        assert status.num_excs == 10
        result = t[t.add3.errortype != None][t.c2, t.add3, t.add3.errortype, t.add3.errormsg].show()
        assert len(result) == 10

    def test_describe(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        fn = lambda c2: np.full((3, 4), c2)
        t.add_column(computed1=fn, type=ArrayType((3, 4), dtype=IntType()))
        t.describe()
        t.comment = 'This is a comment.'
        t.describe()

        # TODO: how to you check the output of these?
        _ = repr(t)
        _ = t._repr_html_()

    def test_common_col_names(self, test_client: pxt.Client) -> None:
        """Make sure that commonly used column names don't collide with Table member vars"""
        cl = test_client
        schema = {'id': IntType(nullable=False), 'name': StringType(nullable=False)}
        tbl = cl.create_table('test', schema)
        status = tbl.insert({'id': id, 'name': str(id)} for id in range(10))
        assert status.num_rows == 10
        assert status.num_excs == 0
        assert tbl.count() == 10
        # we can create references to those column via __getattr__
        _ = tbl.select(tbl.id, tbl.name).collect()
