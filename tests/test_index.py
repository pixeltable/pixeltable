import datetime
import platform
import random
import string
import sys
from pathlib import Path
from typing import Any, Literal, _GenericAlias  # type: ignore[attr-defined]

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip

from .utils import (
    ReloadTester,
    assert_resultset_eq,
    get_sentences,
    reload_catalog,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestIndex:
    # returns string
    @staticmethod
    @pxt.udf
    def bad_embed(x: str) -> str:
        return x

    # returns array w/o size
    @staticmethod
    @pxt.udf
    def bad_embed2(x: str) -> pxt.Array[(None,), pxt.Float]:
        return np.zeros(10)

    def test_similarity_multiple_index(
        self, multi_idx_img_tbl: pxt.Table, clip_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        skip_test_if_not_installed('transformers')
        t = multi_idx_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']

        # similarity query should fail because there are multiple indices
        # img_idx1 and img_idx2 on img column in multi_idx_img_tbl
        with pytest.raises(pxt.Error, match="Column 'img' has multiple embedding indices"):
            _ = t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img), asc=False).limit(1).collect()
        # but we can specify the index to use, and the query should work
        query = (
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='img_idx1'), asc=False).limit(1)
        )
        _ = reload_tester.run_query(query)
        query = (
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='img_idx2'), asc=False).limit(1)
        )
        _ = reload_tester.run_query(query)

        # verify that the result is the same as the original query after reload
        reload_tester.run_reload_test(clear=False)

        # After the query is serialized, dropping the index should raise an error
        # on reload, because the index is no longer available
        t.drop_embedding_index(idx_name='img_idx1')
        with pytest.raises(pxt.Error) as exc_info:
            reload_tester.run_reload_test(clear=False)
        assert "index 'img_idx1' not found" in str(exc_info.value).lower()

        # After the query is serialized, dropping and recreating the index should work
        # on reload, because the index is available again even if it is not the exact
        # same one.
        t.add_embedding_index('img', idx_name='img_idx1', metric='cosine', embedding=clip_embed)
        reload_tester.run_reload_test(clear=True)

    @pytest.mark.parametrize('use_index_name,use_separate_embeddings', [(False, False), (True, False), (False, True)])
    def test_similarity(
        self,
        use_index_name: bool,
        use_separate_embeddings: bool,
        small_img_tbl: pxt.Table,
        clip_embed: pxt.Function,
        reload_tester: ReloadTester,
    ) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        res = t.select(t.img, t.img.localpath, t.img.fileurl).head(1)
        sample_img = res[0, 'img']
        sample_img_localpath = res[0, 'img_localpath']
        sample_img_file_url = res[0, 'img_fileurl']
        assert 'file://' in sample_img_file_url
        sample_img_filename = Path(sample_img_localpath).name
        sample_img_http_url = f'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/imagenette2-160/{sample_img_filename}'

        for metric, is_asc in [('cosine', False), ('ip', False), ('l2', True)]:
            iname = f'idx_{metric}_{is_asc}' if use_index_name else None
            if use_separate_embeddings:
                embed_args = {'string_embed': clip_embed, 'image_embed': clip_embed}
            else:
                embed_args = {'embedding': clip_embed}
            t.add_embedding_index('img', idx_name=iname, metric=metric, **embed_args)  # type: ignore[arg-type]

            # Similarity search on the image itself should reliably retrieve it as the top choice.
            # Make sure it works to give it the image, local path, file:// URL, or http:// URL.
            for img_input in [sample_img, sample_img_localpath, sample_img_file_url, sample_img_http_url]:
                query = (
                    t.select(img=t.img, sim=t.img.similarity(image=img_input, idx=iname))
                    .order_by(t.img.similarity(image=sample_img, idx=iname), asc=is_asc)
                    .limit(1)
                )
                res = reload_tester.run_query(query)
                out_img = res[0, 'img']
                assert sample_img == out_img, f'{metric} failed when using index {iname}'

            # There are only three images of parachutes in small_img_tbl; `clip` is good enough to reliably retrieve
            # the test image from a top-k query (with any metric).
            query = (
                t.select(img=t.img, sim=t.img.similarity(string='parachute', idx=iname))
                .order_by(t.img.similarity(string='parachute', idx=iname), asc=is_asc)
                .limit(5)
            )
            res = reload_tester.run_query(query)
            out_imgs = res['img']
            assert sample_img in out_imgs, f'{metric} failed when using index {iname}'

            # can also be used in a computed column
            validate_update_status(t.add_computed_column(sim=t.img.similarity(string='parachute')))
            t.drop_column('sim')

            reload_tester.run_reload_test()

            t.drop_embedding_index(column='img')

    def test_deprecated_similarity(
        self, small_img_tbl: pxt.Table, clip_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        """
        Test that the deprecated pattern still works, with a warning.
        (Deprecated pattern = calling similarity() without a specific modality)
        """
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', embedding=clip_embed)

        with pytest.warns(
            DeprecationWarning, match=r'Use of similarity\(\) without specifying an explicit modality is deprecated'
        ):
            query = (
                t.select(img=t.img, sim=t.img.similarity(sample_img))
                .order_by(t.img.similarity(sample_img), asc=False)
                .limit(1)
            )
            res = reload_tester.run_query(query)
            out_img = res[0, 'img']
            assert sample_img == out_img, 'deprecated similarity check failed'

        with pytest.warns(
            DeprecationWarning, match=r'Use of similarity\(\) without specifying an explicit modality is deprecated'
        ):
            query = (
                t.select(img=t.img, sim=t.img.similarity('parachute'))
                .order_by(t.img.similarity('parachute'), asc=False)
                .limit(5)
            )
            res = reload_tester.run_query(query)
            out_imgs = res['img']
            assert sample_img in out_imgs, 'deprecated similarity check failed'

        reload_tester.run_reload_test()

        t.drop_embedding_index(column='img')

    def test_query(self, reset_db: None, clip_embed: pxt.Function) -> None:
        skip_test_if_not_installed('transformers')
        queries = pxt.create_table('queries', {'query_text': pxt.String})
        query_rows = [
            {'query_text': 'how much is the stock of AI companies up?'},
            {'query_text': 'what happened to the term machine learning?'},
        ]
        validate_update_status(queries.insert(query_rows))

        chunks = pxt.create_table('test_doc_chunks', {'text': pxt.String})
        chunks.insert(
            [
                {'text': 'the stock of artificial intelligence companies is up 1000%'},
                {
                    'text': 'the term machine learning has fallen out of fashion now that '
                    'AI has been rehabilitated and is now the new hotness'
                },
                {'text': 'machine learning is a subset of artificial intelligence'},
                {'text': 'gas car companies are in danger of being left behind by electric car companies'},
            ]
        )
        chunks.add_embedding_index(column='text', string_embed=clip_embed)

        @pxt.query
        def top_k_chunks(query_text: str) -> pxt.Query:
            return (
                chunks.select(chunks.text, sim=chunks.text.similarity(string=query_text))
                .order_by(chunks.text.similarity(string=query_text), asc=False)
                .limit(5)
            )

        res1 = queries.select(queries.query_text, out=top_k_chunks(queries.query_text)).collect()
        queries.add_computed_column(chunks=top_k_chunks(queries.query_text))
        res2 = queries.collect()

        assert_resultset_eq(res1, res2)

        # Test the deprecated pattern too (similarity() without modality)

        with pytest.warns(
            DeprecationWarning, match=r'Use of similarity\(\) without specifying an explicit modality is deprecated'
        ):

            @pxt.query
            def top_k_chunks_deprecated(query_text: str) -> pxt.Query:
                return (
                    chunks.select(chunks.text, sim=chunks.text.similarity(query_text))
                    .order_by(chunks.text.similarity(query_text), asc=False)
                    .limit(5)
                )

        res1_deprecated = queries.select(queries.query_text, out=top_k_chunks_deprecated(queries.query_text)).collect()
        assert_resultset_eq(res1, res1_deprecated)

        # make sure we can instantiate the query function from the metadata
        reload_catalog()
        queries = pxt.get_table('queries')
        _ = queries.collect()
        # insert more rows in order to run the query function
        validate_update_status(queries.insert(query_rows))

    def test_search_fn(self, small_img_tbl: pxt.Table, clip_embed: pxt.Function) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', metric='cosine', embedding=clip_embed)
        _ = t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img), asc=False).limit(3).collect()

        @pxt.query
        def img_matches(img: PIL.Image.Image) -> pxt.Query:
            return t.select(t.img.localpath).order_by(t.img.similarity(image=img), asc=False).limit(3)

        _ = list(t.select(img=t.img.localpath, matches=img_matches(t.img)).head(1))

    def test_similarity_errors(
        self, indexed_img_tbl: pxt.Table, small_img_tbl: pxt.Table, clip_embed: pxt.Function
    ) -> None:
        skip_test_if_not_installed('transformers')
        t = indexed_img_tbl

        type_failures = (
            ('item', '`str` or `PIL.Image.Image`'),
            ('string', '`str`'),
            ('image', '`str` or `PIL.Image.Image`'),
            ('audio', r'`str` \(path to audio file\)'),
            ('video', r'`str` \(path to video file\)'),
        )

        with pytest.warns(
            DeprecationWarning, match=r'Use of similarity\(\) without specifying an explicit modality is deprecated'
        ):
            for param, expected in type_failures:
                with pytest.raises(pxt.Error, match=rf'similarity\(.*\): expected {expected}; got `tuple`') as exc_info:
                    _ = t.order_by(t.img.similarity(**{param: ('red truck',)})).limit(1).collect()  # type: ignore[arg-type]
            for param, expected in type_failures:
                with pytest.raises(pxt.Error, match=rf'similarity\(.*\): expected {expected}; got `list`') as exc_info:
                    _ = t.order_by(t.img.similarity(**{param: ['red truck']})).limit(1).collect()  # type: ignore[arg-type]

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity(string=t.split)).limit(1).collect()  # type: ignore[arg-type]
        assert 'not an expression' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error, match="No embedding index found for column 'split'"):
            _ = t.order_by(t.split.similarity(string='red truck')).limit(1).collect()

        t = small_img_tbl
        t.add_embedding_index('img', image_embed=clip_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity(string='red truck')).limit(1).collect()
        assert 'does not have a string embedding' in str(exc_info.value).lower()

        t.add_embedding_index('img', embedding=clip_embed)
        with pytest.raises(pxt.Error, match="Column 'img' has multiple embedding indices"):
            _ = t.order_by(t.img.similarity(string='red truck')).limit(1).collect()

        # Similarity fails when attempted on a snapshot
        t_s = pxt.create_snapshot('t_s', t)
        with pytest.raises(pxt.Error, match='Snapshot does not support indices'):
            _ = t_s.order_by(t_s.img.similarity(string='red truck')).limit(1).collect()

        # embedding() fails on a snapshot
        with pytest.raises(pxt.Error, match='Snapshot does not support indices'):
            _ = t_s.select(t_s.img.embedding(idx='other_idx')).limit(2)

        t.drop_embedding_index(idx_name='idx0')
        t.drop_embedding_index(idx_name='idx1')
        t.add_embedding_index('split', string_embed=clip_embed)
        sample_img = t.select(t.img).head(1)[0, 'img']
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.split.similarity(image=sample_img)).limit(1).collect()
        assert 'does not have an image embedding' in str(exc_info.value).lower()

    def test_add_index_after_drop(self, small_img_tbl: pxt.Table, clip_embed: pxt.Function) -> None:
        """Test that an index with the same name can be added after the previous one is dropped"""
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        orig_res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        t.revert()
        # creating an index with the same name again after a revert should be successful
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)
        t.revert()
        # should be true even after reloading from persistence
        reload_catalog()
        t = pxt.get_table('small_img_tbl')
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)

        # same should hold after a drop.
        t.drop_embedding_index(column='img')
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)
        t.drop_embedding_index(idx_name='clip_idx')
        reload_catalog()
        t = pxt.get_table('small_img_tbl')
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)

    def test_add_embedding_index_if_exists(
        self, small_img_tbl: pxt.Table, reload_tester: ReloadTester, clip_embed: pxt.Function
    ) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        initial_indexes = len(t._list_index_info_for_test())

        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 1
        assert indexes[initial_indexes]['_name'] == 'clip_idx'
        clip_idx_id_before = indexes[initial_indexes]['_id']

        # when index name is not provided, the index is created with
        # a newly generated name. And if_exists parameter does not apply
        # and will be ignored.
        t.add_embedding_index('img', embedding=clip_embed, if_exists='error')
        assert len(t._list_index_info_for_test()) == initial_indexes + 2

        t.add_embedding_index('img', embedding=clip_embed, if_exists='invalid')  # type: ignore[arg-type]
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # when index name is provided, if_exists parameter is applied.
        # invalid value is rejected.
        with pytest.raises(pxt.Error) as exc_info:
            t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed, if_exists='invalid')  # type: ignore[arg-type]
        assert (
            "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        )
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # if_exists='error' raises an error if the index name already exists.
        # by default, if_exists='error'.
        with pytest.raises(pxt.Error, match='Duplicate index name'):
            t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed)
        with pytest.raises(pxt.Error, match='Duplicate index name'):
            t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed, if_exists='error')
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # if_exists='ignore' does nothing if the index name already exists.
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed, if_exists='ignore')
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert indexes[initial_indexes]['_name'] == 'clip_idx'
        assert clip_idx_id_before == indexes[initial_indexes]['_id']

        # cannot use if_exists to ignore or replace an existing index
        # that is not an embedding (like, default btree indexes).
        assert indexes[0]['_name'] == 'idx0'
        for _ie in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(pxt.Error, match='not an embedding index'):
                t.add_embedding_index('img', idx_name='idx0', embedding=clip_embed, if_exists=_ie)  # type: ignore[arg-type]
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert indexes[0]['_name'] == 'idx0'
        assert indexes[initial_indexes]['_name'] == 'clip_idx'

        # if_exists='replace' replaces the existing index with the new one.
        t.add_embedding_index('img', idx_name='clip_idx', embedding=clip_embed, if_exists='replace')
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert indexes[initial_indexes]['_name'] != 'clip_idx'
        assert indexes[initial_indexes + 2]['_name'] == 'clip_idx'
        assert clip_idx_id_before != indexes[initial_indexes + 2]['_id']

        # sanity check: use the replaced index to run a query.
        # use the index hint in similarity function to ensure clip_idx is used.
        _ = reload_tester.run_query(
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False).limit(3)
        )

        # sanity check persistence
        reload_tester.run_reload_test()

    def test_update_img(
        self,
        img_tbl: pxt.Table,
        test_tbl: pxt.Table,
        clip_embed: pxt.Function,
        e5_embed: pxt.Function,
        all_mpnet_embed: pxt.Function,
        reload_tester: ReloadTester,
    ) -> None:
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        short_rows = rows[:5]
        new_rows: list[dict[str, Any]] = []
        for n, row in enumerate(short_rows):
            row['pkey'] = n
            new_rows.append(row)

        # create table with fewer rows to speed up testing
        schema = {'pkey': ts.IntType(nullable=False), 'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = 'update_test'
        img_t = pxt.create_table(tbl_name, schema, primary_key='pkey')
        img_t.insert(new_rows)
        print(img_t.head())

        with pytest.raises(pxt.Error, match='property of another column is not allowed'):
            img_t.add_computed_column(emsg=img_t.img.errormsg)

        with pytest.raises(pxt.Error, match='property of another column is not allowed'):
            img_t.add_computed_column(etype=img_t.img.errortype)

        with pytest.raises(AttributeError, match='Unknown method '):
            img_t.add_computed_column(etype=img_t.img.cellmd)

        # Update the first row with a new image
        repl_row = rows[6]
        img_t.update(repl_row, where=img_t.pkey == 0, cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

        # Update the row, removing the image
        repl_row['img'] = None
        img_t.update(repl_row, where=img_t.pkey == 0, cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

        # Update the row with a new image
        repl_row = rows[7]
        repl_row['pkey'] = 0
        with pytest.raises(pxt.Error, match='is a media column and cannot be updated'):
            img_t.batch_update([repl_row], cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

        # Update the row again, looking for an error
        with pytest.raises(pxt.Error, match='is a media column and cannot be updated'):
            img_t.batch_update([repl_row], cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

    def test_embedding_access(self, img_tbl: pxt.Table, e5_embed: pxt.Function) -> None:
        skip_test_if_not_installed('transformers', 'sentence_transformers')
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = 'access_test'
        img_t = pxt.create_table(tbl_name, schema)
        img_t.insert(rows[:5])

        # Add computed column based on the other_idx embedding index
        img_t.add_embedding_index(img_t.category, idx_name='cat_idx', string_embed=e5_embed)
        img_t.add_computed_column(ebd_copy=img_t.category.embedding(idx='cat_idx'))
        img_t.insert([rows[6]])

        # Attempt to drop the embedding index
        with pytest.raises(pxt.Error, match="Cannot drop index 'cat_idx' because the following columns depend on it"):
            img_t.drop_embedding_index(column=img_t.category)

        img_t.add_computed_column(sim=img_t.category.similarity(string='red_truck', idx='cat_idx'))
        with pytest.raises(pxt.ExprEvalError) as exc_info:
            img_t.insert([rows[7]])
        assert 'cannot be used in a computed column' in str(exc_info.value.__cause__)

        img_t.drop_column('sim')
        img_t.drop_column('ebd_copy')
        img_t.drop_embedding_index(column=img_t.category)

    def test_embedding_basic(
        self, img_tbl: pxt.Table, clip_embed: pxt.Function, e5_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        skip_test_if_not_installed('sentence_transformers')
        skip_test_if_not_installed('transformers')

        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = 'index_test'
        img_t = pxt.create_table(tbl_name, schema)
        img_t.insert(rows[:30])
        dummy_img_t = pxt.create_table('dummy', schema)
        dummy_img_t.insert(rows[:10])

        with pytest.raises(pxt.Error) as exc_info:
            # cannot pass another table's column reference
            img_t.add_embedding_index(dummy_img_t.img, embedding=clip_embed)
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        img_t.add_embedding_index('img', embedding=clip_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # cannot pass another table's column reference
            img_t.drop_embedding_index(column=dummy_img_t.img)
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        # predicates on media columns that have both a B-tree and an embedding index still work
        res = img_t.where(img_t.img == rows[0]['img']).collect()
        assert len(res) == 1

        res = img_t.select(img_t.img.embedding()).limit(2).collect()
        assert len(res) == 2
        assert isinstance(res[0, 0], np.ndarray)

        with pytest.raises(pxt.Error) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=clip_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index(img_t.img, idx_name='idx0', image_embed=clip_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        img_t.add_embedding_index(img_t.category, idx_name='cat_idx', string_embed=e5_embed)

        # revert() removes the index
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='category')
        assert 'does not have an index' in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.category)
        assert 'does not have an index' in str(exc_info.value).lower()

        rows = list(img_t.collect())
        status = img_t.update({'split': 'other'}, where=img_t.split == 'test')
        assert status.num_excs == 0

        status = img_t.delete()
        assert status.num_excs == 0

        # revert delete()
        img_t.revert()
        # revert update()
        img_t.revert()

        # make sure we can still do DML after reloading the metadata
        query = img_t.select()
        _ = reload_tester.run_query(query)
        reload_tester.run_reload_test(clear=True)
        img_t = pxt.get_table(tbl_name)
        status = img_t.insert(rows)
        assert status.num_excs == 0

        status = img_t.update({'split': 'other'}, where=img_t.split == 'test')
        assert status.num_excs == 0

        status = img_t.delete()
        assert status.num_excs == 0

        # revert delete()
        img_t.revert()
        # revert update()
        img_t.revert()

        # multiple indices
        img_t.add_embedding_index(img_t.img, idx_name='other_idx', embedding=clip_embed)
        # lookup using the first index, how called idx3
        sim = img_t.img.similarity(string='red truck', idx='idx3')
        res = img_t.order_by(sim, asc=False).limit(1).collect()
        assert len(res) == 1
        # lookup using the second index
        sim = img_t.img.similarity(string='red truck', idx='other_idx')
        res = img_t.order_by(sim, asc=False).limit(1).collect()
        assert len(res) == 1

        r = img_t.select(img_t.img.embedding(idx='other_idx')).limit(2).collect()
        assert len(r) == 2
        assert isinstance(r[0, 0], np.ndarray)

        # embedding() fails when multiple indices are present
        with pytest.raises(pxt.Error, match='has multiple embedding indices'):
            _ = img_t.select(img_t.img.embedding()).collect()

        # Adding an index with an invalid index name fails
        with pytest.raises(pxt.Error, match='Invalid column name'):
            img_t.add_embedding_index(img_t.img, idx_name='BOGUS COL NAME', embedding=clip_embed)

        with pytest.raises(pxt.Error) as exc_info:
            _ = img_t.img.similarity(string='red truck', idx='doesnotexist')
        assert "index 'doesnotexist' not found" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='img')
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        img_t.drop_embedding_index(idx_name='other_idx')

        with pytest.raises(pxt.Error) as exc_info:
            sim = img_t.img.similarity(string='red truck', idx='other_idx')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()
        assert "index 'other_idx' not found" in str(exc_info.value).lower()

        img_t.drop_embedding_index(column=img_t.img)
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert 'does not have an index' in str(exc_info.value).lower()

        # revert() makes the index reappear
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=clip_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

        _ = reload_tester.run_query(img_t.select())

    @pytest.mark.skipif(platform.system() == 'Windows', reason='Segfaulting on Windows runners for unknown reasons')
    def test_view_indices(
        self, reset_db: None, e5_embed: pxt.Function, all_mpnet_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        skip_test_if_not_installed('sentence_transformers')

        # Create a base table
        t = pxt.create_table('t1', {'n': pxt.Int, 's': pxt.String})
        sentences = get_sentences(20)
        status = t.insert({'n': i, 's': s} for i, s in enumerate(sentences))
        validate_update_status(status, 20)

        # Create a view that indexes the base table column
        v = pxt.create_view('v', t.where(t.n % 2 == 0))
        v.add_embedding_index('s', string_embed=all_mpnet_embed)

        query1 = v.select(sim1=v.s.similarity(string=sentences[1]))
        res1 = reload_tester.run_query(query1)

        # Now add an index to the base table, which should be independent of the view index
        t.add_embedding_index('s', string_embed=e5_embed)
        query2 = t.where(t.n % 2 == 0).select(sim2=t.s.similarity(string=sentences[1]))
        res2 = reload_tester.run_query(query2)

        # Now query the view again twice: once with the column referenced as `v.s`, and once as `t.s`
        query3 = v.select(sim3=v.s.similarity(string=sentences[1]))
        res3 = reload_tester.run_query(query3)
        query4 = v.select(sim4=t.s.similarity(string=sentences[1]))
        res4 = reload_tester.run_query(query4)

        # `v.s` should use the view index, while `t.s` should use the base table index
        assert_resultset_eq(res1, res3)
        assert_resultset_eq(res2, res4)

        reload_tester.run_reload_test()

    def test_embedding_errors(self, small_img_tbl: pxt.Table, test_tbl: pxt.Table, clip_embed: pxt.Function) -> None:
        skip_test_if_not_installed('transformers')
        img_t = small_img_tbl

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', metric='badmetric', image_embed=clip_embed)  # type: ignore[arg-type]
        assert 'invalid metric badmetric' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # unknown column
            img_t.add_embedding_index('does_not_exist', idx_name='idx0', image_embed=clip_embed)
        assert 'Unknown column: does_not_exist' in str(exc_info.value)

        with pytest.raises(pxt.Error) as exc_info:
            # no embedding function specified
            img_t.add_embedding_index('img')
        assert '`embed`, `string_embed`, or `image_embed` must be specified' in str(exc_info.value)

        with pytest.raises(pxt.Error, match=r"Type `Int` of column 'c2' is not a valid type for an embedding index."):
            # wrong column type
            test_tbl.add_embedding_index('c2', image_embed=clip_embed)

        with pytest.raises(
            pxt.Error, match=r"The specified embedding function does not support the type `Image` of column 'img'."
        ):
            # missing embedding function
            img_t.add_embedding_index('img', string_embed=clip_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('img', image_embed=clip)
        assert 'must take a single image parameter' in str(exc_info.value).lower()

        with pytest.raises(
            pxt.Error,
            match=r"The specified embedding function does not support the type `String` of column 'category'.",
        ):
            # missing embedding function
            img_t.add_embedding_index('category', image_embed=clip_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('category', string_embed=clip)
        assert 'must take a single string parameter' in str(exc_info.value).lower()

        with pytest.raises(
            pxt.Error,
            match=r'The function `clip` is not a valid embedding: '
            'it must take a single string, image, audio, or video parameter',
        ):
            # no matching signature
            img_t.add_embedding_index('img', embedding=clip)

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed)
        assert 'must return an array' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed2)
        assert 'must return a 1-dimensional array of a specific length' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index()
        assert "exactly one of 'column' or 'idx_name' must be provided" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error, match="Index 'doesnotexist' does not exist"):
            img_t.drop_embedding_index(idx_name='doesnotexist')
        with pytest.raises(pxt.Error, match="Index 'doesnotexist' does not exist"):
            img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='error')

        img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='ignore')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error, match='Unknown column: doesnotexist'):
            img_t.drop_embedding_index(column='doesnotexist')
        # when dropping an index via a column, if_not_exists does not
        # apply to non-existent column; it will still raise error.
        with pytest.raises(pxt.Error, match='Unknown column: doesnotexist'):
            img_t.drop_embedding_index(column='doesnotexist', if_not_exists='invalid')  # type: ignore[arg-type]
        with pytest.raises(AttributeError) as exc_info:
            img_t.drop_embedding_index(column=img_t.doesnotexist)
        assert 'Unknown column: doesnotexist' in str(exc_info.value)

        with pytest.raises(pxt.Error, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column='img')
        with pytest.raises(pxt.Error, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column=img_t.img)
        # when dropping an index via a column, if_not_exists applies if
        # the column does not have any index to drop.
        with pytest.raises(pxt.Error, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column='img', if_not_exists='error')
        img_t.drop_embedding_index(column=img_t.img, if_not_exists='ignore')

        img_t.add_embedding_index('img', idx_name='embed0', embedding=clip_embed)
        img_t.add_embedding_index('img', idx_name='embed1', embedding=clip_embed)

        with pytest.raises(pxt.Error, match="Column 'img' has multiple indices"):
            img_t.drop_embedding_index(column='img')
        with pytest.raises(pxt.Error, match="Column 'img' has multiple indices"):
            img_t.drop_embedding_index(column=img_t.img)

        with pytest.raises(pxt.Error, match="Column 'img' has multiple embedding indices"):
            sim = img_t.img.similarity(string='red truck')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()

        if not Env.get().is_using_cockroachdb:
            # TODO(PXT-941): Revisit embedding index precision behavior for cloud launch
            # In CockroachDB we use VECTOR type that doesn't have the same limitation as pgvector's VECTOR and HALFVEC
            with pytest.raises(
                pxt.Error,
                match="Embedding index's vector dimensionality 4001 exceeds maximum of 4000 for fp16 precision",
            ):
                test_tbl.add_embedding_index(
                    test_tbl.c1, embedding=TestIndex.dummy_embedding.using(n=4001), precision='fp16'
                )
            with pytest.raises(
                pxt.Error,
                match="Embedding index's vector dimensionality 2001 exceeds maximum of 2000 for fp32 precision",
            ):
                test_tbl.add_embedding_index(
                    test_tbl.c1, embedding=TestIndex.dummy_embedding.using(n=2001), precision='fp32'
                )

        with pytest.raises(pxt.Error, match=r"Invalid precision.+Must be one of: \['fp16', 'fp32'\]"):
            test_tbl.add_embedding_index(
                test_tbl.c1,
                embedding=TestIndex.dummy_embedding.using(n=2001),
                precision='invalid',  # type: ignore[arg-type]
            )
        with pytest.raises(pxt.Error, match='is not a valid embedding: it returns an array of invalid length 0'):
            test_tbl.add_embedding_index(test_tbl.c1, embedding=TestIndex.dummy_embedding.using(n=0), precision='fp16')

    def run_btree_test(self, data: list, data_type: type | _GenericAlias) -> pxt.Table:
        t = pxt.create_table('btree_test', {'data': data_type})
        num_rows = len(data)
        rows = [{'data': value} for value in data]
        validate_update_status(t.insert(rows), expected_rows=num_rows)
        median_value = sorted(data)[num_rows // 2]

        assert t.where(t.data == median_value).count() == 1
        assert t.where(t.data < median_value).count() == num_rows // 2
        assert t.where(t.data <= median_value).count() == num_rows // 2 + 1
        assert t.where(t.data > median_value).count() == num_rows // 2
        assert t.where(t.data >= median_value).count() == num_rows // 2 + 1

        return t

    BTREE_TEST_NUM_ROWS = 10001  # ~10k rows: incentivize Postgres to use the index

    def test_int_btree(self, reset_db: None) -> None:
        random.seed(1)
        data = [random.randint(0, 2**63 - 1) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.Int)

    def test_float_btree(self, reset_db: None) -> None:
        random.seed(1)
        data = [random.uniform(0, sys.float_info.max) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.Float)

    def test_string_btree(self, reset_db: None) -> None:
        def create_random_str(n: int) -> str:
            chars = string.ascii_letters + string.digits
            return ''.join(random.choice(chars) for _ in range(n))

        random.seed(1)
        # create random strings of length 200-300 characters
        data = [create_random_str(200 + i % 100) for i in range(self.BTREE_TEST_NUM_ROWS)]
        t = self.run_btree_test(data, pxt.String)

        # edge cases: strings that are at and above the max length
        sorted_data = sorted(data)
        # the index of the first string of length 255
        idx = next(i for i, s in enumerate(sorted_data) if len(s) == 255)
        s = sorted_data[idx]
        assert t.where(t.data == s).count() == 1
        assert t.where(t.data <= s).count() == idx + 1
        assert t.where(t.data < s).count() == idx
        assert t.where(t.data >= s).count() == self.BTREE_TEST_NUM_ROWS - idx
        assert t.where(t.data > s).count() == self.BTREE_TEST_NUM_ROWS - idx - 1

        with pytest.raises(pxt.Error) as exc_info:
            assert len(data[56]) == 256
            _ = t.where(t.data == data[56]).count()
        assert 'String literal too long' in str(exc_info.value)

        # test that Comparison uses BtreeIndex.MAX_STRING_LEN
        t = pxt.create_table('test_max_str_len', {'data': pxt.String})
        rows = [{'data': s}, {'data': s + 'a'}]
        validate_update_status(t.insert(rows), expected_rows=len(rows))
        assert t.where(t.data >= s).count() == 2
        assert t.where(t.data > s).count() == 1

    def test_timestamp_btree(self, reset_db: None) -> None:
        random.seed(1)
        start = datetime.datetime(2000, 1, 1)
        end = datetime.datetime(2020, 1, 1)
        delta = end - start
        delta_secs = int(delta.total_seconds())
        data = [
            start + datetime.timedelta(seconds=random.randint(0, int(delta_secs)))
            for _ in range(self.BTREE_TEST_NUM_ROWS)
        ]
        self.run_btree_test(data, pxt.Timestamp)

    def test_date_btree(self, reset_db: None) -> None:
        random.seed(1)
        start = datetime.date(2000, 1, 1)
        end = datetime.date(2100, 1, 1)
        delta = end - start
        delta_days = int(delta.days)
        assert delta_days > 3 * self.BTREE_TEST_NUM_ROWS
        data = [
            start + datetime.timedelta(days=random.randint(0, int(delta_days))) for _ in range(self.BTREE_TEST_NUM_ROWS)
        ]
        self.run_btree_test(data, pxt.Date)

    @pxt.udf
    @staticmethod
    def dummy_embedding(text: str, n: int) -> pxt.Array[(None,), np.float32]:
        if 'zero' in text:
            arr = np.zeros((n,), dtype=np.float32)
            arr[n // 2 :] = 1
            return arr
        if 'one' in text:
            arr = np.zeros((n,), dtype=np.float32)
            arr[: n // 2] = 1
            return arr
        return np.random.rand(n).astype(np.float32)

    @dummy_embedding.conditional_return_type
    @staticmethod
    def _(n: int) -> ts.ArrayType:
        return ts.ArrayType((n,), dtype=np.dtype('float32'), nullable=False)

    @pytest.mark.parametrize('reload_cat', [True, False], ids=['reload_cat', 'no_reload_cat'])
    @pytest.mark.parametrize('metric', ['l2', 'cosine', 'ip'])
    @pytest.mark.parametrize('precision', ['fp16', 'fp32'])
    def test_embedding_index_precision(
        self,
        reset_db: None,
        reload_cat: bool,
        metric: Literal['cosine', 'ip', 'l2'],
        precision: Literal['fp16', 'fp32'],
    ) -> None:
        # Note: dummy embeddings produced by our test UDF are not normalized, so, strictly speaking, it cannot be
        # used with IP metric, however it appears to work fine anyway, and that's good enough for our test purpose.
        t = pxt.create_table('test', {'rowid': pxt.Int, 'text': pxt.String}, if_exists='replace')
        n = 123
        t.add_embedding_index(
            t.text,
            embedding=TestIndex.dummy_embedding.using(n=n),
            metric=metric,
            precision=precision,
            idx_name='test_idx',
        )
        t.insert(
            [
                {'rowid': 0, 'text': 'string zero'},
                {'rowid': 1, 'text': 'string one'},
                {'rowid': 2, 'text': 'something else'},
            ]
        )

        reload_catalog(reload_cat)

        res = t.select(t.text.embedding()).collect()
        assert len(res.schema) == 1
        assert ts.ArrayType((n,), np.dtype('float32')).matches(res.schema['col_0']), res.schema
        assert len(res) == 3
        assert all(isinstance(row['col_0'], np.ndarray) and row['col_0'].dtype == np.float32 for row in res)

        sim = t.text.similarity(string='zero')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 0

        sim = t.text.similarity(string='one')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 1
