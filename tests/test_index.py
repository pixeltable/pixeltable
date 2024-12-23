import random
import string
import sys
from datetime import datetime, timedelta
from typing import Union, _GenericAlias

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.functions.huggingface import clip_image, clip_text

from .utils import (assert_img_eq, clip_img_embed, clip_text_embed, e5_embed, reload_catalog,
                    skip_test_if_not_installed, validate_update_status, ReloadTester, get_sentences, assert_resultset_eq)


class TestIndex:

    # returns string
    @pxt.udf
    def bad_embed(x: str) -> str:
        return x

    # returns array w/o size
    @pxt.udf
    def bad_embed2(x: str) -> pxt.Array[(None,), pxt.Float]:
        return np.zeros(10)

    def test_similarity(self, small_img_tbl: pxt.Table, reload_tester: ReloadTester) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        for metric, is_asc in [('cosine', False), ('ip', False), ('l2', True)]:
            t.add_embedding_index('img', metric=metric, image_embed=clip_img_embed, string_embed=clip_text_embed)

            df = (
                t.select(img=t.img, sim=t.img.similarity(sample_img))
                .order_by(t.img.similarity(sample_img), asc=is_asc)
                .limit(1)
            )
            res = reload_tester.run_query(df)
            out_img = res[0, 'img']
            assert_img_eq(sample_img, out_img), f'{metric} failed'

            # TODO:  how to verify the output?
            df = (
                t.select(path=t.img.localpath, sim=t.img.similarity('parachute'))
                .order_by(t.img.similarity('parachute'), asc=is_asc)
                .limit(1)
            )
            _ = reload_tester.run_query(df)

            # can also be used in a computed column
            validate_update_status(t.add_column(sim=t.img.similarity('parachute')))
            t.drop_column('sim')

            reload_tester.run_reload_test(clear=True)

            t.drop_embedding_index(column='img')

    def test_query(self, reset_db) -> None:
        skip_test_if_not_installed('transformers')
        queries = pxt.create_table('queries', {'query_text': pxt.String})
        query_rows = [
            {'query_text': 'how much is the stock of AI companies up?'},
            {'query_text': 'what happened to the term machine learning?'}
        ]
        validate_update_status(queries.insert(query_rows))

        chunks = pxt.create_table('test_doc_chunks', {'text': pxt.String})
        chunks.insert([
            {'text': 'the stock of artificial intelligence companies is up 1000%'},
            {'text': 'the term machine learning has fallen out of fashion now that AI has been rehabilitated and is now the new hotness'},
            {'text': 'machine learning is a subset of artificial intelligence'},
            {'text': 'gas car companies are in danger of being left behind by electric car companies'},
        ])
        chunks.add_embedding_index(column='text', string_embed=clip_text_embed)

        @chunks.query
        def top_k_chunks(query_text: str) -> pxt.DataFrame:
            return chunks.select(chunks.text, sim=chunks.text.similarity(query_text)) \
                .order_by(chunks.text.similarity(query_text), asc=False) \
                .limit(5)

        _ = queries.select(queries.query_text, out=chunks.queries.top_k_chunks(queries.query_text)).collect()
        queries.add_column(chunks=chunks.queries.top_k_chunks(queries.query_text))
        _ = queries.collect()

        # make sure we can instantiate the query function from the metadata
        reload_catalog()
        queries = pxt.get_table('queries')
        _ = queries.collect()
        # insert more rows in order to run the query function
        validate_update_status(queries.insert(query_rows))

    def test_search_fn(self, small_img_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', metric='cosine', image_embed=clip_img_embed, string_embed=clip_text_embed)
        _ =  t.select(t.img.localpath).order_by(t.img.similarity(sample_img), asc=False).limit(3).collect()

        @t.query
        def img_matches(img: PIL.Image.Image):
            return t.select(t.img.localpath).order_by(t.img.similarity(img), asc=False).limit(3)

        res = list(t.select(img=t.img.localpath, matches=t.queries.img_matches(t.img)).head(1))

    def test_similarity_errors(self, indexed_img_tbl: pxt.Table, small_img_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        t = indexed_img_tbl
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity(('red truck',))).limit(1).collect()
        assert 'requires a string or' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity(['red truck'])).limit(1).collect()
        assert 'requires a string or a ' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity(t.split)).limit(1).collect()
        assert 'not an expression' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.split.similarity('red truck')).limit(1).collect()
        assert 'no index found' in str(exc_info.value).lower()

        t = small_img_tbl
        t.add_embedding_index('img', image_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert "was created without the 'string_embed' parameter" in str(exc_info.value).lower()

        t.add_embedding_index('img', string_embed=clip_text_embed, image_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()

        t.drop_embedding_index(idx_name='idx0')
        t.drop_embedding_index(idx_name='idx1')
        t.add_embedding_index('split', string_embed=clip_text_embed)
        sample_img = t.select(t.img).head(1)[0, 'img']
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.split.similarity(sample_img)).limit(1).collect()
        assert "was created without the 'image_embed' parameter" in str(exc_info.value).lower()

    def test_add_index_after_drop(self, small_img_tbl: pxt.Table) -> None:
        """ Test the an index with the same name can be added after the previous one is dropped """
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        orig_res = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()
        t.revert()
        # creating an index with the same name again after a revert should be successful
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        res = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()
        assert_resultset_eq(orig_res, res, True)
        t.revert()
        # should be true even after reloading from persistence
        reload_catalog()
        t = pxt.get_table('small_img_tbl')
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        res = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()
        assert_resultset_eq(orig_res, res, True)

        # same should hold after a drop.
        t.drop_embedding_index(column='img')
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        res = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()
        assert_resultset_eq(orig_res, res, True)
        t.drop_embedding_index(idx_name='clip_idx')
        reload_catalog()
        t = pxt.get_table('small_img_tbl')
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        res = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()
        assert_resultset_eq(orig_res, res, True)

    def test_add_embedding_index_if_exists(self, small_img_tbl: pxt.Table, reload_tester: ReloadTester) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        initial_indexes = len(t._list_index_info_for_test())

        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 1
        assert 'clip_idx' == indexes[initial_indexes]['_name']
        clip_idx_id_before = indexes[initial_indexes]['_id']

        # when index name is not provided, the index is created with
        # a newly generated name. And if_exists parameter does not apply
        # and will be ignored.
        t.add_embedding_index('img', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='error')
        assert len(t._list_index_info_for_test()) == initial_indexes + 2

        t.add_embedding_index('img', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='invalid')
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # when index name is provided, if_exists parameter is applied.
        # invalid value is rejected.
        with pytest.raises(pxt.Error) as exc_info:
            t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='invalid')
        assert "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # if_exists='error' raises an error if the index name already exists.
        # by default, if_exists='error'.
        with pytest.raises(pxt.Error) as exc_info:
            t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='error')
        assert 'duplicate index name' in str(exc_info.value).lower()
        assert len(t._list_index_info_for_test()) == initial_indexes + 3

        # if_exists='ignore' does nothing if the index name already exists.
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='ignore')
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert 'clip_idx' == indexes[initial_indexes]['_name']
        assert clip_idx_id_before == indexes[initial_indexes]['_id']

        # cannot use if_exists to ignore or replace an existing index
        # that is not an embedding (like, default btree indexes).
        assert 'idx0' == indexes[0]['_name']
        for _ie in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(pxt.Error) as exc_info:
                t.add_embedding_index('img', idx_name='idx0', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists=_ie)
            assert 'not an embedding index' in str(exc_info.value).lower(), f'for if_exists={_ie}'
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert 'idx0' == indexes[0]['_name']
        assert 'clip_idx' == indexes[initial_indexes]['_name']

        # if_exists='replace' replaces the existing index with the new one.
        t.add_embedding_index('img', idx_name='clip_idx', image_embed=clip_img_embed, string_embed=clip_text_embed, if_exists='replace')
        indexes = t._list_index_info_for_test()
        assert len(indexes) == initial_indexes + 3
        assert 'clip_idx' != indexes[initial_indexes]['_name']
        assert 'clip_idx' == indexes[initial_indexes+2]['_name']
        assert clip_idx_id_before != indexes[initial_indexes+2]['_id']

        # sanity check: use the replaced index to run a query.
        # use the index hint in similary function to ensure clip_idx is used.
        _ = t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3).collect()

        # sanity check persistence
        _ = reload_tester.run_query(t.select(t.img.localpath).order_by(t.img.similarity(sample_img, idx='clip_idx'), asc=False).limit(3))
        # bug: the index hint is not used in the similarity function
        # when we reload the metadata and run the query. PXT-382 tracks
        # it and a fix pending under PR 411. To workaround, drop all
        # other indexes on img column first to ensure clip_idx is used.
        for idx in indexes:
            if idx['_name'] != 'clip_idx':
                t.drop_embedding_index(idx_name=idx['_name'])
        _ = reload_tester.run_reload_test()

    def test_embedding_basic(self, img_tbl: pxt.Table, test_tbl: pxt.Table, reload_tester: ReloadTester) -> None:
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
            img_t.add_embedding_index(dummy_img_t.img, image_embed=clip_img_embed, string_embed=clip_text_embed)
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        img_t.add_embedding_index('img', image_embed=clip_img_embed, string_embed=clip_text_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # cannot pass another table's column reference
            img_t.drop_embedding_index(column=dummy_img_t.img);
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        # predicates on media columns that have both a B-tree and an embedding index still work
        res = img_t.where(img_t.img == rows[0]['img']).collect()
        assert len(res) == 1

        with pytest.raises(pxt.Error) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index(img_t.img, idx_name='idx0', image_embed=clip_img_embed)
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
        df = img_t.select()
        _ = reload_tester.run_query(df)
        _ = reload_tester.run_reload_test(clear=True)
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
        img_t.add_embedding_index(img_t.img, idx_name='other_idx', image_embed=clip_img_embed, string_embed=clip_text_embed)
        with pytest.raises(pxt.Error) as exc_info:
            sim = img_t.img.similarity('red truck')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        # lookup using the first index, how called idx3
        sim = img_t.img.similarity('red truck', idx='idx3')
        res = img_t.order_by(sim, asc=False).limit(1).collect()
        assert len(res) == 1
        # lookup using the second index
        sim = img_t.img.similarity('red truck', idx='other_idx')
        res = img_t.order_by(sim, asc=False).limit(1).collect()
        assert len(res) == 1

        with pytest.raises(pxt.Error) as exc_info:
            _ = img_t.img.similarity('red truck', idx='doesnotexist')
        assert "index 'doesnotexist' not found" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='img')
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        img_t.drop_embedding_index(idx_name='other_idx')

        with pytest.raises(pxt.Error) as exc_info:
            sim = img_t.img.similarity('red truck', idx='other_idx')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()
        assert "index 'other_idx' not found" in str(exc_info.value).lower()

        img_t.drop_embedding_index(column=img_t.img)
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert 'does not have an index' in str(exc_info.value).lower()

        # revert() makes the index reappear
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

        _ = reload_tester.run_query(img_t.select())

        # test that a table with an embedding index can be reloaded
        t = pxt.create_table('t1', {'s': pxt.String})
        sents = get_sentences(3)
        status = t.insert({'s': s} for s in sents)
        t.add_embedding_index('s', string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='all-mpnet-base-v2'))
        df = t.select(sim=t.s.similarity(sents[1]))
        res1 = df.collect()
        _ = reload_tester.run_query(t.select())
        _ = reload_tester.run_query(df)

        # test that a view with an embedding index on a base table column can be reloaded
        t = pxt.create_table('t2', {'s': pxt.String})
        status = t.insert({'s': s} for s in sents)
        v = pxt.create_view('v', t)
        v.add_embedding_index('s', string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='all-mpnet-base-v2'))
        # should work irrespective of whether the column is passed by name or reference
        v.add_embedding_index(v.s, string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='all-mpnet-base-v2'))
        v.add_embedding_index(t.s, string_embed=pxt.functions.huggingface.sentence_transformer.using(model_id='all-mpnet-base-v2'))
        # Expected to verify the following:
        # df = v.select(sim=v.s.similarity(sents[1]))
        # res2 = df.collect()
        # assert_resultset_eq(res1, res2)
        # and
        # _ = reload_tester.run_query(df)
        #
        # But found a bug, instead. PXT-371 tracks this.
        # RCA: The indexes above are on the view, not the base table.
        # Since they are on a column of the base table, the code
        # initializing the SimilarityExpr is looking for the index in
        # the table in the ColumnRef, which is the base table.
        # So it raises error that there's no index.
        # Fix needs discussion.
        with pytest.raises(pxt.Error) as exc_info:
            df = v.select(sim=v.s.similarity(sents[1]))
        assert 'no index found for column' in str(exc_info.value).lower()
        _ = reload_tester.run_query(v.select())

        _ = reload_tester.run_reload_test()

    def test_embedding_errors(self, small_img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        img_t = small_img_tbl

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', metric='badmetric', image_embed=clip_img_embed)
        assert 'invalid metric badmetric' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # unknown column
            img_t.add_embedding_index('does_not_exist', idx_name='idx0', image_embed=clip_img_embed)
        assert "column 'does_not_exist' unknown" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong column type
            test_tbl.add_embedding_index('c2', image_embed=clip_img_embed)
        assert 'requires string or image column' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_t.add_embedding_index('img', string_embed=clip_text_embed)
        assert 'image embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('img', image_embed=clip_image)
        assert 'must take a single image parameter' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_t.add_embedding_index('category', image_embed=clip_img_embed)
        assert 'text embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('category', string_embed=clip_text)
        assert 'must take a single string parameter' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed)
        assert 'must return an array' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed2)
        assert 'must return a 1d array of a specific length' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index()
        assert "exactly one of 'column' or 'idx_name' must be provided" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='doesnotexist')
        assert "index 'doesnotexist' does not exist" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='doesnotexist')
        assert "column 'doesnotexist' unknown" in str(exc_info.value).lower()
        with pytest.raises(AttributeError) as exc_info:
            img_t.drop_embedding_index(column=img_t.doesnotexist)
        assert 'column doesnotexist unknown' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='img')
        assert "column 'img' does not have an index" in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert "column 'img' does not have an index" in str(exc_info.value).lower()

        img_t.add_embedding_index('img', idx_name='embed0', image_embed=clip_img_embed, string_embed=clip_text_embed)
        img_t.add_embedding_index('img', idx_name='embed1', image_embed=clip_img_embed, string_embed=clip_text_embed)

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column='img')
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            sim = img_t.img.similarity('red truck')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()

    def run_btree_test(self, data: list, data_type: Union[type, _GenericAlias]) -> pxt.Table:
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

    def test_int_btree(self, reset_db) -> None:
        random.seed(1)
        data = [random.randint(0, 2 ** 63 - 1) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.Int)

    def test_float_btree(self, reset_db) -> None:
        random.seed(1)
        data = [random.uniform(0, sys.float_info.max) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.Float)

    def test_string_btree(self, reset_db) -> None:
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

    def test_timestamp_btree(self, reset_db) -> None:
        random.seed(1)
        start = datetime(2000, 1, 1)
        end = datetime(2020, 1, 1)
        delta = end - start
        delta_secs = int(delta.total_seconds())
        data = [start + timedelta(seconds=random.randint(0, int(delta_secs))) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.Timestamp)
