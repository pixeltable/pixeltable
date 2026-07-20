import datetime
import random
import string
import sys
from pathlib import Path
from typing import Any, Callable, Literal, _GenericAlias  # type: ignore[attr-defined]

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip

from .utils import (
    CatalogMode,
    ReloadTester,
    assert_resultset_eq,
    get_sentences,
    list_store_indexes,
    local_embedding,
    pxt_raises,
    reload_catalog,
    rerun_on_network_error,
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
        self, multi_idx_img_tbl: pxt.Table, local_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        t = multi_idx_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']

        # similarity query should fail because there are multiple indices
        # img_idx1 and img_idx2 on img column in multi_idx_img_tbl
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'img' has multiple embedding indices"):
            _ = t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img), asc=False).limit(1).collect()
        # but we can specify the index to use, and the query should work
        query_idx1 = (
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='img_idx1'), asc=False).limit(1)
        )
        _ = reload_tester.run_query(query_idx1)
        query = (
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='img_idx2'), asc=False).limit(1)
        )
        _ = reload_tester.run_query(query)

        # verify that the result is the same as the original query after reload (local only: ReloadTester does not
        # yet implement reload semantics for delegated catalogs and skips proxy queries)
        reload_tester.run_reload_test(clear=False)

        # dropping the index makes a query that references it fail at execution, because the index is no longer
        # available
        t.drop_embedding_index(idx_name='img_idx1')
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match=r'(?i).*img_idx1.*not found.*'):
            _ = query_idx1.collect()

        # dropping and recreating the index makes the query work again, because the index is available again even
        # if it is not the exact same one
        t.add_embedding_index('img', idx_name='img_idx1', metric='cosine', embedding=local_embed)
        assert len(query_idx1.collect()) == 1
        reload_tester.run_reload_test(clear=True)

    @pytest.mark.parametrize('use_index_name,use_separate_embeddings', [(False, False), (True, False), (False, True)])
    @rerun_on_network_error()
    def test_similarity(
        self,
        use_index_name: bool,
        use_separate_embeddings: bool,
        small_img_tbl: pxt.Table,
        clip_or_local: tuple[pxt.Function, bool],
        reload_tester: ReloadTester,
        catalog_mode: CatalogMode,
    ) -> None:
        embed, is_dummy_model = clip_or_local
        skip_test_if_not_installed('imagehash')
        if not is_dummy_model:
            skip_test_if_not_installed('transformers')
        t = small_img_tbl
        res = t.select(t.img, t.img.localpath, t.img.fileurl).head(1)
        sample_img = res[0, 'img']
        sample_img_localpath = res[0, 'img_localpath']
        sample_img_file_url = res[0, 'img_fileurl']
        # A PIL image is a self-contained similarity input that works in both modes. A local path, file:// URL, or
        # an http URL rebuilt from the original filename only identifies the same image against a collocated store:
        # over the proxy .localpath is a fetched cache copy (hashed name) and .fileurl is a fetchable daemon URL.
        img_inputs: list[Any] = [sample_img]
        if catalog_mode == 'local':
            assert 'file:/' in sample_img_file_url
            sample_img_filename = Path(sample_img_localpath).name
            sample_img_http_url = f'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/imagenette2-160/{sample_img_filename}'
            img_inputs += [sample_img_localpath, sample_img_file_url, sample_img_http_url]

        for metric, is_asc in [('cosine', False), ('ip', False), ('l2', True)]:
            iname = f'idx_{metric}_{is_asc}' if use_index_name else None
            if use_separate_embeddings:
                embed_args = {'string_embed': embed, 'image_embed': embed}
            else:
                embed_args = {'embedding': embed}
            t.add_embedding_index('img', idx_name=iname, metric=metric, **embed_args)  # type: ignore[arg-type]

            # Similarity search on the image itself should reliably retrieve it as the top choice.
            # Make sure it works to give it the image, local path, file:// URL, or http:// URL.
            for img_input in img_inputs:
                query = (
                    t.select(img=t.img, sim=t.img.similarity(image=img_input, idx=iname))
                    .order_by(t.img.similarity(image=sample_img, idx=iname), asc=is_asc)
                    .limit(1)
                )
                res = reload_tester.run_query(query)
                out_img = res[0, 'img']
                assert sample_img == out_img, f'{metric} failed when using index {iname}'

            # There are only three images of parachutes in small_img_tbl; `clip` is good enough to reliably retrieve
            # the test image from a top-k query (with any metric). The local embedding has no cross-modal semantics,
            # so the retrieval is only asserted for the real model; the query itself runs in both cases.
            query = (
                t.select(img=t.img, sim=t.img.similarity(string='parachute', idx=iname))
                .order_by(t.img.similarity(string='parachute', idx=iname), asc=is_asc)
                .limit(5)
            )
            res = reload_tester.run_query(query)
            out_imgs = res['img']
            if not is_dummy_model:
                assert sample_img in out_imgs, f'{metric} failed when using index {iname}'

            # can also be used in a computed column
            validate_update_status(t.add_computed_column(sim=t.img.similarity(string='parachute')))
            t.drop_column('sim')

            reload_tester.run_reload_test()

            t.drop_embedding_index(column='img')

    def test_deprecated_similarity(
        self, small_img_tbl: pxt.Table, clip_or_local: tuple[pxt.Function, bool], reload_tester: ReloadTester
    ) -> None:
        """
        Test that the deprecated pattern still works, with a warning.
        (Deprecated pattern = calling similarity() without a specific modality)
        """
        embed, is_dummy_model = clip_or_local
        skip_test_if_not_installed('imagehash')
        if not is_dummy_model:
            skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', embedding=embed)

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
            if not is_dummy_model:  # cross-modal retrieval requires a real CLIP model
                assert sample_img in out_imgs, 'deprecated similarity check failed'

        reload_tester.run_reload_test()

        t.drop_embedding_index(column='img')

    def test_query(self, make_catalog_path: Callable[[str], str], local_embed: pxt.Function) -> None:
        # def test_query(self, uses_db: None, local_embed: pxt.Function) -> None:
        p = make_catalog_path
        queries = pxt.create_table(p('queries'), {'query_text': pxt.String})
        query_rows = [
            {'query_text': 'how much is the stock of AI companies up?'},
            {'query_text': 'what happened to the term machine learning?'},
        ]
        validate_update_status(queries.insert(query_rows))

        chunks = pxt.create_table(p('test_doc_chunks'), {'text': pxt.String})
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
        chunks.add_embedding_index(column='text', string_embed=local_embed)

        @pxt.query
        def top_k_chunks(query_text: str) -> pxt.Query:
            return (
                chunks.select(chunks.text, sim=chunks.text.similarity(string=query_text))
                .order_by(chunks.text.similarity(string=query_text), asc=False)
                .limit(5)
            )

        res1 = (
            queries.select(queries.query_text, out=top_k_chunks(queries.query_text))
            .order_by(queries.query_text)
            .collect()
        )
        queries.add_computed_column(chunks=top_k_chunks(queries.query_text))
        res2 = queries.order_by(queries.query_text).collect()

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

        res1_deprecated = (
            queries.select(queries.query_text, out=top_k_chunks_deprecated(queries.query_text))
            .order_by(queries.query_text)
            .collect()
        )
        assert_resultset_eq(res1, res1_deprecated)

        # make sure we can instantiate the query function from the metadata
        reload_catalog()
        queries = pxt.get_table(p('queries'))
        _ = queries.collect()
        # insert more rows in order to run the query function
        validate_update_status(queries.insert(query_rows))

    def test_search_fn(self, small_img_tbl: pxt.Table, local_embed: pxt.Function) -> None:
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', metric='cosine', embedding=local_embed)
        _ = t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img), asc=False).limit(3).collect()

        @pxt.query
        def img_matches(img: PIL.Image.Image) -> pxt.Query:
            return t.select(t.img.localpath).order_by(t.img.similarity(image=img), asc=False).limit(3)

        _ = list(t.select(img=t.img.localpath, matches=img_matches(t.img)).head(1))

    def test_similarity_errors(
        self,
        indexed_img_tbl: pxt.Table,
        small_img_tbl: pxt.Table,
        make_catalog_path: Callable[[str], str],
        local_embed: pxt.Function,
    ) -> None:
        p = make_catalog_path
        t = indexed_img_tbl

        type_failures = (
            ('item', '`str` or `PIL.Image.Image`', pxt.ErrorCode.TYPE_MISMATCH),
            ('string', '`str`', pxt.ErrorCode.TYPE_MISMATCH),
            ('image', '`str` or `PIL.Image.Image`', pxt.ErrorCode.TYPE_MISMATCH),
            ('audio', r'`str` \(path to audio file\)', pxt.ErrorCode.UNSUPPORTED_OPERATION),
            ('video', r'`str` \(path to video file\)', pxt.ErrorCode.UNSUPPORTED_OPERATION),
        )

        with pytest.warns(
            DeprecationWarning, match=r'Use of similarity\(\) without specifying an explicit modality is deprecated'
        ):
            for param, expected, code in type_failures:
                with pxt_raises(code, match=rf'similarity\(.*\): expected {expected}; got `tuple`'):
                    _ = t.order_by(t.img.similarity(**{param: ('red truck',)})).limit(1).collect()  # type: ignore[arg-type]
            for param, expected, code in type_failures:
                with pxt_raises(code, match=rf'similarity\(.*\): expected {expected}; got `list`'):
                    _ = t.order_by(t.img.similarity(**{param: ['red truck']})).limit(1).collect()  # type: ignore[arg-type]

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.order_by(t.img.similarity(string=t.split)).limit(1).collect()  # type: ignore[arg-type]
        assert 'not an expression' in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="No embedding index found for column 'split'"):
            _ = t.order_by(t.split.similarity(string='red truck')).limit(1).collect()

        t = small_img_tbl
        t.add_embedding_index('img', image_embed=local_embed)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.order_by(t.img.similarity(string='red truck')).limit(1).collect()
        assert 'does not have a string embedding' in str(exc_info.value).lower()

        t.add_embedding_index('img', embedding=local_embed)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'img' has multiple embedding indices"):
            _ = t.order_by(t.img.similarity(string='red truck')).limit(1).collect()

        # Similarity fails when attempted on a snapshot
        t_s = pxt.create_snapshot(p('t_s'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Snapshot does not support indices'):
            _ = t_s.order_by(t_s.img.similarity(string='red truck')).limit(1).collect()

        # embedding() fails on a snapshot
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Snapshot does not support indices'):
            _ = t_s.select(t_s.img.embedding(idx='other_idx')).limit(2)

        t.drop_embedding_index(idx_name='idx0')
        t.drop_embedding_index(idx_name='idx1')
        t.add_embedding_index('split', string_embed=local_embed)
        sample_img = t.select(t.img).head(1)[0, 'img']
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.order_by(t.split.similarity(image=sample_img)).limit(1).collect()
        assert 'does not have an image embedding' in str(exc_info.value).lower()

    def test_add_index_after_drop(
        self, small_img_tbl: pxt.Table, make_catalog_path: Callable[[str], str], local_embed: pxt.Function
    ) -> None:
        """Test that an index with the same name can be added after the previous one is dropped"""
        p = make_catalog_path
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
        orig_res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        t.revert()
        # creating an index with the same name again after a revert should be successful
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
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
        t = pxt.get_table(p('small_img_tbl'))
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)

        # same should hold after a drop.
        t.drop_embedding_index(column='img')
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)
        t.drop_embedding_index(idx_name='clip_idx')
        reload_catalog()
        t = pxt.get_table(p('small_img_tbl'))
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
        res = (
            t.select(t.img.localpath)
            .order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False)
            .limit(3)
            .collect()
        )
        assert_resultset_eq(orig_res, res, True)

    def test_add_embedding_index_if_exists(
        self, small_img_tbl: pxt.Table, reload_tester: ReloadTester, local_embed: pxt.Function
    ) -> None:
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']

        def emb_idxs() -> dict[str, Any]:
            return {name: idx for name, idx in t.get_metadata()['indices'].items() if idx['index_type'] == 'embedding'}

        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed)
        assert set(emb_idxs()) == {'clip_idx'}

        # when index name is not provided, duplicates are detected by the index definition (embeddings + metric +
        # precision) on the column. local_embed here has the same definition as the named clip_idx above, so it
        # is a duplicate
        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match='identical embedding index'):
            t.add_embedding_index('img', embedding=local_embed, if_exists='error')
        assert set(emb_idxs()) == {'clip_idx'}

        # if_exists='ignore' makes the duplicate a no-op.
        t.add_embedding_index('img', embedding=local_embed, if_exists='ignore')
        assert set(emb_idxs()) == {'clip_idx'}

        # if_exists is now validated on the unnamed path too.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'if_exists must be one of'):
            t.add_embedding_index('img', embedding=local_embed, if_exists='invalid')  # type: ignore[arg-type]
        assert set(emb_idxs()) == {'clip_idx'}

        # an index that differs in metric is not a duplicate
        t.add_embedding_index('img', embedding=local_embed, metric='ip')
        # an index using a different embedding (bound via .using()) is not a duplicate
        t.add_embedding_index('img', embedding=local_embedding.using(dim=256))
        assert len(emb_idxs()) == 3

        # when index name is provided, if_exists parameter is applied.
        # invalid value is rejected.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed, if_exists='invalid')  # type: ignore[arg-type]
        assert (
            "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        )
        assert len(emb_idxs()) == 3

        # if_exists='error' (the default) raises if the index name already exists, regardless of whether the
        # definition matches: rejection on the named path is by name, not by definition. Vary the metric so each
        # attempt has a definition that differs from the existing cosine clip_idx.
        for metric in ('cosine', 'ip', 'l2'):
            with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match='Duplicate index name'):
                t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed, metric=metric)
            with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match='Duplicate index name'):
                t.add_embedding_index(
                    'img', idx_name='clip_idx', embedding=local_embed, metric=metric, if_exists='error'
                )
        assert len(emb_idxs()) == 3

        # if_exists='ignore' does nothing if the index name already exists: clip_idx keeps its position.
        assert 'clip_idx' in emb_idxs() and len(emb_idxs()) == 3
        assert emb_idxs()['clip_idx']['parameters']['metric'] == 'cosine'

        # if_exists='ignore' on an existing name is a no-op even when the new embedding would fail validation:
        # the name collision is resolved before the new index is constructed.
        t.add_embedding_index('img', idx_name='clip_idx', embedding=self.bad_embed, if_exists='ignore')
        assert 'clip_idx' in emb_idxs() and len(emb_idxs()) == 3

        # cannot use if_exists to ignore or replace an existing index
        # that is not an embedding (like, default btree indexes).
        btree_name = next(
            name
            for name, idx in t.get_metadata()['indices'].items()
            if idx['index_type'] == 'btree' and idx['columns'] == ['img']
        )
        for ie in ('ignore', 'replace', 'replace_force'):
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='not an embedding index'):
                t.add_embedding_index('img', idx_name=btree_name, embedding=local_embed, if_exists=ie)
        assert 'clip_idx' in emb_idxs() and len(emb_idxs()) == 3

        # if_exists='replace' replaces the existing index with the new one (here, with a different metric).
        t.add_embedding_index('img', idx_name='clip_idx', embedding=local_embed, metric='l2', if_exists='replace')
        assert 'clip_idx' in emb_idxs() and len(emb_idxs()) == 3
        assert emb_idxs()['clip_idx']['parameters']['metric'] == 'l2'

        # sanity check: use the replaced index to run a query.
        # use the index hint in similarity function to ensure clip_idx is used.
        _ = reload_tester.run_query(
            t.select(t.img.localpath).order_by(t.img.similarity(image=sample_img, idx='clip_idx'), asc=False).limit(3)
        )

        # sanity check persistence
        reload_tester.run_reload_test()

    @pytest.mark.local('TODO: convert')
    def test_unnamed_duplicate_detection(self, small_img_tbl: pxt.Table, local_embed: pxt.Function) -> None:
        t = small_img_tbl

        def emb_indexes() -> dict[str, Any]:
            return {name: idx for name, idx in t.get_metadata()['indices'].items() if idx['index_type'] == 'embedding'}

        t.add_embedding_index('img', embedding=local_embed)
        assert len(emb_indexes()) == 1
        orig_name = next(iter(emb_indexes()))

        # same definition with no idx_name -> duplicate, governed by if_exists
        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match='identical embedding index'):
            t.add_embedding_index('img', embedding=local_embed)
        t.add_embedding_index('img', embedding=local_embed, if_exists='ignore')
        assert set(emb_indexes()) == {orig_name}

        # replace drops the single matching index and recreates it under a fresh name
        t.add_embedding_index('img', embedding=local_embed, if_exists='replace')
        assert len(emb_indexes()) == 1 and orig_name not in emb_indexes()

        # a different embedding bound via .using() is not a duplicate. Function == compares only self_path, so this
        # guards against comparing function identity instead of the serialized definition.
        t.add_embedding_index('img', embedding=local_embedding.using(dim=256))
        # a different metric is not a duplicate
        t.add_embedding_index('img', embedding=local_embed, metric='ip')
        assert len(emb_indexes()) == 3

        # duplicate detection is per-column: the same embedding on a different column is allowed
        t.add_embedding_index('category', string_embed=local_embed)
        assert len(emb_indexes()) == 4

        # duplicate detection still works against indexes reconstructed from stored metadata after a reload
        reload_catalog()
        t = pxt.get_table('small_img_tbl')
        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match='identical embedding index'):
            t.add_embedding_index('category', string_embed=local_embed)

    def test_update_img(
        self,
        img_tbl: pxt.Table,
        test_tbl: pxt.Table,
        make_catalog_path: Callable[[str], str],
        reload_tester: ReloadTester,
    ) -> None:
        p = make_catalog_path
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        short_rows = rows[:5]
        new_rows: list[dict[str, Any]] = []
        for n, row in enumerate(short_rows):
            row['pkey'] = n
            new_rows.append(row)

        # create table with fewer rows to speed up testing
        schema = {'pkey': pxt.Required[pxt.Int], 'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = p('update_test')
        img_t = pxt.create_table(tbl_name, schema, primary_key='pkey')
        img_t.insert(new_rows)
        print(img_t.head())

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='property of another column is not allowed'):
            img_t.add_computed_column(emsg=img_t.img.errormsg)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='property of another column is not allowed'):
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
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is a media column and cannot be updated'):
            img_t.batch_update([repl_row], cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

        # Update the row again, looking for an error
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is a media column and cannot be updated'):
            img_t.batch_update([repl_row], cascade=True)
        print(img_t.select(img_t.pkey, img_t.img).collect())

    def test_embedding_access(
        self, img_tbl: pxt.Table, make_catalog_path: Callable[[str], str], local_embed: pxt.Function
    ) -> None:
        p = make_catalog_path
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = p('access_test')
        img_t = pxt.create_table(tbl_name, schema)
        img_t.insert(rows[:5])

        # Add computed column based on the other_idx embedding index
        img_t.add_embedding_index(img_t.category, idx_name='cat_idx', string_embed=local_embed)
        img_t.add_computed_column(ebd_copy=img_t.category.embedding(idx='cat_idx'))
        img_t.insert([rows[6]])

        # Attempt to drop the embedding index
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            match="Cannot drop index 'cat_idx' because the following columns depend on it",
        ):
            img_t.drop_embedding_index(column=img_t.category)

        img_t.add_computed_column(sim=img_t.category.similarity(string='red_truck', idx='cat_idx'))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be used in a computed column'):
            img_t.insert([rows[7]])

        img_t.drop_column('sim')
        img_t.drop_column('ebd_copy')
        img_t.drop_embedding_index(column=img_t.category)

    def test_embedding_basic(
        self,
        img_tbl: pxt.Table,
        make_catalog_path: Callable[[str], str],
        local_embed: pxt.Function,
        reload_tester: ReloadTester,
    ) -> None:
        p = make_catalog_path
        skip_test_if_not_installed('imagehash')

        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
        tbl_name = p('index_test')
        img_t = pxt.create_table(tbl_name, schema)
        img_t.insert(rows[:30])
        dummy_img_t = pxt.create_table(p('dummy'), schema)
        dummy_img_t.insert(rows[:10])

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND) as exc_info:
            # cannot pass another table's column reference
            img_t.add_embedding_index(dummy_img_t.img, embedding=local_embed)
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        img_t.add_embedding_index('img', embedding=local_embed)

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND) as exc_info:
            # cannot pass another table's column reference
            img_t.drop_embedding_index(column=dummy_img_t.img)
        assert 'unknown column: dummy.img' in str(exc_info.value).lower()

        # predicates on media columns that have both a B-tree and an embedding index still work
        res = img_t.where(img_t.img == rows[0]['img']).collect()
        assert len(res) == 1

        res = img_t.select(img_t.img.embedding()).limit(2).collect()
        assert len(res) == 2
        assert isinstance(res[0, 0], np.ndarray)

        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=local_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()
        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS) as exc_info:
            img_t.add_embedding_index(img_t.img, idx_name='idx0', image_embed=local_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        img_t.add_embedding_index(img_t.category, idx_name='cat_idx', string_embed=local_embed)

        # revert() removes the index
        img_t.revert()
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
            img_t.drop_embedding_index(column='category')
        assert 'does not have an index' in str(exc_info.value).lower()
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
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
        query = img_t.select().order_by(img_t.img)
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
        img_t.add_embedding_index(img_t.img, idx_name='other_idx', embedding=local_embed)
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
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='has multiple embedding indices'):
            _ = img_t.select(img_t.img.embedding()).collect()

        # Adding an index with an invalid index name fails
        with pxt_raises(pxt.ErrorCode.INVALID_COLUMN_NAME, match='Invalid column name'):
            img_t.add_embedding_index(img_t.img, idx_name='BOGUS COL NAME', embedding=local_embed)

        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
            _ = img_t.img.similarity(string='red truck', idx='doesnotexist')
        assert "index 'doesnotexist' not found" in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            img_t.drop_embedding_index(column='img')
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert "column 'img' has multiple indices" in str(exc_info.value).lower()
        img_t.drop_embedding_index(idx_name='other_idx')

        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
            sim = img_t.img.similarity(string='red truck', idx='other_idx')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()
        assert "index 'other_idx' not found" in str(exc_info.value).lower()

        img_t.drop_embedding_index(column=img_t.img)
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
            img_t.drop_embedding_index(column=img_t.img)
        assert 'does not have an index' in str(exc_info.value).lower()

        # revert() makes the index reappear
        img_t.revert()
        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS) as exc_info:
            img_t.add_embedding_index('img', idx_name='idx0', image_embed=local_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND) as exc_info:
            img_t.drop_embedding_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

        _ = reload_tester.run_query(img_t.select())

    def test_view_indices(
        self, make_catalog_path: Callable[[str], str], local_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        p = make_catalog_path
        # Create a base table
        t = pxt.create_table(p('t1'), {'n': pxt.Int, 's': pxt.String})
        sentences = get_sentences(20)
        status = t.insert({'n': i, 's': s} for i, s in enumerate(sentences))
        validate_update_status(status, 20)

        # Create a view that indexes the base table column
        v = pxt.create_view(p('v'), t.where(t.n % 2 == 0))
        v.add_embedding_index('s', string_embed=local_embed)

        query1 = v.select(sim1=v.s.similarity(string=sentences[1])).order_by(v.n)
        res1 = reload_tester.run_query(query1)

        # Now add an index to the base table, which should be independent of the view index
        t.add_embedding_index('s', string_embed=local_embed)
        query2 = t.where(t.n % 2 == 0).select(sim2=t.s.similarity(string=sentences[1])).order_by(t.n)
        res2 = reload_tester.run_query(query2)

        # Now query the view again twice: once with the column referenced as `v.s`, and once as `t.s`
        query3 = v.select(sim3=v.s.similarity(string=sentences[1])).order_by(v.n)
        res3 = reload_tester.run_query(query3)
        query4 = v.select(sim4=t.s.similarity(string=sentences[1])).order_by(v.n)
        res4 = reload_tester.run_query(query4)

        # `v.s` should use the view index, while `t.s` should use the base table index
        assert_resultset_eq(res1, res3)
        assert_resultset_eq(res2, res4)

        reload_tester.run_reload_test()

    def test_embedding_errors(self, small_img_tbl: pxt.Table, test_tbl: pxt.Table, local_embed: pxt.Function) -> None:
        img_t = small_img_tbl

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            img_t.add_embedding_index('img', metric='badmetric', image_embed=local_embed)  # type: ignore[arg-type]
        assert 'invalid metric badmetric' in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND) as exc_info:
            # unknown column
            img_t.add_embedding_index('does_not_exist', idx_name='idx0', image_embed=local_embed)
        assert 'Unknown column: does_not_exist' in str(exc_info.value)

        with pxt_raises(
            pxt.ErrorCode.MISSING_REQUIRED,
            match=r'`embed`, `string_embed`, `image_embed`, `audio_embed`, `video_embed`, or `document_embed` '
            'must be specified',
        ):
            # no embedding function specified
            img_t.add_embedding_index('img')

        with pxt_raises(
            pxt.ErrorCode.TYPE_MISMATCH, match=r"Type `Int` of column 'c2' is not a valid type for an embedding index."
        ):
            # wrong column type
            test_tbl.add_embedding_index('c2', image_embed=local_embed)

        with pxt_raises(
            pxt.ErrorCode.TYPE_MISMATCH,
            match=r"The specified embedding function does not support the type `Image` of column 'img'.",
        ):
            # missing embedding function
            img_t.add_embedding_index('img', string_embed=local_embed)

        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION) as exc_info:
            # wrong signature
            img_t.add_embedding_index('img', image_embed=clip)
        assert 'must take a single image parameter' in str(exc_info.value).lower()

        with pxt_raises(
            pxt.ErrorCode.TYPE_MISMATCH,
            match=r"The specified embedding function does not support the type `String` of column 'category'.",
        ):
            # missing embedding function
            img_t.add_embedding_index('category', image_embed=local_embed)

        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION) as exc_info:
            # wrong signature
            img_t.add_embedding_index('category', string_embed=clip)
        assert 'must take a single string parameter' in str(exc_info.value).lower()

        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match=r'The function `clip` is not a valid embedding: '
            'it must take a single string, image, audio, video, or document parameter',
        ):
            # no matching signature
            img_t.add_embedding_index('img', embedding=clip)

        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed)
        assert 'must return an array' in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION) as exc_info:
            img_t.add_embedding_index('category', string_embed=self.bad_embed2)
        assert 'must return a 1-dimensional array of a specific length' in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED) as exc_info:
            img_t.drop_embedding_index()
        assert "exactly one of 'column' or 'idx_name' must be provided" in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="Index 'doesnotexist' does not exist"):
            img_t.drop_embedding_index(idx_name='doesnotexist')
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="Index 'doesnotexist' does not exist"):
            img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='error')

        img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='ignore')
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            img_t.drop_embedding_index(idx_name='doesnotexist', if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column: doesnotexist'):
            img_t.drop_embedding_index(column='doesnotexist')
        # when dropping an index via a column, if_not_exists does not
        # apply to non-existent column; it will still raise error.
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column: doesnotexist'):
            img_t.drop_embedding_index(column='doesnotexist', if_not_exists='invalid')  # type: ignore[arg-type]
        with pytest.raises(AttributeError) as exc_info:
            img_t.drop_embedding_index(column=img_t.doesnotexist)
        assert 'Unknown column: doesnotexist' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column='img')
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column=img_t.img)
        # when dropping an index via a column, if_not_exists applies if
        # the column does not have any index to drop.
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match="Column 'img' does not have an index"):
            img_t.drop_embedding_index(column='img', if_not_exists='error')
        img_t.drop_embedding_index(column=img_t.img, if_not_exists='ignore')

        img_t.add_embedding_index('img', idx_name='embed0', embedding=local_embed)
        img_t.add_embedding_index('img', idx_name='embed1', embedding=local_embed)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'img' has multiple indices"):
            img_t.drop_embedding_index(column='img')
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'img' has multiple indices"):
            img_t.drop_embedding_index(column=img_t.img)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'img' has multiple embedding indices"):
            sim = img_t.img.similarity(string='red truck')
            _ = img_t.order_by(sim, asc=False).limit(1).collect()

        if not Env.get().is_using_cockroachdb:
            # TODO(PXT-941): Revisit embedding index precision behavior for cloud launch
            # In CockroachDB we use VECTOR type that doesn't have the same limitation as pgvector's VECTOR and HALFVEC
            with pxt_raises(
                pxt.ErrorCode.INVALID_ARGUMENT,
                match="Embedding index's vector dimensionality 4001 exceeds maximum of 4000 for fp16 precision",
            ):
                test_tbl.add_embedding_index(test_tbl.c1, embedding=local_embedding.using(dim=4001), precision='fp16')
            with pxt_raises(
                pxt.ErrorCode.INVALID_ARGUMENT,
                match="Embedding index's vector dimensionality 2001 exceeds maximum of 2000 for fp32 precision",
            ):
                test_tbl.add_embedding_index(test_tbl.c1, embedding=local_embedding.using(dim=2001), precision='fp32')

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"Invalid precision.+Must be one of: \['fp16', 'fp32'\]"):
            test_tbl.add_embedding_index(
                test_tbl.c1,
                embedding=local_embedding.using(dim=2001),
                precision='invalid',  # type: ignore[arg-type]
            )
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match='is not a valid embedding: it returns an array of invalid length 0',
        ):
            test_tbl.add_embedding_index(test_tbl.c1, embedding=local_embedding.using(dim=0), precision='fp16')

    def run_btree_test(self, p: Callable[[str], str], data: list, data_type: type | _GenericAlias) -> pxt.Table:
        t = pxt.create_table(p('btree_test'), {'data': data_type})
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

    def test_int_btree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        random.seed(1)
        data = [random.randint(0, 2**63 - 1) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(p, data, pxt.Int)

    def test_float_btree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        random.seed(1)
        data = [random.uniform(0, sys.float_info.max) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(p, data, pxt.Float)

    def test_string_btree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path

        def create_random_str(n: int) -> str:
            chars = string.ascii_letters + string.digits
            return ''.join(random.choice(chars) for _ in range(n))

        random.seed(1)
        # create random strings of length 200-300 characters
        data = [create_random_str(200 + i % 100) for i in range(self.BTREE_TEST_NUM_ROWS)]
        t = self.run_btree_test(p, data, pxt.String)

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

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            assert len(data[56]) == 256
            _ = t.where(t.data == data[56]).count()
        assert 'String literal too long' in str(exc_info.value)

        # test that Comparison uses BtreeIndex.MAX_STRING_LEN
        t = pxt.create_table(p('test_max_str_len'), {'data': pxt.String})
        rows = [{'data': s}, {'data': s + 'a'}]
        validate_update_status(t.insert(rows), expected_rows=len(rows))
        assert t.where(t.data >= s).count() == 2
        assert t.where(t.data > s).count() == 1

    def test_timestamp_btree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        random.seed(1)
        start = datetime.datetime(2000, 1, 1)
        end = datetime.datetime(2020, 1, 1)
        delta = end - start
        delta_secs = int(delta.total_seconds())
        data = [
            start + datetime.timedelta(seconds=random.randint(0, int(delta_secs)))
            for _ in range(self.BTREE_TEST_NUM_ROWS)
        ]
        self.run_btree_test(p, data, pxt.Timestamp)

    def test_date_btree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        random.seed(1)
        start = datetime.date(2000, 1, 1)
        end = datetime.date(2100, 1, 1)
        delta = end - start
        delta_days = int(delta.days)
        assert delta_days > 3 * self.BTREE_TEST_NUM_ROWS
        data = [
            start + datetime.timedelta(days=random.randint(0, int(delta_days))) for _ in range(self.BTREE_TEST_NUM_ROWS)
        ]
        self.run_btree_test(p, data, pxt.Date)

    @pytest.mark.parametrize('reload_cat', [True, False], ids=['reload_cat', 'no_reload_cat'])
    @pytest.mark.parametrize('metric', ['l2', 'cosine', 'ip'])
    @pytest.mark.parametrize('precision', ['fp16', 'fp32'])
    def test_embedding_index_precision(
        self,
        make_catalog_path: Callable[[str], str],
        reload_cat: bool,
        metric: Literal['cosine', 'ip', 'l2'],
        precision: Literal['fp16', 'fp32'],
    ) -> None:
        p = make_catalog_path
        t = pxt.create_table(p('test'), {'rowid': pxt.Int, 'text': pxt.String}, if_exists='replace')
        n = 123
        t.add_embedding_index(
            t.text, embedding=local_embedding.using(dim=n), metric=metric, precision=precision, idx_name='test_idx'
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
        assert ts.ArrayType((n,), np.dtype('float32')).matches(res._schema['col_0']), res._schema
        assert len(res) == 3
        assert all(isinstance(row['col_0'], np.ndarray) and row['col_0'].dtype == np.float32 for row in res)

        sim = t.text.similarity(string='zero')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 0

        sim = t.text.similarity(string='one')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 1

    def test_array_column_embedding_index(
        self, make_catalog_path: Callable[[str], str], local_embed: pxt.Function, reload_tester: ReloadTester
    ) -> None:
        p = make_catalog_path
        texts = ['a dog playing in the park', 'a cat sitting on a mat', 'a bird flying in the sky']

        t = pxt.create_table(p('array_embedding_test'), {'id': pxt.Int, 'text': pxt.String})
        validate_update_status(t.insert([{'id': i, 'text': s} for i, s in enumerate(texts)]), expected_rows=3)

        precomputed_embeddings = t.order_by(t.id).select(emb=local_embed(t.text)).collect()['emb']
        dim = len(precomputed_embeddings[0])
        precomputed_embeddings_f64 = [v.astype(np.float64) for v in precomputed_embeddings]

        t.add_column(precomputed_embeddings=pxt.Array[(dim,), np.float32])
        t.add_column(precomputed_embeddings_f64=pxt.Array[(dim,), np.float64])

        for i in range(len(texts)):
            validate_update_status(
                t.where(t.id == i).update(
                    {
                        'precomputed_embeddings': precomputed_embeddings[i],
                        'precomputed_embeddings_f64': precomputed_embeddings_f64[i],
                    }
                ),
                expected_rows=1,
            )

        t.add_computed_column(embedding=local_embed(t.text))

        # embedding index on text column
        t.add_embedding_index('text', idx_name='emd_idx_text', embedding=local_embed, metric='cosine', precision='fp32')

        # embedding index on computed column
        t.add_embedding_index(
            'embedding', idx_name='emb_idx_computed', embedding=local_embed, metric='cosine', precision='fp32'
        )

        # f32 precomputed embedding with string embedding function
        t.add_embedding_index(
            'precomputed_embeddings',
            idx_name='emb_idx_stored',
            string_embed=local_embed,
            metric='cosine',
            precision='fp32',
        )

        # f64 precomputed embedding column without any embedding function, should be searchable by vector
        t.add_embedding_index(
            'precomputed_embeddings_f64', idx_name='emb_idx_stored64', metric='cosine', precision='fp32'
        )

        best = 'a cat sitting on a mat'
        best_vec = precomputed_embeddings[texts.index(best)]

        for col_name, idx_name, has_string_embed_fn in [
            ('text', 'emd_idx_text', True),
            ('embedding', 'emb_idx_computed', True),
            ('precomputed_embeddings', 'emb_idx_stored', True),
            ('precomputed_embeddings_f64', 'emb_idx_stored64', False),
        ]:
            col = getattr(t, col_name)

            # search by string/text
            if has_string_embed_fn:
                sim = col.similarity(string='a cat', idx=idx_name)
                query = t.select(t.id, t.text, sim=sim).order_by(sim, asc=False).limit(3)
                res = reload_tester.run_query(query)
                assert len(res) == 3, col_name
                assert res[0]['text'] == best, col_name
                sim_vals = [r['sim'] for r in res]
                assert all(sim_vals[i] >= sim_vals[i + 1] for i in range(len(sim_vals) - 1)), (
                    f'{col_name}:{idx_name}: similarity scores must be descending; got {sim_vals}'
                )

            # search by embedding vector; should work with all embedding indices
            sim = col.similarity(vector=best_vec, idx=idx_name)
            query = t.select(t.id, t.text, sim=sim).order_by(sim, asc=False).limit(3)
            res = reload_tester.run_query(query)
            assert len(res) == 3
            assert res[0]['text'] == best
            sim_vals = [r['sim'] for r in res]
            assert all(sim_vals[i] >= sim_vals[i + 1] for i in range(len(sim_vals) - 1)), (
                f'precomputed_embeddings({idx_name}): similarity scores must be descending; got {sim_vals}'
            )

        reload_tester.run_reload_test()

    @staticmethod
    @pxt.udf
    def _embed_wrong_shape(x: str) -> pxt.Array[(256,), np.float32]:
        return np.zeros(256, dtype=np.float32)

    def test_array_embedding_index_validation_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        t = pxt.create_table(
            p('arr_val_test'),
            {'id': pxt.Int, 'vec': pxt.Array[(384,), np.float32], 'vec2d': pxt.Array[(10, 10), np.float32]},
            if_exists='replace',
        )
        t.insert([{'id': 0, 'vec': np.zeros(384, dtype=np.float32), 'vec2d': np.zeros((10, 10), dtype=np.float32)}])

        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='shape'):
            t.add_embedding_index('vec', embedding=self._embed_wrong_shape)

        # 2D array columns should be rejected
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='1-dimensional'):
            t.add_embedding_index('vec2d')

        # a non-Function embedding argument is rejected
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='`string_embed` must be a Pixeltable function'):
            t.add_embedding_index('vec', string_embed=str.split)  # type: ignore[arg-type]

    @pytest.mark.parametrize('index_type', ['btree', 'embedding'])
    def test_drop_index(
        self,
        index_type: str,
        make_catalog_path: Callable[[str], str],
        catalog_mode: CatalogMode,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test that indices (B-tree and embedding) are properly dropped, observed through get_metadata(); the
        physical removal from the local Postgres store is additionally checked in local mode."""
        p = make_catalog_path
        t = pxt.create_table(p('index_drop_test'), {'id': pxt.Int, 'text': pxt.String}, if_exists='replace')
        t.insert([{'id': 1, 'text': 'hello world'}, {'id': 2, 'text': 'goodbye'}])

        # Find or create an index to drop, identified by name through the public metadata
        if index_type == 'btree':
            # the table auto-creates a btree index on 'text'
            btree_names = [
                name
                for name, info in t.get_metadata()['indices'].items()
                if info['index_type'] == 'btree' and 'text' in info['columns']
            ]
            assert len(btree_names) == 1, "Should have one B-tree index on 'text'"
            idx_name = btree_names[0]
        else:
            local_embed = request.getfixturevalue('local_embed')
            t.add_embedding_index('text', idx_name='text_idx', string_embed=local_embed)
            idx_name = 'text_idx'

        assert idx_name in t.get_metadata()['indices']

        # the physical Postgres index lives only in the (local) store; verify it there in local mode
        if catalog_mode == 'local':
            idx_info = t._tbl_version.get().idxs_by_name[idx_name]
            store_idx_name = t._tbl_version.get()._store_idx_name(idx_info.id)
            assert store_idx_name in list_store_indexes(t), f'Index {store_idx_name} should exist before drop'

        # Drop it
        if index_type == 'btree':
            t.drop_index(column='text')
        else:
            t.drop_embedding_index(idx_name='text_idx')

        assert idx_name not in t.get_metadata()['indices']
        if catalog_mode == 'local':
            assert store_idx_name not in list_store_indexes(t), f'Index {store_idx_name} should not exist after drop'
        reload_catalog()
        t = pxt.get_table(p('index_drop_test'))
        assert idx_name not in t.get_metadata()['indices']

    def test_similarity_index_lifecycle(
        self, make_catalog_path: Callable[[str], str], local_embed: pxt.Function
    ) -> None:
        """Test similarity when index is dropped, recreated, and column is dropped."""
        p = make_catalog_path
        t = pxt.create_table(p('lifecycle_test'), {'id': pxt.Int, 'text': pxt.String})
        texts = ['a dog playing in the park', 'a cat sitting on a mat', 'a bird flying in the sky']
        validate_update_status(t.insert([{'id': i, 'text': s} for i, s in enumerate(texts)]), expected_rows=3)
        t.add_embedding_index('text', idx_name='emb_idx', string_embed=local_embed)

        sim = t.text.similarity(string='a cat', idx='emb_idx')
        query = t.select(t.id, t.text, sim=sim).order_by(sim, asc=False).limit(3)
        res = query.collect()
        assert res[0]['text'] == 'a cat sitting on a mat'

        # verify query works after catalog reload
        reload_catalog()
        t = pxt.get_table(p('lifecycle_test'))
        res = query.collect()
        assert res[0]['text'] == 'a cat sitting on a mat'

        # drop index: query should fail with a clear error
        t.drop_embedding_index(idx_name='emb_idx')
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND, match=r"(?i).*No embedding index found for column 'text'.*"):
            query.collect()

        # recreate index under same name: query should work again
        t.add_embedding_index('text', idx_name='emb_idx', string_embed=local_embed)
        res = query.collect()
        assert res[0]['text'] == 'a cat sitting on a mat'

        # verify it still works after reload with recreated index
        reload_catalog()
        t = pxt.get_table(p('lifecycle_test'))
        res = query.collect()
        assert res[0]['text'] == 'a cat sitting on a mat'

        # drop the column: query should fail
        t.drop_column('text')
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match=r'(?i).*column was dropped.*'):
            query.collect()

    @pytest.mark.parametrize('reload', [True, False], ids=['reload', 'noreload'])
    def test_similarity_column_snapshot(
        self, make_catalog_path: Callable[[str], str], mpnet_or_local: tuple[pxt.Function, bool], reload: bool
    ) -> None:
        """Tests various edge cases that involve similarity columns and snapshots"""

        embed, is_dummy_model = mpnet_or_local
        if not is_dummy_model:
            skip_test_if_not_installed('sentence_transformers')

        def check(cond: bool) -> None:
            # text-ordering results depend on real-model semantics; only assert them on the very_expensive tier
            if not is_dummy_model:
                assert cond

        p = make_catalog_path
        tbl = pxt.create_table(p('test'), {'text': pxt.String, 'text2': pxt.String})
        tbl.insert([{'text': s, 'text2': s} for s in get_sentences(10)])

        # add a stored similarity column
        tbl.add_embedding_index('text', string_embed=embed, idx_name='embed')
        validate_update_status(tbl.add_computed_column(sim=tbl.text.similarity(string='sunlight', idx='embed')))

        # add an unstored similarity column
        tbl.add_embedding_index('text2', string_embed=embed, idx_name='embed2')
        validate_update_status(
            tbl.add_computed_column(sim_unstored=tbl.text2.similarity(string='sunlight'), stored=False)
        )

        # check the stored sim column in the base table
        res = tbl.select(tbl.text, tbl.sim).order_by(tbl.sim, asc=False).collect()
        check('sunshine' in res[0]['text'])
        check('winter' in res[1]['text'])

        # check the unstored sim column in the base table
        res = tbl.select(tbl.text).order_by(tbl.sim_unstored, asc=False).collect()
        check('sunshine' in res[0]['text'])

        snap = pxt.create_snapshot(p('snap'), tbl)

        # check the stored sim column in the snapshot
        res = snap.select(snap.text, snap.sim).order_by(snap.sim, asc=False).collect()
        check('sunshine' in res[0]['text'])
        check('winter' in res[1]['text'])

        # check the unstored sim column in the snapshot (should raise)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Snapshot does not support indices'):
            snap.select(snap.text).order_by(snap.sim_unstored, asc=False).collect()

        # Drop the top row by similarity, then verify similarity results in the base table and the snapshot
        validate_update_status(tbl.delete(where=pxtf.string.contains(tbl.text, 'sunshine')), expected_rows=1)

        res = tbl.select(tbl.text, tbl.sim).order_by(tbl.sim, asc=False).collect()
        check('winter' in res[0]['text'])

        res = snap.select(snap.text, snap.sim).order_by(snap.sim, asc=False).collect()
        check('sunshine' in res[0]['text'])

        # Drop the value and the similarity columns from base table, then verify that the snapshot still has them
        tbl.drop_column(tbl.sim)
        tbl.drop_column(tbl.text)

        reload_catalog(reload)

        snap = pxt.get_table(p('snap'))
        res = snap.select(snap.text, snap.sim).order_by(snap.sim, asc=False).collect()
        check('sunshine' in res[0]['text'])
        check('winter' in res[1]['text'])

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Snapshot does not support indices'):
            snap.select(snap.text).order_by(snap.sim_unstored, asc=False).collect()
