import PIL.Image
import numpy as np
import pytest

import pixeltable as pxt
from pixeltable.functions.huggingface import clip_image, clip_text
from .utils import clip_text_embed, clip_img_embed, skip_test_if_not_installed, assert_img_eq, e5_embed, reload_catalog


class TestIndex:

    # returns string
    @pxt.udf
    def bad_embed(x: str) -> str:
        return x

    # returns array w/o size
    @pxt.udf(return_type=pxt.ArrayType((None,), dtype=pxt.FloatType()))
    def bad_embed2(x: str) -> np.ndarray:
        return np.zeros(10)

    def test_search(self, small_img_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        for metric, is_asc in [('cosine', False), ('ip', False), ('l2', True)]:
            t.add_embedding_index('img', metric=metric, img_embed=clip_img_embed, text_embed=clip_text_embed)

            res = t.select(img=t.img, sim=t.img.similarity(sample_img))\
                .order_by(t.img.similarity(sample_img), asc=is_asc)\
                .limit(1).collect()
            out_img = res[0, 'img']
            assert_img_eq(sample_img, out_img), f'{metric} failed'

            # TODO:  how to verify the output?
            _ = t.select(path=t.img.localpath, sim=t.img.similarity('parachute')) \
                .order_by(t.img.similarity('parachute'), asc=is_asc) \
                .limit(1).collect()

            t.drop_index(column_name='img')

    @pxt.query
    def top_k_chunks(t: pxt.Table, query_text: str) -> pxt.DataFrame:
        return t.select(t.text, sim=t.text.similarity(query_text))\
            .order_by(t.text.similarity(query_text), asc=False)\
            .limit(5)

    def test_query(self, reset_db) -> None:
        queries = pxt.create_table('queries', schema={'query_text': pxt.StringType()}, )
        queries.insert([{'query_text': 'how much is the stock of AI companies up?'}, {'query_text': 'what happened to the term machine learning?'}])

        test_doc_chunks = pxt.create_table('test_doc_chunks', schema={'text': pxt.StringType()})
        test_doc_chunks.insert([{'text': 'the stock of artificial intelligence companies is up 1000%'},
                            {'text': 'the term machine learning has fallen out of fashion now that AI has been rehabilitated and is now the new hotness'},
                            {'text': 'machine learning is a subset of artificial intelligence'},
                            {'text': 'gas car companies are in danger of being left behind by electric car companies'},
        ])
        test_doc_chunks.add_embedding_index(col_name='text', text_embed=clip_text_embed)
        _ = queries.select(queries.query_text, out=test_doc_chunks.query(self.top_k_chunks)(queries.query_text)).collect()
        queries.add_column(chunks=test_doc_chunks.query(self.top_k_chunks)(queries.query_text))
        _ = queries.collect()
        reload_catalog()
        queries = pxt.get_table('queries')
        _ = queries.collect()
        pass


    def test_search_fn(self, small_img_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        t = small_img_tbl
        sample_img = t.select(t.img).head(1)[0, 'img']
        _ = t.select(t.img.localpath).collect()

        t.add_embedding_index('img', metric='cosine', img_embed=clip_img_embed, text_embed=clip_text_embed)
        _ =  t.select(t.img.localpath).order_by(t.img.similarity(sample_img), asc=False).limit(3).collect()

        @pxt.query
        def img_matches(t: pxt.Table, img: PIL.Image.Image):
            return t.select(t.img.localpath).order_by(t.img.similarity(img), asc=False).limit(3)

        res = list(t.select(img=t.img.localpath, matches=t.query(img_matches)(t.img)).head(1))

    def test_search_errors(self, indexed_img_tbl: pxt.Table, small_img_tbl: pxt.Table) -> None:
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
        t.add_embedding_index('img', img_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert 'was created without the text_embed parameter' in str(exc_info.value).lower()

        t.add_embedding_index('img', text_embed=clip_text_embed, img_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert 'column img has multiple indices' in str(exc_info.value).lower()

        t.drop_index(idx_name='idx0')
        t.drop_index(idx_name='idx1')
        t.add_embedding_index('split', text_embed=clip_text_embed)
        sample_img = t.select(t.img).head(1)[0, 'img']
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.split.similarity(sample_img)).limit(1).collect()
        assert 'was created without the img_embed parameter' in str(exc_info.value).lower()

    def test_embedding_basic(self, img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        schema = {
            'img': pxt.ImageType(nullable=False),
            'category': pxt.StringType(nullable=False),
            'split': pxt.StringType(nullable=False),
        }
        tbl_name = 'index_test'
        img_t = pxt.create_table(tbl_name, schema=schema)
        img_t.insert(rows[:30])

        img_t.add_embedding_index('img', img_embed=clip_img_embed, text_embed=clip_text_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        img_t.add_embedding_index('category', text_embed=e5_embed)

        # revert() removes the index
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(column_name='category')
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
        reload_catalog()
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

        img_t.drop_index(idx_name='idx0')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(column_name='img')
        assert 'does not have an index' in str(exc_info.value).lower()

        # revert() makes the index reappear
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

    def test_errors(self, small_img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        img_t = small_img_tbl

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', metric='badmetric', img_embed=clip_img_embed)
        assert 'invalid metric badmetric' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # unknown column
            img_t.add_embedding_index('does_not_exist', idx_name='idx0', img_embed=clip_img_embed)
        assert 'column does_not_exist unknown' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong column type
            test_tbl.add_embedding_index('c2', img_embed=clip_img_embed)
        assert 'requires string or image column' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_t.add_embedding_index('img', text_embed=clip_text_embed)
        assert 'image embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('img', img_embed=clip_image)
        assert 'but has signature' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_t.add_embedding_index('category', img_embed=clip_img_embed)
        assert 'text embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_t.add_embedding_index('category', text_embed=clip_text)
        assert 'but has signature' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', text_embed=self.bad_embed)
        assert 'must return an array' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('category', text_embed=self.bad_embed2)
        assert 'must return a 1d array of a specific length' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index()
        assert 'exactly one of column_name or idx_name must be provided' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(idx_name='doesnotexist')
        assert 'index doesnotexist does not exist' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(column_name='doesnotexist')
        assert 'column doesnotexist unknown' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(column_name='img')
        assert 'column img does not have an index' in str(exc_info.value).lower()

        img_t.add_embedding_index('img', img_embed=clip_img_embed)
        img_t.add_embedding_index('img', img_embed=clip_img_embed)

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(column_name='img')
        assert 'column img has multiple indices' in str(exc_info.value).lower()
