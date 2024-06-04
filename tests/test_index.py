import sys
from datetime import datetime, timedelta

import numpy as np
import string
import random
import pytest

import pixeltable as pxt
from pixeltable.functions.huggingface import clip_image, clip_text
from .utils import clip_text_embed, clip_img_embed, skip_test_if_not_installed, assert_img_eq, e5_embed, \
    reload_catalog, validate_update_status


class TestIndex:

    # returns string
    @pxt.udf
    def bad_embed(x: str) -> str:
        return x

    # returns array w/o size
    @pxt.udf(return_type=pxt.ArrayType((None,), dtype=pxt.FloatType()))
    def bad_embed2(x: str) -> np.ndarray:
        return np.zeros(10)

    def test_similarity(self, small_img_tbl: pxt.Table) -> None:
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

            t.drop_embedding_index(column_name='img')

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
        t.add_embedding_index('img', img_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert 'was created without the text_embed parameter' in str(exc_info.value).lower()

        t.add_embedding_index('img', text_embed=clip_text_embed, img_embed=clip_img_embed)
        with pytest.raises(pxt.Error) as exc_info:
            _ = t.order_by(t.img.similarity('red truck')).limit(1).collect()
        assert 'column img has multiple indices' in str(exc_info.value).lower()

        t.drop_embedding_index(idx_name='idx0')
        t.drop_embedding_index(idx_name='idx1')
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

        # predicates on media columns that have both a B-tree and an embedding index still work
        res = img_t.where(img_t.img == rows[0]['img']).collect()
        assert len(res) == 1

        with pytest.raises(pxt.Error) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        img_t.add_embedding_index('category', text_embed=e5_embed)

        # revert() removes the index
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column_name='category')
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

        img_t.drop_embedding_index(column_name='img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column_name='img')
        assert 'does not have an index' in str(exc_info.value).lower()

        # revert() makes the index reappear
        img_t.revert()
        with pytest.raises(pxt.Error) as exc_info:
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=clip_img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

    def test_embedding_errors(self, small_img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
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
            img_t.drop_embedding_index()
        assert 'exactly one of column_name or idx_name must be provided' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(idx_name='doesnotexist')
        assert 'index doesnotexist does not exist' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column_name='doesnotexist')
        assert 'column doesnotexist unknown' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column_name='img')
        assert 'column img does not have an index' in str(exc_info.value).lower()

        img_t.add_embedding_index('img', img_embed=clip_img_embed)
        img_t.add_embedding_index('img', img_embed=clip_img_embed)

        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_embedding_index(column_name='img')
        assert 'column img has multiple indices' in str(exc_info.value).lower()

    def run_btree_test(self, data: list, data_type: pxt.ColumnType) -> pxt.Table:
        t = pxt.create_table('btree_test', schema={'data': data_type})
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
        self.run_btree_test(data, pxt.IntType())

    def test_float_btree(self, reset_db) -> None:
        random.seed(1)
        data = [random.uniform(0, sys.float_info.max) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        self.run_btree_test(data, pxt.FloatType())

    def test_string_btree(self, reset_db) -> None:
        def create_random_str(n: int) -> str:
            chars = string.ascii_letters + string.digits
            return ''.join(random.choice(chars) for _ in range(n))

        random.seed(1)
        # create random strings of length 200-300 characters
        data = [create_random_str(200 + i % 100) for i in range(self.BTREE_TEST_NUM_ROWS)]
        t = self.run_btree_test(data, pxt.StringType())

        # edge cases: strings that are at and above the max length
        sorted_data = sorted(data)
        # the index of the first string of length 255
        idx = next(i for i, s in enumerate(sorted_data) if len(s) == 255)
        assert t.where(t.data == sorted_data[idx]).count() == 1
        assert t.where(t.data <= sorted_data[idx]).count() == idx + 1
        assert t.where(t.data < sorted_data[idx]).count() == idx
        assert t.where(t.data >= sorted_data[idx]).count() == self.BTREE_TEST_NUM_ROWS - idx
        assert t.where(t.data > sorted_data[idx]).count() == self.BTREE_TEST_NUM_ROWS - idx - 1

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.where(t.data == data[56]).count()
        assert 'String literal too long' in str(exc_info.value)

    def test_timestamp_btree(self, reset_db) -> None:
        random.seed(1)
        start = datetime(2000, 1, 1)
        end = datetime(2020, 1, 1)
        delta = end - start
        delta_secs = int(delta.total_seconds())
        data = [start + timedelta(seconds=random.randint(0, int(delta_secs))) for _ in range(self.BTREE_TEST_NUM_ROWS)]
        t = self.run_btree_test(data, pxt.TimestampType())
