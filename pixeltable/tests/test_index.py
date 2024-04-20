import PIL.Image
import numpy as np
import pytest

import pixeltable as pxt
from pixeltable.functions.huggingface import clip_image, clip_text
from pixeltable.tests.utils import text_embed, img_embed, skip_test_if_not_installed


class TestIndex:

    # wrong signature
    @pxt.udf
    def bad_embed(x: str) -> str:
        return x

    def test_embedding_basic(self, img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
        skip_test_if_not_installed('transformers')
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        cl = pxt.Client()
        schema = {
            'img': pxt.ImageType(nullable=False),
            'category': pxt.StringType(nullable=False),
            'split': pxt.StringType(nullable=False),
        }
        tbl_name = 'index_test'
        img_t = cl.create_table(tbl_name, schema=schema)
        img_t.insert(rows[:30])

        img_t.add_embedding_index('img', img_embed=img_embed, text_embed=text_embed)

        with pytest.raises(pxt.Error) as exc_info:
            # duplicate name
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        img_t.add_embedding_index('category', text_embed=text_embed)
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
        cl = pxt.Client(reload=True)
        img_t = cl.get_table(tbl_name)
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
            img_t.add_embedding_index('img', idx_name='idx0', img_embed=img_embed)
        assert 'duplicate index name' in str(exc_info.value).lower()

        # dropping the indexed column also drops indices
        img_t.drop_column('img')
        with pytest.raises(pxt.Error) as exc_info:
            img_t.drop_index(idx_name='idx0')
        assert 'does not exist' in str(exc_info.value).lower()

    def test_errors(self, img_tbl: pxt.Table, test_tbl: pxt.Table) -> None:
        img_t = img_tbl
        rows = list(img_t.select(img=img_t.img.fileurl, category=img_t.category, split=img_t.split).collect())
        # create table with fewer rows to speed up testing
        cl = pxt.Client()
        schema = {
            'img': pxt.ImageType(nullable=False),
            'category': pxt.StringType(nullable=False),
            'split': pxt.StringType(nullable=False),
        }
        tbl_name = 'index_test'
        img_t = cl.create_table(tbl_name, schema=schema)
        img_t.insert(rows[:30])

        with pytest.raises(pxt.Error) as exc_info:
            # unknown column
            img_t.add_embedding_index('does_not_exist', idx_name='idx0', img_embed=img_embed)
        assert 'column does_not_exist unknown' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong column type
            test_tbl.add_embedding_index('c2', img_embed=img_embed)
        assert 'requires string or image column' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_tbl.add_embedding_index('img', text_embed=text_embed)
        assert 'image embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_tbl.add_embedding_index('img', img_embed=clip_image)
        assert 'but has signature' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # missing embedding function
            img_tbl.add_embedding_index('category', img_embed=img_embed)
        assert 'text embedding function is required' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # wrong signature
            img_tbl.add_embedding_index('category', text_embed=clip_text)
        assert 'but has signature' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            img_tbl.add_embedding_index('category', text_embed=self.bad_embed)
        assert 'must return an array' in str(exc_info.value).lower()
