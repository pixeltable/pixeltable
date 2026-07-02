# ruff: noqa: F821
# ruff: noqa: N806

import sys
from typing import Callable

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import Column, EmbeddingIndex

from .utils import assert_resultset_eq, capture_console_output, dummy_embedding, schema_from_tbl_md

# Separate model tests, in a different file from test_table_model.py that is declared without
# `from __future__ import annotations`


class TestTableModel2:
    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason='Fails on Python 3.14 without `from __future__ import annotations` (PEP 649)',
    )
    def test_table_model_no_from_future(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = value + 1  # computed column
            descr = pxtf.string.format('Name: {name}', name=name)

            # Test all the custom `Column` properties
            column_with_special_props = Column(
                type=pxt.Video,
                media_validation='on_read',
                custom_metadata={'chicken': 'eggs'},
                comment='This is a column with special properties',
            )
            computed_with_special_props = Column(value=(value / 3), stored=False)
            computed_with_special_props_2 = Column(value=img.rotate(90), destination='.')

            clip_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        expected_path = f'{p("")}/test_table'.lstrip('/')

        print(expected_path)
        with capture_console_output(match=rf'Created {expected_path!r} from model `ExampleTableModel`.'):
            TableModel.create_all(p(''))

        tbl = ExampleTableModel.table
        metadata = tbl.get_metadata()
        assert str(metadata['path']) == expected_path

        # Create an analogous table using the "direct construction" method and verify that the schemas and table
        # behavior align.

        tbl2 = pxt.create_table(
            f'{expected_path}_2',
            {'id': pxt.Required[pxt.Int], 'name': pxt.String, 'value': pxt.Float, 'img': pxt.Image},
        )
        tbl2.add_computed_column(incr=tbl2.value + 1)
        tbl2.add_computed_column(descr=pxtf.string.format('Name: {name}', name=tbl2.name))
        tbl2.add_column(
            column_with_special_props={
                'type': pxt.Video,
                'media_validation': 'on_read',
                'custom_metadata': {'chicken': 'eggs'},
                'comment': 'This is a column with special properties',
            }
        )
        tbl2.add_computed_column(computed_with_special_props=(tbl2.value / 3), stored=False)
        tbl2.add_computed_column(computed_with_special_props_2=tbl2.img.rotate(90), destination='.')
        tbl2.add_embedding_index(tbl2.img, idx_name='clip_idx', embedding=dummy_embedding.using(n=768))
        metadata2 = tbl2.get_metadata()

        assert schema_from_tbl_md(metadata) == schema_from_tbl_md(metadata2)

        tbl.insert([{'id': 1, 'name': 'Alice', 'value': 3.14}])
        tbl2.insert([{'id': 1, 'name': 'Alice', 'value': 3.14}])

        assert_resultset_eq(tbl.collect(), tbl2.collect())
