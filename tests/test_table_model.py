# ruff: noqa: F821
# ruff: noqa: N806

from __future__ import annotations

import textwrap
from typing import Callable

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exceptions as excs
from pixeltable.catalog.model import Column, EmbeddingIndex

from .utils import (
    assert_resultset_eq,
    assert_table_metadata_eq,
    capture_console_output,
    dummy_embedding,
    get_image_files,
    pxt_raises,
    schema_from_tbl_md,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestTableModel:
    @pytest.mark.parametrize('root', ['', 'dir/subdir'])
    def test_table_model_basic(self, root: str, make_catalog_path: Callable[[str], str]) -> None:
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
            computed_with_special_props_2 = Column(value=img.rotate(90))

            clip_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        expected_path = f'{p(root)}/test_table'.lstrip('/')
        if root != '':
            pxt.create_dir(p(root), parents=True)

        print(expected_path)
        with capture_console_output(match=rf'Created {expected_path!r} from model `ExampleTableModel`.'):
            TableModel.create_all(p(root))

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
        tbl2.add_computed_column(computed_with_special_props_2=tbl2.img.rotate(90))
        tbl2.add_embedding_index(tbl2.img, idx_name='clip_idx', embedding=dummy_embedding.using(n=768))
        metadata2 = tbl2.get_metadata()

        assert schema_from_tbl_md(metadata) == schema_from_tbl_md(metadata2)

        tbl.insert([{'id': 1, 'name': 'Alice', 'value': 3.14}])
        tbl2.insert([{'id': 1, 'name': 'Alice', 'value': 3.14}])

        assert_resultset_eq(tbl.collect(), tbl2.collect())

        if p(root) != '':
            return  # Exact metadata comparison only applies to the '' case

        metadata_dict = dict(tbl.get_metadata())
        metadata_dict.pop('id')
        metadata_dict.pop('version_created')
        print(metadata_dict)
        assert_table_metadata_eq(
            {
                'name': 'test_table',
                'path': 'test_table',
                'columns': {
                    'id': {
                        'name': 'id',
                        'type_': 'Required[Int]',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': False,
                        'computed_with': None,
                        'is_builtin': None,
                        'depends_on': None,
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'name': {
                        'name': 'name',
                        'type_': 'String',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': False,
                        'computed_with': None,
                        'is_builtin': None,
                        'depends_on': None,
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'value': {
                        'name': 'value',
                        'type_': 'Float',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': False,
                        'computed_with': None,
                        'is_builtin': None,
                        'depends_on': None,
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'img': {
                        'name': 'img',
                        'type_': 'Image',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': 'on_write',
                        'is_computed': False,
                        'computed_with': None,
                        'is_builtin': None,
                        'depends_on': None,
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'incr': {
                        'name': 'incr',
                        'type_': 'Float',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': True,
                        'computed_with': 'value + 1',
                        'is_builtin': True,
                        'depends_on': [('test_table', 'value')],
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'descr': {
                        'name': 'descr',
                        'type_': 'Required[String]',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': True,
                        'computed_with': "format('Name: {name}', name=name)",
                        'is_builtin': True,
                        'depends_on': [('test_table', 'name')],
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'column_with_special_props': {
                        'name': 'column_with_special_props',
                        'type_': 'Video',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': 'on_read',
                        'is_computed': False,
                        'computed_with': None,
                        'is_builtin': None,
                        'depends_on': None,
                        'defined_in': 'test_table',
                        'comment': 'This is a column with special properties',
                        'custom_metadata': {'chicken': 'eggs'},
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'computed_with_special_props': {
                        'name': 'computed_with_special_props',
                        'type_': 'Float',
                        'version_added': 0,
                        'is_stored': False,
                        'is_primary_key': False,
                        'media_validation': None,
                        'is_computed': True,
                        'computed_with': 'value / 3',
                        'is_builtin': True,
                        'depends_on': [('test_table', 'value')],
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                    'computed_with_special_props_2': {
                        'name': 'computed_with_special_props_2',
                        'type_': 'Image',
                        'version_added': 0,
                        'is_stored': True,
                        'is_primary_key': False,
                        'media_validation': 'on_write',
                        'is_computed': True,
                        'computed_with': 'img.rotate(90)',
                        'is_builtin': True,
                        'depends_on': [('test_table', 'img')],
                        'defined_in': 'test_table',
                        'comment': None,
                        'custom_metadata': None,
                        'is_iterator_col': False,
                        'destination': None,
                    },
                },
                'indices': {
                    'idx0': {'name': 'idx0', 'columns': ['id'], 'index_type': 'btree', 'parameters': None},
                    'idx1': {'name': 'idx1', 'columns': ['name'], 'index_type': 'btree', 'parameters': None},
                    'idx2': {'name': 'idx2', 'columns': ['value'], 'index_type': 'btree', 'parameters': None},
                    'idx3': {'name': 'idx3', 'columns': ['img'], 'index_type': 'btree', 'parameters': None},
                    'idx4': {'name': 'idx4', 'columns': ['incr'], 'index_type': 'btree', 'parameters': None},
                    'idx5': {'name': 'idx5', 'columns': ['descr'], 'index_type': 'btree', 'parameters': None},
                    'idx6': {
                        'name': 'idx6',
                        'columns': ['column_with_special_props'],
                        'index_type': 'btree',
                        'parameters': None,
                    },
                    'clip_idx': {
                        'name': 'clip_idx',
                        'columns': ['img'],
                        'index_type': 'embedding',
                        'parameters': {
                            'metric': 'cosine',
                            'embedding': 'dummy_embedding(img, n=768)',
                            'embedding_functions': ['dummy_embedding(text, n=768)', 'dummy_embedding(img, n=768)'],
                        },
                    },
                },
                'is_versioned': True,
                'is_view': False,
                'is_snapshot': False,
                'version': 1,
                'schema_version': 0,
                'comment': None,
                'custom_metadata': None,
                'media_validation': 'on_write',
                'primary_key': None,
                'kind': 'table',
                'base': None,
                'iterator_call': None,
            },
            tbl.get_metadata(),
        )

    def test_all_table_exprs(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        TableModel = pxt.model_base()

        class AllExprsTableModel(TableModel, name='all_exprs_table'):
            id: pxt.Int
            name: pxt.String
            value: pxt.Float
            arr: pxt.Array
            img: pxt.Image
            arith_add = value + 1
            arith_radd = 1 + value
            arith_mul = value * 2
            arith_rmul = 2 * value
            array_slice = arr[:, 1:3]
            column_property_ref = img.fileurl  # type: ignore[attr-defined]
            column_ref = name
            comparison = value > 0.0
            compound_predicate = (value > 0.0) & (name != 'test')
            function_call = pxtf.math.floor(value)
            in_predicate = name.isin(['Alice', 'Bob', 'Charlie'])  # type: ignore[attr-defined]
            inline_array = pxt.array([value, value + 1, value + 2])
            inline_dict = {'name': name, 'img': img}  # noqa: RUF012
            inline_list = [name, img]  # noqa: RUF012
            is_null = name == None
            method_ref = name.upper()
            # similarity = name.similarity('similar string')
            string_add = name + ' suffix'
            string_radd = 'prefix ' + name
            string_mul = name * 3
            string_rmul = 3 * name
            type_cast = arr.astype(pxt.Array[(2, 3), np.float32])

        expected_path = p('all_exprs_table')
        TableModel.create_all(p(''))
        tbl = AllExprsTableModel.table

        # Create an analogous table using the "direct construction" method and verify that the schemas and table
        # behavior align.
        tbl2 = pxt.create_table(
            f'{expected_path}_2',
            {'id': pxt.Int, 'name': pxt.String, 'value': pxt.Float, 'arr': pxt.Array, 'img': pxt.Image},
        )
        tbl2.add_computed_column(arith_add=tbl2.value + 1)
        tbl2.add_computed_column(arith_radd=1 + tbl2.value)
        tbl2.add_computed_column(arith_mul=tbl2.value * 2)
        tbl2.add_computed_column(arith_rmul=2 * tbl2.value)
        tbl2.add_computed_column(array_slice=tbl2.arr[:, 1:3])
        tbl2.add_computed_column(column_property_ref=tbl2.img.fileurl)
        tbl2.add_computed_column(column_ref=tbl2.name)
        tbl2.add_computed_column(comparison=tbl2.value > 0.0)
        tbl2.add_computed_column(compound_predicate=(tbl2.value > 0.0) & (tbl2.name != 'test'))
        tbl2.add_computed_column(function_call=pxtf.math.floor(tbl2.value))
        tbl2.add_computed_column(in_predicate=tbl2.name.isin(['Alice', 'Bob', 'Charlie']))
        tbl2.add_computed_column(inline_array=pxt.array([tbl2.value, tbl2.value + 1, tbl2.value + 2]))
        tbl2.add_computed_column(inline_dict={'name': tbl2.name, 'img': tbl2.img})
        tbl2.add_computed_column(inline_list=[tbl2.name, tbl2.img])
        tbl2.add_computed_column(is_null=(tbl2.name == None))
        tbl2.add_computed_column(method_ref=tbl2.name.upper())
        tbl2.add_computed_column(string_add=(tbl2.name + ' suffix'))
        tbl2.add_computed_column(string_radd=('prefix ' + tbl2.name))
        tbl2.add_computed_column(string_mul=tbl2.name * 3)
        tbl2.add_computed_column(string_rmul=3 * tbl2.name)
        tbl2.add_computed_column(type_cast=tbl2.arr.astype(pxt.Array[(2, 3), np.float32]))

        assert schema_from_tbl_md(tbl.get_metadata()) == schema_from_tbl_md(tbl2.get_metadata())

        sample_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        row = {'id': 1, 'name': 'Alice', 'value': 3.14, 'arr': sample_arr, 'img': None}
        validate_update_status(tbl.insert([row]), expected_rows=1)
        validate_update_status(tbl2.insert([row]), expected_rows=1)

        assert_resultset_eq(tbl.collect(), tbl2.collect())

    @pytest.mark.parametrize('root', ['', 'dir/subdir'])
    def test_view_model(self, root: str, make_catalog_path: Callable[[str], str]) -> None:
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = value + 1
            descr = pxtf.string.format('Name: {name}', name=name)

            clip_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        class ExampleViewModel(TableModel, name='test_view', base=ExampleTableModel):
            view_col_1: pxt.Image
            view_col_2 = view_col_1.rotate(90)
            view_col_3 = ExampleTableModel.img.rotate(90)  # Also try dereferencing a base table column

            view_idx = EmbeddingIndex(view_col_2, embedding=dummy_embedding.using(n=768))
            view_idx_on_base_tbl_col = EmbeddingIndex(ExampleTableModel.img, embedding=dummy_embedding.using(n=768))

        class ExampleSubviewModel(TableModel, name='test_subview', base=ExampleViewModel):
            subview_col_1 = ExampleTableModel.img.rotate(180)
            subview_col_2 = ExampleViewModel.view_col_1.rotate(270)
            subview_col_3 = subview_col_2.rotate(30)

        class ExampleViewModelFromQuery(
            TableModel,
            name='test_view_from_query',
            base=ExampleTableModel.select(
                ExampleTableModel.value,
                ExampleTableModel.img,
                ExampleTableModel.value + 1,
                plusone=(ExampleTableModel.value + 1),
            ).where(ExampleTableModel.value > 0.5),
        ):
            view_col_1: pxt.Image
            view_col_2 = view_col_1.rotate(90)
            view_col_3 = ExampleTableModel.img.rotate(90)
            view_col_4 = plusone + 5  # type: ignore[name-defined]

            view_idx = EmbeddingIndex(view_col_2, embedding=dummy_embedding.using(n=768))
            view_idx_on_base_tbl_col = EmbeddingIndex(ExampleTableModel.img, embedding=dummy_embedding.using(n=768))

        class ExampleSubviewModelFromQuery(
            TableModel,
            name='test_subview_from_query',
            base=ExampleViewModelFromQuery.where(ExampleTableModel.value > 1.0),
        ):
            subview_col_1 = ExampleTableModel.img.rotate(180)
            subview_col_2 = ExampleViewModel.view_col_1.rotate(270)
            subview_col_3 = subview_col_2.rotate(30)

        prefix = '' if root == '' else f'{root}/'
        if root != '':
            pxt.create_dir(p(root), parents=True)

        with capture_console_output(
            match=rf'Created {p(f"{prefix}test_table")!r} from model `ExampleTableModel`.\n'
            rf'Created {p(f"{prefix}test_view")!r} from model `ExampleViewModel`.\n'
            rf'Created {p(f"{prefix}test_subview")!r} from model `ExampleSubviewModel`.\n'
            rf'Created {p(f"{prefix}test_view_from_query")!r} from model `ExampleViewModelFromQuery`.\n'
            rf'Created {p(f"{prefix}test_subview_from_query")!r} from model `ExampleSubviewModelFromQuery`.'
        ):
            TableModel.create_all(p(root))

        # Create analogous tables/views using the "direct construction" method and verify that the schemas (columns
        # and indices) align with the model-based ones. (The models default to `create_default_idxs=True`, including
        # for views, whereas `pxt.create_view()` defaults to `False`; pass it explicitly to match.)
        tbl2 = pxt.create_table(
            p(f'{prefix}test_table_2'),
            {'id': pxt.Required[pxt.Int], 'name': pxt.String, 'value': pxt.Float, 'img': pxt.Image},
        )
        tbl2.add_computed_column(incr=tbl2.value + 1)
        tbl2.add_computed_column(descr=pxtf.string.format('Name: {name}', name=tbl2.name))
        tbl2.add_embedding_index('img', idx_name='clip_idx', embedding=dummy_embedding.using(n=768))

        view2 = pxt.create_view(
            p(f'{prefix}test_view_2'), tbl2, additional_columns={'view_col_1': pxt.Image}, create_default_idxs=True
        )
        view2.add_computed_column(view_col_2=view2.view_col_1.rotate(90))
        view2.add_computed_column(view_col_3=view2.img.rotate(90))
        view2.add_embedding_index('view_col_2', idx_name='view_idx', embedding=dummy_embedding.using(n=768))
        view2.add_embedding_index('img', idx_name='view_idx_on_base_tbl_col', embedding=dummy_embedding.using(n=768))

        subview2 = pxt.create_view(p(f'{prefix}test_subview_2'), view2, create_default_idxs=True)
        subview2.add_computed_column(subview_col_1=subview2.img.rotate(180))
        subview2.add_computed_column(subview_col_2=subview2.view_col_1.rotate(270))
        subview2.add_computed_column(subview_col_3=subview2.subview_col_2.rotate(30))

        view_from_query2 = pxt.create_view(
            p(f'{prefix}test_view_from_query_2'),
            tbl2.select(tbl2.value, tbl2.img, tbl2.value + 1, plusone=tbl2.value + 1).where(tbl2.value > 0.5),
            additional_columns={'view_col_1': pxt.Image},
            create_default_idxs=True,
        )
        view_from_query2.add_computed_column(view_col_2=view_from_query2.view_col_1.rotate(90))
        view_from_query2.add_computed_column(view_col_3=view_from_query2.img.rotate(90))
        view_from_query2.add_computed_column(view_col_4=view_from_query2.plusone + 5)
        view_from_query2.add_embedding_index('view_col_2', idx_name='view_idx', embedding=dummy_embedding.using(n=768))
        view_from_query2.add_embedding_index(
            'img', idx_name='view_idx_on_base_tbl_col', embedding=dummy_embedding.using(n=768)
        )

        subview_from_query2 = pxt.create_view(
            p(f'{prefix}test_subview_from_query_2'),
            view_from_query2.where(view_from_query2.value > 1.0),
            create_default_idxs=True,
        )
        subview_from_query2.add_computed_column(subview_col_1=subview_from_query2.img.rotate(180))
        subview_from_query2.add_computed_column(subview_col_2=subview_from_query2.view_col_1.rotate(270))
        subview_from_query2.add_computed_column(subview_col_3=subview_from_query2.subview_col_2.rotate(30))

        images = get_image_files()
        rows = [
            {'id': 1, 'name': 'Alice', 'value': 3.14, 'img': images[0]},
            {'id': 2, 'name': 'Bob', 'value': 2.71, 'img': images[1]},
        ]
        ExampleTableModel.insert(rows)
        tbl2.insert(rows)

        for mtbl, atbl in (
            (ExampleTableModel.table, tbl2),
            (ExampleViewModel.table, view2),
            (ExampleSubviewModel.table, subview2),
            (ExampleViewModelFromQuery.table, view_from_query2),
            (ExampleSubviewModelFromQuery.table, subview_from_query2),
        ):
            assert schema_from_tbl_md(mtbl.get_metadata()) == schema_from_tbl_md(atbl.get_metadata())
            assert_resultset_eq(mtbl.order_by(mtbl.value).collect(), atbl.order_by(atbl.value).collect())

    def test_view_model_with_iterator(self, make_catalog_path: Callable[[str], str]) -> None:
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            image: pxt.Image

        class ExampleViewModel(
            TableModel,
            name='test_view',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.image, (256, 256)),
        ):
            view_col_1 = ExampleTableModel.value + 1
            view_col_2 = tile.rotate(90)  # type: ignore[name-defined]  # `tile` is defined by the iterator

        class ExampleViewModelFromQuery(
            TableModel,
            name='test_view_from_query',
            base=ExampleTableModel.select(
                ExampleTableModel.id, ExampleTableModel.image, rot=ExampleTableModel.image.rotate(90)
            ),
            iterator=pxtf.image.tile_iterator(ExampleTableModel.image, (256, 256)),
        ):
            view_col_1 = tile.rotate(90)  # type: ignore[name-defined]

        TableModel.create_all(p(''))
        tbl = ExampleTableModel.table
        view = ExampleViewModel.table
        view_from_query = ExampleViewModelFromQuery.table

        # Create analogous tables/views using the "direct construction" method and verify that the schemas (columns
        # and indices) align with the model-based ones. (The models default to `create_default_idxs=True`, including
        # for views, whereas `pxt.create_view()` defaults to `False`; pass it explicitly to match.)
        tbl2 = pxt.create_table(
            p('test_table_2'), {'id': pxt.Required[pxt.Int], 'name': pxt.String, 'value': pxt.Float, 'image': pxt.Image}
        )

        view2 = pxt.create_view(
            p('test_view_2'), tbl2, iterator=pxtf.image.tile_iterator(tbl2.image, (256, 256)), create_default_idxs=True
        )
        view2.add_computed_column(view_col_1=(tbl2.value + 1))
        view2.add_computed_column(view_col_2=view2.tile.rotate(90))

        view_from_query2 = pxt.create_view(
            p('test_view_from_query_2'),
            tbl2.select(tbl2.id, tbl2.image, rot=tbl2.image.rotate(90)),
            iterator=pxtf.image.tile_iterator(tbl2.image, (256, 256)),
            create_default_idxs=True,
        )
        view_from_query2.add_computed_column(view_col_1=view_from_query2.tile.rotate(90))

        images = get_image_files()
        rows = [
            {'id': 1, 'name': 'Alice', 'value': 3.14, 'image': images[0]},
            {'id': 2, 'name': 'Bob', 'value': 2.71, 'image': images[1]},
        ]
        ExampleTableModel.insert(rows)
        tbl2.insert(rows)

        assert schema_from_tbl_md(tbl.get_metadata()) == schema_from_tbl_md(tbl2.get_metadata())
        assert schema_from_tbl_md(view.get_metadata()) == schema_from_tbl_md(view2.get_metadata())
        assert schema_from_tbl_md(view_from_query.get_metadata()) == schema_from_tbl_md(view_from_query2.get_metadata())

        assert_resultset_eq(tbl.order_by(tbl.id).collect(), tbl2.order_by(tbl2.id).collect())
        assert_resultset_eq(view.order_by(view.id, view.pos).collect(), view2.order_by(view2.id, view2.pos).collect())
        assert_resultset_eq(
            view_from_query.order_by(view_from_query.id, view_from_query.pos).collect(),
            view_from_query2.order_by(view_from_query2.id, view_from_query2.pos).collect(),
        )

    def test_diff_all(self, make_catalog_path: Callable[[str], str]) -> None:
        """`diff_all()` reports added/dropped columns and an iterator mismatch against already-created tables."""
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        root = p('')

        # A base with a table model and a view model, 4 columns each. `create_default_idxs=False` keeps the diff
        # focused on columns and the iterator (default indexes are not part of a model's declared `__indexes__`).
        TableModel = pxt.model_base()

        class ExampleTable(TableModel, name='test_table', create_default_idxs=False):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            image: pxt.Image
            idx1 = EmbeddingIndex(image, embedding=dummy_embedding.using(n=768))
            idx2 = EmbeddingIndex(image, embedding=dummy_embedding.using(n=512))

        class ExampleView(
            TableModel,
            name='test_view',
            base=ExampleTable,
            iterator=pxtf.image.tile_iterator(ExampleTable.image, (256, 256)),
        ):
            vc1 = ExampleTable.id + 1
            vc2 = ExampleTable.id + 2
            vc3 = ExampleTable.id + 3
            vc4 = ExampleTable.id + 4

        # Created as a view; V2 redeclares it as a table, producing a kind mismatch.
        class ExampleKind(TableModel, name='test_kind', base=ExampleTable):
            kc1 = ExampleTable.value + 1
            kc2 = ExampleTable.value + 2

        TableModel.create_all(root)

        # Re-diffing the original models reports no differences (in particular, the view's iterator round-trips).
        with capture_console_output() as out:
            TableModel.diff_all(root)
        assert out.getvalue().strip() == 'Catalog is up to date.'

        # A fresh base whose models correspond to the created tables (same names), but with: two columns added and
        # two dropped in the table, and a mismatched iterator (128 vs. 256) in the view.
        TableModelV2 = pxt.model_base()

        class ExampleTableV2(TableModelV2, name='test_table', create_default_idxs=False):
            id: pxt.Required[pxt.Int]
            image: pxt.Image
            extra1: pxt.Int  # added
            extra2: pxt.String  # added
            # 'name' and 'value' dropped
            idx1 = EmbeddingIndex(image, embedding=dummy_embedding.using(n=768))  # kept
            idx3 = EmbeddingIndex(image, embedding=dummy_embedding.using(n=256))  # added
            # 'idx2' dropped

        class ExampleViewV2(
            TableModelV2,
            name='test_view',
            base=ExampleTableV2,
            iterator=pxtf.image.tile_iterator(ExampleTableV2.image, (128, 128)),  # mismatched tile size
        ):
            vc1 = ExampleTableV2.id + 1
            vc2 = ExampleTableV2.id + 2
            vextra1: pxt.Int
            vextra2: pxt.String

        # Redeclares 'test_kind' (created above as a view) as a table, with the same columns; only the kind differs.
        class ExampleKindV2(TableModelV2, name='test_kind'):
            kc1: pxt.Float
            kc2: pxt.Float

        # A model with no corresponding table in the catalog; it would be created.
        class ExampleNewV2(TableModelV2, name='test_new'):
            id: pxt.Required[pxt.Int]
            data: pxt.String

        with capture_console_output() as out:
            TableModelV2.diff_all(root)
        assert (
            out.getvalue().strip()
            == textwrap.dedent("""
            Table 'test_table' (from model `ExampleTableV2`) has differences:
              the following columns are new to the model, and will be ADDED:
                'extra1' = {'type': Int | None}
                'extra2' = {'type': String | None}
              the following columns are no longer in the model, and will be DROPPED:
                'name'
                'value'
              the following indexes are new to the model, and will be ADDED:
                'idx3' = EmbeddingIndex(column=image, embedding=dummy_embedding(text, n=256))
              the following indexes are no longer in the model, and will be DROPPED:
                'idx2'
            View 'test_view' (from model `ExampleViewV2`) has differences:
              iterator mismatch (FATAL):
                model iterator   : tile_iterator(image, [128, 128])
                existing iterator: tile_iterator(image, [256, 256])
              the following columns are new to the model, and will be ADDED:
                'vextra1' = {'type': Int | None}
                'vextra2' = {'type': String | None}
              the following columns are no longer in the model, and will be DROPPED:
                'vc3'
                'vc4'
            Table 'test_kind' (from model `ExampleKindV2`) has differences:
              kind mismatch (FATAL): `ExampleKindV2` specifies a table, but 'test_kind' is a view
            Table 'test_new' (from model `ExampleNewV2`) does not yet exist, and will be CREATED.
            """).strip()
        )

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r'One or more tables cannot be updated, because their models are inconsistent with the existing'
        ):
            TableModelV2.update_all(root)

    def test_table_model_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        """Reproduce each error condition raised by `pixeltable.catalog.model`."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r'`name` must be a valid Pixeltable identifier'):

            class BadTableName(TableModel, name='invalid! table@name'):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_ARGUMENT,
            match=r'model `BadIterTable`: `iterator` can only be specified together with a `base`.',
        ):

            class BadIterTable(TableModel, name='bad_iter_table', iterator='not_allowed'):
                pass

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'Empty table schema not allowed.'):

            class EmptyTableModel(TableModel, name='empty_table'):
                pass

        class ValidTableModel(TableModel, name='valid_table'):
            id: pxt.Int

        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match='must be a valid iterator reference'):

            class BadIterRef(TableModel, name='bad_iter_ref', base=ValidTableModel, iterator='not a valid iterator'):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_ARGUMENT, match=r"`media_validation` must be one of: \['on_read', 'on_write'\]"
        ):

            class BadMediaValidation(TableModel, name='bad_media_validation', media_validation='on_ragnarok'):
                pass

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Conflicting type annotation for column 'name'."):

            class TypeConflict(TableModel, name='type_conflict'):
                name: pxt.Int = Column(type=pxt.String)  # type: ignore[assignment]

        with pxt_raises(
            excs.ErrorCode.INVALID_ARGUMENT,
            match=r'model `InvalidBase`: `base` must be a valid base table reference '
            r'\(another Pixeltable model, or a query over a model\).',
        ):

            class InvalidBase(TableModel, name='invalid_base', base=42):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'Pixeltable schemas must be direct subclasses of a model_base\(\).'
        ):

            class SubclassedModel(ValidTableModel, name='subclassed_model'):
                x: pxt.Int

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r"has name 'dup_name', but that name was previously used by `FirstDup`"
        ):

            class FirstDup(TableModel, name='dup_name'):
                id: pxt.Int

            class SecondDup(TableModel, name='dup_name'):
                id: pxt.Int

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'must define `type` or `value`, but not both'):

            class BadColSpec(TableModel, name='bad_col_spec'):
                id: pxt.Int
                bad = Column()

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'Cannot set a type annotation for index'):

            class IdxTypeConflict(TableModel, name='idx_type_conflict'):
                img: pxt.Image
                my_idx: pxt.Int = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))  # type: ignore[assignment]

        # `references columns that are not in the model's scope` is raised at `create()` time, when a computed
        # column refers to a column outside the model (here, a column belonging to a different, unbound model).
        class OtherModel(TableModel, name='other_model'):
            x: pxt.Int

        class RefsOutOfScope(TableModel, name='refs_out_of_scope'):
            y: pxt.Int
            bad = OtherModel.x + 1

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"references columns that are not in the model's scope"):
            RefsOutOfScope._create(p(''))

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Column 'plus': duplicate definition"):

            class DuplicateColumn(TableModel, name='duplicate_column'):
                id: pxt.Int
                plus = id + 1
                plus = id + 2

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Index 'dup_idx': duplicate definition"):

            class DuplicateIndex(TableModel, name='duplicate_index'):
                img: pxt.Image
                dup_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))
                dup_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Column 'bad': invalid value"):

            class InvalidValue(TableModel, name='invalid_value'):
                id: pxt.Int
                bad = object()

        # A model column may not redefine a name already provided by the base query...
        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r"'doubled' is already defined by the base query; it cannot be redeclared.",
        ):

            class QueryColCollision(
                TableModel, name='query_col_collision', base=ValidTableModel.select(doubled=ValidTableModel.id * 2)
            ):
                doubled = ValidTableModel.id * 3

        # ...or by the iterator.
        class ImageModel(TableModel, name='image_model'):
            img: pxt.Image

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r"'tile' is already defined by the iterator; it cannot be redeclared."
        ):

            class IterColCollision(
                TableModel,
                name='iter_col_collision',
                base=ImageModel,
                iterator=pxtf.image.tile_iterator(ImageModel.img, (256, 256)),
            ):
                tile = 5

        # Forwarded `Table` methods that aren't available on a placeholder query raise `AttributeError` when the
        # model isn't yet bound to an actual table.
        with pytest.raises(AttributeError, match=r'is not yet bound to an actual table'):
            ValidTableModel.collect()

        # `ModelQuery` clause methods reject being specified more than once in a `ViewModel` base query.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`select\(\)` list already specified'):
            ValidTableModel.select(ValidTableModel.id).select(ValidTableModel.id)

        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r'Invalid name: bad name'):
            ValidTableModel.select(**{'bad name': ValidTableModel.id})

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`where\(\)` clause already specified'):
            ValidTableModel.where(ValidTableModel.id > 0).where(ValidTableModel.id > 0)  # type: ignore[arg-type]

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`group_by\(\)` clause already specified'):
            ValidTableModel.group_by(ValidTableModel.id).group_by(ValidTableModel.id)  # type: ignore[call-overload]

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`limit\(\)` clause already specified'):
            ValidTableModel.limit(10).limit(5)

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`sample\(\)` clause already specified'):
            ValidTableModel.sample(n=10).sample(n=5)

    def test_table_model_validation_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        """Errors that arise from a schema mismatch between a model and an existing table."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        t = pxt.create_table(p('test_table'), {'id': pxt.Required[pxt.Int], 'name': pxt.String, 'img': pxt.Image})
        _ = pxt.create_view(p('test_view'), t)
        _ = pxt.create_snapshot(p('test_snapshot'), t)

        t_ok = pxt.create_table(p('ok_table'), {'id': pxt.Required[pxt.Int], 'name': pxt.String, 'img': pxt.Image})
        _ = pxt.create_view(p('test_view_2'), t_ok)
        _ = pxt.create_view(p('test_iter_view'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))
        _ = pxt.create_view(p('test_iter_view_2'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))
        _ = pxt.create_view(p('test_iter_view_3'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))

        class BadTableModel(TableModel, name='test_view'):
            id: pxt.Required[pxt.Int]

        class ExampleTableModel(TableModel, name='ok_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            img: pxt.Image

        class BadViewModel(TableModel, name='test_table', base=ExampleTableModel):
            pass

        class BadViewModel2(TableModel, name='test_snapshot', base=ExampleTableModel):
            pass

        class GoodIterViewModel(
            TableModel,
            name='test_iter_view',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.img, (256, 256)),
        ):
            pass

        class IteratorMismatch(
            TableModel,
            name='test_iter_view_2',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.img, (128, 128)),
        ):
            pass

        class MissingIterator(TableModel, name='test_iter_view_3', base=ExampleTableModel):
            pass

        class ExtraneousIterator(
            TableModel,
            name='test_view_2',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.img, (256, 256)),
        ):
            pass

        ExampleTableModel._create(p(''))  # should succeed; schema matches existing table

        # The validation errors below are raised by `Catalog.create_from_model` in the catalog that owns the
        # table, so the paths they report are in-db paths (no proxy prefix) — identical in local and proxy modes.
        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"model `BadTableModel` is defined as a table, but the existing 'test_view' is a view.",
        ):
            BadTableModel._create(p(''))

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"model `BadViewModel` is defined as a view, but the existing 'test_table' is a table.",
        ):
            BadViewModel._create(p(''))

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"model `BadViewModel2` is defined as a view, but the existing 'test_snapshot' is a snapshot.",
        ):
            BadViewModel2._create(p(''))

        GoodIterViewModel._create(p(''))

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"Iterator for model `IteratorMismatch` does not match the existing table 'test_iter_view_2'.",
        ):
            IteratorMismatch._create(p(''))

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"Iterator for model `MissingIterator` does not match the existing table 'test_iter_view_3'.",
        ):
            MissingIterator._create(p(''))

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r"Iterator for model `ExtraneousIterator` does not match the existing table 'test_view_2'.",
        ):
            ExtraneousIterator._create(p(''))
