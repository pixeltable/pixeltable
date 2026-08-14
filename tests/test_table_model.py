# ruff: noqa: F821
# ruff: noqa: N806
# ruff: noqa: RUF012

from __future__ import annotations

import textwrap
from typing import Callable

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exceptions as excs
from pixeltable.catalog.model import BtreeIndex, Column, EmbeddingIndex

from .utils import (
    assert_resultset_eq,
    assert_table_metadata_eq,
    btree_idxs,
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
            id: pxt.Int
            name: pxt.String | None
            value: pxt.Float | None
            img: pxt.Image | None
            incr = value + 1  # computed column
            descr = pxtf.string.format('Name: {name}', name=name)

            # Test all the custom Column properties
            column_with_special_props = Column(
                type=pxt.Video | None,
                media_validation='on_read',
                custom_metadata={'chicken': 'eggs'},
                comment='This is a column with special properties',
            )
            computed_with_special_props = Column(value=(value / 3), stored=False)
            computed_with_special_props_2 = Column(value=img.rotate(90))

            __indexes__ = [
                BtreeIndex(id),
                EmbeddingIndex(descr, embedding=dummy_embedding.using(n=512)),
                EmbeddingIndex(img, embedding=dummy_embedding.using(n=768), name='clip_idx'),
            ]

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
            {'id': pxt.Int, 'name': pxt.String | None, 'value': pxt.Float | None, 'img': pxt.Image | None},
        )
        tbl2.add_computed_column(incr=tbl2.value + 1)
        tbl2.add_computed_column(descr=pxtf.string.format('Name: {name}', name=tbl2.name))
        tbl2.add_column(
            column_with_special_props={
                'type': pxt.Video | None,
                'media_validation': 'on_read',
                'custom_metadata': {'chicken': 'eggs'},
                'comment': 'This is a column with special properties',
            }
        )
        tbl2.add_computed_column(computed_with_special_props=(tbl2.value / 3), stored=False)
        tbl2.add_computed_column(computed_with_special_props_2=tbl2.img.rotate(90))
        tbl2.add_btree_index(tbl2.id)
        tbl2.add_embedding_index(tbl2.descr, embedding=dummy_embedding.using(n=512))
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
                        'type_': 'Int',
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
                        'type_': 'String | None',
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
                        'type_': 'Float | None',
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
                        'type_': 'Image | None',
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
                        'type_': 'Float | None',
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
                        'type_': 'String',
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
                        'type_': 'Video | None',
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
                        'type_': 'Float | None',
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
                        'type_': 'Image | None',
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
                'indexes': {
                    'idx0': {'columns': ['id'], 'index_type': 'btree', 'name': 'idx0', 'parameters': None},
                    'idx1': {
                        'columns': ['descr'],
                        'index_type': 'embedding',
                        'name': 'idx1',
                        'parameters': {
                            'embedding': 'dummy_embedding(descr, n=512)',
                            'embedding_functions': ['dummy_embedding(text, n=512)', 'dummy_embedding(img, n=512)'],
                            'metric': 'cosine',
                            'precision': 'fp16',
                        },
                    },
                    'clip_idx': {
                        'name': 'clip_idx',
                        'columns': ['img'],
                        'index_type': 'embedding',
                        'parameters': {
                            'metric': 'cosine',
                            'precision': 'fp16',
                            'embedding': 'dummy_embedding(img, n=768)',
                            'embedding_functions': ['dummy_embedding(text, n=768)', 'dummy_embedding(img, n=768)'],
                        },
                    },
                },
                'is_data_versioned': True,
                'has_default_idxs': False,
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
                'view_filter': None,
                'view_sample': None,
                'iterator_call': None,
            },
            tbl.get_metadata(),
        )

    def test_btree_index_declaration(self, make_catalog_path: Callable[[str], str]) -> None:
        root = make_catalog_path('')
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None

            __indexes__ = [BtreeIndex(name), BtreeIndex(img)]

        class ExampleViewModel(TableModel, name='test_view', base=ExampleTableModel):
            vc: pxt.Int | None

        TableModel.create_all(root)
        tbl = ExampleTableModel.table
        ExampleTableModel.insert([{'id': 1, 'name': 'a', 'img': get_image_files()[0]}])

        assert btree_idxs(tbl) == {'idx0': 'name', 'idx1': 'img'}
        assert len(btree_idxs(ExampleViewModel.table)) == 0

        # Rename an index
        TM_rename = pxt.model_base()

        class RenamedTableModel(TM_rename, name='test_table'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None

            __indexes__ = [BtreeIndex(name), BtreeIndex(img)]

        class ViewOwnCol(TM_rename, name='test_view', base=RenamedTableModel):
            vc: pxt.Int | None

            __indexes__ = [BtreeIndex(vc)]

        TM_rename.update_all(root, allow_destructive=True)
        assert btree_idxs(tbl) == {'idx0': 'name', 'idx1': 'img'}
        assert btree_idxs(ExampleViewModel.table) == {'idx0': 'vc'}

    def test_default_idxs_diff(self, make_catalog_path: Callable[[str], str]) -> None:
        """Verifies how model diff interacts with has_default_idxs."""
        p = make_catalog_path
        root = p('')
        TableModel = pxt.model_base()

        class WithDefaults(TableModel, name='defaults_table', has_default_idxs=True):
            id: pxt.Int
            name: pxt.String | None

        class NoDefaults(TableModel, name='no_defaults_table'):
            id: pxt.Int
            name: pxt.String | None

            __indexes__ = [BtreeIndex(name)]

        TableModel.create_all(root)
        tbl_with_defaults = WithDefaults.table

        assert set(btree_idxs(tbl_with_defaults).values()) == {'id', 'name'}
        assert tbl_with_defaults.get_metadata()['has_default_idxs'] is True

        # New column in defaults_table gets a B-tree index automatically
        TableModelV2 = pxt.model_base()

        class WithDefaultsV2(TableModelV2, name='defaults_table', has_default_idxs=True):
            id: pxt.Int
            name: pxt.String | None
            extra: pxt.Int | None

        diff = TableModelV2.get_model_diff(root)['defaults_table']
        assert diff['resolution'] == 'update_additive'
        TableModelV2.update_all(root)
        assert set(btree_idxs(tbl_with_defaults).values()) == {'id', 'name', 'extra'}
        assert TableModelV2.get_model_diff(root)['defaults_table']['resolution'] == 'up_to_date'

        # has_default_idxs can't be changed
        TableModelV3 = pxt.model_base()

        class WithDefaultsV3(TableModelV3, name='defaults_table', has_default_idxs=False):
            id: pxt.Int
            name: pxt.String | None
            extra: pxt.Int | None

            __indexes__ = [BtreeIndex(name)]

        class NoDefaultsV3(TableModelV3, name='no_defaults_table', has_default_idxs=True):
            id: pxt.Int
            name: pxt.String | None

        assert TableModelV3.get_model_diff(root)['defaults_table']['resolution'] == 'unsupported'
        assert TableModelV3.get_model_diff(root)['no_defaults_table']['resolution'] == 'unsupported'
        with capture_console_output(
            match=r'the following table properties have changed \(FATAL\):\n'
            r'\s*has_default_idxs: model=False, existing=True'
        ):
            TableModelV3.diff_all(root)

    def test_btree_index_validation(self, make_catalog_path: Callable[[str], str]) -> None:
        """`update_all()` and `create_all()` enforce the same B-tree eligibility rules as `Table.add_btree_index()`."""
        root = make_catalog_path('')
        TableModel = pxt.model_base()

        class Base(TableModel, name='base'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None
            unstored = Column(value=img.rotate(90), stored=False)

            __indexes__ = [BtreeIndex(id)]

        class V(TableModel, name='v', base=Base):
            vc: pxt.Int | None

        TableModel.create_all(root)

        # An ineligible column.
        TM_unstored = pxt.model_base()

        class BaseUnstored(TM_unstored, name='base'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None
            unstored = Column(value=img.rotate(90), stored=False)

            __indexes__ = [BtreeIndex(id), BtreeIndex(unstored)]

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="unstored column 'unstored'"):
            TM_unstored.update_all(root)

        # A view cannot index a base table's column.
        TM_base_col = pxt.model_base()

        class BaseForView(TM_base_col, name='base'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None
            unstored = Column(value=img.rotate(90), stored=False)

            __indexes__ = [BtreeIndex(id)]

        class ViewOnBaseCol(TM_base_col, name='v', base=BaseForView):
            vc: pxt.Int | None

            __indexes__ = [BtreeIndex(BaseForView.name)]

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='belongs to a base table'):
            TM_base_col.update_all(root)

        # None of the rejected changes were applied.
        assert btree_idxs(Base.table) == {'idx0': 'id'}
        assert btree_idxs(V.table) == {}

        # The same rule holds when the view declares the index up front (create_all instead update update_all)
        TM_create = pxt.model_base()

        class BaseAtCreate(TM_create, name='base_at_create'):
            id: pxt.Int
            name: pxt.String | None

        class ViewOnBaseColAtCreate(TM_create, name='v_at_create', base=BaseAtCreate):
            doubled = BaseAtCreate.id * 2

            __indexes__ = [BtreeIndex(BaseAtCreate.name)]

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='belongs to a base table'):
            TM_create.create_all(root)

    def test_index_name_collision_on_update(self, make_catalog_path: Callable[[str], str]) -> None:
        """`update_all()` rejects a declared index whose name is taken by one of the table's existing indexes."""
        p = make_catalog_path
        root = p('')
        TableModel = pxt.model_base()

        class Defaults(TableModel, name='defaults', has_default_idxs=True):
            txt: pxt.String | None

        TableModel.create_all(root)

        assert btree_idxs(Defaults.table) == {'idx0': 'txt'}

        TM_collision = pxt.model_base()

        class DefaultsWithIdx0(TM_collision, name='defaults', has_default_idxs=True):
            txt: pxt.String | None

            __indexes__ = [EmbeddingIndex(txt, embedding=dummy_embedding.using(n=768), name='idx0')]

        with pxt_raises(pxt.ErrorCode.INDEX_ALREADY_EXISTS, match="Index 'idx0' already exists on column 'txt'"):
            TM_collision.update_all(root)

        assert btree_idxs(Defaults.table) == {'idx0': 'txt'}

    def test_all_table_exprs(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        TableModel = pxt.model_base()

        class AllExprsTableModel(TableModel, name='all_exprs_table'):
            id: pxt.Int | None
            name: pxt.String | None
            value: pxt.Float | None
            arr: pxt.Array | None
            img: pxt.Image | None
            arith_add = value + 1
            arith_radd = 1 + value
            arith_mul = value * 2
            arith_rmul = 2 * value
            array_slice = arr[:, 1:3]
            column_property_ref = img.fileurl  # type: ignore[union-attr]
            column_ref = name
            comparison = value > 0.0
            compound_predicate = (value > 0.0) & (name != 'test')
            function_call = pxtf.math.floor(value)
            in_predicate = name.isin(['Alice', 'Bob', 'Charlie'])  # type: ignore[union-attr]
            inline_array = pxt.array([value, value + 1, value + 2])
            inline_dict = {'name': name, 'img': img}
            inline_list = [name, img]
            is_null = name == None
            method_ref = name.upper()
            # similarity = name.similarity('similar string')
            string_add = name + ' suffix'
            string_radd = 'prefix ' + name
            string_mul = name * 3
            string_rmul = 3 * name
            type_cast = arr.astype(pxt.Array[(2, 3), np.float32] | None)

        expected_path = p('all_exprs_table')
        TableModel.create_all(p(''))
        tbl = AllExprsTableModel.table

        # Create an analogous table using the "direct construction" method and verify that the schemas and table
        # behavior align.
        tbl2 = pxt.create_table(
            f'{expected_path}_2',
            {
                'id': pxt.Int | None,
                'name': pxt.String | None,
                'value': pxt.Float | None,
                'arr': pxt.Array | None,
                'img': pxt.Image | None,
            },
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
        tbl2.add_computed_column(type_cast=tbl2.arr.astype(pxt.Array[(2, 3), np.float32] | None))

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

        class ExampleTableModel(TableModel, name='test_table', has_default_idxs=True):
            id: pxt.Int
            name: pxt.String | None
            value: pxt.Float | None
            img: pxt.Image | None
            incr = value + 1
            descr = pxtf.string.format('Name: {name}', name=name)

            __indexes__ = [EmbeddingIndex(img, embedding=dummy_embedding.using(n=768), name='clip_idx')]

        class ExampleViewModel(TableModel, name='test_view', base=ExampleTableModel, has_default_idxs=True):
            view_col_1: pxt.Image | None
            view_col_2 = view_col_1.rotate(90)
            view_col_3 = ExampleTableModel.img.rotate(90)  # Also try dereferencing a base table column

            __indexes__ = [
                EmbeddingIndex(view_col_2, embedding=dummy_embedding.using(n=768), name='view_idx'),
                EmbeddingIndex(
                    ExampleTableModel.img, embedding=dummy_embedding.using(n=768), name='view_idx_on_base_tbl_col'
                ),
            ]

        class ExampleSubviewModel(TableModel, name='test_subview', base=ExampleViewModel, has_default_idxs=True):
            subview_col_1 = ExampleTableModel.img.rotate(180)
            subview_col_2 = ExampleViewModel.view_col_1.rotate(270)
            subview_col_3 = subview_col_2.rotate(30)

        class ExampleViewModelFromQuery(
            TableModel,
            name='test_view_from_query',
            base=ExampleTableModel.select(
                ExampleTableModel.value, ExampleTableModel.img, plusone=(ExampleTableModel.value + 1)
            ).where(ExampleTableModel.value > 0.5),
            has_default_idxs=True,
        ):
            view_col_1: pxt.Image | None
            view_col_2 = view_col_1.rotate(90)
            view_col_3 = ExampleTableModel.img.rotate(90)
            view_col_4 = plusone + 5  # type: ignore[name-defined]

            __indexes__ = [
                EmbeddingIndex(view_col_2, embedding=dummy_embedding.using(n=768), name='view_idx'),
                EmbeddingIndex(
                    ExampleTableModel.img, embedding=dummy_embedding.using(n=768), name='view_idx_on_base_tbl_col'
                ),
            ]

        class ExampleSubviewModelFromQuery(
            TableModel,
            name='test_subview_from_query',
            base=ExampleViewModelFromQuery.where(ExampleTableModel.value > 1.0),
            has_default_idxs=True,
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
        # and indices) align with the model-based ones.
        tbl2 = pxt.create_table(
            p(f'{prefix}test_table_2'),
            {'id': pxt.Int, 'name': pxt.String | None, 'value': pxt.Float | None, 'img': pxt.Image | None},
            has_default_idxs=True,
        )
        tbl2.add_computed_column(incr=tbl2.value + 1)
        tbl2.add_computed_column(descr=pxtf.string.format('Name: {name}', name=tbl2.name))
        tbl2.add_embedding_index('img', idx_name='clip_idx', embedding=dummy_embedding.using(n=768))

        view2 = pxt.create_view(
            p(f'{prefix}test_view_2'), tbl2, additional_columns={'view_col_1': pxt.Image | None}, has_default_idxs=True
        )
        view2.add_computed_column(view_col_2=view2.view_col_1.rotate(90))
        view2.add_computed_column(view_col_3=view2.img.rotate(90))
        view2.add_embedding_index('view_col_2', idx_name='view_idx', embedding=dummy_embedding.using(n=768))
        view2.add_embedding_index('img', idx_name='view_idx_on_base_tbl_col', embedding=dummy_embedding.using(n=768))

        subview2 = pxt.create_view(p(f'{prefix}test_subview_2'), view2, has_default_idxs=True)
        subview2.add_computed_column(subview_col_1=subview2.img.rotate(180))
        subview2.add_computed_column(subview_col_2=subview2.view_col_1.rotate(270))
        subview2.add_computed_column(subview_col_3=subview2.subview_col_2.rotate(30))

        view_from_query2 = pxt.create_view(
            p(f'{prefix}test_view_from_query_2'),
            tbl2.select(tbl2.value, tbl2.img, plusone=tbl2.value + 1).where(tbl2.value > 0.5),
            additional_columns={'view_col_1': pxt.Image | None},
            has_default_idxs=True,
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
            has_default_idxs=True,
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

    def test_view_model_shadows_base_column(self, make_catalog_path: Callable[[str], str]) -> None:
        """A view model column cannot shadow a base column, the same as create_view()."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ExampleViewModel(TableModel, name='test_view', base=ExampleTableModel):
            value = ExampleTableModel.value * 100.0

        with pxt_raises(
            excs.ErrorCode.COLUMN_ALREADY_EXISTS, match=r"Column 'value' already exists in the base table 'test_table'"
        ):
            TableModel.create_all(p(''))

    def test_update_all_adds_shadowing_column(self, make_catalog_path: Callable[[str], str]) -> None:
        """update_all() cannot add a view column that shadows a base column, the same as add_computed_column()."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ExampleViewModel(TableModel, name='test_view', base=ExampleTableModel):
            vc1 = ExampleTableModel.id + 1

        TableModel.create_all(p(''))
        ExampleTableModel.insert([{'id': 1, 'value': 2.0}])

        TableModelV2 = pxt.model_base()

        class ExampleTableModelV2(TableModelV2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        class ExampleViewModelV2(TableModelV2, name='test_view', base=ExampleTableModelV2):
            vc1 = ExampleTableModelV2.id + 1
            value = ExampleTableModelV2.value * 100.0

        with pxt_raises(
            excs.ErrorCode.COLUMN_ALREADY_EXISTS, match=r"Column 'value' already exists in the base table 'test_table'"
        ):
            TableModelV2.update_all(p(''))
        # nothing was applied: value still reads through to the base's column
        t = ExampleViewModel.table
        assert t.select(t.value).collect()['value'] == [2.0]

    def test_view_model_index_on_iterator_column(self, make_catalog_path: Callable[[str], str]) -> None:
        """An embedding index in a view model can name a column produced by the view's iterator."""
        skip_test_if_not_installed('spacy')
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            doc_text: pxt.String | None

        class ExampleViewModel(
            TableModel,
            name='test_view',
            base=ExampleTableModel,
            iterator=pxtf.string.string_splitter(ExampleTableModel.doc_text, separators='sentence'),
        ):
            # text is an output column of the iterator, not one declared by this model
            __indexes__ = [
                EmbeddingIndex(text, embedding=dummy_embedding.using(n=32), name='ix')  # type: ignore[name-defined]
            ]

        TableModel.create_all(p(''))
        ExampleTableModel.insert([{'id': 1, 'doc_text': 'One sentence. Two sentence.'}])

        idx_md = ExampleViewModel.get_metadata()['indexes']['ix']
        assert idx_md['columns'] == ['text']
        assert idx_md['index_type'] == 'embedding'
        view = ExampleViewModel.table
        sim = view.text.similarity(string='One sentence.')
        assert len(view.order_by(sim, asc=False).limit(1).collect()) == 1

    def test_view_model_iterator_column_shadows_base(self, make_catalog_path: Callable[[str], str]) -> None:
        """An iterator output shadows a base column of the same name, so the model's text is the chunk text
        throughout: the column, the index declared on it, and queries against it."""
        skip_test_if_not_installed('spacy')
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            text: pxt.String | None  # the document text; shadowed in the view by the iterator's text output

        class ExampleViewModel(
            TableModel,
            name='test_view',
            base=ExampleTableModel,
            iterator=pxtf.string.string_splitter(ExampleTableModel.text, separators='sentence'),
        ):
            __indexes__ = [
                EmbeddingIndex(text, embedding=dummy_embedding.using(n=32), name='ix')  # type: ignore[name-defined]
            ]

        TableModel.create_all(p(''))
        ExampleTableModel.insert([{'id': 1, 'text': 'One sentence. Two sentence.'}])

        view = ExampleViewModel.table
        assert view.columns() == ['pos', 'text', 'id']
        assert [r['text'] for r in view.order_by(view.pos).collect()] == ['One sentence.', 'Two sentence.']
        idx_md = ExampleViewModel.get_metadata()['indexes']['ix']
        assert idx_md['columns'] == ['text']
        sim = view.text.similarity(string='One sentence.')
        assert len(view.order_by(sim, asc=False).limit(1).collect()) == 1

    def test_view_model_column_collides_with_iterator(self, make_catalog_path: Callable[[str], str]) -> None:
        """A model column cannot reuse the name of one of the view's iterator outputs."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            doc_text: pxt.String | None

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r"'text' is already defined by the iterator; it cannot be redeclared"
        ):

            class ExampleViewModel(
                TableModel,
                name='test_view',
                base=ExampleTableModel,
                iterator=pxtf.string.string_splitter(ExampleTableModel.doc_text, separators='sentence'),
            ):
                text = ExampleTableModel.doc_text + '!'

        # the same collision through the non-model API
        TableModel.create_all(p(''))
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"'text' .* produced by the iterator"):
            pxt.create_view(
                p('other_view'),
                ExampleTableModel.table,
                iterator=pxtf.string.string_splitter(ExampleTableModel.table.doc_text, separators='sentence'),
                additional_columns={'text': pxt.String | None},
            )

    def test_view_model_with_iterator(self, make_catalog_path: Callable[[str], str]) -> None:
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTableModel(TableModel, name='test_table'):
            id: pxt.Int
            name: pxt.String | None
            value: pxt.Float | None
            image: pxt.Image | None

        class ExampleViewModel(
            TableModel,
            name='test_view',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.image, (256, 256)),
        ):
            view_col_1 = ExampleTableModel.value + 1
            view_col_2 = tile.rotate(90)  # type: ignore[name-defined]  # tile is defined by the iterator

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
        # and indices) align with the model-based ones. (The models default to has_default_idxs=True, including
        # for views, whereas pxt.create_view() defaults to False; pass it explicitly to match.)
        tbl2 = pxt.create_table(
            p('test_table_2'),
            {'id': pxt.Int, 'name': pxt.String | None, 'value': pxt.Float | None, 'image': pxt.Image | None},
        )

        view2 = pxt.create_view(p('test_view_2'), tbl2, iterator=pxtf.image.tile_iterator(tbl2.image, (256, 256)))
        view2.add_computed_column(view_col_1=(tbl2.value + 1))
        view2.add_computed_column(view_col_2=view2.tile.rotate(90))

        view_from_query2 = pxt.create_view(
            p('test_view_from_query_2'),
            tbl2.select(tbl2.id, tbl2.image, rot=tbl2.image.rotate(90)),
            iterator=pxtf.image.tile_iterator(tbl2.image, (256, 256)),
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
        """diff_all() reports added/dropped columns and an iterator mismatch against already-created tables."""
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        root = p('')

        # A base with a table model and a view model, 4 columns each. has_default_idxs=False keeps the diff
        # focused on columns and the iterator (default indexes are not part of a model's declared __indexes__).
        TableModel = pxt.model_base()

        # In V2, test_table exercises every alterable column property across a few kept columns: score (type),
        # image (media_validation), and the computed derived (value expression, stored, comment, custom_metadata).
        # It also changes table-level properties (comment, custom_metadata).
        class ExampleTable(
            TableModel, name='test_table', has_default_idxs=False, comment='before', custom_metadata={'origin': 'v1'}
        ):
            id: pxt.Int
            name: pxt.String | None
            value: pxt.Float | None
            image: pxt.Image | None
            score: pxt.Float | None
            derived = Column(value=id + 1, comment='before', custom_metadata={'v': 1})

            __indexes__ = [
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=768), name='idx1'),
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=512), name='idx2'),
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=1024), name='idx3'),
            ]

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

        class ExampleQueryView(
            TableModel,
            name='test_query_view',
            base=ExampleTable.select(ExampleTable.id, id_copy=ExampleTable.id, plusone=(ExampleTable.value + 1))
            .where(ExampleTable.id > 0)
            .sample(n=10, seed=1),
        ):
            fc1 = ExampleTable.id + 1

        # Created as a view; V2 redeclares it as a table, producing a kind mismatch.
        class ExampleKind(TableModel, name='test_kind', base=ExampleTable):
            kc1 = ExampleTable.value + 1
            kc2 = ExampleTable.value + 2

        TableModel.create_all(root)

        # Re-diffing the original models reports no differences (in particular, the view's iterator round-trips).
        assert all(d['resolution'] == 'up_to_date' for d in TableModel.get_model_diff(root).values())
        with capture_console_output() as out:
            TableModel.diff_all(root)
        assert out.getvalue().strip() == 'Catalog is up to date.'

        # A fresh base whose models correspond to the created tables (same names), but with: two columns added and
        # two dropped in the table, and a mismatched iterator (128 vs. 256) in the view.
        TableModelV2 = pxt.model_base()

        class ExampleTableV2(
            TableModelV2, name='test_table', has_default_idxs=False, comment='after', custom_metadata={'origin': 'v2'}
        ):
            id: pxt.Int
            image = Column(type=pxt.Image | None, media_validation='on_read')  # kept, media_validation changed
            score: pxt.Int | None  # kept, but its type changed (Float -> Int)
            derived = Column(value=id + 100, stored=False, comment='after', custom_metadata={'v': 2})  # 4 props changed
            extra1: pxt.Int | None  # added
            extra2: pxt.String | None  # added
            # 'name' and 'value' dropped

            __indexes__ = [
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=768), name='idx1'),  # kept
                EmbeddingIndex(
                    image, embedding=dummy_embedding.using(n=1024), precision='fp32', name='idx3'
                ),  # changed
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=256), name='idx4'),  # added
                # 'idx2' dropped
            ]

        class ExampleViewV2(
            TableModelV2,
            name='test_view',
            base=ExampleTableV2,
            iterator=pxtf.image.tile_iterator(ExampleTableV2.image, (128, 128)),  # mismatched tile size
        ):
            vc1 = ExampleTableV2.id + 1
            vc2 = ExampleTableV2.id + 2
            vextra1: pxt.Int | None
            vextra2: pxt.String | None

        class ExampleQueryViewV2(
            TableModelV2,
            name='test_query_view',
            base=ExampleTableV2.select(ExampleTableV2.id, ExampleTableV2.extra1, plustwo=(ExampleTableV2.id + 2))
            .where(ExampleTableV2.id > 5)
            .sample(n=20, seed=2),
        ):
            id_copy = Column(value=ExampleTableV2.id, stored=False)
            fc1 = ExampleTableV2.id + 1

        # Redeclares 'test_kind' (created above as a view) as a table, with the same columns; only the kind differs.
        class ExampleKindV2(TableModelV2, name='test_kind'):
            kc1: pxt.Float | None
            kc2: pxt.Float | None

        # A model with no corresponding table in the catalog; it would be created.
        class ExampleNewV2(TableModelV2, name='test_new'):
            id: pxt.Int
            data: pxt.String | None

        with capture_console_output() as out:
            TableModelV2.diff_all(root)
        assert (
            out.getvalue().strip()
            == textwrap.dedent("""
            Table 'test_table' (from model `ExampleTableV2`) has differences:
              the following table properties have changed (FATAL):
                comment: model='after', existing='before'
                custom_metadata: model={'origin': 'v2'}, existing={'origin': 'v1'}
              the following columns have altered properties (FATAL):
                'derived' value: model='id + 100', existing='id + 1'
                'derived' stored: model=False, existing=True
                'derived' comment: model='after', existing='before'
                'derived' custom_metadata: model={'v': 2}, existing={'v': 1}
                'image' media_validation: model='on_read', existing='on_write'
                'score' type: model='Int | None', existing='Float | None'
              the following columns are new to the model, and will be ADDED:
                'extra1' = {'type': Int | None}
                'extra2' = {'type': String | None}
              the following columns are no longer in the model, and will be DROPPED:
                'name'
                'value'
              the following indexes are new to the model, and will be ADDED:
                EmbeddingIndex(column=image, embedding=dummy_embedding(text, n=256), name='idx4')
              the following indexes are no longer in the model, and will be DROPPED:
                'idx2'
              the following named indexes have altered properties (FATAL):
                'idx3'
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
            View 'test_query_view' (from model `ExampleQueryViewV2`) has differences:
              filter mismatch (FATAL):
                model filter   : id > 5
                existing filter: id > 0
              sample mismatch (FATAL):
                model sample   : sample(n=20, n_per_stratum=None, fraction=None, seed=2, [])
                existing sample: sample(n=10, n_per_stratum=None, fraction=None, seed=1, [])
              the following columns are new to the model, and will be ADDED:
                'extra1' = {'value': extra1, 'stored': False}
                'plustwo' = {'value': id + 2, 'stored': True}
              the following columns are no longer in the model, and will be DROPPED:
                'plusone'
            Table 'test_kind' (from model `ExampleKindV2`) has differences:
              kind mismatch (FATAL): `ExampleKindV2` specifies a table, but 'test_kind' is a view
              the following columns have altered properties (FATAL):
                'kc1' value: model=None, existing='value + 1'
                'kc2' value: model=None, existing='value + 2'
            Table 'test_new' (from model `ExampleNewV2`) does not yet exist, and will be CREATED.
            """).strip()
        )

        # `get_model_diff()` returns the same information in structured form (the source of the report above).
        diffs = TableModelV2.get_model_diff(root)

        # every diff records the table it was computed against, so that an update can tell whether the catalog
        # moved underneath it; a model whose table doesn't exist yet has nothing to record
        for d in diffs.values():
            if not d['exists']:
                assert d['tbl_id'] is None and d['schema_versions'] is None
                continue
            md = pxt.get_table(d['path']).get_metadata()
            assert d['tbl_id'] == md['id']
            assert d['schema_versions'][md['id']] == md['schema_version']

        # those two are the catalog's ids and versions, so they are checked above instead of spelled out below
        without_identity = {
            name: {k: v for k, v in d.items() if k not in ('tbl_id', 'schema_versions')} for name, d in diffs.items()
        }
        assert without_identity == {
            'test_table': {
                'path': p('test_table'),
                'model_cls': 'ExampleTableV2',
                'kind': 'table',
                'exists': True,
                'resolution': 'unsupported',
                'ops': [
                    {
                        'target': 'table',
                        'name': 'comment',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': 'after',
                        'existing': 'before',
                        'description': "table property 'comment': model='after', existing='before'",
                        'details': {},
                    },
                    {
                        'target': 'table',
                        'name': 'custom_metadata',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {'origin': 'v2'},
                        'existing': {'origin': 'v1'},
                        'description': "table property 'custom_metadata': "
                        "model={'origin': 'v2'}, existing={'origin': 'v1'}",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'derived',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {
                            'value': 'id + 100',
                            'stored': False,
                            'comment': 'after',
                            'custom_metadata': {'v': 2},
                        },
                        'existing': {
                            'value': 'id + 1',
                            'stored': True,
                            'comment': 'before',
                            'custom_metadata': {'v': 1},
                        },
                        'description': "column 'derived' has altered properties: "
                        'value, stored, comment, custom_metadata',
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'image',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {'media_validation': 'on_read'},
                        'existing': {'media_validation': 'on_write'},
                        'description': "column 'image' has altered properties: media_validation",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'score',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {'type': 'Int | None'},
                        'existing': {'type': 'Float | None'},
                        'description': "column 'score' has altered properties: type",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'extra1',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': Int | None}",
                        'existing': None,
                        'description': "column 'extra1' will be added",
                        'details': {'type': 'Int | None'},
                    },
                    {
                        'target': 'column',
                        'name': 'extra2',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': String | None}",
                        'existing': None,
                        'description': "column 'extra2' will be added",
                        'details': {'type': 'String | None'},
                    },
                    {
                        'target': 'column',
                        'name': 'name',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "column 'name' will be dropped",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'value',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "column 'value' will be dropped",
                        'details': {},
                    },
                    {
                        'description': "named index 'idx3' has altered properties",
                        'details': {'index_ref': {'index_type': 'embedding', 'columns': ['image'], 'name': 'idx3'}},
                        'existing': {
                            'columns': ['image'],
                            'index_type': 'embedding',
                            'name': 'idx3',
                            'parameters': {
                                'embedding': 'dummy_embedding(image, n=1024)',
                                'embedding_functions': [
                                    'dummy_embedding(text, n=1024)',
                                    'dummy_embedding(img, n=1024)',
                                ],
                                'metric': 'cosine',
                                'precision': 'fp16',
                            },
                        },
                        'model': 'EmbeddingIndex(column=image, embedding=dummy_embedding(text, '
                        "n=1024), precision='fp32', name='idx3')",
                        'name': 'idx3',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'target': 'index',
                    },
                    {
                        'target': 'index',
                        'name': 'idx4',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "EmbeddingIndex(column=image, embedding=dummy_embedding(text, n=256), name='idx4')",
                        'existing': None,
                        'description': "EmbeddingIndex 'idx4' will be added",
                        'details': {'index_ref': {'index_type': 'embedding', 'columns': ['image'], 'name': 'idx4'}},
                    },
                    {
                        'target': 'index',
                        'name': 'idx2',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "index 'idx2' will be dropped",
                        'details': {'index_ref': {'index_type': 'embedding', 'columns': ['image'], 'name': 'idx2'}},
                    },
                ],
            },
            'test_view': {
                'path': p('test_view'),
                'model_cls': 'ExampleViewV2',
                'kind': 'view',
                'exists': True,
                'resolution': 'unsupported',
                'ops': [
                    {
                        'target': 'table',
                        'name': 'iterator',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': 'tile_iterator(image, [128, 128])',
                        'existing': 'tile_iterator(image, [256, 256])',
                        'description': "iterator mismatch: model='tile_iterator(image, [128, 128])', "
                        "existing='tile_iterator(image, [256, 256])'",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'vextra1',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': Int | None}",
                        'existing': None,
                        'description': "column 'vextra1' will be added",
                        'details': {'type': 'Int | None'},
                    },
                    {
                        'target': 'column',
                        'name': 'vextra2',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': String | None}",
                        'existing': None,
                        'description': "column 'vextra2' will be added",
                        'details': {'type': 'String | None'},
                    },
                    {
                        'target': 'column',
                        'name': 'vc3',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "column 'vc3' will be dropped",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'vc4',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "column 'vc4' will be dropped",
                        'details': {},
                    },
                ],
            },
            'test_query_view': {
                'path': p('test_query_view'),
                'model_cls': 'ExampleQueryViewV2',
                'kind': 'view',
                'exists': True,
                'resolution': 'unsupported',
                'ops': [
                    {
                        'target': 'table',
                        'name': 'view_filter',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': 'id > 5',
                        'existing': 'id > 0',
                        'description': "view_filter mismatch: model='id > 5', existing='id > 0'",
                        'details': {},
                    },
                    {
                        'target': 'table',
                        'name': 'view_sample',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': 'sample(n=20, n_per_stratum=None, fraction=None, seed=2, [])',
                        'existing': 'sample(n=10, n_per_stratum=None, fraction=None, seed=1, [])',
                        'description': 'view_sample mismatch: '
                        "model='sample(n=20, n_per_stratum=None, fraction=None, seed=2, [])', "
                        "existing='sample(n=10, n_per_stratum=None, fraction=None, seed=1, [])'",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'extra1',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'value': extra1, 'stored': False}",
                        'existing': None,
                        'description': "column 'extra1' will be added",
                        'details': {'type': 'Int | None', 'value': 'extra1'},
                    },
                    {
                        'target': 'column',
                        'name': 'plustwo',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'value': id + 2, 'stored': True}",
                        'existing': None,
                        'description': "column 'plustwo' will be added",
                        'details': {'type': 'Int', 'value': 'id + 2'},
                    },
                    {
                        'target': 'column',
                        'name': 'plusone',
                        'op': 'drop',
                        'severity': 'destructive',
                        'model': None,
                        'existing': None,
                        'description': "column 'plusone' will be dropped",
                        'details': {},
                    },
                ],
            },
            'test_kind': {
                'path': p('test_kind'),
                'model_cls': 'ExampleKindV2',
                'kind': 'table',
                'exists': True,
                'resolution': 'unsupported',
                'ops': [
                    {
                        'target': 'table',
                        'name': 'kind',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': 'table',
                        'existing': 'view',
                        'description': "`ExampleKindV2` specifies a table, but 'test_kind' is a view",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'kc1',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {'value': None},
                        'existing': {'value': 'value + 1'},
                        'description': "column 'kc1' has altered properties: value",
                        'details': {},
                    },
                    {
                        'target': 'column',
                        'name': 'kc2',
                        'op': 'alter',
                        'severity': 'unsupported',
                        'model': {'value': None},
                        'existing': {'value': 'value + 2'},
                        'description': "column 'kc2' has altered properties: value",
                        'details': {},
                    },
                ],
            },
            'test_new': {
                'path': p('test_new'),
                'model_cls': 'ExampleNewV2',
                'kind': 'table',
                'exists': False,
                'resolution': 'create',
                'ops': [
                    {
                        'target': 'column',
                        'name': 'data',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': String | None}",
                        'existing': None,
                        'description': "column 'data' will be added",
                        'details': {'type': 'String | None'},
                    },
                    {
                        'target': 'column',
                        'name': 'id',
                        'op': 'add',
                        'severity': 'additive',
                        'model': "{'type': Int}",
                        'existing': None,
                        'description': "column 'id' will be added",
                        'details': {'type': 'Int'},
                    },
                ],
            },
        }

        with pxt_raises(
            excs.ErrorCode.SCHEMA_MISMATCH,
            match=r'One or more tables cannot be updated, because their models are inconsistent with the existing',
        ):
            TableModelV2.update_all(root)

    def test_update_all(self, make_catalog_path: Callable[[str], str]) -> None:
        """`update_all()` applies purely additive changes (new columns and indexes) to existing tables."""
        skip_test_if_not_installed('imagehash')

        p = make_catalog_path
        root = p('')

        TableModel = pxt.model_base()

        class ExampleTable(TableModel, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            image: pxt.Image | None

            __indexes__ = [EmbeddingIndex(image, embedding=dummy_embedding.using(n=768), name='embed_a')]

        class ExampleView(TableModel, name='test_view', base=ExampleTable):
            vc1 = ExampleTable.value + 1

        class ExampleQueryView(
            TableModel,
            name='test_query_view',
            base=ExampleTable.select(ExampleTable.id, ExampleTable.value, plusone=(ExampleTable.value + 1))
            .where(ExampleTable.value > 0.5)
            .sample(n=10, seed=1),
        ):
            fc1 = ExampleTable.id + 1

        TableModel.create_all(root)

        images = get_image_files()
        ExampleTable.insert([{'id': 1, 'value': 1.0, 'image': images[0]}, {'id': 2, 'value': 2.0, 'image': images[1]}])

        # A fresh base whose models match the created tables plus purely additive changes: new columns and new
        # embedding/b-tree indexes on the table, new columns on the views. No drops, no kind/iterator mismatch.
        TableModelV2 = pxt.model_base()

        class ExampleTableV2(TableModelV2, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            image: pxt.Image | None
            plus_ten = value + 10  # new computed column
            plus_fifteen = plus_ten + 5  # new computed column that depends on a new column
            plus_sixty = plus_fifteen + 45
            note: pxt.String | None  # new (plain) column
            new_image: pxt.Image | None

            __indexes__ = [
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=768), name='embed_a'),  # kept
                EmbeddingIndex(image, embedding=dummy_embedding.using(n=512), name='embed_b'),  # new index
                EmbeddingIndex(new_image, embedding=dummy_embedding.using(n=256), name='embed_c'),  # new on new column
                BtreeIndex(id),  # new
            ]

        class ExampleViewV2(TableModelV2, name='test_view', base=ExampleTableV2):
            vc1 = ExampleTableV2.value + 1
            vc2 = ExampleTableV2.value + 2  # new column
            plus_twenty = ExampleTableV2.plus_ten + 10  # new column that depends on a new column of the base table

        class ExampleQueryViewV2(
            TableModelV2,
            name='test_query_view',
            base=ExampleTableV2.select(
                ExampleTableV2.id,
                ExampleTableV2.value,
                ExampleTableV2.note,
                plusone=(ExampleTableV2.value + 1),
                plustwo=(ExampleTableV2.value + 2),
            )
            .where(ExampleTableV2.value > 0.5)
            .sample(n=10, seed=1),
        ):
            fc1 = ExampleTableV2.id + 1

        # Purely additive, so no `allow_destructive` needed.
        TableModelV2.update_all(root)

        # The new columns and indexes are present on the table; the new column is present on the view.
        tbl_md = ExampleTableV2.get_metadata()
        assert 'plus_ten' in tbl_md['columns']
        assert 'note' in tbl_md['columns']
        assert set(tbl_md['indexes'].keys()) == {'embed_a', 'embed_b', 'embed_c', 'idx0'}
        assert tbl_md['indexes']['idx0']['index_type'] == 'btree'
        assert tbl_md['indexes']['idx0']['columns'] == ['id']
        assert 'vc2' in ExampleViewV2.get_metadata()['columns']

        # The new computed column is backfilled for the existing rows.
        tbl = ExampleTableV2.table
        res = tbl.order_by(tbl.id).select(tbl.id, tbl.plus_ten).collect()
        assert res['plus_ten'] == [11.0, 12.0]

        # A third base that both drops and adds columns, on the table and the view. The dropped columns
        # (`plus_*`, `note`, `vc1`) have no dependents, so the only obstacle is that dropping is destructive.
        TableModelV3 = pxt.model_base()

        class ExampleTableV3(TableModelV3, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            image: pxt.Image | None
            doubled = value * 2  # added
            label: pxt.String | None  # added
            # 'plus_ten', 'plus_fifteen', and 'note' dropped

            __indexes__ = [EmbeddingIndex(image, embedding=dummy_embedding.using(n=512), name='embed_b')]
            # embed_a, embed_c, and idx0 are dropped

        class ExampleViewV3(TableModelV3, name='test_view', base=ExampleTableV3):
            vc2 = ExampleTableV3.value + 2  # kept
            vc3 = ExampleTableV3.value + 3  # added
            # 'vc1' dropped

        class ExampleQueryViewV3(
            TableModelV3,
            name='test_query_view',
            # 'note' and 'plusone' dropped from the query
            base=ExampleTableV3.select(ExampleTableV3.id, ExampleTableV3.value, plustwo=(ExampleTableV3.value + 2))
            .where(ExampleTableV3.value > 0.5)
            .sample(n=10, seed=1),
        ):
            fc1 = ExampleTableV3.id + 1

        # Refuses without opt-in, since columns are being dropped.
        with pxt_raises(excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE, match='destructive'):
            TableModelV3.update_all(root)

        # Succeeds with the opt-in.
        TableModelV3.update_all(root, allow_destructive=True)

        tbl_md = ExampleTableV3.get_metadata()
        assert {'doubled', 'label'} <= set(tbl_md['columns'].keys())
        assert not ({'plus_ten', 'note'} & set(tbl_md['columns'].keys()))
        assert set(tbl_md['indexes'].keys()) == {'embed_b'}
        view_md = ExampleViewV3.get_metadata()
        assert 'vc3' in view_md['columns'] and 'vc1' not in view_md['columns']

        # Try inserting something at the end of all the updates.
        images = get_image_files()
        rows = [
            {'id': 3, 'value': 3.0, 'image': images[2], 'label': 'three'},
            {'id': 4, 'value': 4.0, 'image': images[3], 'label': 'four'},
        ]
        ExampleTableV3.insert(rows)

        # the sample view guarantees no row order, so order the query rather than the result
        v = ExampleQueryViewV3
        res = v.order_by(v.plustwo).collect()
        assert res['plustwo'] == [3.0, 4.0, 5.0, 6.0]

    def test_update_all_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        """`update_all()` raises an error if a model's schema is inconsistent with the existing table."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTable(TableModel, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            img: pxt.Image | None

            __indexes__ = [EmbeddingIndex(img, embedding=dummy_embedding.using(n=768), name='idx')]

        TableModel.create_all(p(''))

        # Add a view manually, not visible to the model_base
        v = pxt.create_view(p('test_view'), ExampleTable.table)
        v.add_computed_column(vc1=(ExampleTable.value + 1))
        v.add_computed_column(vc2=(ExampleTable.img.embedding()))  # type: ignore[attr-defined]

        TableModelV2 = pxt.model_base()

        # Drop the `value` column, but without dropping the dependent column `vc1` in the manually added view
        class ExampleTableV2(TableModelV2, name='test_table'):
            id: pxt.Int
            img: pxt.Image | None

            __indexes__ = [EmbeddingIndex(img, embedding=dummy_embedding.using(n=768), name='idx')]

        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r"Column 'value' was removed from the model for 'test_table', but cannot be dropped "
            r'because the following depend on it:\nvc1',
        ):
            TableModelV2.update_all(p(''), allow_destructive=True)

        TableModelV3 = pxt.model_base()

        # Drop the `idx` index, but without dropping the dependent column `vc1` in the manually added view
        class ExampleTableV3(TableModelV3, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None
            img: pxt.Image | None

        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r"Index 'idx' was removed from the model for 'test_table', but cannot be dropped "
            r'because the following depend on it:\nvc2',
        ):
            TableModelV3.update_all(p(''), allow_destructive=True)

    def test_drop_col_with_view_index(self, make_catalog_path: Callable[[str], str]) -> None:
        """update_all() cannot drop a column that a view's index is built on."""
        p = make_catalog_path
        base = pxt.create_table(p('base_t'), {'c0': pxt.String | None, 'c1': pxt.String | None})
        v = pxt.create_view(p('view_t'), base)
        v.add_embedding_index('c0', idx_name='v_idx', embedding=dummy_embedding.using(n=32))

        TableModel = pxt.model_base()

        # the view isn't part of the model, so its index survives the update and still depends on the base column
        class BaseV2(TableModel, name='base_t'):
            c1: pxt.String | None

        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r"Column 'c0' was removed from the model for 'base_t', but cannot be dropped "
            r"because the following depend on it:\nindex 'v_idx' on 'view_t'",
        ):
            TableModel.update_all(p(''), allow_destructive=True)
        assert 'c0' in pxt.get_table(p('base_t')).columns()

        # dropping the index first unblocks it: nothing is left depending on the column
        v.drop_embedding_index(idx_name='v_idx')
        TableModel.update_all(p(''), allow_destructive=True)
        assert pxt.get_table(p('base_t')).columns() == ['c1']

    def test_update_all_view_predicate(self, make_catalog_path: Callable[[str], str]) -> None:
        """update_all() cannot drop a column that a view's predicate references."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class ExampleTable(TableModel, name='test_table'):
            id: pxt.Int
            value: pxt.Float | None

        TableModel.create_all(p(''))
        ExampleTable.insert([{'id': 1, 'value': 5.0}])

        # Add views manually, not visible to the model_base; both filter on the same column
        t = ExampleTable.table
        pxt.create_view(p('test_view'), t.where(t.value > 1.0))
        pxt.create_view(p('other_view'), t.where(t.value > 2.0))

        TableModelV2 = pxt.model_base()

        # Drop the value column that the views' predicates filter on
        class ExampleTableV2(TableModelV2, name='test_table'):
            id: pxt.Int

        with pxt_raises(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            match=r'Cannot drop the following columns, because view predicates depend on them:\n'
            r'column: value, view: other_view, predicate: value > 2.0\n'
            r'column: value, view: test_view, predicate: value > 1.0',
        ):
            TableModelV2.update_all(p(''), allow_destructive=True)

        assert 'value' in ExampleTable.table.columns()

    def test_table_model_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        """Reproduce each error condition raised by pixeltable.catalog.model."""
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
            id: pxt.Int | None

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
                name: pxt.Int | None = Column(type=pxt.String | None)  # type: ignore[assignment]

        with pxt_raises(
            excs.ErrorCode.INVALID_ARGUMENT,
            match=r'model `InvalidBase`: `base` must be a valid base table reference '
            r'\(another Pixeltable model, or a query over a model\).',
        ):

            class InvalidBase(TableModel, name='invalid_base', base=42):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_ARGUMENT,
            match=r'`base` select\(\) list may contain only direct column references or named expressions, '
            r'but contains an anonymous compound expression: id \+ 1',
        ):

            class InvalidBaseQuery(
                TableModel, name='invalid_base_query', base=ValidTableModel.select(ValidTableModel.id + 1)
            ):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'Pixeltable schemas must be direct subclasses of a model_base\(\).'
        ):

            class SubclassedModel(ValidTableModel, name='subclassed_model'):
                x: pxt.Int | None

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r"has name 'dup_name', but that name was previously used by `FirstDup`"
        ):

            class FirstDup(TableModel, name='dup_name'):
                id: pxt.Int | None

            class SecondDup(TableModel, name='dup_name'):
                id: pxt.Int | None

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'must define `type` or `value`, but not both'):

            class BadColSpec(TableModel, name='bad_col_spec'):
                id: pxt.Int | None
                bad = Column()

        # `references columns that are not in the model's scope` is raised at `create()` time, when a computed
        # column refers to a column outside the model (here, a column belonging to a different, unbound model).
        class OtherModel(TableModel, name='other_model'):
            x: pxt.Int | None

        class RefsOutOfScope(TableModel, name='refs_out_of_scope'):
            y: pxt.Int | None
            bad = OtherModel.x + 1

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"references columns that are not in the model's scope"):
            RefsOutOfScope._create(p(''))

        # rejected by the class definition itself, before _create() is ever reached
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'cannot combine `has_default_idxs=True`'):

            class DefaultsPlusBtree(TableModel, name='defaults_plus_btree_table', has_default_idxs=True):
                id: pxt.Int
                name: pxt.String | None

                __indexes__ = [BtreeIndex(name)]

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Column 'plus': duplicate definition"):

            class DuplicateColumn(TableModel, name='duplicate_column'):
                id: pxt.Int
                plus = id + 1
                plus = id + 2

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'model `IndexesNotSequence`: `__indexes__` must be a sequence of'
        ):

            class IndexesNotSequence(TableModel, name='indexes_not_sequence'):
                id: pxt.Int

                __indexes__ = 170

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'model `InvalidIndex`: `__indexes__` must be a sequence of'
        ):

            class InvalidIndex(TableModel, name='invalid_index'):
                id: pxt.Int

                __indexes__ = [BtreeIndex(id), 'a string is definitely not an index']

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'model `InvalidBtreeIndex`: Invalid BtreeIndex column reference: 42'
        ):

            class InvalidBtreeIndex(TableModel, name='invalid_btree_index'):
                id: pxt.Int

                __indexes__ = [BtreeIndex(42)]

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r'model `InvalidEmbeddingIndex`: Invalid EmbeddingIndex column reference: 42',
        ):

            class InvalidEmbeddingIndex(TableModel, name='invalid_embedding_index'):
                id: pxt.Int

                __indexes__ = [EmbeddingIndex(42, embedding=dummy_embedding.using(n=768))]

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r"model `InvalidIndexName`: Invalid EmbeddingIndex name: 'not an identifier'",
        ):

            class InvalidIndexName(TableModel, name='invalid_index_name'):
                id: pxt.Int

                __indexes__ = [EmbeddingIndex(id, embedding=dummy_embedding.using(n=768), name='not an identifier')]

        with pxt_raises(
            pxt.ErrorCode.INVALID_SCHEMA, match=r"model `DuplicateBtreeIndex`: multiple B-tree indexes for column 'id'."
        ):

            class DuplicateBtreeIndex(TableModel, name='duplicate_btree_index'):
                id: pxt.Int
                name: pxt.String | None
                img: pxt.Image | None
                unstored = Column(value=img.rotate(90), stored=False)

                __indexes__ = [BtreeIndex(id), BtreeIndex(id)]

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r"model `UnnamedEmbeddingIndexes`: column 'text' has multiple embedding indexes; "
            'they must be given explicit names',
        ):

            class UnnamedEmbeddingIndexes(TableModel, name='unnamed_embedding_indexes'):
                text: pxt.String | None

                __indexes__ = [
                    EmbeddingIndex(text, embedding=dummy_embedding.using(n=768)),
                    EmbeddingIndex(text, embedding=dummy_embedding.using(n=1024)),
                ]

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'model `DuplicateNamedIndex`: index names must be unique'
        ):

            class DuplicateNamedIndex(TableModel, name='duplicate_named_index'):
                text: pxt.String | None
                img: pxt.Image | None

                __indexes__ = [
                    EmbeddingIndex(text, embedding=dummy_embedding.using(n=768), name='dup_idx_name'),
                    EmbeddingIndex(img, embedding=dummy_embedding.using(n=768), name='dup_idx_name'),
                ]

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r"Column 'bad': invalid value"):

            class InvalidValue(TableModel, name='invalid_value'):
                id: pxt.Int | None
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
            img: pxt.Image | None

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

        t = pxt.create_table(p('test_table'), {'id': pxt.Int, 'name': pxt.String | None, 'img': pxt.Image | None})
        _ = pxt.create_view(p('test_view'), t)
        _ = pxt.create_snapshot(p('test_snapshot'), t)

        t_ok = pxt.create_table(p('ok_table'), {'id': pxt.Int, 'name': pxt.String | None, 'img': pxt.Image | None})
        _ = pxt.create_view(p('test_view_2'), t_ok)
        _ = pxt.create_view(p('test_iter_view'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))
        _ = pxt.create_view(p('test_iter_view_2'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))
        _ = pxt.create_view(p('test_iter_view_3'), t_ok, iterator=pxtf.image.tile_iterator(t_ok.img, (256, 256)))

        class BadTableModel(TableModel, name='test_view'):
            id: pxt.Int

        class ExampleTableModel(TableModel, name='ok_table'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None

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

        # `diff_all()` reports every mismatch between a model and its existing table at once.
        with capture_console_output() as out:
            TableModel.diff_all(p(''))
        report = out.getvalue()

        # Kind mismatches: table-vs-view, view-vs-table, view-vs-snapshot.
        assert "kind mismatch (FATAL): `BadTableModel` specifies a table, but 'test_view' is a view" in report
        assert "kind mismatch (FATAL): `BadViewModel` specifies a view, but 'test_table' is a table" in report
        assert "kind mismatch (FATAL): `BadViewModel2` specifies a view, but 'test_snapshot' is a snapshot" in report

        # Iterator mismatches: a differing iterator, a missing one, and an extraneous one.
        assert 'tile_iterator(img, [128, 128])' in report  # IteratorMismatch: model's iterator
        assert 'model iterator   : None' in report  # MissingIterator: model has no iterator
        assert 'existing iterator: None' in report  # ExtraneousIterator: existing view has no iterator

        # Models that match their existing tables produce no differences.
        assert '`ExampleTableModel`' not in report
        assert '`GoodIterViewModel`' not in report

        # `create_all()` only creates; it refuses to run when any existing table differs from its model.
        with pxt_raises(excs.ErrorCode.SCHEMA_MISMATCH, match=r'Call `update_all\(\)` instead'):
            TableModel.create_all(p(''))
