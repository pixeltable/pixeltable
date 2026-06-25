# ruff: noqa: F821

from typing import Any, Literal

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exceptions as excs, exprs
from pixeltable.catalog.model import Column, EmbeddingIndex

from .utils import assert_table_metadata_eq, dummy_embedding, pxt_raises


class TestTableModel:
    def test_table_model_basic(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = value + 1
            descr = pxtf.string.format('Name: {name}', name=name)

            clip_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        tbl = ExampleTableModel.create()
        metadata: dict[str, Any] = dict(tbl.get_metadata())

        metadata.pop('id')
        metadata.pop('version_created')
        print(metadata)

        assert {name: info['type_'] for name, info in metadata['columns'].items()} == {
            'id': 'Required[Int]',
            'name': 'String',
            'value': 'Float',
            'incr': 'Float',
            'img': 'Image',
            'descr': 'Required[String]',
        }

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
                },
                'indices': {
                    'idx0': {'name': 'idx0', 'columns': ['id'], 'index_type': 'btree', 'parameters': None},
                    'idx1': {'name': 'idx1', 'columns': ['name'], 'index_type': 'btree', 'parameters': None},
                    'idx2': {'name': 'idx2', 'columns': ['value'], 'index_type': 'btree', 'parameters': None},
                    'idx3': {'name': 'idx3', 'columns': ['img'], 'index_type': 'btree', 'parameters': None},
                    'idx4': {'name': 'idx4', 'columns': ['incr'], 'index_type': 'btree', 'parameters': None},
                    'idx5': {'name': 'idx5', 'columns': ['descr'], 'index_type': 'btree', 'parameters': None},
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
                'schema_version': 1,
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

    def test_table_model_query(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = value + 1
            descr = pxtf.string.format('Name: {name}', name=name)

        ExampleTableModel.create()
        ExampleTableModel.insert(
            [
                {'id': 1, 'name': 'Alice', 'value': 3.14},
                {'id': 2, 'name': 'Bob', 'value': 2.71},
                {'id': 3, 'name': 'Charlie', 'value': 1.41},
            ]
        )
        assert isinstance(ExampleTableModel.id, exprs.ColumnRef)
        assert isinstance(ExampleTableModel.descr, exprs.ColumnRef)

        results = (
            ExampleTableModel.select(ExampleTableModel.id, ExampleTableModel.descr)
            .order_by(ExampleTableModel.id)
            .collect()
        )
        assert list(results) == [
            {'id': 1, 'descr': 'Name: Alice'},
            {'id': 2, 'descr': 'Name: Bob'},
            {'id': 3, 'descr': 'Name: Charlie'},
        ]

    def test_all_table_exprs(self, uses_db: None) -> None:
        class AllExprsTableModel(pxt.TableModel, name='all_exprs_table'):
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
            column_property_ref = img.fileurl
            column_ref = name
            comparison = value > 0.0
            compound_predicate = (value > 0.0) & (name != 'test')
            function_call = pxtf.math.floor(value)
            in_predicate = name.isin(['Alice', 'Bob', 'Charlie'])
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
            type_cast = arr.astype(pxt.Array[np.float32])  # type: ignore[misc]

        tbl = AllExprsTableModel.create()

        metadata = tbl.get_metadata()
        assert {name: info['type_'] for name, info in metadata['columns'].items()} == {
            'arith_add': 'Float',
            'arith_mul': 'Float',
            'arith_radd': 'Float',
            'arith_rmul': 'Float',
            'arr': 'Array',
            'array_slice': 'Array',
            'column_property_ref': 'String',
            'column_ref': 'String',
            'comparison': 'Required[Bool]',
            'compound_predicate': 'Required[Bool]',
            'function_call': 'Float',
            'id': 'Int',
            'img': 'Image',
            'in_predicate': 'Required[Bool]',
            'inline_array': 'Required[Array[(3,), float32]]',
            'inline_dict': "Required[Json[{'name': String | None, 'img': Image | None}]]",
            'inline_list': 'Required[Json[(String | None, Image | None)]]',
            'is_null': 'Required[Bool]',
            'method_ref': 'String',
            'name': 'String',
            'string_add': 'String',
            'string_mul': 'String',
            'string_radd': 'String',
            'string_rmul': 'String',
            'type_cast': 'Array[float32]',
            'value': 'Float',
        }

    @pytest.mark.parametrize('create_all', [False, True])
    @pytest.mark.parametrize('spec_type', ['model', 'query'])
    def test_view_model(self, spec_type: Literal['model', 'query'], create_all: bool, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = value + 1
            descr = pxtf.string.format('Name: {name}', name=name)

            clip_idx = EmbeddingIndex(img, embedding=dummy_embedding.using(n=768))

        spec: Any
        match spec_type:
            case 'model':
                spec = ExampleTableModel
            case 'query':
                spec = ExampleTableModel.select(
                    ExampleTableModel.value,
                    ExampleTableModel.img,
                    ExampleTableModel.value + 1,
                    plusone=(ExampleTableModel.value + 1),
                ).where(
                    ExampleTableModel.value > 0.5  # type: ignore[arg-type]
                )

        class ExampleViewModel(pxt.ViewModel, name='test_view', base=spec):
            view_col_1: pxt.Image
            view_col_2 = view_col_1.rotate(90)
            view_col_3 = ExampleTableModel.img.rotate(90)  # Also try dereferencing a base table column

            if spec_type == 'query':
                view_col_4 = plusone + 5  # type: ignore[name-defined]  # Deference a column from the select list

            view_idx = EmbeddingIndex(view_col_2, embedding=dummy_embedding.using(n=768))
            view_idx_on_base_tbl_col = EmbeddingIndex(ExampleTableModel.img, embedding=dummy_embedding.using(n=768))

        match spec_type:
            case 'model':
                spec = ExampleViewModel
            case 'query':
                spec = ExampleViewModel.where(ExampleTableModel.value > 1.0)

        class ExampleViewModel2(pxt.ViewModel, name='test_view_2', base=ExampleViewModel):
            subview_col_1 = ExampleTableModel.img.rotate(180)
            subview_col_2 = ExampleViewModel.view_col_1.rotate(270)
            subview_col_3 = subview_col_2.rotate(30)

        if create_all:
            pxt.create_all()
        else:
            _ = ExampleTableModel.create()
            _ = ExampleViewModel.create()
            _ = ExampleViewModel2.create()

    def test_view_model_with_iterator(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel, name='test_table'):
            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            image: pxt.Image

        class ExampleViewModel(
            pxt.ViewModel,
            name='test_view',
            base=ExampleTableModel,
            iterator=pxtf.image.tile_iterator(ExampleTableModel.image, (256, 256)),
        ):
            view_col_1 = ExampleTableModel.value + 1
            view_col_2 = tile.rotate(90)  # type: ignore[name-defined]  # `tile` is defined by the iterator

        _ = ExampleTableModel.create()
        _ = ExampleViewModel.create()

    def test_table_model_errors(self, uses_db: None) -> None:
        """Reproduce each error condition raised by `pixeltable.catalog.model`."""

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'`name` must be a valid Pixeltable identifier'):

            class BadTableName(pxt.TableModel, name='invalid! table@name'):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r'`base` not allowed for a `TableModel`; `BadBaseTable` must subclass `ViewModel` instead.',
        ):

            class BadBaseTable(pxt.TableModel, name='bad_base_table', base='not_allowed'):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r'`iterator` not allowed for a `TableModel`; `BadIterTable` must subclass `ViewModel` instead.',
        ):

            class BadIterTable(pxt.TableModel, name='bad_iter_table', iterator='not_allowed'):
                pass

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'Empty `TableModel` not allowed.'):

            class EmptyTableModel(pxt.TableModel, name='empty_table'):
                pass

        class ValidTableModel(pxt.TableModel, name='valid_table'):
            id: pxt.Int

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='must be a valid iterator reference'):

            class BadIterRef(pxt.ViewModel, name='bad_iter_ref', base=ValidTableModel, iterator='not a valid iterator'):
                pass

        # Type annotation conflicts with the `type=` argument in `Column()`.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'conflicts with the `type=` argument'):

            class TypeConflict(pxt.TableModel, name='type_conflict'):
                name: pxt.Int = Column(type=pxt.String)  # type: ignore[assignment]

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'TableModel `NoTableName` must specify a `name`.'):

            class NoTableName(pxt.TableModel):
                id: pxt.Int

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'ViewModel `NoTableNameView` must specify a `name`.'):

            class NoTableNameView(pxt.ViewModel, base=ValidTableModel):
                pass

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'ViewModel `NoBase` must specify a `base`.'):

            class NoBase(pxt.ViewModel, name='no_base'):
                pass

        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA,
            match=r'ViewModel `InvalidBase`: `base` must be a valid base table reference '
            r'\(another `TableModel` or `ViewModel`, or a query over a model\).',
        ):

            class InvalidBase(pxt.ViewModel, name='invalid_base', base=42):
                pass
