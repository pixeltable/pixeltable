from typing import Any

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exceptions as excs, exprs
from pixeltable.catalog.model import Column, EmbeddingIndex

from .utils import assert_table_metadata_eq, dummy_embedding, pxt_raises


class TestTableModel:
    def test_table_model_basic(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel):
            __table_name__ = 'test_table'

            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = ExampleTableModel.value + 1
            descr = pxtf.string.format('Name: {name}', name=ExampleTableModel.name)

            clip_idx = EmbeddingIndex(ExampleTableModel.img, embedding=dummy_embedding.using(n=768))

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
        class ExampleTableModel(pxt.TableModel):
            __table_name__ = 'test_table'

            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = ExampleTableModel.value + 1
            descr = pxtf.string.format('Name: {name}', name=ExampleTableModel.name)

        with pxt_raises(excs.ErrorCode.PATH_NOT_FOUND, match="Path 'test_table' does not exist"):
            _ = ExampleTableModel.id

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
        class AllExprsTableModel(pxt.TableModel):
            __table_name__ = 'all_exprs_table'

            id: pxt.Int
            name: pxt.String
            value: pxt.Float
            arr: pxt.Array
            img: pxt.Image
            arith_add = Column.value + 1
            arith_radd = 1 + Column.value
            arith_mul = Column.value * 2
            arith_rmul = 2 * Column.value
            array_slice = Column.arr[:, 1:3]
            column_property_ref = Column.img.fileurl
            column_ref = Column.name
            comparison = Column.value > 0.0
            compound_predicate = (Column.value > 0.0) & (Column.name != 'test')
            function_call = pxtf.math.floor(Column.value)
            in_predicate = Column.name.isin(['Alice', 'Bob', 'Charlie'])
            inline_array = pxt.array([Column.value, Column.value + 1, Column.value + 2])
            inline_dict = {'name': Column.name, 'img': Column.img}  # noqa: RUF012
            inline_list = [Column.name, Column.img]  # noqa: RUF012
            is_null = Column.name == None
            method_ref = Column.name.upper()
            # similarity = Column.name.similarity('similar string')
            string_add = Column.name + ' suffix'
            string_radd = 'prefix ' + Column.name
            string_mul = Column.name * 3
            string_rmul = 3 * Column.name
            type_cast = Column.arr.astype(pxt.Array[np.float32])  # type: ignore[misc]

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
            'type_cast': 'Required[Array[float32]]',
            'value': 'Float',
        }

    @pytest.mark.parametrize('spec_type', ['model', 'name', 'query', 'table'])
    def test_view_model(self, spec_type: str, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel):
            __table_name__ = 'test_table'

            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            img: pxt.Image
            incr = Column.value + 1
            descr = pxtf.string.format('Name: {name}', name=Column.name)

            clip_idx = EmbeddingIndex(Column.img, embedding=dummy_embedding.using(n=768))

        _ = ExampleTableModel.create()

        spec: Any
        match spec_type:
            case 'model':
                spec = ExampleTableModel
            case 'name':
                spec = 'test_table'
            case 'query':
                spec = ExampleTableModel.select(ExampleTableModel.value, ExampleTableModel.img).where(
                    ExampleTableModel.value > 0.5  # type: ignore[arg-type]
                )
            case 'table':
                spec = ExampleTableModel.table

        class ExampleViewModel(pxt.ViewModel):
            __table_name__ = 'test_view'
            __base_table__ = spec

            view_col_1: pxt.Image
            view_col_2 = Column.view_col_1.rotate(90)
            view_col_3 = Column.img.rotate(90)  # Also try dereferencing a base table column

            view_idx = EmbeddingIndex(Column.view_col_2, embedding=dummy_embedding.using(n=768))

        _ = ExampleViewModel.create()

        match spec_type:
            case 'model':
                spec = ExampleViewModel
            case 'name':
                spec = 'test_view'
            case 'query':
                spec = ExampleViewModel.where(ExampleViewModel.value > 1.0)
            case 'table':
                spec = ExampleViewModel.table

        class ExampleViewModel2(pxt.ViewModel):
            __table_name__ = 'test_view_2'
            __base_table__ = spec

            subview_col_1 = Column.img.rotate(180)
            subview_col_2 = Column.view_col_1.rotate(270)

        _ = ExampleViewModel2.create()

    def test_table_model_errors(self, uses_db: None) -> None:
        """Reproduce each error condition raised by `pixeltable.catalog.model`."""

        # `_PlaceholderFactory.__getattr__` rejects identifiers that aren't valid column names
        # (e.g., names starting with `_`).
        with pytest.raises(AttributeError, match='Invalid column name'):
            _ = Column._invalid

        # `__base_table__` is not allowed on a TableModel.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='__base_table__ not allowed for a TableModel'):

            class BadBaseTable(pxt.TableModel):
                __table_name__ = 'bad_base_table'
                __base_table__ = 'unused'

        # `__iterator__` is not allowed on a TableModel.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='__iterator__ not allowed for a TableModel'):

            class BadIterTable(pxt.TableModel):
                __table_name__ = 'bad_iter_table'
                __iterator__ = 'unused'

        # `__iterator__` on a ViewModel must be a `GeneratingFunctionCall`.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='must be a valid iterator reference'):

            class BadIterRef(pxt.ViewModel):
                __table_name__ = 'bad_iter_ref'
                __base_table__ = 'unused'
                __iterator__ = 'not a generating function call'

        # Type annotation conflicts with the `type=` argument in `Column()`.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='conflicts with the `type=` argument'):

            class TypeConflict(pxt.TableModel):
                __table_name__ = 'type_conflict'
                name: pxt.Int = Column(type=pxt.String)  # type: ignore[assignment]

        # A TableModel subclass must define `__table_name__`.
        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'TableModel `NoTableName` does not define a __table_name__'
        ):

            class NoTableName(pxt.TableModel):
                id: pxt.Int

        # A ViewModel subclass must define `__table_name__`.
        with pxt_raises(
            excs.ErrorCode.INVALID_SCHEMA, match=r'ViewModel `NoTableNameView` does not define a __table_name__'
        ):

            class NoTableNameView(pxt.ViewModel):
                __base_table__ = 'unused'

        # A ViewModel must define `__base_table__`.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match=r'ViewModel `NoBase` does not define a __base_table__'):

            class NoBase(pxt.ViewModel):
                __table_name__ = 'no_base'

        # `__base_table__` must be a name, an existing Table/Query, or a TableModel/ViewModel class.
        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='Invalid __base_table__'):

            class InvalidBase(pxt.ViewModel):
                __table_name__ = 'invalid_base'
                __base_table__ = 42

        # A computed column references a column that was never declared on the model.
        # This error fires at `create()` time during placeholder substitution, not at class-definition time.
        class UndefinedColRef(pxt.TableModel):
            __table_name__ = 'undef_col_ref'

            name: pxt.String
            bogus = Column.nonexistent + 1

        with pxt_raises(excs.ErrorCode.INVALID_SCHEMA, match='references undefined column'):
            UndefinedColRef.create()
