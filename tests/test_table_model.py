import numpy as np

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import Column

from .utils import assert_table_metadata_eq


class TestTableModel:
    def test_table_model(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel):
            __table_name__ = 'test_table'

            id: pxt.Required[pxt.Int]
            name: pxt.String
            value: pxt.Float
            incr = Column.value + 1
            descr = pxtf.string.format('Name: {name}', name=Column.name)

        tbl = ExampleTableModel.create()
        metadata = tbl.get_metadata()

        assert {name: info['type_'] for name, info in metadata['columns'].items()} == {
            'id': 'Required[Int]',
            'name': 'String',
            'value': 'Float',
            'incr': 'Float',
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
                    'incr': {
                        'name': 'incr',
                        'type_': 'Float',
                        'version_added': 1,
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
                        'version_added': 2,
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
                    'idx3': {'name': 'idx3', 'columns': ['incr'], 'index_type': 'btree', 'parameters': None},
                    'idx4': {'name': 'idx4', 'columns': ['descr'], 'index_type': 'btree', 'parameters': None},
                },
                'is_versioned': True,
                'is_replica': False,
                'is_view': False,
                'is_snapshot': False,
                'version': 2,
                'schema_version': 2,
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
            # array_slice = Column.arr[:, 1:3]
            # column_property_ref = Column.img.fileurl
            column_ref = Column.name
            comparison = Column.value > 0.0
            compound_predicate = (Column.value > 0.0) & (Column.name != 'test')
            function_call = pxtf.math.floor(Column.value)
            # in_predicate = Column.name.isin(['Alice', 'Bob', 'Charlie'])
            # inline_array = pxt.array([Column.value, Column.value + 1, Column.value + 2])
            # inline_list = [Column.name, Column.img]
            is_null = Column.name == None
            # method_ref = Column.name.upper()
            # similarity = Column.name.similarity('similar string')
            string_add = Column.name + ' suffix'
            string_radd = 'prefix ' + Column.name
            string_mul = Column.name * 3
            string_rmul = 3 * Column.name
            type_cast = Column.arr.astype(pxt.Array[np.float32])

        tbl = AllExprsTableModel.create()
