import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import Column

from .utils import assert_table_metadata_eq


class TestTableModel:
    def test_table_model(self, uses_db: None) -> None:
        class ExampleTableModel(pxt.TableModel):
            __table_name__ = 'test_table'

            id: pxt.Int
            name: pxt.String
            value: pxt.Float
            descr = pxtf.string.format('Name: {name}', name=Column.name)

        tbl = ExampleTableModel.create()
        assert_table_metadata_eq(
            {
                'base': None,
                'columns': {
                    'descr': {
                        'comment': None,
                        'computed_with': "format('Name: {name}', name=name)",
                        'custom_metadata': None,
                        'defined_in': 'test_table',
                        'depends_on': [('test_table', 'name')],
                        'destination': None,
                        'is_builtin': True,
                        'is_computed': True,
                        'is_iterator_col': False,
                        'is_primary_key': False,
                        'is_stored': True,
                        'media_validation': None,
                        'name': 'descr',
                        'type_': 'Required[String]',
                        'version_added': 1,
                    },
                    'id': {
                        'comment': None,
                        'computed_with': None,
                        'custom_metadata': None,
                        'defined_in': 'test_table',
                        'depends_on': None,
                        'destination': None,
                        'is_builtin': None,
                        'is_computed': False,
                        'is_iterator_col': False,
                        'is_primary_key': False,
                        'is_stored': True,
                        'media_validation': None,
                        'name': 'id',
                        'type_': 'Int',
                        'version_added': 0,
                    },
                    'name': {
                        'comment': None,
                        'computed_with': None,
                        'custom_metadata': None,
                        'defined_in': 'test_table',
                        'depends_on': None,
                        'destination': None,
                        'is_builtin': None,
                        'is_computed': False,
                        'is_iterator_col': False,
                        'is_primary_key': False,
                        'is_stored': True,
                        'media_validation': None,
                        'name': 'name',
                        'type_': 'String',
                        'version_added': 0,
                    },
                    'value': {
                        'comment': None,
                        'computed_with': None,
                        'custom_metadata': None,
                        'defined_in': 'test_table',
                        'depends_on': None,
                        'destination': None,
                        'is_builtin': None,
                        'is_computed': False,
                        'is_iterator_col': False,
                        'is_primary_key': False,
                        'is_stored': True,
                        'media_validation': None,
                        'name': 'value',
                        'type_': 'Float',
                        'version_added': 0,
                    },
                },
                'comment': None,
                'custom_metadata': None,
                'indices': {
                    'idx0': {'columns': ['id'], 'index_type': 'btree', 'name': 'idx0', 'parameters': None},
                    'idx1': {'columns': ['name'], 'index_type': 'btree', 'name': 'idx1', 'parameters': None},
                    'idx2': {'columns': ['value'], 'index_type': 'btree', 'name': 'idx2', 'parameters': None},
                    'idx3': {'columns': ['descr'], 'index_type': 'btree', 'name': 'idx3', 'parameters': None},
                },
                'is_replica': False,
                'is_snapshot': False,
                'is_versioned': True,
                'is_view': False,
                'iterator_call': None,
                'kind': 'table',
                'media_validation': 'on_write',
                'name': 'test_table',
                'path': 'test_table',
                'primary_key': None,
                'schema_version': 1,
                'version': 1,
            },
            tbl.get_metadata(),
        )
