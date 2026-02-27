from typing import Any

import numpy as np
import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    Renames val_t to col_type in Literals, and updates how Literal types are serialized so that col_type is always
    present and always a full ColumnType dict (never a bare type-name string).

    E.g. before: {'val': [[...], [...]], 'val_t': 'ARRAY', '_classname': 'Literal'}
    After: {'val': [[...], [...]], 'col_type': {'_classname': 'ArrayType', 'nullable': False, 'shape': [2, 3],
           'numpy_dtype': 'int64'}, '_classname': 'Literal'}
    """
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get('_classname') != 'Literal' or 'val' not in value:
        return None

    # convert_29 calls Literal.as_dict, so it's possible to encounter already converted literals here.
    if 'col_type' in value:
        assert isinstance(value['col_type'], dict)
        return None

    if 'val_t' not in value:
        # Add col_type for Literals that previously didn't serialize it (int, float, string, bool, json, None/invalid)
        col_type = ts.ColumnType.infer_literal_type(value['val'])
        assert col_type is not None, f'Failed to infer literal type for {value["val"]}'
        value['col_type'] = col_type.as_dict()
        return key, value

    val_t = value.pop('val_t')
    assert isinstance(val_t, str), val_t

    # val_t is a bare string; convert from old string format to new dict format
    match val_t:
        case 'ARRAY':
            array = np.array(value['val'])
            col_type = ts.ArrayType.from_literal(array)
            assert col_type is not None, array
            col_type_dict = col_type.as_dict()
        case 'TIMESTAMP':
            col_type_dict = {'_classname': 'TimestampType', 'nullable': False}
        case 'UUID':
            col_type_dict = {'_classname': 'UUIDType', 'nullable': False}
        case 'DATE':
            col_type_dict = {'_classname': 'DateType', 'nullable': False}
        case 'BINARY':
            col_type_dict = {'_classname': 'BinaryType', 'nullable': False}
        case _:
            raise AssertionError(f'Unrecognized Literal type: {val_t}')

    value['col_type'] = col_type_dict
    return key, value
