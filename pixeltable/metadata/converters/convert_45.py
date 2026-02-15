from typing import Any

import numpy as np
import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    Updates how Literal types are serialized.

    E.g. before: {'val': [[...], [...]], 'val_t': 'ARRAY', '_classname': 'Literal'}
    After: {'val': [[...], [...]], 'val_t': {'_classname': 'ArrayType', 'nullable': False, 'shape': [2, 3],
           'numpy_dtype': 'int64'}, '_classname': 'Literal'}
    """
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if 'val' not in value or 'val_t' not in value:
        return None
    updated_val_t = None
    match value['val_t']:
        case 'ARRAY':
            array = np.array(value['val'])
            col_type = ts.ArrayType.from_literal(array)
            assert col_type is not None, array
            updated_val_t = col_type.as_dict()
        case 'TIMESTAMP':
            updated_val_t = {'_classname': 'TimestampType', 'nullable': False}
        case 'UUID':
            updated_val_t = {'_classname': 'UUIDType', 'nullable': False}
        case 'DATE':
            updated_val_t = {'_classname': 'DateType', 'nullable': False}
        case 'BINARY':
            updated_val_t = {'_classname': 'BinaryType', 'nullable': False}

    if updated_val_t is not None:
        value['val_t'] = updated_val_t
        return key, value
    return None
