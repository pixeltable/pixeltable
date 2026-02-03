from typing import Any

import numpy as np
import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    Before: Array literals simply had 'ARRAY' as their type indicator.
    After: Array literals are accompanied by the full type info, including dtype.
    """
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if 'val' not in value or value.get('val_t', None) != 'ARRAY':
        return None
    array = np.array(value['val'])
    col_type = ts.ArrayType.from_literal(array)
    assert col_type is not None, array
    value['val_t'] = col_type.as_dict()
    return key, value
