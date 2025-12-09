from typing import Any

import numpy as np
import sqlalchemy as sql

from pixeltable import type_system as ts
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=43)
def _(engine: sql.engine.Engine) -> None:
    """Converts ArrayTypes by replacing legacy dtype (which was a pxt Type ID) to numpy dtype."""
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get('_classname', None) != 'ArrayType':
        return None
    if 'numpy_dtype' in value:
        return None
    assert 'dtype' in value

    legacy_dtype_val = value['dtype']
    new_dtype: np.dtype | None
    if legacy_dtype_val is None:
        new_dtype = None
    else:
        legacy_dtype = ts.ColumnType.Type(legacy_dtype_val)
        new_dtype = ts.ArrayType.pxt_dtype_to_numpy_dtype.get(legacy_dtype, None)
        if new_dtype is None:
            raise ValueError(f'Unrecognized dtype: {legacy_dtype_val} ({legacy_dtype}) in {key}, {value}')

    del value['dtype']
    if new_dtype is None:
        value['numpy_dtype'] = None
    elif new_dtype == np.str_:
        # str(np.str_) would be something like '<U'
        value['numpy_dtype'] = 'str'
    else:
        value['numpy_dtype'] = str(new_dtype)
    return key, value
