"""
Pixeltable UDFs for `JsonType`.

Example:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf

t = pxt.get_table(...)
t.select(pxtf.json.make_list(t.json_col)).collect()
```
"""

import itertools
import json
from typing import Any, Iterator, Literal

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import exprs, type_system as ts
from pixeltable.utils.code import local_public_names


@pxt.udf
def dumps(obj: pxt.Json) -> str:
    """
    Serialize a JSON object to a string.

    Equivalent to [`json.dumps()`](https://docs.python.org/3/library/json.html#json.dumps).

    Args:
        obj: A JSON-serializable object (dict, list, or scalar).

    Returns:
        A JSON-formatted string.
    """
    return json.dumps(obj)


@dumps.to_sql
def _(obj: sql.ColumnElement) -> sql.ColumnElement:
    return obj.cast(sql.Text)


@pxt.uda
class make_list(pxt.Aggregator):
    """
    Collects arguments into a list.
    """

    def __init__(self) -> None:
        self.output: list[Any] = []

    def update(self, obj: pxt.Json) -> None:
        if obj is None:
            return
        self.output.append(obj)

    def value(self) -> list[Any]:
        return self.output


@pxt.iterator
def list_iterator(
    elements: list[dict] | None = None, *, method: Literal['strict', 'truncated', 'padded'] = 'strict', **kwargs: list
) -> Iterator[dict]:
    assert (elements is None) != (len(kwargs) == 0)

    if elements is not None:
        yield from elements

    else:
        zipped: Iterator[tuple]
        match method:
            case 'strict':
                zipped = zip(*kwargs.values(), strict=True)
            case 'truncated':
                zipped = zip(*kwargs.values(), strict=False)
            case 'padded':
                zipped = itertools.zip_longest(*kwargs.values(), fillvalue=None)
        for el in zipped:
            yield dict(zip(kwargs.keys(), el))


@list_iterator.conditional_output_schema
def _(bound_args: dict[str, exprs.Expr]) -> dict[str, type]:
    elements = bound_args['elements']
    el_col_type = elements.col_type
    if (
        not isinstance(el_col_type, ts.JsonType)
        or el_col_type.type_schema is None
        or not isinstance(el_col_type.type_schema.type_spec, list)
        or len(el_col_type.type_schema.type_spec) != 0
    ):
        raise TypeError(f'Expected a type for `elements` matching `list[dict]`; got {el_col_type}')
    dict_type = el_col_type.type_schema.variadic_type
    if (
        not isinstance(dict_type, ts.JsonType)
        or dict_type.type_schema is None
        or not isinstance(dict_type.type_schema.type_spec, dict)
    ):
        raise TypeError(f'Expected a type for `elements` matching `list[dict]`; got {el_col_type}')

    return dict_type.type_schema.type_spec  # type: ignore[return-value]


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
