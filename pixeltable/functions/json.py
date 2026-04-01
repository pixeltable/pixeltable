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
from pixeltable import exceptions as excs, exprs, type_system as ts
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
    """
    Iterator over elements of a list or lists. There are two distinct call patterns: either a single positional
    argument; or one or more keyword arguments.

    - If a single positional argument is specified, as in `list_iterator(t.col)`, then the elements of `t.col` must
        contain lists of dictionaries with matching signatures (identical keys and compatible value types). The
        iterator will yield one new column for each key in the dictionaries, and one output row per element in the
        lists.
    - If multiple keyword arguments are specified, as in `list_iterator(val_1=t.col_1, val_2=t.col_2)`, then the
        elements of each input column must contain lists, but not necessarily lists of dictionaries. The iterator
        will yield one new column for each keyword argument, zipping together the individual lists.

    All of the inputs must be *typed* `Json` expressions. Untyped Json will be rejected (the type schema is
    necessary in order for Pixeltable to determine the types of the output columns).

    Args:
        elements: A list of dictionaries to iterate over. The dictionary keys will be used as column names in the
            output. Cannot be specified together with keyword arguments.
        method: Only applies when called with keyword arguments. Determines how to handle lists of different lengths:

            - `'strict'`: Raises an error if the input lists have different lengths.
            - `'truncated'`: Iterates until the shortest input list is exhausted, ignoring any remaining elements in
                longer lists.
            - `'padded'`: Iterates until the longest input list is exhausted, yielding `None` for any missing
                elements from shorter lists.
        **kwargs: One or more lists to iterate over. The kwarg names will be used as column names in the output.
            Cannot be specified together with `elements`.
    """
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
    if bound_args.get('elements') is not None:
        if len(bound_args) > 1:
            raise excs.Error('list_iterator(): Cannot specify both `elements` and keyword arguments')
        elements = bound_args['elements']
        el_col_type = elements.col_type
        if (
            not isinstance(el_col_type, ts.JsonType)
            or el_col_type.type_schema is None
            or not isinstance(el_col_type.type_schema.type_spec, list)
            or len(el_col_type.type_schema.type_spec) != 0
        ):
            raise excs.Error(
                f'list_iterator(): Expected a type for `elements` matching `list[dict]`; got `{el_col_type}`'
            )
        dict_type = el_col_type.type_schema.variadic_type
        if (
            not isinstance(dict_type, ts.JsonType)
            or dict_type.type_schema is None
            or not isinstance(dict_type.type_schema.type_spec, dict)
        ):
            raise excs.Error(
                f'list_iterator(): Expected a type for `elements` matching `list[dict]`; got `{el_col_type}`'
            )
        return dict_type.type_schema.type_spec  # type: ignore[return-value]

    else:  # bound_args.get('element') is None
        method = bound_args.get('method', 'strict')
        kwargs = bound_args['kwargs']
        if len(kwargs) == 0:
            raise excs.Error('list_iterator(): No inputs provided')
        type_schema: dict[str, ts.ColumnType] = {}
        for name, expr in kwargs.items():
            if (
                not isinstance(expr.col_type, ts.JsonType)
                or expr.col_type.type_schema is None
                or not isinstance(expr.col_type.type_schema.type_spec, list)
            ):
                raise excs.Error(
                    f'list_iterator(): Expected a type for `{name}` matching `list`; got `{expr.col_type}`'
                )
            common_supertype = ts.ColumnType.common_supertype(
                [*expr.col_type.type_schema.type_spec, expr.col_type.type_schema.variadic_type]
            )
            assert common_supertype is not None  # at worst it is `Json`
            type_schema[name] = common_supertype
        return type_schema  # type: ignore[return-value]


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
