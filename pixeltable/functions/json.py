"""
Pixeltable UDFs for `JsonType`.

Example:

>>> import pixeltable as pxt
>>> import pixeltable.functions as pxtf
>>> t = pxt.get_table(...)
>>> t.select(pxtf.json.make_list(t.json_col)).collect()
"""

import builtins
import itertools
import json
from typing import Any, Iterator, Literal

import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import array as pg_array

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


def _jsonb_object_length(obj: sql.ColumnElement) -> sql.ColumnElement:
    """SQL expression for the number of keys in a jsonb object."""
    return sql.select(sql.func.count()).select_from(sql.func.jsonb_object_keys(obj)).scalar_subquery()


def _jsonb_as_text(obj: sql.ColumnElement) -> sql.ColumnElement:
    """SQL expression for a jsonb string's underlying text (the empty path '{}' selects the whole value)."""
    return obj.op('#>>')(pg_array([], type_=sql.Text))


@pxt.udf(is_method=True)
def len(self: pxt.Json) -> int:
    """
    Return the number of elements in a JSON array, keys in a JSON object, or characters in a JSON string.

    Not defined for numbers or booleans. A `null` value (or missing path) yields `null`.

    Example:

        >>> t.select(t.detections.bboxes.len()).collect()
    """
    if isinstance(self, (list, dict, str)):
        return builtins.len(self)
    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'len() is not defined for a JSON {type(self).__name__}')


@len.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    type_of = sql.func.jsonb_typeof(self)
    return sql.case(
        (self.is_(None), sql.null()),
        (type_of == 'null', sql.null()),
        (type_of == 'array', sql.func.jsonb_array_length(self)),
        (type_of == 'object', _jsonb_object_length(self)),
        (type_of == 'string', sql.func.length(_jsonb_as_text(self))),
        # a number or boolean is not sized: reuse jsonb_array_length's native scalar error
        else_=sql.func.jsonb_array_length(self),
    )


@pxt.udf(is_method=True)
def is_empty(self: pxt.Json | None) -> bool:
    """
    Return `True` if the value is `null`, an empty array, an empty object, or an empty string; `False` otherwise
    (including for numbers and booleans).

    Example:

        >>> t.where(~t.detections.bboxes.is_empty()).collect()
    """
    if self is None:
        return True
    if isinstance(self, (list, dict, str)):
        return builtins.len(self) == 0
    return False


@is_empty.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    type_of = sql.func.jsonb_typeof(self)
    return sql.case(
        (self.is_(None), sql.literal(True)),
        (type_of == 'null', sql.literal(True)),
        (type_of == 'array', sql.func.jsonb_array_length(self) == 0),
        (type_of == 'object', _jsonb_object_length(self) == 0),
        (type_of == 'string', _jsonb_as_text(self) == ''),
        else_=sql.literal(False),
    )


@pxt.udf(is_method=True)
def contains(self: pxt.Json, value: pxt.Json) -> bool:
    """
    Return `True` if `value` is an element of a JSON array or a key of a JSON object; `False` otherwise.

    Example:

        >>> t.where(t.detections.labels.contains('person')).collect()
    """
    if isinstance(self, list):
        return value in self
    if isinstance(self, dict):
        return value in self
    return False


# TODO: add a to_sql for contains(). A jsonb value parameter binds as an untyped literal that Postgres can't
# resolve during EXPLAIN, so the natural @> / jsonb_exists translation fails; it needs explicit jsonb typing.


@pxt.udf(is_method=True)
def get(self: pxt.Json | None, key: str, default: pxt.Json | None = None) -> pxt.Json:
    """
    Return the value of `key` if the value is a JSON object containing it, otherwise `default`.

    Example:

        >>> t.select(t.metadata.get('author', default='unknown')).collect()
    """
    if isinstance(self, dict):
        return self.get(key, default)
    return default


@get.conditional_return_type
def _(self: exprs.Expr, key: str) -> ts.ColumnType:
    schema = self.col_type.type_schema if isinstance(self.col_type, ts.JsonType) else None
    if schema is not None and isinstance(schema.type_spec, dict) and key in schema.type_spec:
        return schema.type_spec[key].copy(nullable=True)
    return ts.JsonType(nullable=True)


# TODO: add a to_sql for get(). The jsonb default parameter binds as an untyped literal that Postgres can't
# resolve during EXPLAIN (see contains()).


def _require_number_array(arr: Any, fn_name: str) -> list:
    if not isinstance(arr, list) or builtins.any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in arr):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'{fn_name}() is only defined for a JSON array of numbers'
        )
    return arr


@pxt.udf(is_method=True)
def sum(self: pxt.Json) -> float:
    """
    Return the sum of the numbers in a JSON array (0 for an empty array).

    Example:

        >>> t.select(t.detections.scores.sum()).collect()
    """
    return float(builtins.sum(_require_number_array(self, 'sum')))


def _number_array_agg(self: sql.ColumnElement, agg_fn: Any, empty_value: int | None = None) -> sql.ColumnElement:
    """
    SQL for a numeric aggregate over a jsonb array; a non-array reuses jsonb_array_length's native error.

    For agg_fn=min, this produces:

        CASE WHEN jsonb_typeof(self) = 'array'
             THEN (SELECT min(e::numeric) FROM jsonb_array_elements(self) AS e)
             ELSE jsonb_array_length(self)
        END

    When empty_value is given (eg, 0 for sum), the subquery is wrapped in coalesce(..., empty_value) so that
    an empty array yields empty_value rather than null.
    """
    elem = sql.func.jsonb_array_elements(self).column_valued('e')
    agg: sql.ColumnElement = sql.select(agg_fn(elem.cast(sql.Numeric))).scalar_subquery()
    if empty_value is not None:
        agg = sql.func.coalesce(agg, empty_value)
    return sql.case((sql.func.jsonb_typeof(self) == 'array', agg), else_=sql.func.jsonb_array_length(self))


@sum.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return _number_array_agg(self, sql.func.sum, empty_value=0)


@pxt.udf(is_method=True)
def min(self: pxt.Json) -> float:
    """
    Return the smallest number in a JSON array, or `null` if the array is empty.

    Example:

        >>> t.select(t.detections.scores.min()).collect()
    """
    nums = _require_number_array(self, 'min')
    return float(builtins.min(nums)) if builtins.len(nums) > 0 else None


@min.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return _number_array_agg(self, sql.func.min)


@pxt.udf(is_method=True)
def max(self: pxt.Json) -> float:
    """
    Return the largest number in a JSON array, or `null` if the array is empty.

    Example:

        >>> t.select(t.detections.scores.max()).collect()
    """
    nums = _require_number_array(self, 'max')
    return float(builtins.max(nums)) if builtins.len(nums) > 0 else None


@max.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return _number_array_agg(self, sql.func.max)


@pxt.udf(is_method=True)
def mean(self: pxt.Json) -> float:
    """
    Return the arithmetic mean of the numbers in a JSON array, or `null` if the array is empty.

    Example:

        >>> t.select(t.detections.scores.mean()).collect()
    """
    nums = _require_number_array(self, 'mean')
    return builtins.sum(nums) / builtins.len(nums) if builtins.len(nums) > 0 else None


@mean.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return _number_array_agg(self, sql.func.avg)


@pxt.udf(is_method=True)
def count(self: pxt.Json, value: pxt.Json) -> int:
    """
    Return the number of times `value` occurs in a JSON array.

    Example:

        >>> t.select(t.detections.labels.count('person')).collect()
    """
    if isinstance(self, list):
        return self.count(value)
    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'count() is only defined for a JSON array')


@pxt.udf(is_method=True)
def keys(self: pxt.Json) -> list[str]:
    """
    Return the keys of a JSON object. `keys()`, `values()`, and `items()` share the same ordering.

    Example:

        >>> t.select(t.metadata.keys()).collect()
    """
    if isinstance(self, dict):
        return builtins.list(self.keys())
    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'keys() is only defined for a JSON object')


@pxt.udf(is_method=True)
def values(self: pxt.Json) -> pxt.Json:
    """
    Return the values of a JSON object. `keys()`, `values()`, and `items()` share the same ordering.

    Example:

        >>> t.select(t.metadata.values()).collect()
    """
    if isinstance(self, dict):
        return builtins.list(self.values())
    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'values() is only defined for a JSON object')


@values.conditional_return_type
def _(self: exprs.Expr) -> ts.ColumnType:
    # jsonb does not preserve key order, so the values can't be typed as an ordered tuple; use a variadic list
    # of the common supertype of the value types instead
    schema = self.col_type.type_schema if isinstance(self.col_type, ts.JsonType) else None
    if schema is not None and isinstance(schema.type_spec, dict) and builtins.len(schema.type_spec) > 0:
        supertype = ts.ColumnType.common_supertype(schema.type_spec.values())
        if supertype is not None:
            return ts.JsonType(ts.JsonType.TypeSchema([], variadic_type=supertype), nullable=True)
    return ts.JsonType(nullable=True)


@pxt.udf(is_method=True)
def items(self: pxt.Json) -> pxt.Json:
    """
    Return the `[key, value]` pairs of a JSON object as a list. `keys()`, `values()`, and `items()` share the
    same ordering.

    Example:

        >>> t.select(t.metadata.items()).collect()
    """
    if isinstance(self, dict):
        return [[k, v] for k, v in self.items()]
    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'items() is only defined for a JSON object')


@pxt.udf(is_method=True)
def flatten(self: pxt.Json) -> pxt.Json:
    """
    Concatenate the elements of a JSON array one level deep; non-array elements are kept as-is.

    Example:

        >>> t.select(t.chunks.flatten()).collect()
    """
    if not isinstance(self, list):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'flatten() is only defined for a JSON array')
    result: list = []
    for x in self:
        if isinstance(x, list):
            result.extend(x)
        else:
            result.append(x)
    return result


@pxt.udf(display_name='sort')
def _sort(list_: pxt.Json | None, asc: bool = True) -> pxt.Json:
    """Return a new JSON array with the elements sorted by their natural ordering."""
    if not isinstance(list_, list):
        return None
    try:
        return sorted(list_, reverse=not asc)
    except TypeError as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            'sort(): the array elements are not orderable; pass a key, e.g. .sort(lambda R: R.field)',
        ) from e


@_sort.conditional_return_type
def _(list_: exprs.Expr) -> ts.ColumnType:
    # sorting reproduces the source elements in a new order, so the element type carries through unchanged
    if isinstance(list_.col_type, ts.JsonType):
        return list_.col_type.copy(nullable=True)
    return ts.JsonType(nullable=True)


def _variadic_element_type(col_type: ts.ColumnType) -> ts.ColumnType | None:
    """The element type of a variadic-list JsonType (Json[[T]]), or None if `ct` isn't one."""
    if isinstance(col_type, ts.JsonType) and col_type.type_schema is not None:
        schema = col_type.type_schema
        if (
            isinstance(schema.type_spec, list)
            and builtins.len(schema.type_spec) == 0
            and schema.variadic_type is not None
        ):
            return schema.variadic_type
    return None


@flatten.conditional_return_type
def _(self: exprs.Expr) -> ts.ColumnType:
    # Json[[[T]]] (a variadic list of variadic lists) flattens one level to Json[[T]]
    outer_element = _variadic_element_type(self.col_type)
    inner_element = _variadic_element_type(outer_element) if outer_element is not None else None
    if inner_element is not None:
        return ts.JsonType(ts.JsonType.TypeSchema([], variadic_type=inner_element), nullable=True)
    return ts.JsonType(nullable=True)


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
    elements: list[dict] | None = None, *, mode: Literal['strict', 'truncated', 'padded'] = 'strict', **kwargs: list
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
        mode: Only applies when called with keyword arguments. Determines how to handle lists of different lengths:

            - `'strict'`: Raises an error if the input lists have different lengths.
            - `'truncated'`: Iterates until the shortest input list is exhausted, ignoring any remaining elements in
                longer lists.
            - `'padded'`: Iterates until the longest input list is exhausted, yielding `None` for any missing
                elements from shorter lists.
        **kwargs: One or more lists to iterate over. The kwarg names will be used as column names in the output.
            Cannot be specified together with `elements`.
    """
    assert (elements is None) != (builtins.len(kwargs) == 0)

    if elements is not None:
        yield from elements

    else:
        # TODO: Clean up the way kwargs are passed to the iterator (this works, but it's a bit clunk with
        #     unnecessary indirection)
        kwargs_: dict[str, list] = kwargs['kwargs']  # type: ignore[assignment]
        zipped: Iterator[tuple]
        match mode:
            case 'strict':
                zipped = zip(*kwargs_.values(), strict=True)
            case 'truncated':
                zipped = zip(*kwargs_.values(), strict=False)
            case 'padded':
                zipped = itertools.zip_longest(*kwargs_.values(), fillvalue=None)
        for el in zipped:
            yield dict(zip(kwargs_.keys(), el, strict=True))


@list_iterator.conditional_output_schema
def _(bound_args: dict[str, exprs.Expr]) -> dict[str, type]:
    if bound_args.get('elements') is not None:
        if 'mode' in bound_args:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'list_iterator(): `mode` argument cannot be used with `elements`'
            )
        if builtins.len(bound_args) > 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'list_iterator(): Cannot specify both `elements` and keyword arguments',
            )
        elements = bound_args['elements']

        el_col_type = elements.col_type
        if (
            not isinstance(el_col_type, ts.JsonType)
            or el_col_type.type_schema is None
            or not isinstance(el_col_type.type_schema.type_spec, list)
            or builtins.len(el_col_type.type_schema.type_spec) != 0
        ):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'list_iterator(): Expected a type for `elements` matching `list[dict]`; got `{el_col_type}`',
            )
        dict_type = el_col_type.type_schema.variadic_type
        if (
            not isinstance(dict_type, ts.JsonType)
            or dict_type.type_schema is None
            or not isinstance(dict_type.type_schema.type_spec, dict)
        ):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'list_iterator(): Expected a type for `elements` matching `list[dict]`; got `{el_col_type}`',
            )

        return dict_type.type_schema.type_spec  # type: ignore[return-value]

    else:  # bound_args.get('element') is None
        mode = bound_args.get('mode')
        kwargs = bound_args.get('kwargs', {})  # type: ignore[var-annotated]
        if builtins.len(kwargs) == 0:
            raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'list_iterator(): No inputs provided')

        output_schema: dict[str, ts.ColumnType] = {}
        for name, expr in kwargs.items():
            assert isinstance(expr, exprs.Expr)
            if (
                not isinstance(expr.col_type, ts.JsonType)
                or expr.col_type.type_schema is None
                or not isinstance(expr.col_type.type_schema.type_spec, list)
            ):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'list_iterator(): Expected a type for `{name}` matching `list`; got `{expr.col_type}`',
                )
            type_schema = expr.col_type.type_schema
            relevant_types = type_schema.type_spec
            assert isinstance(relevant_types, list)
            if type_schema.variadic_type is not None:
                relevant_types = [*relevant_types, type_schema.variadic_type]
            common_supertype = ts.ColumnType.common_supertype(relevant_types)
            assert common_supertype is not None  # at worst it is `Json`
            if mode is not None and (not isinstance(mode, exprs.Literal) or mode.val == 'padded'):
                # If `mode` is 'padded' or a non-constant expression, then it's possible we may get a None value in the
                # output, so we need to make the type nullable.
                common_supertype = common_supertype.copy(nullable=True)
            output_schema[name] = common_supertype

        return output_schema  # type: ignore[return-value]


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
