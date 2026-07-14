import builtins
import typing
from typing import Any, Callable

import sqlalchemy as sql

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.utils.code import local_public_names


# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: ts.ColumnType | type) -> exprs.Expr:
    expr.col_type = ts.ColumnType.normalize_type(target_type)
    return expr


T = typing.TypeVar('T')


@func.uda(allows_window=True, type_substitutions=({T: int | None}, {T: float | None}))  # type: ignore[misc]
class sum(func.Aggregator, typing.Generic[T]):
    """
    Aggregate function that computes the sum of non-null values of a numeric column or grouping.

    Args:
        val: The numeric value to add to the sum.

    Returns:
        The sum of the non-null values, or `None` if there are no non-null values.

    Examples:
        Sum the values in the `value` column of the table `tbl`:

        >>> tbl.select(pxt.functions.sum(tbl.value)).collect()

        Group by the `category` column and compute the sum of the `value` column for each category,
        assigning the name `'category_total'` to the new column:

        >>> tbl.group_by(tbl.category).select(
        ...     tbl.category, category_total=pxt.functions.sum(tbl.value)
        ... ).collect()
    """

    def __init__(self) -> None:
        self.sum: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val  # type: ignore[operator]

    def value(self) -> T:
        return self.sum


@sum.to_sql
def _(val: sql.ColumnElement) -> sql.ColumnElement | None:
    # This can produce a Decimal. We are deliberately avoiding an explicit cast to a Bigint here, because that can
    # cause overflows in Postgres. We're instead doing the conversion to the target type in SqlNode.__iter__().
    return sql.sql.func.sum(val)


@func.uda(
    allows_window=True,
    # Allow counting non-null values of any type
    # TODO: should we have an "Any" type that can be used here?
    type_substitutions=tuple({T: t | None} for t in ts.ALL_PIXELTABLE_TYPES),  # type: ignore[misc]
)
class count(func.Aggregator, typing.Generic[T]):
    """
    Aggregate function that counts the number of non-null values in a column or grouping.

    Args:
        val: The value to count.

    Returns:
        The count of non-null values.

    Examples:
        Count the number of non-null values in the `value` column of the table `tbl`:

        >>> tbl.select(pxt.functions.count(tbl.value)).collect()

        Group by the `category` column and compute the count of non-null values in the `value` column
        for each category, assigning the name `'category_count'` to the new column:

        >>> tbl.group_by(tbl.category).select(
        ...     tbl.category, category_count=pxt.functions.count(tbl.value)
        ... ).collect()
    """

    def __init__(self) -> None:
        self.count = 0

    def update(self, val: T) -> None:
        if val is not None:
            self.count += 1

    def value(self) -> int:
        return self.count


@count.to_sql
def _(val: sql.ColumnElement) -> sql.ColumnElement | None:
    return sql.sql.func.count(val)


@func.uda(
    allows_window=True,
    type_substitutions=tuple({T: t | None} for t in (str, int, float, bool, ts.Timestamp)),  # type: ignore[misc]
)
class min(func.Aggregator, typing.Generic[T]):
    """
    Aggregate function that computes the minimum value in a column or grouping.

    Args:
        val: The value to compare.

    Returns:
        The minimum value, or `None` if there are no non-null values.

    Examples:
        Compute the minimum value in the `value` column of the table `tbl`:

        >>> tbl.select(pxt.functions.min(tbl.value)).collect()

        Group by the `category` column and compute the minimum value in the `value` column for each category,
        assigning the name `'category_min'` to the new column:

        >>> tbl.group_by(tbl.category).select(
        ...     tbl.category, category_min=pxt.functions.min(tbl.value)
        ... ).collect()
    """

    def __init__(self) -> None:
        self.val: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.min(self.val, val)  # type: ignore[call-overload]

    def value(self) -> T:
        return self.val


@min.to_sql
def _(val: sql.ColumnElement) -> sql.ColumnElement | None:
    if val.type.python_type is bool:
        # TODO: min/max aggregation of booleans is not supported in Postgres (but it is in Python).
        # Right now we simply force the computation to be done in Python; we might consider implementing an alternate
        # way of doing it in SQL. (min/max of booleans is simply logical and/or, respectively.)
        return None
    return sql.sql.func.min(val)


@func.uda(
    allows_window=True,
    type_substitutions=tuple({T: t | None} for t in (str, int, float, bool, ts.Timestamp)),  # type: ignore[misc]
)
class max(func.Aggregator, typing.Generic[T]):
    """
    Aggregate function that computes the maximum value in a column or grouping.

    Args:
        val: The value to compare.

    Returns:
        The maximum value, or `None` if there are no non-null values.

    Examples:
        Compute the maximum value in the `value` column of the table `tbl`:

        >>> tbl.select(pxt.functions.max(tbl.value)).collect()

        Group by the `category` column and compute the maximum value in the `value` column for each category,
        assigning the name `'category_max'` to the new column:

        >>> tbl.group_by(tbl.category).select(
        ...     tbl.category, category_max=pxt.functions.max(tbl.value)
        ... ).collect()
    """

    def __init__(self) -> None:
        self.val: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.max(self.val, val)  # type: ignore[call-overload]

    def value(self) -> T:
        return self.val


@max.to_sql
def _(val: sql.ColumnElement) -> sql.ColumnElement | None:
    if val.type.python_type is bool:
        # TODO: see comment in @min.to_sql.
        return None
    return sql.sql.func.max(val)


@func.uda(type_substitutions=({T: int | None}, {T: float | None}))  # type: ignore[misc]
class mean(func.Aggregator, typing.Generic[T]):
    """
    Aggregate function that computes the mean (average) of non-null values of a numeric column or grouping.

    Args:
        val: The numeric value to include in the mean.

    Returns:
        The mean of the non-null values, or `None` if there are no non-null values.

    Examples:
        Compute the mean of the values in the `value` column of the table `tbl`:

        >>> tbl.select(pxt.functions.mean(tbl.value)).collect()

        Group by the `category` column and compute the mean of the `value` column for each category,
        assigning the name `'category_mean'` to the new column:

        >>> tbl.group_by(tbl.category).select(
        ...     tbl.category, category_mean=pxt.functions.mean(tbl.value)
        ... ).collect()
    """

    def __init__(self) -> None:
        self.sum: T = None
        self.count = 0

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val  # type: ignore[operator]
        self.count += 1

    def value(self) -> float | None:  # Always a float
        if self.count == 0:
            return None
        return self.sum / self.count  # type: ignore[operator]


@mean.to_sql
def _(val: sql.ColumnElement) -> sql.ColumnElement | None:
    return sql.sql.func.avg(val)


def _relative_path_root(expr: exprs.Expr) -> exprs.Expr:
    """A relative-path root typed as the element type of the list that expr produces (untyped if unknown)."""
    element_type = expr.col_type.array_element_type() if isinstance(expr.col_type, ts.JsonType) else None
    return exprs.json_path.JsonPath(None, root_type=element_type)


def map(expr: exprs.Expr, fn: Callable[[exprs.Expr], Any]) -> exprs.Expr:
    """
    Applies a function to each element of a JSON array, producing a new array.

    `map()` is used like a UDF, for example in `select()` or `add_computed_column()`.

    Args:
        expr: The array to map over; an expression of type `pxt.Json` that resolves to a JSON array. If its
            elements have a known type (e.g. the column is declared `pxt.Json[[int]]`), that type is available to
            `fn` and carries through to the result.
        fn: A Python function (typically a lambda) applied to each element to produce its replacement. It receives
            `x`, a stand-in for a single array element, and returns the value to store in its place. Operate on
            `x` exactly as you would on a column: arithmetic, indexing (`x[0]`), field access (`x.field`), and
            JSON methods (`x.len()`, `x.sum()`, etc.) all work.

    Returns:
        A new array holding `fn(x)` for each element `x` of `expr`. If `expr` is `null` or does not resolve to a
        JSON array, the result is `null`.

    Examples:
        Given a table `tbl` with a `pxt.Json` column `data` holding lists of numbers, add a column that doubles
        each number:

        >>> tbl.add_computed_column(
        ...     doubled=pxt.functions.map(tbl.data, lambda x: x * 2)
        ... )

        When `data` holds lists of objects such as `{'score': 0.9, 'label': 'cat'}`, extract each score:

        >>> tbl.select(
        ...     scores=pxt.functions.map(tbl.data, lambda x: x.score)
        ... ).collect()

    See also [`filter()`][pixeltable.functions.filter], which keeps a subset of the elements.
    """
    target_expr: exprs.Expr
    try:
        target_expr = exprs.Expr.from_object(fn(_relative_path_root(expr)))
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            'Failed to evaluate map function. '
            '(The `fn` argument to `map()` must produce a valid Pixeltable expression.)',
        ) from e
    return exprs.JsonMapper(expr, target_expr)


def filter(expr: exprs.Expr, predicate: Callable[[exprs.Expr], Any]) -> exprs.Expr:
    """
    Keeps the elements of a JSON array for which a predicate holds, producing a new array.

    `filter()` is used like a UDF, for example in `select()` or `add_computed_column()`.

    Args:
        expr: The array to filter; an expression of type `pxt.Json` that resolves to a JSON array. Its element
            type is preserved in the result.
        predicate: A Python function (typically a lambda) that decides which elements to keep. It receives `x`, a
            stand-in for a single array element, and returns a boolean condition; the element is kept when it is
            true. Operate on `x` exactly as you would on a column: comparisons, indexing (`x[0]`), field access
            (`x.field`), and JSON methods all work. Combine multiple conditions with `&` and `|`, and negate with
            `~` (not Python `and`/`or`/`not`). Example: `lambda x: (x > 0) & (x < 10)`.

    Returns:
        A new array containing the elements of `expr` for which `predicate` is true, unchanged. If `expr` is
        `null` or does not resolve to a JSON array, the result is `null`.

    Examples:
        Given a table `tbl` with a `pxt.Json` column `data` holding lists of numbers, add a column that keeps only
        the positive numbers:

        >>> tbl.add_computed_column(
        ...     positives=pxt.functions.filter(tbl.data, lambda x: x > 0)
        ... )

        When `data` holds lists of objects such as `{'score': 0.9, 'label': 'cat'}`, keep only the high-confidence
        ones:

        >>> tbl.select(
        ...     confident=pxt.functions.filter(tbl.data, lambda x: x.score >= 0.9)
        ... ).collect()

    See also [`map()`][pixeltable.functions.map], which transforms elements.
    """
    filter_expr: exprs.Expr
    try:
        filter_expr = exprs.Expr.from_object(predicate(_relative_path_root(expr)))
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            'Failed to evaluate filter predicate. '
            '(The `predicate` argument to `filter()` must produce a valid Pixeltable expression.)',
        ) from e
    return exprs.JsonMapper(expr, None, filter_expr=filter_expr)


def sort(expr: exprs.Expr, *, key: Callable[[exprs.Expr], Any] | None = None, asc: bool = True) -> exprs.Expr:
    """
    Sorts the elements of a JSON array, producing a new array.

    `sort()` is used like a UDF, for example in `select()` or `add_computed_column()`.

    Args:
        expr: The array to sort; an expression of type `pxt.Json` that resolves to a JSON array. Its element type
            is preserved in the result.
        key: An optional Python function (typically a lambda) that produces the value each element is ordered by.
            It receives `x`, a stand-in for a single array element, and returns the sort key. Operate on `x`
            exactly as you would on a column: arithmetic, indexing (`x[0]`), field access (`x.field`), and JSON
            methods all work. When `key` is omitted, the elements are ordered by their own natural ordering.
        asc: Whether to sort in ascending (the default) or descending order.

    Returns:
        A new array with the elements of `expr` in sorted order. If `expr` is `null` or does not resolve to a JSON
        array, the result is `null`. Sorting a list of scalars without a `key`, or by non-orderable keys, raises if
        the values are not mutually comparable (matching Python's `sorted()`).

    Examples:
        Given a table `tbl` with a `pxt.Json` column `data` holding lists of numbers, add a column that sorts each
        list in descending order:

        >>> tbl.add_computed_column(ranked=pxt.functions.sort(tbl.data, asc=False))

        When `data` holds lists of objects such as `{'score': 0.9, 'label': 'cat'}`, sort each list by score:

        >>> tbl.select(
        ...     by_score=pxt.functions.sort(
        ...         tbl.data, key=lambda x: x.score, asc=False
        ...     )
        ... ).collect()

    See also [`map()`][pixeltable.functions.map] and [`filter()`][pixeltable.functions.filter].
    """
    if key is None:
        # no per-element expression to evaluate: sort the materialized list directly, without building the nested
        # rows a JsonMapper would. Imported here rather than at module scope to avoid a circular import with json.
        from pixeltable.functions import json

        result = json._sort(expr) if asc else json._sort(expr, asc=False)
        # render as a method call (x.sort(...)) rather than as a plain call to the private _sort UDF
        result.is_method_call = True
        return result
    key_expr: exprs.Expr
    try:
        key_expr = exprs.Expr.from_object(key(_relative_path_root(expr)))
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            'Failed to evaluate sort key. (The `key` argument to `sort()` must produce a valid Pixeltable expression.)',
        ) from e
    return exprs.JsonMapper(expr, None, key_expr=key_expr, asc=asc)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
