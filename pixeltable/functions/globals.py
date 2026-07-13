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
    Applies a mapping function to each element of a list.

    Args:
        expr: The list expression to map over; must be an expression of type `pxt.Json`.
        fn: An operation on Pixeltable expressions that will be applied to each element of the JSON array.

    Examples:
        Given a table `tbl` with a column `data` of type `pxt.Json` containing lists of integers, add a computed
        column that produces new lists with each integer doubled:

        >>> tbl.add_computed_column(
        ...     doubled=pxt.functions.map(t.data, lambda x: x * 2)
        ... )
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
    Selects the elements of a list for which a predicate is true.

    Args:
        expr: The list expression to filter; must be an expression of type `pxt.Json`.
        predicate: An operation on Pixeltable expressions that produces a boolean condition for each element of
            the JSON array. Only elements for which the condition is true are retained.

    Examples:
        Given a table `tbl` with a column `data` of type `pxt.Json` containing lists of integers, add a computed
        column that keeps only the positive integers of each list:

        >>> tbl.add_computed_column(
        ...     positives=pxt.functions.filter(t.data, lambda x: x > 0)
        ... )
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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
