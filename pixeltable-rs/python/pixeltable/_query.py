"""`pxt.Query` and `pxt.ResultSet`: thin wrappers around the backend's `QuerySpec` and `ResultSet`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, TypeVar

import pandas as pd
import pydantic

from pixeltable import _core
from pixeltable._unimplemented import unimplemented

if TYPE_CHECKING:
    from pixeltable._table import Table
    from pixeltable._update_status import UpdateStatus
    from pixeltable.type_system import ColumnType

__all__ = ['Query', 'ResultCursor', 'ResultSet']


class ResultSet:
    """
    A dataset obtained by executing a [`Query`][pixeltable.Query].

    - `len(result)` returns the number of rows
    - `result[i]` returns the `i`th row as a `dict` mapping column names to values
    - `result['col']` returns a `list` of all values in the column named `'col'`
    - `result[i, 'col']` returns the specific value in the `i`th row and column `'col'`
    """

    _core: _core.ResultSet

    def __init__(self, core: _core.ResultSet):
        self._core = core

    @property
    def schema(self) -> dict[str, str]:
        """The result columns as a mapping from name to its type string."""
        return {name: repr(col_type) for name, col_type in self._schema.items()}

    @property
    def _schema(self) -> dict[str, ColumnType]:
        return self._core.schema()

    @property
    def _col_names(self) -> list[str]:
        return self._core.column_names()

    def __len__(self) -> int:
        return len(self._core)

    def __getitem__(self, index: Any) -> Any:
        return self._core.getitem(index)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._core.rows())

    def to_pandas(self) -> pd.DataFrame:
        """Convert the `ResultSet` to a Pandas `DataFrame`."""
        return self._core.to_pandas()

    def __repr__(self) -> str:
        return repr(self.to_pandas())

    def __str__(self) -> str:
        return self.to_pandas().to_string()

    def _repr_html_(self) -> str:
        return self.to_pandas().to_html(index=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultSet):
            return False
        return self.to_pandas().equals(other.to_pandas())

    def __hash__(self) -> int:
        return hash(tuple(self._col_names))

    BaseModelT = TypeVar('BaseModelT', bound=pydantic.BaseModel)

    @unimplemented('ResultSet.to_pydantic()')
    def to_pydantic(self, model: type[BaseModelT]) -> Iterator[BaseModelT]: ...


class ResultCursor:
    """Cursor that iterates over query results, yielding [`Row`][pixeltable.Row] objects.

    A cursor is pending (created), open (iterating) or closed. Iteration opens it and exhaustion closes it; it can
    also be used as a context manager.
    """

    _core: _core.ResultCursor
    _schema: dict[str, ColumnType]

    def __init__(self, query: Query):
        self._core = query._spec.cursor()
        self._schema = query.schema

    @property
    def schema(self) -> dict[str, str]:
        """The result columns as a mapping from name to its type string."""
        return {name: repr(col_type) for name, col_type in self._schema.items()}

    @property
    def _closed(self) -> bool:
        return self._core._closed

    def open(self) -> None:
        """Start the query; raises if the cursor is already open or closed."""
        self._core.open()

    def close(self) -> None:
        """Release the query's resources (idempotent); a closed cursor cannot be reopened."""
        self._core.close()

    def __enter__(self) -> ResultCursor:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._core)

    def __repr__(self) -> str:
        cols = ', '.join(f'{name}: {col_type}' for name, col_type in self._schema.items())
        return f'ResultCursor({self._core.state_name}, columns=[{cols}])'


class Query:
    """A query over a table: it can be executed, and its rows updated or deleted."""

    _spec: _core.QuerySpec

    def __init__(self, spec: _core.QuerySpec):
        self._spec = spec

    @classmethod
    def _from_table(cls, tbl: Table) -> Query:
        return cls(_core.QuerySpec(str(tbl._id), tbl._version))

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Column names and types in this Query."""
        return self._spec.schema()

    def has_select_list(self) -> bool:
        return self._spec.has_select_list

    def select(self, *items: Any, **named_items: Any) -> Query:
        """Select columns or expressions."""
        return Query(self._spec.select([(e, None) for e in items] + [(e, k) for k, e in named_items.items()]))

    def where(self, pred: _core.Expr) -> Query:
        """Filter rows based on a predicate."""
        return Query(self._spec.where_(pred))

    def order_by(self, *expr_list: _core.Expr, asc: bool = True) -> Query:
        """Order the rows by the given expressions (ascending nulls last, descending nulls first)."""
        return Query(self._spec.order_by(list(expr_list), asc))

    def limit(self, n: int, offset: int | None = None) -> Query:
        """Limit the number of rows, optionally skipping `offset` rows first."""
        return Query(self._spec.limit(n, offset))

    def collect(self) -> ResultSet:
        return ResultSet(self._spec.collect())

    def show(self, n: int = 20) -> ResultSet:
        return self.limit(n).collect()

    def head(self, n: int = 10) -> ResultSet:
        """The first n rows, in insertion order of the underlying table."""
        return ResultSet(self._spec.head(n))

    def tail(self, n: int = 10) -> ResultSet:
        """The last n rows, in insertion order of the underlying table."""
        return ResultSet(self._spec.tail(n))

    def count(self) -> int:
        """The number of rows in the Query."""
        return self._spec.count()

    def as_dict(self) -> dict[str, Any]:
        return self._spec.as_dict()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        return cls(_core.QuerySpec.from_dict(d))

    def _descriptors(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{'Name': n, 'Type': t, 'Expression': e} for n, t, e in self._spec.describe_rows()],
            columns=['Name', 'Type', 'Expression'],
        )

    def __repr__(self) -> str:
        return self._descriptors().to_string(index=False)

    def _repr_html_(self) -> str:
        return self._descriptors().to_html(index=False)

    def describe(self) -> None:
        print(repr(self))

    @unimplemented('Query.join()')
    def join(self, other: Table, on: Any = None, how: str = 'inner') -> Query: ...

    @unimplemented('Query.group_by()')
    def group_by(self, *grouping_items: Any) -> Query: ...

    @unimplemented('Query.distinct()')
    def distinct(self) -> Query:
        """
        Remove duplicate rows from this Query.

        Note that grouping will be applied to the rows based on the select clause of this Query.
        In the absence of a select clause, by default, all columns are selected in the grouping.

        Each selected expression must be expressible in SQL. A JSON path, including one passed through
        `astype()`, is not. Executing the query raises `RequestError` rather than returning duplicate rows.
        Store the value in a computed column of a scalar type and select that column.

        Examples:
            Select unique addresses from table `addresses`.

            >>> results = addresses.distinct()

            Select unique cities in table `addresses`

            >>> results = addresses.select(addresses.city).distinct()
        """

    @unimplemented('Query.sample()')
    def sample(self, *args: Any, **kwargs: Any) -> Query: ...

    def cursor(self) -> ResultCursor:
        """A [`ResultCursor`][pixeltable.ResultCursor] over the query results."""
        return ResultCursor(self)

    @unimplemented('Query.update()')
    def update(self, value_spec: dict[str, Any], cascade: bool = True) -> UpdateStatus: ...

    @unimplemented('Query.delete()')
    def delete(self) -> UpdateStatus: ...

    @unimplemented('Query.recompute_columns()')
    def recompute_columns(self, *columns: Any, errors_only: bool = False, cascade: bool = True) -> UpdateStatus: ...

    @unimplemented('Query.to_pytorch_dataset()')
    def to_pytorch_dataset(self, image_format: str = 'pt') -> Any: ...

    @unimplemented('Query.to_coco_dataset()')
    def to_coco_dataset(self) -> Path: ...

    def __hash__(self) -> int:
        return hash(repr(self._spec.as_dict()))
