from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypedDict

from pixeltable import exceptions as excs
from pixeltable.type_system import ColumnType


class CellError(TypedDict):
    """Error info for a cell."""

    errortype: str
    errormsg: str


class Row(Mapping[str, Any]):
    """A dict-like wrapper over a single result row.

    Supports key access (`row['col']`), membership (`'col' in row`), iteration over keys, and the standard `get()`,
    `keys()`, `values()`, and `items()` methods.

    The `errors` property holds error info (`{'errortype': ..., 'errormsg': ...}`) for each cell whose
    evaluation failed; the `index_values` property holds the values of embedding indexes defined on the
    row's table. Both are keyed by column or index name.
    """

    _data: tuple[Any, ...]
    _columns: dict[str, int]
    _col_types: dict[str, ColumnType]
    _errors: dict[str, CellError]
    _index_values: dict[str, Any]

    def __init__(
        self,
        data: Iterable[Any],
        columns: dict[str, int],
        col_types: dict[str, ColumnType],
        errors: dict[str, CellError] | None = None,
        index_values: dict[str, Any] | None = None,
    ):
        self._data = tuple(data)
        self._columns = columns
        self._col_types = col_types
        self._errors = errors or {}
        self._index_values = index_values or {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._columns:
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Column {key!r} does not exist in the row.')
        return self._data[self._columns[key]]

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._columns:
            return default
        return self._data[self._columns[key]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __contains__(self, key: object) -> bool:
        return key in self._columns

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        return 'Row({' + ', '.join(f'{k!r}: {v!r}' for k, v in self.items()) + '})'

    @property
    def errors(self) -> dict[str, CellError]:
        """Error information for each cell of this row whose evaluation failed, keyed by column or index name.

        A failed cell holds `None` as its value and records its error here.
        """
        return self._errors

    @property
    def index_values(self) -> dict[str, Any]:
        """The embedding values for embedding indexes defined on the row's table, keyed by index name."""
        return self._index_values

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this row's values.

        - `None`: preserved as `None`
        - Timestamp, Date: ISO 8601 string
        - UUID: string
        - Array: Python list (via `tolist()`)
        - Json: validated for serializability, kept as native Python
        - Binary: omitted (not representable in JSON)
        - All others: unchanged
        """
        result: dict[str, Any] = {}
        for col_name, col_type in self._col_types.items():
            val = self[col_name]
            if col_type.is_binary_type():
                continue
            elif val is None:
                result[col_name] = None
            elif col_type.is_timestamp_type() or col_type.is_date_type():
                result[col_name] = val.isoformat()
            elif col_type.is_uuid_type():
                result[col_name] = str(val)
            elif col_type.is_array_type():
                result[col_name] = val.tolist()
            elif col_type.is_json_type():
                try:
                    json.dumps(val)
                except (TypeError, ValueError) as err:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT,
                        f'Column {col_name!r} contains a value that is not JSON-serializable: {err}',
                    ) from err
                result[col_name] = val
            else:
                result[col_name] = val
        return result


class RowBatch(Sequence[Row]):
    """A sequence of [`Row`][pixeltable.Row] instances that share a common schema.

    Supports indexing (`batch[0]`), iteration, `len()`.
    """

    _col_types: dict[str, ColumnType]
    _columns: dict[str, int]
    _rows: list[Row]

    def __init__(
        self,
        data: Iterable[Iterable[Any]],
        col_types: dict[str, ColumnType],
        errors: Sequence[dict[str, CellError]] | None = None,
        index_values: Sequence[dict[str, Any]] | None = None,
    ):
        self._col_types = col_types
        self._columns = {name: i for i, name in enumerate(col_types)}
        self._rows = [
            Row(
                row_data,
                self._columns,
                self._col_types,
                errors=errors[i] if errors is not None else None,
                index_values=index_values[i] if index_values is not None else None,
            )
            for i, row_data in enumerate(data)
        ]

    @property
    def schema(self) -> dict[str, str]:
        """The batch's column names and types, in column order."""
        return {name: t._to_str(as_schema=True) for name, t in self._col_types.items()}

    @property
    def column_names(self) -> list[str]:
        """The batch's column names, in column order."""
        return list(self._columns)

    def to_json(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable list of row dicts (see [`Row.to_json()`][pixeltable.Row.to_json])."""
        return [row.to_json() for row in self._rows]

    def _map_values(self, fn: Callable[[Any], Any]) -> RowBatch:
        """Return a new RowBatch with fn applied to every column and index value."""
        return RowBatch(
            [tuple(fn(val) for val in row._data) for row in self._rows],
            self._col_types,
            errors=[row.errors for row in self._rows],
            index_values=[{name: fn(val) for name, val in row.index_values.items()} for row in self._rows],
        )

    def __getitem__(self, index: Any) -> Any:
        return self._rows[index]

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    # unhashable, like dict and list
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        """Test utility"""
        if isinstance(other, RowBatch):
            return self._rows == other._rows
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return len(self._rows) == len(other) and all(a == b for a, b in zip(self._rows, other))
        return NotImplemented

    def __repr__(self) -> str:
        return 'RowBatch(' + repr([dict(row) for row in self._rows]) + ')'
