from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypedDict, TypeVar

from pixeltable import exceptions as excs
from pixeltable.catalog.globals import fold_identifier
from pixeltable.type_system import ColumnType

_V = TypeVar('_V')


class CellError(TypedDict):
    """Error info for a cell."""

    errortype: str
    errormsg: str


class _FoldedKeyMapping(Mapping[str, _V]):
    """Read-only view over a mapping whose keys are already folded; folds the lookup key.

    Iteration, keys() and len() pass through, so callers still see the stored (folded) spellings.
    """

    _data: Mapping[str, _V]

    def __init__(self, data: Mapping[str, _V]):
        self._data = data

    def __getitem__(self, key: str) -> _V:
        return self._data[fold_identifier(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and fold_identifier(key) in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class Row(Mapping[str, Any]):
    """A dict-like wrapper over a single result row.

    Supports key access (`row['col']`), membership (`'col' in row`), iteration over keys, and the standard `get()`,
    `keys()`, `values()`, and `items()` methods.

    The `errors` property holds error info (`{'errortype': ..., 'errormsg': ...}`) for each cell whose
    evaluation failed; the `index_values` property holds the values of embedding indexes defined on the
    row's table. Both are keyed by column or index name.

    Every key a Row is constructed with is a stored name, and so is already case-folded because they come from
    the catalog. The caller-supplied keys, on the other hand, are folded, so that `row['MyCol']`, `'MYCOL' in row` and
    `row.errors['MyCol']` all work correctly. Iteration and `keys()` pass through and yield the stored spellings.
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

    def _slot(self, key: object) -> int | None:
        """Index of the column `key` names, or None if this row has no such column."""
        return self._columns.get(fold_identifier(key)) if isinstance(key, str) else None

    def __getitem__(self, key: str) -> Any:
        idx = self._slot(key)
        if idx is None:
            raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Column {key!r} does not exist in the row.')
        return self._data[idx]

    def get(self, key: str, default: Any = None) -> Any:
        idx = self._slot(key)
        return default if idx is None else self._data[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __contains__(self, key: object) -> bool:
        return self._slot(key) is not None

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        return f'Row({dict(self)})'

    @property
    def errors(self) -> Mapping[str, CellError]:
        """Error information for each cell of this row whose evaluation failed, keyed by column or index name.

        A failed cell holds `None` as its value and records its error here.

        The returned mapping folds the lookup key, so `row.errors['MyCol']` and `row.errors['mycol']` both work.
        """
        return _FoldedKeyMapping(self._errors)

    @property
    def index_values(self) -> Mapping[str, Any]:
        """The embedding values for embedding indexes defined on the row's table, keyed by index name.

        The returned mapping folds the lookup key, so `row.index_values['MyIdx']` and `row.index_values['myidx']`
        both work.
        """
        return _FoldedKeyMapping(self._index_values)

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
        return {name: repr(t) for name, t in self._col_types.items()}

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
            errors=[row._errors for row in self._rows],
            index_values=[{name: fn(val) for name, val in row._index_values.items()} for row in self._rows],
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
