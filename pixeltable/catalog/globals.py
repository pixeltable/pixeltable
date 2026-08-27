from __future__ import annotations

import dataclasses
import enum
import itertools
from typing import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from typing_extensions import TypeForm

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.metadata import schema
from pixeltable.types import ColumnSpec

if TYPE_CHECKING:
    from pixeltable import exprs, index
    from pixeltable.globals import TableDataSource

    from .column import Column
    from .types import ColumnVersionMd

# name of the position column in a component view
_POS_COLUMN_NAME = 'pos'
_ROWID_COLUMN_NAME = '_rowid'

# Set of symbols that are predefined in the `InsertableTable` class (and are therefore not allowed as column names).
# This will be populated lazily to avoid circular imports.
_PREDEF_SYMBOLS: set[str] | None = None


@dataclasses.dataclass
class DirEntry:
    dir: schema.Dir | None
    dir_entries: dict[str, DirEntry]
    table: schema.Table | None

    # Only populated for table entries when get_dir_contents() was called with with_error_counts=True;
    # None otherwise (including for directory entries).
    table_error_count: int | None = None


class IndexSpec(NamedTuple):
    """
    A declared spec names its column; a resolved one identifies it. A column of the table being created is
    identified by the Column instance the builder assigns an id to; one that already exists, in this table or a
    base, by its metadata.

    TODO: make this generic in the column type, so that a declared spec is an IndexSpec[str] and a
    resolved one an IndexSpec[Column].
    """

    indexed_column: 'str | Column | ColumnVersionMd'
    idx_name: str | None  # None for an unnamed index
    idx: index.IndexBase


class MediaValidation(enum.Enum):
    ON_READ = 0
    ON_WRITE = 1

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> MediaValidation:
        try:
            return cls[name.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{error_prefix} must be one of: [{val_strs}]'
            ) from None


class OnErrorParam(enum.Enum):
    """Supported values for the on_error parameter"""

    ABORT = 'abort'
    IGNORE = 'ignore'

    @classmethod
    def is_valid(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in [c.value for c in cls]
        return False

    @classmethod
    def fail_on_exception(cls, v: str) -> bool:
        if not cls.is_valid(v):
            raise ValueError(f'Invalid value for on_error: {v}')
        return v.lower() == cls.ABORT.value


class IfExistsParam(enum.Enum):
    ERROR = 0
    IGNORE = 1
    REPLACE = 2
    REPLACE_FORCE = 3

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{param_name} must be one of: [{val_strs}]'
            ) from None


class IfNotExistsParam(enum.Enum):
    ERROR = 0
    IGNORE = 1

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfNotExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{param_name} must be one of: [{val_strs}]'
            ) from None


def is_valid_identifier(name: str, *, allow_hyphens: bool = False) -> bool:
    # If allow_hyphens=True, we allow hyphens to appear in the name, but we still do not permit a name to start with one
    adj_name = name.replace('-', '_') if allow_hyphens else name
    return adj_name.isidentifier() and not name.startswith('-') and not name.startswith('_')


def is_system_column_name(name: str) -> bool:
    from pixeltable.catalog import InsertableTable, View

    global _PREDEF_SYMBOLS  # noqa: PLW0603
    if _PREDEF_SYMBOLS is None:
        _PREDEF_SYMBOLS = set(itertools.chain(dir(InsertableTable), dir(View)))
    return name in _PREDEF_SYMBOLS


def col_type_from_spec(column_spec: ColumnSpec) -> ts.ColumnType:
    """The ColumnType that a column defined by `column_spec` will have."""
    if 'type' in column_spec:
        return ts.ColumnType.normalize_type(column_spec['type'], allow_builtin_types=False)
    assert 'value' in column_spec
    return column_spec['value'].col_type


def normalize_schema(schema: Mapping[str, TypeForm | ColumnSpec | exprs.Expr]) -> dict[str, ColumnSpec]:
    """Canonicalize a create_table schema to a {name: ColumnSpec} mapping with resolved ColumnTypes."""
    from pixeltable import exprs

    from .column import Column

    result: dict[str, ColumnSpec] = {}
    for name, spec in schema.items():
        if isinstance(spec, exprs.Expr):
            result[name] = {'value': spec}
            continue
        if isinstance(spec, dict):
            col_spec: dict[str, Any] = dict(spec)
            Column._validate_column_spec(name, cast(ColumnSpec, col_spec))
        elif isinstance(spec, ts.ColumnType) or ts.is_type_form(spec):
            col_spec = {'type': spec}
        else:
            raise excs.RequestError(excs.ErrorCode.TYPE_MISMATCH, f'Invalid spec for column {name!r}: {spec!r}')
        if col_spec.get('type') is not None:
            col_spec['type'] = ts.ColumnType.normalize_type(col_spec['type'], allow_builtin_types=False)
        result[name] = cast(ColumnSpec, col_spec)
    return result


def is_hf_dataset(source: 'TableDataSource' | None) -> bool:
    try:
        import datasets  # type: ignore[import-untyped]
    except ImportError:
        return False
    return isinstance(
        source, (datasets.Dataset, datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict)
    )
