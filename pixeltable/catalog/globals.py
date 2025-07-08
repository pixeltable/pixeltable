from __future__ import annotations

import enum
import itertools
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')

# name of the position column in a component view
_POS_COLUMN_NAME = 'pos'
_ROWID_COLUMN_NAME = '_rowid'

# Set of symbols that are predefined in the `InsertableTable` class (and are therefore not allowed as column names).
# This will be populated lazily to avoid circular imports.
_PREDEF_SYMBOLS: Optional[set[str]] = None


@dataclass(frozen=True)
class QColumnId:
    """Qualified column id"""

    tbl_id: UUID
    col_id: int


class MediaValidation(enum.Enum):
    ON_READ = 0
    ON_WRITE = 1

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> MediaValidation:
        try:
            return cls[name.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{error_prefix} must be one of: [{val_strs}]') from None


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
            raise excs.Error(f'{param_name} must be one of: [{val_strs}]') from None


class IfNotExistsParam(enum.Enum):
    ERROR = 0
    IGNORE = 1

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfNotExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{param_name} must be one of: [{val_strs}]') from None


def is_valid_identifier(name: str, allow_system_identifiers: bool = False) -> bool:
    return name.isidentifier() and (allow_system_identifiers or not name.startswith('_'))


def is_valid_path(path: str, empty_is_valid: bool, allow_system_paths: bool = False) -> bool:
    if path == '':
        return empty_is_valid
    return all(is_valid_identifier(part, allow_system_paths) for part in path.split('.'))


def is_system_column_name(name: str) -> bool:
    from pixeltable.catalog import InsertableTable, View

    global _PREDEF_SYMBOLS  # noqa: PLW0603
    if _PREDEF_SYMBOLS is None:
        _PREDEF_SYMBOLS = set(itertools.chain(dir(InsertableTable), dir(View)))
    return name == _POS_COLUMN_NAME or name in _PREDEF_SYMBOLS
