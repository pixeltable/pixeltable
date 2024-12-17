from __future__ import annotations
import dataclasses
import enum
import itertools
import logging
from typing import Optional

import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')

# name of the position column in a component view
_POS_COLUMN_NAME = 'pos'
_ROWID_COLUMN_NAME = '_rowid'

# Set of symbols that are predefined in the `InsertableTable` class (and are therefore not allowed as column names).
# This will be populated lazily to avoid circular imports.
_PREDEF_SYMBOLS: Optional[set[str]] = None


@dataclasses.dataclass
class UpdateStatus:
    """
    Information about updates that resulted from a table operation.
    """
    num_rows: int = 0
    # TODO: disambiguate what this means: # of slots computed or # of columns computed?
    num_computed_values: int = 0
    num_excs: int = 0
    updated_cols: list[str] = dataclasses.field(default_factory=list)
    cols_with_excs: list[str] = dataclasses.field(default_factory=list)

    def __iadd__(self, other: 'UpdateStatus') -> 'UpdateStatus':
        self.num_rows += other.num_rows
        self.num_computed_values += other.num_computed_values
        self.num_excs += other.num_excs
        self.updated_cols = list(dict.fromkeys(self.updated_cols + other.updated_cols))
        self.cols_with_excs = list(dict.fromkeys(self.cols_with_excs + other.cols_with_excs))
        return self


class MediaValidation(enum.Enum):
    ON_READ = 0
    ON_WRITE = 1

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> MediaValidation:
        try:
            return cls[name.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__.keys())
            raise excs.Error(f'{error_prefix} must be one of: [{val_strs}]')

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
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__.keys())
            raise excs.Error(f'{param_name} must be one of: [{val_strs}]')

class IfNotExistsParam(enum.Enum):
    ERROR = 'error'
    IGNORE = 'ignore'

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfNotExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            raise excs.Error(f'{param_name} must be one of: {[e.value for e in cls]}')

def is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith('_')

def is_valid_path(path: str, empty_is_valid : bool) -> bool:
    if path == '':
        return empty_is_valid

    for part in path.split('.'):
        if not is_valid_identifier(part):
            return False
    return True

def is_system_column_name(name: str) -> bool:
    from pixeltable.catalog import InsertableTable, View

    global _PREDEF_SYMBOLS
    if _PREDEF_SYMBOLS is None:
        _PREDEF_SYMBOLS = set(itertools.chain(dir(InsertableTable), dir(View)))
    return name == _POS_COLUMN_NAME or name in _PREDEF_SYMBOLS
