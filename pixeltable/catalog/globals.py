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


@dataclasses.dataclass()
class RowCountStats:
    """
    Statistics about the counts of rows affected by a table operation.
    """

    ins_rows: int = 0  # rows inserted
    del_rows: int = 0  # rows deleted
    upd_rows: int = 0  # rows updated
    exc_rows: int = 0  # rows that had exceptions during the operation
    num_excs: int = 0  # total number of exceptions

    @property
    def is_empty(self) -> bool:
        """
        Return true if the row count stats are empty (i.e., all counts are zero).
        """
        return (
            self.ins_rows == 0
            and self.del_rows == 0
            and self.upd_rows == 0
            and self.exc_rows == 0
            and self.num_excs == 0
        )

    def accumulate(self, other: 'RowCountStats') -> 'RowCountStats':
        """
        Accumulate the row count stats from another RowCountStats object.
        """
        return RowCountStats(
            ins_rows=self.ins_rows + other.ins_rows,
            del_rows=self.del_rows + other.del_rows,
            upd_rows=self.upd_rows + other.upd_rows,
            exc_rows=self.exc_rows + other.exc_rows,
            num_excs=self.num_excs + other.num_excs,
        )


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

    # stats for the rows affected by the operation
    row_count_stats: RowCountStats = dataclasses.field(default_factory=lambda: RowCountStats())

    # stats for changes cascaded to other tables
    cascade_row_count_stats: RowCountStats = dataclasses.field(default_factory=lambda: RowCountStats())

    def accumulate(self, other: 'UpdateStatus', cascade: bool = False) -> None:
        """
        Accumulate the update status from another UpdateStatus object.
        """
        self.num_rows += other.num_rows
        self.num_computed_values += other.num_computed_values
        self.num_excs += other.num_excs
        self.updated_cols = list(dict.fromkeys(self.updated_cols + other.updated_cols))
        self.cols_with_excs = list(dict.fromkeys(self.cols_with_excs + other.cols_with_excs))

        if cascade:
            # If the other stats are a result of a cascaded operation, accumulate all stats into cascade stats
            self.cascade_row_count_stats = self.cascade_row_count_stats.accumulate(other.row_count_stats)
        else:
            self.row_count_stats = self.row_count_stats.accumulate(other.row_count_stats)
        self.cascade_row_count_stats = self.cascade_row_count_stats.accumulate(other.cascade_row_count_stats)

    @property
    def insert_msg(self) -> str:
        """Return a message describing the results of an insert operation."""
        if self.num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = (
                f' across {len(self.cols_with_excs)} column{"" if len(self.cols_with_excs) == 1 else "s"}'
            )
            cols_with_excs_str += f' ({", ".join(self.cols_with_excs)})'
        msg = (
            f'Inserted {self.num_rows} row{"" if self.num_rows == 1 else "s"} '
            f'with {self.num_excs} error{"" if self.num_excs == 1 else "s"}{cols_with_excs_str}.'
        )
        return msg


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
