from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


@dataclass(frozen=True)
class RowCountStats:
    """
    Statistics about the counts of rows affected by a table operation.
    """

    ins_rows: int = 0  # rows inserted
    del_rows: int = 0  # rows deleted
    upd_rows: int = 0  # rows updated
    num_excs: int = 0  # total number of exceptions
    # TODO: disambiguate what this means: # of slots computed or # of columns computed?
    computed_values: int = 0  # number of computed values (e.g., computed columns) affected by the operation

    @property
    def num_rows(self) -> int:
        return self.ins_rows + self.del_rows + self.upd_rows

    def insert_to_update(self) -> 'RowCountStats':
        """
        Convert insert row count stats to update row count stats.
        This is used when an insert operation is treated as an update.
        """
        return RowCountStats(
            ins_rows=0,
            del_rows=self.del_rows,
            upd_rows=self.upd_rows + self.ins_rows,
            num_excs=self.num_excs,
            computed_values=self.computed_values,
        )

    def __add__(self, other: 'RowCountStats') -> 'RowCountStats':
        """
        Add the stats from two RowCountStats objects together.
        """
        return RowCountStats(
            ins_rows=self.ins_rows + other.ins_rows,
            del_rows=self.del_rows + other.del_rows,
            upd_rows=self.upd_rows + other.upd_rows,
            num_excs=self.num_excs + other.num_excs,
            computed_values=self.computed_values + other.computed_values,
        )


@dataclass(frozen=True)
class UpdateStatus:
    """
    Information about changes to table data or table schema
    """

    updated_cols: list[str] = field(default_factory=list)
    cols_with_excs: list[str] = field(default_factory=list)

    # stats for the rows affected by the operation
    row_count_stats: RowCountStats = field(default_factory=RowCountStats)

    # stats for changes cascaded to other tables
    cascade_row_count_stats: RowCountStats = field(default_factory=RowCountStats)

    # stats for the rows affected by the operation in an external store
    ext_row_count_stats: RowCountStats = field(default_factory=RowCountStats)

    @property
    def num_rows(self) -> int:
        return self.row_count_stats.num_rows + self.cascade_row_count_stats.num_rows

    @property
    def num_excs(self) -> int:
        return self.row_count_stats.num_excs + self.cascade_row_count_stats.num_excs

    @property
    def num_computed_values(self) -> int:
        return self.row_count_stats.computed_values + self.cascade_row_count_stats.computed_values

    def insert_to_update(self) -> 'UpdateStatus':
        """
        Convert the update status from an insert operation to an update operation.
        This is used when an insert operation is treated as an update.
        """
        return UpdateStatus(
            updated_cols=self.updated_cols,
            cols_with_excs=self.cols_with_excs,
            row_count_stats=self.row_count_stats.insert_to_update(),
            cascade_row_count_stats=self.cascade_row_count_stats.insert_to_update(),
            ext_row_count_stats=self.ext_row_count_stats,
        )

    def to_cascade(self) -> 'UpdateStatus':
        """
        Convert the update status to a cascade update status.
        This is used when an operation cascades changes to other tables.
        """
        return UpdateStatus(
            updated_cols=self.updated_cols,
            cols_with_excs=self.cols_with_excs,
            row_count_stats=RowCountStats(),
            cascade_row_count_stats=self.cascade_row_count_stats + self.row_count_stats,
            ext_row_count_stats=self.ext_row_count_stats,
        )

    def __add__(self, other: 'UpdateStatus') -> UpdateStatus:
        """
        Add the update status from two UpdateStatus objects together.
        """
        return UpdateStatus(
            updated_cols=list(dict.fromkeys(self.updated_cols + other.updated_cols)),
            cols_with_excs=list(dict.fromkeys(self.cols_with_excs + other.cols_with_excs)),
            row_count_stats=self.row_count_stats + other.row_count_stats,
            cascade_row_count_stats=self.cascade_row_count_stats + other.cascade_row_count_stats,
            ext_row_count_stats=self.ext_row_count_stats + other.ext_row_count_stats,
        )

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

    @classmethod
    def __cnt_str(cls, cnt: int, item: str) -> str:
        assert cnt > 0
        return f'{cnt} {item}{"" if cnt == 1 else "s"}'

    def _repr_pretty_(self, p: 'RepresentationPrinter', cycle: bool) -> None:
        messages = []
        # Combine row count stats and cascade row count stats
        stats = self.row_count_stats + self.cascade_row_count_stats
        if stats.ins_rows > 0:
            messages.append(f'{self.__cnt_str(stats.ins_rows, "row")} inserted')
        if stats.del_rows > 0:
            messages.append(f'{self.__cnt_str(stats.del_rows, "row")} deleted')
        if stats.upd_rows > 0:
            messages.append(f'{self.__cnt_str(stats.upd_rows, "row")} updated')
        if stats.computed_values > 0:
            messages.append(f'{self.__cnt_str(stats.computed_values, "value")} computed')
        if stats.num_excs > 0:
            messages.append(self.__cnt_str(stats.num_excs, 'exception'))
        p.text(', '.join(messages) + '.' if len(messages) > 0 else 'No rows affected.')

    @property
    def pxt_rows_updated(self) -> int:
        """
        Returns the number of Pixeltable rows that were updated as a result of the operation.
        """
        return (self.row_count_stats + self.cascade_row_count_stats).upd_rows

    @property
    def external_rows_updated(self) -> int:
        return self.ext_row_count_stats.upd_rows

    @property
    def external_rows_created(self) -> int:
        return self.ext_row_count_stats.ins_rows

    @property
    def external_rows_deleted(self) -> int:
        return self.ext_row_count_stats.del_rows

    @property
    def ext_num_rows(self) -> int:
        return self.ext_row_count_stats.num_rows
