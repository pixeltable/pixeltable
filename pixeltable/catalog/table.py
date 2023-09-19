from __future__ import annotations

import dataclasses
import logging
from typing import Optional, List, Dict, Any, Union
from uuid import UUID

from .table_base import TableBase
from .column import Column
from .table_version import TableVersion
from pixeltable import exceptions as exc

_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


_logger = logging.getLogger('pixeltable')

class Table(TableBase):
    """Base class for MutableTable and View"""

    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int = 0
        # TODO: disambiguate what this means: # of slots computed or # of columns computed?
        num_computed_values: int = 0
        num_excs: int = 0
        updated_cols: List[str] = dataclasses.field(default_factory=list)
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    def __init__(self, id: UUID, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(id, dir_id, tbl_version.name, tbl_version)


    def add_column(self, col: Column, print_stats: bool = False) -> Table.UpdateStatus:
        """Adds a column to the table.

        Args:
            col: The column to add.

        Returns:
            execution status

        Raises:
            Error: If the column name is invalid or already exists.

        Examples:
            Add an int column with ``None`` values:

            >>> tbl.add_column(Column('new_col', IntType()))

            For a table with int column ``x``, add a column that is the factorial of ``x``. Note that the names of
            the parameters of the ``computed_with`` Callable must correspond to existing column names (the column
            values are then passed as arguments to the Callable):

            >>> tbl.add_column(Column('factorial', IntType(), computed_with=lambda x: math.factorial(x)))

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees (note that in this case, the column type is inferred from the ``computed_with`` expression):

            >>> tbl.add_column(Column('rotated', computed_with=tbl.frame.rotate(90)))
            'added ...'
        """
        self._check_is_dropped()
        return self.tbl_version.add_column(col, print_stats=print_stats)

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.

        Args:
            name: The name of the column to drop.

        Raises:
            Error: If the column does not exist or if it is referenced by a computed column.

        Example:
            >>> tbl.drop_column('factorial')
        """
        self._check_is_dropped()
        self.tbl_version.drop_column(name)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist or if the new name is invalid or already exists.

        Example:
            >>> tbl.rename_column('factorial', 'fac')
        """
        self._check_is_dropped()
        self.tbl_version.rename_column(old_name, new_name)

    def insert(self, rows: List[List[Any]], columns: List[str] = [], print_stats: bool = False) -> Table.UpdateStatus:
        """Insert rows into table.

        Args:
            rows: A list of rows to insert. Each row is a list of values, one for each column.
            columns: A list of column names that specify the columns present in ``rows``.
                If ``columns`` is empty, all non-computed columns are present in ``rows``.
            print_stats: If ``True``, print statistics about the cost of computed columns.

        Returns:
            execution status

        Raises:
            Error: If the number of columns in ``rows`` does not match the number of columns in the table or in
            ``columns``.

        Examples:
            Insert two rows into a table with three int columns ``a``, ``b``, and ``c``. Note that the ``columns``
            argument is required here because ``rows`` only contain two columns.

            >>> tbl.insert([[1, 1], [2, 2]], columns=['a', 'b'])

            Assuming a table with columns ``video``, ``frame`` and ``frame_idx`` and set up for automatic frame extraction,
            insert a single row containing a video file path (the video contains 100 frames). The row will be expanded
            into 100 rows, one for each frame, and the ``frame`` and ``frame_idx`` columns will be populated accordingly.
            Note that the ``columns`` argument is unnecessary here because only the ``video`` column is required.

            >>> tbl.insert([['/path/to/video.mp4']])

        """
        if not isinstance(rows, list):
            raise exc.Error('rows must be a list of lists')
        if len(rows) == 0:
            raise exc.Error('rows must not be empty')
        for row in rows:
            if not isinstance(row, list):
                raise exc.Error('rows must be a list of lists')
        if not isinstance(columns, list):
            raise exc.Error('columns must be a list of column names')
        for col_name in columns:
            if not isinstance(col_name, str):
                raise exc.Error('columns must be a list of column names')

        insertable_col_names = self.tbl_version.get_insertable_col_names()
        if len(columns) == 0 and len(rows[0]) != len(insertable_col_names):
                if len(rows[0]) < len(insertable_col_names):
                    raise exc.Error((
                        f'Table {self.name} has {len(insertable_col_names)} user-supplied columns, but the data only '
                        f'contains {len(rows[0])} columns. In this case, you need to specify the column names with the '
                        f"'columns' parameter."))
                else:
                    raise exc.Error((
                        f'Table {self.name} has {len(insertable_col_names)} user-supplied columns, but the data '
                        f'contains {len(rows[0])} columns. '))

        # make sure that each row contains the same number of values
        num_col_vals = len(rows[0])
        for i in range(1, len(rows)):
            if len(rows[i]) != num_col_vals:
                raise exc.Error(
                    f'Inconsistent number of column values in rows: row 0 has {len(rows[0])}, '
                    f'row {i} has {len(rows[i])}')

        if len(columns) == 0:
            columns = insertable_col_names
        if len(rows[0]) != len(columns):
            raise exc.Error(
                f'The number of column values in rows ({len(rows[0])}) does not match the given number of column names '
                f'({", ".join(columns)})')

        self.tbl_version.check_input_rows(rows, columns)
        return self.tbl_version.insert(rows, columns, print_stats=print_stats)

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> Table.UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
        """
        return self.tbl_version.update(value_spec, where, cascade)

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> Table.UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        return self.tbl_version.delete(where)

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        self.tbl_version.revert()

