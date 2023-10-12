from __future__ import annotations

import dataclasses
import logging
from typing import Optional, List, Dict, Any, Union
from uuid import UUID

from .table_base import TableBase
from .column import Column
from .table_version import TableVersion
from pixeltable import exceptions as exc


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

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        self.tbl_version.revert()

