from __future__ import annotations

import logging
from typing import Optional, List, Any
from uuid import UUID

import sqlalchemy.orm as orm

from .column import Column
from .table_version import TableVersion
from .mutable_table import MutableTable
from pixeltable.env import Env
from pixeltable import exceptions as exc


_logger = logging.getLogger('pixeltable')

class InsertableTable(MutableTable):
    """A :py:class:`MutableTable` that allows inserting and deleting rows.
    """
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
            cls, dir_id: UUID, name: str, cols: List[Column],
            num_retained_versions: int,
    ) -> InsertableTable:
        cls._verify_columns(cols)
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, None, None, num_retained_versions, None, None, session)
            tbl = cls(dir_id, tbl_version)
            session.commit()
            _logger.info(f'created table {name}, id={tbl_version.id}')
            return tbl

    def insert(self, rows: List[List[Any]], columns: List[str] = [], print_stats: bool = False) -> MutableTable.UpdateStatus:
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
                    f'MutableTable {self.name} has {len(insertable_col_names)} user-supplied columns, but the data only '
                    f'contains {len(rows[0])} columns. In this case, you need to specify the column names with the '
                    f"'columns' parameter."))
            else:
                raise exc.Error((
                    f'MutableTable {self.name} has {len(insertable_col_names)} user-supplied columns, but the data '
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
        result = self.tbl_version.insert(rows, columns, print_stats=print_stats)

        if result.num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = \
                f'across {len(result.cols_with_excs)} column{"" if len(result.cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join(result.cols_with_excs)})'
        msg = (
            f'inserted {result.num_rows} row{"" if result.num_rows == 1 else "s"} '
            f'with {result.num_excs} error{"" if result.num_excs == 1 else "s"} {cols_with_excs_str}'
        )
        print(msg)
        _logger.info(f'InsertableTable {self.name}: {msg}')
        return result

    def delete(self, where: Optional['pixeltable.exprs.Predicate'] = None) -> MutableTable.UpdateStatus:
        """Delete rows in this table.
        Args:
            where: a Predicate to filter rows to delete.
        """
        from pixeltable.exprs import Predicate
        from pixeltable.plan import Planner
        if where is not None:
            if not isinstance(where, Predicate):
                raise exc.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.tbl_version, where)
            if analysis_info.similarity_clause is not None:
                raise exc.Error('nearest() cannot be used with delete()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise exc.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        return self.tbl_version.delete(where)
