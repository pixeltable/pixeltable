from __future__ import annotations

import logging
from typing import Optional, List, Any, Dict
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
        cls._verify_user_columns(cols)
        with orm.Session(Env.get().engine, future=True) as session:
            tbl_version = TableVersion.create(
                dir_id, name, cols, None, None, num_retained_versions, None, None, session)
            tbl = cls(dir_id, tbl_version)
            session.commit()
            _logger.info(f'created table {name}, id={tbl_version.id}')
            return tbl

    def insert(
            self, rows: List[Dict[str, Any]], print_stats: bool = False, fail_on_exception : bool = True
    ) -> MutableTable.UpdateStatus:
        """Insert rows into table.

        Args:
            rows: A list of rows to insert, each of which is a dictionary mapping column names to values.
            print_stats: If ``True``, print statistics about the cost of computed columns.
            fail_on_exception:
                Determines how exceptions in computed columns and invalid media files (e.g., corrupt images)
                are handled.
                If ``False``, store error information (accessible as column properties 'errortype' and 'errormsg')
                for those cases, but continue inserting rows.
                If ``True``, raise an exception that aborts the insert.
        Returns:
            execution status

        Raises:
            Error: if a row does not match the table schema or contains values for computed columns

        Examples:
            Insert two rows into a table with three int columns ``a``, ``b``, and ``c``. Column ``c`` is nullable.

            >>> tbl.insert([{'a': 1, 'b': 1, 'c': 1}, {'a': 2, 'b': 2}])
        """
        if not isinstance(rows, list):
            raise exc.Error('rows must be a list of dictionaries')
        if len(rows) == 0:
            raise exc.Error('rows must not be empty')
        for row in rows:
            if not isinstance(row, dict):
                raise exc.Error('rows must be a list of dictionaries')
        self._validate_input_rows(rows)
        result = self.tbl_version.insert(rows, print_stats=print_stats, fail_on_exception=fail_on_exception)

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

    def _validate_input_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Verify that the input rows match the table schema"""
        valid_col_names = set(self.column_names())
        reqd_col_names = set(self.tbl_version.get_required_col_names())
        computed_col_names = set(self.tbl_version.get_computed_col_names())
        for row in rows:
            assert isinstance(row, dict)
            col_names = set(row.keys())
            if len(reqd_col_names - col_names) > 0:
                raise exc.Error(f'Missing required column(s) ({", ".join(reqd_col_names - col_names)}) in row {row}')

            for col_name, val in row.items():
                if col_name not in valid_col_names:
                    raise exc.Error(f'Unknown column name {col_name} in row {row}')
                if col_name in computed_col_names:
                    raise exc.Error(f'Value for computed column {col_name} in row {row}')

                # validate data
                col = self.tbl_version.get_column(col_name)
                try:
                    # basic sanity checks here
                    checked_val = col.col_type.create_literal(val)
                    row[col_name] = checked_val
                except TypeError as e:
                    msg = str(e)
                    raise exc.Error(f'Error in column {col.name}: {msg[0].lower() + msg[1:]}\nRow: {row}')

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
