from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, overload
from uuid import UUID

import sqlalchemy.orm as orm

import pixeltable
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
from pixeltable.env import Env

from .catalog import Catalog
from .globals import UpdateStatus
from .table import Table
from .table_version import TableVersion
from .table_version_path import TableVersionPath

_logger = logging.getLogger('pixeltable')


class InsertableTable(Table):
    """A `Table` that allows inserting and deleting rows."""

    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        tbl_version_path = TableVersionPath(tbl_version)
        super().__init__(tbl_version.id, dir_id, tbl_version.name, tbl_version_path)

    @classmethod
    def _display_name(cls) -> str:
        return 'table'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def _create(
            cls, dir_id: UUID, name: str, schema: Dict[str, ts.ColumnType], primary_key: List[str],
            num_retained_versions: int, comment: str
    ) -> InsertableTable:
        columns = cls._create_columns(schema)
        cls._verify_schema(columns)
        column_names = [col.name for col in columns]
        for pk_col in primary_key:
            if pk_col not in column_names:
                raise excs.Error(f'Primary key column {pk_col} not found in table schema')
            col = columns[column_names.index(pk_col)]
            if col.col_type.nullable:
                raise excs.Error(f'Primary key column {pk_col} cannot be nullable')
            col.is_pk = True

        with orm.Session(Env.get().engine, future=True) as session:
            _, tbl_version = TableVersion.create(session, dir_id, name, columns, num_retained_versions, comment)
            tbl = cls(dir_id, tbl_version)
            session.commit()
            cat = Catalog.get()
            cat.tbl_dependents[tbl._id] = []
            cat.tbls[tbl._id] = tbl

            _logger.info(f'Created table `{name}`, id={tbl_version.id}')
            print(f'Created table `{name}`.')
            return tbl

    @overload
    def insert(
            self, rows: Iterable[Dict[str, Any]], /, *, print_stats: bool = False, fail_on_exception: bool = True
    ) -> UpdateStatus: ...

    @overload
    def insert(self, *, print_stats: bool = False, fail_on_exception: bool = True, **kwargs: Any) -> UpdateStatus: ...

    def insert(
            self, rows: Optional[Iterable[dict[str, Any]]] = None, /, *, print_stats: bool = False,
            fail_on_exception: bool = True, **kwargs: Any
    ) -> UpdateStatus:
        if rows is None:
            rows = [kwargs]
        else:
            rows = list(rows)
            if len(kwargs) > 0:
                raise excs.Error('`kwargs` cannot be specified unless `rows is None`.')

        if not isinstance(rows, list):
            raise excs.Error('rows must be a list of dictionaries')
        if len(rows) == 0:
            raise excs.Error('rows must not be empty')
        for row in rows:
            if not isinstance(row, dict):
                raise excs.Error('rows must be a list of dictionaries')
        self._validate_input_rows(rows)
        result = self._tbl_version.insert(rows, print_stats=print_stats, fail_on_exception=fail_on_exception)

        if result.num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = \
                f' across {len(result.cols_with_excs)} column{"" if len(result.cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join(result.cols_with_excs)})'
        msg = (
            f'Inserted {result.num_rows} row{"" if result.num_rows == 1 else "s"} '
            f'with {result.num_excs} error{"" if result.num_excs == 1 else "s"}{cols_with_excs_str}.'
        )
        print(msg)
        _logger.info(f'InsertableTable {self._name}: {msg}')
        return result

    def _validate_input_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Verify that the input rows match the table schema"""
        valid_col_names = set(self.column_names())
        reqd_col_names = set(self._tbl_version_path.tbl_version.get_required_col_names())
        computed_col_names = set(self._tbl_version_path.tbl_version.get_computed_col_names())
        for row in rows:
            assert isinstance(row, dict)
            col_names = set(row.keys())
            if len(reqd_col_names - col_names) > 0:
                raise excs.Error(f'Missing required column(s) ({", ".join(reqd_col_names - col_names)}) in row {row}')

            for col_name, val in row.items():
                if col_name not in valid_col_names:
                    raise excs.Error(f'Unknown column name {col_name} in row {row}')
                if col_name in computed_col_names:
                    raise excs.Error(f'Value for computed column {col_name} in row {row}')

                # validate data
                col = self._tbl_version_path.get_column(col_name)
                try:
                    # basic sanity checks here
                    checked_val = col.col_type.create_literal(val)
                    row[col_name] = checked_val
                except TypeError as e:
                    msg = str(e)
                    raise excs.Error(f'Error in column {col.name}: {msg[0].lower() + msg[1:]}\nRow: {row}')

    def delete(self, where: Optional['pixeltable.exprs.Expr'] = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """
        return self._tbl_version.delete(where=where)
