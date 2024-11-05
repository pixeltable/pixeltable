from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Optional, overload
from uuid import UUID

import sqlalchemy.orm as orm

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.filecache import FileCache

from .catalog import Catalog
from .globals import MediaValidation, UpdateStatus
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
        cls, dir_id: UUID, name: str, schema: dict[str, ts.ColumnType], df: Optional[pxt.DataFrame],
        primary_key: list[str], num_retained_versions: int, comment: str, media_validation: MediaValidation
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
            _, tbl_version = TableVersion.create(
                session, dir_id, name, columns, num_retained_versions=num_retained_versions, comment=comment,
                media_validation=media_validation)
            tbl = cls(dir_id, tbl_version)
            # TODO We need to commit before doing the insertion, in order to avoid a primary key (version) collision
            #   when the table metadata gets updated. Once we have a notion of user-defined transactions in
            #   Pixeltable, we can wrap the create/insert in a transaction to avoid this.
            session.commit()
            if df is not None:
                # A DataFrame was provided, so insert its contents into the table
                # (using the same DB session as the table creation)
                tbl_version.insert(None, df, conn=session.connection(), fail_on_exception=True)
            session.commit()
            cat = Catalog.get()
            cat.tbl_dependents[tbl._id] = []
            cat.tbls[tbl._id] = tbl

            _logger.info(f'Created table `{name}`, id={tbl_version.id}')
            print(f'Created table `{name}`.')
            return tbl

    def get_metadata(self) -> dict[str, Any]:
        md = super().get_metadata()
        md['is_view'] = False
        md['is_snapshot'] = False
        return md

    @overload
    def insert(
        self,
        rows: Iterable[dict[str, Any]],
        /,
        *,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort'
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self,
        *,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        **kwargs: Any
    ) -> UpdateStatus: ...

    def insert(  # type: ignore[misc]
        self,
        rows: Optional[Iterable[dict[str, Any]]] = None,
        /,
        *,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        **kwargs: Any
    ) -> UpdateStatus:
        if rows is None:
            rows = [kwargs]
        else:
            rows = list(rows)
            if len(kwargs) > 0:
                raise excs.Error('`kwargs` cannot be specified unless `rows is None`.')

        fail_on_exception = on_error == 'abort'

        if not isinstance(rows, list):
            raise excs.Error('rows must be a list of dictionaries')
        if len(rows) == 0:
            raise excs.Error('rows must not be empty')
        for row in rows:
            if not isinstance(row, dict):
                raise excs.Error('rows must be a list of dictionaries')
        self._validate_input_rows(rows)
        status = self._tbl_version.insert(rows, None, print_stats=print_stats, fail_on_exception=fail_on_exception)

        if status.num_excs == 0:
            cols_with_excs_str = ''
        else:
            cols_with_excs_str = \
                f' across {len(status.cols_with_excs)} column{"" if len(status.cols_with_excs) == 1 else "s"}'
            cols_with_excs_str += f' ({", ".join(status.cols_with_excs)})'
        msg = (
            f'Inserted {status.num_rows} row{"" if status.num_rows == 1 else "s"} '
            f'with {status.num_excs} error{"" if status.num_excs == 1 else "s"}{cols_with_excs_str}.'
        )
        print(msg)
        _logger.info(f'InsertableTable {self._name}: {msg}')
        FileCache.get().emit_eviction_warnings()
        return status

    def _validate_input_rows(self, rows: list[dict[str, Any]]) -> None:
        """Verify that the input rows match the table schema"""
        valid_col_names = set(self._schema.keys())
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

    def delete(self, where: Optional['pxt.exprs.Expr'] = None) -> UpdateStatus:
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
