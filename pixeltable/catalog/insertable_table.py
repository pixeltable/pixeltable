from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, overload
from uuid import UUID

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.utils.filecache import FileCache

from .globals import MediaValidation
from .table import Table
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.globals import TableDataSource
    from pixeltable.io.table_data_conduit import TableDataConduit

_logger = logging.getLogger('pixeltable')


class OnErrorParameter(enum.Enum):
    """Supported values for the on_error parameter"""

    ABORT = 'abort'
    IGNORE = 'ignore'

    @classmethod
    def is_valid(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in [c.value for c in cls]
        return False

    @classmethod
    def fail_on_exception(cls, v: Any) -> bool:
        if not cls.is_valid(v):
            raise ValueError(f'Invalid value for on_error: {v}')
        if isinstance(v, str):
            return v.lower() != cls.IGNORE.value
        return True


class InsertableTable(Table):
    """A `Table` that allows inserting and deleting rows."""

    def __init__(self, dir_id: UUID, tbl_version: TableVersionHandle):
        tbl_version_path = TableVersionPath(tbl_version)
        super().__init__(tbl_version.id, dir_id, tbl_version.get().name, tbl_version_path)
        self._tbl_version = tbl_version

    @classmethod
    def _display_name(cls) -> str:
        return 'table'

    @classmethod
    def _create(
        cls,
        dir_id: UUID,
        name: str,
        schema: dict[str, ts.ColumnType],
        df: Optional[pxt.DataFrame],
        primary_key: list[str],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
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

        _, tbl_version = TableVersion.create(
            dir_id,
            name,
            columns,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation,
        )
        tbl = cls(dir_id, TableVersionHandle.create(tbl_version))
        # TODO We need to commit before doing the insertion, in order to avoid a primary key (version) collision
        #   when the table metadata gets updated. Once we have a notion of user-defined transactions in
        #   Pixeltable, we can wrap the create/insert in a transaction to avoid this.
        session = Env.get().session
        session.commit()
        if df is not None:
            # A DataFrame was provided, so insert its contents into the table
            # (using the same DB session as the table creation)
            tbl_version.insert(None, df, fail_on_exception=True)
        session.commit()

        _logger.info(f'Created table `{name}`, id={tbl_version.id}')
        Env.get().console_logger.info(f'Created table `{name}`.')
        return tbl

    def _get_metadata(self) -> dict[str, Any]:
        md = super()._get_metadata()
        md['base'] = None
        md['is_view'] = False
        md['is_snapshot'] = False
        return md

    @overload
    def insert(
        self,
        source: Optional[TableDataSource] = None,
        /,
        *,
        source_format: Optional[Literal['csv', 'excel', 'parquet', 'json']] = None,
        schema_overrides: Optional[dict[str, ts.ColumnType]] = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self, /, *, on_error: Literal['abort', 'ignore'] = 'abort', print_stats: bool = False, **kwargs: Any
    ) -> UpdateStatus: ...

    def insert(
        self,
        source: Optional[TableDataSource] = None,
        /,
        *,
        source_format: Optional[Literal['csv', 'excel', 'parquet', 'json']] = None,
        schema_overrides: Optional[dict[str, ts.ColumnType]] = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        from pixeltable.catalog import Catalog
        from pixeltable.io.table_data_conduit import UnkTableDataConduit

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            table = self
            if source is None:
                source = [kwargs]
                kwargs = None

            tds = UnkTableDataConduit(
                source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
            )
            data_source = tds.specialize()
            if data_source.source_column_map is None:
                data_source.src_pk = []

            assert isinstance(table, Table)
            data_source.add_table_info(table)
            data_source.prepare_for_insert_into_table()

            fail_on_exception = OnErrorParameter.fail_on_exception(on_error)
            return table.insert_table_data_source(
                data_source=data_source, fail_on_exception=fail_on_exception, print_stats=print_stats
            )

    def insert_table_data_source(
        self, data_source: TableDataConduit, fail_on_exception: bool, print_stats: bool = False
    ) -> pxt.UpdateStatus:
        """Insert row batches into this table from a `TableDataConduit`."""
        from pixeltable.catalog import Catalog
        from pixeltable.io.table_data_conduit import DFTableDataConduit

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            if isinstance(data_source, DFTableDataConduit):
                status = pxt.UpdateStatus()
                status += self._tbl_version.get().insert(
                    rows=None, df=data_source.pxt_df, print_stats=print_stats, fail_on_exception=fail_on_exception
                )
            else:
                status = pxt.UpdateStatus()
                for row_batch in data_source.valid_row_batch():
                    status += self._tbl_version.get().insert(
                        rows=row_batch, df=None, print_stats=print_stats, fail_on_exception=fail_on_exception
                    )

        Env.get().console_logger.info(status.insert_msg)

        FileCache.get().emit_eviction_warnings()
        return status

    def _validate_input_rows(self, rows: list[dict[str, Any]]) -> None:
        """Verify that the input rows match the table schema"""
        valid_col_names = set(self._get_schema().keys())
        reqd_col_names = set(self._tbl_version_path.tbl_version.get().get_required_col_names())
        computed_col_names = set(self._tbl_version_path.tbl_version.get().get_computed_col_names())
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
                    raise excs.Error(f'Error in column {col.name}: {msg[0].lower() + msg[1:]}\nRow: {row}') from e

    def delete(self, where: Optional['exprs.Expr'] = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            return self._tbl_version.get().delete(where=where)

    def _get_base_table(self) -> Optional['Table']:
        return None

    @property
    def _effective_base_versions(self) -> list[Optional[int]]:
        return []

    def _table_descriptor(self) -> str:
        return f'Table {self._path()!r}'
