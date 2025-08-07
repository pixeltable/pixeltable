from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, overload
from uuid import UUID

import pydantic

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

    def _display_name(self) -> str:
        assert not self._tbl_version_path.is_replica()
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
                raise excs.Error(f'Primary key column {pk_col!r} not found in table schema.')
            col = columns[column_names.index(pk_col)]
            if col.col_type.nullable:
                raise excs.Error(f'Primary key column {pk_col!r} cannot be nullable.')
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

        _logger.info(f'Created table {name!r}, id={tbl_version.id}')
        Env.get().console_logger.info(f'Created table {name!r}.')
        return tbl

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

    def insert_pydantic(self, rows: Iterable[pydantic.BaseModel]) -> UpdateStatus:

        from pixeltable.catalog import Catalog

        rows_list = list(rows)
        if not rows_list:
            return UpdateStatus()

        model_class = type(rows_list[0])
        self._validate_pydantic_model(model_class)
        pxt_rows = [row.model_dump() for row in rows_list]
        # explicitly check that all required columns are present and non-None in the rows, because we ignore nullability
        # when validating the pydantic model
        reqd_col_names = [col.name for col in self._tbl_version_path.columns() if col.is_required_for_insert]
        for i, pxt_row in enumerate(pxt_rows):
            for col_name in reqd_col_names:
                if pxt_row.get(col_name) is None:
                    raise excs.Error(f'Missing required column {col_name!r} in {repr(rows_list[i])}')

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            status = self._tbl_version.get().insert(rows=pxt_rows, df=None, print_stats=False, fail_on_exception=True)

        Env.get().console_logger.info(status.insert_msg)
        FileCache.get().emit_eviction_warnings()
        return status

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
        return self._display_str()
