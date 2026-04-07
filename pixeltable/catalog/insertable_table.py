from __future__ import annotations

import enum
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence, overload
from uuid import UUID

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec
from pixeltable.utils.filecache import FileCache

from .column import Column
from .globals import MediaValidation
from .table import Table
from .table_version import TableVersion, TableVersionMd
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .tbl_ops import CreateStoreTableOp, CreateTableMdOp, OpStatus, TableOp
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
        name: str,
        schema: dict[str, type | ColumnSpec | exprs.Expr],
        primary_key: list[str],
        num_retained_versions: int,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        create_default_idxs: bool,
    ) -> tuple[TableVersionMd, list[TableOp]]:
        columns = [Column.create(name, spec) for name, spec in schema.items()]
        cls._verify_schema(columns)
        column_names = [col.name for col in columns]
        for pk_col in primary_key:
            if pk_col not in column_names:
                raise excs.Error(f'Primary key column {pk_col!r} not found in table schema.')
            col = columns[column_names.index(pk_col)]
            if col.col_type.nullable:
                raise excs.Error(
                    f'Primary key column {pk_col!r} cannot be nullable. '
                    f'Declare it as `Required` instead: `pxt.Required[pxt.{col.col_type._to_base_str()}]`'
                )
            col.is_pk = True

        md = TableVersion.create_initial_md(
            name,
            columns,
            num_retained_versions,
            comment,
            custom_metadata,
            media_validation,
            create_default_idxs=create_default_idxs,
            view_md=None,
        )

        ops = [
            CreateTableMdOp(tbl_id=md.tbl_md.tbl_id, op_sn=0, num_ops=2, status=OpStatus.PENDING),
            CreateStoreTableOp(tbl_id=md.tbl_md.tbl_id, op_sn=1, num_ops=2, status=OpStatus.PENDING),
        ]
        return md, ops

    @overload
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
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
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        from pixeltable.io.table_data_conduit import TableDataConduit

        if source is not None and isinstance(source, Sequence) and len(source) == 0:
            raise excs.Error('Cannot insert an empty sequence.')
        fail_on_exception = OnErrorParameter.fail_on_exception(on_error)

        if source is None:
            source = [kwargs]
            kwargs = None

        data_source = TableDataConduit.create(
            source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
        )
        if data_source.source_column_map is None:
            data_source.src_pk = []

        data_source.add_table_info(self)
        data_source.prepare_for_insert_into_table()

        return self.insert_table_data_source(
            data_source=data_source, fail_on_exception=fail_on_exception, print_stats=print_stats
        )

    def insert_table_data_source(
        self, data_source: TableDataConduit, fail_on_exception: bool, print_stats: bool = False
    ) -> pxt.UpdateStatus:
        """Insert row batches into this table from a `TableDataConduit`."""
        from pixeltable.io.table_data_conduit import QueryTableDataConduit

        start_ts = time.perf_counter()
        status = pxt.UpdateStatus()
        with get_runtime().catalog.begin_xact(
            for_write=True, tvp_write_targets=[self._tbl_version_path], lock_mutable_tree=True
        ):
            if isinstance(data_source, QueryTableDataConduit):
                status += self._tbl_version.get().insert(
                    rows=None, query=data_source.pxt_query, print_stats=print_stats, fail_on_exception=fail_on_exception
                )
            else:
                for row_batch in data_source.valid_row_batch():
                    status += self._tbl_version.get().insert(
                        rows=row_batch, query=None, print_stats=print_stats, fail_on_exception=fail_on_exception
                    )

        Env.get().console_logger.info(status.insert_msg(start_ts))
        FileCache.get().emit_eviction_warnings()
        return status

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """
        with get_runtime().catalog.begin_xact(
            for_write=True, tvp_write_targets=[self._tbl_version_path], lock_mutable_tree=True
        ):
            return self._tbl_version.get().delete(where=where)

    def _get_base_table(self) -> 'Table' | None:
        return None

    @property
    def _effective_base_versions(self) -> list[int | None]:
        return []

    def _table_descriptor(self) -> str:
        return self._display_str()
