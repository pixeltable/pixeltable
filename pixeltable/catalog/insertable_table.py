from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from uuid import UUID

import pixeltable as pxt
from pixeltable import exceptions as excs, telemetry, type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.filecache import FileCache

from .column import Column
from .globals import IndexSpec, MediaValidation, OnErrorParam
from .local_table import LocalTable
from .table_path import TableVersionPath
from .table_version_handle import TableVersionHandle
from .tbl_ops import CreateTableMdOp, TableOp, TableOpsBuilder
from .types import TableVersionMd
from .update_status import UpdateStatus
from .utils import create_table_version_md

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.globals import TableDataSource
    from pixeltable.io.data_sources import SqlDataSource
    from pixeltable.io.table_data_conduit import TableDataConduit

    from .table import Table


class InsertableTable(LocalTable):
    """A `Table` that allows inserting and deleting rows."""

    def __init__(self, tbl_version: TableVersionHandle):
        tbl_version_path = TableVersionPath(tbl_version)
        super().__init__(tbl_version.id, tbl_version_path)
        self._tbl_version = tbl_version

    def _display_name(self) -> str:
        return 'table'

    @classmethod
    def _create(
        cls,
        tbl_id: UUID,
        name: str,
        columns: list[Column],
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        has_default_idxs: bool,
        is_data_versioned: bool,
        additional_idxs: list[IndexSpec],
    ) -> tuple[TableVersionMd, list[TableOp]]:
        cls._verify_schema(columns)
        cols_by_name = {col.name: col for col in columns if col.name is not None}
        assert all(isinstance(spec.indexed_column, str) for spec in additional_idxs)
        resolved_idxs = [
            IndexSpec(indexed_column=cols_by_name[cast(str, spec.indexed_column)], idx_name=spec.idx_name, idx=spec.idx)
            for spec in additional_idxs
        ]

        md = create_table_version_md(
            tbl_id,
            name,
            columns,
            comment,
            custom_metadata,
            media_validation,
            has_default_idxs=has_default_idxs,
            view_md=None,
            is_data_versioned=is_data_versioned,
            additional_idxs=resolved_idxs,
        )

        ops = (
            TableOpsBuilder(md.tbl_md.tbl_id, tbl_version=md.tbl_md.current_version)
            .add(CreateTableMdOp, is_view=False)
            .build()
        )
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
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self,
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
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
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        from pixeltable.io.table_data_conduit import TableDataConduit

        self._validate_insert_source(source)
        fail_on_exception = OnErrorParam.fail_on_exception(on_error)

        if source is None:
            source = [kwargs]
            kwargs = None

        with telemetry.span('pixeltable.insert', set_current=True):
            with telemetry.span('pixeltable.data_source.prepare'):
                data_source = TableDataConduit.create(
                    source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
                )
                data_source.add_table_info(self)
                data_source.prepare_for_insert_into_table()

            return self._insert_table_data_source(
                data_source=data_source,
                fail_on_exception=fail_on_exception,
                print_stats=print_stats,
                return_rows=return_rows,
            )

    def _insert_table_data_source(
        self,
        data_source: TableDataConduit,
        fail_on_exception: bool,
        print_stats: bool = False,
        return_rows: bool = False,
    ) -> pxt.UpdateStatus:
        """Insert row batches into this table from a `TableDataConduit`."""
        from pixeltable.io.table_data_conduit import QueryTableDataConduit

        start_ts = time.perf_counter()
        status = pxt.UpdateStatus()
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            # in on_error='abort' mode the exec raises the internal ExprEvalError on the first failing cell;
            # convert it to a user-facing Error (on_error='ignore' records per-cell errors and never raises)
            try:
                if isinstance(data_source, QueryTableDataConduit):
                    status += self._tbl_version.get().insert(
                        source=None,
                        query=data_source.pxt_query,
                        print_stats=print_stats,
                        fail_on_exception=fail_on_exception,
                    )
                else:
                    for row_batch in data_source.valid_row_batch():
                        status += self._tbl_version.get().insert(
                            source=row_batch,
                            query=None,
                            print_stats=print_stats,
                            fail_on_exception=fail_on_exception,
                            return_rows=return_rows,
                        )
            except excs.ExprEvalError as e:
                excs.raise_from_expr_eval_err(e)

        Env.get().console_logger.info(status.insert_msg(start_ts))
        FileCache.get().emit_eviction_warnings()
        return status

    def _insert_sql_source(
        self,
        sql_source: SqlDataSource,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        send_connect_url: bool = False,
    ) -> pxt.UpdateStatus:
        """Stream a SqlDataSource into this table through a single insert plan.

        Assumes the source's columns have already been validated against this table's schema.
        """
        fail_on_exception = OnErrorParam.fail_on_exception(on_error)
        start_ts = time.perf_counter()
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            status = self._tbl_version.get().insert(
                source=sql_source,
                query=None,
                print_stats=print_stats,
                fail_on_exception=fail_on_exception,
                return_rows=return_rows,
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
        self._validate_where(where)
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            return self._tbl_version.get().delete(where=where)

    def _get_base_table(self) -> 'Table' | None:
        return None

    @property
    def _effective_base_versions(self) -> list[int | None]:
        return []

    def _table_descriptor(self, path: 'pxt.catalog.Path | None' = None) -> str:
        return self._display_str(path)
