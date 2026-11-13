from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from uuid import UUID

import pixeltable as pxt
from pixeltable import exceptions as excs, telemetry, telemetry_schemas, type_system as ts
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
    from pixeltable.api import TableDataSource
    from pixeltable.io.data_sources import SqlDataSource
    from pixeltable.io.source_adapter import SourceAdapter

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
        if_exists: Literal['error', 'ignore'] = 'error',
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
        if_exists: Literal['error', 'ignore'] = 'error',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @telemetry.spanned('pixeltable.insert', set_current=True)
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        if_exists: Literal['error', 'ignore'] = 'error',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        from pixeltable.io.source_adapter import SourceAdapter

        self._validate_insert_source(source)
        fail_on_exception = OnErrorParam.fail_on_exception(on_error)
        if if_exists not in ('error', 'ignore'):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f"insert(): if_exists must be 'error' or 'ignore', not {if_exists!r}; to update existing rows, use "
                "batch_update(rows, if_not_exists='insert')",
            )

        if source is None:
            source = [kwargs]
            kwargs = None

        telemetry.add_attrs(telemetry.func_span(), **telemetry_schemas.OpAttrs(table_id=str(self._id)))
        with telemetry.span('pixeltable.data_source.prepare'):
            data_source = SourceAdapter.create(
                source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
            )

        status = self._insert_table_data_source(
            data_source=data_source,
            fail_on_exception=fail_on_exception,
            print_stats=print_stats,
            return_rows=return_rows,
            skip_existing_keys=if_exists == 'ignore',
        )
        telemetry.add_attrs(telemetry.func_span(), **telemetry_schemas.op_attrs_from_update_status(status))
        return status

    def _insert_table_data_source(
        self,
        data_source: SourceAdapter,
        fail_on_exception: bool,
        print_stats: bool = False,
        return_rows: bool = False,
        skip_existing_keys: bool = False,
    ) -> pxt.UpdateStatus:
        """Insert the rows of a `SourceAdapter` into this table."""
        from pixeltable.io.source_adapter import QueryAdapter

        start_ts = time.perf_counter()
        query = data_source.pxt_query if isinstance(data_source, QueryAdapter) else None
        read_tbl_keys = query.referenced_tbl_keys() if query is not None else None
        # ON CONFLICT DO NOTHING skips a concurrent writer's key only under READ COMMITTED, not REPEATABLE READ
        read_committed = skip_existing_keys and not self._tbl_version_path.is_data_versioned()
        with get_runtime().catalog.begin_data_write_xact(
            read_tbl_keys=read_tbl_keys, tvps=[self._tbl_version_path], read_committed=read_committed
        ):
            if skip_existing_keys:
                self._validate_key_matching()
            data_source.add_table_info(self)
            data_source.prepare_for_insert_into_table()
            # in on_error='abort' mode the exec raises the internal ExprEvalError on the first failing cell;
            # convert it to a user-facing Error (on_error='ignore' records per-cell errors and never raises)
            try:
                if query is not None:
                    status = self._tbl_version.get().insert(
                        source=None,
                        query=query,
                        print_stats=print_stats,
                        fail_on_exception=fail_on_exception,
                        skip_existing_keys=skip_existing_keys,
                    )
                else:
                    status = self._tbl_version.get().insert(
                        source=data_source.valid_rows(),
                        query=None,
                        print_stats=print_stats,
                        fail_on_exception=fail_on_exception,
                        return_rows=return_rows,
                        skip_existing_keys=skip_existing_keys,
                    )
            except excs.ExprEvalError as e:
                excs.raise_from_expr_eval_err(e)

        Env.get().console_logger.info(status.insert_msg(start_ts))
        FileCache.get().emit_eviction_warnings()
        return status

    def _validate_key_matching(self) -> None:
        """Raise an error if insert(if_exists='ignore') cannot match this table's rows by primary key."""
        pk_cols = self._tbl_version.get().primary_key_columns()
        if len(pk_cols) == 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f"insert(): if_exists='ignore' matches rows by primary key, and {self._display_str()} does not "
                'have one',
            )
        for col in pk_cols:
            if col.col_type.is_json_type() or col.col_type.is_array_type():
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f"insert(): if_exists='ignore' does not support the {col.col_type} primary key column {col.name!r}",
                )

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
        with get_runtime().catalog.begin_data_write_xact(tvps=[self._tbl_version_path]):
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

    @telemetry.spanned('pixeltable.delete', set_current=True)
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
        telemetry.add_attrs(telemetry.func_span(), **telemetry_schemas.OpAttrs(table_id=str(self._id)))
        with get_runtime().catalog.begin_data_write_xact(tvps=[self._tbl_version_path]):
            tv = self._tbl_version.get()
            status = tv.delete(where=where)
            telemetry.add_attrs(telemetry.func_span(), **telemetry_schemas.OpAttrs(table=tv.name, version=tv.version))
            telemetry.add_attrs(telemetry.func_span(), **telemetry_schemas.op_attrs_from_update_status(status))
            return status

    def _get_base_table(self) -> 'Table' | None:
        return None

    @property
    def _effective_base_versions(self) -> list[int | None]:
        return []

    def _table_descriptor(self, path: 'pxt.catalog.Path | None' = None) -> str:
        return self._display_str(path)
