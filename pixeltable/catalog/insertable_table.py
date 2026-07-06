from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence, overload
from uuid import UUID

import pydantic

import pixeltable as pxt
from pixeltable import exceptions as excs, hooks, type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.filecache import FileCache

from .column import Column
from .globals import MediaValidation
from .local_table import LocalTable
from .table_path import TableVersionPath
from .table_version import TableVersion, TableVersionMd
from .table_version_handle import TableVersionHandle
from .tbl_ops import CreateStoreTableOp, CreateTableMdOp, TableOp, TableOpsBuilder
from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.globals import TableDataSource
    from pixeltable.io.data_sources import SqlDataSource
    from pixeltable.io.table_data_conduit import TableDataConduit

    from .table import Table


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
        name: str,
        columns: list[Column],
        primary_key: list[str],
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        create_default_idxs: bool,
        is_versioned: bool,
        tbl_id: UUID | None = None,
    ) -> tuple[TableVersionMd, list[TableOp]]:
        cls._verify_schema(columns)
        column_names = [col.name for col in columns]
        for pk_col in primary_key:
            if pk_col not in column_names:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND, f'Primary key column {pk_col!r} not found in table schema.'
                )
            col = columns[column_names.index(pk_col)]
            if col.col_type.nullable:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Primary key column {pk_col!r} cannot be nullable. '
                    f'Declare it as `Required` instead: `pxt.Required[pxt.{col.col_type._to_base_str()}]`',
                )
            col.is_pk = True

        md = TableVersion.create_initial_md(
            name,
            columns,
            comment,
            custom_metadata,
            media_validation,
            create_default_idxs=create_default_idxs,
            view_md=None,
            is_versioned=is_versioned,
            tbl_id=tbl_id,
        )

        ops = (
            TableOpsBuilder(md.tbl_md.tbl_id, tbl_version=md.tbl_md.current_version)
            .add(CreateTableMdOp)
            .add(CreateStoreTableOp)
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
        fail_on_exception = OnErrorParameter.fail_on_exception(on_error)

        if source is None:
            source = [kwargs]
            kwargs = None

        with hooks.span('pixeltable.insert', set_current=True):
            with hooks.span('pixeltable.data_source.prepare'):
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

    def compute(
        self,
        source: Sequence[dict[str, Any]] | Sequence[pydantic.BaseModel],
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> list[dict[str, Any]]:
        from pixeltable.io.table_data_conduit import PydanticTableDataConduit, RowDataTableDataConduit, TableDataConduit

        # str/bytes are technically Sequences; reject them explicitly so we don't fall through to
        # TableDataConduit.create() which would treat a string as a path/URL and trigger file I/O.
        if isinstance(source, (str, bytes)) or not isinstance(source, Sequence) or len(source) == 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'compute() requires a non-empty sequence of dicts or pydantic models; got {type(source).__name__}',
            )
        fail_on_exc = OnErrorParameter.fail_on_exception(on_error)
        # TableDataConduit.is_rowdata_structure() only accepts list (not arbitrary Sequence) for the
        # dict-source dispatch, so normalize to list here.
        data_source = TableDataConduit.create(list(source))
        if not isinstance(data_source, (RowDataTableDataConduit, PydanticTableDataConduit)):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'compute() requires a sequence of dicts or pydantic models; got {type(source).__name__}',
            )
        data_source.add_table_info(self)
        data_source.prepare_for_insert_into_table()

        result: list[dict[str, Any]] = []
        try:
            with get_runtime().catalog.begin_xact(read_tbl_ids=[self._id]):
                for row_batch in data_source.valid_row_batch():
                    result.extend(self._tbl_version.get().compute(row_batch, fail_on_exc=fail_on_exc))
        except excs.ExprEvalError as e:
            excs.raise_from_expr_eval_err(e)

        FileCache.get().emit_eviction_warnings()
        return result

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
    ) -> pxt.UpdateStatus:
        """Stream a SqlDataSource into this table through a single insert plan.

        Assumes the source's columns have already been validated against this table's schema.
        """
        fail_on_exception = OnErrorParameter.fail_on_exception(on_error)
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
