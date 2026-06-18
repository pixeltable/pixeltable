from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pixeltable import exceptions as excs

from .table_proxy import TableProxy

if TYPE_CHECKING:
    from pixeltable import exprs, type_system as ts

    from ..globals import TableDataSource
    from .table_path import TableMdPath
    from .update_status import UpdateStatus


class ViewProxy(TableProxy):
    """A proxy for a hosted view handle."""

    def _display_name(self) -> str:
        return 'snapshot' if self._tbl_md_path.is_snapshot() else 'view'

    @property
    def _query_path(self) -> TableMdPath:
        # a pure snapshot has no physical table of its own; queries run against its base at the pinned
        # snapshot version (mirrors a local View(snapshot_only) whose _tbl_version_path is the base path)
        md_path = self._tbl_md_path
        if md_path.is_pure_snapshot():
            assert md_path.base is not None
            return md_path.base
        return md_path

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
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, f'{self._display_str()}: Cannot insert into a {self._display_name()}.'
        )

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, f'{self._display_str()}: Cannot delete from a {self._display_name()}.'
        )
