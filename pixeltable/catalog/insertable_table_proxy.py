from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pydantic

from pixeltable import exceptions as excs
from pixeltable.utils import parse_local_file_path

from .table_proxy import TableProxy

if TYPE_CHECKING:
    from pixeltable import exprs, type_system as ts
    from pixeltable._query import Query

    from ..globals import TableDataSource
    from .update_status import UpdateStatus


class InsertableTableProxy(TableProxy):
    """A proxy for a hosted InsertableTable handle."""

    def _display_name(self) -> str:
        return 'table'

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
        if source_format is not None or schema_overrides is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'Hosted insert does not support `source_format` or `schema_overrides` yet.',
            )
        if source is None:
            # the kwargs form (t.insert(col=val, ...)) is a single row
            source = [kwargs]

        # a Query source runs on the server (same hosted catalog) and its result rows are inserted there
        from pixeltable._query import Query

        if isinstance(source, Query):
            return self._insert_query(source, on_error=on_error, print_stats=print_stats, return_rows=return_rows)

        if not isinstance(source, list) or len(source) == 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'Hosted insert supports only a non-empty list of dicts or pydantic models.',
            )

        # pydantic models are validated and converted to dicts on the client (the model classes aren't
        # importable on the server); plain dicts are shipped as-is
        if isinstance(source[0], pydantic.BaseModel):
            source = self._pydantic_to_rows(source)

        rows: list[dict[str, Any]] = []
        media_cols = {cvmd.name for cvmd in self._tbl_md_path.column_md() if cvmd.col_type.is_media_type()}
        for source_row in source:
            if not isinstance(source_row, dict):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, 'Hosted insert rows must be dicts or pydantic models.'
                )
            # for now: reject local media
            for col in media_cols:
                value = source_row.get(col)
                # a media value may only cross the wire as a URL; local files require binary upload (TODO)
                if value is not None and (not isinstance(value, str) or parse_local_file_path(value) is not None):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Inserting local media into a hosted table is not supported yet (column {col!r}).',
                    )
            rows.append(source_row)

        return self._dispatch(
            'insert', {'rows': rows, 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows}
        )

    def _insert_query(
        self, query: 'Query', *, on_error: Literal['abort', 'ignore'], print_stats: bool, return_rows: bool
    ) -> UpdateStatus:
        if query._from_clause.catalog_uri != self._catalog_uri:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'Inserting from a query in a different catalog is not supported.'
            )
        return self._dispatch(
            'insert_query',
            {'query': query.as_dict(), 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows},
        )

    def _pydantic_to_rows(self, models: list[Any]) -> list[dict[str, Any]]:
        """Validate pydantic models against this table's schema and convert them to insertable dicts.

        Mirrors the local insert path's pydantic handling, run on the client so the same validation and
        error messages apply.
        """
        from pixeltable.io.table_data_conduit import PydanticTableDataConduit

        converter = PydanticTableDataConduit(models)
        converter.tbl_name = self._name()
        schema: dict[str, ts.ColumnType] = {}
        for col_md in self._tbl_md_path.column_md():
            if col_md.name is None:
                continue
            schema[col_md.name] = col_md.col_type
            if col_md.is_computed:
                converter.computed_col_names.add(col_md.name)
            elif not col_md.col_type.nullable:
                converter.reqd_col_names.add(col_md.name)
        converter.pxt_schema = schema
        converter.prepare_for_insert_into_table()
        return converter.pxt_rows

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        return self._dispatch('delete', {'where': where})
