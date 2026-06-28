from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Literal, Sequence, cast
from uuid import UUID

import pydantic

from pixeltable import exceptions as excs

from .table_path import TableMdPath
from .table_proxy import TableProxy

if TYPE_CHECKING:
    from pixeltable import exprs, type_system as ts
    from pixeltable._query import Query
    from pixeltable.service.proxy_client import ProxyClient

    from ..globals import TableDataSource
    from .table import Table
    from .update_status import UpdateStatus


class InsertableTableProxy(TableProxy):
    """A proxy for a hosted InsertableTable handle."""

    def __init__(self, id: UUID, tbl_md_path: TableMdPath, client: 'ProxyClient'):
        super().__init__(id, None, tbl_md_path, client)

    def _display_name(self) -> str:
        return 'table'

    def _get_base_table(self) -> 'Table' | None:
        return None

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
        from pixeltable.io.table_data_conduit import (
            PydanticTableDataConduit,
            QueryTableDataConduit,
            RowDataTableDataConduit,
            TableDataConduit,
        )

        self._validate_insert_source(source)
        if source is None:
            # the kwargs form (t.insert(col=val, ...)) is a single row
            source = [kwargs]
            kwargs = None
        if isinstance(source, Iterator):
            # materialize a generator/iterator of rows, matching the local insert path
            source = list(source)

        # source classification (and its 'unsupported data source type' error) is shared with the local insert path
        data_source = TableDataConduit.create(
            source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
        )

        # a Table or Query source runs on the server against the same hosted catalog
        if isinstance(data_source, QueryTableDataConduit):
            return self._insert_query(
                data_source.pxt_query, on_error=on_error, print_stats=print_stats, return_rows=return_rows
            )
        # dict/pydantic rows are shipped to the daemon, which validates and inserts them through the same conduit
        if isinstance(data_source, PydanticTableDataConduit):
            rows = self._pydantic_to_rows(cast('list[Any]', source))
        elif isinstance(data_source, RowDataTableDataConduit):
            assert data_source.raw_rows is not None
            rows = data_source.raw_rows
        else:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Hosted insert does not support a {type(source).__name__} source yet.',
            )
        rows = self._wrap_media_uploads(rows)
        return self._dispatch(
            'insert', {'rows': rows, 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows}
        )

    def compute(
        self,
        source: Sequence[dict[str, Any]] | Sequence[pydantic.BaseModel],
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> list[dict[str, Any]]:
        # str/bytes are technically Sequences; reject them explicitly (with a clear message) rather than letting
        # them fall through to the element-type check or be interpreted as paths/URLs
        if isinstance(source, (str, bytes)) or not isinstance(source, Sequence) or len(source) == 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'compute() requires a non-empty sequence of dicts or pydantic models',
            )
        if not all(isinstance(row, (dict, pydantic.BaseModel)) for row in source):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'compute() requires a sequence of dicts or pydantic models'
            )
        rows = self._wrap_media_uploads(self._prepare_rows(list(source)))
        return self._dispatch('compute', {'rows': rows, 'on_error': on_error})

    def _insert_query(
        self, query: 'Query', *, on_error: Literal['abort', 'ignore'], print_stats: bool, return_rows: bool
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        if query._from_clause.catalog_uri != self._catalog_uri:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'Inserting from a query in a different catalog is not supported.'
            )
        bound_args['query'] = query.as_dict()
        return self._dispatch('insert_query', bound_args)

    def _wrap_media_uploads(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap local media-column file paths as MediaFileUpload so they ship to the daemon as binary parts.

        Remote URLs (http/s3/...) and non-path values are left unchanged, matching the local insert path (which
        stores remote URLs as-is and fetches them on access).
        """
        import urllib.parse
        import urllib.request

        from pixeltable.service.proxy_protocol import MediaFileUpload

        media_cols = {
            col_md.name
            for col_md in self._tbl_md_path.column_md()
            if col_md.name is not None and col_md.col_type.is_media_type()
        }
        if len(media_cols) == 0:
            return rows

        def local_path(val: str) -> str | None:
            # mirrors DataRow.__setitem__'s local-vs-remote classification
            parsed = urllib.parse.urlparse(val)
            if len(parsed.scheme) <= 1:
                return val  # bare local path (scheme <= 1 also covers Windows drive letters)
            if parsed.scheme == 'file':
                return urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
            return None  # remote URL

        wrapped: list[dict[str, Any]] = []
        for row in rows:
            new_row = dict(row)
            for name in media_cols & new_row.keys():
                val = new_row[name]
                if isinstance(val, str):
                    p = local_path(val)
                    if p is not None:
                        new_row[name] = MediaFileUpload(p)
            wrapped.append(new_row)
        return wrapped

    def _prepare_rows(self, source: list[Any]) -> list[dict[str, Any]]:
        """
        Validate and normalize a non-empty list of dict/pydantic source rows for the hosted catalog:
        - pydantic models are validated and converted to dicts on the client (the model classes aren't
          importable on the server)
        - plain dicts are shipped as-is

        Local media-column paths are wrapped for upload separately, by the caller, via `_wrap_media_uploads()`.
        """
        if isinstance(source[0], pydantic.BaseModel):
            source = self._pydantic_to_rows(source)
        rows: list[dict[str, Any]] = []
        for source_row in source:
            if not isinstance(source_row, dict):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, 'Hosted table rows must be dicts or pydantic models.'
                )
            rows.append(source_row)
        return rows

    def _pydantic_to_rows(self, models: list[Any]) -> list[dict[str, Any]]:
        """Validate pydantic models against this table's schema and convert them to insertable dicts.

        Mirrors the local insert path's pydantic handling.
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
        bound_args = self._dispatch_args(locals())
        self._validate_where(where)
        return self._dispatch('delete', bound_args)
