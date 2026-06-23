from __future__ import annotations

import collections.abc
from typing import TYPE_CHECKING, Any, Literal, cast
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
        if isinstance(source, collections.abc.Iterator):
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
        return self._dispatch(
            'insert', {'rows': rows, 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows}
        )

    def compute(
        self,
        source: collections.abc.Sequence[dict[str, Any]] | collections.abc.Sequence[pydantic.BaseModel],
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> list[dict[str, Any]]:
        # str/bytes are technically Sequences; reject them explicitly rather than treating them as paths/URLs.
        if isinstance(source, (str, bytes)) or not isinstance(source, collections.abc.Sequence) or len(source) == 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'compute() requires a non-empty sequence of dicts or pydantic models; got {type(source).__name__}',
            )
        if not all(isinstance(row, (dict, pydantic.BaseModel)) for row in source):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'compute() requires a sequence of dicts or pydantic models; got {type(source).__name__}',
            )
        rows = self._prepare_rows(list(source))
        return self._dispatch('compute', {'rows': rows, 'on_error': on_error})

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

    def _prepare_rows(self, source: list[Any]) -> list[dict[str, Any]]:
        """
        Validate and normalize a non-empty list of dict/pydantic source rows for the hosted catalog:
        - pydantic models are validated and converted to dicts on the client (the model classes aren't
          importable on the server)
        - plain dicts are shipped as-is
        - local media paths are shipped unchanged, for now
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
        self._validate_where(where)
        return self._dispatch('delete', {'where': where})
