from __future__ import annotations

import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Iterator, Literal, Sequence, cast
from uuid import UUID

import pyarrow.parquet as pq
import pydantic

from pixeltable import exceptions as excs
from pixeltable.utils import arrow

from .globals import is_hf_dataset
from .table_path import TableMdPath
from .table_proxy import TableProxy
from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable import exprs, type_system as ts
    from pixeltable._query import Query
    from pixeltable.io.data_sources import SqlDataSource
    from pixeltable.service.proxy_client import ProxyClient

    from ..globals import TableDataSource
    from .table import Table


# byte budget per arrow batch when materializing a SQL source to parquet
_SQL_PARQUET_BATCH_BYTES = 256 * 2**20


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
            PandasTableDataConduit,
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

        if isinstance(source, str):
            # A file/directory/URL source is sent to the daemon as-is. The exception is a source with media columns,
            # whose (possibly local) files must be sent to the daemon, so we read the source into rows here and route it
            # through the media-upload path.
            if len(self._media_column_names()) > 0:
                # parse the source into rows client-side (to upload its media files), honoring the same
                # schema_overrides and reader options (extra_fields) the local/non-media paths receive.
                # Normalize the overrides to ColumnType (schema inference requires instances, not type markers).
                data_source = TableDataConduit.create(
                    source,
                    source_format=source_format,
                    src_schema_overrides=self._normalize_schema_overrides(schema_overrides),
                    extra_fields=kwargs,
                )
                data_source.src_pk = []
                data_source.infer_schema()
                rows = self._convert_local_paths([row for batch in data_source.valid_row_batch() for row in batch])
                return self._dispatch(
                    'insert',
                    {'rows': rows, 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows},
                )
            return self._insert_source_file(
                source,
                source_format=source_format,
                schema_overrides=schema_overrides,
                extra_fields=kwargs,
                on_error=on_error,
                print_stats=print_stats,
                return_rows=return_rows,
            )

        # a HuggingFace dataset is sent to the daemon in its own on-disk form and reconstructed there, rather
        # than materialized into rows on the client
        if is_hf_dataset(source):
            return self._insert_hf_dataset(
                source,
                schema_overrides=schema_overrides,
                on_error=on_error,
                print_stats=print_stats,
                return_rows=return_rows,
                extra_fields=kwargs,
            )

        # source classification/error conditions are shared with the local insert path
        data_source = TableDataConduit.create(
            source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
        )

        if isinstance(data_source, QueryTableDataConduit):
            return self._insert_query(
                data_source.pxt_query, on_error=on_error, print_stats=print_stats, return_rows=return_rows
            )

        # in-memory data sources are sent as row dicts
        if isinstance(data_source, PydanticTableDataConduit):
            rows = self._pydantic_to_rows(cast('list[Any]', source))
        elif isinstance(data_source, RowDataTableDataConduit):
            assert data_source.raw_rows is not None
            rows = data_source.raw_rows
        elif isinstance(data_source, PandasTableDataConduit):
            data_source.src_pk = []
            data_source.infer_schema()  # populates valid_rows (df -> pxt-native dicts)
            assert data_source.valid_rows is not None
            rows = data_source.valid_rows
        else:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Hosted insert does not support a {type(source).__name__} source yet.',
            )
        rows = self._convert_local_paths(rows)
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
        rows = self._convert_local_paths(self._prepare_rows(list(source)))
        return self._dispatch('compute', {'rows': rows, 'on_error': on_error})

    def _insert_query(
        self, query: 'Query', *, on_error: Literal['abort', 'ignore'], print_stats: bool, return_rows: bool
    ) -> UpdateStatus:
        bound_args = self._dispatch_args(locals())
        if query._from_clause.catalog_uri != self._catalog_uri:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'Inserting from a query against a different database not supported.',
            )
        bound_args['query'] = query.as_dict()
        return self._dispatch('insert_query', bound_args)

    def _insert_sql_source(
        self,
        sql_source: 'SqlDataSource',
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        send_connect_url: bool = False,
    ) -> 'UpdateStatus':
        """Import a SQL source."""
        if send_connect_url:
            return self._insert_sql(sql_source, on_error=on_error, print_stats=print_stats, return_rows=return_rows)

        schema = self._get_schema()
        src_schema = {name: schema[name] for name in sql_source.col_names}

        tmp_dir = tempfile.mkdtemp()
        try:
            parquet_path = os.path.join(tmp_dir, 'sql_source.parquet')
            writer: pq.ParquetWriter | None = None
            try:
                for batch in arrow.record_batches_from_rows(
                    src_schema, self._sql_source_rows(sql_source, src_schema), _SQL_PARQUET_BATCH_BYTES
                ):
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, batch.schema)
                    writer.write_batch(batch)
            finally:
                # ParquetWriter writes the file footer on close(), so the parquet is incomplete until then; close
                # here so the file is complete before it is read below, and so an error frees the handle for cleanup
                if writer is not None:
                    writer.close()
            if writer is None:
                # the source produced no rows; the destination table is already created, so nothing to send
                return UpdateStatus()
            return self._insert_source_file(
                parquet_path,
                source_format='parquet',
                schema_overrides=None,
                extra_fields=None,
                on_error=on_error,
                print_stats=print_stats,
                return_rows=return_rows,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _sql_source_rows(
        self, sql_source: 'SqlDataSource', schema: dict[str, 'ts.ColumnType']
    ) -> Iterator[dict[str, Any]]:
        """
        Stream the SQL query result as row dicts keyed by output column name, enforcing non-nullable columns the way
        SqlDataNode does (the store accepts NULLs, so the check belongs here).
        """
        result = sql_source.conn.execute(  # type: ignore[call-overload]
            sql_source.select_stmt, execution_options={'stream_results': True}
        )
        for sa_row in result:
            row: dict[str, Any] = {}
            for name, val in zip(sql_source.col_names, sa_row, strict=True):
                if val is None and not schema[name].nullable:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, f'Error in column {name}: expected non-None value'
                    )
                row[name] = val
            yield row

    def _insert_sql(
        self, sql_source: 'SqlDataSource', *, on_error: Literal['abort', 'ignore'], print_stats: bool, return_rows: bool
    ) -> 'UpdateStatus':
        """Send the source connection URL (incl credentials) + compiled SQL so the daemon can run the query itself."""
        engine = sql_source.conn.engine
        sql_text = str(sql_source.select_stmt.compile(dialect=engine.dialect, compile_kwargs={'literal_binds': True}))
        return self._dispatch(
            'insert_sql_source',
            {
                'connect_url': engine.url.render_as_string(hide_password=False),
                'sql_text': sql_text,
                'col_names': list(sql_source.col_names),
                'on_error': on_error,
                'print_stats': print_stats,
                'return_rows': return_rows,
            },
        )

    def _local_path(self, val: str) -> str | None:
        """The local filesystem path for a bare path or file:// URL; None for a remote (http/s3/...) URL.

        Mirrors DataRow.__setitem__()'s local-vs-remote classification.
        """
        parsed = urllib.parse.urlparse(val)
        if len(parsed.scheme) <= 1:
            return val  # bare local path (scheme <= 1 also covers Windows drive letters)
        if parsed.scheme == 'file':
            return urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        return None  # remote URL

    def _media_column_names(self) -> set[str]:
        return {
            col_md.name
            for col_md in self._tbl_md_path.column_md()
            if col_md.name is not None and col_md.col_type.is_media_type()
        }

    def _insert_source_file(
        self,
        source: str,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None,
        schema_overrides: dict[str, ts.ColumnType] | None,
        extra_fields: dict[str, Any] | None,
        on_error: Literal['abort', 'ignore'],
        print_stats: bool,
        return_rows: bool,
    ) -> UpdateStatus:
        """Send a media-free file/directory/URL source to the daemon."""
        from pixeltable.service.proxy_protocol import LocalFile

        local = self._local_path(source)
        wire_source: Any
        source_dir_name: str | None = None
        if local is None:
            wire_source = source  # remote URL: the daemon reads it directly
        elif os.path.isdir(local):
            # send every file in the directory; the daemon reassembles them into a directory of the same name
            wire_source = [LocalFile(os.path.join(local, name)) for name in sorted(os.listdir(local))]
            source_dir_name = os.path.basename(os.path.normpath(local))
        else:
            wire_source = LocalFile(local)
        return self._dispatch(
            'insert_source',
            {
                'source': wire_source,
                'source_dir_name': source_dir_name,
                'source_format': source_format,
                'schema_overrides': self._normalize_schema_overrides(schema_overrides),
                'extra_fields': extra_fields,
                'on_error': on_error,
                'print_stats': print_stats,
                'return_rows': return_rows,
            },
        )

    def _insert_hf_dataset(
        self,
        source: 'TableDataSource',
        *,
        schema_overrides: dict[str, ts.ColumnType] | None,
        on_error: Literal['abort', 'ignore'],
        print_stats: bool,
        return_rows: bool,
        extra_fields: dict[str, Any] | None,
    ) -> UpdateStatus:
        """
        Send a HuggingFace dataset to the daemon via its on-disk serialization.
        - unfortunately HF Datasets are not reliably self-identifying, so we can't just send the Dataset's identity
          and re-instantiate it on the daemon
        - datasets.save_to_disk() writes a self-contained Arrow copy (media bytes embedded, full feature metadata)
        - this avoids materialization of the dataset in Python
        """
        from pixeltable.service.proxy_protocol import LocalFile

        # a streaming dataset (IterableDataset/IterableDatasetDict) has no on-disk serialization; supporting it
        # would mean reading the entire stream into client memory, which is exactly what sending the on-disk
        # form avoids
        dataset = cast(Any, source)
        if not hasattr(dataset, 'save_to_disk'):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'Importing a streaming HuggingFace dataset into a hosted table is not supported yet; '
                'load the dataset without streaming=True.',
            )

        tmp_dir = tempfile.mkdtemp()
        try:
            dataset.save_to_disk(tmp_dir)
            files: list[dict[str, Any]] = []
            for dir_path, _, names in os.walk(tmp_dir):
                for name in sorted(names):
                    abs_path = os.path.join(dir_path, name)
                    rel_path = os.path.relpath(abs_path, tmp_dir).replace(os.sep, '/')
                    files.append({'relpath': rel_path, 'upload': LocalFile(abs_path)})
            return self._dispatch(
                'insert_hf_dataset',
                {
                    'files': files,
                    'schema_overrides': self._normalize_schema_overrides(schema_overrides),
                    'extra_fields': extra_fields or {},
                    'on_error': on_error,
                    'print_stats': print_stats,
                    'return_rows': return_rows,
                },
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _normalize_schema_overrides(
        self, schema_overrides: dict[str, ts.ColumnType] | None
    ) -> dict[str, ts.ColumnType] | None:
        """Normalize override values to ColumnType so they serialize, matching globals.create_table()."""
        from pixeltable import type_system as ts

        if schema_overrides is None:
            return None
        return {
            name: t if isinstance(t, ts.ColumnType) else ts.ColumnType.normalize_type(t, nullable_default=True)
            for name, t in schema_overrides.items()
        }

    def _convert_local_paths(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert local file path values of media columns to LocalFile for correct request serialization."""
        from pixeltable.service.proxy_protocol import LocalFile

        media_cols = self._media_column_names()
        if len(media_cols) == 0:
            return rows

        converted: list[dict[str, Any]] = []
        for row in rows:
            new_row = dict(row)
            for name in media_cols & new_row.keys():
                val = new_row[name]
                if isinstance(val, str):
                    p = self._local_path(val)
                    if p is not None:
                        new_row[name] = LocalFile(p)
            converted.append(new_row)
        return converted

    def _prepare_rows(self, source: list[Any]) -> list[dict[str, Any]]:
        """
        Validate and normalize a non-empty list of dict/pydantic source rows for the hosted catalog:
        - pydantic models are validated and converted to dicts on the client (the model classes aren't
          importable on the server)
        - plain dicts are sent as-is
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
