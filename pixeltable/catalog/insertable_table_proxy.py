from __future__ import annotations

import os
import shutil
import tempfile
import urllib.parse
import urllib.request
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

        # a file/directory/URL source (parquet, ...) is read by the daemon from a shipped file rather than
        # materialized into rows on the client -- except when the table has media columns, whose (possibly local)
        # files must be shipped, so we read the source into rows here and route it through the media-upload path
        if isinstance(source, str):
            if self._has_media_columns():
                rows = self._wrap_media_uploads(self._read_source_rows(source, source_format=source_format))
                return self._dispatch(
                    'insert',
                    {'rows': rows, 'on_error': on_error, 'print_stats': print_stats, 'return_rows': return_rows},
                )
            return self._insert_source_file(
                source,
                source_format=source_format,
                schema_overrides=schema_overrides,
                on_error=on_error,
                print_stats=print_stats,
                return_rows=return_rows,
            )

        # a HuggingFace dataset is shipped to the daemon in its own on-disk form and reconstructed there, rather
        # than materialized into rows on the client
        if self._is_hf_dataset(source):
            return self._insert_hf_dataset(
                source,
                schema_overrides=schema_overrides,
                on_error=on_error,
                print_stats=print_stats,
                return_rows=return_rows,
                extra_fields=kwargs,
            )

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

    def _local_path(self, val: str) -> str | None:
        """The local filesystem path for a bare path or file:// URL; None for a remote (http/s3/...) URL.

        Mirrors DataRow.__setitem__'s local-vs-remote classification.
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

    def _has_media_columns(self) -> bool:
        return len(self._media_column_names()) > 0

    def _read_source_rows(
        self, source: str, *, source_format: Literal['csv', 'excel', 'parquet', 'json'] | None
    ) -> list[dict[str, Any]]:
        """Read a file/directory/URL source into rows on the client.

        Used only when the table has media columns: the source's media values may be local file paths, whose
        bytes must be shipped to the daemon (see _wrap_media_uploads). The values are kept in their source form;
        the daemon validates and types them against the table schema.
        """
        from pixeltable.io.table_data_conduit import TableDataConduit

        data_source = TableDataConduit.create(source, source_format=source_format)
        data_source.src_pk = []
        data_source.infer_schema()
        return [row for batch in data_source.valid_row_batch() for row in batch]

    def _insert_source_file(
        self,
        source: str,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None,
        schema_overrides: dict[str, ts.ColumnType] | None,
        on_error: Literal['abort', 'ignore'],
        print_stats: bool,
        return_rows: bool,
    ) -> UpdateStatus:
        """Ship a media-free file/directory/URL source to the daemon, which reads it through its own conduit."""
        from pixeltable.service.proxy_protocol import MediaFileUpload

        local = self._local_path(source)
        wire_source: Any
        source_dir_name: str | None = None
        if local is None:
            wire_source = source  # remote URL: the daemon reads it directly
        elif os.path.isdir(local):
            # ship every file in the directory; the daemon reassembles them into a directory of the same name
            wire_source = [MediaFileUpload(os.path.join(local, name)) for name in sorted(os.listdir(local))]
            source_dir_name = os.path.basename(os.path.normpath(local))
        else:
            wire_source = MediaFileUpload(local)
        return self._dispatch(
            'insert_source',
            {
                'source': wire_source,
                'source_dir_name': source_dir_name,
                'source_format': source_format,
                'schema_overrides': self._normalize_schema_overrides(schema_overrides),
                'on_error': on_error,
                'print_stats': print_stats,
                'return_rows': return_rows,
            },
        )

    def _is_hf_dataset(self, source: 'TableDataSource' | None) -> bool:
        try:
            import datasets  # type: ignore[import-untyped]
        except ImportError:
            return False
        return isinstance(
            source, (datasets.Dataset, datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict)
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
        Ship a HuggingFace dataset to the daemon via its on-disk serialization.
        - unfortunately HF Datasets are not reliably self-identifying, so we can't just ship the Dataset's identity
          and re-instantiate it on the daemon
        - datasets.save_to_disk() writes a self-contained Arrow copy (media bytes embedded, full feature metadata)
        - this avoids materialization of the dataset in Python
        """
        from pixeltable.service.proxy_protocol import MediaFileUpload

        # a streaming dataset (IterableDataset/IterableDatasetDict) has no on-disk serialization; supporting it
        # would mean reading the entire stream into client memory, which is exactly what shipping the on-disk
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
                    files.append({'relpath': rel_path, 'upload': MediaFileUpload(abs_path)})
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

    def _wrap_media_uploads(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap local media-column file paths as MediaFileUpload so they ship to the daemon as binary parts.

        Remote URLs (http/s3/...) and non-path values are left unchanged, matching the local insert path (which
        stores remote URLs as-is and fetches them on access).
        """
        from pixeltable.service.proxy_protocol import MediaFileUpload

        media_cols = self._media_column_names()
        if len(media_cols) == 0:
            return rows

        wrapped: list[dict[str, Any]] = []
        for row in rows:
            new_row = dict(row)
            for name in media_cols & new_row.keys():
                val = new_row[name]
                if isinstance(val, str):
                    p = self._local_path(val)
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
