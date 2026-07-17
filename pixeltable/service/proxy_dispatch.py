"""Server-side dispatch for the proxy service.

Routes a request on (class, method) to a handler that runs the operation against the local catalog and
returns a serializable result; errors are returned as excs.Error.to_dict().
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import shutil
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, cast
from uuid import UUID, uuid4

import sqlalchemy as sql

from pixeltable import exceptions as excs
from pixeltable._query import Query
from pixeltable.catalog import InsertableTable, Path, TablePathKey, retry_loop
from pixeltable.catalog.table_version import TableVersionKey
from pixeltable.env import Env
from pixeltable.io.data_sources import SqlDataSource
from pixeltable.runtime import get_runtime
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore

from . import proxy_protocol
from .proxy_protocol import PROTOCOL_VERSION, LocalFile, MediaPath, ProxyRequest, ProxyResponse

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pixeltable.catalog import LocalTable
    from pixeltable.catalog.globals import TableVersionMd
    from pixeltable.catalog.update_status import UpdateStatus


def handle(request_json: str, request_parts: list[bytes]) -> tuple[str, list[bytes]]:
    """Entry point for an incoming proxy request; always returns a ProxyResponse as (JSON head, binary parts)."""
    request = ProxyRequest.model_validate_json(request_json)
    request._binary_parts = request_parts
    path_label = request.path_key.get('tbl_key', request.path_key) if request.path_key else ''
    _logger.debug('%s.%s %s', request.class_name, request.method, path_label)
    t0 = time.monotonic()
    try:
        if request.protocol_version != PROTOCOL_VERSION:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Unsupported proxy protocol version: {request.protocol_version} (server expects {PROTOCOL_VERSION})',
            )
        key = (request.class_name, request.method)
        table_handler = _TABLE_HANDLERS.get(key)
        if table_handler is not None:
            # Table method
            assert request.path_key is not None and request.snapshot_path_key is not None
            cat = get_runtime().catalog
            tbl = _resolve_tbl(TablePathKey.from_dict(request.path_key))
            is_mutation = request.method in _MUTATION_METHODS
            if is_mutation:
                # refuse to run if the client's snapshot_path_key is behind the current schema
                snapshot_key = TablePathKey.from_dict(request.snapshot_path_key)
                with cat.begin_xact(for_write=False):
                    md = cat.read_md_for_export(tbl)
                if snapshot_key != _current_key(md):
                    # return the current md and is_stale_md=True so the client refreshes and retries
                    return _encode_response(ProxyResponse(current_md=md, is_stale_md=True))

            result = _convert_result(key, table_handler(request, tbl))
            if not is_mutation:
                # a read leaves the schema unchanged, so the client's md stays valid; no need to send it back
                return _encode_response(ProxyResponse(result=result))

            # a mutation bumps the table version; return the new md so the client's path refreshes
            with cat.begin_xact(for_write=False):
                md = cat.read_md_for_export(tbl)
            _logger.debug('%s.%s %s (%.2fs)', request.class_name, request.method, path_label, time.monotonic() - t0)
            return _encode_response(ProxyResponse(result=result, current_md=md))

        handler = _HANDLERS.get(key)
        if handler is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Unsupported proxy method: {request.class_name}.{request.method}'
            )
        result = _convert_result(key, handler(request))
        _logger.debug('%s.%s (%.2fs)', request.class_name, request.method, time.monotonic() - t0)
        return _encode_response(ProxyResponse(result=result))

    except excs.Error as e:
        if e.detail is not None:
            # the client only gets the message; keep the diagnostic detail (e.g. an evaluation stack trace)
            # for whoever reads the server logs
            _logger.info('Error detail handling %s.%s:\n%s', request.class_name, request.method, e.detail)
        _logger.info('%s.%s error (%.2fs)', request.class_name, request.method, time.monotonic() - t0)
        error_dict = e.to_dict()
        error_dict['message'] = _restore_upload_names(error_dict['message'], request._uploaded_names)
        if 'cause' in error_dict:
            error_dict['cause'] = _restore_upload_names(error_dict['cause'], request._uploaded_names)
        return _encode_response(ProxyResponse(error=error_dict))

    except Exception:
        # An unexpected server-side failure. Log the full traceback for debugging, but return only a short
        # reference id to the client: server internals (stack frames, filesystem paths) must not cross the wire.
        ref = uuid4().hex
        tb = traceback.format_exc()
        _logger.error(
            'Internal error (ref %s) handling %s.%s (%.2fs):\n%s',
            ref,
            request.class_name,
            request.method,
            time.monotonic() - t0,
            tb,
        )
        err = excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'Internal proxy error (ref: {ref})')
        error_dict = err.to_dict()
        if os.environ.get('PXTTEST_IN_CI'):
            error_dict['detail'] = tb
        return _encode_response(ProxyResponse(error=error_dict))

    finally:
        # best-effort removal of this request's uploaded temp files; missing_ok covers a file that a handler moved
        # into a reassembled directory (those directories are removed by the handler that creates them)
        for temp_path in request._uploaded_names:
            try:
                pathlib.Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _encode_local_path(value: Any) -> Any:
    """Encode local file paths as LocalFile/MediaPath."""
    if not isinstance(value, str):
        return value
    path = parse_local_file_path(value)
    if path is None:
        return value  # remote URL: the client fetches it directly
    if TempStore.contains_path(path):
        return LocalFile(str(path))
    media_dir = Env.get().media_dir.resolve()
    resolved = path.resolve()
    if resolved == media_dir or media_dir in resolved.parents:
        return MediaPath(resolved.relative_to(media_dir).as_posix())
    cache_dir = Env.get().file_cache_dir.resolve()
    if resolved == cache_dir or cache_dir in resolved.parents:
        # a file-cache copy of remote media (e.g. from .localpath): send its bytes, since the daemon's local
        # path can't be resolved by the client
        # TODO: send the url and have the client fetch it directly?
        return LocalFile(str(path))
    return value


def _convert_result(key: tuple[str, str], result: Any) -> Any:
    """Apply this method's registered output converter (if any) to prepare its result for the wire."""
    converter = _RESULT_CONVERTERS.get(key)
    return result if converter is None else converter(result)


def _encode_response(response: ProxyResponse) -> tuple[str, list[bytes]]:
    """Encode a response as (JSON head, binary parts), moving any binary values in it out to the parts list."""
    proxy_protocol.serialize_response(response)
    return response.model_dump_json(), response._binary_parts


def _restore_upload_names(text: str, uploaded_names: dict[str, str]) -> str:
    """Replace any uploaded-file temp path in text with the client's original filename."""
    for temp_path, original_name in uploaded_names.items():
        text = text.replace(temp_path, original_name)
    return text


def _deserialize_args(request: ProxyRequest) -> dict:
    """Deserialize the request's args inside a read transaction.

    Rebuilding Expr/Query/Function values from the wire loads table-version metadata, which needs a retry loop.
    Handlers that carry such values must deserialize through here rather than calling
    proxy_protocol.deserialize_request() directly.
    """

    @retry_loop(for_write=False)
    def deserialize() -> dict:
        return proxy_protocol.deserialize_request(request)

    return deserialize()


def _create_table(request: ProxyRequest) -> tuple[list, bool]:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    tbl, was_created = cat.create_table(**kwargs)
    with cat.begin_xact(for_write=False):
        md = cat.read_md_for_export(tbl)
    return md, was_created


def _create_view(request: ProxyRequest) -> tuple[list, bool]:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    tbl, was_created = cat.create_view(**kwargs)
    with cat.begin_xact(for_write=False):
        md = cat.read_md_for_export(tbl)
    return md, was_created


def _create_from_model(request: ProxyRequest) -> tuple[list, bool]:
    kwargs = _deserialize_args(request)
    base_dict = kwargs.pop('base')

    @retry_loop(for_write=False)
    def build_base() -> Any:
        return None if base_dict is None else Query.from_dict(base_dict)

    cat = get_runtime().catalog
    tbl, was_created = cat.create_from_model(base=build_base(), **kwargs)
    with cat.begin_xact(for_write=False):
        md = cat.read_md_for_export(tbl)
    return md, was_created


def _get_table(request: ProxyRequest) -> list | None:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    tbl = cat.get_table(**kwargs)
    if tbl is None:
        return None
    with cat.begin_xact(for_write=False):
        return cat.read_md_for_export(tbl)


def _get_table_by_id(request: ProxyRequest) -> list | None:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    with cat.begin_xact(for_write=False):  # get_table_by_id must run inside a transaction
        tbl = cat.get_table_by_id(**kwargs)
        return None if tbl is None else cat.read_md_for_export(tbl)


def _catalog_method(request: ProxyRequest) -> Any:
    """Generic handler for catalog methods whose return value is directly serializable."""
    method = getattr(get_runtime().catalog, request.method)
    return method(**proxy_protocol.deserialize_request(request))


def _resolve_tbl(path_key: TablePathKey) -> LocalTable:
    tbl_id, effective_version = path_key.keys[0].tbl_id, path_key.keys[0].effective_version
    cat = get_runtime().catalog
    with cat.begin_xact(for_write=False):
        tbl = cat.get_table_by_id(tbl_id, effective_version)
    if tbl is None:
        raise excs.table_was_dropped(tbl_id)
    return tbl


def _current_key(md: list[TableVersionMd]) -> TablePathKey:
    return TablePathKey(tuple(TableVersionKey(UUID(m.tbl_md.tbl_id), m.version_md.version) for m in md))


def _get_metadata(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl.get_metadata()


def _list_views(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl.list_views(recursive=proxy_protocol.deserialize_request(request)['recursive'])


def _get_versions(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl.get_versions(proxy_protocol.deserialize_request(request)['n'])


def _insert(request: ProxyRequest, tbl: LocalTable) -> Any:
    # only an InsertableTableProxy dispatches 'insert', so a non-InsertableTable here is an internal error
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    return tbl.insert(
        kwargs['rows'],
        on_error=kwargs['on_error'],
        print_stats=kwargs['print_stats'],
        return_rows=kwargs['return_rows'],
    )


def _insert_source(request: ProxyRequest, tbl: LocalTable) -> Any:
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    # 'source' is a local temp path (a sent file), a remote URL string, or a directory tree (a sent directory).
    # tbl.insert() reads a local path or a URL directly; a directory tree is reassembled here.
    source = kwargs['source']
    root: pathlib.Path | None = None
    if isinstance(source, list):
        root = TempStore.create_path()
        source = str(proxy_protocol.decode_dir_tree(source, root))
    try:
        return tbl.insert(
            source,
            source_format=kwargs['source_format'],
            schema_overrides=kwargs['schema_overrides'],
            on_error=kwargs['on_error'],
            print_stats=kwargs['print_stats'],
            return_rows=kwargs['return_rows'],
            **(kwargs['extra_fields'] or {}),
        )
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)


def _insert_hf_dataset(request: ProxyRequest, tbl: LocalTable) -> Any:
    import datasets  # type: ignore[import-untyped]

    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    # reassemble the sent save_to_disk() directory tree and load it back
    root = TempStore.create_path()
    try:
        dataset_dir = proxy_protocol.decode_dir_tree(kwargs['files'], root)
        dataset = datasets.load_from_disk(str(dataset_dir))
        return tbl.insert(
            dataset,
            schema_overrides=kwargs['schema_overrides'],
            on_error=kwargs['on_error'],
            print_stats=kwargs['print_stats'],
            return_rows=kwargs['return_rows'],
            **(kwargs['extra_fields'] or {}),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _insert_sql_source(request: ProxyRequest, tbl: LocalTable) -> Any:
    # only an InsertableTableProxy with send_connect_url=True dispatches this, so a non-InsertableTable is an
    # internal error
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    # the daemon connects to the source database itself and streams it in through the normal SqlDataNode path
    engine = sql.create_engine(kwargs['connect_url'])
    try:
        with engine.connect() as conn:
            # text(...) executes like a SELECT here; SqlDataNode reads its rows positionally by col_names
            sql_source = SqlDataSource(
                select_stmt=cast(Any, sql.text(kwargs['sql_text'])), col_names=kwargs['col_names'], conn=conn
            )
            return tbl._insert_sql_source(
                sql_source,
                on_error=kwargs['on_error'],
                print_stats=kwargs['print_stats'],
                return_rows=kwargs['return_rows'],
            )
    finally:
        engine.dispose()


def _insert_query(request: ProxyRequest, tbl: LocalTable) -> Any:
    from pixeltable._query import Query

    # only an InsertableTableProxy dispatches 'insert_query', so a non-InsertableTable here is an internal error
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)

    # from_dict() loads the query's table metadata, which is disallowed inside a plain transaction
    @retry_loop(for_write=False)
    def build() -> Query:
        return Query.from_dict(kwargs['query'])

    return tbl.insert(
        build(), on_error=kwargs['on_error'], print_stats=kwargs['print_stats'], return_rows=kwargs['return_rows']
    )


def _compute(request: ProxyRequest, tbl: LocalTable) -> Any:
    # only an InsertableTableProxy dispatches 'compute', so a non-InsertableTable here is an internal error
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    return tbl.compute(kwargs['rows'], on_error=kwargs['on_error'])


def _update(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.update(
        kwargs['value_spec'], where=kwargs['where'], cascade=kwargs['cascade'], return_rows=kwargs['return_rows']
    )


def _batch_update(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.batch_update(
        kwargs['rows'],
        cascade=kwargs['cascade'],
        if_not_exists=kwargs['if_not_exists'],
        return_rows=kwargs['return_rows'],
    )


def _delete(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl.delete(where=_deserialize_args(request)['where'])


def _revert(request: ProxyRequest, tbl: LocalTable) -> None:
    tbl.revert()


def _add_columns(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.add_columns(kwargs['schema'], if_exists=kwargs['if_exists'])


def _add_column(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.add_column(if_exists=kwargs['if_exists'], **kwargs['columns'])


def _add_computed_column(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.add_computed_column(
        stored=kwargs['stored'],
        destination=kwargs['destination'],
        custom_metadata=kwargs['custom_metadata'],
        comment=kwargs['comment'],
        print_stats=kwargs['print_stats'],
        on_error=kwargs['on_error'],
        if_exists=kwargs['if_exists'],
        **kwargs['columns'],
    )


def _drop_column(request: ProxyRequest, tbl: LocalTable) -> None:
    kwargs = _deserialize_args(request)
    tbl.drop_column(kwargs['column'], if_not_exists=kwargs['if_not_exists'])


def _rename_column(request: ProxyRequest, tbl: LocalTable) -> None:
    kwargs = _deserialize_args(request)
    tbl.rename_column(kwargs['old_name'], kwargs['new_name'])


def _add_embedding_index(request: ProxyRequest, tbl: LocalTable) -> None:
    kwargs = _deserialize_args(request)
    tbl.add_embedding_index(
        kwargs['column'],
        idx_name=kwargs['idx_name'],
        embedding=kwargs['embedding'],
        string_embed=kwargs['string_embed'],
        image_embed=kwargs['image_embed'],
        metric=kwargs['metric'],
        precision=kwargs['precision'],
        if_exists=kwargs['if_exists'],
    )


def _drop_embedding_index(request: ProxyRequest, tbl: LocalTable) -> None:
    kwargs = _deserialize_args(request)
    tbl.drop_embedding_index(
        column=kwargs['column'], idx_name=kwargs['idx_name'], if_not_exists=kwargs['if_not_exists']
    )


def _drop_index(request: ProxyRequest, tbl: LocalTable) -> None:
    kwargs = _deserialize_args(request)
    tbl.drop_index(column=kwargs['column'], idx_name=kwargs['idx_name'], if_not_exists=kwargs['if_not_exists'])


def _recompute_columns(request: ProxyRequest, tbl: LocalTable) -> Any:
    kwargs = _deserialize_args(request)
    return tbl.recompute_columns(
        *kwargs['columns'], where=kwargs['where'], errors_only=kwargs['errors_only'], cascade=kwargs['cascade']
    )


def _get_path(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl._path()


def _describe(request: ProxyRequest, tbl: LocalTable) -> Any:
    # rebase the title onto the client's catalog so it shows the table's full pxt:// path
    args = proxy_protocol.deserialize_request(request)
    catalog_uri = Path.parse(args['catalog_uri'], allow_empty_path=True)
    display_path = dataclasses.replace(tbl._path(), org=catalog_uri.org, db=catalog_uri.db)
    helper = tbl._descriptors(path=display_path)
    return {'str': helper.to_string(), 'html': helper.to_html()}


def _run_query(query_dict: dict, run: 'Callable[[Any], Any]') -> dict:
    from pixeltable._query import Query

    # from_dict() loads metadata
    @retry_loop(for_write=False)
    def build() -> Query:
        return Query.from_dict(query_dict)

    rs = run(build())
    return {'schema': dict(rs._schema), 'rows': [list(row._data) for row in rs._rows]}


def _query_collect(request: ProxyRequest) -> dict:
    payload = proxy_protocol.deserialize_request(request)
    return _run_query(payload['query'], lambda q: q._collect(args=payload.get('args'), media_as_urls=True))


def _query_head(request: ProxyRequest) -> dict:
    args = proxy_protocol.deserialize_request(request)
    return _run_query(args['query'], lambda q: q._head(args['n'], media_as_urls=True))


def _query_tail(request: ProxyRequest) -> dict:
    args = proxy_protocol.deserialize_request(request)
    return _run_query(args['query'], lambda q: q._tail(args['n'], media_as_urls=True))


def _query_count(request: ProxyRequest) -> int:
    from pixeltable._query import Query

    query_dict = proxy_protocol.deserialize_request(request)['query']

    @retry_loop(for_write=False)
    def build() -> Query:
        return Query.from_dict(query_dict)

    return build().count()


def _encode_row_media(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converter for row dicts containing references to local files."""
    for row in rows:
        for name in row:
            row[name] = _encode_local_path(row[name])
    return rows


def _encode_update_status(status: UpdateStatus) -> UpdateStatus:
    """Converter for handlers returning an UpdateStatus with rows."""
    if status.rows is not None:
        _encode_row_media(status.rows)  # mutates the row dicts in place (UpdateStatus is frozen)
    return status


def _encode_result_set(result: dict) -> dict:
    """Converter for query terminals returning {schema, rows}."""
    result['rows'] = [[_encode_local_path(v) for v in row] for row in result['rows']]
    return result


# Catalog methods: handler(request) -> serializable result.
_HANDLERS: dict[tuple[str, str], Callable[[ProxyRequest], Any]] = {
    ('CatalogBase', 'create_table'): _create_table,
    ('CatalogBase', 'create_view'): _create_view,
    ('CatalogBase', 'create_from_model'): _create_from_model,
    ('CatalogBase', 'get_table'): _get_table,
    ('CatalogBase', 'get_table_by_id'): _get_table_by_id,
    ('CatalogBase', 'move'): _catalog_method,
    ('CatalogBase', 'drop_table'): _catalog_method,
    ('CatalogBase', 'get_dir_contents'): _catalog_method,
    ('CatalogBase', 'create_dir'): _catalog_method,
    ('CatalogBase', 'drop_dir'): _catalog_method,
    ('Query', 'collect'): _query_collect,
    ('Query', 'head'): _query_head,
    ('Query', 'tail'): _query_tail,
    ('Query', 'count'): _query_count,
}

# Mutations are gated on schema-version staleness (handle() withholds + returns current_md if the client's
# snapshot_path_key is behind); reads run unconditionally.
_MUTATION_METHODS: frozenset[str] = frozenset(
    {
        'insert',
        'insert_source',
        'insert_hf_dataset',
        'insert_sql_source',
        'insert_query',
        'update',
        'delete',
        'batch_update',
        'revert',
        'add_columns',
        'add_column',
        'add_computed_column',
        'drop_column',
        'rename_column',
        'add_embedding_index',
        'drop_embedding_index',
        'drop_index',
        'recompute_columns',
    }
)

# Path-bearing Table methods: handler(request, tbl) -> result; handle() resolves tbl and sends current md back.
_TABLE_HANDLERS: dict[tuple[str, str], Callable[[ProxyRequest, 'LocalTable'], Any]] = {
    ('Table', 'get_metadata'): _get_metadata,
    ('Table', '_path'): _get_path,
    ('Table', 'describe'): _describe,
    ('Table', 'list_views'): _list_views,
    ('Table', 'get_versions'): _get_versions,
    ('Table', 'insert'): _insert,
    ('Table', 'insert_source'): _insert_source,
    ('Table', 'insert_hf_dataset'): _insert_hf_dataset,
    ('Table', 'insert_sql_source'): _insert_sql_source,
    ('Table', 'insert_query'): _insert_query,
    ('Table', 'compute'): _compute,
    ('Table', 'update'): _update,
    ('Table', 'delete'): _delete,
    ('Table', 'batch_update'): _batch_update,
    ('Table', 'revert'): _revert,
    ('Table', 'add_columns'): _add_columns,
    ('Table', 'add_column'): _add_column,
    ('Table', 'add_computed_column'): _add_computed_column,
    ('Table', 'drop_column'): _drop_column,
    ('Table', 'rename_column'): _rename_column,
    ('Table', 'add_embedding_index'): _add_embedding_index,
    ('Table', 'drop_embedding_index'): _drop_embedding_index,
    ('Table', 'drop_index'): _drop_index,
    ('Table', 'recompute_columns'): _recompute_columns,
}

# (class, method) -> output converter, for the methods whose result can carry media. handle() applies the
# converter (if any) to a handler's return value before serialization.
_RESULT_CONVERTERS: dict[tuple[str, str], Callable[[Any], Any]] = {
    ('Query', 'collect'): _encode_result_set,
    ('Query', 'head'): _encode_result_set,
    ('Query', 'tail'): _encode_result_set,
    ('Table', 'insert'): _encode_update_status,
    ('Table', 'insert_source'): _encode_update_status,
    ('Table', 'insert_hf_dataset'): _encode_update_status,
    ('Table', 'insert_sql_source'): _encode_update_status,
    ('Table', 'insert_query'): _encode_update_status,
    ('Table', 'compute'): _encode_row_media,
    ('Table', 'update'): _encode_update_status,
    ('Table', 'batch_update'): _encode_update_status,
}
