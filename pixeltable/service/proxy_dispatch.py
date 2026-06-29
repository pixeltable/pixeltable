"""Server-side dispatch for the proxy service.

Routes a request on (class, method) to a handler that runs the operation against the local catalog and
returns a serializable result; errors are returned as excs.Error.to_dict().
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import shutil
import traceback
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid4

from pixeltable import exceptions as excs
from pixeltable.catalog import InsertableTable, Path, TablePathKey, retry_loop
from pixeltable.catalog.table_version import TableVersionKey
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.local_store import TempStore

from . import proxy_protocol
from .proxy_protocol import PROTOCOL_VERSION, MediaFileUpload, MediaUrlRef, ProxyRequest, ProxyResponse

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pixeltable.catalog import LocalTable
    from pixeltable.catalog.globals import TableVersionMd


def handle(request_json: str, request_parts: list[bytes]) -> tuple[str, list[bytes]]:
    """Entry point for an incoming proxy request; always returns a ProxyResponse as (JSON head, binary parts)."""
    request = ProxyRequest.model_validate_json(request_json)
    request._binary_parts = request_parts
    response_parts: list[bytes] = []
    try:
        if request.protocol_version != PROTOCOL_VERSION:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Unsupported proxy protocol version: {request.protocol_version} (server expects {PROTOCOL_VERSION})',
            )
        key = (request.class_name, request.method)
        table_handler = _TABLE_HANDLERS.get(key)
        if table_handler is not None:
            # path-bearing Table method. For a mutation, refuse to run if the client's snapshot_path_key is
            # behind the current schema (return is_stale_md + current_md so it refreshes and retries); reads
            # run unconditionally. Either way, ship the table's current md back so the client's path refreshes.
            assert request.path_key is not None and request.snapshot_path_key is not None
            cat = get_runtime().catalog
            tbl = _resolve_tbl(TablePathKey.from_dict(request.path_key))
            if request.method in _MUTATION_METHODS:
                snapshot_key = TablePathKey.from_dict(request.snapshot_path_key)
                with cat.begin_xact(for_write=False):
                    md = cat.read_md_for_export(tbl)
                if snapshot_key != _current_key(md):
                    resp = ProxyResponse(current_md=proxy_protocol.serialize(md, response_parts), is_stale_md=True)
                    return resp.model_dump_json(), response_parts
            result = _convert_result(key, table_handler(request, tbl))
            with cat.begin_xact(for_write=False):
                md = cat.read_md_for_export(tbl)
            resp = ProxyResponse(
                result=proxy_protocol.serialize(result, response_parts),
                current_md=proxy_protocol.serialize(md, response_parts),
            )
            return resp.model_dump_json(), response_parts
        handler = _HANDLERS.get(key)
        if handler is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Unsupported proxy method: {request.class_name}.{request.method}'
            )
        result = _convert_result(key, handler(request))
        resp = ProxyResponse(result=proxy_protocol.serialize(result, response_parts))
        return resp.model_dump_json(), response_parts
    except excs.Error as e:
        if e.detail is not None:
            # the client only gets the message; keep the diagnostic detail (e.g. an evaluation stack trace)
            # for whoever reads the server logs
            _logger.info('Error detail handling %s.%s:\n%s', request.class_name, request.method, e.detail)
        return ProxyResponse(error=e.to_dict()).model_dump_json(), []
    except Exception:
        # An unexpected server-side failure. Log the full traceback for debugging, but return only a short
        # reference id to the client: server internals (stack frames, filesystem paths) must not cross the wire.
        ref = uuid4().hex
        _logger.error(
            'Internal error (ref %s) handling %s.%s:\n%s',
            ref,
            request.class_name,
            request.method,
            traceback.format_exc(),
        )
        err = excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'Internal proxy error (ref: {ref})')
        return ProxyResponse(error=err.to_dict()).model_dump_json(), []


def _result_local_path(val: str) -> pathlib.Path | None:
    """Resolve a result media value to a local fs path (bare path or file:// URI), or None for a remote URL."""
    parsed = urllib.parse.urlparse(val)
    if len(parsed.scheme) <= 1:
        return pathlib.Path(val)  # bare local path (scheme <= 1 also covers Windows drive letters)
    if parsed.scheme == 'file':
        return pathlib.Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
    return None


def _media_to_wire(value: Any) -> Any:
    """Convert a single media-column result value for the wire.

    Transient media (a file under the daemon TempStore, e.g. a computed-on-the-fly transform) is shipped back as
    bytes via MediaFileUpload, landing in the client's TempStore. Persisted media (under the daemon media dir, e.g.
    a stored column) is returned as a MediaUrlRef the client fetches lazily from the daemon's /media endpoint into
    its FileCache. Remote URLs and non-string values (e.g. an in-memory PIL.Image) pass through unchanged.

    Callers apply this only to values their result schema identifies as media.
    """
    if not isinstance(value, str):
        return value
    path = _result_local_path(value)
    if path is None:
        return value  # remote URL
    if TempStore.contains_path(path):
        return MediaFileUpload(str(path))
    media_dir = Env.get().media_dir.resolve()
    resolved = path.resolve()
    if resolved == media_dir or media_dir in resolved.parents:
        return MediaUrlRef(resolved.relative_to(media_dir).as_posix())
    return value


def _convert_result(key: tuple[str, str], result: Any) -> Any:
    """Apply this method's registered output converter (if any) to prepare its result for the wire."""
    converter = _RESULT_CONVERTERS.get(key)
    return result if converter is None else converter(result)


def _deserialize_args(request: ProxyRequest) -> dict:
    """Deserialize the request's args inside a read transaction.

    Rebuilding Expr/Query/Function values from the wire loads table-version metadata, which needs a retry loop.
    Handlers that carry such values must deserialize through here rather than calling proxy_protocol.deserialize()
    directly.
    """

    @retry_loop(for_write=False)
    def deserialize() -> dict:
        return proxy_protocol.deserialize(request.args, request._binary_parts)

    return deserialize()


def _create_table(request: ProxyRequest) -> tuple[list, bool]:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    tbl, was_created = cat.create_table(**kwargs)
    with cat.begin_xact(for_write=False):
        md = cat.read_md_for_export(tbl)
    return md, was_created


def _create_view(request: ProxyRequest) -> list:
    kwargs = _deserialize_args(request)
    cat = get_runtime().catalog
    tbl = cat.create_view(**kwargs)
    with cat.begin_xact(for_write=False):
        return cat.read_md_for_export(tbl)


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
    return method(**proxy_protocol.deserialize(request.args, request._binary_parts))


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
    return tbl.list_views(recursive=proxy_protocol.deserialize(request.args, request._binary_parts)['recursive'])


def _get_versions(request: ProxyRequest, tbl: LocalTable) -> Any:
    return tbl.get_versions(proxy_protocol.deserialize(request.args, request._binary_parts)['n'])


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
    # only an InsertableTableProxy dispatches 'insert_source', so a non-InsertableTable here is an internal error
    assert isinstance(tbl, InsertableTable), tbl
    kwargs = _deserialize_args(request)
    # 'source' is a local temp path (a shipped file), a remote URL string, or a list of temp paths (a shipped
    # directory). tbl.insert() reads a local path or a URL directly; a directory is reassembled here.
    source = kwargs['source']
    if isinstance(source, list):
        # reassemble the shipped directory, keeping its original name so format detection (e.g. by a *.parquet
        # name) matches the local path
        dest_dir = TempStore.create_path() / kwargs['source_dir_name']
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in source:
            shutil.move(path, dest_dir / pathlib.Path(path).name)
        source = str(dest_dir)
    return tbl.insert(
        source,
        source_format=kwargs['source_format'],
        schema_overrides=kwargs['schema_overrides'],
        on_error=kwargs['on_error'],
        print_stats=kwargs['print_stats'],
        return_rows=kwargs['return_rows'],
    )


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
    args = proxy_protocol.deserialize(request.args, request._binary_parts)
    catalog_uri = Path.parse(args['catalog_uri'], allow_empty_path=True)
    display_path = dataclasses.replace(tbl._path(), org=catalog_uri.org, db=catalog_uri.db)
    helper = tbl._descriptors(path=display_path)
    return {'str': helper.to_string(), 'html': helper.to_html()}


def _run_query_terminal(query_dict: dict, run: 'Callable[[Any], Any]') -> dict:
    from pixeltable._query import Query, emit_media_as_urls  # lazy: _query pulls in plan/exec

    # from_dict() might load metadata
    @retry_loop(for_write=False)
    def build() -> Query:
        return Query.from_dict(query_dict)

    # the schema keeps the original media column types; only the row values become URLs the client fetches
    with emit_media_as_urls():
        rs = run(build())
        return {'schema': dict(rs._schema), 'rows': [list(row._data) for row in rs._rows]}


def _query_collect(request: ProxyRequest) -> dict:
    payload = proxy_protocol.deserialize(request.args, request._binary_parts)
    return _run_query_terminal(payload['query'], lambda q: q._collect(args=payload.get('args')))


def _query_head(request: ProxyRequest) -> dict:
    args = proxy_protocol.deserialize(request.args, request._binary_parts)
    return _run_query_terminal(args['query'], lambda q: q.head(args['n']))


def _query_tail(request: ProxyRequest) -> dict:
    args = proxy_protocol.deserialize(request.args, request._binary_parts)
    return _run_query_terminal(args['query'], lambda q: q.tail(args['n']))


def _query_count(request: ProxyRequest) -> int:
    from pixeltable._query import Query

    query_dict = proxy_protocol.deserialize(request.args, request._binary_parts)['query']

    @retry_loop(for_write=False)
    def build() -> Query:
        return Query.from_dict(query_dict)

    return build().count()


# Output converters: prepare a handler's return value for the wire. A media-bearing result ships its transient
# (computed-on-the-fly) media back as bytes via MediaFileUpload; each converter knows the structure its handler
# returns. Methods without a converter need no media handling.
def _convert_row_media(rows: list[dict[str, Any]]) -> None:
    """In place: ship transient media in named result rows (keyed by column name) back to the client as bytes."""
    for row in rows:
        for name in row:
            row[name] = _media_to_wire(row[name])


def _encode_update_status(result: Any) -> Any:
    """Converter for handlers returning an UpdateStatus with rows (insert/insert_query/update/batch_update)."""
    _convert_row_media(result.rows or [])
    return result


def _encode_compute_result(result: Any) -> Any:
    """Converter for compute(), which returns a list of result-row dicts."""
    _convert_row_media(result)
    return result


def _encode_result_set(result: dict) -> dict:
    """Converter for query terminals returning {schema, rows} (rows are positional value lists).

    With emit_media_as_urls() active, a file-backed media column arrives here as its file URL (a string),
    which _media_to_wire maps to a MediaUrlRef or MediaFileUpload; an in-memory computed image arrives as a
    PIL image, which passes through to the `image` serialization tag. Applied per value rather than by schema
    because a query may also select a media column's `.fileurl`/`.localpath` (a String-typed expr), whose path
    value needs the same treatment; _media_to_wire is a no-op on non-media strings.
    """
    result['rows'] = [[_media_to_wire(v) for v in row] for row in result['rows']]
    return result


# Catalog methods: handler(request) -> serializable result.
_HANDLERS: dict[tuple[str, str], Callable[[ProxyRequest], Any]] = {
    ('CatalogBase', 'create_table'): _create_table,
    ('CatalogBase', 'create_view'): _create_view,
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

# Path-bearing Table methods: handler(request, tbl) -> result; handle() resolves tbl and ships current md back.
_TABLE_HANDLERS: dict[tuple[str, str], Callable[[ProxyRequest, 'LocalTable'], Any]] = {
    ('Table', 'get_metadata'): _get_metadata,
    ('Table', '_path'): _get_path,
    ('Table', 'describe'): _describe,
    ('Table', 'list_views'): _list_views,
    ('Table', 'get_versions'): _get_versions,
    ('Table', 'insert'): _insert,
    ('Table', 'insert_source'): _insert_source,
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
    ('Table', 'insert_query'): _encode_update_status,
    ('Table', 'compute'): _encode_compute_result,
    ('Table', 'update'): _encode_update_status,
    ('Table', 'batch_update'): _encode_update_status,
}
