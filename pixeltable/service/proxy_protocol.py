"""Wire protocol for delegated catalog execution (the 'proxy' service).

A single generic request carries (class, method, args); dispatch routes on (class, method). Arguments
and return values are encoded by a type-driven serializer that round-trips Pixeltable's own objects
(column types, expressions, paths, enums, TableVersionMd) via their existing serialization, so adding a
new method is "register a handler + make sure its arg/return types serialize" -- no new models.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import io
import json
import math
import pathlib
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Generic, TypedDict, TypeVar
from uuid import UUID, uuid4

import numpy as np
import PIL.Image
from pydantic import BaseModel, PrivateAttr

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.catalog.dir import Dir
from pixeltable.catalog.globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation
from pixeltable.catalog.model import BtreeIndex, EmbeddingIndex
from pixeltable.catalog.path import Path
from pixeltable.catalog.table_path import TablePath, TablePathKey, TableVersionPath
from pixeltable.catalog.types import TableVersionMd
from pixeltable.catalog.update_status import RowCountStats, UpdateStatus
from pixeltable.env import Env
from pixeltable.metadata import VERSION as MD_SCHEMA_VERSION, schema
from pixeltable.query_clauses import SampleClause
from pixeltable.row import RowBatch
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import FileDestination, ObjectOps, ObjectStoreBase

if TYPE_CHECKING:
    from pixeltable._query import Query

PROTOCOL_VERSION = 3

# Reserved key marking a type-tagged value: {_TAG: <type-name>, 'v': <payload>}.
_TAG = '$pxt'


T = TypeVar('T')


class PartSink(abc.ABC, Generic[T]):
    """Destination for a request's binary values during serialization.

    Subclasses may handle the data in a variety of ways, and return a value of type `T` that refers to the stored
    data (in a subclass-specific way).
    """

    binary_parts: list[bytes]

    def __init__(self) -> None:
        self.binary_parts = []

    def add_inline(self, data: bytes) -> int:
        self.binary_parts.append(data)
        return len(self.binary_parts) - 1

    @abc.abstractmethod
    def add_media_bytes(self, data: bytes, extension: str) -> T:
        """Add an in-memory media value; returns a reference to the value."""

    @abc.abstractmethod
    def add_media_file(self, path: str) -> T:
        """Add a file-backed media value; returns a part index (inline) or an object key (out of band)."""

    def flush(self) -> None:
        """Complete any work the sink deferred while serializing."""


class InlinePartSink(PartSink[int]):
    """A PartSink that inlines everything into binary_parts (referenced by index from the JSON head)."""

    binary_parts: list[bytes]

    def add_media_bytes(self, data: bytes, extension: str) -> int:
        return self.add_inline(data)

    def add_media_file(self, path: str) -> int:
        with open(path, 'rb') as f:
            return self.add_inline(f.read())


class PxtStorePartSink(PartSink[int | str]):
    """PartSink that uploads media parts to the hosted db's home bucket. The parts will be deposited in the
    uploads/ folder of the db's home bucket, in a per-request subfolder uploads/<request-uuid>.

    The RPC then carries only the object keys; the daemon localizes the objects before dispatch; see
    proxy_dispatch._prefetch_remote_parts(). Objects under uploads/ expire via a bucket lifecycle rule, so
    they must never become stored cell values.

    Each part's key is minted during serialization, but the transfer itself is deferred to flush() so that a
    request's uploads run concurrently rather than one per media value.

    Scalars (tags 'bytes'/'ndarray') always stay inline.
    """

    _MAX_UPLOAD_THREADS = 16

    _org: str
    _db: str
    _key_prefix: str  # 'uploads/<request-uuid>/'
    _num_media_parts: int
    _store: ObjectStoreBase | None  # built on the first flush, so scalar requests skip the overhead of construction
    _pending: list[tuple[pathlib.Path, str, bool]]  # (local path, object key, remove the path after uploading it)

    def __init__(self, org: str, db: str) -> None:
        super().__init__()
        self._org = org
        self._db = db
        self._key_prefix = f'uploads/{uuid4().hex}/'
        self._num_media_parts = 0
        self._store = None
        self._pending = []

    def _get_store(self) -> ObjectStoreBase:
        if self._store is None:
            # the prefix in the URI scopes the store's temp credentials to this request's uploads
            self._store = ObjectOps.get_store(f'pxtfs://{self._org}:{self._db}/home/{self._key_prefix}', False)
        return self._store

    def add_media_bytes(self, data: bytes, extension: str) -> str:
        # stage to a temp file so all uploads go through the file path (boto3's transfer manager); flush()
        # removes the staged file once it has been uploaded
        tmp_path = TempStore.create_path(extension=extension)
        tmp_path.write_bytes(data)
        return self._add_pending(tmp_path, remove_after_upload=True)

    def add_media_file(self, path: str) -> str:
        return self._add_pending(pathlib.Path(path), remove_after_upload=False)

    def _add_pending(self, path: pathlib.Path, *, remove_after_upload: bool) -> str:
        """Mint this part's object key and queue its upload for flush()."""
        key = f'{self._key_prefix}{self._num_media_parts}{path.suffix}'
        self._num_media_parts += 1
        self._pending.append((path, key, remove_after_upload))
        return key

    def flush(self) -> None:
        """Upload the queued media parts concurrently.

        Repeated references to one path are not coalesced into a single object: the daemon moves each
        localized file into the media store (ObjectOps.put_file_resolved), which would consume a shared one.
        """
        if len(self._pending) == 0:
            return
        pending, self._pending = self._pending, []
        # fetch credentials and build the store once, here: the boto3 client it holds is bound to it at
        # construction (see S3Store.client()), so the upload threads share that one client, as boto3 permits
        store = self._get_store()

        def upload(item: tuple[pathlib.Path, str, bool]) -> None:
            path, key, remove_after_upload = item
            try:
                url = f'pxtfs://{self._org}:{self._db}/home/{key}'
                store.copy_local_file(path, FileDestination(url=url, remote_key=key))
            finally:
                if remove_after_upload:
                    path.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=min(self._MAX_UPLOAD_THREADS, len(pending))) as executor:
            list(executor.map(upload, pending))


@dataclasses.dataclass
class LocalFile:
    """A reference to a local file so _serialize()/_deserialize() can handle it correctly."""

    path: str


@dataclasses.dataclass
class MediaPath:
    """References persisted daemon media (under the media dir) by a media-dir-relative path."""

    path: str


class ProxyRequest(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    schema_version: int = MD_SCHEMA_VERSION
    class_name: str  # the CatalogBase/TableBase method's defining class
    method: str

    # TablePathKey.as_dict() for Table methods, used to obtain a Table instance
    path_key: dict | None = None

    # TablePathKey.as_dict() of the expected concrete versions, for staleness validation
    snapshot_path_key: dict | None = None

    args: dict[str, Any]  # method kwargs
    request_id: str | None = None  # set for mutating methods (idempotency); unused for now

    # raw binary parts, referenced by index from the tagged values in args
    _binary_parts: list[bytes] = PrivateAttr(default_factory=list)

    # temp path -> the client's original filename; needed for informative error messages
    _uploaded_names: dict[str, str] = PrivateAttr(default_factory=dict)

    # object key -> local temp path, for media parts the client uploaded out of band; populated by the
    # server before dispatch (see proxy_dispatch._prefetch_remote_parts)
    _remote_parts: dict[str, str] = PrivateAttr(default_factory=dict)


class ProxyResponse(TypedDict, total=False):
    """The fields a proxy response carries; encode_response() turns them into the body's JSON head.

    A response sets either result or error, and current_md only after a mutation or a stale-md refusal.
    """

    result: Any  # return value
    error: dict[str, Any]  # excs.Error.to_dict(), set instead of result on failure

    # serialized TableMdPath (list[TableVersionMd]); returned after a mutation so the client refreshes its md
    current_md: Any

    # True if the request's snapshot_path_key was behind the current schema version
    is_stale_md: bool


def _serialize(obj: Any, sink: PartSink) -> Any:
    """Encode a Python value to a json-serializable dict that can be deserialized by _deserialize().

    Binary values go to sink: inlined into its binary_parts (referenced by index), or routed out of band by
    a media-capable sink (referenced by object key).
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        # nan/inf are valid Float cell values but are lost (rendered as null) by JSON serialization
        return {_TAG: 'float', 'v': repr(obj)}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, IfExistsParam):
        return {_TAG: 'IfExistsParam', 'v': obj.name}
    if isinstance(obj, IfNotExistsParam):
        return {_TAG: 'IfNotExistsParam', 'v': obj.name}
    if isinstance(obj, MediaValidation):
        return {_TAG: 'MediaValidation', 'v': obj.name}
    if isinstance(obj, ts.ColumnType):
        return {_TAG: 'ColumnType', 'v': obj.as_dict()}
    if isinstance(obj, Path):
        # Paths on the wire are in-db (the target catalog is implied by routing), so org/db are dropped.
        return {_TAG: 'Path', 'v': {'components': list(obj.components), 'version': obj.version}}
    if isinstance(obj, TableVersionMd):
        return {_TAG: 'TableVersionMd', 'v': obj.as_dict()}
    if isinstance(obj, TablePath):
        # the path's version key; the server rebuilds a TableVersionPath from it
        return {_TAG: 'TablePathKey', 'v': obj.key().as_dict()}
    if isinstance(obj, exprs.Expr):
        return {_TAG: 'Expr', 'v': obj.as_dict()}
    if isinstance(obj, SampleClause):
        return {_TAG: 'SampleClause', 'v': obj.as_dict()}
    if isinstance(obj, func.Function):
        return {_TAG: 'Function', 'v': obj.as_dict()}
    if isinstance(obj, func.GeneratingFunctionCall):
        return {_TAG: 'GeneratingFunctionCall', 'v': obj.as_dict()}
    if isinstance(obj, EmbeddingIndex):
        # A declarative model's embedding-index spec (a dataclass of an Expr column ref + embedding Functions +
        # scalars); serialize field-by-field so the nested Exprs/Functions round-trip via their own handlers.
        return {
            _TAG: 'EmbeddingIndex',
            'v': {f.name: _serialize(getattr(obj, f.name), InlinePartSink()) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, BtreeIndex):
        # A declarative model's B-tree-index spec (a dataclass wrapping an Expr column ref).
        return {
            _TAG: 'BtreeIndex',
            'v': {f.name: _serialize(getattr(obj, f.name), InlinePartSink()) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, DirEntry):
        # only the fields any get_dir_contents() consumer reads: dir presence, table id/md, error count
        return {
            _TAG: 'DirEntry',
            'v': {
                'is_dir': obj.dir is not None,
                'table': None if obj.table is None else {'id': _serialize(obj.table.id, sink), 'md': obj.table.md},
                'dir_entries': {name: _serialize(child, sink) for name, child in obj.dir_entries.items()},
                'table_error_count': obj.table_error_count,
            },
        }
    if isinstance(obj, UpdateStatus):
        d = dataclasses.asdict(obj)
        d['rows'] = _serialize(obj.rows, sink)  # returned rows may hold non-JSON scalars (timestamps, etc.)
        return {_TAG: 'UpdateStatus', 'v': d}
    if isinstance(obj, RowBatch):
        return {
            _TAG: 'RowBatch',
            'v': {
                'schema': {name: t.as_dict() for name, t in obj._col_types.items()},
                'rows': [[_serialize(val, sink) for val in row._data] for row in obj],
                'errors': [row.errors for row in obj],
                'index_values': [
                    {name: _serialize(val, sink) for name, val in row.index_values.items()} for row in obj
                ],
            },
        }
    if isinstance(obj, Dir):
        # a Dir is an identity-only handle; only its id crosses the wire
        return {_TAG: 'Dir', 'v': str(obj._id)}
    if isinstance(obj, UUID):
        return {_TAG: 'UUID', 'v': str(obj)}
    if isinstance(obj, datetime.datetime):
        return {_TAG: 'datetime', 'v': obj.isoformat()}
    if isinstance(obj, datetime.date):
        # check after datetime (datetime is a subclass of date)
        return {_TAG: 'date', 'v': obj.isoformat()}
    if isinstance(obj, pathlib.Path):
        return str(obj)  # filesystem paths travel as strings
    if isinstance(obj, bytes):
        # a Binary cell, or an array column's stored byte form as returned by compute()
        # TODO: We should be coalescing these into out-of-band uploads via add_media_bytes(), not inlining them
        #     in HTTP requests [PXT-1314]
        return {_TAG: 'bytes', 'v': sink.add_inline(obj)}
    if isinstance(obj, np.ndarray):
        # TODO: We should be coalescing these into out-of-band uploads via add_media_bytes(), not inlining them
        #     in HTTP requests [PXT-1314]
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)  # .npy carries dtype and shape
        return {_TAG: 'ndarray', 'v': sink.add_inline(buf.getvalue())}
    if isinstance(obj, PIL.Image.Image):
        # an in-memory image; file-backed media travels as a path
        buf = io.BytesIO()
        fmt = obj.format or 'PNG'
        obj.save(buf, format=fmt)
        return {_TAG: 'image', 'format': fmt, 'v': sink.add_media_bytes(buf.getvalue(), f'.{fmt.lower()}')}
    if isinstance(obj, LocalFile):
        # carry the original file name so the receiver can restore it in error messages (its temp copy uses an
        # opaque name) and preserve the extension for media-type detection
        return {_TAG: 'file', 'name': pathlib.Path(obj.path).name, 'v': sink.add_media_file(obj.path)}
    if isinstance(obj, MediaPath):
        return {_TAG: 'mediapath', 'v': obj.path}
    if isinstance(obj, list):
        return [_serialize(x, sink) for x in obj]
    if isinstance(obj, tuple):
        return {_TAG: 'tuple', 'v': [_serialize(x, sink) for x in obj]}
    if isinstance(obj, dict):
        if _TAG in obj or any(not isinstance(k, str) for k in obj):
            # store as ordered key/value pairs, which keeps a key colliding with the reserved tag out of the top
            # level and preserves keys that json cannot represent (json object keys are always strings)
            return {_TAG: 'rawdict', 'v': [[_serialize(k, sink), _serialize(val, sink)] for k, val in obj.items()]}
        return {k: _serialize(v, sink) for k, v in obj.items()}
    raise AssertionError(f'cannot serialize {type(obj).__name__} for the proxy protocol')


def _check_valid_fn(fn: func.Function) -> func.Function:
    if isinstance(fn, func.InvalidFunction):
        raise excs.NotFoundError(
            excs.ErrorCode.FUNCTION_NOT_FOUND,
            f'The request references the UDF `{fn.self_path}`, '
            'but that UDF is not defined in the remote database.\n'
            'You can use `pxt db update` to deploy a new version of the UDF to the remote database.',
        )
    return fn


def _check_valid_expr(expr: exprs.Expr) -> exprs.Expr:
    if not expr.is_valid:
        raise excs.NotFoundError(
            excs.ErrorCode.FUNCTION_NOT_FOUND,
            f'{expr.validation_error.protocol_error_msg()}\n'
            'You can use `pxt db update` to deploy a new version of the UDF to the remote database.',
        )
    return expr


def _check_valid_gfc(gfc: func.GeneratingFunctionCall) -> func.GeneratingFunctionCall:
    if not gfc.is_valid:
        raise excs.NotFoundError(
            excs.ErrorCode.FUNCTION_NOT_FOUND,
            f'{gfc.validation_error.protocol_error_msg()}\n'
            'You can use `pxt db update` to deploy a new version of the iterator to the remote database.',
        )
    return gfc


def check_query(query: 'Query') -> 'Query':
    for e in query._component_exprs():
        _check_valid_expr(e)
    return query


def _deserialize(
    obj: Any,
    binary_parts: list[bytes],
    uploaded_names: dict[str, str] | None = None,
    remote_parts: dict[str, str] | None = None,
) -> Any:
    """Inverse of _serialize(). When uploaded_names is provided, each 'file' arg maps its temp path to the
    original filename in it. remote_parts maps each out-of-band media part's object key to a pre-downloaded
    local temp path.

    A container whose values all come back unchanged is returned as it was, so a large result that holds no
    encoded value is walked rather than rebuilt."""
    if isinstance(obj, list):
        deserialized_list: list[Any] | None = None
        for i, elem in enumerate(obj):
            deserialized_elem = _deserialize(elem, binary_parts, uploaded_names, remote_parts)
            if deserialized_elem is not elem:
                if deserialized_list is None:
                    deserialized_list = list(obj)
                deserialized_list[i] = deserialized_elem
        return obj if deserialized_list is None else deserialized_list

    if isinstance(obj, dict):
        tag = obj.get(_TAG)
        if not isinstance(tag, str) or 'v' not in obj:
            decoded: dict[Any, Any] | None = None
            for k, val in obj.items():
                d = _deserialize(val, binary_parts, uploaded_names, remote_parts)
                if d is not val:
                    if decoded is None:
                        decoded = dict(obj)
                    decoded[k] = d
            return obj if decoded is None else decoded

        v = obj['v']
        if tag == 'float':
            return float(v)  # nan/inf
        if tag == 'rawdict':
            return {
                _deserialize(k, binary_parts, uploaded_names, remote_parts): _deserialize(
                    val, binary_parts, uploaded_names, remote_parts
                )
                for k, val in v
            }
        if tag == 'tuple':
            return tuple(_deserialize(x, binary_parts, uploaded_names, remote_parts) for x in v)
        if tag == 'bytes':
            return binary_parts[v]
        if tag == 'ndarray':
            return np.load(io.BytesIO(binary_parts[v]), allow_pickle=False)
        if tag == 'image':
            # a str v is an object key of an out-of-band media part, resolved to a pre-downloaded local path
            img = PIL.Image.open(
                _remote_part_path(v, remote_parts) if isinstance(v, str) else io.BytesIO(binary_parts[v])
            )
            img.load()  # read pixels now so the result doesn't depend on the transient buffer/file
            return img
        if tag == 'file':
            if isinstance(v, str):
                # an object key of an out-of-band media part; return its pre-downloaded local path
                dest_str = _remote_part_path(v, remote_parts)
            else:
                # write the sent bytes to an opaque temp path (extension preserved for media-type detection)
                # TODO: We still need this because bytes/ndarrays are still inlined into HTTP requests; once that's
                #     fixed, this code branch can be removed (v will always be a str) [PXT-1314]
                dest = TempStore.create_path(extension=pathlib.Path(obj['name']).suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, 'wb') as f:
                    f.write(binary_parts[v])
                dest_str = str(dest)
            # record the original file name so an error can reference it rather than the temp path
            if uploaded_names is not None:
                uploaded_names[dest_str] = obj['name']
            return dest_str
        if tag == 'mediapath':
            # persisted daemon media; the client localizes it from the daemon's /media endpoint (see ProxyClient)
            return MediaPath(v)
        if tag == 'IfExistsParam':
            return IfExistsParam[v]
        if tag == 'IfNotExistsParam':
            return IfNotExistsParam[v]
        if tag == 'MediaValidation':
            return MediaValidation[v]
        if tag == 'ColumnType':
            return ts.ColumnType.from_dict(v)
        if tag == 'Path':
            return Path.from_components(tuple(v['components']), version=v['version'])
        if tag == 'TableVersionMd':
            return TableVersionMd.from_dict(v)
        if tag == 'TablePathKey':
            return TableVersionPath.from_key(TablePathKey.from_dict(v))
        if tag == 'Expr':
            return _check_valid_expr(exprs.Expr.from_dict(v))
        if tag == 'SampleClause':
            return SampleClause.from_dict(v)
        if tag == 'Function':
            return _check_valid_fn(func.Function.from_dict(v))
        if tag == 'GeneratingFunctionCall':
            return _check_valid_gfc(func.GeneratingFunctionCall.from_dict(v))
        if tag == 'EmbeddingIndex':
            return EmbeddingIndex(**{name: _deserialize(val, []) for name, val in v.items()})
        if tag == 'BtreeIndex':
            return BtreeIndex(**{name: _deserialize(val, []) for name, val in v.items()})
        if tag == 'DirEntry':
            table = v['table']
            return DirEntry(
                dir=schema.Dir(md={}) if v['is_dir'] else None,
                dir_entries={
                    name: _deserialize(child, binary_parts, uploaded_names, remote_parts)
                    for name, child in v['dir_entries'].items()
                },
                table=None
                if table is None
                else schema.Table(
                    id=_deserialize(table['id'], binary_parts, uploaded_names, remote_parts), md=table['md']
                ),
                table_error_count=v['table_error_count'],
            )
        if tag == 'UpdateStatus':
            d = dict(v)
            d['rows'] = _deserialize(d['rows'], binary_parts)
            for field in ('row_count_stats', 'cascade_row_count_stats'):
                d[field] = RowCountStats(**d[field])
            return UpdateStatus(**d)
        if tag == 'RowBatch':
            return RowBatch(
                [tuple(_deserialize(val, binary_parts, uploaded_names) for val in row_data) for row_data in v['rows']],
                {name: ts.ColumnType.from_dict(t) for name, t in v['schema'].items()},
                errors=v['errors'],
                index_values=[
                    {name: _deserialize(val, binary_parts, uploaded_names) for name, val in iv.items()}
                    for iv in v['index_values']
                ],
            )
        if tag == 'Dir':
            return Dir(UUID(v))
        if tag == 'UUID':
            return UUID(v)
        if tag == 'datetime':
            return datetime.datetime.fromisoformat(v)
        if tag == 'date':
            return datetime.date.fromisoformat(v)
        # a json value of its own that happens to carry the reserved key
        return {k: _deserialize(val, binary_parts, uploaded_names, remote_parts) for k, val in obj.items()}
    return obj


def _remote_part_path(key: str, remote_parts: dict[str, str] | None) -> str:
    """Resolve an out-of-band media part's object key to its pre-downloaded local path."""
    if remote_parts is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'Cannot localize uploaded media object {key!r}: this receiver has no access to uploaded objects',
        )
    if key not in remote_parts:
        raise excs.RequestError(
            excs.ErrorCode.STORAGE_NOT_FOUND,
            f'Cannot localize uploaded media object {key!r}: object was not prefetched on this receiver',
        )
    return remote_parts[key]


def collect_remote_keys(args: Any) -> list[str]:
    """Return the object keys of all out-of-band media parts ('file'/'image' tags whose 'v' is a str) in
    serialized args, deduplicated in encounter order."""
    keys: dict[str, None] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            if obj.get(_TAG) in ('file', 'image') and isinstance(obj.get('v'), str):
                keys[obj['v']] = None
            else:
                for value in obj.values():
                    walk(value)

    walk(args)
    return list(keys)


def serialize_args(args: dict[str, Any], sink: PartSink) -> dict[str, Any]:
    """Encode a request's args for the wire; binary values go to sink (see _serialize())."""
    wire_args = _serialize(args, sink)
    assert isinstance(wire_args, dict)
    sink.flush()  # an out-of-band sink defers its transfers to here, where they can run concurrently
    return wire_args


def deserialize_request(request: ProxyRequest) -> dict[str, Any]:
    """Decode request.args, recording each uploaded file's temp-path-to-original-name mapping on the request."""
    return _deserialize(request.args, request._binary_parts, request._uploaded_names, request._remote_parts or None)


def encode_local_path(value: Any) -> Any:
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


def deserialize_value(value: Any, parts: list[bytes]) -> Any:
    """Decode a value carried by a response, resolving its binary references from parts."""
    return _deserialize(value, parts)


_dumps = json.JSONEncoder(separators=(',', ':')).encode


def value_encoder(sink: PartSink) -> Callable[[Any], str]:
    """A json encoder that writes what json can and hands the rest to _serialize().

    Faster than encoding a _serialize()ed copy, since json walks the value in C. A value json writes itself
    reaches the receiver in json's form: nan and inf as NaN and Infinity, a tuple as an array, a dict key as
    a string. json writes a dict as-is, tag key and all, and an inlined object inside it never reaches the
    hook, so a caller runs serialize_value() over a value that can hold json of its own.
    """
    return json.JSONEncoder(separators=(',', ':'), default=lambda obj: _serialize(obj, sink)).encode


def escape_json(value: Any) -> Any:
    """Rewrite the dicts in a json value that json cannot write as data: the ones carrying the reserved tag
    key, and the ones with a key json would coerce to a string. Returns value itself if there is nothing to
    rewrite, so an ordinary value costs no allocation.

    An inlined object needs no rewriting: json cannot write it either way, so it reaches _serialize() through
    the encoder's hook.
    """
    if isinstance(value, dict):
        if _TAG in value or not all(isinstance(k, str) for k in value):
            return {_TAG: 'rawdict', 'v': [[k, escape_json(v)] for k, v in value.items()]}
        escaped: dict[str, Any] | None = None
        for k, v in value.items():
            e = escape_json(v)
            if e is not v:
                if escaped is None:
                    escaped = dict(value)
                escaped[k] = e
        return value if escaped is None else escaped
    if isinstance(value, list):
        escaped_list: list[Any] | None = None
        for i, v in enumerate(value):
            e = escape_json(v)
            if e is not v:
                if escaped_list is None:
                    escaped_list = list(value)
                escaped_list[i] = e
        return value if escaped_list is None else escaped_list
    return value


def response_body(
    result_json: bytes,
    parts: list[bytes],
    *,
    error_json: bytes = b'null',
    current_md_json: bytes = b'null',
    is_stale_md: bool = False,
) -> bytes:
    """The wire body for a response whose fields are already encoded.

    Both the generic path and a caller that encodes its own result (see Query._collect_content()) write their
    body here, so the head has one layout.
    """
    head = bytearray(b'{"result":')
    head += result_json
    head += b',"error":'
    head += error_json
    head += b',"current_md":'
    head += current_md_json
    head += b',"is_stale_md":'
    head += b'true' if is_stale_md else b'false'
    head += b'}'
    return encode_body(bytes(head), parts)


def encode_response(response: ProxyResponse) -> bytes:
    """The wire body for a response, moving any binary values in it out to the body's parts."""
    sink = InlinePartSink()
    result_json = _dumps(_serialize(response.get('result'), sink)).encode()
    current_md_json = _dumps(_serialize(response.get('current_md'), sink)).encode()
    error = response.get('error')
    return response_body(
        result_json,
        sink.binary_parts,
        error_json=b'null' if error is None else _dumps(error).encode(),
        current_md_json=current_md_json,
        is_stale_md=response.get('is_stale_md', False),
    )


def encode_dir_tree(dir_path: pathlib.Path) -> list[dict[str, Any]]:
    """Encode a local directory tree for transport: one {relpath, file} entry per file. relpath includes
    dir_path's own name as its first component, so decode_dir_tree() rebuilds the tree under a directory of the
    same name (which source-format detection keys on, e.g. a *.parquet directory)."""
    return [
        {'relpath': path.relative_to(dir_path.parent).as_posix(), 'file': LocalFile(str(path))}
        for path in sorted(dir_path.rglob('*'))
        if path.is_file()
    ]


def decode_dir_tree(files: list[dict[str, Any]], root: pathlib.Path) -> pathlib.Path:
    """Inverse of encode_dir_tree(): rebuild the entries under root and return the reassembled tree's top-level
    directory (named after the original source directory). The caller owns root and removes it to clean up."""
    for entry in files:
        dest = root / entry['relpath']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(entry['file'], dest)
    tops = {pathlib.PurePosixPath(entry['relpath']).parts[0] for entry in files}
    assert len(tops) == 1, tops
    return root / next(iter(tops))


_U32 = struct.Struct('>I')


def encode_body(head: bytes, binary_parts: list[bytes]) -> bytes:
    out = [_U32.pack(len(head)), head, _U32.pack(len(binary_parts))]
    for part in binary_parts:
        out.append(_U32.pack(len(part)))
        out.append(part)
    return b''.join(out)


def decode_body(body: bytes) -> tuple[bytes, list[bytes]]:
    view = memoryview(body)
    offset = 0

    def take(n: int) -> bytes:
        nonlocal offset
        chunk = view[offset : offset + n]
        if len(chunk) != n:
            raise ValueError('truncated framed body')
        offset += n
        return bytes(chunk)

    def take_u32() -> int:
        return _U32.unpack(take(4))[0]

    head = take(take_u32())
    n_parts = take_u32()
    binary_parts = [take(take_u32()) for _ in range(n_parts)]
    if offset != len(view):
        raise ValueError('trailing bytes after framed body')
    return head, binary_parts
