"""Wire protocol for delegated catalog execution (the 'proxy' service).

A single generic request carries (class, method, args); dispatch routes on (class, method). Arguments
and return values are encoded by a type-driven serializer that round-trips Pixeltable's own objects
(column types, expressions, paths, enums, TableVersionMd) via their existing serialization, so adding a
new method is "register a handler + make sure its arg/return types serialize" -- no new models.
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import math
import pathlib
import shutil
import struct
from typing import Any
from uuid import UUID

import numpy as np
import PIL.Image
from pydantic import BaseModel, PrivateAttr

from pixeltable import exprs, func, type_system as ts
from pixeltable.catalog.dir import Dir
from pixeltable.catalog.globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation, TableVersionMd
from pixeltable.catalog.model import EmbeddingIndex
from pixeltable.catalog.path import Path
from pixeltable.catalog.table_path import TablePath, TablePathKey, TableVersionPath
from pixeltable.catalog.update_status import RowCountStats, UpdateStatus
from pixeltable.metadata import VERSION as MD_SCHEMA_VERSION, schema
from pixeltable.query_clauses import SampleClause
from pixeltable.utils.local_store import TempStore

PROTOCOL_VERSION = 1

# Reserved key marking a type-tagged value: {_TAG: <type-name>, 'v': <payload>}.
_TAG = '$pxt'


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

    # raw binary parts referenced by 'blob' tags in args
    _binary_parts: list[bytes] = PrivateAttr(default_factory=list)

    # temp path -> the client's original filename; needed for informative error messages
    _uploaded_names: dict[str, str] = PrivateAttr(default_factory=dict)


class ProxyResponse(BaseModel):
    result: Any = None  # return value
    error: dict[str, Any] | None = None  # excs.Error.to_dict(), set instead of result on failure

    # serialized TableMdPath (list[TableVersionMd]); returned after a mutation so the client refreshes its md
    current_md: Any = None

    # True if the request's snapshot_path_key was behind the current schema version
    is_stale_md: bool = False

    # raw binary parts referenced by 'blob' tags in result/current_md
    _binary_parts: list[bytes] = PrivateAttr(default_factory=list)


def _add_part(binary_parts: list[bytes], data: bytes) -> int:
    """Append a binary part and return its index, referenced by a 'blob' tag in the JSON."""
    binary_parts.append(data)
    return len(binary_parts) - 1


def _serialize(obj: Any, binary_parts: list[bytes]) -> Any:
    """Encode a Python value to a json-serializable dict that can be deserialized by _deserialize().

    Binary values are appended to binary_parts as raw bytes and referenced inside the dict by index.
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
            'v': {f.name: _serialize(getattr(obj, f.name), []) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, DirEntry):
        # only the fields any get_dir_contents() consumer reads: dir presence, table id/md, error count
        return {
            _TAG: 'DirEntry',
            'v': {
                'is_dir': obj.dir is not None,
                'table': None
                if obj.table is None
                else {'id': _serialize(obj.table.id, binary_parts), 'md': obj.table.md},
                'dir_entries': {name: _serialize(child, binary_parts) for name, child in obj.dir_entries.items()},
                'table_error_count': obj.table_error_count,
            },
        }
    if isinstance(obj, UpdateStatus):
        d = dataclasses.asdict(obj)
        d['rows'] = _serialize(obj.rows, binary_parts)  # returned rows may hold non-JSON scalars (timestamps, etc.)
        return {_TAG: 'UpdateStatus', 'v': d}
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
        return {_TAG: 'bytes', 'v': _add_part(binary_parts, obj)}
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)  # .npy carries dtype and shape
        return {_TAG: 'ndarray', 'v': _add_part(binary_parts, buf.getvalue())}
    if isinstance(obj, PIL.Image.Image):
        # an in-memory image; file-backed media travels as a path
        buf = io.BytesIO()
        fmt = obj.format or 'PNG'
        obj.save(buf, format=fmt)
        return {_TAG: 'image', 'format': fmt, 'v': _add_part(binary_parts, buf.getvalue())}
    if isinstance(obj, LocalFile):
        with open(obj.path, 'rb') as f:
            data = f.read()
        # carry the original file name so the receiver can restore it in error messages (its temp copy uses an
        # opaque name) and preserve the extension for media-type detection
        return {_TAG: 'file', 'name': pathlib.Path(obj.path).name, 'v': _add_part(binary_parts, data)}
    if isinstance(obj, MediaPath):
        return {_TAG: 'mediapath', 'v': obj.path}
    if isinstance(obj, list):
        return [_serialize(x, binary_parts) for x in obj]
    if isinstance(obj, tuple):
        return {_TAG: 'tuple', 'v': [_serialize(x, binary_parts) for x in obj]}
    if isinstance(obj, dict):
        if _TAG in obj:
            # a user dict whose own key collides with the reserved tag: store it as ordered key/value pairs so
            # the tag no longer sits at the top level and the dict round-trips
            return {_TAG: 'rawdict', 'v': [[k, _serialize(val, binary_parts)] for k, val in obj.items()]}
        return {k: _serialize(v, binary_parts) for k, v in obj.items()}
    raise AssertionError(f'cannot serialize {type(obj).__name__} for the proxy protocol')


def _deserialize(obj: Any, binary_parts: list[bytes], uploaded_names: dict[str, str] | None = None) -> Any:
    """Inverse of _serialize(). When uploaded_names is provided, each 'file' arg maps its temp path to the
    original filename in it."""
    if isinstance(obj, list):
        return [_deserialize(x, binary_parts, uploaded_names) for x in obj]
    if isinstance(obj, dict):
        tag = obj.get(_TAG)
        if tag is None:
            return {k: _deserialize(v, binary_parts, uploaded_names) for k, v in obj.items()}
        v = obj['v']
        if tag == 'float':
            return float(v)  # nan/inf
        if tag == 'rawdict':
            # a user dict whose own key collided with the reserved tag; stored as ordered key/value pairs
            return {k: _deserialize(val, binary_parts, uploaded_names) for k, val in v}
        if tag == 'tuple':
            return tuple(_deserialize(x, binary_parts, uploaded_names) for x in v)
        if tag == 'bytes':
            return binary_parts[v]
        if tag == 'ndarray':
            return np.load(io.BytesIO(binary_parts[v]), allow_pickle=False)
        if tag == 'image':
            img = PIL.Image.open(io.BytesIO(binary_parts[v]))
            img.load()  # read pixels now so the result doesn't depend on the transient buffer
            return img
        if tag == 'file':
            # write the sent bytes to an opaque temp path (extension preserved for media-type detection); record
            # the original file name so an error can reference it rather than the temp path
            parts_idx = v
            dest = TempStore.create_path(extension=pathlib.Path(obj['name']).suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(binary_parts[parts_idx])
            if uploaded_names is not None:
                uploaded_names[str(dest)] = obj['name']
            return str(dest)
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
            return exprs.Expr.from_dict(v)
        if tag == 'SampleClause':
            return SampleClause.from_dict(v)
        if tag == 'Function':
            return func.Function.from_dict(v)
        if tag == 'GeneratingFunctionCall':
            return func.GeneratingFunctionCall.from_dict(v)
        if tag == 'EmbeddingIndex':
            return EmbeddingIndex(**{name: _deserialize(val, []) for name, val in v.items()})
        if tag == 'DirEntry':
            table = v['table']
            return DirEntry(
                dir=schema.Dir(md={}) if v['is_dir'] else None,
                dir_entries={
                    name: _deserialize(child, binary_parts, uploaded_names) for name, child in v['dir_entries'].items()
                },
                table=None
                if table is None
                else schema.Table(id=_deserialize(table['id'], binary_parts, uploaded_names), md=table['md']),
                table_error_count=v['table_error_count'],
            )
        if tag == 'UpdateStatus':
            d = dict(v)
            d['rows'] = _deserialize(d['rows'], binary_parts)
            for field in ('row_count_stats', 'cascade_row_count_stats'):
                d[field] = RowCountStats(**d[field])
            return UpdateStatus(**d)
        if tag == 'Dir':
            return Dir(UUID(v))
        if tag == 'UUID':
            return UUID(v)
        if tag == 'datetime':
            return datetime.datetime.fromisoformat(v)
        if tag == 'date':
            return datetime.date.fromisoformat(v)
        raise AssertionError(f'unknown proxy serialization tag: {tag!r}')
    return obj


def serialize_request(request: ProxyRequest) -> None:
    """Encode request.args in place, appending any binary values to request._binary_parts for transport."""
    request.args = _serialize(request.args, request._binary_parts)


def deserialize_request(request: ProxyRequest) -> dict[str, Any]:
    """Decode request.args, recording each uploaded file's temp-path-to-original-name mapping on the request."""
    return _deserialize(request.args, request._binary_parts, request._uploaded_names)


def serialize_response(response: ProxyResponse) -> None:
    """Encode response.result and response.current_md in place, appending binary values to response._binary_parts."""
    response.result = _serialize(response.result, response._binary_parts)
    response.current_md = _serialize(response.current_md, response._binary_parts)


def deserialize_response(response: ProxyResponse, value: Any) -> Any:
    """Decode a value carried by response (its result or current_md), resolving binary references from it."""
    return _deserialize(value, response._binary_parts)


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
