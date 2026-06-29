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
import pathlib
from typing import Any
from uuid import UUID

import numpy as np
import PIL.Image
from pydantic import BaseModel, PrivateAttr

from pixeltable import exprs, func, type_system as ts
from pixeltable.catalog.dir import Dir
from pixeltable.catalog.globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation, TableVersionMd
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
class MediaFileUpload:
    """Wraps a local media file so serialize() ships its bytes as a binary part; deserialize() writes them to the
    local TempStore and yields the new path. Used to move file-backed media across the proxy boundary in both
    directions (client insert/compute args and daemon transient result rows)."""

    path: str


@dataclasses.dataclass
class MediaUrlRef:
    """References persisted daemon media (under the media dir) by a media-dir-relative path. The daemon emits it for
    result media; the client fetches it from the daemon's /media endpoint into its FileCache (see ProxyClient).
    deserialize() yields it unchanged so the client can find and localize it after decoding the response."""

    ref: str


class ProxyRequest(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    schema_version: int = MD_SCHEMA_VERSION
    class_name: str  # the CatalogBase/TableBase method's defining class
    method: str
    # TablePathKey.as_dict() forms for path-bearing Table methods: the effective path_key (what to
    # resolve) and the believed snapshot_path_key (concrete versions, for staleness validation).
    path_key: dict | None = None
    snapshot_path_key: dict | None = None
    args: dict[str, Any]  # method kwargs, type-driven-serialized
    request_id: str | None = None  # set for mutating methods (idempotency); unused for now

    # raw binary parts referenced by 'blob' tags in args; carried by the framing layer, not in the JSON
    _binary_parts: list[bytes] = PrivateAttr(default_factory=list)


class ProxyResponse(BaseModel):
    result: Any = None  # type-driven-serialized return value
    error: dict[str, Any] | None = None  # excs.Error.to_dict(), set instead of result on failure
    current_md: Any = None  # serialized TableMdPath (list[TableVersionMd]); set for path-bearing methods
    is_stale_md: bool = False  # True if the request's snapshot_path_key was behind the current schema version

    # raw binary parts referenced by 'blob' tags in result/current_md; carried by the framing layer, not the JSON
    _binary_parts: list[bytes] = PrivateAttr(default_factory=list)


def _add_part(binary_parts: list[bytes], data: bytes) -> int:
    """Append a binary part and return its index, referenced by a 'blob' tag in the JSON."""
    binary_parts.append(data)
    return len(binary_parts) - 1


def serialize(obj: Any, binary_parts: list[bytes]) -> Any:
    """Encode a Python value to a json-serializable dict that can be deserialized by deserialize().

    Binary values are appended to binary_parts as raw bytes and referenced inside the dict by index.
    """
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
    if isinstance(obj, DirEntry):
        # only the fields any get_dir_contents() consumer reads: dir presence, table id/md, error count
        return {
            _TAG: 'DirEntry',
            'v': {
                'is_dir': obj.dir is not None,
                'table': None
                if obj.table is None
                else {'id': serialize(obj.table.id, binary_parts), 'md': obj.table.md},
                'dir_entries': {name: serialize(child, binary_parts) for name, child in obj.dir_entries.items()},
                'table_error_count': obj.table_error_count,
            },
        }
    if isinstance(obj, UpdateStatus):
        d = dataclasses.asdict(obj)
        d['rows'] = serialize(obj.rows, binary_parts)  # returned rows may hold non-JSON scalars (timestamps, etc.)
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
    if isinstance(obj, MediaFileUpload):
        with open(obj.path, 'rb') as f:
            data = f.read()
        # carry the original file name so the receiver's temp copy keeps it (e.g. for media validation errors)
        return {_TAG: 'mediafile', 'name': pathlib.Path(obj.path).name, 'v': _add_part(binary_parts, data)}
    if isinstance(obj, MediaUrlRef):
        return {_TAG: 'mediaurl', 'v': obj.ref}
    if isinstance(obj, list):
        return [serialize(x, binary_parts) for x in obj]
    if isinstance(obj, tuple):
        return {_TAG: 'tuple', 'v': [serialize(x, binary_parts) for x in obj]}
    if isinstance(obj, dict):
        if _TAG in obj:
            # a user dict whose own key collides with the reserved tag: store it as ordered key/value pairs so
            # the tag no longer sits at the top level and the dict round-trips
            return {_TAG: 'rawdict', 'v': [[k, serialize(val, binary_parts)] for k, val in obj.items()]}
        return {k: serialize(v, binary_parts) for k, v in obj.items()}
    raise AssertionError(f'cannot serialize {type(obj).__name__} for the proxy protocol')


def deserialize(obj: Any, binary_parts: list[bytes]) -> Any:
    """Inverse of serialize()."""
    if isinstance(obj, list):
        return [deserialize(x, binary_parts) for x in obj]
    if isinstance(obj, dict):
        tag = obj.get(_TAG)
        if tag is None:
            return {k: deserialize(v, binary_parts) for k, v in obj.items()}
        v = obj['v']
        if tag == 'rawdict':
            # a user dict whose own key collided with the reserved tag; stored as ordered key/value pairs
            return {k: deserialize(val, binary_parts) for k, val in v}
        if tag == 'tuple':
            return tuple(deserialize(x, binary_parts) for x in v)
        if tag == 'bytes':
            return binary_parts[v]
        if tag == 'ndarray':
            return np.load(io.BytesIO(binary_parts[v]), allow_pickle=False)
        if tag == 'image':
            img = PIL.Image.open(io.BytesIO(binary_parts[v]))
            img.load()  # read pixels now so the result doesn't depend on the transient buffer
            return img
        if tag == 'mediafile':
            # write the shipped bytes into the local TempStore, preserving the original file name, and hand back
            # the new path
            dest = TempStore.create_path(name=obj['name'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(binary_parts[v])
            return str(dest)
        if tag == 'mediaurl':
            # persisted daemon media; the client localizes it from the daemon's /media endpoint (see ProxyClient)
            return MediaUrlRef(v)
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
        if tag == 'DirEntry':
            table = v['table']
            return DirEntry(
                dir=schema.Dir(md={}) if v['is_dir'] else None,
                dir_entries={name: deserialize(child, binary_parts) for name, child in v['dir_entries'].items()},
                table=None
                if table is None
                else schema.Table(id=deserialize(table['id'], binary_parts), md=table['md']),
                table_error_count=v['table_error_count'],
            )
        if tag == 'UpdateStatus':
            d = dict(v)
            d['rows'] = deserialize(d['rows'], binary_parts)
            for field in ('row_count_stats', 'cascade_row_count_stats', 'ext_row_count_stats'):
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
