"""The UDF servers of the pixeltable.udf setting, independent of the provider that runs them.

Contains the encoding of a call's request and response, the header with a server's secret, and the lookup of the
servers' endpoints.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import threading
from typing import Any

from typing_extensions import Self

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.env import Env, UdfEndpoints
from pixeltable.service import db, proxy_protocol

# the environment variables with the endpoints of a hosted database's UDF servers, as JSON objects:
# UdfEndpoints.server_urls and UdfEndpoints.headers
SERVERS_VAR = 'PXT_UDF_SERVERS'
HEADERS_VAR = 'PXT_UDF_SERVERS_HEADERS'

# the name of the HTTP header with the secret that UdfServer requires of a call
UDF_SECRET_HEADER_NAME = 'Pxt-Udf-Secret'

# serializes the calls of _load_endpoints()
_load_lock = threading.Lock()


@dataclasses.dataclass(frozen=True)
class UdfCallRequest:
    """One call to a UDF, as sent to UdfServer.run().

    As in FnCallArgs, a call has either args and kwargs, for one row, or batch_args and batch_kwargs, with one value
    per row, for a batched UDF.
    """

    # UdfServer.run() calls the UDF's overload with this signature
    signature: func.Signature
    # the name of the calling process's default time zone (Env.default_time_zone)
    time_zone: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    batch_args: list[list[Any]] | None = None
    batch_kwargs: dict[str, list[Any]] | None = None

    def encode(self, fn_call: exprs.FunctionCall) -> bytes:
        # self.signature gives no column type to a value passed through *args or **kwargs
        arg_types = [arg.col_type for arg in fn_call.args]
        kwarg_types = {name: arg.col_type for name, arg in fn_call.kwargs.items()}
        encode_media = proxy_protocol.encode_media_files
        wire: dict[str, Any] = {'signature': self.signature.as_dict(), 'time_zone': self.time_zone}
        if self.batch_args is not None:
            assert self.batch_kwargs is not None
            wire['batch_args'] = [
                [encode_media(val, arg_types[i]) for val in vals] for i, vals in enumerate(self.batch_args)
            ]
            wire['batch_kwargs'] = {
                name: [encode_media(val, kwarg_types[name]) for val in vals] for name, vals in self.batch_kwargs.items()
            }
        else:
            assert self.args is not None and self.kwargs is not None
            wire['args'] = [encode_media(val, arg_types[i]) for i, val in enumerate(self.args)]
            wire['kwargs'] = {name: encode_media(val, kwarg_types[name]) for name, val in self.kwargs.items()}
        return _encode_body(wire)

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        return cls(
            func.Signature.from_dict(wire['signature']),
            wire['time_zone'],
            wire.get('args'),
            wire.get('kwargs'),
            wire.get('batch_args'),
            wire.get('batch_kwargs'),
        )


@dataclasses.dataclass(frozen=True)
class UdfCallResponse:
    """The result of a UdfServer.run() call."""

    # one value per row of the request; None if error is set
    values: list[Any] | None
    # the exception raised by the UDF, which becomes the error of every row of the request
    error: Exception | None
    # how long the UDF ran in the container
    duration_seconds: float
    # the provider's ID of the container that ran the call
    container_id: str | None

    def encode(self, return_type: ts.ColumnType) -> bytes:
        values = self.values
        if values is not None:
            values = [proxy_protocol.encode_media_files(val, return_type) for val in values]
        return _encode_body(
            {
                'values': values,
                'error': None if self.error is None else self._pickled_error(),
                'duration_seconds': self.duration_seconds,
                'container_id': self.container_id,
            }
        )

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        error = None if wire['error'] is None else pickle.loads(wire['error'])
        return cls(wire['values'], error, wire['duration_seconds'], wire['container_id'])

    def _pickled_error(self) -> bytes:
        """Pickle error, or a RuntimeError with its type and message if error does not survive pickling."""
        assert self.error is not None
        try:
            data = pickle.dumps(self.error)
            pickle.loads(data)
        except Exception:
            data = pickle.dumps(RuntimeError(f'{type(self.error).__name__}: {self.error}'))
        return data


def _encode_body(wire: dict[str, Any]) -> bytes:
    sink = proxy_protocol.InlinePartSink()
    head = json.dumps(proxy_protocol.serialize_args(wire, sink), separators=(',', ':')).encode()
    return proxy_protocol.encode_body(head, sink.binary_parts)


def _decode_body(body: bytes) -> dict[str, Any]:
    head, parts = proxy_protocol.decode_body(body)
    return proxy_protocol.deserialize_value(json.loads(head), parts)


def udf_endpoints() -> UdfEndpoints:
    """Return the process's endpoints of the UDF servers, looked up if Env has none."""
    env = Env.get()
    endpoints = env.udf_endpoints
    if endpoints is None:
        with _load_lock:
            endpoints = env.udf_endpoints
            if endpoints is None:
                endpoints = _load_endpoints()
                env.udf_endpoints = endpoints
    return endpoints


def _load_endpoints() -> UdfEndpoints:
    """Return the endpoints in the environment, or the local database's endpoints from local_db_udf_servers()."""
    if os.environ.get(SERVERS_VAR, '') != '':
        return UdfEndpoints(_env_object(SERVERS_VAR), _env_object(HEADERS_VAR))
    hosted_db = Env.get().hosted_db()
    if hosted_db is not None:
        org, db_name = hosted_db
        # the control plane sets SERVERS_VAR in the pods of a release with UDF servers
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            "The `pixeltable.udf` setting has no effect here: the database's release was deployed without UDF "
            f'servers ({SERVERS_VAR} is not set). Run `pxt db update pxt://{org}:{db_name}` to deploy them.',
        )
    servers = db.local_db_udf_servers()
    return UdfEndpoints(servers.urls, servers.headers)


def _env_object(var: str) -> dict[str, str]:
    """Return the JSON object of strings in the environment variable var."""
    text = os.environ.get(var, '')
    if text == '':
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, f'{var} is not set in the environment')
    invalid = excs.RequestError(
        excs.ErrorCode.INVALID_CONFIGURATION, f'{var} in the environment is not a JSON object of strings'
    )
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise invalid from None
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        raise invalid
    return value
