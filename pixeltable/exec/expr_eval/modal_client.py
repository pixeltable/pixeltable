"""The calls to the UDFs in udf_resources, which run on Modal Servers (pixeltable.serving.modal_server)."""

import asyncio
import dataclasses
import json
import logging
import pickle
import time
from typing import Any

import httpx
from typing_extensions import Self

from pixeltable import exceptions as excs, func, type_system as ts
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.service import proxy_protocol

from .interfaces import FnCallArgs

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass(frozen=True)
class UdfCallRequest:
    """The arguments of a ModalServer.run() call, shaped as in FnCallArgs."""

    # selects the overload of the server's UDF
    signature: func.Signature
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    batch_args: list[list[Any]] | None = None
    batch_kwargs: dict[str, list[Any]] | None = None

    def encode(self) -> bytes:
        params = self.signature.parameters
        params_by_pos = self.signature.parameters_by_pos
        encode_media = proxy_protocol.encode_media_files
        wire: dict[str, Any] = {'signature': self.signature.as_dict()}
        if self.batch_args is not None:
            assert self.batch_kwargs is not None
            wire['batch_args'] = [
                [encode_media(val, params_by_pos[i].col_type) for val in vals] for i, vals in enumerate(self.batch_args)
            ]
            wire['batch_kwargs'] = {
                name: [encode_media(val, params[name].col_type) for val in vals]
                for name, vals in self.batch_kwargs.items()
            }
        else:
            assert self.args is not None and self.kwargs is not None
            wire['args'] = [encode_media(val, params_by_pos[i].col_type) for i, val in enumerate(self.args)]
            wire['kwargs'] = {name: encode_media(val, params[name].col_type) for name, val in self.kwargs.items()}
        return _encode_body(wire)

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        return cls(
            func.Signature.from_dict(wire['signature']),
            wire.get('args'),
            wire.get('kwargs'),
            wire.get('batch_args'),
            wire.get('batch_kwargs'),
        )


@dataclasses.dataclass(frozen=True)
class UdfCallResponse:
    """The result of a ModalServer.run() call."""

    # one per row; None if error is set
    values: list[Any] | None
    # the UDF's exception, which fails every row
    error: Exception | None
    # run time of the UDF
    duration_seconds: float
    # the Modal container that ran the call
    task_id: str | None

    def encode(self, return_type: ts.ColumnType) -> bytes:
        values = self.values
        if values is not None:
            values = [proxy_protocol.encode_media_files(val, return_type) for val in values]
        return _encode_body(
            {
                'values': values,
                'error': None if self.error is None else self._pickled_error(),
                'duration_seconds': self.duration_seconds,
                'task_id': self.task_id,
            }
        )

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        error = None if wire['error'] is None else pickle.loads(wire['error'])
        return cls(wire['values'], error, wire['duration_seconds'], wire['task_id'])

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


# how long a call waits for a container with room while the Server answers 503
_CONTAINER_WAIT_SECONDS = 300


@register_client('modal', credential_param='proxy_secret')
def _(proxy_key: str, proxy_secret: str) -> httpx.AsyncClient:
    # HTTP/2 sends concurrent calls over one connection
    return httpx.AsyncClient(http2=True, headers={'Modal-Key': proxy_key, 'Modal-Secret': proxy_secret})


# ModalApp's key in Env.get_resource_pool_info()
_APP_REGISTRY_KEY = 'modal-app'


class ModalServerClient:
    """The process's client for one UDF's Modal Server."""

    udf_path: str
    timeout_seconds: int
    # set by connect()
    url: str | None

    def __init__(self, udf_path: str, timeout_seconds: int) -> None:
        self.udf_path = udf_path
        self.timeout_seconds = timeout_seconds
        self.url = None

    async def connect(self) -> None:
        """Look up the Server's URL."""
        if self.url is None:
            # raises MISSING_CREDENTIALS without a proxy token, before a deploy
            get_runtime().get_client('modal')
            Env.get().require_package('modal')
            from pixeltable.serving.modal_server import ModalApp  # modal is an optional package

            app = Env.get().get_resource_pool_info(_APP_REGISTRY_KEY, ModalApp)
            # a deploy takes seconds to minutes and calls Modal's blocking API
            self.url = await asyncio.to_thread(app.server_url, self.udf_path)

    async def call(self, call_args: FnCallArgs) -> UdfCallResponse:
        """Run call_args on the Server."""
        assert self.url is not None
        request_body = UdfCallRequest(
            call_args.pxt_fn.signature, call_args.args, call_args.kwargs, call_args.batch_args, call_args.batch_kwargs
        ).encode()
        http_client: httpx.AsyncClient = get_runtime().get_client('modal')
        start = time.monotonic()
        response = await http_client.post(self.url, content=request_body, timeout=self.timeout_seconds)
        attempt = 0
        # TODO: detect a Server whose containers fail in ModalServer.start(), instead of waiting _CONTAINER_WAIT_SECONDS
        while response.status_code == 503 and time.monotonic() - start < _CONTAINER_WAIT_SECONDS:
            await asyncio.sleep(min(1.0, 0.1 * 2**attempt))
            attempt += 1
            # probe with a GET, which has no body to upload again
            if (await http_client.get(self.url, timeout=self.timeout_seconds)).status_code != 503:
                response = await http_client.post(self.url, content=request_body, timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Modal call of {self.udf_path} failed with status {response.status_code}: {response.text}',
            )
        result = UdfCallResponse.decode(response.content)
        _logger.debug(
            f'Modal call of {self.udf_path}: {len(call_args.rows)} rows in {time.monotonic() - start:.2f}s, '
            f'UDF {result.duration_seconds:.2f}s, container {result.task_id}'
        )
        return result
