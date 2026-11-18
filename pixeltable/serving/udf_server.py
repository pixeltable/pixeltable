"""An HTTP server that runs the calls to one UDF. Its client is pixeltable.exec.expr_eval.modal_client."""

import asyncio
import hmac
import threading
import time
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from pixeltable import exceptions as excs, func
from pixeltable.env import Env
from pixeltable.func.symbols import resolve_symbol
from pixeltable.service.udf_servers import UDF_SECRET_HEADER_NAME, UdfCallRequest, UdfCallResponse
from pixeltable.utils.local_store import TempStore


class UdfServer:
    """Runs one UDF's calls, one at a time."""

    setting_path: str  # udf's path as used in the config setting for which this server is created
    overloads: list[func.CallableFunction]
    # the provider's ID of the container that runs this server
    container_id: str | None
    # a POST must have this value in UDF_SECRET_HEADER_NAME
    secret: str
    lock: threading.Lock

    def __init__(self, setting_path: str, container_id: str | None, secret: str) -> None:
        # initializes the Env now instead of in the first call
        Env.get()  # doesn't require a local database
        self.setting_path = setting_path
        fn = resolve_symbol(setting_path)
        assert isinstance(fn, func.CallableFunction)
        self.overloads = fn._resolved_fns
        self.container_id = container_id
        # an empty secret would equal the value of a missing header
        assert secret != ''
        self.secret = secret
        self.lock = threading.Lock()

    def start(self, port: int) -> None:
        """Serve the UDF's calls on port, in a thread of its own."""
        # only a serving process needs hypercorn
        import hypercorn.asyncio
        import hypercorn.config

        config = hypercorn.config.Config()
        config.bind = [f'0.0.0.0:{port}']
        # with a shutdown_trigger, hypercorn doesn't install signal handlers, which Python allows only in the main
        # thread
        serve = hypercorn.asyncio.serve(self.handle, config, shutdown_trigger=asyncio.Event().wait)  # type: ignore[arg-type]
        threading.Thread(target=asyncio.run, args=(serve,), daemon=True).start()

    async def handle(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Answer a POST with the result of run() on its body."""
        if scope['type'] != 'http':
            return
        # the checks come before reading the body, so a caller without the secret can't make the server buffer one
        if scope['method'] != 'POST':
            # GETs come from Modal, which polls /metrics, and from ModalServerClient._monitor(), which probes for a
            # running container
            status, body = 404, b''
        elif not self._is_authorized(scope):
            status, body = 401, b''
        else:
            request_body = await self._read_body(receive)
            try:
                status, body = 200, await asyncio.to_thread(self.run, request_body)
            except Exception as exc:
                status, body = 500, f'{type(exc).__name__}: {exc}'.encode()
        await send(
            {'type': 'http.response.start', 'status': status, 'headers': [(b'content-length', str(len(body)).encode())]}
        )
        await send({'type': 'http.response.body', 'body': body})

    async def _read_body(self, receive: Callable[[], Awaitable[dict[str, Any]]]) -> bytes:
        """Read the request body from the ASGI messages of receive()."""
        request_body = bytearray()
        while True:
            message = await receive()
            request_body += message.get('body', b'')
            if not message.get('more_body', False):
                return bytes(request_body)

    def _is_authorized(self, scope: dict[str, Any]) -> bool:
        """Whether the request with this ASGI scope has self.secret in its UDF_SECRET_HEADER_NAME header."""
        # scope['headers'] is a list of (name, value) pairs of bytes, with the names in lowercase
        header_name = UDF_SECRET_HEADER_NAME.lower().encode()
        sent = next((value for name, value in scope['headers'] if name == header_name), b'')
        # compare_digest() takes the same time for every wrong value, so a caller can't guess the secret from timings
        return hmac.compare_digest(sent, self.secret.encode())

    def run(self, request_body: bytes) -> bytes:
        """Run the UdfCallRequest in request_body and return the encoded UdfCallResponse."""
        with self.lock:
            try:
                request = UdfCallRequest.decode(request_body)
                # the UDF sees the caller's default time zone, as in a local run
                Env.get().default_time_zone = ZoneInfo(request.time_zone)
                fn = next((overload for overload in self.overloads if overload.signature == request.signature), None)
                if fn is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_STATE,
                        f"The server's version of {self.setting_path!r} has no signature {request.signature}",
                    )
                start = time.monotonic()
                values: list[Any] | None = None
                error: Exception | None = None
                try:
                    if request.batch_args is not None:
                        assert request.batch_kwargs is not None
                        values = fn.exec_batch(request.batch_args, request.batch_kwargs)
                    else:
                        assert request.args is not None and request.kwargs is not None
                        values = [fn.py_fn(*request.args, **request.kwargs)]
                except Exception as exc:
                    error = exc
                response = UdfCallResponse(values, error, time.monotonic() - start, self.container_id)
                return response.encode(fn.signature.get_return_type())
            finally:
                # self.lock lets only one call run at a time, so the TempStore contains only this call's files
                TempStore.clear()
