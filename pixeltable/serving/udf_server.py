"""An HTTP server that runs the calls to one UDF. Its client is pixeltable.exec.expr_eval.modal_client."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from pixeltable import exceptions as excs, func
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.exec.expr_eval.modal_client import UdfCallRequest, UdfCallResponse
from pixeltable.func.symbols import resolve_symbol
from pixeltable.utils.local_store import TempStore


class UdfServer:
    """Runs one UDF's calls, one at a time."""

    udf_path: str
    overloads: list[func.CallableFunction]
    # the provider's ID of the container that runs this server
    container_id: str | None
    lock: threading.Lock

    def __init__(self, udf_path: str, container_id: str | None, project_dir: Path | None = None) -> None:
        """project_dir: the root of the unpacked project, with its config file"""
        if project_dir is not None:
            Config.init(reinit=True, project_root=project_dir)
        # initializes the Env now instead of in the first call
        Env.get()  # doesn't require a local database
        self.udf_path = udf_path
        fn = resolve_symbol(udf_path)
        assert isinstance(fn, func.CallableFunction)
        self.overloads = fn._resolved_fns
        self.container_id = container_id
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
        request_body = bytearray()
        while True:
            message = await receive()
            request_body += message.get('body', b'')
            if not message.get('more_body', False):
                break
        if scope['method'] != 'POST':
            status, body = 404, b''
        else:
            try:
                status, body = 200, await asyncio.to_thread(self.run, bytes(request_body))
            except Exception as exc:
                status, body = 500, f'{type(exc).__name__}: {exc}'.encode()
        await send(
            {'type': 'http.response.start', 'status': status, 'headers': [(b'content-length', str(len(body)).encode())]}
        )
        await send({'type': 'http.response.body', 'body': body})

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
                        f"The server's version of {self.udf_path!r} has no signature {request.signature}",
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
