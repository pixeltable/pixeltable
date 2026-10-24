"""Running coroutines returned by async UDFs (called by the backend)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Coroutine

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
        return _loop


def run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    """Runs `coro` to completion and returns its result (in a separate thread if this thread runs an event loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _get_loop().run_until_complete(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
