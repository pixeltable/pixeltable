"""The event loop that runs async UDFs, and what the backend asks Python about failed calls.

The backend (`exec::dispatch`) decides which calls run and when; this module only provides the asyncio side: one
persistent event loop on a dedicated daemon thread, to which the backend hands coroutines in groups (`submit()`)
and which reports each finished task to a backend callback. `run_coroutine()` runs a single coroutine to completion
for the rare call that is not part of a batch.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
import re
import threading
import time
import urllib.error
from http import HTTPStatus
from typing import Any, Callable, Coroutine

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_pid: int | None = None
_lock = threading.Lock()


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()


def get_loop() -> asyncio.AbstractEventLoop:
    """The loop async UDFs run on (started on first use, and again in a forked child)."""
    global _loop, _thread, _pid
    with _lock:
        if _loop is None or _pid != os.getpid() or _thread is None or not _thread.is_alive():
            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(target=_run_loop, args=(loop, ready), name='pixeltable-async-udfs', daemon=True)
            thread.start()
            ready.wait()
            _loop, _thread, _pid = loop, thread, os.getpid()
        return _loop


def in_loop_thread() -> bool:
    """True when called from the thread that runs the loop (a blocking wait for the loop there would deadlock)."""
    return _thread is not None and threading.current_thread() is _thread


def new_group() -> set[asyncio.Task]:
    """A set that collects the tasks of one dispatch, so that they can be cancelled together."""
    return set()


def submit(items: list[tuple[Any, Coroutine[Any, Any, Any]]], done: Callable[[Any, asyncio.Task], None], group: set) -> None:
    """Starts a task for each (token, coroutine) on the loop; `done(token, task)` is called when a task finishes."""
    get_loop().call_soon_threadsafe(_start, items, done, group)


def _start(items: list[tuple[Any, Coroutine[Any, Any, Any]]], done: Callable[[Any, asyncio.Task], None], group: set) -> None:
    loop = asyncio.get_running_loop()
    for token, coro in items:
        task = loop.create_task(coro)
        group.add(task)
        task.add_done_callback(group.discard)
        task.add_done_callback(functools.partial(done, token))


def cancel(group: set) -> None:
    """Cancels the unfinished tasks of a group."""
    if _loop is not None and not _loop.is_closed():
        _loop.call_soon_threadsafe(_cancel, group)


def _cancel(group: set) -> None:
    for task in list(group):
        task.cancel()


def run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    """Runs `coro` to completion on the loop and returns its result (from the loop's own thread, which must not
    block on the loop, on a temporary loop in another thread)."""
    if in_loop_thread():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result()


# ------------------------------------------------------------------------------------------------ retries
# A port of upstream's `pixeltable.utils.http.is_retryable_error()`: whether a failed call of a function in a resource
# pool is worth retrying, judged from the attributes and the message of the exception.

_RETRIABLE_ERROR_INDICATORS = (
    'rate limit',
    'too many requests',
    '429',
    'quota exceeded',
    'throttled',
    'rate exceeded',
    'connection error',
    'timed out',
)
_RETRY_AFTER_PATTERNS = (
    r'retry after (\d+(?:\.\d+)?)\s*seconds?',
    r'try again in (\d+(?:\.\d+)?)\s*seconds?',
    r'wait (\d+(?:\.\d+)?)\s*seconds?',
    r'retry-after:\s*(\d+(?:\.\d+)?)',
)
_RETRIABLE_HTTP_STATUSES: dict[str, int] = {
    'TOO_MANY_REQUESTS': HTTPStatus.TOO_MANY_REQUESTS.value,
    'SERVICE_UNAVAILABLE': HTTPStatus.SERVICE_UNAVAILABLE.value,
    'REQUEST_TIMEOUT': HTTPStatus.REQUEST_TIMEOUT.value,
    'GATEWAY_TIMEOUT': HTTPStatus.GATEWAY_TIMEOUT.value,
}


def retry_info(exc: BaseException) -> tuple[bool, float | None]:
    """(is retryable, the delay the error asks for in seconds, if any)."""
    if not isinstance(exc, Exception):
        return False, None
    err_md = _error_metadata(exc)
    if (err_md is None or not err_md[0]) and hasattr(exc, 'response'):
        err_md = _error_metadata(exc.response)
    if err_md is not None and err_md[0]:
        retry_after = err_md[1]
        return True, retry_after if retry_after is not None and retry_after >= 0 else None
    if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)) and not isinstance(
        exc, urllib.error.HTTPError
    ):
        return True, None
    msg = str(exc).lower()
    if any(indicator in msg for indicator in _RETRIABLE_ERROR_INDICATORS):
        delay = _delay_from_message(msg)
        return True, delay if delay is not None and delay >= 0 else None
    return False, None


def _error_metadata(obj: Any) -> tuple[bool, float | None] | None:
    is_retryable: bool | None = None
    delay: float | None = None
    for attr in ('status', 'code', 'status_code'):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            is_retryable = val in _RETRIABLE_HTTP_STATUSES.values() or str(val).upper() in _RETRIABLE_HTTP_STATUSES
    if hasattr(obj, 'headers'):
        delay = _delay_from_headers(obj.headers)
        if delay is not None:
            is_retryable = True
    return (is_retryable, delay) if is_retryable is not None else None


def _delay_from_headers(headers: Any) -> float | None:
    if headers is None:
        return None
    if hasattr(headers, 'get'):
        header_dict = headers
    else:
        try:
            header_dict = dict(headers)
        except (TypeError, ValueError):
            return None
    header_dict = {str(k).lower().replace('-', ''): v for k, v in header_dict.items()}
    for key, absolute in (('retryafter', False), ('xratelimitreset', True), ('xratelimitresetafter', False)):
        val = header_dict.get(key)
        if val is None:
            continue
        try:
            return max(0.0, float(val) - time.time()) if absolute else float(val)
        except (ValueError, TypeError):
            continue
    return None


def _delay_from_message(msg: str) -> float | None:
    for pattern in _RETRY_AFTER_PATTERNS:
        match = re.search(pattern, msg)
        if match is not None:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                continue
    return None

