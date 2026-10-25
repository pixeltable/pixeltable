"""The event loop that runs async UDFs, and what the backend asks Python about failed calls.

The backend (`exec::dispatch`) decides which calls run and when; this module only provides the asyncio side: one
persistent event loop on a dedicated daemon thread, to which the backend hands coroutines in groups (`submit()`)
and which reports each finished task to a backend callback. `run_coroutine()` runs a single coroutine to completion
for the rare call that is not part of a batch.

Every coroutine runs wrapped in `_guarded()`, which turns its outcome into a value, also when it raises a
`BaseException` (`KeyboardInterrupt`, `SystemExit`, a `CancelledError` of its own): asyncio would let such an
exception escape the loop and stop it, and the task's waiter would never hear of it. The loop persists until
`shutdown()` (at exit, by the test hooks, and in a forked child, which starts a new one); a later call starts a new
one.
"""

from __future__ import annotations

import asyncio
import atexit
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

# the loop runs `_beat()` every `_BEAT_INTERVAL` seconds: a beat that is long overdue means a callback (a UDF that
# blocks instead of awaiting) holds the loop
_BEAT_INTERVAL = 1.0
_last_beat = 0.0
# how long the loop may be held before a dispatcher waiting for it warns (`report_stall()`)
_STALL_WARNING_S = 30.0


def _mark_loop_thread() -> None:
    # nested evaluation from this thread must not wait for the persistent loop (see `exec::dispatch`)
    from pixeltable import _core

    _core._mark_udf_loop_thread()


def _beat(loop: asyncio.AbstractEventLoop) -> None:
    global _last_beat
    _last_beat = time.monotonic()
    loop.call_later(_BEAT_INTERVAL, _beat, loop)


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    _mark_loop_thread()
    asyncio.set_event_loop(loop)
    loop.call_soon(_beat, loop)
    loop.call_soon(ready.set)
    while True:
        try:
            loop.run_forever()
        except BaseException:  # noqa: S110
            # raised by a callback (task outcomes are values, see _guarded()); the loop itself is intact
            pass
        with _lock:
            # stopped by shutdown(), or by user code: keep serving the tasks of other callers unless shut down
            if _loop is not loop or loop.is_closed():
                return


def get_loop() -> asyncio.AbstractEventLoop:
    """The loop async UDFs run on (started on first use, again after shutdown(), and in a forked child)."""
    with _lock:
        return _get_loop_locked()


def _get_loop_locked() -> asyncio.AbstractEventLoop:
    global _loop, _thread, _pid
    if _loop is None or _pid != os.getpid() or _thread is None or not _thread.is_alive():
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        thread = threading.Thread(target=_run_loop, args=(loop, ready), name='pixeltable-async-udfs', daemon=True)
        thread.start()
        ready.wait()
        _loop, _thread, _pid = loop, thread, os.getpid()
    return _loop


def loop_alive() -> bool:
    """False if the loop that tasks were submitted to stopped running (its thread ended)."""
    return _thread is not None and _thread.is_alive() and _pid == os.getpid()


def report_stall() -> bool:
    """Logs a warning if the loop has not run for more than `_STALL_WARNING_S` seconds (a UDF blocks it); returns True
    if it did."""
    stalled = time.monotonic() - _last_beat - _BEAT_INTERVAL
    if not loop_alive() or stalled <= _STALL_WARNING_S:
        return False
    from pixeltable import _console

    # a user-visible warning (upstream's console logger: stdout)
    _console.warning(
        f'Async UDF calls are waiting for the event loop that runs them, which has not run for {stalled:.0f} s: '
        'an async UDF blocks it (a synchronous call that does not await)'
    )
    return True


def in_loop_thread() -> bool:
    """True when called from the thread that runs the loop (a blocking wait for the loop there would deadlock)."""
    return _thread is not None and threading.current_thread() is _thread


async def _guarded(coro: Coroutine[Any, Any, Any]) -> tuple[bool, Any]:
    """`(True, result)` or `(False, exception)`; a cancellation of the task itself (`cancel()`) propagates."""
    try:
        return True, await coro
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling() > 0:
            raise
        return False, exc
    except BaseException as exc:
        return False, exc


def _unwrap(outcome: tuple[bool, Any]) -> Any:
    ok, value = outcome
    if not ok:
        raise value
    return value


def new_group() -> set[asyncio.Task]:
    """A set that collects the tasks of one dispatch, so that they can be cancelled together."""
    return set()


def submit(items: list[tuple[Any, Coroutine[Any, Any, Any]]], done: Callable[[Any, asyncio.Task], None], group: set) -> None:
    """Starts a task for each (token, coroutine) on the loop; `done(token, task)` is called when a task finishes, and
    the task's result is `_guarded()`'s."""
    with _lock:
        # under the lock, so that shutdown() sees these tasks
        _get_loop_locked().call_soon_threadsafe(_start, items, done, group)


def _start(items: list[tuple[Any, Coroutine[Any, Any, Any]]], done: Callable[[Any, asyncio.Task], None], group: set) -> None:
    loop = asyncio.get_running_loop()
    for token, coro in items:
        task = loop.create_task(_guarded(coro))
        group.add(task)
        task.add_done_callback(group.discard)
        task.add_done_callback(functools.partial(done, token))


def cancel(group: set) -> None:
    """Cancels the unfinished tasks of a group."""
    loop = _loop
    if loop is not None and not loop.is_closed():
        try:
            loop.call_soon_threadsafe(_cancel, group)
        except RuntimeError:  # closed meanwhile
            pass


def _cancel(group: set) -> None:
    for task in list(group):
        task.cancel()


def _run_isolated(coro: Coroutine[Any, Any, Any]) -> Any:
    _mark_loop_thread()
    return asyncio.run(coro)


def run_coroutine(coro: Coroutine[Any, Any, Any], isolated: bool = False) -> Any:
    """Runs `coro` to completion and returns its result: on the persistent loop, or, when the caller must not wait for
    that loop (`isolated`: it runs a loop itself, perhaps that one, or holds the catalog lock), on a temporary loop in
    another thread."""
    guarded = _guarded(coro)
    if isolated or in_loop_thread() or asyncio._get_running_loop() is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return _unwrap(pool.submit(_run_isolated, guarded).result())
    return _unwrap(asyncio.run_coroutine_threadsafe(guarded, get_loop()).result())


def shutdown(only_if_idle: bool = False) -> bool:
    """Stops and closes the persistent loop (cancelling its unfinished tasks, unless `only_if_idle`, which leaves a busy
    loop alone); the next async call starts a new one. Returns False if the loop was left running."""
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        if loop is None or thread is None or _pid != os.getpid():
            _loop = _thread = None
            return True
        if thread is threading.current_thread():
            return False
        if thread.is_alive() and not loop.is_closed():
            try:
                pending = asyncio.run_coroutine_threadsafe(_pending_tasks(), loop).result(timeout=5)
            except Exception:
                pending = 0
            if pending > 0 and only_if_idle:
                return False
        _loop = _thread = None
    if thread.is_alive() and not loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(_finish(), loop).result(timeout=10)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
    if not thread.is_alive() and not loop.is_closed():
        loop.close()
    return True


async def _pending_tasks() -> int:
    me = asyncio.current_task()
    return sum(1 for t in asyncio.all_tasks() if t is not me and not t.done())


async def _finish() -> None:
    """Cancels the loop's other tasks and waits for them, then shuts down async generators and the default executor."""
    me = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    loop = asyncio.get_running_loop()
    await loop.shutdown_asyncgens()
    await loop.shutdown_default_executor()


atexit.register(shutdown)


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

