from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, TypeVar

import nest_asyncio  # type: ignore[import-untyped]
import sqlalchemy as sql
from rich.progress import Progress
from sqlalchemy import orm

from pixeltable.env import Env

if TYPE_CHECKING:
    from pixeltable.catalog.catalog import Catalog

_logger = logging.getLogger('pixeltable')
_thread_local = threading.local()

SERIALIZABLE_ISOLATION_LEVEL = 'SERIALIZABLE'

_T = TypeVar('_T')


class Runtime:
    """
    Global context for a thread executing Pixeltable API calls.

    All state that cannot be shared process-wide (and would therefore be located in Env) is stored here.
    """

    _catalog: Catalog | None
    conn: sql.Connection | None
    session: orm.Session | None
    isolation_level: str | None
    _progress: Progress | None
    _event_loop: asyncio.AbstractEventLoop | None
    _run_coro_executor: concurrent.futures.ThreadPoolExecutor

    def __init__(self) -> None:
        # Catalog is created lazily to avoid circular initialization:
        # Catalog.__init__() calls _init_store() which needs begin_xact() which calls get_runtime().
        self._catalog = None
        self.conn = None
        self.session = None
        self.isolation_level = None
        self._progress = None
        self._event_loop = None
        self._run_coro_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def in_xact(self) -> bool:
        return self.conn is not None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            from pixeltable.catalog.catalog import Catalog

            self._catalog = Catalog()
        return self._catalog

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        if self._event_loop is None:
            self._init_event_loop()
        return self._event_loop

    def _init_event_loop(self) -> None:
        try:
            # check if we are already in an event loop (eg, Jupyter's); if so, patch it to allow
            # multiple run_until_complete()
            running_loop = asyncio.get_running_loop()
            self._event_loop = running_loop
            _logger.debug('Patched running loop')
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            # we set a deliberately long duration to avoid warnings getting printed to the console in debug mode
            self._event_loop.slow_callback_duration = 3600

        # TODO: remove unconditional nest_asyncio.apply()
        if True or Env.get().is_notebook():  # noqa: SIM222
            # Jupyter notebooks have their own event loop, which we need to patch to allow nested run_until_complete()
            nest_asyncio.apply(self._event_loop)
        if _logger.isEnabledFor(logging.DEBUG):
            self._event_loop.set_debug(True)

    def run_coro(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine synchronously in a separate thread with its own event loop."""
        return self._run_coro_executor.submit(asyncio.run, coro).result()

    @contextmanager
    def begin_xact(self, *, for_write: bool = False) -> Iterator[sql.Connection]:
        """Start or join a database transaction.

        Prefer Catalog.begin_xact() unless there is a specific reason to call this directly.

        Args:
            for_write: if True, uses serializable isolation; if False, uses repeatable_read

        TODO: repeatable read is not available in Cockroachdb; instead, run queries against a snapshot TVP
        that avoids tripping over any pending ops
        """
        from pixeltable.env import Env

        if not self.in_xact:
            assert self.session is None
            try:
                self.isolation_level = SERIALIZABLE_ISOLATION_LEVEL
                with (
                    Env.get().engine.connect().execution_options(isolation_level=self.isolation_level) as conn,
                    orm.Session(conn) as session,
                    conn.begin(),
                ):
                    self.conn = conn
                    self.session = session
                    yield conn
            finally:
                self.session = None
                self.conn = None
                self.isolation_level = None
        else:
            assert self.session is not None
            assert self.isolation_level == SERIALIZABLE_ISOLATION_LEVEL or not for_write
            yield self.conn

    def start_progress(self, create_fn: Callable[[], Progress]) -> Progress:
        if self._progress is None:
            self._progress = create_fn()
            self._progress.start()
        return self._progress

    def stop_progress(self) -> None:
        from pixeltable.env import Env

        if self._progress is None:
            return
        try:
            self._progress.stop()
        except Exception as e:
            _logger.warning(f'Error stopping progress: {e}')
        finally:
            self._progress = None

        # if we're running in a notebook, we need to clear the Progress output manually
        if Env.get().is_notebook():
            try:
                from IPython.display import clear_output

                clear_output(wait=False)
            except ImportError:
                pass

    @contextmanager
    def report_progress(self) -> Iterator[None]:
        """Context manager for the Progress instance."""
        try:
            yield
        finally:
            self.stop_progress()


def get_runtime() -> Runtime:
    """Get the current thread's Runtime instance, creating one if needed."""
    runtime = getattr(_thread_local, 'runtime', None)
    if runtime is None:
        runtime = Runtime()
        _thread_local.runtime = runtime
    return runtime


def reset_runtime() -> None:
    """Reset the current thread's Runtime instance. Used for testing."""
    runtime = getattr(_thread_local, 'runtime', None)
    if runtime is not None:
        if runtime._catalog is not None:
            # Invalidate all existing TableVersion instances to force reloading of metadata,
            for tbl_version in runtime._catalog._tbl_versions.values():
                tbl_version.is_validated = False
        if runtime._event_loop is not None:
            # Don't close a loop we didn't create (e.g. Jupyter's)
            try:
                loop = runtime._event_loop
                if not loop.is_running():
                    loop.close()
            except Exception:
                pass
        runtime._run_coro_executor.shutdown(wait=False)
    _thread_local.runtime = Runtime()
