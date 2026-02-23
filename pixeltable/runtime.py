"""Per-thread execution context for concurrent Pixeltable access.

Each thread gets its own Runtime instance containing a Catalog and DB connection state.
Process-level resources (engine, paths, logging, config) stay shared on Env.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator

import sqlalchemy as sql
from rich.progress import Progress
from sqlalchemy import orm

if TYPE_CHECKING:
    from pixeltable.catalog.catalog import Catalog

_logger = logging.getLogger('pixeltable')
_thread_local = threading.local()

SERIALIZABLE_ISOLATION_LEVEL = 'SERIALIZABLE'


class Runtime:
    """
    Per-thread execution context. Each thread gets its own Runtime instance.
    """

    _catalog: Catalog | None
    conn: sql.Connection | None
    session: orm.Session | None
    isolation_level: str | None
    _progress: Progress | None

    def __init__(self) -> None:
        # Catalog is created lazily to avoid circular initialization:
        # Catalog.__init__() calls _init_store() which needs begin_xact() which calls get_runtime().
        self._catalog = None
        self.conn = None
        self.session = None
        self.isolation_level = None
        self._progress = None

    @property
    def in_xact(self) -> bool:
        return self.conn is not None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            from pixeltable.catalog.catalog import Catalog

            self._catalog = Catalog()
        return self._catalog

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
    if runtime is not None and runtime._catalog is not None:
        # Invalidate all existing TableVersion instances to force reloading of metadata,
        for tbl_version in runtime._catalog._tbl_versions.values():
            tbl_version.is_validated = False
    _thread_local.runtime = Runtime()
