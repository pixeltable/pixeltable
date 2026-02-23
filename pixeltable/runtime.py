"""Per-thread execution context for concurrent Pixeltable access.

Each thread gets its own Runtime instance containing a Catalog and DB connection state.
Process-level resources (engine, paths, logging, config) stay shared on Env.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import sqlalchemy as sql
from sqlalchemy import orm

if TYPE_CHECKING:
    from pixeltable.catalog.catalog import Catalog

_thread_local = threading.local()


class Runtime:
    """Per-thread execution context. Each thread gets its own Runtime instance."""

    def __init__(self) -> None:
        # Catalog is created lazily to avoid circular initialization:
        # Catalog.__init__() calls _init_store() which needs Env.begin_xact() which calls get_runtime().
        self._catalog: Catalog | None = None
        # Connection/transaction state (per-thread, moved from Env)
        self.conn: sql.Connection | None = None
        self.session: orm.Session | None = None
        self.isolation_level: str | None = None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            from pixeltable.catalog.catalog import Catalog

            self._catalog = Catalog()
        return self._catalog


def get_runtime() -> Runtime:
    """Get the current thread's Runtime, creating one if needed."""
    runtime = getattr(_thread_local, 'runtime', None)
    if runtime is None:
        runtime = Runtime()
        _thread_local.runtime = runtime
    return runtime


def reset_runtime() -> None:
    """Replace the current thread's Runtime with a fresh one. Used for testing."""
    old_runtime = getattr(_thread_local, 'runtime', None)
    if old_runtime is not None and old_runtime._catalog is not None:
        # Invalidate all existing TableVersion instances to force reloading of metadata,
        # matching the behavior of the old Catalog.clear().
        for tbl_version in old_runtime._catalog._tbl_versions.values():
            tbl_version.is_validated = False
    _thread_local.runtime = Runtime()
