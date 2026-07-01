from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, Literal, TypeVar
from uuid import UUID
from weakref import WeakKeyDictionary

import sqlalchemy as sql
from rich.progress import Progress
from sqlalchemy import orm

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils import fault_injection

if TYPE_CHECKING:
    from pixeltable._query import Query
    from pixeltable.catalog.catalog import Catalog
    from pixeltable.catalog.catalog_base import CatalogBase
    from pixeltable.catalog.path import Path
    from pixeltable.catalog.table import Table
    from pixeltable.exec import ExecPlan

_logger = logging.getLogger(__name__)
_thread_local = threading.local()

_XACT_ISOLATION_LEVEL = 'REPEATABLE READ'

_T = TypeVar('_T')


class Runtime:
    """
    Global context for a thread executing Pixeltable API calls.

    All state that cannot be shared process-wide (and would therefore be located in Env) is stored here.
    """

    # catalogs keyed by their catalog_uri Path (ROOT_PATH = the in-process catalog)
    _catalogs: dict[Path, CatalogBase]

    conn: sql.Connection | None
    session: orm.Session | None
    isolation_level: str | None
    _progress: Progress | None
    _event_loop: asyncio.AbstractEventLoop | None  # event loop for this thread
    _run_coro_executor: concurrent.futures.ThreadPoolExecutor | None
    fault_manager: Any

    # True if this thread's runtime was populated from another thread via copy_db_context()
    context_inherited: bool

    # we need to cache client instances on a per-thread basis because some of them are thread-specific (eg, async
    # clients which are tied to an event loop)
    _clients: dict[str, Any]

    # Per-thread cache of compiled query plans, keyed by Query identity. Entries auto-expire when the Query is
    # garbage-collected.
    plan_cache: WeakKeyDictionary[Query, ExecPlan]

    def __init__(self) -> None:
        # Catalogs are created lazily to avoid circular initialization:
        # Catalog.__init__() calls _init_store() which needs begin_xact() which calls get_runtime().
        self._catalogs = {}
        self.conn = None
        self.session = None
        self.isolation_level = None
        self._progress = None
        self._event_loop = None
        self._run_coro_executor = None
        self._clients = {}
        self.context_inherited = False
        self.fault_manager = fault_injection.create_fault_manager()
        self.plan_cache = WeakKeyDictionary()

    def copy_db_context(self, other: Runtime) -> None:
        """Copy the db-related state from another Runtime instance."""
        self.conn = other.conn
        self.session = other.session
        self.isolation_level = other.isolation_level
        # share the same catalog instances, but with an independent map so each thread can add catalogs
        # without racing
        self._catalogs = dict(other._catalogs)
        self._progress = other._progress
        self.context_inherited = True
        self.fault_manager = other.fault_manager

    @property
    def in_xact(self) -> bool:
        return self.conn is not None

    @property
    def catalog(self) -> Catalog:
        """The local Catalog instance."""
        from pixeltable.catalog.catalog import Catalog
        from pixeltable.catalog.path import ROOT_PATH

        cat = self._catalogs.get(ROOT_PATH)
        if cat is None:
            cat = Catalog()
            self._catalogs[ROOT_PATH] = cat
        assert isinstance(cat, Catalog)
        return cat

    def get_catalog(self, path: Path) -> CatalogBase:
        """Return the catalog the given path lives in, creating it on first use."""
        from pixeltable.catalog.path import ROOT_PATH

        catalog_uri = path.catalog_uri
        if catalog_uri == ROOT_PATH:  # the in-process catalog
            return self.catalog
        cat = self._catalogs.get(catalog_uri)
        if cat is None:
            cat = self._make_proxy_catalog(catalog_uri)
            self._catalogs[catalog_uri] = cat
        return cat

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None:
        """Load the table with the given id, routing to whichever catalog owns it.

        The owning catalog is determined from the URI Env records when a table is first loaded; tables that
        haven't been seen yet resolve to the local catalog.
        """
        cat = self.get_catalog(Env.get().tbl_catalog_uri(tbl_id))
        return cat.get_table_by_id(tbl_id, version=version, ignore_if_dropped=ignore_if_dropped)

    def _make_proxy_catalog(self, catalog_uri: Path) -> CatalogBase:
        from pixeltable.catalog.catalog_proxy import CatalogProxy

        assert catalog_uri.db is not None

        if catalog_uri.org == 'local':
            from pixeltable.service import proxy_daemon
            from pixeltable.service.proxy_client import ProxyHttpClient

            info = proxy_daemon.read_port_lock(catalog_uri.db)
            if info is None:
                db = catalog_uri.db
                raise excs.NotFoundError(
                    excs.ErrorCode.SERVICE_NOT_FOUND,
                    f'No local proxy is running for {db!r}. Start it with: pxt localproxy start {db}',
                )
            return CatalogProxy(catalog_uri, ProxyHttpClient(f'http://127.0.0.1:{info["port"]}'))

        # Cloud-hosted database: connect via TLS tunnel to the proxy sidecar.
        import os
        from pixeltable.service.proxy_cloud_client import ProxyCloudClient

        api_key = Env.get().pxt_api_key
        if api_key is None:
            raise excs.AuthorizationError(
                excs.ErrorCode.MISSING_CREDENTIALS,
                f'A Pixeltable API key is required to connect to hosted database {catalog_uri!r}. '
                'Set PIXELTABLE_API_KEY or add api_key to your config.',
            )
        # PIXELTABLE_CLOUD_HOST is a domain suffix override (e.g. "dev.pxt.run").
        # The full host is composed as {org}-{db}.{domain}.  Omit to use the
        # production default (pxt.run), which proxy_cloud_client already encodes.
        cloud_domain = os.environ.get('PIXELTABLE_CLOUD_HOST') or None
        port_override = 9000
        if cloud_domain and ':' in cloud_domain:
            cloud_domain, _, port_str = cloud_domain.rpartition(':')
            port_override = int(port_str)
        host = f'{catalog_uri.org}-{catalog_uri.db}.{cloud_domain}' if cloud_domain else None
        no_verify = os.environ.get('PIXELTABLE_CLOUD_NO_VERIFY', '') in ('1', 'true', 'yes')
        return CatalogProxy(
            catalog_uri,
            ProxyCloudClient(
                catalog_uri.org, catalog_uri.db, api_key,
                host=host, port=port_override, no_verify=no_verify,
            ),
        )

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

        if _logger.isEnabledFor(logging.DEBUG):
            self._event_loop.set_debug(True)

    def get_client(self, name: str) -> Any:
        """Gets the client with the specified name, initializing it if necessary."""
        client = self._clients.get(name)
        if client is not None:
            return client
        client = Env.get().create_client(name)
        self._clients[name] = client
        return client

    def run_coro(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine synchronously in a separate thread with its own persistent event loop."""
        if self._run_coro_executor is None:
            self._run_coro_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def run(coro: Coroutine[Any, Any, _T]) -> _T:
            # this runs in the _run_coro_executor's thread, with its own Runtime instance
            return get_runtime().event_loop.run_until_complete(coro)

        return self._run_coro_executor.submit(run, coro).result()

    @contextmanager
    def begin_xact(
        self,
        *,
        for_write: bool = False,
        isolation_level: Literal['READ COMMITTED', 'REPEATABLE READ', 'SERIALIZABLE'] | None = None,
    ) -> Iterator[sql.Connection]:
        """Start or join a database transaction.

        Prefer Catalog.begin_xact() unless there is a specific reason to call this directly.

        Args:
            for_write: unused (TODO use or remove)
            isolation_level: if specified, the isolation level for the new transaction. Can only be set when starting
                the outermost transaction.

        TODO: repeatable read is not available in Cockroachdb; instead, run queries against a snapshot TVP
        that avoids tripping over any pending ops
        """
        from pixeltable.env import Env

        if not self.in_xact:
            assert self.session is None
            try:
                self.isolation_level = isolation_level or _XACT_ISOLATION_LEVEL
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
            assert isolation_level is None or isolation_level == self.isolation_level, (
                f'cannot change isolation level to {isolation_level!r} for a joined transaction '
                f'(current: {self.isolation_level!r})'
            )
            assert self.isolation_level == _XACT_ISOLATION_LEVEL or not for_write
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
        from pixeltable.catalog.path import ROOT_PATH

        local_catalog = runtime._catalogs.get(ROOT_PATH)
        if local_catalog is not None:
            from pixeltable.catalog.catalog import Catalog

            assert isinstance(local_catalog, Catalog)
            # Invalidate all existing TableVersion instances to force reloading of metadata,
            for tbl_version in local_catalog._tbl_versions.values():
                tbl_version.is_validated = False
        if runtime._event_loop is not None:
            # Don't close a loop we didn't create (e.g. Jupyter's)
            try:
                loop = runtime._event_loop
                if not loop.is_running():
                    loop.close()
            except Exception:
                pass
        runtime._clients.clear()
        if runtime._run_coro_executor is not None:
            runtime._run_coro_executor.shutdown(wait=False)
    _thread_local.runtime = Runtime()
