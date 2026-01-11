from __future__ import annotations

import asyncio
import builtins
import datetime
import glob
import http.server
import importlib
import importlib.util
import inspect
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import types
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from sys import stdout
from typing import Any, Callable, Iterator, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import nest_asyncio  # type: ignore[import-untyped]
import pixeltable_pgserver
import sqlalchemy as sql
import tzlocal
from pillow_heif import register_heif_opener  # type: ignore[import-untyped]
from rich.progress import Progress
from sqlalchemy import orm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.utils.console_output import ConsoleLogger, ConsoleMessageFilter, ConsoleOutputHandler, map_level
from pixeltable.utils.dbms import CockroachDbms, Dbms, PostgresqlDbms
from pixeltable.utils.http_server import make_server
from pixeltable.utils.object_stores import ObjectPath
from pixeltable.utils.sql import add_option_to_db_url

_logger = logging.getLogger('pixeltable')

T = TypeVar('T')


class Env:
    """
    Store runtime globals for both local and non-local environments.
    For a local environment, Pixeltable uses an embedded PostgreSQL server that runs locally in a separate process.
    For a non-local environment, Pixeltable uses a connection string to the externally managed database.
    """

    SERIALIZABLE_ISOLATION_LEVEL = 'SERIALIZABLE'

    _instance: Env | None = None
    __initializing: bool = False
    _log_fmt_str = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'

    _media_dir: Path | None
    _file_cache_dir: Path | None  # cached object files with external URL
    _dataset_cache_dir: Path | None  # cached datasets (eg, pytorch or COCO)
    _log_dir: Path | None  # log files
    _tmp_dir: Path | None  # any tmp files
    _sa_engine: sql.engine.base.Engine | None
    _pgdata_dir: Path | None
    _db_name: str | None
    _db_server: pixeltable_pgserver.PostgresServer | None  # set only when running in local environment
    _db_url: str | None
    _default_time_zone: ZoneInfo | None
    _verbosity: int

    # info about optional packages that are utilized by some parts of the code
    __optional_packages: dict[str, PackageInfo]

    _httpd: http.server.HTTPServer | None
    _http_address: str | None
    _logger: logging.Logger
    _default_log_level: int
    _logfilename: str | None
    _log_to_stdout: bool
    _module_log_level: dict[str, int]  # module name -> log level
    _file_cache_size_g: float
    _default_input_media_dest: str | None
    _default_output_media_dest: str | None
    _pxt_api_key: str | None
    _stdout_handler: logging.StreamHandler
    _default_video_encoder: str | None
    _initialized: bool
    _progress: Progress | None

    _resource_pool_info: dict[str, Any]
    _current_conn: sql.Connection | None
    _current_session: orm.Session | None
    _current_isolation_level: str | None
    _dbms: Dbms | None
    _event_loop: asyncio.AbstractEventLoop | None  # event loop for ExecNode

    @classmethod
    def get(cls) -> Env:
        if cls._instance is None:
            cls._init_env()
        return cls._instance

    @classmethod
    def _init_env(cls, reinit_db: bool = False) -> None:
        assert not cls.__initializing, 'Circular env initialization detected.'
        cls.__initializing = True
        if cls._instance is not None:
            cls._instance._clean_up()
        cls._instance = None
        env = Env()
        try:
            env._set_up(reinit_db=reinit_db)
            env._upgrade_metadata()
            cls._instance = env
        finally:
            # Reset the initializing flag, even if setup fails.
            # This prevents the environment from being left in a broken state.
            cls.__initializing = False

    def __init__(self) -> None:
        assert self._instance is None, 'Env is a singleton; use Env.get() to access the instance'

        self._media_dir = None  # computed media files
        self._file_cache_dir = None  # cached object files with external URL
        self._dataset_cache_dir = None  # cached datasets (eg, pytorch or COCO)
        self._log_dir = None  # log files
        self._tmp_dir = None  # any tmp files
        self._sa_engine = None
        self._pgdata_dir = None
        self._db_name = None
        self._db_server = None
        self._db_url = None
        self._default_time_zone = None
        self.__optional_packages = {}
        self._httpd = None
        self._http_address = None
        self._default_video_encoder = None

        # logging-related state
        self._logger = logging.getLogger('pixeltable')
        self._logger.setLevel(logging.DEBUG)  # allow everything to pass, we filter in _log_filter()
        self._logger.propagate = False
        self._logger.addFilter(self._log_filter)
        self._default_log_level = logging.INFO
        self._logfilename = None
        self._log_to_stdout = False
        self._module_log_level = {}  # module name -> log level

        # create logging handler to also log to stdout
        self._stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self._stdout_handler.setFormatter(logging.Formatter(self._log_fmt_str))
        self._initialized = False
        self._progress = None

        self._resource_pool_info = {}
        self._current_conn = None
        self._current_session = None
        self._current_isolation_level = None
        self._dbms = None
        self._event_loop = None

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

        # always allow nested event loops, we need that to run async udfs synchronously (eg, for SimilarityExpr);
        # see run_coroutine_synchronously()
        nest_asyncio.apply()
        if _logger.isEnabledFor(logging.DEBUG):
            self._event_loop.set_debug(True)

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        if self._event_loop is None:
            self._init_event_loop()
        return self._event_loop

    @property
    def db_url(self) -> str:
        assert self._db_url is not None
        return self._db_url

    @property
    def http_address(self) -> str:
        assert self._http_address is not None
        return self._http_address

    @property
    def user(self) -> str | None:
        return Config.get().get_string_value('user')

    @user.setter
    def user(self, user: str | None) -> None:
        if user is None:
            if 'PIXELTABLE_USER' in os.environ:
                del os.environ['PIXELTABLE_USER']
        else:
            os.environ['PIXELTABLE_USER'] = user

    @property
    def default_time_zone(self) -> ZoneInfo | None:
        return self._default_time_zone

    @default_time_zone.setter
    def default_time_zone(self, tz: ZoneInfo | None) -> None:
        """
        This is not a publicly visible setter; it is only for testing purposes.
        """
        if tz is None:
            tz_name = self._get_tz_name()
        else:
            assert isinstance(tz, ZoneInfo)
            tz_name = tz.key
        self.engine.dispose()
        self._create_engine(time_zone_name=tz_name)

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @property
    def conn(self) -> sql.Connection | None:
        assert self._current_conn is not None
        return self._current_conn

    @property
    def session(self) -> orm.Session | None:
        assert self._current_session is not None
        return self._current_session

    @property
    def dbms(self) -> Dbms | None:
        assert self._dbms is not None
        return self._dbms

    @property
    def is_using_cockroachdb(self) -> bool:
        assert self._dbms is not None
        return isinstance(self._dbms, CockroachDbms)

    @property
    def in_xact(self) -> bool:
        return self._current_conn is not None

    @property
    def is_local(self) -> bool:
        assert self._db_url is not None  # is_local should be called only after db initialization
        return self._db_server is not None

    def is_interactive(self) -> bool:
        """Return True if running in an interactive environment."""
        if getattr(builtins, '__IPYTHON__', False):
            return True
        # Python interactive shell
        if hasattr(sys, 'ps1'):
            return True
        # for script execution, __main__ has __file__
        import __main__

        return not hasattr(__main__, '__file__')

    def is_notebook(self) -> bool:
        """Return True if running in a Jupyter notebook."""
        try:
            shell = get_ipython()  # type: ignore[name-defined]
            return 'ZMQInteractiveShell' in str(shell)
        except NameError:
            return False

    def start_progress(self, create_fn: Callable[[], Progress]) -> Progress:
        if self._progress is None:
            self._progress = create_fn()
            self._progress.start()
        return self._progress

    def stop_progress(self) -> None:
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None

        # if we're running in a notebook, we need to clear the Progress output manually
        if self.is_notebook():
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

    @contextmanager
    def begin_xact(self, *, for_write: bool = False) -> Iterator[sql.Connection]:
        """
        Call Catalog.begin_xact() instead, unless there is a specific reason to call this directly.

        for_write: if True, uses serializable isolation; if False, uses repeatable_read

        TODO: repeatable read is not available in Cockroachdb; instead, run queries against a snapshot TVP
        that avoids tripping over any pending ops
        """
        if self._current_conn is None:
            assert self._current_session is None
            try:
                self._current_isolation_level = self.SERIALIZABLE_ISOLATION_LEVEL
                with (
                    self.engine.connect().execution_options(isolation_level=self._current_isolation_level) as conn,
                    orm.Session(conn) as session,
                    conn.begin(),
                ):
                    self._current_conn = conn
                    self._current_session = session
                    yield conn
            finally:
                self._current_session = None
                self._current_conn = None
                self._current_isolation_level = None
        else:
            assert self._current_session is not None
            assert self._current_isolation_level == self.SERIALIZABLE_ISOLATION_LEVEL or not for_write
            yield self._current_conn

    def configure_logging(
        self,
        *,
        to_stdout: bool | None = None,
        level: int | None = None,
        add: str | None = None,
        remove: str | None = None,
    ) -> None:
        """Configure logging.

        Args:
            to_stdout: if True, also log to stdout
            level: default log level
            add: comma-separated list of 'module name:log level' pairs; ex.: add='video:10'
            remove: comma-separated list of module names
        """
        if to_stdout is not None:
            self.log_to_stdout(to_stdout)
        if level is not None:
            self.set_log_level(level)
        if add is not None:
            for module, level_str in [t.split(':') for t in add.split(',')]:
                self.set_module_log_level(module, int(level_str))
        if remove is not None:
            for module in remove.split(','):
                self.set_module_log_level(module, None)
        if to_stdout is None and level is None and add is None and remove is None:
            self.print_log_config()

    def print_log_config(self) -> None:
        print(f'logging to {self._logfilename}')
        print(f'{"" if self._log_to_stdout else "not "}logging to stdout')
        print(f'default log level: {logging.getLevelName(self._default_log_level)}')
        print(
            f'module log levels: '
            f'{",".join([name + ":" + logging.getLevelName(val) for name, val in self._module_log_level.items()])}'
        )

    def log_to_stdout(self, enable: bool = True) -> None:
        self._log_to_stdout = enable
        if enable:
            self._logger.addHandler(self._stdout_handler)
        else:
            self._logger.removeHandler(self._stdout_handler)

    def set_log_level(self, level: int) -> None:
        self._default_log_level = level

    def set_module_log_level(self, module: str, level: int | None) -> None:
        if level is None:
            self._module_log_level.pop(module, None)
        else:
            self._module_log_level[module] = level

    def is_installed_package(self, package_name: str) -> bool:
        assert package_name in self.__optional_packages
        return self.__optional_packages[package_name].is_installed

    def _log_filter(self, record: logging.LogRecord) -> bool:
        if record.name == 'pixeltable':
            # accept log messages from a configured pixeltable module (at any level of the module hierarchy)
            path_parts = list(Path(record.pathname).parts)
            path_parts.reverse()
            if 'pixeltable' not in path_parts:
                return False
            max_idx = path_parts.index('pixeltable')
            for module_name in path_parts[:max_idx]:
                if module_name in self._module_log_level and record.levelno >= self._module_log_level[module_name]:
                    return True
        return record.levelno >= self._default_log_level

    @property
    def console_logger(self) -> ConsoleLogger:
        return self._console_logger

    def _get_tz_name(self) -> str:
        """Get the time zone name from the configuration, or the system local time zone if not specified.

        Returns:
            str: The time zone name.
        """
        tz_name = Config.get().get_string_value('time_zone')
        if tz_name is not None:
            # Validate tzname
            if not isinstance(tz_name, str):
                self._logger.error('Invalid time zone specified in configuration.')
            else:
                try:
                    _ = ZoneInfo(tz_name)
                except ZoneInfoNotFoundError:
                    self._logger.error(f'Invalid time zone specified in configuration: {tz_name}')
        else:
            tz_name = tzlocal.get_localzone_name()
        return tz_name

    def _set_up(self, echo: bool = False, reinit_db: bool = False) -> None:
        if self._initialized:
            return

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        config = Config.get()

        self._initialized = True

        self._media_dir = Config.get().home / 'media'
        self._file_cache_dir = Config.get().home / 'file_cache'
        self._dataset_cache_dir = Config.get().home / 'dataset_cache'
        self._log_dir = Config.get().home / 'logs'
        self._tmp_dir = Config.get().home / 'tmp'

        self._media_dir.mkdir(exist_ok=True)
        self._file_cache_dir.mkdir(exist_ok=True)
        self._dataset_cache_dir.mkdir(exist_ok=True)
        self._log_dir.mkdir(exist_ok=True)
        self._tmp_dir.mkdir(exist_ok=True)

        self._file_cache_size_g = config.get_float_value('file_cache_size_g')
        if self._file_cache_size_g is None:
            raise excs.Error(
                'pixeltable/file_cache_size_g is missing from configuration\n'
                f'(either add a `file_cache_size_g` entry to the `pixeltable` section of {Config.get().config_file},\n'
                'or set the PIXELTABLE_FILE_CACHE_SIZE_G environment variable)'
            )

        self._default_input_media_dest = config.get_string_value('input_media_dest')
        self._default_output_media_dest = config.get_string_value('output_media_dest')
        for mode, uri in (('input', self._default_input_media_dest), ('output', self._default_output_media_dest)):
            if uri is not None:
                try:
                    _ = ObjectPath.parse_object_storage_addr(uri, False)
                except Exception as e:
                    raise excs.Error(f'Invalid {mode} media destination URI: {uri}') from e

        self._pxt_api_key = config.get_string_value('api_key')

        # Disable spurious warnings:
        # Suppress tqdm's ipywidgets warning in Jupyter environments
        warnings.filterwarnings('ignore', message='IProgress not found')
        # suppress Rich's ipywidgets warning in Jupyter environments
        warnings.filterwarnings('ignore', message='install "ipywidgets" for Jupyter support')
        if config.get_bool_value('hide_warnings'):
            # Disable more warnings
            warnings.simplefilter('ignore', category=UserWarning)
            warnings.simplefilter('ignore', category=FutureWarning)

        # if we're running in a Jupyter notebook, warn about missing ipywidgets
        if self.is_notebook() and importlib.util.find_spec('ipywidgets') is None:
            warnings.warn(
                'Progress reporting is disabled because ipywidgets is not installed. '
                'To fix this, run: `pip install ipywidgets`',
                stacklevel=1,
            )

        # Set verbosity level for user visible console messages
        self._verbosity = config.get_int_value('verbosity')
        if self._verbosity is None:
            self._verbosity = 1
        stdout_handler = ConsoleOutputHandler(stream=stdout)
        stdout_handler.setLevel(map_level(self._verbosity))
        stdout_handler.addFilter(ConsoleMessageFilter())
        self._logger.addHandler(stdout_handler)
        self._console_logger = ConsoleLogger(self._logger)

        # configure _logger to log to a file
        self._logfilename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        fh = logging.FileHandler(self._log_dir / self._logfilename, mode='w')
        fh.setFormatter(logging.Formatter(self._log_fmt_str))
        self._logger.addHandler(fh)

        # configure sqlalchemy logging
        sql_logger = logging.getLogger('sqlalchemy.engine')
        sql_logger.setLevel(logging.INFO)
        sql_logger.addHandler(fh)
        sql_logger.propagate = False

        # configure pyav logging
        av_logfilename = self._logfilename.replace('.log', '_av.log')
        av_fh = logging.FileHandler(self._log_dir / av_logfilename, mode='w')
        av_fh.setFormatter(logging.Formatter(self._log_fmt_str))
        av_logger = logging.getLogger('libav')
        av_logger.addHandler(av_fh)
        av_logger.propagate = False

        # configure web-server logging
        http_logfilename = self._logfilename.replace('.log', '_http.log')
        http_fh = logging.FileHandler(self._log_dir / http_logfilename, mode='w')
        http_fh.setFormatter(logging.Formatter(self._log_fmt_str))
        http_logger = logging.getLogger('pixeltable.http.server')
        http_logger.addHandler(http_fh)
        http_logger.propagate = False

        self.clear_tmp_dir()
        tz_name = self._get_tz_name()

        # configure pixeltable database
        self._init_db(config)

        if reinit_db and not self.is_local:
            raise excs.Error(
                'Reinitializing pixeltable database is not supported when running in non-local environment'
            )

        if reinit_db and self._store_db_exists():
            self._drop_store_db()

        create_db = not self._store_db_exists()
        if create_db:
            self._logger.info(f'creating database at: {self.db_url}')
            self._create_store_db()
        else:
            self._logger.info(f'found database at: {self.db_url}')

        # Create the SQLAlchemy engine. This will also set the default time zone.
        self._create_engine(time_zone_name=tz_name, echo=echo)

        # Create catalog tables and system metadata
        self._init_metadata()

        self.console_logger.info(f'Connected to Pixeltable database at: {self.db_url}')

        # we now have a home directory and db; start other services
        self._set_up_runtime()
        self.log_to_stdout(False)

    def _init_db(self, config: Config) -> None:
        """
        Initialize the pixeltable database along with its associated DBMS.
        """
        db_connect_str = config.get_string_value('DB_CONNECT_STR')
        if db_connect_str is not None:
            try:
                db_url = sql.make_url(db_connect_str)
            except sql.exc.ArgumentError as e:
                error = f'Invalid db connection string {db_connect_str}: {e}'
                self._logger.error(error)
                raise excs.Error(error) from e
            self._db_url = db_url.render_as_string(hide_password=False)
            self._db_name = db_url.database  # use the dbname given in connect string
            dialect = db_url.get_dialect().name
            if dialect == 'cockroachdb':
                self._dbms = CockroachDbms(db_url)
            else:
                raise excs.Error(f'Unsupported DBMS {dialect}')
            # Check if database exists
            if not self._store_db_exists():
                error = f'Database {self._db_name!r} does not exist'
                self._logger.error(error)
                raise excs.Error(error)
            self._logger.info(f'Using database at: {self.db_url}')
        else:
            self._db_name = config.get_string_value('db') or 'pixeltable'
            self._pgdata_dir = Path(os.environ.get('PIXELTABLE_PGDATA', str(Config.get().home / 'pgdata')))
            # cleanup_mode=None will leave the postgres process running after Python exits
            # cleanup_mode='stop' will terminate the postgres process when Python exits
            # On Windows, we need cleanup_mode='stop' because child processes are killed automatically when the parent
            # process (such as Terminal or VSCode) exits, potentially leaving it in an unusable state.
            cleanup_mode = 'stop' if platform.system() == 'Windows' else None
            self._db_server = pixeltable_pgserver.get_server(self._pgdata_dir, cleanup_mode=cleanup_mode)
            self._db_url = self._db_server.get_uri(database=self._db_name, driver='psycopg')
            self._dbms = PostgresqlDbms(sql.make_url(self._db_url))
        assert self._dbms is not None
        assert self._db_url is not None
        assert self._db_name is not None

    @retry(
        stop=stop_after_attempt(3),  # Stop after 3 attempts
        wait=wait_exponential_jitter(initial=0.2, max=1.0, jitter=0.2),  # Exponential backoff with jitter
    )
    def _init_metadata(self) -> None:
        """
        Create pixeltable metadata tables and system metadata.
        This is an idempotent operation.

        Retry logic handles race conditions when multiple Pixeltable processes
        attempt to initialize metadata tables simultaneously. The first process may succeed
        in creating tables while others encounter database constraints (e.g., "table already exists").
        Exponential backoff with jitter reduces contention between competing processes.
        """
        assert self._sa_engine is not None
        from pixeltable import metadata

        self._logger.debug('Creating pixeltable metadata')
        metadata.schema.base_metadata.create_all(self._sa_engine, checkfirst=True)
        metadata.create_system_info(self._sa_engine)

    def _create_engine(self, time_zone_name: str, echo: bool = False) -> None:
        # Add timezone option to connection string
        updated_url = add_option_to_db_url(self.db_url, f'-c timezone={time_zone_name}')

        self._sa_engine = sql.create_engine(
            updated_url, echo=echo, isolation_level=self._dbms.transaction_isolation_level
        )

        self._logger.info(f'Created SQLAlchemy engine at: {self.db_url}')
        self._logger.info(f'Engine dialect: {self._sa_engine.dialect.name}')
        self._logger.info(f'Engine driver : {self._sa_engine.dialect.driver}')

        with self.engine.begin() as conn:
            tz_name = conn.execute(sql.text('SHOW TIME ZONE')).scalar()
            assert isinstance(tz_name, str)
            self._logger.info(f'Database time zone is now: {tz_name}')
            self._default_time_zone = ZoneInfo(tz_name)
            if self.is_using_cockroachdb:
                # This could be set when the database is created, but we set it now
                conn.execute(sql.text('SET null_ordered_last = true;'))
                null_ordered_last = conn.execute(sql.text('SHOW null_ordered_last')).scalar()
                assert isinstance(null_ordered_last, str)
                self._logger.info(f'Database null_ordered_last is now: {null_ordered_last}')

    def _store_db_exists(self) -> bool:
        assert self._db_name is not None
        # don't try to connect to self.db_name, it may not exist
        engine = sql.create_engine(self._dbms.default_system_db_url(), future=True)
        try:
            with engine.begin() as conn:
                stmt = f"SELECT COUNT(*) FROM pg_database WHERE datname = '{self._db_name}'"
                result = conn.scalar(sql.text(stmt))
                assert result <= 1
                return result == 1
        finally:
            engine.dispose()

    def _create_store_db(self) -> None:
        assert self._db_name is not None
        # create the db
        engine = sql.create_engine(self._dbms.default_system_db_url(), future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                stmt = self._dbms.create_db_stmt(preparer.quote(self._db_name))
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

        # enable pgvector
        engine = sql.create_engine(self.db_url, future=True, isolation_level='AUTOCOMMIT')
        try:
            with engine.begin() as conn:
                conn.execute(sql.text('CREATE EXTENSION vector'))
        finally:
            engine.dispose()

    def _pgserver_terminate_connections_stmt(self) -> str:
        return f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{self._db_name}'
                AND pid <> pg_backend_pid()
            """

    def _drop_store_db(self) -> None:
        assert self._db_name is not None
        engine = sql.create_engine(self._dbms.default_system_db_url(), future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                # terminate active connections
                if self._db_server is not None:
                    conn.execute(sql.text(self._pgserver_terminate_connections_stmt()))
                # drop db
                stmt = self._dbms.drop_db_stmt(preparer.quote(self._db_name))
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

    def _upgrade_metadata(self) -> None:
        from pixeltable import metadata

        metadata.upgrade_md(self._sa_engine)

    @property
    def pxt_api_key(self) -> str | None:
        return self._pxt_api_key

    def get_client(self, name: str) -> Any:
        """
        Gets the client with the specified name, initializing it if necessary.

        Args:
            - name: The name of the client
        """
        # Return the existing client if it has already been constructed
        with _registered_clients_lock:
            cl = _registered_clients[name]
            if cl.client_obj is not None:
                return cl.client_obj  # Already initialized

        # Retrieve parameters required to construct the requested client.
        init_kwargs: dict[str, Any] = {}
        for param in cl.params.values():
            # Determine the type of the parameter for proper config parsing.
            pname = param.name
            t = param.annotation
            # Deference T | None
            if typing.get_origin(t) in (typing.Union, types.UnionType):
                args = typing.get_args(t)
                if args[0] is type(None):
                    t = args[1]
                elif args[1] is type(None):
                    t = args[0]
            assert isinstance(t, type), t
            arg: Any = Config.get().get_value(pname, t, section=name)
            if arg is not None:
                init_kwargs[pname] = arg
            elif param.default is inspect.Parameter.empty:
                raise excs.Error(
                    f'`{name}` client not initialized: parameter `{pname}` is not configured.\n'
                    f'To fix this, specify the `{name.upper()}_{pname.upper()}` environment variable, '
                    f'or put `{pname.lower()}` in the `{name.lower()}` section of $PIXELTABLE_HOME/config.toml.'
                )

        # Construct the requested client
        with _registered_clients_lock:
            if cl.client_obj is not None:
                return cl.client_obj  # Already initialized
            cl.client_obj = cl.init_fn(**init_kwargs)
            self._logger.info(f'Initialized `{name}` client with parameters: {init_kwargs}.')
            return cl.client_obj

    def _start_web_server(self) -> None:
        """
        The http server root is the file system root.
        eg: /home/media/foo.mp4 is located at http://127.0.0.1:{port}/home/media/foo.mp4
        On Windows, the server will translate paths like http://127.0.0.1:{port}/c:/media/foo.mp4
        This arrangement enables serving objects hosted within _home,
        as well as external objects inserted into pixeltable or produced by pixeltable.
        The port is chosen dynamically to prevent conflicts.
        """
        # Port 0 means OS picks one for us.
        self._httpd = make_server('127.0.0.1', 0)
        port = self._httpd.server_address[1]
        self._http_address = f'http://127.0.0.1:{port}'

        def run_server() -> None:
            logging.log(logging.INFO, f'running web server at {self._http_address}')
            self._httpd.serve_forever()

        # Run the server in a separate thread
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    def _set_up_runtime(self) -> None:
        """Check for and start runtime services"""
        register_heif_opener()
        self._start_web_server()
        self.__register_packages()

    @property
    def default_video_encoder(self) -> str | None:
        if self._default_video_encoder is None:
            self._default_video_encoder = self._determine_default_video_encoder()
        return self._default_video_encoder

    def _determine_default_video_encoder(self) -> str | None:
        """
        Returns the first available encoder from a list of candidates.

        TODO:
        - the user might prefer a hardware-accelerated encoder (eg, h264_nvenc or h264_videotoolbox)
        - allow user override via a config option 'video_encoder'
        """
        # look for available encoders, in this order
        candidates = [
            'libx264',  # GPL, best quality
            'libopenh264',  # BSD
        ]

        try:
            # Get list of available encoders
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10, check=True)

            if result.returncode == 0:
                available_encoders = result.stdout
                for encoder in candidates:
                    # ffmpeg -encoders output format: " V..... encoder_name  description"
                    if f' {encoder} ' in available_encoders:
                        _logger.debug(f'Using H.264 encoder: {encoder}')
                        return encoder
        except Exception:
            pass
        return None

    def __register_packages(self) -> None:
        """Declare optional packages that are utilized by some parts of the code."""
        self.__register_package('accelerate')
        self.__register_package('anthropic')
        self.__register_package('azure.storage.blob', library_name='azure-storage-blob')
        self.__register_package('boto3')
        self.__register_package('datasets')
        self.__register_package('diffusers')
        self.__register_package('fal_client', library_name='fal-client')
        self.__register_package('fiftyone')
        self.__register_package('fireworks', library_name='fireworks-ai')
        self.__register_package('google.cloud.storage', library_name='google-cloud-storage')
        self.__register_package('google.genai', library_name='google-genai')
        self.__register_package('groq')
        self.__register_package('huggingface_hub', library_name='huggingface-hub')
        self.__register_package('label_studio_sdk', library_name='label-studio-sdk')
        self.__register_package('lancedb')
        self.__register_package('librosa')
        self.__register_package('llama_cpp', library_name='llama-cpp-python')
        self.__register_package('markitdown')
        self.__register_package('mcp')
        self.__register_package('mistralai')
        self.__register_package('mistune')
        self.__register_package('ollama')
        self.__register_package('openai')
        self.__register_package('openpyxl')
        self.__register_package('pyarrow')
        self.__register_package('pydantic')
        self.__register_package('replicate')
        self.__register_package('reve')
        self.__register_package('scenedetect')
        self.__register_package('sentencepiece')
        self.__register_package('sentence_transformers', library_name='sentence-transformers')
        self.__register_package('snowflake.sqlalchemy', library_name='snowflake-sqlalchemy')
        self.__register_package('soundfile')
        self.__register_package('spacy')
        self.__register_package('tiktoken')
        self.__register_package('timm')
        self.__register_package('together')
        self.__register_package('torch')
        self.__register_package('torchaudio')
        self.__register_package('torchvision')
        self.__register_package('transformers')
        self.__register_package('twelvelabs')
        self.__register_package('voyageai')
        self.__register_package('whisper', library_name='openai-whisper')
        self.__register_package('whisperx')
        self.__register_package('yolox', library_name='pixeltable-yolox')

    def __register_package(self, package_name: str, library_name: str | None = None) -> None:
        is_installed: bool
        try:
            is_installed = importlib.util.find_spec(package_name) is not None
        except ModuleNotFoundError:
            # This can happen if the parent of `package_name` is not installed.
            is_installed = False
        self.__optional_packages[package_name] = PackageInfo(
            is_installed=is_installed,
            library_name=library_name or package_name,  # defaults to package_name unless specified otherwise
        )

    def require_binary(self, binary_name: str) -> None:
        if not shutil.which(binary_name):
            raise excs.Error(f'{binary_name} is not installed or not in PATH. Please install it to use this feature.')

    def require_package(
        self, package_name: str, min_version: list[int] | None = None, not_installed_msg: str | None = None
    ) -> None:
        """
        Checks whether the specified optional package is available. If not, raises an exception
        with an error message informing the user how to install it.
        """
        assert package_name in self.__optional_packages
        package_info = self.__optional_packages[package_name]

        if not package_info.is_installed:
            # Check again whether the package has been installed.
            # We do this so that if a user gets an "optional library not found" error message, they can
            # `pip install` the library and re-run the Pixeltable operation without having to restart
            # their Python session.
            package_info.is_installed = importlib.util.find_spec(package_name) is not None
            if not package_info.is_installed:
                # Still not found.
                if not_installed_msg is None:
                    not_installed_msg = f'This feature requires the `{package_name}` package'
                raise excs.Error(
                    f'{not_installed_msg}. To install it, run: `pip install -U {package_info.library_name}`'
                )

        if min_version is None:
            return

        # check whether we have a version >= the required one
        if package_info.version is None:
            module = importlib.import_module(package_name)
            package_info.version = [int(x) for x in module.__version__.split('.')]

        if min_version > package_info.version:
            raise excs.Error(
                f'The installed version of package `{package_name}` is '
                f'{".".join(str(v) for v in package_info.version)}, '
                f'but version >={".".join(str(v) for v in min_version)} is required. '
                f'To fix this, run: `pip install -U {package_info.library_name}`'
            )

    def clear_tmp_dir(self) -> None:
        for path in glob.glob(f'{self._tmp_dir}/*'):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    # def get_resource_pool_info(self, pool_id: str, pool_info_cls: Type[T] | None) -> T:
    def get_resource_pool_info(self, pool_id: str, make_pool_info: Callable[[], T] | None = None) -> T:
        """Returns the info object for the given id, creating it if necessary."""
        info = self._resource_pool_info.get(pool_id)
        if info is None and make_pool_info is not None:
            info = make_pool_info()
            self._resource_pool_info[pool_id] = info
        return info

    @property
    def media_dir(self) -> Path:
        assert self._media_dir is not None
        return self._media_dir

    @property
    def default_input_media_dest(self) -> str | None:
        return self._default_input_media_dest

    @property
    def default_output_media_dest(self) -> str | None:
        return self._default_output_media_dest

    @property
    def file_cache_dir(self) -> Path:
        assert self._file_cache_dir is not None
        return self._file_cache_dir

    @property
    def dataset_cache_dir(self) -> Path:
        assert self._dataset_cache_dir is not None
        return self._dataset_cache_dir

    @property
    def tmp_dir(self) -> Path:
        assert self._tmp_dir is not None
        return self._tmp_dir

    @property
    def engine(self) -> sql.engine.base.Engine:
        assert self._sa_engine is not None
        return self._sa_engine

    def _clean_up(self) -> None:
        """
        Internal cleanup method that properly closes all resources and resets state.
        This is called before destroying the singleton instance.
        """
        assert self._current_session is None
        assert self._current_conn is None

        # Stop HTTP server
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception as e:
                _logger.warning(f'Error stopping HTTP server: {e}')

        # First terminate all connections to the database
        if self._db_server is not None:
            assert self._dbms is not None
            assert self._db_name is not None
            try:
                temp_engine = sql.create_engine(self._dbms.default_system_db_url(), isolation_level='AUTOCOMMIT')
                try:
                    with temp_engine.begin() as conn:
                        conn.execute(sql.text(self._pgserver_terminate_connections_stmt()))
                        _logger.info(f"Terminated all connections to database '{self._db_name}'")
                except Exception as e:
                    _logger.warning(f'Error terminating database connections: {e}')
                finally:
                    temp_engine.dispose()
            except Exception as e:
                _logger.warning(f'Error stopping database server: {e}')

        # Dispose of SQLAlchemy engine (after stopping db server)
        if self._sa_engine is not None:
            try:
                self._sa_engine.dispose()
            except Exception as e:
                _logger.warning(f'Error disposing engine: {e}')

        # Close event loop
        if self._event_loop is not None:
            try:
                if self._event_loop.is_running():
                    self._event_loop.stop()
                self._event_loop.close()
            except Exception as e:
                _logger.warning(f'Error closing event loop: {e}')

        # Remove logging handlers
        for handler in self._logger.handlers[:]:
            try:
                handler.close()
                self._logger.removeHandler(handler)
            except Exception as e:
                _logger.warning(f'Error removing handler: {e}')


def register_client(name: str) -> Callable:
    """Decorator that registers a third-party API client for use by Pixeltable.

    The decorated function is an initialization wrapper for the client, and can have
    any number of string parameters, with a signature such as:

    ```
    def my_client(api_key: str, url: str) -> my_client_sdk.Client:
        return my_client_sdk.Client(api_key=api_key, url=url)
    ```

    The initialization wrapper will not be called immediately; initialization will
    be deferred until the first time the client is used. At initialization time,
    Pixeltable will attempt to load the client parameters from config. For each
    config parameter:
    - If an environment variable named MY_CLIENT_API_KEY (for example) is set, use it;
    - Otherwise, look for 'api_key' in the 'my_client' section of config.toml.

    If all config parameters are found, Pixeltable calls the initialization function;
    otherwise it throws an exception.

    Args:
        - name (str): The name of the API client (e.g., 'openai' or 'label-studio').
    """

    def decorator(fn: Callable) -> None:
        sig = inspect.signature(fn)
        params = dict(sig.parameters)
        with _registered_clients_lock:
            _registered_clients[name] = ApiClient(init_fn=fn, params=params)

    return decorator


_registered_clients_lock: threading.Lock = threading.Lock()
_registered_clients: dict[str, ApiClient] = {}


@dataclass
class ApiClient:
    init_fn: Callable
    params: dict[str, inspect.Parameter]
    client_obj: Any | None = None


@dataclass
class PackageInfo:
    is_installed: bool
    library_name: str  # pypi library name (may be different from package name)
    version: list[int] | None = None  # installed version, as a list of components (such as [3,0,2] for "3.0.2")


TIME_FORMAT = '%H:%M.%S %f'
# As far as rate limiting goes, we try not go lower than 5% of the capacity because we don't have perfect information
# about the rate limits and the usage
TARGET_RATE_LIMIT_RESOURCE_FRACT = 0.05


@dataclass
class RateLimitsInfo:
    """
    Abstract base class for resource pools made up of rate limits for different resources.

    Rate limits and currently remaining resources are periodically reported via record().

    Subclasses provide operational customization via:
    - get_retry_delay()
    - get_request_resources(self, ...) -> dict[str, int]
    with parameters that are a subset of those of the udf that creates the subclass's instance
    """

    # get_request_resources:
    # - Returns estimated resources needed for a specific request (ie, a single udf call) as a dict (key: resource name)
    # - parameters are a subset of those of the udf
    # - this is not a class method because the signature depends on the instantiating udf
    get_request_resources: Callable[..., dict[str, int]]

    resource_limits: dict[str, RateLimitInfo] = field(default_factory=dict)
    has_exc: bool = False

    def debug_str(self) -> str:
        return ','.join(info.debug_str() for info in self.resource_limits.values())

    def is_initialized(self) -> bool:
        return len(self.resource_limits) > 0

    def reset(self) -> None:
        self.resource_limits.clear()

    def record(self, request_ts: datetime.datetime, reset_exc: bool = False, **kwargs: Any) -> None:
        """Update self.resource_limits with the provided rate limit info.
        Args:
            - request_ts: time at which the request was made
            - reset_exc: if True, reset the has_exc flag
        """
        if len(self.resource_limits) == 0:
            self.resource_limits = {k: RateLimitInfo(k, request_ts, *v) for k, v in kwargs.items() if v is not None}
            # TODO: remove
            for info in self.resource_limits.values():
                _logger.debug(f'Updated resource state: {info}')
        else:
            if self.has_exc and not reset_exc:
                # ignore updates until we're asked to reset
                _logger.debug(f'rate_limits.record(): ignoring update {kwargs}')
                return
            self.has_exc = False
            for k, v in kwargs.items():
                if v is not None:
                    self.resource_limits[k].update(request_ts, *v)
                    _logger.debug(f'Updated resource state: {self.resource_limits[k]}')

    def record_exc(self, request_ts: datetime.datetime, exc: Exception) -> None:
        """Update self.resource_limits based on the exception headers
        Args:
            - request_ts: time at which the request that caused the exception was made
            - exc: the exception raised"""
        self.has_exc = True

    def get_retry_delay(self, exc: Exception, attempt: int) -> float | None:
        """Returns number of seconds to wait before retry, or None if not retryable"""
        # Find the highest wait until at least 5% availability of all resources
        max_wait = 0.0
        for limit_info in self.resource_limits.values():
            time_until = limit_info.estimated_resource_refill_delay(
                math.ceil(TARGET_RATE_LIMIT_RESOURCE_FRACT * limit_info.limit)
            )
            if time_until is not None:
                max_wait = max(max_wait, time_until)
        return max_wait if max_wait > 0 else None


@dataclass
class RateLimitInfo:
    """Container for rate limit-related information for a single resource."""

    resource: str
    request_start_ts: datetime.datetime
    limit: int
    remaining: int
    reset_at: datetime.datetime

    def debug_str(self) -> str:
        return (
            f'{self.resource}@{self.request_start_ts.strftime(TIME_FORMAT)}: '
            f'{self.limit}/{self.remaining}/{self.reset_at.strftime(TIME_FORMAT)}'
        )

    def update(
        self, request_start_ts: datetime.datetime, limit: int, remaining: int, reset_at: datetime.datetime
    ) -> None:
        # Responses can come out of order, especially for failed requests. We need to be careful not to overwrite
        # the current state with less up-to-date information. We use request_start_ts as a proxy for rate limit info
        # recency.
        if self.request_start_ts > request_start_ts:
            # The current state is more up-to-date than the update
            _logger.debug(
                f'Ignoring out-of-date update for {self.resource}. Current request_start_ts: '
                f'{self.request_start_ts}, update: {request_start_ts}'
            )
            return
        self.request_start_ts = request_start_ts
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at

    def estimated_resource_refill_delay(self, target_remaining: int) -> float | None:
        """Estimate time in seconds until remaining resources reaches target_remaining.
        Assumes linear replenishment of resources over time.
        Returns None if unable to estimate.
        """
        if self.remaining >= target_remaining:
            return 0
        if self.request_start_ts >= self.reset_at:
            return 0
        if self.limit < target_remaining:
            return None

        # Estimate resource refill rate based on the recorded state and timestamps. Assumes linear refill.
        refill_rate = (self.limit - self.remaining) / (self.reset_at - self.request_start_ts).total_seconds()
        assert refill_rate > 0, f'self={self}, target_remaining={target_remaining}'

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        time_until = (target_remaining - self.remaining) / refill_rate - (now - self.request_start_ts).total_seconds()
        return max(0, math.ceil(time_until))

    def __repr__(self) -> str:
        return (
            f'RateLimitInfo(resource={self.resource}, request_start_ts={self.request_start_ts}, '
            f'remaining={self.remaining}/{self.limit} ({(100 * self.remaining / self.limit):.1f}%), '
            f'reset_at={self.reset_at})'
        )


@dataclass
class RuntimeCtx:
    """
    Container for runtime data provided by the execution system to udfs.

    Udfs that accept the special _runtime_ctx parameter receive an instance of this class.
    """

    # Indicates a retry attempt following a rate limit error (error code: 429). Requires a 'rate-limits' resource pool.
    # If True, call RateLimitsInfo.record() with reset_exc=True.
    is_retry: bool = False
