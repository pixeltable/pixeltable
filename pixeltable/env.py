from __future__ import annotations

import datetime
import glob
import http.server
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pixeltable_pgserver
import sqlalchemy as sql
import yaml
from tqdm import TqdmWarning

import pixeltable.exceptions as excs
from pixeltable import metadata
from pixeltable.utils.http_server import make_server

if TYPE_CHECKING:
    import spacy


class Env:
    """
    Store for runtime globals.
    """

    _instance: Optional[Env] = None
    _log_fmt_str = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'

    _home: Optional[Path]
    _media_dir: Optional[Path]
    _file_cache_dir: Optional[Path]  # cached media files with external URL
    _dataset_cache_dir: Optional[Path]  # cached datasets (eg, pytorch or COCO)
    _log_dir: Optional[Path]  # log files
    _tmp_dir: Optional[Path]  # any tmp files
    _sa_engine: Optional[sql.engine.base.Engine]
    _pgdata_dir: Optional[Path]
    _db_name: Optional[str]
    _db_server: Optional[pixeltable_pgserver.PostgresServer]
    _db_url: Optional[str]
    _default_time_zone: Optional[ZoneInfo]

    # info about installed packages that are utilized by some parts of the code;
    # package name -> version; version == []: package is installed, but we haven't determined the version yet
    _installed_packages: dict[str, Optional[list[int]]]

    _spacy_nlp: Optional[spacy.Language]
    _httpd: Optional[http.server.HTTPServer]
    _http_address: Optional[str]
    _logger: logging.Logger
    _default_log_level: int
    _logfilename: Optional[str]
    _log_to_stdout: bool
    _module_log_level: dict[str, int]  # module name -> log level
    _config_file: Optional[Path]
    _config: Optional[dict[str, Any]]
    _stdout_handler: logging.StreamHandler
    _initialized: bool

    @classmethod
    def get(cls) -> Env:
        if cls._instance is None:
            cls._init_env()
        return cls._instance

    @classmethod
    def _init_env(cls, reinit_db: bool = False) -> None:
        env = Env()
        env._set_up(reinit_db=reinit_db)
        env._upgrade_metadata()
        cls._instance = env

    def __init__(self):
        self._home = None
        self._media_dir = None  # computed media files
        self._file_cache_dir = None  # cached media files with external URL
        self._dataset_cache_dir = None  # cached datasets (eg, pytorch or COCO)
        self._log_dir = None  # log files
        self._tmp_dir = None  # any tmp files
        self._sa_engine = None
        self._pgdata_dir = None
        self._db_name = None
        self._db_server = None
        self._db_url = None
        self._default_time_zone = None

        self._installed_packages = {}
        self._spacy_nlp = None
        self._httpd = None
        self._http_address = None

        # logging-related state
        self._logger = logging.getLogger('pixeltable')
        self._logger.setLevel(logging.DEBUG)  # allow everything to pass, we filter in _log_filter()
        self._logger.propagate = False
        self._logger.addFilter(self._log_filter)
        self._default_log_level = logging.INFO
        self._logfilename = None
        self._log_to_stdout = False
        self._module_log_level = {}  # module name -> log level

        self._config_file = None
        self._config = None

        # create logging handler to also log to stdout
        self._stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self._stdout_handler.setFormatter(logging.Formatter(self._log_fmt_str))
        self._initialized = False

    @property
    def config(self):
        return self._config

    @property
    def db_url(self) -> str:
        assert self._db_url is not None
        return self._db_url

    @property
    def http_address(self) -> str:
        assert self._http_address is not None
        return self._http_address

    @property
    def default_time_zone(self) -> Optional[ZoneInfo]:
        return self._default_time_zone

    @default_time_zone.setter
    def default_time_zone(self, tz: Optional[ZoneInfo]) -> None:
        """
        This is not a publicly visible setter. It is only intended to be set during initialization
        or for testing purposes.
        """
        if tz is None:
            # System time zone. Since Postgres and Python may interpret the default time zone differently,
            # we first set the Postgres time zone to its default, and then query Postgres for the IANA zone
            # name, using the latter to set an explicit default time zone in Python.
            with self.engine.begin() as conn:
                conn.execute(sql.text('SET TIME ZONE LOCAL'))
            with self.engine.begin() as conn:
                tz_name = conn.execute(sql.text('SHOW TIME ZONE')).scalar()
                assert isinstance(tz_name, str)
                self._logger.info(f'Database time zone is now: LOCAL; IANA equivalent is (from Postgres): {tz_name}')
                self._default_time_zone = ZoneInfo(tz_name)
        else:
            with self.engine.begin() as conn:
                # Explicit default time zone.
                conn.execute(sql.text(f"SET TIME ZONE '{tz.key}'"))
                self._logger.info(f'Database time zone is now: {tz.key}')
                self._default_time_zone = tz

    def configure_logging(
        self,
        *,
        to_stdout: Optional[bool] = None,
        level: Optional[int] = None,
        add: Optional[str] = None,
        remove: Optional[str] = None,
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

    def set_module_log_level(self, module: str, level: Optional[int]) -> None:
        if level is None:
            self._module_log_level.pop(module, None)
        else:
            self._module_log_level[module] = level

    def is_installed_package(self, package_name: str) -> bool:
        return self._installed_packages[package_name] is not None

    def _log_filter(self, record: logging.LogRecord) -> bool:
        if record.name == 'pixeltable':
            # accept log messages from a configured pixeltable module (at any level of the module hierarchy)
            path_parts = list(Path(record.pathname).parts)
            path_parts.reverse()
            max_idx = path_parts.index('pixeltable')
            for module_name in path_parts[:max_idx]:
                if module_name in self._module_log_level and record.levelno >= self._module_log_level[module_name]:
                    return True
        if record.levelno >= self._default_log_level:
            return True
        else:
            return False

    def _set_up(self, echo: bool = False, reinit_db: bool = False) -> None:
        if self._initialized:
            return

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self._initialized = True
        home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        assert self._home is None or self._home == home
        self._home = home
        self._config_file = Path(os.environ.get('PIXELTABLE_CONFIG', str(self._home / 'config.yaml')))
        self._media_dir = self._home / 'media'
        self._file_cache_dir = self._home / 'file_cache'
        self._dataset_cache_dir = self._home / 'dataset_cache'
        self._log_dir = self._home / 'logs'
        self._tmp_dir = self._home / 'tmp'

        # Read in the config
        if os.path.isfile(self._config_file):
            with open(self._config_file, 'r') as stream:
                try:
                    self._config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    self._logger.error(f'Could not read config file: {self._config_file}')
                    self._config = {}
        else:
            self._config = {}

        # Disable spurious warnings
        warnings.simplefilter('ignore', category=TqdmWarning)
        if 'hide_warnings' in self._config and self._config['hide_warnings']:
            # Disable more warnings
            warnings.simplefilter('ignore', category=UserWarning)

        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        if not self._home.exists():
            # we don't have our logger set up yet, so print to stdout
            print(f'Creating a Pixeltable instance at: {self._home}')
            self._home.mkdir()
            # TODO (aaron-siegel) This is the existing behavior, but it seems scary. If something happens to
            # self._home, it will cause the DB to be destroyed even if pgdata is in an alternate location.
            # PROPOSAL: require `reinit_db` to be set explicitly to destroy the DB.
            reinit_db = True

        if not self._media_dir.exists():
            self._media_dir.mkdir()
        if not self._file_cache_dir.exists():
            self._file_cache_dir.mkdir()
        if not self._dataset_cache_dir.exists():
            self._dataset_cache_dir.mkdir()
        if not self._log_dir.exists():
            self._log_dir.mkdir()
        if not self._tmp_dir.exists():
            self._tmp_dir.mkdir()

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

        # empty tmp dir
        for path in glob.glob(f'{self._tmp_dir}/*'):
            os.remove(path)

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self._pgdata_dir = Path(os.environ.get('PIXELTABLE_PGDATA', str(self._home / 'pgdata')))

        # in pixeltable_pgserver.get_server(): cleanup_mode=None will leave db on for debugging purposes
        self._db_server = pixeltable_pgserver.get_server(self._pgdata_dir, cleanup_mode=None)
        self._db_url = self._db_server.get_uri(database=self._db_name, driver='psycopg')

        if reinit_db:
            if self._store_db_exists():
                self._drop_store_db()

        if not self._store_db_exists():
            self._logger.info(f'creating database at {self.db_url}')
            self._create_store_db()
            self._create_engine(echo=echo)
            from pixeltable.metadata import schema
            schema.Base.metadata.create_all(self._sa_engine)
            metadata.create_system_info(self._sa_engine)
        else:
            self._logger.info(f'found database {self.db_url}')
            self._create_engine(echo=echo)

        print(f'Connected to Pixeltable database at: {self.db_url}')

        self._default_time_zone: Optional[ZoneInfo] = None

        # Configure default time zone
        tz: Optional[ZoneInfo] = None
        tzname = os.environ.get('PXT_TIME_ZONE', self._config.get('pxt_time_zone', None))
        if tzname is not None:
            if not isinstance(tzname, str):
                self._logger.error(f'Invalid time zone specified in configuration.')
            else:
                try:
                    tz = ZoneInfo(tzname)
                except ZoneInfoNotFoundError:
                    self._logger.error(f'Invalid time zone specified in configuration: {tzname}')
        # Set the default time zone; we need to do this even if tz is None, in order to set the system time zone
        # properly
        self.default_time_zone = tz

        # we now have a home directory and db; start other services
        self._set_up_runtime()
        self.log_to_stdout(False)

    def _create_engine(self, echo: bool = False) -> None:
        self._sa_engine = sql.create_engine(self.db_url, echo=echo, future=True, isolation_level='AUTOCOMMIT')

    def _store_db_exists(self) -> bool:
        assert self._db_name is not None
        # don't try to connect to self.db_name, it may not exist
        db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(db_url, future=True)
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
        pg_db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(pg_db_url, future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                # use C collation to get standard C/Python-style sorting
                stmt = (
                    f"CREATE DATABASE {preparer.quote(self._db_name)} "
                    "ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
                )
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

        # enable pgvector
        store_db_url = self._db_server.get_uri(database=self._db_name, driver='psycopg')
        engine = sql.create_engine(store_db_url, future=True, isolation_level='AUTOCOMMIT')
        try:
            with engine.begin() as conn:
                conn.execute(sql.text('CREATE EXTENSION vector'))
        finally:
            engine.dispose()

    def _drop_store_db(self) -> None:
        assert self._db_name is not None
        db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(db_url, future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                # terminate active connections
                stmt = (f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{self._db_name}'
                    AND pid <> pg_backend_pid()
                """)
                conn.execute(sql.text(stmt))
                # drop db
                stmt = f'DROP DATABASE {preparer.quote(self._db_name)}'
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

    def _upgrade_metadata(self) -> None:
        metadata.upgrade_md(self._sa_engine)

    def get_client(self, name: str) -> Any:
        """
        Gets the client with the specified name, initializing it if necessary.

        Args:
            - name: The name of the client
        """
        cl = _registered_clients[name]
        if cl.client_obj is not None:
            return cl.client_obj  # Already initialized

        # Construct a client. For each client parameter, first check if the parameter is in the environment;
        # if not, look in Pixeltable config from `config.yaml`.

        init_kwargs: dict[str, str] = {}
        for param in cl.param_names:
            environ = f'{name.upper()}_{param.upper()}'
            if environ in os.environ:
                init_kwargs[param] = os.environ[environ]
            elif name.lower() in self._config and param in self._config[name.lower()]:
                init_kwargs[param] = self._config[name.lower()][param.lower()]
            if param not in init_kwargs or init_kwargs[param] == '':
                raise excs.Error(
                    f'`{name}` client not initialized: parameter `{param}` is not configured.\n'
                    f'To fix this, specify the `{environ}` environment variable, or put `{param.lower()}` in '
                    f'the `{name.lower()}` section of $PIXELTABLE_HOME/config.yaml.'
                )

        cl.client_obj = cl.init_fn(**init_kwargs)
        self._logger.info(f'Initialized `{name}` client.')
        return cl.client_obj

    def _start_web_server(self) -> None:
        """
        The http server root is the file system root.
        eg: /home/media/foo.mp4 is located at http://127.0.0.1:{port}/home/media/foo.mp4
        in windows, the server will translate paths like http://127.0.0.1:{port}/c:/media/foo.mp4
        This arrangement enables serving media hosted within _home,
        as well as external media inserted into pixeltable or produced by pixeltable.
        The port is chosen dynamically to prevent conflicts.
        """
        # Port 0 means OS picks one for us.
        self._httpd = make_server('127.0.0.1', 0)
        port = self._httpd.server_address[1]
        self._http_address = f'http://127.0.0.1:{port}'

        def run_server():
            logging.log(logging.INFO, f'running web server at {self._http_address}')
            self._httpd.serve_forever()

        # Run the server in a separate thread
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    def _set_up_runtime(self) -> None:
        """Check for and start runtime services"""
        self._start_web_server()
        self._check_installed_packages()

    def _check_installed_packages(self) -> None:
        def check(package: str) -> None:
            if importlib.util.find_spec(package) is not None:
                self._installed_packages[package] = []
            else:
                self._installed_packages[package] = None

        check('toml')
        check('datasets')
        check('torch')
        check('torchvision')
        check('transformers')
        check('sentence_transformers')
        check('whisper')
        check('yolox')
        check('whisperx')
        check('boto3')
        check('fitz')  # pymupdf
        check('pyarrow')
        check('spacy')  # TODO: deal with en-core-web-sm
        if self.is_installed_package('spacy'):
            import spacy

            self._spacy_nlp = spacy.load('en_core_web_sm')
        check('tiktoken')
        check('openai')
        check('anthropic')
        check('together')
        check('fireworks')
        check('label_studio_sdk')
        check('openpyxl')

    def require_package(self, package: str, min_version: Optional[list[int]] = None) -> None:
        assert package in self._installed_packages
        if self._installed_packages[package] is None:
            raise excs.Error(f'Package {package} is not installed')
        if min_version is None:
            return

        # check whether we have a version >= the required one
        if not self._installed_packages[package]:
            m = importlib.import_module(package)
            module_version = [int(x) for x in m.__version__.split('.')]
            self._installed_packages[package] = module_version
        installed_version = self._installed_packages[package]
        if len(min_version) < len(installed_version):
            normalized_min_version = min_version + [0] * (len(installed_version) - len(min_version))
        if any([a < b for a, b in zip(installed_version, normalized_min_version)]):
            raise excs.Error(
                (
                    f'The installed version of package {package} is {".".join(str(v) for v in installed_version)}, '
                    f'but version  >={".".join(str(v) for v in min_version)} is required'
                )
            )

    def num_tmp_files(self) -> int:
        return len(glob.glob(f'{self._tmp_dir}/*'))

    def create_tmp_path(self, extension: str = '') -> Path:
        return self._tmp_dir / f'{uuid.uuid4()}{extension}'

    @property
    def home(self) -> Path:
        assert self._home is not None
        return self._home

    @property
    def media_dir(self) -> Path:
        assert self._media_dir is not None
        return self._media_dir

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

    @property
    def spacy_nlp(self) -> spacy.Language:
        assert self._spacy_nlp is not None
        return self._spacy_nlp


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
    - Otherwise, look for 'api_key' in the 'my_client' section of config.yaml.

    If all config parameters are found, Pixeltable calls the initialization function;
    otherwise it throws an exception.

    Args:
        - name (str): The name of the API client (e.g., 'openai' or 'label-studio').
    """
    def decorator(fn: Callable) -> None:
        global _registered_clients
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        _registered_clients[name] = ApiClient(init_fn=fn, param_names=param_names)

    return decorator


_registered_clients: dict[str, ApiClient] = {}


@dataclass
class ApiClient:
    init_fn: Callable
    param_names: list[str]
    client_obj: Optional[Any] = None
