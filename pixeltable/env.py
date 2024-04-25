from __future__ import annotations

import datetime
import glob
import http.server
import importlib
import importlib.util
import logging
import os
import socketserver
import sys
import threading
import uuid
import warnings
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

import pgserver
import sqlalchemy as sql
import yaml
from sqlalchemy_utils.functions import database_exists, create_database, drop_database
from tqdm import TqdmWarning

import pixeltable.exceptions as excs
from pixeltable import metadata


class Env:
    """
    Store for runtime globals.
    """
    _instance: Optional[Env] = None
    _log_fmt_str = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'

    @classmethod
    def get(cls) -> Env:
        if cls._instance is None:
            cls._instance = Env()
        return cls._instance

    def __init__(self):
        self._home: Optional[Path] = None
        self._media_dir: Optional[Path] = None  # computed media files
        self._file_cache_dir: Optional[Path] = None  # cached media files with external URL
        self._dataset_cache_dir: Optional[Path] = None  # cached datasets (eg, pytorch or COCO)
        self._log_dir: Optional[Path] = None  # log files
        self._tmp_dir: Optional[Path] = None  # any tmp files
        self._sa_engine: Optional[sql.engine.base.Engine] = None
        self._pgdata_dir : Optional[Path] = None
        self._db_name: Optional[str] = None
        self._db_server: Optional[pgserver.PostgresServer] = None
        self._db_url: Optional[str] = None

        # info about installed packages that are utilized by some parts of the code;
        # package name -> version; version == []: package is installed, but we haven't determined the version yet
        self._installed_packages: Dict[str, Optional[List[int]]] = {}
        self._nos_client: Optional[Any] = None
        self._spacy_nlp: Optional[Any] = None  # spacy.Language
        self._httpd: Optional[socketserver.TCPServer] = None
        self._http_address: Optional[str] = None

        self._registered_clients: dict[str, Any] = {}

        # logging-related state
        self._logger = logging.getLogger('pixeltable')
        self._logger.setLevel(logging.DEBUG)  # allow everything to pass, we filter in _log_filter()
        self._logger.propagate = False
        self._logger.addFilter(self._log_filter)
        self._default_log_level = logging.INFO
        self._logfilename: Optional[str] = None
        self._log_to_stdout = False
        self._module_log_level: Dict[str, int] = {}  # module name -> log level

        # config
        self._config_file: Optional[Path] = None
        self._config: Optional[Dict[str, Any]] = None

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

    def print_log_config(self) -> None:
        print(f'logging to {self._logfilename}')
        print(f'{"" if self._log_to_stdout else "not "}logging to stdout')
        print(f'default log level: {logging.getLevelName(self._default_log_level)}')
        print(
            f'module log levels: '
            f'{",".join([name + ":" + logging.getLevelName(val) for name, val in self._module_log_level.items()])}')

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

    def set_up(self, echo: bool = False, reinit_db: bool = False) -> None:
        if self._initialized:
            return

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

        # empty tmp dir
        for path in glob.glob(f'{self._tmp_dir}/*'):
            os.remove(path)

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self._pgdata_dir = Path(os.environ.get('PIXELTABLE_PGDATA', str(self._home / 'pgdata')))

        # in pgserver.get_server(): cleanup_mode=None will leave db on for debugging purposes
        self._db_server = pgserver.get_server(self._pgdata_dir, cleanup_mode=None)
        self._db_url = self._db_server.get_uri(database=self._db_name)

        if reinit_db:
            if database_exists(self.db_url):
                drop_database(self.db_url)

        if not database_exists(self.db_url):
            self._logger.info(f'creating database at {self.db_url}')
            create_database(self.db_url)
            self._sa_engine = sql.create_engine(self.db_url, echo=echo, future=True)
            from pixeltable.metadata import schema
            schema.Base.metadata.create_all(self._sa_engine)
            metadata.create_system_info(self._sa_engine)
            # enable pgvector
            with self._sa_engine.begin() as conn:
                conn.execute(sql.text('CREATE EXTENSION vector'))
        else:
            self._logger.info(f'found database {self.db_url}')
            if self._sa_engine is None:
                self._sa_engine = sql.create_engine(self.db_url, echo=echo, future=True)

        print(f'Connected to Pixeltable database at: {self.db_url}')

        # we now have a home directory and db; start other services
        self._set_up_runtime()
        self.log_to_stdout(False)

        # Disable spurious warnings
        warnings.simplefilter("ignore", category=TqdmWarning)

    def upgrade_metadata(self) -> None:
        metadata.upgrade_md(self._sa_engine)

    def _create_nos_client(self) -> None:
        import nos
        self._logger.info('connecting to NOS')
        nos.init(logging_level=logging.DEBUG)
        self._nos_client = nos.client.InferenceClient()
        self._logger.info('waiting for NOS')
        self._nos_client.WaitForServer()

        # now that we have a client, we can create the module
        import importlib
        try:
            importlib.import_module('pixeltable.functions.nos')
            # it's already been created
            return
        except ImportError:
            pass
        from pixeltable.functions.util import create_nos_modules
        _ = create_nos_modules()

    def get_client(self, name: str, init: Callable, environ: Optional[str] = None) -> Any:
        """
        Gets the client with the specified name, using `init` to construct one if necessary.

        - name: The name of the client
        - init: A `Callable` with signature `fn(api_key: str) -> Any` that constructs a client object
        - environ: The name of the environment variable to use for the API key, if no API key is found in config
            (defaults to f'{name.upper()}_API_KEY')
        """
        if name in self._registered_clients:
            return self._registered_clients[name]

        if environ is None:
            environ = f'{name.upper()}_API_KEY'

        if name in self._config and 'api_key' in self._config[name]:
            api_key = self._config[name]['api_key']
        else:
            api_key = os.environ.get(environ)
        if api_key is None or api_key == '':
            raise excs.Error(f'`{name}` client not initialized (no API key configured).')

        client = init(api_key)
        self._registered_clients[name] = client
        self._logger.info(f'Initialized `{name}` client.')
        return client

    def _start_web_server(self) -> None:
        """
        The http server root is the file system root.
        eg: /home/media/foo.mp4 is located at http://127.0.0.1:{port}/home/media/foo.mp4
        This arrangement enables serving media hosted within _home,
        as well as external media inserted into pixeltable or produced by pixeltable.
        The port is chosen dynamically to prevent conflicts.
        """
        # Port 0 means OS picks one for us.
        address = ("127.0.0.1", 0)
        class FixedRootHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory='/', **kwargs)
        self._httpd = socketserver.TCPServer(address, FixedRootHandler)
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

        check('datasets')
        check('torch')
        check('torchvision')
        check('transformers')
        check('sentence_transformers')
        check('yolox')
        check('boto3')
        check('pyarrow')
        check('spacy')  # TODO: deal with en-core-web-sm
        if self.is_installed_package('spacy'):
            import spacy
            self._spacy_nlp = spacy.load('en_core_web_sm')
        check('tiktoken')
        check('openai')
        check('together')
        check('fireworks')
        check('nos')
        if self.is_installed_package('nos'):
            self._create_nos_client()

    def require_package(self, package: str, min_version: Optional[List[int]] = None) -> None:
        assert package in self._installed_packages
        if self._installed_packages[package] is None:
            raise excs.Error(f'Package {package} is not installed')
        if min_version is None:
            return

        # check whether we have a version >= the required one
        if self._installed_packages[package] == []:
            m = importlib.import_module(package)
            module_version = [int(x) for x in m.__version__.split('.')]
            self._installed_packages[package] = module_version
        installed_version = self._installed_packages[package]
        if len(min_version) < len(installed_version):
            normalized_min_version = min_version + [0] * (len(installed_version) - len(min_version))
        if any([a < b for a, b in zip(installed_version, normalized_min_version)]):
            raise excs.Error((
                f'The installed version of package {package} is {".".join([str[v] for v in installed_version])}, '
                f'but version  >={".".join([str[v] for v in min_version])} is required'))

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
    def nos_client(self) -> Any:
        return self._nos_client

    @property
    def spacy_nlp(self) -> Any:
        assert self._spacy_nlp is not None
        return self._spacy_nlp