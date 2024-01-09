import datetime
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
import sqlalchemy as sql
from sqlalchemy_utils.functions import database_exists, create_database, drop_database
import psycopg2
import pgserver
import docker
import logging
import sys
import platform
import glob

from pixeltable import metadata


class Env:
    """
    Store for runtime globals.
    """
    _instance: Optional['Env'] = None
    _log_fmt_str = '%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d: %(message)s'

    @classmethod
    def get(cls) -> 'Env':
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
        self._db_name: Optional[str] = None
        self._db_url: Optional[str] = None
        self._store_container: Optional[docker.models.containers.Container] = None
        self._nos_client: Optional[Any] = None
        self._openai_client: Optional[Any] = None

        # logging-related state
        self._logger = logging.getLogger('pixeltable')
        self._logger.setLevel(logging.DEBUG)  # allow everything to pass, we filter in _log_filter()
        self._logger.propagate = False
        self._logger.addFilter(self._log_filter)
        self._default_log_level = logging.INFO
        self._logfilename: Optional[str] = None
        self._log_to_stdout = False
        self._module_log_level: Dict[str, int] = {}  # module name -> log level

        # create logging handler to also log to stdout
        self._stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self._stdout_handler.setFormatter(logging.Formatter(self._log_fmt_str))
        self._inited = False

    def db_url(self, hide_passwd=False) -> str:
        return self._db_url
    
    @property
    def db_service_url(self) -> str:
        return self._db_url

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

    def _log_filter(self, record: logging.LogRecord) -> bool:
        if record.module in self._module_log_level and record.levelno >= self._module_log_level[record.module]:
            return True
        if record.levelno >= self._default_log_level:
            return True
        return False

    def set_up(self, echo: bool = False) -> None:
        if self._inited:
            return
        
        self._inited = True
        self.log_to_stdout(True)
        home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        assert self._home is None or self._home == home
        self._home = home
        self._media_dir = self._home / 'media'
        self._file_cache_dir = self._home / 'file_cache'
        self._dataset_cache_dir = self._home / 'dataset_cache'
        self._log_dir = self._home / 'logs'
        self._tmp_dir = self._home / 'tmp'

        env_pgdata = os.environ.get('PIXELTABLE_PGDATA')
        self._pgdata_dir = Path(env_pgdata) if env_pgdata is not None else (self._home / 'pgdata')

        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        if not self._home.exists():
            msg = f'setting up Pixeltable at {self._home}, db at {self.db_url(hide_passwd=True)}'
            # we don't have our logger set up yet, so print to stdout
            print(msg)
            self._home.mkdir()
            init_home_dir = True
        else:
            init_home_dir = False

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
        sql_logger = logging.getLogger('sqlalchemy.engine')
        sql_logger.setLevel(logging.INFO)
        sql_logger.addHandler(fh)

        # empty tmp dir
        for path in glob.glob(f'{self._tmp_dir}/*'):
            os.remove(path)

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self.db_server = pgserver.get_server(self._pgdata_dir)
        self._db_url = self.db_server.get_uri(database=self._db_name)

        if init_home_dir:
            if database_exists(self.db_url()):
                drop_database(self.db_url())

        if not database_exists(self.db_url()):
            self._logger.info('creating database')
            create_database(self.db_url())
            self._sa_engine = sql.create_engine(self.db_url(), echo=echo, future=True)
            from pixeltable.metadata import schema
            schema.Base.metadata.create_all(self._sa_engine)
            metadata.create_system_info(self._sa_engine)
            # enable pgvector
            with self._sa_engine.begin() as conn:
                conn.execute(sql.text('CREATE EXTENSION vector'))
        else:
            self._logger.info(f'found database {self.db_url(hide_passwd=True)}')
            if self._sa_engine is None:
                self._sa_engine = sql.create_engine(self.db_url(), echo=echo, future=True)
            metadata.upgrade_md(self._sa_engine)

        # we now have a home directory and db; start runtime containers
        self._set_up_runtime()
        self.log_to_stdout(False)

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
        nos_modules = create_nos_modules()
        import pixeltable.func as func
        for mod in nos_modules:
            func.FunctionRegistry.get().register_module(mod)

    def _create_openai_client(self) -> None:
        if not 'OPENAI_API_KEY' in os.environ:
            return
        import openai
        self._logger.info('connecting to OpenAI')
        self._openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def _set_up_runtime(self) -> None:
        """Check for and start runtime services"""
        try:
            import nos
            self._create_nos_client()
        except ImportError:
            pass
        try:
            import openai
            self._create_openai_client()
        except ImportError:
            pass

    def _is_apple_cpu(self) -> bool:
        return sys.platform == 'darwin' and platform.processor() == 'arm'

    def num_tmp_files(self) -> int:
        return len(glob.glob(f'{self._tmp_dir}/*'))

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
    def openai_client(self) -> Any:
        return self._openai_client