import datetime
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
import sqlalchemy as sql
from sqlalchemy_utils.functions import database_exists, create_database, drop_database
import psycopg2
import docker
import logging
import sys
import platform

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
        self._img_dir: Optional[Path] = None  # computed images
        self._filecache_dir: Optional[Path] = None  # cached media files with external URL
        self._log_dir: Optional[Path] = None  # log files
        self._tmp_dir: Optional[Path] = None  # any tmp files
        self._sa_engine: Optional[sql.engine.base.Engine] = None
        self._db_name: Optional[str] = None
        self._db_user: Optional[str] = None
        self._db_password: Optional[str] = None
        self._db_name: Optional[str] = None
        self._db_port: Optional[int] = None
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

    def db_url(self, hide_passwd: bool = False) -> str:
        assert self._db_user is not None
        assert self._db_password is not None
        assert self._db_name is not None
        assert self._db_port is not None
        return (
            f'postgresql://{self._db_user}:{"*****" if hide_passwd else self._db_password}'
            f'@localhost:{self._db_port}/{self._db_name}'
        )

    @property
    def db_service_url(self) -> str:
        assert self._db_user is not None
        assert self._db_password is not None
        assert self._db_port is not None
        return f'postgresql://{self._db_user}:{self._db_password}@localhost:{self._db_port}'

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
        self.log_to_stdout(True)
        home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        assert self._home is None or self._home == home
        self._home = home
        self._img_dir = self._home / 'images'
        self._filecache_dir = self._home / 'filecache'
        self._log_dir = self._home / 'logs'
        self._tmp_dir = self._home / 'tmp'
        self._cache_dir = self._home / 'cache'

        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self._db_user = os.environ.get('PIXELTABLE_DB_USER', 'postgres')
        self._db_password = os.environ.get('PIXELTABLE_DB_PASSWORD', 'pgpassword')
        self._db_port = os.environ.get('PIXELTABLE_DB_PORT', '6543')

        if not self._home.exists():
            msg = f'setting up Pixeltable at {self._home}, db at {self.db_url(hide_passwd=True)}'
            # we don't have our logger set up yet, so print to stdout
            print(msg)
            self._home.mkdir()
            init_home_dir = True
        else:
            init_home_dir = False

        if not self._img_dir.exists():
            self._img_dir.mkdir()
        if not self._filecache_dir.exists():
            self._filecache_dir.mkdir()
        if not self._log_dir.exists():
            self._log_dir.mkdir()
        if not self._tmp_dir.exists():
            self._tmp_dir.mkdir()
        if not self._cache_dir.exists():
            self._cache_dir.mkdir()

        # configure _logger to log to a file
        self._logfilename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        fh = logging.FileHandler(self._log_dir / self._logfilename, mode='w')
        fh.setFormatter(logging.Formatter(self._log_fmt_str))
        self._logger.addHandler(fh)
        sql_logger = logging.getLogger('sqlalchemy.engine')
        sql_logger.setLevel(logging.INFO)
        sql_logger.addHandler(fh)

        # we now have a home directory; start runtime containers
        self._set_up_runtime()

        if init_home_dir:
            self.tear_down()

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

        self.log_to_stdout(False)

    def _set_up_postgres(self) -> None:
        cl = docker.from_env()
        try:
            self._store_container = cl.containers.get('pixeltable-store')
            self._logger.info('found store container')
        except docker.errors.NotFound:
            self._logger.info('starting store container')
            self._store_container = cl.containers.run(
                self._postgres_image(),
                detach=True,
                name='pixeltable-store',
                ports={'5432/tcp': self._db_port},
                environment={
                    'POSTGRES_USER': self._db_user,
                    'POSTGRES_PASSWORD': self._db_password,
                    'POSTGRES_DB': self._db_name,
                    'PGDATA': '/var/lib/postgresql/data',
                },
                volumes={
                    str(self._home / 'pgdata'): {'bind': '/var/lib/postgresql/data', 'mode': 'rw'},
                },
                remove=True,
            )
            self._wait_for_postgres()

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
        self._set_up_postgres()

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

    def _postgres_is_up(self) -> bool:
        """
        Returns true if the service is up, false otherwise.
        """
        try:
            conn = psycopg2.connect(self.db_service_url)
            conn.close()
            self._logger.info(f'connected to {self.db_service_url}')
            return True
        except psycopg2.OperationalError:
            return False

    def _wait_for_postgres(self, num_attempts: int = 20) -> None:
        """
        Waits for the service to be up.
        """
        i = 0
        while not self._postgres_is_up() and i < num_attempts:
            self._logger.info('waiting for store container to start...')
            time.sleep(i + 1)
            i += 1
        if not self._postgres_is_up():
            self._logger.info('could not find store container')
            raise RuntimeError(f'Postgres is not running: {self.db_service_url}')

    def _is_apple_cpu(self) -> bool:
        return sys.platform == 'darwin' and platform.processor() == 'arm'

    def _postgres_image(self) -> str:
        if self._is_apple_cpu():
            return 'ankane/pgvector:latest'
            #return 'arm64v8/postgres:15-alpine'
        else:
            #return 'postgres:15-alpine'
            return 'ankane/pgvector:latest'

    def tear_down(self) -> None:
        if database_exists(self.db_url()):
            drop_database(self.db_url())

    @property
    def home(self) -> Path:
        assert self._home is not None
        return self._home

    @property
    def img_dir(self) -> Path:
        assert self._img_dir is not None
        return self._img_dir

    @property
    def filecache_dir(self) -> Path:
        assert self._filecache_dir is not None
        return self._filecache_dir

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