import os
import time
from typing import Optional
from pathlib import Path
import importlib
import subprocess
import shutil
import sqlalchemy as sql
from sqlalchemy_utils.functions import database_exists, create_database, drop_database
import psycopg2
import docker


class Env:
    """
    Store for runtime globals.
    """
    _instance: Optional['Env'] = None

    @classmethod
    def get(cls) -> 'Env':
        if cls._instance is None:
            cls._instance = Env()
        return cls._instance

    def __init__(self):
        self._home: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._img_dir: Optional[Path] = None
        self._nnidx_dir: Optional[Path] = None
        self._tmp_frames_dir: Optional[Path] = None
        self._filecache_dir: Optional[Path] = None
        self._sa_engine: Optional[sql.engine.base.Engine] = None
        self._db_name: Optional[str] = None
        self._max_filecache_size: Optional[int] = None
        self._db_user: Optional[str] = None
        self._db_password: Optional[str] = None
        self._db_name: Optional[str] = None
        self._db_port: Optional[int] = None
        self._store_container: Optional[docker.models.containers.Container] = None

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

    def set_up(self, echo: bool = False) -> None:
        home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        self.set_home(home)
        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self._db_user = os.environ.get('PIXELTABLE_DB_USER', 'postgres')
        self._db_password = os.environ.get('PIXELTABLE_DB_PASSWORD', 'pgpassword')
        self._db_port = os.environ.get('PIXELTABLE_DB_PORT', '6543')
        self._max_filecache_size = int(os.environ.get('PIXELTABLE_MAX_FILECACHE_SIZE', f'{10 * 1024 * 1024 * 1024}'))

        if not self._home.exists():
            print(f'setting up Pixeltable at {self._home}, db at {self.db_url(hide_passwd=True)}')
            self._home.mkdir()
            init_home_dir = True
        else:
            init_home_dir = False

        # we now have a home directory; start runtime containers
        self._set_up_runtime()

        if init_home_dir:
            self._img_dir.mkdir()
            self._nnidx_dir.mkdir()
            self._tmp_frames_dir.mkdir()
            self._filecache_dir.mkdir()
            self.tear_down()
            if not database_exists(self.db_url()):
                print('creating database')
                create_database(self.db_url())
            print('creating engine')
            self._sa_engine = sql.create_engine(self.db_url(), echo=echo, future=True)
            from pixeltable import store
            store.Base.metadata.create_all(self._sa_engine)
        else:
            if not database_exists(self.db_url()):
                raise RuntimeError(f'Database not found: {self.db_url(hide_passwd=True)}')
            if self._sa_engine is None:
                self._sa_engine = sql.create_engine(self.db_url(), echo=echo, future=True)
            # discard tmp frames
            shutil.rmtree(self._tmp_frames_dir)
            self._tmp_frames_dir.mkdir()

    def _set_up_runtime(self) -> None:
        """
        Start store and runtime containers.
        """
        cl = docker.from_env()
        try:
            self._store_container = cl.containers.get('pixeltable-store')
        except docker.errors.NotFound:
            print('starting store container')
            self._store_container = cl.containers.run(
                'postgres:15-alpine',
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
                    str(self._home / 'pgdata') : {'bind': '/var/lib/postgresql/data', 'mode': 'rw'},
                },
                remove=True,
            )
            self._wait_for_postgres()
        return

    def _postgres_is_up(self) -> bool:
        """
        Returns true if the service is up, false otherwise.
        """
        try:
            conn = psycopg2.connect(self.db_service_url)
            conn.close()
            return True
        except psycopg2.OperationalError:
            return False

    def _wait_for_postgres(self, num_attempts: int = 20) -> None:
        """
        Waits for the service to be up.
        """
        i = 0
        while not self._postgres_is_up() and i < num_attempts:
            print('waiting for store container to start...')
            time.sleep(i + 1)
            i += 1
        if not self._postgres_is_up():
            raise RuntimeError(f'Postgres is not running: {self.db_service_url}')

    def tear_down(self) -> None:
        if database_exists(self.db_url()):
            drop_database(self.db_url())

    def set_home(self, home: Path) -> None:
        if self._home is not None:
            return
        self._home = home
        self._db_path = self._home / 'db.sqlite3'
        self._img_dir = self._home / 'images'
        self._nnidx_dir = self._home / 'nnidxs'
        self._tmp_frames_dir = self._home / 'tmp_frames'
        self._filecache_dir = self._home / 'filecache'

    @property
    def home(self) -> Path:
        assert self._home is not None
        return self._home

    @property
    def img_dir(self) -> Path:
        assert self._img_dir is not None
        return self._img_dir

    @property
    def nnidx_dir(self) -> Path:
        assert self._nnidx_dir is not None
        return self._nnidx_dir

    @property
    def tmp_frames_dir(self) -> Path:
        assert self._tmp_frames_dir is not None
        return self._tmp_frames_dir

    @property
    def filecache_dir(self) -> Path:
        assert self._filecache_dir is not None
        return self._filecache_dir

    @property
    def engine(self) -> sql.engine.base.Engine:
        assert self._sa_engine is not None
        return self._sa_engine

    @property
    def max_filecache_size(self) -> int:
        assert self._max_filecache_size is not None
        return self._max_filecache_size
