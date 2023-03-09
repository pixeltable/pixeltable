from typing import Optional
from pathlib import Path
import shutil
import sqlalchemy as sql
from sqlalchemy_utils.functions import database_exists, create_database, drop_database


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

    def set_up(
            self, home_str: Optional[str], db_name: Optional[str], echo: bool = False,
            db_user: Optional[str] = None, db_password: Optional[str] = None, db_host: Optional[str] = None,
            db_port: Optional[int] = None, max_filecache_size: Optional[int] = None) -> None:
        home = Path.home() / '.pixeltable' if home_str is None else Path(home_str)
        if db_name is None:
            db_name = 'pixeltable'
        self.set_home(home)
        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        self._db_name = db_name
        if db_user is None:
            db_url = f'postgresql:///{self._db_name}'
        else:
            assert db_password is not None
            if db_host is None:
                db_host = 'localhost'
            if db_port is None:
                db_port = 5432
            db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{self._db_name}'

        if not self._home.exists():
            print(f'setting up Pixeltable at {self._home}, db at {db_url}')
            self._home.mkdir()
            self._img_dir.mkdir()
            self._nnidx_dir.mkdir()
            self._tmp_frames_dir.mkdir()
            self._filecache_dir.mkdir()
            self.tear_down()
            if not database_exists(db_url):
                create_database(db_url)
            self._sa_engine = sql.create_engine(db_url, echo=echo, future=True)
            from pixeltable import store
            store.Base.metadata.create_all(self._sa_engine)
        else:
            if not database_exists(db_url):
                raise RuntimeError(f'Database not found: {db_url}')
            if self._sa_engine is None:
                self._sa_engine = sql.create_engine(db_url, echo=echo, future=True)
            # discard tmp frames
            shutil.rmtree(self._tmp_frames_dir)
            self._tmp_frames_dir.mkdir()

        # a 10GB file cache by default
        self._max_filecache_size = 10*1024*1024*1024 if max_filecache_size is None else max_filecache_size

    def tear_down(self) -> None:
        db_url = f'postgresql:///{self._db_name}'
        if database_exists(db_url):
            drop_database(db_url)

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
