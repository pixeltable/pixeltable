from typing import Optional
from pathlib import Path
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
        self._tmp_video_dir: Optional[Path] = None
        self._sa_engine: Optional[sql.engine.base.Engine] = None
        self._db_name: Optional[str] = None

    def set_up(
        self, home_parent: Optional[Path] = Path.home(), db_name: str = 'pixeltable', echo: bool = False
    ) -> None:
        self.set_home(home_parent / '.pixeltable')
        if self._home.exists() and not self._home.is_dir():
            raise RuntimeError(f'{self._home} is not a directory')

        self._db_name = db_name
        db_url = f'postgresql:///{self._db_name}'

        if not self._home.exists():
            print(f'creating {self._home}')
            self._home.mkdir()
            self._img_dir.mkdir()
            self._nnidx_dir.mkdir()
            self._tmp_video_dir.mkdir()
            self.tear_down()
            if not database_exists(db_url):
                create_database(db_url)
            self._sa_engine = sql.create_engine(db_url, echo=echo, future=True)
            from pixeltable import store
            store.Base.metadata.create_all(self._sa_engine)
        else:
            if self._sa_engine is None:
                self._sa_engine = sql.create_engine(db_url, echo=echo, future=True)

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
        self._tmp_video_dir = self._home / 'tmp_videos'

    @property
    def img_dir(self) -> Path:
        assert self._img_dir is not None
        return self._img_dir

    @property
    def nnidx_dir(self) -> Path:
        assert self._nnidx_dir is not None
        return self._nnidx_dir

    @property
    def tmp_video_dir(self) -> Path:
        assert self._tmp_video_dir is not None
        return self._tmp_video_dir

    @property
    def engine(self) -> sql.engine.base.Engine:
        assert self._sa_engine is not None
        return self._sa_engine
