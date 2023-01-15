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
        self.home: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.img_dir: Optional[Path] = None
        self.nnidx_dir: Optional[Path] = None
        self.tmp_video_dir: Optional[Path] = None
        self.sa_engine: Optional[sql.engine.base.Engine] = None
        self.db_name: Optional[str] = None

    def set_up(
        self, home_parent: Optional[Path] = Path.home(), db_name: str = 'pixeltable', echo: bool = False
    ) -> None:
        self.set_home(home_parent / '.pixeltable')
        if self.home.exists() and not self.home.is_dir():
            raise RuntimeError(f'{self.home} is not a directory')

        self.db_name = db_name
        db_url = f'postgresql:///{self.db_name}'

        if not self.home.exists():
            print(f'creating {self.home}')
            self.home.mkdir()
            self.img_dir.mkdir()
            self.nnidx_dir.mkdir()
            self.tear_down()
            if not database_exists(db_url):
                create_database(db_url)
            self.sa_engine = sql.create_engine(db_url, echo=echo, future=True)
            from pixeltable import store
            store.Base.metadata.create_all(self.sa_engine)
        else:
            if self.sa_engine is None:
                self.sa_engine = sql.create_engine(db_url, echo=echo, future=True)

    def tear_down(self) -> None:
        db_url = f'postgresql:///{self.db_name}'
        if database_exists(db_url):
            drop_database(db_url)

    def set_home(self, home: Path) -> None:
        if self.home is not None:
            return
        self.home = home
        self.db_path = self.home / 'db.sqlite3'
        self.img_dir = self.home / 'images'
        self.nnidx_dir = self.home / 'nnidxs'
        self.tmp_video_dir = self.home / 'tmp_videos'

    def get_img_dir(self) -> Path:
        assert self.img_dir is not None
        return self.img_dir

    def get_nnidx_dir(self) -> Path:
        assert self.nnidx_dir is not None
        return self.nnidx_dir

    def get_engine(self) -> sql.engine.base.Engine:
        assert self.sa_engine is not None
        return self.sa_engine
