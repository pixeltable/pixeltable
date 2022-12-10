from typing import Optional
from pathlib import Path
import sqlalchemy as sql
from sqlalchemy_utils.functions import database_exists, create_database, drop_database


__home: Optional[Path] = None
__db_path: Optional[Path] = None
__img_dir: Optional[Path] = None
__nnidx_dir: Optional[Path] = None
__sa_engine: Optional[sql.engine.base.Engine] = None

def get_home() -> Path:
    if __home is None:
        # initialize to defaults
        set_home(Path.home / '.pixeltable')
    assert __home is not None
    return __home

def get_db_path() -> Path:
    assert __db_path is not None
    return __db_path

def get_img_dir() -> Path:
    assert __img_dir is not None
    return __img_dir

def get_nnidx_dir() -> Path:
    assert __nnidx_dir is not None
    return __nnidx_dir

def set_home(home: Path) -> None:
    global __home, __db_path, __img_dir, __nnidx_dir
    __home = home
    __db_path = __home / 'db.sqlite3'
    __img_dir = __home / 'images'
    __nnidx_dir = __home / 'nnidxs'

def get_engine() -> sql.engine.base.Engine:
    assert __sa_engine is not None
    return __sa_engine

def init_env(home_parent: Optional[Path] = Path.home(), db_name: str = 'pixeltable', echo: bool = False) -> None:
    set_home(home_parent / '.pixeltable')
    if __home.exists() and not __home.is_dir():
        raise RuntimeError(f'{__home} is not a directory')

    global __sa_engine
    db_url = f'postgresql:///{db_name}'

    if not __home.exists():
        print(f'creating {__home}')
        __home.mkdir()
        _ = __home
        __img_dir.mkdir()
        __nnidx_dir.mkdir()
        teardown_env(db_name)
        if not database_exists(db_url):
            create_database(db_url)
        __sa_engine = sql.create_engine(db_url, echo=echo, future=True)
        from pixeltable import store
        store.Base.metadata.create_all(__sa_engine)
    else:
        if __sa_engine is None:
            __sa_engine = sql.create_engine(db_url, echo=echo, future=True)

def teardown_env(db_name: str) -> None:
    db_url = f'postgresql:///{db_name}'
    if database_exists(db_url):
        drop_database(db_url)
