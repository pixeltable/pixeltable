import sqlite3

import pixeltable.constants as constants
from .client import Client
from .dataframe import DataFrame
from .catalog import Db, MutableTable, TableSnapshot
import pixeltable.store as store

__all__ = [
    'Client',
    'DataFrame',
    'Db',
    'MutableTable',
    'TableSnapshot',
]


# initialization
if constants.PIXELTABLE_HOME.exists() and not constants.PIXELTABLE_HOME.is_dir():
    raise RuntimeError(f'{constants.PIXELTABLE_HOME} is not a directory')
if not constants.PIXELTABLE_HOME.exists():
    print(f'creating {constants.PIXELTABLE_HOME}')
    constants.PIXELTABLE_HOME.mkdir()
    constants.PIXELTABLE_IMG_DIR.mkdir()
    with sqlite3.connect(constants.PIXELTABLE_DB) as conn:
        store.init_db()