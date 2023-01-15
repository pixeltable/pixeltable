from typing import List, Dict

import sqlalchemy.orm as orm

from pixeltable import catalog, store
from pixeltable.env import Env

__all__ = [
    'Client',
]


class Client:
    """Missing docstring.
    """

    def __init__(self) -> None:
        self.db_cache: Dict[str, catalog.Db] = {}
        Env.get().set_up()

    def create_db(self, name: str) -> catalog.Db:
        db = catalog.Db.create(name)
        self.db_cache[name] = db
        return db

    def drop_db(self, name: str, force: bool = False) -> None:
        if not force:
            return
        if name in self.db_cache:
            self.db_cache[name].delete()
            del self.db_cache[name]
        else:
            catalog.Db.load(name).delete()

    def get_db(self, name: str) -> catalog.Db:
        if name in self.db_cache:
            return self.db_cache[name]
        db = catalog.Db.load(name)
        self.db_cache[name] = db
        return db

    def list_dbs(self) -> List[str]:
        with orm.Session(store.engine) as session:
            return [r[0] for r in session.query(store.Db.name)]
