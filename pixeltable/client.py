from typing import List, Dict
import os
import pandas as pd

import sqlalchemy.orm as orm

from pixeltable import catalog, store
from pixeltable.env import Env
from pixeltable.function import FunctionRegistry

__all__ = [
    'Client',
]


class Client:
    """Missing docstring.
    """

    def __init__(self) -> None:
        self.db_cache: Dict[str, catalog.Db] = {}
        Env.get().set_up(
            os.environ.get('PIXELTABLE_HOME'), os.environ.get('PIXELTABLE_DB'),
            db_user=os.environ.get('PIXELTABLE_DB_USER'), db_password=os.environ.get('PIXELTABLE_DB_PASSWORD'),
            db_host=os.environ.get('PIXELTABLE_DB_HOST'), db_port=os.environ.get('PIXELTABLE_DB_PORT'))

    def list_functions(self) -> pd.DataFrame:
        func_info = FunctionRegistry.get().list_functions()
        paths = ['.'.join(info.fqn.split('.')[:-1]) for info in func_info]
        names = [info.fqn.split('.')[-1] for info in func_info]
        pd_df = pd.DataFrame({
            'Path': paths,
            'Name': names,
            'Parameter Types': [', '.join([str(col_type) for col_type in info.param_types]) for info in func_info],
            'Return Type': [str(info.return_type) for info in func_info],
            'Is Agg': [info.is_agg for info in func_info],
            'Library': [info.is_library_fn for info in func_info],
        })
        pd_df = pd_df.style.set_properties(**{'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])  # center-align headings
        return pd_df.hide(axis='index')

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
