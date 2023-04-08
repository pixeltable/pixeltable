from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import logging

import sqlalchemy.orm as orm

from pixeltable import catalog, store
from pixeltable.env import Env
from pixeltable.function import FunctionRegistry
from pixeltable import exceptions as exc
from pixeltable.utils.filecache import FileCache
from pixeltable.utils import store_utils

__all__ = [
    'Client',
]


class Client:
    """
    Client for interacting with a Pixeltable environment.
    """

    def __init__(self) -> None:
        self.db_cache: Dict[str, catalog.Db] = {}
        Env.get().set_up()

    def logging(
            self, *, to_stdout: Optional[bool] = None, level: Optional[int] = None,
            add: Optional[str] = None, remove: Optional[str] = None
    ) -> None:
        """
        Configure logging.

        to_stdout: if True, also log to stdout
        level: default log level
        add: comma-separated list of (module name, log level) tuples
        remove: comma-separated list of module names
        """
        if to_stdout is not None:
            Env.get().log_to_stdout(to_stdout)
        if level is not None:
            Env.get().set_log_level(level)
        if add is not None:
            for module, level in [t.split(':') for t in add.split(',')]:
                Env.get().set_module_log_level(module, int(level))
        if remove is not None:
            for module in remove.split(','):
                Env.get().set_module_log_level(module, None)
        if to_stdout is None and level is None and add is None and remove is None:
            Env.get().print_log_config()

    def list_functions(self) -> pd.DataFrame:
        func_info = FunctionRegistry.get().list_functions()
        paths = ['.'.join(info.fqn.split('.')[:-1]) for info in func_info]
        names = [info.fqn.split('.')[-1] for info in func_info]
        pd_df = pd.DataFrame({
            'Path': paths,
            'Name': names,
            'Parameters': [
                ', '.join([p[0] + ': ' + str(p[1]) for p in info.signature.parameters]) for info in func_info
            ],
            'Return Type': [str(info.signature.return_type) for info in func_info],
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

    def drop_db(self, name: str, force: bool = False, ignore_errors: bool = False) -> None:
        if not force:
            return
        if name in self.db_cache:
            self.db_cache[name].delete()
            del self.db_cache[name]
        else:
            try:
                db = catalog.Db.load(name)
            except exc.UnknownEntityError as e:
                if ignore_errors:
                    return
                else:
                    raise e
            db.delete()

    def get_db(self, name: str) -> catalog.Db:
        if name in self.db_cache:
            return self.db_cache[name]
        db = catalog.Db.load(name)
        self.db_cache[name] = db
        return db

    def list_dbs(self) -> List[str]:
        with orm.Session(Env.get().engine) as session:
            return [r[0] for r in session.query(store.Db.name)]

    # TODO: why is this not resolved?
    #def cache_stats(self) -> pd.io.formats.style.Styler:
    def cache_stats(self) -> Any:
        stats = FileCache.get().stats()
        hit_ratio = stats.num_hits / stats.num_requests
        df = pd.DataFrame(data={
            'a': ['# requests', '# hits', '# misses', 'hit %', '# evictions'],
            'b': [
                stats.num_requests, stats.num_hits, stats.num_requests - stats.num_hits, round(hit_ratio * 100),
                stats.num_evictions
            ]})
        return df.style.hide().hide(axis='columns').set_properties(**{'text-align': 'left'})

    # TODO: why is this not resolved?
    #def cache_util(self) -> pd.io.formats.style.Styler:
    def cache_util(self) -> Any:
        names = {(name[0], name[1]): (name[2], name[3], name[4]) for name in store_utils.column_names()}

        cache = FileCache.get()
        util = cache.stats().util
        db_names = [names[(entry.tbl_id, entry.col_id)][0] for entry in util]
        tbl_names = [names[(entry.tbl_id, entry.col_id)][1] for entry in util]
        col_names = [names[(entry.tbl_id, entry.col_id)][2] for entry in util]
        sizes = [entry.total_size for entry in util]
        num_files = [entry.num_files for entry in util]
        rel_sizes = [round(100 * entry.total_size / cache.capacity, 2) for entry in util]
        df = pd.DataFrame(data={
            'Db': db_names,
            'Table': tbl_names,
            'Column': col_names,
            'Total Size': sizes,
            '# Files': num_files,
            '% Capacity': rel_sizes,
        })
        return df.style.hide().set_properties(**{'text-align': 'left'})
