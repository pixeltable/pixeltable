import logging
import os
import time

import pytest

import pixeltable as pxt
from pixeltable.env import Env

from .utils import reload_catalog


class TestDbms:
    def test_create_table(self):
        if os.environ.get('PIXELTABLE_DB_CONNECT_STR') is None:
            logging.error('PIXELTABLE_DB_CONNECT_STR is not set, skipping test')
            return
        Env._init_env(reinit_db=False)
        pxt.init()
        Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)
        now = int(time.time() * 1000)
        dir_name = f'dir1_{now}'
        table_name = f'{dir_name}.table1_{now}'
        dir = pxt.create_dir(dir_name, if_exists='ignore')
        assert dir is not None
        schema = {'c1': pxt.String, 'c2': pxt.Int, 'c3': pxt.Float, 'c4': pxt.Timestamp}
        tbl = pxt.create_table(table_name, schema)
        assert tbl is not None
        schema = tbl.get_metadata()['schema']

        reload_catalog()
        tbl = pxt.get_table(table_name)
        assert tbl is not None
        assert tbl.get_metadata()['schema'] == schema

        dirs = pxt.list_dirs('', recursive=True)
        assert dir_name in dirs

        # cleanup
        pxt.drop_table(table_name)
        pxt.drop_dir(dir_name)

        dirs = pxt.list_dirs('', recursive=True)
        assert dir_name not in dirs

        with pytest.raises(pxt.Error, match='does not exist'):
            _ = pxt.get_table(table_name)
