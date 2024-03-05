import glob
import logging
import os
import subprocess

import pgserver

import pixeltable as pt
from pixeltable.env import Env
from pixeltable.tests.conftest import clean_db

_logger = logging.getLogger('pixeltable')


class TestMigration:

    def test_db_migration(self, init_env) -> None:
        env = Env.get()
        for dump_file in glob.glob('pixeltable/tests/data/dbdumps/*.dump'):
            _logger.info(f'Testing migration from DB dump {dump_file}.')
            _logger.info(f'DB URL: {env.db_url}')
            clean_db(restore_tables=False)
            pg_package_dir = os.path.dirname(pgserver.__file__)
            pg_restore_binary = f'{pg_package_dir}/pginstall/bin/pg_restore'
            _logger.info(f'Using pg_restore binary at: {pg_restore_binary}')
            with open(dump_file, 'r') as dump:
                subprocess.run(
                    [pg_restore_binary, '-d', env.db_url, '-U', 'postgres'],
                    stdin=dump,
                    check=True
                )
            # TODO (asiegel) This will test that the migration succeeds without raising any exceptions.
            # We should also add some assertions to sanity-check the outcome.
            _ = pt.Client()
