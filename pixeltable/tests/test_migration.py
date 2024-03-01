import glob
import logging
import subprocess

from sqlalchemy.orm import declarative_base
from sqlalchemy_utils.functions import create_database, database_exists, drop_database

import pixeltable as pt
from pixeltable.env import Env

import sqlalchemy as sql

_logger = logging.getLogger('pixeltable')


class TestMigration:

    def test_db_migration(self, init_env) -> None:
        env = Env.get()
        for dump_file in glob.glob('pixeltable/tests/data/dbdumps/*.dump'):
            _logger.info(f'Testing migration from DB dump {dump_file}.')
            _logger.info(f'DB URL: {env.db_url}')
            # (asiegel) This will drop all tables in the `test` db. It would be cleaner to drop
            # and recreate the database, but that's not possible the way the unit tests are
            # currently set up, since the Env object is shared statically across all tests.
            # [We can't use `Catalog.clear()` here, since we also need to destroy `systeminfo` etc]
            sql_md = declarative_base().metadata
            sql_md.reflect(Env.get().engine)
            sql_md.drop_all(bind=Env.get().engine)
            with open(dump_file, 'r') as dump:
                subprocess.run(
                    ['/usr/local/bin/pg_restore', '-d', env.db_url, '-U', 'postgres'],
                    stdin=dump,
                    check=True
                )
            # TODO (asiegel) This will test that the migration succeeds without raising any exceptions.
            # We should also add some assertions to sanity-check the outcome.
            _ = pt.Client()
