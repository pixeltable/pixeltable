import glob
import logging
import os
import platform
import subprocess
import sys

import pgserver
import pytest
import sqlalchemy.orm as orm

from pixeltable.env import Env
from pixeltable.metadata import SystemInfo
from .conftest import clean_db
from .utils import reload_catalog

_logger = logging.getLogger('pixeltable')


class TestMigration:

    @pytest.mark.skipif(platform.system() == 'Windows', reason='Does not run on Windows')
    @pytest.mark.skipif(sys.version_info >= (3, 11), reason='Does not run on Python 3.11+ (due to pickling issue)')
    def test_db_migration(self, init_env) -> None:
        env = Env.get()
        pg_package_dir = os.path.dirname(pgserver.__file__)
        pg_restore_binary = f'{pg_package_dir}/pginstall/bin/pg_restore'
        _logger.info(f'Using pg_restore binary at: {pg_restore_binary}')
        dump_files = glob.glob('pixeltable/tests/data/dbdumps/*.dump.gz')
        dump_files.sort()
        assert len(dump_files) > 0

        versions_tested = set()
        for dump_file in dump_files:
            _logger.info(f'Testing migration from DB dump {dump_file}.')
            _logger.info(f'DB URL: {env.db_url}')
            clean_db(restore_tables=False)
            with open(dump_file, 'rb') as dump:
                gunzip_process = subprocess.Popen(
                    ["gunzip", "-c"],
                    stdin=dump,
                    stdout=subprocess.PIPE
                )
                subprocess.run(
                    [pg_restore_binary, '-d', env.db_url, '-U', 'postgres'],
                    stdin=gunzip_process.stdout,
                    check=True
                )
            with orm.Session(env.engine, future=True) as session:
                system_info = session.query(SystemInfo).one().md
                md_version = system_info['schema_version']
                versions_tested.add(md_version)
            # TODO(aaron-siegel) This will test that the migration succeeds without raising any exceptions.
            # We should also add some assertions to sanity-check the outcome.
            reload_catalog()
