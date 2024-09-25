import glob
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

import pixeltable_pgserver
import pytest
import sqlalchemy.orm as orm

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.exprs import FunctionCall, Literal
from pixeltable.func import CallableFunction
from pixeltable.metadata import VERSION, SystemInfo
from pixeltable.metadata.notes import VERSION_NOTES

from .conftest import clean_db
from .utils import reload_catalog, skip_test_if_not_installed

_logger = logging.getLogger('pixeltable')


class TestMigration:

    @pytest.mark.skipif(platform.system() == 'Windows', reason='Does not run on Windows')
    @pytest.mark.skipif(sys.version_info >= (3, 11), reason='Does not run on Python 3.11+ (due to pickling issue)')
    def test_db_migration(self, init_env) -> None:
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('toml')
        import toml

        env = Env.get()
        pg_package_dir = os.path.dirname(pixeltable_pgserver.__file__)
        pg_restore_binary = f'{pg_package_dir}/pginstall/bin/pg_restore'
        _logger.info(f'Using pg_restore binary at: {pg_restore_binary}')
        dump_files = glob.glob('tests/data/dbdumps/*.dump.gz')
        dump_files.sort()
        assert len(dump_files) > 0
        versions_found: list[int] = []

        for dump_file in dump_files:
            _logger.info(f'Testing migration from DB dump {dump_file}.')
            info_file = dump_file.rstrip('.dump.gz') + '-info.toml'
            with open(info_file, 'r', encoding='utf-8') as fp:
                info = toml.load(fp)
                old_version = info['pixeltable-dump']['metadata-version']
                assert isinstance(old_version, int)
                _logger.info(f'Migrating from version: {old_version} -> {VERSION}')
                versions_found.append(old_version)

            # For this test we need the raw DB URL, without a driver qualifier. (The driver qualifier is needed by
            # SQLAlchemy, but command-line Postgres won't know how to interpret it.)
            db_url = env._db_server.get_uri(env._db_name)
            _logger.info(f'DB URL: {db_url}')
            clean_db(restore_tables=False)
            with open(dump_file, 'rb') as dump:
                gunzip_process = subprocess.Popen(
                    ["gunzip", "-c"],
                    stdin=dump,
                    stdout=subprocess.PIPE
                )
                subprocess.run(
                    [pg_restore_binary, '-d', db_url, '-U', 'postgres'],
                    stdin=gunzip_process.stdout,
                    check=True
                )

            with orm.Session(env.engine) as session:
                md = session.query(SystemInfo).one().md
                assert md['schema_version'] == old_version

            env._upgrade_metadata()

            with orm.Session(env.engine) as session:
                md = session.query(SystemInfo).one().md
                assert md['schema_version'] == VERSION

            reload_catalog()

            # TODO(aaron-siegel) We need many more of these sorts of checks.
            if 13 <= old_version <= 14:
                self._run_v13_tests()
            if old_version == 14:
                self._run_v14_tests()
            if old_version >= 15:
                self._run_v15_tests()
            if old_version >= 17:
                self._run_v17_tests()

        _logger.info(f'Verified DB dumps with versions: {versions_found}')
        assert VERSION in versions_found, (
            f'No DB dump found for current schema version {VERSION}. You can generate one with:\n'
            f'`python pixeltable/tool/create_test_db_dump.py`\n'
            f'`mv target/*.dump.gz target/*.toml tests/data/dbdumps`')
        assert VERSION in VERSION_NOTES, (
            f'No version notes found for current schema version {VERSION}. '
            f'Please add them to pixeltable/metadata/notes.py.')

    @classmethod
    def _run_v13_tests(cls) -> None:
        """Tests that apply to DB artifacts of version 13-14."""
        t = pxt.get_table('views.empty_view')
        # Test that the batched function is properly loaded as batched
        expr = t['batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

    @classmethod
    def _run_v14_tests(cls) -> None:
        """Tests that apply to DB artifacts of version ==14."""
        t = pxt.get_table('views.sample_view')
        # Test that stored batched functions are properly loaded as batched
        expr = t['test_udf_batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

    @classmethod
    def _run_v15_tests(cls) -> None:
        """Tests that apply to DB artifacts of version 15+."""
        # Test that computed column metadata of tables and views loads properly by forcing
        # the tables to describe themselves
        pxt.get_table('base_table').describe()
        pxt.get_table('views.view').describe()
        pxt.get_table('views.snapshot').describe()
        pxt.get_table('views.view_of_views').describe()
        pxt.get_table('views.empty_view').describe()

        v = pxt.get_table('views.view')
        e = pxt.get_table('views.empty_view')

        # Test that batched functions are properly loaded as batched
        expr = e['empty_view_batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

        # Test that stored batched functions are properly loaded as batched
        expr = v['view_test_udf_batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

        # Test that timestamp literals are properly stored as aware datetimes
        expr = v['view_timestamp_const_1'].col.value_expr
        assert isinstance(expr, Literal) and isinstance(expr.val, datetime) and expr.val.tzinfo is not None

        # Test that timestamp columns are properly converted to aware columns (TIMESTAMPTZ in Postgres)
        ts1 = v.select(v.c5).head(1)[0]['c5']
        assert isinstance(ts1, datetime)
        assert ts1.tzinfo is not None  # ensure timestamps are aware

    @classmethod
    def _run_v17_tests(cls) -> None:
        from pixeltable.io.external_store import MockProject
        from pixeltable.io.label_studio import LabelStudioProject

        t = pxt.get_table('base_table')
        v = pxt.get_table('views.view')

        # Test that external stores are loaded properly.
        assert len(v.external_stores) == 2
        stores = list(v._tbl_version.external_stores.values())
        assert len(stores) == 2
        store0 = stores[0]
        assert isinstance(store0, MockProject)
        assert store0.get_export_columns() == {'int_field': pxt.IntType()}
        assert store0.get_import_columns() == {'str_field': pxt.StringType()}
        assert store0.col_mapping == {v.view_test_udf.col: 'int_field', t.c1.col: 'str_field'}
        store1 = stores[1]
        assert isinstance(store1, LabelStudioProject)
        assert store1.project_id == 4171780

        # Test that the stored proxies were retained properly
        assert len(store1.stored_proxies) == 1
        assert t.base_table_image_rot.col in store1.stored_proxies
