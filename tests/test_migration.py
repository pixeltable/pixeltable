import glob
import logging
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pixeltable_pgserver
import pytest
import sqlalchemy as sql
import toml
from sqlalchemy import orm

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.exprs import FunctionCall, Literal
from pixeltable.func import CallableFunction
from pixeltable.func.signature import Batch
from pixeltable.metadata import VERSION, SystemInfo
from pixeltable.metadata.converters.convert_48 import _table_modifier as _pk_table_modifier
from pixeltable.metadata.converters.util import convert_table_md
from pixeltable.metadata.notes import VERSION_NOTES
from pixeltable.metadata.schema import Table, TableSchemaVersion, TableVersion

from .conftest import clean_db
from .utils import (
    SAMPLE_IMAGE_URL,
    get_audio_files,
    get_documents,
    get_video_files,
    reload_catalog,
    rerun,
    skip_test_if_not_installed,
    validate_update_status,
)

_logger = logging.getLogger('pixeltable')


class TestMigration:
    @rerun(reruns=3, reruns_delay=8)  # Deal with occasional concurrency issues
    @pytest.mark.skipif(platform.system() == 'Windows', reason='Does not run on Windows')
    @pytest.mark.skipif(sys.version_info >= (3, 11), reason='Runs only on Python 3.10 (due to pickling issue)')
    def test_db_migration(self, init_env: None) -> None:
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('label_studio_sdk')

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
            info_file = dump_file.removesuffix('.dump.gz') + '-info.toml'
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
            clean_db(drop_md_tables=True)
            with open(dump_file, 'rb') as dump:
                gunzip_process = subprocess.Popen(['gunzip', '-c'], stdin=dump, stdout=subprocess.PIPE)
                subprocess.run(
                    [pg_restore_binary, '-d', db_url, '-U', 'postgres'], stdin=gunzip_process.stdout, check=True
                )

            with orm.Session(env.engine) as session:
                md = session.query(SystemInfo).one().md
                assert md['schema_version'] == old_version

            # Older database artifacts may contain references to UDFs that exist solely as part of the DB migration
            # test framework (and are not part of the Pixeltable UDF library), but have been moved or renamed. We
            # perform a manual database "migration" to alter these specific UDF names before proceeding with the
            # main part of the migration test.
            with orm.Session(env.engine) as session:
                convert_table_md(env.engine, substitution_fn=self.__substitute_md)

            # make sure we run the env db setup
            Env._init_env()
            env = Env.get()

            with orm.Session(env.engine) as session:
                md = session.query(SystemInfo).one().md
                assert md['schema_version'] == VERSION

            # Most DB artifacts were created using Python 3.9, but there is a pickling incompatibility between
            # Python 3.9 and 3.10 that affects the specific UDF `test_udf_stored_batched`. Eventually pickled UDFs
            # will go away; until we find a better solution, the workaround is to surgically replace references to
            # `test_udf_stored_batched` in the DB artifact metadata with a non-pickled variant.
            # TODO: Remove this workaround once we implement a better solution for dealing with legacy pickled UDFs.
            with orm.Session(env.engine) as session:
                convert_table_md(env.engine, substitution_fn=self.__replace_pickled_udfs)

            try:
                reload_catalog()

                # TODO: We need many more of these sorts of checks.
                if 12 <= old_version <= 14:
                    self._run_v12_tests()
                if 13 <= old_version <= 14:
                    self._run_v13_tests()
                if old_version == 14:
                    self._run_v14_tests()
                if old_version >= 15:
                    self._run_v15_tests()
                if old_version >= 17:
                    self._run_v17_tests()
                if old_version >= 19:
                    self._run_v19_tests(old_version)
                if old_version >= 30:
                    self._run_v30_tests()
                if old_version >= 33:
                    self._verify_v33()
                if old_version >= 45:
                    self._verify_v45()
                if old_version == 48:
                    self._verify_v49()
                # self._verify_v24(old_version)

                pxt.drop_table('sample_table', force=True)

            except Exception as e:
                raise RuntimeError(
                    f'Migration test failed on version {old_version} with `{e.__class__.__qualname__}`: {e}'
                ) from e

        _logger.info(f'Verified DB dumps with versions: {versions_found}')
        assert VERSION in versions_found, (
            f'No DB dump found for current schema version {VERSION}. You can generate one with:\n'
            f'`rm target/*.dump.gz target/*.toml; python tool/create_test_db_dump.py'
            f' && mv target/*.dump.gz target/*.toml tests/data/dbdumps`'
        )
        assert VERSION in VERSION_NOTES, (
            f'No version notes found for current schema version {VERSION}. '
            f'Please add them to pixeltable/metadata/notes.py.'
        )

    @classmethod
    def _run_v12_tests(cls) -> None:
        """Tests that apply to DB artifacts of version 12-14."""
        pxt.get_table('sample_table').describe()
        pxt.get_table('views/sample_view').describe()
        pxt.get_table('views/sample_snapshot').describe()

    @classmethod
    def _run_v13_tests(cls) -> None:
        """Tests that apply to DB artifacts of version 13-14."""
        t = pxt.get_table('views/empty_view')
        # Test that the batched function is properly loaded as batched
        expr = t['batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

    @classmethod
    def _run_v14_tests(cls) -> None:
        """Tests that apply to DB artifacts of version ==14."""
        t = pxt.get_table('views/sample_view')
        # Test that stored batched functions are properly loaded as batched
        expr = t['test_udf_batched'].col.value_expr
        assert isinstance(expr, FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched

    @classmethod
    def _run_v15_tests(cls) -> None:
        """Tests that apply to DB artifacts of version 15+."""
        # Test that computed column metadata of tables and views loads properly by forcing
        # the tables to describe themselves
        pxt.get_table('base_table').describe()
        pxt.get_table('views/view').describe()
        pxt.get_table('views/snapshot').describe()
        pxt.get_table('views/view_of_views').describe()
        pxt.get_table('views/empty_view').describe()

        v = pxt.get_table('views/view')
        e = pxt.get_table('views/empty_view')

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

        # Test that InlineLists are properly loaded
        inline_list_exprs = (
            v.where(v.c2 == 19).select(v.base_table_inline_list_exprs).head(1)['base_table_inline_list_exprs'][0]
        )
        assert inline_list_exprs == ['test string 19', ['test string 19', 19]]
        inline_list_mixed = (
            v.where(v.c2 == 19).select(v.base_table_inline_list_mixed).head(1)['base_table_inline_list_mixed'][0]
        )
        assert inline_list_mixed == [1, 'a', 'test string 19', [1, 'a', 'test string 19'], 1, 'a']

        # Test that InlineDicts are properly loaded
        inline_dict = v.where(v.c2 == 19).select(v.base_table_inline_dict).head(1)['base_table_inline_dict'][0]
        assert inline_dict == {'int': 22, 'dict': {'key': 'val'}, 'expr': 'test string 19'}

    @classmethod
    def _run_v17_tests(cls) -> None:
        from pixeltable.io.external_store import MockProject
        from pixeltable.io.label_studio import LabelStudioProject

        t = pxt.get_table('base_table')
        v = pxt.get_table('views/view')

        # Test that external stores are loaded properly.
        assert len(v.external_stores()) == 2
        stores = list(v._tbl_version.get().external_stores.values())
        assert len(stores) == 2
        store0 = stores[0]
        assert isinstance(store0, MockProject)
        assert store0.get_export_columns() == {'int_field': ts.IntType()}
        assert store0.get_import_columns() == {'str_field': ts.StringType()}
        assert store0.col_mapping == {v.view_test_udf.col.handle: 'int_field', t.c1.col.handle: 'str_field'}
        store1 = stores[1]
        assert isinstance(store1, LabelStudioProject)
        assert store1.project_id == 4171780

        # Test that the stored proxies were retained properly
        assert len(store1.stored_proxies) == 1
        assert t.base_table_image_rot.col.handle in store1.stored_proxies

    @classmethod
    def _run_v19_tests(cls, version: int) -> None:
        assert version >= 19
        t = pxt.get_table('base_table')
        row = {
            'c1': 'test string 21',
            'c1n': 'test string 21',
            'c2': 21,
            'c3': 21.0,
            'c4': True,
            'c5': datetime.now(),
            'c6': {
                'f1': 'test string 21',
                'f2': 21,
                'f3': float(21.0),
                'f4': True,
                'f5': [1.0, 2.0, 3.0, 4.0],
                'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
            },
            'c7': [],
            'c8': SAMPLE_IMAGE_URL,
        }
        if version >= 45:
            row['c9'] = get_audio_files()[0]
            row['c10'] = get_video_files()[0]
            row['c11'] = get_documents()[0]
            row['c12'] = np.zeros((10,), dtype=np.float64)
            row['c13'] = uuid.uuid4()
            row['c14'] = datetime.now().date()
            row['c16'] = b'\xca\xfe'
            row['c17'] = np.ones((1, 2, 3), dtype=np.bool_)
            row['c18'] = np.zeros((2, 10), dtype=np.str_)
        status = t.insert([row])
        validate_update_status(status)
        inline_list_mixed = (
            t.where(t.c2 == 21).select(t.base_table_inline_list_mixed).head(1)['base_table_inline_list_mixed'][0]
        )
        assert inline_list_mixed == [1, 'a', 'test string 21', [1, 'a', 'test string 21'], 1, 'a']

    @staticmethod
    def __substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
        if k == 'path' and v == 'pixeltable.tool.embed_udf.clip_text_embed':
            return 'path', 'tool.embed_udf.clip_text_embed'
        return None

    @staticmethod
    def __replace_pickled_udfs(k: str | None, v: Any) -> tuple[str | None, Any] | None:
        # The following set of conditions uniquely identifies FunctionCall instances in the artifacts whose function
        # is `test_udf_stored_batched`. See comment above re: pickled UDFs in Python 3.10.
        # TODO: Remove this method once we implement a better solution for dealing with legacy pickled UDFs.
        if (
            isinstance(v, dict)
            and v.get('_classname') == 'FunctionCall'
            and 'id' in v['fn']
            and len(v['kwarg_idxs']) == 1
        ):
            del v['fn']['id']
            v['fn']['path'] = replacement_batched_udf.self_path
            v['fn']['signature'] = replacement_batched_udf.signature.as_dict()

        return k, v

    @classmethod
    def _run_v30_tests(cls) -> None:
        with Env.get().engine.begin() as conn:
            for row in conn.execute(sql.select(Table.id, Table.md)):
                tbl_id = str(row[0])
                table_md = row[1]
                assert table_md['tbl_id'] == tbl_id
            for row in conn.execute(sql.select(TableVersion)):
                tbl_id = str(row[0])
                version = row[1]
                table_version_md = row[2]
                assert table_version_md['tbl_id'] == tbl_id
                assert table_version_md['version'] == version
            for row in conn.execute(sql.select(TableSchemaVersion)):
                tbl_id = str(row[0])
                schema_version = row[1]
                table_schema_version_md = row[2]
                assert table_schema_version_md['tbl_id'] == tbl_id
                assert table_schema_version_md['schema_version'] == schema_version

    @classmethod
    def _verify_v33(cls) -> None:
        with Env.get().engine.begin() as conn:
            for row in conn.execute(sql.select(Table.md)):
                table_md = row[0]
                for col_md in table_md['column_md'].values():
                    assert col_md['is_pk'] is not None

    @classmethod
    def _verify_v49(cls) -> None:
        """Verify primary key index migration (v48→v49 converter).

        - pk_test_good: should have a PK index and retain its PK metadata.
        - pk_test_bad: PK index creation should have failed (duplicates), so PK metadata is erased.

        Only asserts on these tables if they exist in the dump (they were added alongside the
        v48→v49 converter; older dumps won't have them).
        """
        found_good = False
        found_bad = False
        with Env.get().engine.begin() as conn:
            for row in conn.execute(sql.select(Table.id, Table.md)):
                tbl_id, table_md = row[0], row[1]

                if table_md['name'] == 'pk_test_good':
                    found_good = True
                    # PK metadata should be intact
                    assert table_md['primary_index_md'] is not None
                    pk_col_ids = table_md['primary_index_md']['indexed_col_ids']
                    assert len(pk_col_ids) > 0
                    for col_id in pk_col_ids:
                        assert table_md['column_md'][str(col_id)]['is_pk'] is True

                    # Unique index should exist in PostgreSQL
                    store_name = f'tbl_{tbl_id.hex}'
                    idx_name = f'pk_idx_{tbl_id.hex}'
                    idx_row = conn.execute(
                        sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
                        {'tbl': store_name, 'idx': idx_name},
                    ).fetchone()
                    assert idx_row is not None, f'PK index {idx_name} should exist on {store_name}'

                elif table_md['name'] == 'pk_test_bad':
                    found_bad = True
                    # PK metadata should have been erased
                    assert table_md['primary_index_md'] is None
                    for col_md in table_md['column_md'].values():
                        assert col_md['is_pk'] is False

                    # No PK index should exist
                    store_name = f'tbl_{tbl_id.hex}'
                    idx_name = f'pk_idx_{tbl_id.hex}'
                    idx_row = conn.execute(
                        sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
                        {'tbl': store_name, 'idx': idx_name},
                    ).fetchone()
                    assert idx_row is None, f'PK index {idx_name} should NOT exist on {store_name}'

        if found_good or found_bad:
            assert found_good, 'pk_test_good table should be present if pk_test_bad is'
            assert found_bad, 'pk_test_bad table should be present if pk_test_good is'

    @classmethod
    def _verify_v45(cls) -> None:
        t = pxt.get_table('base_table')
        v = pxt.get_table('views.view')
        s = pxt.get_table('views.snapshot_non_pure')
        vv = pxt.get_table('views.view_of_views')
        no_comment = pxt.get_table('string_splitter')

        # Verify comment and custom_metadata for base_table
        assert t.get_metadata()['comment'] == 'This is a test table.'
        assert t.get_metadata()['custom_metadata'] == {'key': 'value'}

        # Verify column-level comment and custom_metadata
        assert t.get_metadata()['columns']['c1n']['comment'] == 'Nullable version of c1'
        assert t.get_metadata()['columns']['c8']['custom_metadata'] == {'source': 'test'}

        # Verify comment and custom_metadata for view
        assert v.get_metadata()['comment'] == 'This is a test view.'
        assert v.get_metadata()['custom_metadata'] == {'view_key': 'view_value'}

        # Verify comment and custom_metadata for snapshot_non_pure
        assert s.get_metadata()['comment'] == 'This is a test snapshot.'
        assert s.get_metadata()['custom_metadata'] == {'snapshot_key': 'snapshot_value'}
        # Verify the additional column in the non-pure snapshot
        assert 's1' in s.columns()

        # Verify comment and custom_metadata for view_of_views
        assert vv.get_metadata()['comment'] == 'This is a test view of views.'
        assert vv.get_metadata()['custom_metadata'] == {'view_of_views_key': 'view_of_views_value'}

        # TODO: Once we migrate we should have no more '' as comments
        assert no_comment.get_metadata()['comment'] in (None, '')
        assert no_comment.get_metadata()['custom_metadata'] in (None, '')


class TestPkMigration:
    """Integration tests for the v48->v49 primary key index migration converter.

    These tests create real tables in a live database, inject PK metadata without
    creating the corresponding PostgreSQL index, then run the converter directly
    to verify it handles both the success and failure cases correctly.
    """

    MAX_VERSION = 9223372036854775807  # 2^63 - 1 (must match convert_48.py and Table.MAX_VERSION)

    @staticmethod
    def _inject_pk_metadata_for_col(
        engine: sql.engine.Engine, table_name: str, pk_col_name: str
    ) -> tuple[uuid.UUID, int]:
        """Inject PK metadata directly into a table that was created without a PK.

        Returns (table_uuid, pk_col_id) for later verification.
        """
        with engine.begin() as conn:
            for row in conn.execute(sql.select(Table.id, Table.md)):
                tbl_id, table_md = row[0], row[1]
                if table_md['name'] != table_name:
                    continue

                target_col_id: int | None = None
                t = pxt.get_table(table_name)
                for col in t._tbl_version.get().cols:
                    if col.name == pk_col_name:
                        target_col_id = col.id
                        break

                assert target_col_id is not None, f'Column {pk_col_name!r} not found in table {table_name!r}'

                table_md['column_md'][str(target_col_id)]['is_pk'] = True
                next_id = max((int(k) for k in table_md.get('index_md', {})), default=-1) + 1
                table_md['primary_index_md'] = {
                    'id': next_id,
                    'name': f'pk{tbl_id.hex}',
                    'indexed_col_tbl_id': str(tbl_id),
                    'indexed_col_ids': [target_col_id],
                }

                conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=table_md))
                return tbl_id, target_col_id

        raise RuntimeError(f'Table {table_name!r} not found in metadata')

    @staticmethod
    def _get_table_md(engine: sql.engine.Engine, table_name: str) -> tuple[uuid.UUID, dict]:
        """Retrieve (tbl_id, table_md) for a table by name."""
        with engine.begin() as conn:
            for row in conn.execute(sql.select(Table.id, Table.md)):
                tbl_id, table_md = row[0], row[1]
                if table_md['name'] == table_name:
                    return tbl_id, table_md
        raise RuntimeError(f'Table {table_name!r} not found in metadata')

    @staticmethod
    def _pk_index_exists(engine: sql.engine.Engine, tbl_id: uuid.UUID) -> bool:
        """Check whether the pk_idx_<hex> index exists in PostgreSQL for the given table."""
        store_name = f'tbl_{tbl_id.hex}'
        idx_name = f'pk_idx_{tbl_id.hex}'
        with engine.begin() as conn:
            result = conn.execute(
                sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
                {'tbl': store_name, 'idx': idx_name},
            ).fetchone()
        return result is not None

    def test_pk_migration_good_table_gets_index(self, uses_db: None) -> None:
        """A table with unique PK values should get a unique index after migration."""
        engine = Env.get().engine
        t = pxt.create_table('pk_good', {'id': pxt.Required[pxt.Int], 'name': pxt.Required[pxt.String]})
        t.insert([{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}, {'id': 3, 'name': 'Charlie'}])
        tbl_id, pk_col_id = self._inject_pk_metadata_for_col(engine, 'pk_good', 'id')
        assert not self._pk_index_exists(engine, tbl_id), 'PK index should not exist before migration'

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert self._pk_index_exists(engine, tbl_id), 'PK index should exist after migration'
        _, table_md = self._get_table_md(engine, 'pk_good')
        assert table_md['primary_index_md'] is not None
        assert table_md['primary_index_md']['indexed_col_ids'] == [pk_col_id]
        assert table_md['column_md'][str(pk_col_id)]['is_pk'] is True

    def test_pk_migration_bad_table_erases_pk(self, uses_db: None) -> None:
        """A table with duplicate PK values should have its PK metadata erased after migration."""
        engine = Env.get().engine
        t = pxt.create_table('pk_bad', {'id': pxt.Required[pxt.Int], 'name': pxt.Required[pxt.String]})
        t.insert([{'id': 1, 'name': 'Alice'}, {'id': 1, 'name': 'Bob'}, {'id': 2, 'name': 'Charlie'}])
        tbl_id, _pk_col_id = self._inject_pk_metadata_for_col(engine, 'pk_bad', 'id')
        assert not self._pk_index_exists(engine, tbl_id), 'PK index should not exist before migration'

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert not self._pk_index_exists(engine, tbl_id), 'PK index should NOT exist after migration (duplicates)'
        _, table_md = self._get_table_md(engine, 'pk_bad')
        assert table_md['primary_index_md'] is None, 'primary_index_md should be None after failed index creation'
        for col_md in table_md['column_md'].values():
            assert col_md['is_pk'] is False, f'is_pk should be False for all columns, got True for col {col_md["id"]}'

    def test_pk_migration_skips_table_without_pk(self, uses_db: None) -> None:
        """A table without PK metadata should be left untouched by the converter."""
        engine = Env.get().engine
        t = pxt.create_table('no_pk', {'id': pxt.Required[pxt.Int], 'name': pxt.Required[pxt.String]})
        t.insert([{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}])
        tbl_id, table_md_before = self._get_table_md(engine, 'no_pk')
        assert table_md_before['primary_index_md'] is None

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert not self._pk_index_exists(engine, tbl_id), 'No PK index should be created for table without PK'
        _, table_md_after = self._get_table_md(engine, 'no_pk')
        assert table_md_after['primary_index_md'] is None

    def test_pk_migration_skips_existing_index(self, uses_db: None) -> None:
        """If the PK index already exists, the converter should skip it without error."""
        engine = Env.get().engine
        t = pxt.create_table('pk_exists', {'id': pxt.Required[pxt.Int], 'name': pxt.Required[pxt.String]})
        t.insert([{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}])
        tbl_id, pk_col_id = self._inject_pk_metadata_for_col(engine, 'pk_exists', 'id')

        # Pre-create the index so the converter encounters it already present
        store_name = f'tbl_{tbl_id.hex}'
        idx_name = f'pk_idx_{tbl_id.hex}'
        with engine.begin() as conn:
            conn.execute(
                sql.text(
                    f'CREATE UNIQUE INDEX {idx_name} ON {store_name} '
                    f'USING btree (col_{pk_col_id}) WHERE v_max = {self.MAX_VERSION}'
                )
            )
        assert self._pk_index_exists(engine, tbl_id), 'Pre-created index should exist'

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert self._pk_index_exists(engine, tbl_id), 'PK index should still exist after converter skips it'
        _, table_md = self._get_table_md(engine, 'pk_exists')
        assert table_md['primary_index_md'] is not None
        assert table_md['column_md'][str(pk_col_id)]['is_pk'] is True

    def test_pk_migration_string_column_uses_left_truncation(self, uses_db: None) -> None:
        """A PK on a string column should use left(col, 256) in the index expression."""
        engine = Env.get().engine
        t = pxt.create_table('pk_str', {'code': pxt.Required[pxt.String], 'val': pxt.Required[pxt.Int]})
        t.insert([{'code': 'aaa', 'val': 1}, {'code': 'bbb', 'val': 2}, {'code': 'ccc', 'val': 3}])
        tbl_id, pk_col_id = self._inject_pk_metadata_for_col(engine, 'pk_str', 'code')

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert self._pk_index_exists(engine, tbl_id), 'PK index should exist for string PK column'
        store_name = f'tbl_{tbl_id.hex}'
        idx_name = f'pk_idx_{tbl_id.hex}'
        with engine.begin() as conn:
            result = conn.execute(
                sql.text('SELECT indexdef FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
                {'tbl': store_name, 'idx': idx_name},
            ).fetchone()
        assert result is not None
        # PostgreSQL may render the function as left(...) or "left"(...) depending on quoting rules
        indexdef_lower = result[0].lower()
        assert 'left' in indexdef_lower and '256' in indexdef_lower, (
            f'String PK index should use left() truncation with length 256, got: {result[0]}'
        )
        _, table_md = self._get_table_md(engine, 'pk_str')
        assert table_md['primary_index_md'] is not None
        assert table_md['column_md'][str(pk_col_id)]['is_pk'] is True

    def test_pk_migration_idempotent(self, uses_db: None) -> None:
        """Running the converter twice should be safe (second run skips because index exists)."""
        engine = Env.get().engine
        t = pxt.create_table('pk_idem', {'id': pxt.Required[pxt.Int], 'name': pxt.Required[pxt.String]})
        t.insert([{'id': 10, 'name': 'X'}, {'id': 20, 'name': 'Y'}])
        tbl_id, pk_col_id = self._inject_pk_metadata_for_col(engine, 'pk_idem', 'id')

        convert_table_md(engine, table_modifier=_pk_table_modifier)
        assert self._pk_index_exists(engine, tbl_id)

        # Second run should skip gracefully
        convert_table_md(engine, table_modifier=_pk_table_modifier)
        assert self._pk_index_exists(engine, tbl_id)

        _, table_md = self._get_table_md(engine, 'pk_idem')
        assert table_md['primary_index_md'] is not None
        assert table_md['column_md'][str(pk_col_id)]['is_pk'] is True

    def test_pk_migration_good_and_bad_together(self, uses_db: None) -> None:
        """When both good and bad PK tables exist, the converter handles each correctly."""
        engine = Env.get().engine
        t_good = pxt.create_table('pk_both_good', {'id': pxt.Required[pxt.Int], 'data': pxt.Required[pxt.String]})
        t_good.insert([{'id': 1, 'data': 'a'}, {'id': 2, 'data': 'b'}, {'id': 3, 'data': 'c'}])
        t_bad = pxt.create_table('pk_both_bad', {'id': pxt.Required[pxt.Int], 'data': pxt.Required[pxt.String]})
        t_bad.insert([{'id': 1, 'data': 'x'}, {'id': 1, 'data': 'y'}, {'id': 2, 'data': 'z'}])
        good_tbl_id, good_col_id = self._inject_pk_metadata_for_col(engine, 'pk_both_good', 'id')
        bad_tbl_id, _bad_col_id = self._inject_pk_metadata_for_col(engine, 'pk_both_bad', 'id')

        convert_table_md(engine, table_modifier=_pk_table_modifier)

        assert self._pk_index_exists(engine, good_tbl_id), 'Good table should have PK index'
        _, good_md = self._get_table_md(engine, 'pk_both_good')
        assert good_md['primary_index_md'] is not None
        assert good_md['column_md'][str(good_col_id)]['is_pk'] is True

        assert not self._pk_index_exists(engine, bad_tbl_id), 'Bad table should NOT have PK index'
        _, bad_md = self._get_table_md(engine, 'pk_both_bad')
        assert bad_md['primary_index_md'] is None
        for col_md in bad_md['column_md'].values():
            assert col_md['is_pk'] is False


@pxt.udf(batch_size=4)
def replacement_batched_udf(strings: Batch[str], *, upper: bool = True) -> Batch[pxt.String]:
    return [string.upper() if upper else string.lower() for string in strings]
