import glob
import json
import logging
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

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
from pixeltable.metadata.converters.convert_45 import _convert_table_and_versions
from pixeltable.metadata.converters.util import convert_table_md, convert_table_schema_version_md
from pixeltable.metadata.notes import VERSION_NOTES
from pixeltable.metadata.schema import Table, TableSchemaVersion, TableVersion

from .conftest import clean_db
from .utils import reload_catalog, rerun, skip_test_if_not_installed, validate_update_status

_logger = logging.getLogger('pixeltable')


class TestMigration:
    # @rerun(reruns=3, reruns_delay=8)  # Deal with occasional concurrency issues
    # @pytest.mark.skipif(platform.system() == 'Windows', reason='Does not run on Windows')
    # @pytest.mark.skipif(sys.version_info >= (3, 11), reason='Runs only on Python 3.10 (due to pickling issue)')
    def test_db_migration(self, init_env: None) -> None:
        # skip_test_if_not_installed('transformers')
        # skip_test_if_not_installed('label_studio_sdk')

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
            clean_db(restore_md_tables=False)
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
                convert_table_schema_version_md(env.engine, schema_column_updater=self.__replace_pickled_udfs)

            reload_catalog()

            # TODO(aaron-siegel) We need many more of these sorts of checks.
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
                self._run_v19_tests()
            if old_version >= 30:
                self._run_v30_tests()
            if old_version >= 45:
                self._verify_v45()

            pxt.drop_table('sample_table', force=True)

        _logger.info(f'Verified DB dumps with versions: {versions_found}')
        assert VERSION in versions_found, (
            f'No DB dump found for current schema version {VERSION}. You can generate one with:\n'
            f'`python tool/create_test_db_dump.py`\n'
            f'`mv target/*.dump.gz target/*.toml tests/data/dbdumps`'
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
    def _run_v19_tests(cls) -> None:
        t = pxt.get_table('base_table')
        status = t.insert(
            c1='test string 21',
            c1n='test string 21',
            c2=21,
            c3=21.0,
            c4=True,
            c5=datetime.now(),
            c6={
                'f1': 'test string 21',
                'f2': 21,
                'f3': float(21.0),
                'f4': True,
                'f5': [1.0, 2.0, 3.0, 4.0],
                'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
            },
            c7=[],
        )
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
    def __replace_pickled_udfs(column_md: dict) -> None:
        # The following set of conditions uniquely identifies FunctionCall instances in the artifacts whose function
        # is `test_udf_stored_batched`. See comment above re: pickled UDFs in Python 3.10.
        # TODO: Remove this method once we implement a better solution for dealing with legacy pickled UDFs.
        try:
            if (
                column_md['value_expr']['_classname'] == 'FunctionCall'
                and 'id' in column_md['value_expr']['fn']
                and len(column_md['value_expr']['kwarg_idxs']) == 1
            ):
                del column_md['value_expr']['fn']['id']
                column_md['value_expr']['fn']['path'] = replacement_batched_udf.self_path
                column_md['value_expr']['fn']['signature'] = replacement_batched_udf.signature.as_dict()
        except TypeError:
            return

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
    def _verify_v45(cls) -> None:
        with Env.get().engine.begin() as conn:
            for row in conn.execute(sql.select(Table.md)):
                pass  # TODO

    def test_convert_45(self) -> None:
        table_md = deepcopy(_TABLE_MD)
        schema_version_md = deepcopy(_SCHEMA_VERSION_MD)
        _convert_table_and_versions(table_md, schema_version_md)
        assert table_md == _TABLE_MD_EXPECTED
        assert set(schema_version_md.keys()) == set(_SCHEMA_VERSION_MD_EXPECTED.keys())
        for ver in schema_version_md:
            assert schema_version_md[ver] == _SCHEMA_VERSION_MD_EXPECTED[ver], ver


@pxt.udf(batch_size=4)
def replacement_batched_udf(strings: Batch[str], *, upper: bool = True) -> Batch[pxt.String]:
    return [string.upper() if upper else string.lower() for string in strings]


_TABLE_MD = json.loads("""
 {
     "name": "test",
     "user": null,
     "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
     "view_md": null,
     "view_sn": 0,
     "index_md": {
         "0": {
             "id": 0,
             "name": "idx0",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 0,
             "index_val_col_id": 2,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 0,
             "schema_version_drop": null,
             "index_val_undo_col_id": 3
         },
         "1": {
             "id": 1,
             "name": "idx1",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 1,
             "index_val_col_id": 4,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 0,
             "schema_version_drop": 2,
             "index_val_undo_col_id": 5
         },
         "2": {
             "id": 2,
             "name": "idx2",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 6,
             "index_val_col_id": 7,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 3,
             "schema_version_drop": null,
             "index_val_undo_col_id": 8
         }
     },
     "column_md": {
         "0": {
             "id": 0,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "1": {
             "id": 1,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "2": {
             "id": 2,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": {
                 "col_id": 0,
                 "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                 "_classname": "ColumnRef",
                 "tbl_version": null,
                 "reference_tbl": null,
                 "perform_validation": false
             },
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "3": {
             "id": 3,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "4": {
             "id": 4,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": {
                 "col_id": 1,
                 "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                 "_classname": "ColumnRef",
                 "tbl_version": null,
                 "reference_tbl": null,
                 "perform_validation": false
             },
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "5": {
             "id": 5,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "6": {
             "id": 6,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 3,
             "schema_version_drop": null
         },
         "7": {
             "id": 7,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": {
                 "col_id": 6,
                 "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                 "_classname": "ColumnRef",
                 "tbl_version": null,
                 "reference_tbl": null,
                 "perform_validation": false
             },
             "destination": null,
             "schema_version_add": 3,
             "schema_version_drop": null
         },
         "8": {
             "id": 8,
             "is_pk": false,
             "stored": true,
             "col_type": {
                 "nullable": true,
                 "_classname": "IntType"
             },
             "value_expr": null,
             "destination": null,
             "schema_version_add": 3,
             "schema_version_drop": null
         }
     },
     "tbl_state": 0,
     "is_replica": false,
     "next_col_id": 9,
     "next_idx_id": 3,
     "next_row_id": 10,
     "pending_stmt": null,
     "additional_md": {
     },
     "current_version": 4,
     "external_stores": [
     ],
     "has_pending_ops": false,
     "current_schema_version": 3
 }""")

_SCHEMA_VERSION_MD = {
    0: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "media_validation": null
                        },
                        "1": {
                            "pos": 1,
                            "name": "c2",
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 0,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": null
                }
                 """),
    2: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 2,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": 0
                }
                 """),
    3: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "media_validation": null
                        },
                        "6": {
                            "pos": 1,
                            "name": "c3",
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 3,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": 2
                }
                 """),
}

_TABLE_MD_EXPECTED = json.loads("""
 {
     "name": "test",
     "user": null,
     "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
     "view_md": null,
     "view_sn": 0,
     "index_md": {
         "0": {
             "id": 0,
             "name": "idx0",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 0,
             "index_val_col_id": 2,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 0,
             "schema_version_drop": null,
             "index_val_undo_col_id": 3
         },
         "1": {
             "id": 1,
             "name": "idx1",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 1,
             "index_val_col_id": 4,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 0,
             "schema_version_drop": 2,
             "index_val_undo_col_id": 5
         },
         "2": {
             "id": 2,
             "name": "idx2",
             "class_fqn": "pixeltable.index.btree.BtreeIndex",
             "init_args": {
             },
             "indexed_col_id": 6,
             "index_val_col_id": 7,
             "indexed_col_tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
             "schema_version_add": 3,
             "schema_version_drop": null,
             "index_val_undo_col_id": 8
         }
     },
     "column_md": {
         "0": {
             "id": 0,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "1": {
             "id": 1,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "2": {
             "id": 2,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "3": {
             "id": 3,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": null
         },
         "4": {
             "id": 4,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "5": {
             "id": 5,
             "stored": true,
             "schema_version_add": 0,
             "schema_version_drop": 2
         },
         "6": {
             "id": 6,
             "stored": true,
             "schema_version_add": 3,
             "schema_version_drop": null
         },
         "7": {
             "id": 7,
             "stored": true,
             "schema_version_add": 3,
             "schema_version_drop": null
         },
         "8": {
             "id": 8,
             "stored": true,
             "schema_version_add": 3,
             "schema_version_drop": null
         }
     },
     "tbl_state": 0,
     "is_replica": false,
     "next_col_id": 9,
     "next_idx_id": 3,
     "next_row_id": 10,
     "pending_stmt": null,
     "additional_md": {
     },
     "current_version": 4,
     "external_stores": [
     ],
     "has_pending_ops": false,
     "current_schema_version": 3
 }
 """)

_SCHEMA_VERSION_MD_EXPECTED = {
    0: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "1": {
                            "pos": 1,
                            "name": "c2",
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "2": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": {
                                "col_id": 0,
                                "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                                "_classname": "ColumnRef",
                                "tbl_version": null,
                                "reference_tbl": null,
                                "perform_validation": false
                            },
                            "destination": null,
                            "media_validation": null
                        },
                        "3": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "4": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": {
                                "col_id": 1,
                                "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                                "_classname": "ColumnRef",
                                "tbl_version": null,
                                "reference_tbl": null,
                                "perform_validation": false
                            },
                            "destination": null,
                            "media_validation": null
                        },
                        "5": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 0,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": null
                }
    """),
    2: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "2": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": {
                                "col_id": 0,
                                "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                                "_classname": "ColumnRef",
                                "tbl_version": null,
                                "reference_tbl": null,
                                "perform_validation": false
                            },
                            "destination": null,
                            "media_validation": null
                        },
                        "3": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 2,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": 0
                }
    """),
    3: json.loads("""
                {
                    "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                    "columns": {
                        "0": {
                            "pos": 0,
                            "name": "c1",
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "2": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": {
                                "col_id": 0,
                                "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                                "_classname": "ColumnRef",
                                "tbl_version": null,
                                "reference_tbl": null,
                                "perform_validation": false
                            },
                            "destination": null,
                            "media_validation": null
                        },
                        "3": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "6": {
                            "pos": 1,
                            "name": "c3",
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        },
                        "7": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": {
                                "col_id": 6,
                                "tbl_id": "c463b145-7085-4d03-a197-1f7835c78d6e",
                                "_classname": "ColumnRef",
                                "tbl_version": null,
                                "reference_tbl": null,
                                "perform_validation": false
                            },
                            "destination": null,
                            "media_validation": null
                        },
                        "8": {
                            "pos": null,
                            "name": null,
                            "is_pk": false,
                            "col_type": {
                                "nullable": true,
                                "_classname": "IntType"
                            },
                            "value_expr": null,
                            "destination": null,
                            "media_validation": null
                        }
                    },
                    "comment": "",
                    "additional_md": {
                    },
                    "schema_version": 3,
                    "media_validation": "on_write",
                    "num_retained_versions": 10,
                    "preceding_schema_version": 2
                }
    """),
}
