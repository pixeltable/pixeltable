import datetime
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import pixeltable_pgserver
import toml

import pixeltable as pxt
import pixeltable.metadata as metadata
from pixeltable.env import Env
from pixeltable.func import Batch
from pixeltable.io.external_store import Project
from pixeltable.type_system import BoolType, FloatType, ImageType, IntType, JsonType, StringType, TimestampType

_logger = logging.getLogger('pixeltable')


class Dumper:
    def __init__(self, output_dir: str = 'target', db_name: str = 'pxtdump') -> None:
        if sys.version_info >= (3, 10):
            raise RuntimeError(
                'This script must be run on Python 3.9. '
                'DB dumps are incompatible across versions due to issues with pickling anonymous UDFs.'
            )

        self.output_dir = pathlib.Path(output_dir)
        shared_home = pathlib.Path(os.environ.get('PIXELTABLE_HOME', '~/.pixeltable')).expanduser()
        mock_home_dir = self.output_dir / '.pixeltable'
        mock_home_dir.mkdir(parents=True, exist_ok=True)
        os.environ['PIXELTABLE_HOME'] = str(mock_home_dir)
        os.environ['PIXELTABLE_CONFIG'] = str(shared_home / 'config.toml')
        os.environ['PIXELTABLE_DB'] = db_name
        os.environ['PIXELTABLE_PGDATA'] = str(shared_home / 'pgdata')

        Env._init_env(reinit_db=True)

        Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)

    def dump_db(self) -> None:
        md_version = metadata.VERSION
        dump_file = self.output_dir / f'pixeltable-v{md_version:03d}-test.dump.gz'
        _logger.info(f'Creating database dump at: {dump_file}')
        pg_package_dir = os.path.dirname(pixeltable_pgserver.__file__)
        pg_dump_binary = f'{pg_package_dir}/pginstall/bin/pg_dump'
        _logger.info(f'Using pg_dump binary at: {pg_dump_binary}')
        # We need the raw DB URL, without a driver qualifier.  (The driver qualifier is needed by
        # SQLAlchemy, but command-line Postgres won't know how to interpret it.)
        db_url = Env.get()._db_server.get_uri(Env.get()._db_name)
        with open(dump_file, 'wb') as dump:
            pg_dump_process = subprocess.Popen(
                (pg_dump_binary, db_url, '-U', 'postgres', '-Fc'), stdout=subprocess.PIPE
            )
            subprocess.run(('gzip', '-9'), stdin=pg_dump_process.stdout, stdout=dump, check=True)
            if pg_dump_process.poll() != 0:
                # wait for a 2s before checking again & raising error
                time.sleep(2)
            if pg_dump_process.poll() != 0:
                raise RuntimeError(f'pg_dump failed with return code {pg_dump_process.returncode}')
        info_file = self.output_dir / f'pixeltable-v{md_version:03d}-test-info.toml'
        git_sha = subprocess.check_output(('git', 'rev-parse', 'HEAD')).decode('ascii').strip()
        user = os.environ.get('USER', os.environ.get('USERNAME'))
        info_dict = {
            'pixeltable-dump': {
                'metadata-version': md_version,
                'git-sha': git_sha,
                'datetime': datetime.datetime.now(tz=datetime.timezone.utc),
                'user': user,
            }
        }
        with open(info_file, 'w') as info:
            toml.dump(info_dict, info)

    # Expression types, predicate types, embedding indices, views on views
    def create_tables(self) -> None:
        schema = {
            'c1': StringType(nullable=False),
            'c1n': StringType(nullable=True),
            'c2': IntType(nullable=False),
            'c3': FloatType(nullable=False),
            'c4': BoolType(nullable=False),
            'c5': TimestampType(nullable=False),
            'c6': JsonType(nullable=False),
            'c7': JsonType(nullable=False),
            'c8': ImageType(nullable=True),
        }
        t = pxt.create_table('base_table', schema, primary_key='c2')

        num_rows = 20
        d1 = {
            'f1': 'test string 1',
            'f2': 1,
            'f3': 1.0,
            'f4': True,
            'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
        }
        d2 = [d1, d1]

        c1_data = [f'test string {i}' for i in range(num_rows)]
        c2_data = [i for i in range(num_rows)]
        c3_data = [float(i) for i in range(num_rows)]
        c4_data = [bool(i % 2) for i in range(num_rows)]
        c5_data = [datetime.datetime.now()] * num_rows
        c6_data = [
            {
                'f1': f'test string {i}',
                'f2': i,
                'f3': float(i),
                'f4': bool(i % 2),
                'f5': [1.0, 2.0, 3.0, 4.0],
                'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
            }
            for i in range(num_rows)
        ]
        c7_data = [d2] * num_rows
        rows = [
            {
                'c1': c1_data[i],
                'c1n': c1_data[i] if i % 10 != 0 else None,
                'c2': c2_data[i],
                'c3': c3_data[i],
                'c4': c4_data[i],
                'c5': c5_data[i],
                'c6': c6_data[i],
                'c7': c7_data[i],
                'c8': None,
            }
            for i in range(num_rows)
        ]

        self.__add_expr_columns(t, 'base_table')
        t.insert(rows)

        pxt.create_dir('views')

        # simple view
        v = pxt.create_view('views.view', t.where(t.c2 < 50))
        self.__add_expr_columns(v, 'view')

        # snapshot
        _ = pxt.create_snapshot('views.snapshot', t.where(t.c2 >= 75))

        # view of views
        vv = pxt.create_view('views.view_of_views', v.where(t.c2 >= 25))
        self.__add_expr_columns(vv, 'view_of_views')

        # empty view
        e = pxt.create_view('views.empty_view', t.where(t.c2 == 4171780))
        assert e.count() == 0
        self.__add_expr_columns(e, 'empty_view', include_expensive_functions=True)

        # Add external stores
        from pixeltable.io.external_store import MockProject

        v._link_external_store(
            MockProject.create(
                v,
                'project',
                {'int_field': pxt.IntType()},
                {'str_field': pxt.StringType()},
                {'view_test_udf': 'int_field', 'c1': 'str_field'},
            )
        )
        # We're just trying to test metadata here, so it's ok to link a false Label Studio project.
        # We include a computed image column in order to ensure the creation of a stored proxy.
        from pixeltable.io.label_studio import LabelStudioProject

        col_mapping = Project.validate_columns(
            v,
            {'str_field': pxt.StringType(), 'img_field': pxt.ImageType()},
            {},
            {'view_function_call': 'str_field', 'base_table_image_rot': 'img_field'},
        )
        project = LabelStudioProject('ls_project_0', 4171780, media_import_method='file', col_mapping=col_mapping)
        v._link_external_store(project)
        # Sanity check that the stored proxy column did get created
        assert len(project.stored_proxies) == 1
        assert t.base_table_image_rot.col in project.stored_proxies

    def __add_expr_columns(self, t: pxt.Table, col_prefix: str, include_expensive_functions: bool = False) -> None:
        def add_computed_column(col_name: str, col_expr: Any, stored: bool = True) -> None:
            t.add_computed_column(**{f'{col_prefix}_{col_name}': col_expr}, stored=stored)

        # arithmetic_expr
        add_computed_column('plus', t.c2 + 6)
        add_computed_column('minus', t.c2 - 5)
        add_computed_column('times', t.c3 * 1.2)
        add_computed_column('div', t.c3 / 1.7)
        add_computed_column('mod', t.c2 % 11)

        # column_property_ref
        add_computed_column('fileurl', t.c8.fileurl)
        add_computed_column('localpath', t.c8.localpath)

        # comparison
        add_computed_column('lt', t.c2 < t.c3)
        add_computed_column('le', t.c2 <= t.c3)
        add_computed_column('gt', t.c2 > t.c3)
        add_computed_column('ge', t.c2 >= t.c3)
        add_computed_column('ne', t.c2 != t.c3)
        add_computed_column('eq', t.c2 == t.c3)

        # compound_predicate
        add_computed_column('and', (t.c2 >= 5) & (t.c2 < 8))
        add_computed_column('or', (t.c2 > 1) | t.c4)
        add_computed_column('not', ~(t.c2 > 20))

        # function_call
        add_computed_column(
            'function_call', pxt.functions.string.format('{0} {key}', t.c1, key=t.c1)
        )  # library function
        add_computed_column('test_udf', test_udf_stored(t.c2))  # stored udf
        add_computed_column('test_udf_batched', test_udf_stored_batched(t.c1, upper=False))  # batched stored udf
        if include_expensive_functions:
            # batched library function
            add_computed_column(
                'batched', pxt.functions.huggingface.clip(t.c1, model_id='openai/clip-vit-base-patch32')
            )

        # image_member_access
        add_computed_column('image_mode', t.c8.mode)
        add_computed_column('image_rot', t.c8.rotate(180), stored=False)

        # in_predicate
        add_computed_column('isin_1', t.c1.isin(['test string 1', 'test string 2', 'test string 3']))
        add_computed_column('isin_2', t.c2.isin([1, 2, 3, 4, 5]))
        add_computed_column('isin_3', t.c2.isin(t.c6.f5))

        # inline_array, inline_list, inline_dict
        add_computed_column('inline_array_1', pxt.array([[1, 2, 3], [4, 5, 6]]))
        add_computed_column('inline_array_2', pxt.array([['a', 'b', 'c'], ['d', 'e', 'f']]))
        add_computed_column('inline_array_exprs', pxt.array([[t.c2, t.c2 + 1], [t.c2 + 2, t.c2]]))
        add_computed_column('inline_array_mixed', pxt.array([[1, t.c2], [3, t.c2]]))
        add_computed_column('inline_list_1', [[1, 2, 3], [4, 5, 6]])
        add_computed_column('inline_list_2', [['a', 'b', 'c'], ['d', 'e', 'f']])
        add_computed_column('inline_list_exprs', [t.c1, [t.c1n, t.c2]])
        add_computed_column('inline_list_mixed', [1, 'a', t.c1, [1, 'a', t.c1n], 1, 'a'])
        add_computed_column('inline_dict', {'int': 22, 'dict': {'key': 'val'}, 'expr': t.c1})

        # is_null
        add_computed_column('isnull', t.c1 == None)

        # json_mapper and json_path
        add_computed_column('json_mapper', t.c6[3])
        add_computed_column('json_path', t.c6.f1)
        add_computed_column('json_path_nested', t.c6.f6.f7)
        add_computed_column('json_path_star', t.c6.f5['*'])
        add_computed_column('json_path_idx', t.c6.f5[3])
        add_computed_column('json_path_slice', t.c6.f5[1:3:2])

        # literal
        add_computed_column('str_const', 'str')
        add_computed_column('int_const', 5)
        add_computed_column('float_const', 5.0)
        add_computed_column('timestamp_const_1', datetime.datetime.now())
        add_computed_column('timestamp_const_2', datetime.datetime.now().astimezone(ZoneInfo('America/Anchorage')))

        # type_cast
        add_computed_column('astype', t.c2.astype(FloatType()))

        # .apply
        add_computed_column('c2_to_string', t.c2.apply(str))
        add_computed_column('c6_to_string', t.c6.apply(json.dumps))
        add_computed_column('c6_back_to_json', t[f'{col_prefix}_c6_to_string'].apply(json.loads))

        t.add_embedding_index(
            f'{col_prefix}_function_call',
            string_embed=pxt.functions.huggingface.clip.using(model_id='openai/clip-vit-base-patch32'),
        )

        if t.get_metadata()['is_view']:
            # Add an embedding index to the view that is on a column in the base table
            t.add_embedding_index(
                'base_table_function_call',
                string_embed=pxt.functions.huggingface.clip.using(model_id='openai/clip-vit-base-patch32'),
            )

        # query()
        @pxt.query
        def q1(i: int) -> pxt.DataFrame:
            # this breaks; TODO: why?
            # return t.where(t.c2 < i)
            return t.where(t.c2 < i).select(t.c1, t.c2)

        add_computed_column('query_output', q1(t.c2))

        @pxt.query
        def q2(s: str) -> pxt.DataFrame:
            sim = t[f'{col_prefix}_function_call'].similarity(s)
            return t.order_by(sim, asc=False).select(t[f'{col_prefix}_function_call']).limit(5)

        add_computed_column('sim_output', q2(t.c1))


@pxt.udf(_force_stored=True)
def test_udf_stored(n: int) -> int:
    return n + 1


@pxt.udf(batch_size=4, _force_stored=True)
def test_udf_stored_batched(strings: Batch[str], *, upper: bool = True) -> Batch[str]:
    return [string.upper() if upper else string.lower() for string in strings]


def main() -> None:
    _logger.info('Creating pixeltable test artifact.')
    dumper = Dumper()
    dumper.create_tables()
    dumper.dump_db()


if __name__ == '__main__':
    main()
