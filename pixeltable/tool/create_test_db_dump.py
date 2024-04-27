import datetime
import json
import logging
import os
import pathlib
import subprocess

import pgserver
import toml

import pixeltable as pxt
import pixeltable.metadata as metadata
from pixeltable.env import Env
from pixeltable.type_system import \
    StringType, IntType, FloatType, BoolType, TimestampType, JsonType

_logger = logging.getLogger('pixeltable')


class Dumper:

    def __init__(self, output_dir='target', db_name='pxtdump') -> None:
        self.output_dir = pathlib.Path(output_dir)
        shared_home = pathlib.Path(os.environ.get('PIXELTABLE_HOME', '~/.pixeltable')).expanduser()
        mock_home_dir = self.output_dir / '.pixeltable'
        mock_home_dir.mkdir(parents=True, exist_ok=True)
        os.environ['PIXELTABLE_HOME'] = str(mock_home_dir)
        os.environ['PIXELTABLE_CONFIG'] = str(shared_home / 'config.yaml')
        os.environ['PIXELTABLE_DB'] = db_name
        os.environ['PIXELTABLE_PGDATA'] = str(shared_home / 'pgdata')

        Env.get().set_up(reinit_db=True)
        self.cl = pxt.Client()
        self.cl.logging(level=logging.DEBUG, to_stdout=True)

    def dump_db(self) -> None:
        md_version = metadata.VERSION
        dump_file = self.output_dir / f'pixeltable-v{md_version:03d}-test.dump.gz'
        _logger.info(f'Creating database dump at: {dump_file}')
        pg_package_dir = os.path.dirname(pgserver.__file__)
        pg_dump_binary = f'{pg_package_dir}/pginstall/bin/pg_dump'
        _logger.info(f'Using pg_dump binary at: {pg_dump_binary}')
        with open(dump_file, 'wb') as dump:
            pg_dump_process = subprocess.Popen(
                [pg_dump_binary, Env.get().db_url, '-U', 'postgres', '-Fc'],
                stdout=subprocess.PIPE
            )
            subprocess.run(
                ["gzip", "-9"],
                stdin=pg_dump_process.stdout,
                stdout=dump,
                check=True
            )
        info_file = self.output_dir / f'pixeltable-v{md_version:03d}-test-info.toml'
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        user = os.environ.get('USER', os.environ.get('USERNAME'))
        info_dict = {'pixeltable-dump': {
            'metadata-version': md_version,
            'git-sha': git_sha,
            'datetime': datetime.datetime.utcnow(),
            'user': user
        }}
        with open(info_file, 'w') as info:
            toml.dump(info_dict, info)

    # TODO: Add additional features to the test DB dump (ideally it should exercise
    # every major pixeltable DB feature)
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
        }
        t = self.cl.create_table('sample_table', schema, primary_key='c2')
        t.add_column(c8=[[1, 2, 3], [4, 5, 6]])

        # Add columns for .astype converters to ensure they're persisted properly
        t.add_column(c2_as_float=t.c2.astype(FloatType()))

        # Add columns for .apply
        t.add_column(c2_to_string=t.c2.apply(str))
        t.add_column(c6_to_string=t.c6.apply(json.dumps))
        t.add_column(c6_back_to_json=t.c6_to_string.apply(json.loads))

        num_rows = 100
        d1 = {
            'f1': 'test string 1',
            'f2': 1,
            'f3': 1.0,
            'f4': True,
            'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {
                'f7': 'test string 2',
                'f8': [1.0, 2.0, 3.0, 4.0],
            },
        }
        d2 = [d1, d1]

        c1_data = [f'test string {i}' for i in range(num_rows)]
        c2_data = [i for i in range(num_rows)]
        c3_data = [float(i) for i in range(num_rows)]
        c4_data = [bool(i % 2) for i in range(num_rows)]
        c5_data = [datetime.datetime.now()] * num_rows
        c6_data = []
        for i in range(num_rows):
            d = {
                'f1': f'test string {i}',
                'f2': i,
                'f3': float(i),
                'f4': bool(i % 2),
                'f5': [1.0, 2.0, 3.0, 4.0],
                'f6': {
                    'f7': 'test string 2',
                    'f8': [1.0, 2.0, 3.0, 4.0],
                },
            }
            c6_data.append(d)

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
            }
            for i in range(num_rows)
        ]
        t.insert(rows)
        self.cl.create_dir('views')
        v = self.cl.create_view('views.sample_view', t, filter=(t.c2 < 50))
        _ = self.cl.create_view('views.sample_snapshot', t, filter=(t.c2 >= 75), is_snapshot=True)
        # Computed column using a library function
        v['str_format'] = pxt.functions.string.str_format('{0} {key}', t.c1, key=t.c1)
        # Computed column using a bespoke udf
        v['test_udf'] = test_udf(t.c2)
        # astype
        v['astype'] = t.c1.astype(pxt.FloatType())
        # computed column using a stored function
        v['stored'] = t.c1.apply(lambda x: f'Hello, {x}', col_type=pxt.StringType())


@pxt.udf
def test_udf(n: int) -> int:
    return n + 1


def main() -> None:
    _logger.info("Creating pixeltable test artifact.")
    dumper = Dumper()
    dumper.create_tables()
    dumper.dump_db()


if __name__ == "__main__":
    main()
