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
from pixeltable.func import Batch
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

        Env._init_env(reinit_db=True)

        Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)

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
        t = pxt.create_table('sample_table', schema, primary_key='c2')

        # Add columns for InlineArray and InlineDict
        t.add_column(c8=[[1, 2, 3], [4, 5, 6]])
        t.add_column(c9=[['a', 'b', 'c'], ['d', 'e', 'f']])
        t.add_column(c10=[t.c1, [t.c1n, t.c2]])
        t.add_column(c11={'int': 22, 'dict': {'key': 'val'}, 'expr': t.c1})

        # InPredicate
        t.add_column(isin_1=t.c1.isin(['test string 1', 'test string 2', 'test string 3']))
        t.add_column(isin_2=t.c2.isin([1, 2, 3, 4, 5]))
        t.add_column(isin_3=t.c2.isin(t.c6.f5))

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
        pxt.create_dir('views')
        v = pxt.create_view('views.sample_view', t, filter=(t.c2 < 50))
        _ = pxt.create_view('views.sample_snapshot', t, filter=(t.c2 >= 75), is_snapshot=True)
        e = pxt.create_view('views.empty_view', t, filter=t.c2 == 4171780)
        assert e.count() == 0
        # Computed column using a library function
        v['str_format'] = pxt.functions.string.str_format('{0} {key}', t.c1, key=t.c1)
        # Computed column using a bespoke stored udf
        v['test_udf'] = test_udf_stored(t.c2)
        # Computed column using a batched function
        # (apply this to the empty view, since it's a "heavyweight" function)
        e['batched'] = pxt.functions.huggingface.clip_text(t.c1, model_id='openai/clip-vit-base-patch32')
        # computed column using a stored batched function
        v['test_udf_batched'] = test_udf_stored_batched(t.c1, upper=False)
        # astype
        v['astype'] = t.c1.astype(pxt.FloatType())

        # Add remotes
        from pixeltable.datatransfer.remote import MockRemote
        v.link(
            MockRemote('remote', {'int_field': pxt.IntType()}, {'str_field': pxt.StringType()}),
            col_mapping={'test_udf': 'int_field', 'c1': 'str_field'}
        )
        # We're just trying to test metadata here, so reach "under the covers" and link a fake
        # Label Studio project without validation (so we don't need a real Label Studio server)
        from pixeltable.datatransfer.label_studio import LabelStudioProject
        v.tbl_version_path.tbl_version.link(
            LabelStudioProject(4171780, media_import_method='file'),
            col_mapping={'str_format': 'str_format'}
        )


@pxt.udf(_force_stored=True)
def test_udf_stored(n: int) -> int:
    return n + 1


@pxt.udf(batch_size=4, _force_stored=True)
def test_udf_stored_batched(strings: Batch[str], *, upper: bool = True) -> Batch[str]:
    return [string.upper() if upper else string.lower() for string in strings]


def main() -> None:
    _logger.info("Creating pixeltable test artifact.")
    dumper = Dumper()
    dumper.create_tables()
    dumper.dump_db()


if __name__ == "__main__":
    main()
