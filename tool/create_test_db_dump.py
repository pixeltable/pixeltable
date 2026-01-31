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
from pixeltable import functions as pxtf, metadata, type_system as ts
from pixeltable.env import Env
from pixeltable.func import Batch
from pixeltable.io.external_store import Project
from pixeltable.iterators.base import ComponentIterator

_logger = logging.getLogger('pixeltable')


# We use URLs (not local files) so that the dumps are portable. We also use repo references with fixed SHAs for all
# URLs, so that the dumps will survive any future repo reorganizations.

SAMPLE_IMAGE_URLS = (
    'https://raw.githubusercontent.com/pixeltable/pixeltable/870cf9c49a368e2c17bf53e6fde48554e546abd7/'
    'docs/resources/images/000000000016.jpg',
    'https://raw.githubusercontent.com/pixeltable/pixeltable/870cf9c49a368e2c17bf53e6fde48554e546abd7/'
    'docs/resources/images/000000000019.jpg',
)
SAMPLE_VIDEO_URLS = (
    'https://raw.githubusercontent.com/pixeltable/pixeltable/d8b91c59d6f1742ba75c20f318c0f9a2ae729768/'
    'tests/data/videos/bangkok_half_res.mp4',
    'https://raw.githubusercontent.com/pixeltable/pixeltable/1418d69125cd19ff09f8f368b65c248d8dcd7377/'
    'tests/data/videos/v_shooting_01_01.mpg',
)
SAMPLE_AUDIO_URLS = (
    'https://raw.githubusercontent.com/pixeltable/pixeltable/cab695b0df06286cd88857036adcb5efc3fd122a/'
    'tests/data/audio/jfk_1961_0109_cityuponahill-excerpt.flac',
    'https://raw.githubusercontent.com/pixeltable/pixeltable/d8b91c59d6f1742ba75c20f318c0f9a2ae729768/'
    'tests/data/audio/sample.mp3',
)
SAMPLE_DOCUMENT_URLS = (
    'https://raw.githubusercontent.com/pixeltable/pixeltable/d8b91c59d6f1742ba75c20f318c0f9a2ae729768/'
    'tests/data/documents/layout-parser-paper.pdf',
    'https://raw.githubusercontent.com/pixeltable/pixeltable/d8b91c59d6f1742ba75c20f318c0f9a2ae729768/'
    'tests/data/documents/1706.03762.pdf',
)


class CustomLegacyIterator(ComponentIterator):
    """This is preserved in code for the benefit of version <= 45 database dumps that reference it."""

    input_text: str
    expand_by: int
    idx: int

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, ts.ColumnType]:
        return {'text': ts.StringType(), 'expand_by': ts.IntType()}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {'output_text': ts.StringType(), 'unstored_text': ts.StringType()}, ['unstored_text']

    def __init__(self, text: str, expand_by: int) -> None:
        self.input_text = text
        self.expand_by = expand_by
        self.idx = 0

    def __next__(self) -> dict[str, Any]:
        if self.idx >= self.expand_by:
            raise StopIteration
        result = {
            'output_text': f'stored {self.input_text} {self.idx}',
            'unstored_text': f'unstored {self.input_text} {self.idx}',
        }
        self.idx += 1
        return result

    def close(self) -> None:
        pass

    def set_pos(self, pos: int, **kwargs: Any) -> None:
        assert 0 <= pos < self.expand_by
        self.idx = pos


class Dumper:
    def __init__(self, output_dir: str = 'target', db_name: str = 'pxtdump') -> None:
        if sys.version_info >= (3, 11):
            raise RuntimeError(
                'This script must be run on Python 3.10. '
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
        with open(info_file, 'w', encoding='utf-8') as info:
            toml.dump(info_dict, info)

    # Expression types, predicate types, embedding indices, views on views
    def create_tables(self) -> None:
        import tool.create_test_db_dump  # noqa: PLW0406  # we need a self-reference since this module is run as main

        schema = {
            'c1': pxt.Required[pxt.String],
            'c1n': pxt.String,
            'c2': pxt.Required[pxt.Int],
            'c3': pxt.Required[pxt.Float],
            'c4': pxt.Required[pxt.Bool],
            'c5': pxt.Required[pxt.Timestamp],
            'c6': pxt.Required[pxt.Json],
            'c7': pxt.Required[pxt.Json],
            'c8': pxt.Image,
            'c9': pxt.Audio,
            'c10': pxt.Video,
            'c11': pxt.Document,
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
        c2_data = list(range(num_rows))
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
                'c8': SAMPLE_IMAGE_URLS[i] if i < len(SAMPLE_IMAGE_URLS) else None,
                'c9': SAMPLE_AUDIO_URLS[i] if i < len(SAMPLE_AUDIO_URLS) else None,
                'c10': SAMPLE_VIDEO_URLS[i] if i < len(SAMPLE_VIDEO_URLS) else None,
                'c11': SAMPLE_DOCUMENT_URLS[i] if i < len(SAMPLE_DOCUMENT_URLS) else None,
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
                {'int_field': ts.IntType()},
                {'str_field': ts.StringType()},
                {'view_test_udf': 'int_field', 'c1': 'str_field'},
            )
        )
        # We're just trying to test metadata here, so it's ok to link a false Label Studio project.
        # We include a computed image column in order to ensure the creation of a stored proxy.
        from pixeltable.io.label_studio import LabelStudioProject

        col_mapping = Project.validate_columns(
            v,
            {'str_field': ts.StringType(), 'img_field': ts.ImageType()},
            {},
            {'view_function_call': 'str_field', 'base_table_image_rot': 'img_field'},
        )
        project = LabelStudioProject('ls_project_0', 4171780, media_import_method='file', col_mapping=col_mapping)
        v._link_external_store(project)
        # Sanity check that the stored proxy column did get created
        assert len(project.stored_proxies) == 1
        assert t.base_table_image_rot.col.handle in project.stored_proxies

        # Various iterators
        pxt.create_view('string_splitter', t, iterator=pxtf.string.string_splitter(t.c1, 'sentence'))
        pxt.create_view('tile_iterator', t, iterator=pxtf.image.tile_iterator(t.c8, (64, 64), overlap=(16, 16)))
        pxt.create_view('frame_iterator_1', t, iterator=pxtf.video.frame_iterator(t.c10, fps=1, use_legacy_schema=True))
        pxt.create_view('frame_iterator_2', t, iterator=pxtf.video.frame_iterator(t.c10, num_frames=5))
        pxt.create_view('frame_iterator_3', t, iterator=pxtf.video.frame_iterator(t.c10, keyframes_only=True))
        pxt.create_view(
            'document_splitter', t, iterator=pxtf.document.document_splitter(t.c11, 'page', elements=['text'])
        )
        # audio_splitter and video_splitter produce local files as output, so we can't include any outputs here
        # (the dumps won't be portable). But we can create filter views with no output, and that at least tests
        # that the iterator metadata survives the migration. By choosing an appropriate filter we ensure that
        # the insert statement in test_migration.py *does* produce data, providing further validation.
        # TODO: "Bundling" local media files with the db dumps would increase test coverage.
        pxt.create_view(
            'audio_splitter',
            t.where(t.c2 >= len(SAMPLE_AUDIO_URLS)),
            iterator=pxtf.audio.audio_splitter(
                t.c9, segment_duration_sec=10.0, overlap_sec=1.0, min_segment_duration_sec=5.0
            ),
        )
        pxt.create_view(
            'video_splitter',
            t.where(t.c2 >= len(SAMPLE_VIDEO_URLS)),
            iterator=pxtf.video.video_splitter(
                t.c10, duration=10.0, overlap=1.0, min_segment_duration=5.0, mode='fast'
            ),
        )
        pxt.create_view(
            'video_splitter_2',
            t.where(t.c2 >= len(SAMPLE_VIDEO_URLS)),
            iterator=pxtf.video.video_splitter(t.c10, segment_times=[3.0, 6.0], mode='accurate'),
        )

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
        add_computed_column('function_call', pxtf.string.format('{0} {key}', t.c1, key=t.c1))  # library function
        add_computed_column('test_udf', test_udf_stored(t.c2))  # stored udf
        add_computed_column('test_udf_batched', test_udf_stored_batched(t.c1, upper=False))  # batched stored udf
        if include_expensive_functions:
            # batched library function
            add_computed_column('batched', pxtf.huggingface.clip(t.c1, model_id='openai/clip-vit-base-patch32'))

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
        add_computed_column('astype', t.c2.astype(ts.FloatType()))

        # .apply
        add_computed_column('c2_to_string', t.c2.apply(str))
        add_computed_column('c6_to_string', t.c6.apply(json.dumps))
        add_computed_column('c6_back_to_json', t[f'{col_prefix}_c6_to_string'].apply(json.loads))

        t.add_embedding_index(
            f'{col_prefix}_function_call',
            string_embed=pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32'),
        )

        if t.get_metadata()['is_view']:
            # Add an embedding index to the view that is on a column in the base table
            t.add_embedding_index(
                'base_table_function_call',
                string_embed=pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32'),
            )

        # query()
        @pxt.query
        def q1(i: int) -> pxt.Query:
            # this breaks; TODO: why?
            # return t.where(t.c2 < i)
            return t.where(t.c2 < i).select(t.c1, t.c2)

        add_computed_column('query_output', q1(t.c2))

        @pxt.query
        def q2(s: str) -> pxt.Query:
            sim = t[f'{col_prefix}_function_call'].similarity(string=s)
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
