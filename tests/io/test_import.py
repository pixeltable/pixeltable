import json
from pathlib import Path
from typing import Callable

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts

from ..utils import ensure_s3_pytest_resources_access, get_image_files, pxt_raises, rerun

EXPECTED_SCHEMA = {
    'name': ts.StringType(nullable=True),
    'human': ts.BoolType(nullable=True),
    'parents': ts.JsonType(nullable=True),
    'age': ts.FloatType(nullable=True),
    'metadata': ts.JsonType(nullable=True),
    'children': ts.IntType(nullable=True),
}


EXPECTED_SCHEMA_WITH_JSON_INFERENCE = {
    'name': ts.StringType(nullable=True),
    'human': ts.BoolType(nullable=True),
    'parents': ts.JsonType(ts.JsonType.TypeSchema([ts.StringType(), ts.StringType()]), nullable=True),
    'age': ts.FloatType(nullable=True),
    'metadata': ts.JsonType(ts.JsonType.TypeSchema({'first_appearance': ts.StringType()}), nullable=True),
    'children': ts.IntType(nullable=True),
}


class TestImport:
    def test_import_rows(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        with open(example, encoding='utf-8') as fp:
            data = json.loads(fp.read())
        t1 = pxt.io.import_rows(p('example1'), data)
        assert t1.count() == 4
        assert t1._get_schema() == EXPECTED_SCHEMA

        t2 = pxt.io.import_rows(p('example2'), data, schema_overrides={'children': pxt.Float})
        assert t2.count() == 4
        assert t2._get_schema() == EXPECTED_SCHEMA | {'children': ts.FloatType(nullable=True)}

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            pxt.io.import_rows(p('example3'), [{'only_none': None}])
        assert 'The following columns have no non-null values: only_none' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            pxt.io.import_rows(p('example4'), [{'col': 1}], schema_overrides={'not_col': pxt.String})
        assert 'Some column(s) specified in `schema_overrides` are not present in the source: not_col' in str(
            exc_info.value
        )

        with pxt_raises(pxt.ErrorCode.INVALID_TYPE) as exc_info:
            pxt.io.import_rows(p('example5'), [{'col': 1}, {'col': 'value'}])
        assert (
            'Could not infer type of column `col`; '
            "the value in row 1 does not match preceding type Int | None: 'value'" in str(exc_info.value)
        )

        with pxt_raises(pxt.ErrorCode.INVALID_TYPE) as exc_info:
            pxt.io.import_rows(p('example6'), [{'col': str}])
        assert (
            "Could not infer type for column `col`; the value in row 0 has an unsupported type: <class 'type'>"
            in str(exc_info.value)
        )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            pxt.io.import_rows(p('example7'), [{'__unusable_name': 'abc'}])
        assert 'Column names must be valid pixeltable identifiers' in str(exc_info.value)

    def test_insert_rows(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        with open(example, encoding='utf-8') as fp:
            data = json.loads(fp.read())
        t1 = pxt.io.import_rows(p('example1'), data)
        assert t1.count() == 4
        t1.insert(data)
        assert t1.count() == 8

        t2 = pxt.io.import_rows(p('example2'), data, schema_overrides={'children': pxt.Float})
        assert t2.count() == 4
        t2.insert(data)
        assert t2.count() == 8

    @rerun(reruns=3, reruns_delay=15, only_rerun=['429', 'Too Many Requests'])
    def test_import_json(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        jeopardy = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/json/jeopardy.json'

        # `example.json` has a variety of datatypes and tests both nullable and non-nullable columns
        t1 = pxt.io.import_json(p('example'), str(example))
        assert t1.count() == 4
        assert t1._get_schema() == EXPECTED_SCHEMA

        # `jeopardy.json` is a larger dataset; we try loading it as a URL to test both file and URL loading
        t2 = pxt.io.import_json(p('jeopardy'), jeopardy)
        assert t2.count() == 10000

    @pytest.mark.parametrize(
        'source',
        [
            'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/json/example.json',
            's3://pxt-test/pytest-resources/example.json',
        ],
    )
    @rerun(reruns=3, reruns_delay=15, only_rerun=['429', 'Too Many Requests'])
    def test_import_json_from_remote(self, make_catalog_path: Callable[[str], str], source: str) -> None:
        p = make_catalog_path
        if source.startswith('s3://'):
            ensure_s3_pytest_resources_access()
        tab = pxt.create_table(p('from_remote_json'), source=source, source_format='json')
        assert tab.count() == 4
        assert tab._get_schema() == EXPECTED_SCHEMA

    @rerun(reruns=3, reruns_delay=15, only_rerun=['429', 'Too Many Requests'])
    def test_insert_json(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        jeopardy = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/json/jeopardy.json'

        # `example.json` has a variety of datatypes and tests both nullable and non-nullable columns
        t1 = pxt.io.import_json(p('example'), str(example))
        assert t1.count() == 4
        t1.insert(str(example))
        assert t1.count() == 8

        # `jeopardy.json` is a larger dataset; we try loading it as a URL to test both file and URL loading
        t2 = pxt.io.import_json(p('jeopardy'), jeopardy)
        assert t2.count() == 10000
        t2.insert(jeopardy)
        assert t2.count() == 20000

    def test_insert_file_reader_options(self, make_catalog_path: Callable[[str], str], tmp_path: Path) -> None:
        """A file source's reader options (kwargs) and schema_overrides are honored, including for a table
        with a media column."""
        p = make_catalog_path

        # a ';'-delimited CSV: the reader option is needed to split the columns correctly
        csv_path = tmp_path / 'data.csv'
        csv_path.write_text('a;b\n1;x\n2;y\n')

        t = pxt.create_table(p('rdopt'), {'a': pxt.Int, 'b': pxt.String})
        t.insert(str(csv_path), delimiter=';')
        rows = list(t.order_by(t.a).collect())
        assert [r['a'] for r in rows] == [1, 2]
        assert [r['b'] for r in rows] == ['x', 'y']

        # schema_overrides is honored on a file source (column 'a' kept as a string)
        t2 = pxt.create_table(p('rdopt_ovr'), {'a': pxt.String, 'b': pxt.String})
        t2.insert(str(csv_path), delimiter=';', schema_overrides={'a': ts.StringType(nullable=True)})
        assert [r['a'] for r in t2.order_by(t2.a).collect()] == ['1', '2']

        # a table with a media column, fed a ';'-delimited CSV whose image column holds local file paths
        img_paths = get_image_files()[:2]
        media_csv = tmp_path / 'media.csv'
        media_csv.write_text('k;img\n' + ''.join(f'{i};{path}\n' for i, path in enumerate(img_paths)))
        tm = pxt.create_table(p('rdopt_media'), {'k': pxt.Int, 'img': pxt.Image})
        tm.insert(str(media_csv), delimiter=';')
        media_rows = list(tm.order_by(tm.k).collect())
        assert [r['k'] for r in media_rows] == [0, 1]
        assert all(isinstance(r['img'], PIL.Image.Image) for r in media_rows)
