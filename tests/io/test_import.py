import json
from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs


class TestImport:
    def test_import_rows(self, reset_db) -> None:
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        with open(example) as fp:
            data = json.loads(fp.read())
        t1 = pxt.io.import_rows('example1', data)
        assert t1.count() == 4
        assert t1._schema == {
            'name': pxt.StringType(nullable=True),
            'human': pxt.BoolType(nullable=True),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.FloatType(nullable=True),
            'metadata': pxt.JsonType(nullable=True),
            'children': pxt.IntType(nullable=True),
        }

        t2 = pxt.io.import_rows('example2', data, schema_overrides={'children': pxt.FloatType(nullable=True)})
        assert t2.count() == 4
        assert t2._schema == {
            'name': pxt.StringType(nullable=True),
            'human': pxt.BoolType(nullable=True),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.FloatType(nullable=True),
            'metadata': pxt.JsonType(nullable=True),
            'children': pxt.FloatType(nullable=True),
        }

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_rows('example3', [{'only_none': None}])
        assert 'The following columns have no non-null values: only_none' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_rows('example4', [{'col': 1}], schema_overrides={'not_col': pxt.StringType()})
        assert 'Some column(s) specified in `schema_overrides` are not present in the source: not_col' in str(
            exc_info.value
        )

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_rows('example5', [{'col': 1}, {'col': 'value'}])
        assert (
            "Could not infer type of column `col`; the value in row 1 does not match preceding type Optional[Int]: 'value'"
            in str(exc_info.value)
        )

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_rows('example6', [{'col': str}])
        assert (
            "Could not infer type for column `col`; the value in row 0 has an unsupported type: <class 'type'>"
            in str(exc_info.value)
        )

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_rows('example7', [{'__unusable_name': 'abc'}])
        assert 'Column names must be valid pixeltable identifiers' in str(exc_info.value)

    def test_import_json(self, reset_db) -> None:
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        jeopardy = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/json/jeopardy.json'

        # `example.json` has a variety of datatypes and tests both nullable and non-nullable columns
        t1 = pxt.io.import_json('example', str(example))
        assert t1.count() == 4
        assert t1._schema == {
            'name': pxt.StringType(nullable=True),
            'human': pxt.BoolType(nullable=True),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.FloatType(nullable=True),
            'metadata': pxt.JsonType(nullable=True),
            'children': pxt.IntType(nullable=True),
        }
        # `jeopardy.json` is a larger dataset; we try loading it as a URL to test both file and URL loading
        t2 = pxt.io.import_json('jeopardy', jeopardy)
        assert t2.count() == 10000
