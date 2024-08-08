import json
from pathlib import Path

import pixeltable as pxt


class TestImport:

    def test_import_data(self, reset_db) -> None:
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        with open(example) as fp:
            data = json.loads(fp.read())
        t1 = pxt.io.import_data('example1', data)
        assert t1.count() == 4
        assert t1.column_types() == {
            'name': pxt.StringType(),
            'human': pxt.BoolType(),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.IntType(nullable=True)
        }
        t2 = pxt.io.import_data('example2', data, schema_overrides={'age': pxt.FloatType(nullable=True)})
        assert t2.count() == 4
        assert t2.column_types() == {
            'name': pxt.StringType(),
            'human': pxt.BoolType(),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.FloatType(nullable=True)
        }

    def test_import_json(self, reset_db) -> None:
        example = Path(__file__).parent.parent / 'data' / 'json' / 'example.json'
        jeopardy = 'https://raw.githubusercontent.com/pixeltable/pixeltable/master/tests/data/json/jeopardy.json'

        # `example.json` has a variety of datatypes and tests both nullable and non-nullable columns
        t1 = pxt.io.import_json('example', str(example))
        assert t1.count() == 4
        assert t1.column_types() == {
            'name': pxt.StringType(),
            'human': pxt.BoolType(),
            'parents': pxt.JsonType(nullable=True),
            'age': pxt.IntType(nullable=True)
        }
        # `jeopardy.json` is a larger dataset; we try loading it as a URL to test both file and URL loading
        t2 = pxt.io.import_json('jeopardy', jeopardy)
        assert t2.count() == 10000
