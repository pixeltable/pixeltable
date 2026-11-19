"""Models from a TypeScript schema file, replayed through the Python DSL.

The tests feed in the data that @pixeltable/sdk's exportModels() prints; Node evaluates a schema file only when the
CLI reads one.
"""

from typing import Any

import pytest

import pixeltable as pxt
from pixeltable.catalog.model import typescript

from .utils import DatabaseRoot, pxt_raises

_FILMS_COLUMNS: list[list[Any]] = [
    ['title', {'type': {'type': 'string', 'nullable': False}}],
    ['year', {'type': {'type': 'int', 'nullable': False}}],
    ['budget', {'type': {'type': 'float', 'nullable': True}}],
    ['revenue', {'type': {'type': 'float', 'nullable': True}, 'comment': 'worldwide gross'}],
    ['title_upper', {'value': {'method': 'upper', 'on': {'col': 'title'}, 'args': [], 'kwargs': {}}}],
    ['title_len', {'value': {'method': 'len', 'on': {'col': 'title_upper'}, 'args': [], 'kwargs': {}}}],
    ['profit', {'value': {'op': 'sub', 'args': [{'col': 'revenue'}, {'col': 'budget'}]}}],
    ['margin', {'value': {'op': 'div', 'args': [{'col': 'profit'}, {'col': 'budget'}]}}],
    ['is_hit', {'value': {'op': 'gt', 'args': [{'col': 'margin'}, 1]}}],
    ['decade', {'value': {'op': 'mul', 'args': [{'op': 'floorDiv', 'args': [{'col': 'year'}, 10]}, 10]}}],
    [
        'label',
        {
            'value': {
                'call': 'pixeltable.functions.string.format',
                'args': ['{0} ({1})', {'col': 'title'}, {'col': 'year'}],
                'kwargs': {},
            }
        },
    ],
]


def _export(columns: list[list[Any]]) -> dict[str, Any]:
    return {'v': 1, 'models': [{'cls': 'Films', 'name': 'films', 'options': {}, 'columns': columns, 'indexes': []}]}


class TestTypescriptModels:
    def test_computed_columns(self, db_root: DatabaseRoot) -> None:
        root = db_root.make_catalog_path('')
        diffs = typescript.models_from_export(_export(_FILMS_COLUMNS), module='app').update_all(root)
        assert diffs['films'].resolution == 'create'

        t = pxt.get_table(db_root.make_catalog_path('films'))
        assert t.get_metadata()['columns']['revenue']['comment'] == 'worldwide gross'
        t.insert(
            [
                {'title': 'Arrival', 'year': 2016, 'budget': 50.0, 'revenue': 200.0},
                {'title': 'Primer', 'year': 2004, 'budget': 0.5, 'revenue': 1.0},
            ]
        )
        rows = list(
            t.select(t.title_upper, t.title_len, t.profit, t.margin, t.is_hit, t.decade, t.label)
            .order_by(t.year)
            .collect()
        )
        assert rows == [
            {
                'title_upper': 'PRIMER',
                'title_len': 6,
                'profit': 0.5,
                'margin': 1.0,
                'is_hit': False,
                'decade': 2000,
                'label': 'Primer (2004)',
            },
            {
                'title_upper': 'ARRIVAL',
                'title_len': 7,
                'profit': 150.0,
                'margin': 3.0,
                'is_hit': True,
                'decade': 2010,
                'label': 'Arrival (2016)',
            },
        ]

        # the same export again leaves the table up to date
        again = typescript.models_from_export(_export(_FILMS_COLUMNS), module='app')
        assert again.get_model_diff(root)['films'].resolution == 'up_to_date'

    def test_schema_changes(self, db_root: DatabaseRoot) -> None:
        root = db_root.make_catalog_path('')
        typescript.models_from_export(_export(_FILMS_COLUMNS), module='app').update_all(root)

        added = [*_FILMS_COLUMNS, ['is_long', {'value': {'op': 'gte', 'args': [{'col': 'title_len'}, 7]}}]]
        base = typescript.models_from_export(_export(added), module='app')
        assert base.get_model_diff(root)['films'].resolution == 'update_additive'
        base.update_all(root)
        assert 'is_long' in pxt.get_table(db_root.make_catalog_path('films')).columns()

        dropped = typescript.models_from_export(_export(_FILMS_COLUMNS), module='app')
        with pxt_raises(pxt.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE):
            dropped.update_all(root)
        dropped.update_all(root, allow_destructive=True)
        assert 'is_long' not in pxt.get_table(db_root.make_catalog_path('films')).columns()

    @pytest.mark.parametrize(
        ('value', 'code', 'match'),
        [
            # an expression's internals are not methods
            ({'method': 'as_dict', 'on': {'col': 'title'}}, 'INVALID_SCHEMA', r"has no method or property 'as_dict'"),
            ({'prop': 'upper', 'on': {'col': 'title'}}, 'INVALID_SCHEMA', r"'upper' is a method of String"),
            ({'col': 'missing'}, 'INVALID_SCHEMA', r"unknown column 'missing'"),
            ({'call': 'os.system', 'args': ['true']}, 'INVALID_SCHEMA', r"'os.system' is not a Pixeltable function"),
            ({'call': 'pixeltable.functions.string.__builtins__'}, 'INVALID_SCHEMA', r'invalid function path'),
            ({'op': 'xor', 'args': [True, False]}, 'INVALID_SCHEMA', r"unknown operator 'xor'"),
            # the DSL's own error, prefixed with the schema file's model and column
            (
                {'op': 'sub', 'args': [{'col': 'title'}, 1]},
                'TYPE_MISMATCH',
                r"model `Films`, column 'x': title - 1: - requires numeric types",
            ),
        ],
    )
    def test_refused_nodes(self, init_env: None, value: dict[str, Any], code: str, match: str) -> None:
        columns = [['title', {'type': {'type': 'string', 'nullable': False}}], ['x', {'value': value}]]
        with pxt_raises(pxt.ErrorCode[code], match=match):
            typescript.models_from_export(_export(columns), module='app')
