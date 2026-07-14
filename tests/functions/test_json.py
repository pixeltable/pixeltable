# mypy: disable-error-code="misc"


import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import pxt_raises, rerun, skip_test_if_no_config, skip_test_if_not_installed

pytestmark = pytest.mark.local('UDF/integration test')


class TestJson:
    def test_list_iterator(self, uses_db: None) -> None:
        schema = {
            'id': pxt.Int,
            'col_1': pxt.Json[[{'a': pxt.Int, 'b': pxt.String, 'c': pxt.Json[[int]]}]],
            'col_2': pxt.Json[[int]],
            'col_3': pxt.Json[[str]],
            'col_4': pxt.Json[[int]],
        }
        t = pxt.create_table('test_table', schema)
        t.insert(
            {
                'id': i,
                'col_1': [{'a': j, 'b': f'string_{j}', 'c': list(range(j))} for j in range(i)],
                'col_2': list(range(i)),
                'col_3': [f'string_{j}' for j in range(i)],
                'col_4': list(range(i + 1)),  # Longer list
            }
            for i in range(50)
        )

        # Type 1: Iterate over a list of dicts

        v1 = pxt.create_view('test_view_1', t, iterator=pxtf.json.list_iterator(t.col_1))
        schema = {col: col_md['type_'] for col, col_md in v1.get_metadata()['columns'].items()}
        assert schema == {
            'pos': 'Required[Int]',
            'a': 'Required[Int]',
            'b': 'Required[String]',
            'c': 'Required[Json[(Int, ...)]]',
            'id': 'Int',
            'col_1': "Json[(Json[{'a': Int, 'b': String, 'c': Json[(Int, ...)]}], ...)]",
            'col_2': 'Json[(Int, ...)]',
            'col_3': 'Json[(String, ...)]',
            'col_4': 'Json[(Int, ...)]',
        }
        res = v1.order_by(v1.id, v1.pos).collect()
        assert res['a'] == [j for i in range(50) for j in range(i)]
        assert res['b'] == [f'string_{j}' for i in range(50) for j in range(i)]
        assert res['c'] == [list(range(j)) for i in range(50) for j in range(i)]

        # Type 2: Iterate over keyword arguments of lists

        v2 = pxt.create_view('test_view_2', t, iterator=pxtf.json.list_iterator(my_int=t.col_2, my_str=t.col_3))
        schema = {col: col_md['type_'] for col, col_md in v2.get_metadata()['columns'].items()}
        assert schema == {
            'pos': 'Required[Int]',
            'my_int': 'Required[Int]',
            'my_str': 'Required[String]',
            'id': 'Int',
            'col_1': "Json[(Json[{'a': Int, 'b': String, 'c': Json[(Int, ...)]}], ...)]",
            'col_2': 'Json[(Int, ...)]',
            'col_3': 'Json[(String, ...)]',
            'col_4': 'Json[(Int, ...)]',
        }

        # Unequal length lists with mode='strict'
        with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match=r'zip\(\) argument 2 is shorter than argument 1'):
            _ = pxt.create_view('test_view_3', t, iterator=pxtf.json.list_iterator(my_int=t.col_4, my_str=t.col_3))

        # Unequal length lists with mode='truncated'
        v4 = pxt.create_view(
            'test_view_4', t, iterator=pxtf.json.list_iterator(my_int=t.col_4, my_str=t.col_3, mode='truncated')
        )
        res = v4.order_by(v4.id, v4.pos).collect()
        assert res['my_int'] == [j for i in range(50) for j in range(i)]
        assert res['my_str'] == [f'string_{j}' for i in range(50) for j in range(i)]

        # Unequal length lists with mode='padded'
        v5 = pxt.create_view(
            'test_view_5', t, iterator=pxtf.json.list_iterator(my_int=t.col_4, my_str=t.col_3, mode='padded')
        )
        res = v5.order_by(v5.id, v5.pos).collect()
        assert res['my_int'] == [j for i in range(50) for j in range(i + 1)]
        assert res['my_str'] == [f'string_{j}' if j < i else None for i in range(50) for j in range(i + 1)]

    def test_len_and_is_empty(self, uses_db: None) -> None:
        t = pxt.create_table('json_len', {'id': pxt.Int, 'j': pxt.Json})
        t.insert(
            [
                {'id': 1, 'j': [1, 2, 3]},
                {'id': 2, 'j': []},
                {'id': 3, 'j': {'a': 1, 'b': 2}},
                {'id': 4, 'j': {}},
                {'id': 5, 'j': 'hello'},
                {'id': 6, 'j': ''},
                {'id': 7, 'j': None},
            ]
        )

        # len: array elements, object keys, string characters; null -> null
        rs = t.select(t.id, t.j, n=t.j.len()).collect()
        assert rs.schema['n'] == 'Int'
        assert {r['id']: r['n'] for r in rs} == {1: 3, 2: 0, 3: 2, 4: 0, 5: 5, 6: 0, 7: None}
        # SQL pushdown in a filter exercises the array, object and string cases of the to_sql translation
        assert sorted(r['id'] for r in t.where(t.j.len() > 0).select(t.id).collect()) == [1, 3, 5]

        # is_empty: null and empty arrays/objects/strings are empty; numbers/booleans are not
        rs = t.select(t.id, t.j, e=t.j.is_empty()).collect()
        assert rs.schema['e'] == 'Required[Bool]'
        assert {r['id']: r['e'] for r in rs} == {1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: True}
        assert sorted(r['id'] for r in t.where(t.j.is_empty()).select(t.id).collect()) == [2, 4, 6, 7]

        # map() evaluates the function per element in Python (no SQL pushdown in a nested scope)
        tm = pxt.create_table('json_len_map', {'sized': pxt.Json, 'mixed': pxt.Json})
        tm.insert([{'sized': [[1, 2, 3], 'ab', {'a': 1}, [], {}, ''], 'mixed': [[1], [], 'x', '', 0, None]}])
        rs = tm.select(o=pxtf.map(tm.sized, lambda x: x.len())).collect()
        assert rs.schema['o'] == 'Json[(Int | None, ...)]'
        assert [r['o'] for r in rs] == [[3, 2, 1, 0, 0, 0]]
        rs = tm.select(o=pxtf.map(tm.mixed, lambda x: x.is_empty())).collect()
        assert rs.schema['o'] == 'Json[(Bool, ...)]'
        assert [r['o'] for r in rs] == [[False, True, False, True, False, True]]

        # len() of a number is undefined; it raises cleanly in Python and as a raw DB error when pushed down
        tnum = pxt.create_table('json_len_num', {'scalar': pxt.Json, 'lst': pxt.Json})
        tnum.insert([{'scalar': 5, 'lst': [5]}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='not defined for a JSON'):
            tnum.select(o=pxtf.map(tnum.lst, lambda x: x.len())).collect()
        # TODO: the pushed-down scalar error surfaces as a raw DB error, not a clean pxt.Error; wrap it.
        with pytest.raises(Exception, match='scalar'):
            tnum.select(tnum.scalar, n=tnum.scalar.len()).collect()

        # TODO: selecting a pushed-down json function without also selecting the source column fails, because the
        # column's slot is never materialized (cell-reconstruction bug); re-enable once fixed.
        # assert [r['n'] for r in t.order_by(t.id).select(n=t.j.len()).collect()] == [3, 0, 2, 0, 5, 0, None]

    def test_contains(self, uses_db: None) -> None:
        t = pxt.create_table('json_contains', {'id': pxt.Int, 'j': pxt.Json})
        t.insert(
            [
                {'id': 1, 'j': ['person', 'car']},
                {'id': 2, 'j': ['dog']},
                {'id': 3, 'j': {'person': 1}},  # object: key membership
                {'id': 4, 'j': 'person'},  # string: no containment notion here
                {'id': 5, 'j': None},
            ]
        )
        rs = t.select(t.id, t.j, c=t.j.contains('person')).collect()
        assert rs.schema['c'] == 'Bool'
        assert {r['id']: r['c'] for r in rs} == {1: True, 2: False, 3: True, 4: False, 5: None}
        assert sorted(r['id'] for r in t.where(t.j.contains('person')).select(t.id).collect()) == [1, 3]

    def test_get(self, uses_db: None) -> None:
        t = pxt.create_table('json_get', {'id': pxt.Int, 'j': pxt.Json})
        t.insert(
            [
                {'id': 1, 'j': {'author': 'alice'}},
                {'id': 2, 'j': {'x': 1}},  # missing key -> default
                {'id': 3, 'j': [1, 2]},  # non-object -> default
                {'id': 4, 'j': None},  # null -> default
            ]
        )
        rs = t.select(t.id, t.j, a=t.j.get('author', default='unknown')).collect()
        assert rs.schema['a'] == 'Json'  # untyped input
        assert {r['id']: r['a'] for r in rs} == {1: 'alice', 2: 'unknown', 3: 'unknown', 4: 'unknown'}
        # default is None when unspecified
        rs = t.select(t.id, t.j, a=t.j.get('author')).collect()
        assert rs.schema['a'] == 'Json'
        assert {r['id']: r['a'] for r in rs} == {1: 'alice', 2: None, 3: None, 4: None}

        # return type: a literal key resolves against a typed object schema; a missing key, a non-object schema
        # and an untyped input all fall back to Json
        td = pxt.create_table(
            'json_get_typed', {'d': pxt.Json[{'scores': pxt.Json[[float]]}], 'lst': pxt.Json[[int]], 'raw': pxt.Json}
        )
        td.insert([{'d': {'scores': [0.1, 0.9]}, 'lst': [10, 20], 'raw': {'x': 7}}])
        rs = td.select(
            hit=td.d.get('scores'), miss=td.d.get('nope'), non_dict=td.lst.get('x'), untyped=td.raw.get('x')
        ).collect()
        assert rs.schema == {'hit': 'Json[(Float, ...)]', 'miss': 'Json', 'non_dict': 'Json', 'untyped': 'Json'}
        assert {k: rs[0][k] for k in ('hit', 'miss', 'non_dict', 'untyped')} == {
            'hit': [0.1, 0.9],
            'miss': None,
            'non_dict': None,
            'untyped': 7,
        }
        # the resolved element type chains into a downstream numeric reduction
        rs = td.select(m=td.d.get('scores').mean()).collect()
        assert rs.schema['m'] == 'Float'
        assert rs[0]['m'] == 0.5

    def test_numeric_agg(self, uses_db: None) -> None:
        t = pxt.create_table('json_reduce', {'id': pxt.Int, 'j': pxt.Json})
        t.insert(
            [
                {'id': 1, 'j': [3, 1, 2]},
                {'id': 2, 'j': [1.5, 2.5]},
                {'id': 3, 'j': []},  # empty: sum 0, others null
                {'id': 4, 'j': None},  # null -> null
            ]
        )
        rs = t.select(t.id, t.j, o=t.j.sum()).collect()
        assert rs.schema['o'] == 'Float'
        assert {r['id']: r['o'] for r in rs} == {1: 6.0, 2: 4.0, 3: 0.0, 4: None}
        rs = t.select(t.id, t.j, o=t.j.min()).collect()
        assert rs.schema['o'] == 'Float'
        assert {r['id']: r['o'] for r in rs} == {1: 1.0, 2: 1.5, 3: None, 4: None}
        rs = t.select(t.id, t.j, o=t.j.max()).collect()
        assert rs.schema['o'] == 'Float'
        assert {r['id']: r['o'] for r in rs} == {1: 3.0, 2: 2.5, 3: None, 4: None}
        rs = t.select(t.id, t.j, o=t.j.mean()).collect()
        assert rs.schema['o'] == 'Float'
        assert {r['id']: r['o'] for r in rs} == {1: 2.0, 2: 2.0, 3: None, 4: None}

        # SQL pushdown in a filter exercises each aggregate's to_sql translation
        assert sorted(r['id'] for r in t.where(t.j.sum() > 4).select(t.id).collect()) == [1]
        assert sorted(r['id'] for r in t.where(t.j.min() < 1.5).select(t.id).collect()) == [1]
        assert sorted(r['id'] for r in t.where(t.j.max() > 2.5).select(t.id).collect()) == [1]
        assert sorted(r['id'] for r in t.where(t.j.mean() >= 2.0).select(t.id).collect()) == [1, 2]

        # map() evaluates the aggregates per element in Python, including the empty-array cases
        tm = pxt.create_table('json_reduce_map', {'rows': pxt.Json})
        tm.insert([{'rows': [[3, 1, 2], [1.5, 2.5], []]}])
        for agg, expected in (
            (lambda x: x.sum(), [6.0, 4.0, 0.0]),
            (lambda x: x.min(), [1.0, 1.5, None]),
            (lambda x: x.max(), [3.0, 2.5, None]),
            (lambda x: x.mean(), [2.0, 2.0, None]),
        ):
            rs = tm.select(o=pxtf.map(tm.rows, agg)).collect()
            assert rs.schema['o'] == 'Json[(Float | None, ...)]'
            assert [r['o'] for r in rs] == [expected]

        # not defined for a non-numeric array: the Python body raises a clean error, while the pushed-down form
        # surfaces a raw DB cast error (TODO'd in json.py)
        ts = pxt.create_table('json_reduce_err', {'rows': pxt.Json, 'j': pxt.Json})
        ts.insert([{'rows': [['a', 'b']], 'j': ['a', 'b']}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='array of numbers'):
            ts.select(o=pxtf.map(ts.rows, lambda x: x.mean())).collect()
        with pytest.raises(Exception, match='numeric'):
            ts.select(ts.j, o=ts.j.mean()).collect()

    def test_count(self, uses_db: None) -> None:
        t = pxt.create_table('json_count', {'id': pxt.Int, 'j': pxt.Json})
        t.insert([{'id': 1, 'j': ['a', 'b', 'a', 'a']}, {'id': 2, 'j': []}, {'id': 3, 'j': None}])
        rs = t.select(t.id, t.j, o=t.j.count('a')).collect()
        assert rs.schema['o'] == 'Int'
        assert {r['id']: r['o'] for r in rs} == {1: 3, 2: 0, 3: None}
        # count() is only defined for an array
        te = pxt.create_table('json_count_err', {'j': pxt.Json})
        te.insert([{'j': {'a': 1}}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='JSON array'):
            te.select(te.j, o=te.j.count('a')).collect()

    def test_object_accessors(self, uses_db: None) -> None:
        t = pxt.create_table('json_obj', {'id': pxt.Int, 'j': pxt.Json})
        t.insert([{'id': 1, 'j': {'a': 1, 'b': 2}}, {'id': 2, 'j': {}}, {'id': 3, 'j': None}])
        rs = t.select(t.id, t.j, o=t.j.keys()).collect()
        assert rs.schema['o'] == 'Json[(String, ...)]'  # keys are always strings
        assert {r['id']: r['o'] for r in rs} == {1: ['a', 'b'], 2: [], 3: None}
        rs = t.select(t.id, t.j, o=t.j.values()).collect()
        assert rs.schema['o'] == 'Json'  # untyped input: value types are unknown
        assert {r['id']: r['o'] for r in rs} == {1: [1, 2], 2: [], 3: None}
        rs = t.select(t.id, t.j, o=t.j.items()).collect()
        assert rs.schema['o'] == 'Json'  # items() never carries key/value type information
        assert {r['id']: r['o'] for r in rs} == {1: [['a', 1], ['b', 2]], 2: [], 3: None}

        # values() over a typed object schema: the common supertype of the value types for a homogeneous object,
        # Json when the value types have no common supertype
        tv = pxt.create_table(
            'json_values_typed',
            {'homog': pxt.Json[{'a': pxt.Int, 'b': pxt.Int}], 'hetero': pxt.Json[{'a': pxt.Int, 'b': pxt.String}]},
        )
        tv.insert([{'homog': {'a': 1, 'b': 2}, 'hetero': {'a': 1, 'b': 'x'}}])
        rs = tv.select(homog=tv.homog.values(), hetero=tv.hetero.values()).collect()
        assert rs.schema == {'homog': 'Json[(Int, ...)]', 'hetero': 'Json[(Json, ...)]'}
        assert rs[0]['homog'] == [1, 2]
        assert rs[0]['hetero'] == [1, 'x']

        # keys(), values() and items() are only defined for an object
        te = pxt.create_table('json_obj_err', {'j': pxt.Json})
        te.insert([{'j': [1, 2]}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='JSON object'):
            te.select(te.j, o=te.j.keys()).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='JSON object'):
            te.select(te.j, o=te.j.values()).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='JSON object'):
            te.select(te.j, o=te.j.items()).collect()

    def test_flatten(self, uses_db: None) -> None:
        t = pxt.create_table('json_flatten', {'id': pxt.Int, 'j': pxt.Json})
        t.insert(
            [
                {'id': 1, 'j': [[1, 2], [3], [4, 5]]},
                {'id': 2, 'j': [[1], 2, [3]]},  # non-array elements pass through as-is
                {'id': 3, 'j': None},
            ]
        )
        rs = t.select(t.id, t.j, o=t.j.flatten()).collect()
        assert rs.schema['o'] == 'Json'  # untyped input
        assert {r['id']: r['o'] for r in rs} == {1: [1, 2, 3, 4, 5], 2: [1, 2, 3], 3: None}

        # return type: Json[[[T]]] flattens one level to Json[[T]]; a single-level list or an untyped input
        # flattens to Json
        tt = pxt.create_table('json_flatten_typed', {'nested': pxt.Json[[[int]]], 'flat': pxt.Json[[int]]})
        tt.insert([{'nested': [[1, 2], [3]], 'flat': [4, 5]}])
        rs = tt.select(nested=tt.nested.flatten(), flat=tt.flat.flatten()).collect()
        assert rs.schema == {'nested': 'Json[(Int, ...)]', 'flat': 'Json'}
        assert rs[0]['nested'] == [1, 2, 3]
        assert rs[0]['flat'] == [4, 5]

        # flatten() is only defined for an array
        te = pxt.create_table('json_flatten_err', {'j': pxt.Json})
        te.insert([{'j': {'a': 1}}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='JSON array'):
            te.select(te.j, o=te.j.flatten()).collect()

    def test_map_filter_edge_cases(self, uses_db: None) -> None:
        t = pxt.create_table('json_map_filter', {'id': pxt.Int, 'j': pxt.Json})
        # empty-list and null sources
        t.insert([{'id': 1, 'j': [1, -2, 3, -4]}, {'id': 2, 'j': []}, {'id': 3, 'j': None}])
        # an empty source list yields an empty result; a null source yields null
        assert {r['id']: r['o'] for r in t.select(t.id, o=t.j.map(lambda x: x * 2)).collect()} == {
            1: [2, -4, 6, -8],
            2: [],
            3: None,
        }
        assert {r['id']: r['o'] for r in t.select(t.id, o=t.j.filter(lambda x: x > 0)).collect()} == {
            1: [1, 3],
            2: [],
            3: None,
        }

        # method/field name collision
        # a field colliding with a method name stays reachable via subscript, while `.map` is the method
        tc = pxt.create_table('json_map_filter_collide', {'j': pxt.Json})
        tc.insert([{'j': {'map': 5, 'nums': [1, 2, 3]}}])
        assert tc.select(o=tc.j['map']).collect()['o'] == [5]
        assert tc.select(o=tc.j.nums.map(lambda x: x + 1)).collect()['o'] == [[2, 3, 4]]

    def test_sort(self, uses_db: None) -> None:
        t = pxt.create_table('json_sort', {'id': pxt.Int, 'nums': pxt.Json[[int]], 'objs': pxt.Json})
        t.insert(
            [
                {
                    'id': 1,
                    'nums': [3, 1, 2],
                    'objs': [{'s': 0.5, 'l': 'b'}, {'s': 0.9, 'l': 'a'}, {'s': 0.1, 'l': 'c'}],
                },
                {'id': 2, 'nums': [], 'objs': []},  # empty list sorts to empty
                {'id': 3, 'nums': None, 'objs': None},  # null source sorts to null
            ]
        )

        # keyless sort preserves the element type of a typed list
        res = t.select(t.id, a=t.nums.sort(), d=t.nums.sort(asc=False)).order_by(t.id).collect()
        assert res.schema['a'] == 'Json[(Int, ...)]'
        assert res['a'] == [[1, 2, 3], [], None]
        assert res['d'] == [[3, 2, 1], [], None]
        # the function form is interchangeable with the method form
        assert t.select(a=pxtf.sort(t.nums)).order_by(t.id).collect()['a'] == res['a']

        # keyed sort orders dicts by a per-element key and reproduces the dicts themselves
        res2 = t.select(t.id, by_score=t.objs.sort(key=lambda x: x.s, asc=False)).order_by(t.id).collect()
        assert [[o['l'] for o in v] if v is not None else v for v in res2['by_score']] == [['a', 'b', 'c'], [], None]

        # a keyless sort of non-orderable elements raises
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'sort\(\): the array elements are not orderable'):
            t.select(o=t.objs.sort()).collect()

        # a keyed sort with non-orderable keys raises (a null mixed with numbers)
        tn = pxt.create_table('json_sort_null_key', {'j': pxt.Json})
        tn.insert([{'j': [{'v': 1}, {'v': None}, {'v': 3}]}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'sort\(\): the sort keys are not orderable'):
            tn.select(o=tn.j.sort(key=lambda x: x.v)).collect()

    @pytest.mark.very_expensive  # Downloads a Hugging Face model
    @rerun(reruns=3, reruns_delay=15, only_rerun=['429', 'Too Many Requests'])
    def test_list_iterator_appl(self, uses_db: None) -> None:
        """
        Fully worked example of flattening object detection output.
        """
        skip_test_if_no_config('token', 'hf')
        skip_test_if_not_installed('transformers')

        t = pxt.create_table('img_table', {'id': pxt.Int, 'img': pxt.Image})
        t.insert(
            {
                'id': id,
                'img': f'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/{name}',
            }
            for id, name in enumerate(('000000000009.jpg', '000000000016.jpg'))
        )
        t.add_computed_column(
            detections=pxtf.huggingface.detr_for_object_detection(
                t.img, model_id='facebook/detr-resnet-50', threshold=0.8
            )
        )
        v = pxt.create_view(
            'detections_view',
            t,
            iterator=pxtf.json.list_iterator(
                score=t.detections.scores,
                label=t.detections.labels,
                label_text=t.detections.label_text,
                box=t.detections.boxes,
            ),
        )
        res = v.order_by(v.id, v.pos).collect()
        assert len(res) == 13
        assert set(res['label_text']) == {'baseball bat', 'bowl', 'broccoli', 'orange', 'person', 'sports ball'}

    def test_list_iterator_errors(self, uses_db: None) -> None:
        t = pxt.create_table(
            'test_table',
            {
                'col_1': pxt.Json[[{'a': pxt.Int}]],
                'col_2': pxt.Json[[int]],
                'col_3': pxt.String,  # primitive type
                'col_4': pxt.Json,  # untyped json
                'col_5': pxt.Json[list],  # typed json, but with an untyped subscript
            },
        )

        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            match=r'list_iterator\(\): `mode` argument cannot be used with `elements`',
        ):
            pxtf.json.list_iterator(t.col_1, mode='truncated')

        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            match=r'list_iterator\(\): Cannot specify both `elements` and keyword arguments',
        ):
            pxtf.json.list_iterator(t.col_1, my_int=t.col_2)

        invalid_type_cases = (t.col_2, t.col_3, t.col_4, t.col_5)
        for expr in invalid_type_cases:
            with pxt_raises(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                match=r'list_iterator\(\): Expected a type for `elements` matching `list\[dict\]`',
            ):
                pxtf.json.list_iterator(expr)

        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match=r'list_iterator\(\): No inputs provided'):
            pxtf.json.list_iterator()

        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match=r'list_iterator\(\): No inputs provided'):
            pxtf.json.list_iterator(mode='padded')

        invalid_type_cases_2 = (t.col_3, t.col_4)
        for expr in invalid_type_cases_2:
            with pxt_raises(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                match=r'list_iterator\(\): Expected a type for `my_int` matching `list`',
            ):
                pxtf.json.list_iterator(my_int=expr)
