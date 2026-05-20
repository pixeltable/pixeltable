# mypy: disable-error-code="misc"


import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import pxt_raises, skip_test_if_not_installed


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

    def test_list_iterator_appl(self, uses_db: None) -> None:
        """
        Fully worked example of flattening object detection output.
        """
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
