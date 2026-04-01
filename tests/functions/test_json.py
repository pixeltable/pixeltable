import pixeltable as pxt
import pixeltable.functions as pxtf


class TestJson:
    def test_list_iterator(self, uses_db: None) -> None:
        schema = {
            'id': pxt.Int,
            'col_1': pxt.Json[[{'a': pxt.Int, 'b': pxt.String, 'c': pxt.Json[[int]]}]],
            'col_2': pxt.Json[[int]],
            'col_3': pxt.Json[[str]],
        }
        t = pxt.create_table('test_table', schema)
        t.insert(
            {
                'id': i,
                'col_1': [{'a': j, 'b': f'string_{j}', 'c': list(range(j))} for j in range(i)],
                'col_2': list(range(i)),
                'col_3': [f'string_{j}' for j in range(i)],
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
        }
        res = v1.order_by(v1.id, v1.pos).collect()
        assert res['a'] == [j for i in range(50) for j in range(i)]
        assert res['b'] == [f'string_{j}' for i in range(50) for j in range(i)]
        assert res['c'] == [list(range(j)) for i in range(50) for j in range(i)]
