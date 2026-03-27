"""Tests for pixeltable.dashboard.bridge — the translation layer between Pixeltable APIs and the dashboard REST API."""

import numpy as np

import pixeltable as pxt
from pixeltable.dashboard import bridge


@pxt.udf
def dummy_embed(text: str) -> pxt.Array[(3,), pxt.Float]:
    return np.array([1.0, 2.0, 3.0])


class TestBridge:
    def test_directory_tree_empty(self, uses_db: None) -> None:
        assert bridge.get_directory_tree() == []

    def test_directory_tree(self, uses_db: None) -> None:
        pxt.create_dir('a')
        pxt.create_dir('a/b')
        t = pxt.create_table('a/t1', {'c1': pxt.String})
        pxt.create_table('a/b/t2', {'c1': pxt.String})
        pxt.create_view('a/v', t)
        pxt.create_view('a/snap', t, is_snapshot=True)

        tree = bridge.get_directory_tree()
        assert len(tree) == 1
        root = tree[0]
        assert root['name'] == 'a'
        assert root['kind'] == 'directory'

        children_by_name = {n['name']: n for n in root['children']}
        assert children_by_name['t1']['kind'] == 'table'
        assert children_by_name['v']['kind'] == 'view'
        assert children_by_name['snap']['kind'] == 'snapshot'
        assert children_by_name['t1']['error_count'] == 0
        assert children_by_name['b']['kind'] == 'directory'
        assert len(children_by_name['b']['children']) == 1

    def test_table_metadata_basic(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String, 'c2': pxt.Required[pxt.Int]}, primary_key='c2')
        t.add_computed_column(upper=t.c1.upper())
        t.insert([{'c1': 'hello', 'c2': 1}])

        result = bridge.get_table_metadata('md/t')
        assert result['path'] == 'md/t'
        assert result['name'] == 't'
        assert result['is_view'] is False
        assert result['is_snapshot'] is False
        assert result['base'] is None
        assert result['iterator_expr'] is None
        assert isinstance(result['version'], int)

        # columns is a dict keyed by column name
        assert isinstance(result['columns'], dict)
        assert 'c1' in result['columns']
        assert result['columns']['c1']['is_computed'] is False
        assert result['columns']['c2']['is_primary_key'] is True
        assert result['columns']['upper']['is_computed'] is True
        assert result['columns']['upper']['computed_with'] is not None

    def test_table_metadata_view(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/base', {'c1': pxt.String})
        pxt.create_view('md/v', t)
        result = bridge.get_table_metadata('md/v')
        assert result['is_view'] is True
        assert result['base'] == 'md/base'

    def test_table_metadata_indices(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String})
        t.add_embedding_index('c1', embedding=dummy_embed)
        result = bridge.get_table_metadata('md/t')
        # indices is a dict keyed by index name
        assert isinstance(result['indices'], dict)
        assert len(result['indices']) > 0
        idx = next(iter(result['indices'].values()))
        assert 'name' in idx
        assert 'columns' in idx
        assert 'index_type' in idx

    def test_table_data_basic(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String, 'c2': pxt.Int})
        t.insert([{'c1': 'hello', 'c2': 1}, {'c1': 'world', 'c2': 2}])

        result = bridge.get_table_data('td/t')
        assert result['total_count'] == 2
        assert len(result['rows']) == 2
        col_names = [c['name'] for c in result['columns']]
        assert 'c1' in col_names
        assert 'c2' in col_names
        for col in result['columns']:
            assert 'type' in col
            assert 'is_media' in col
            assert 'is_computed' in col

    def test_table_data_empty(self, uses_db: None) -> None:
        pxt.create_dir('td')
        pxt.create_table('td/t', {'c1': pxt.String})
        result = bridge.get_table_data('td/t')
        assert result['rows'] == []
        assert result['total_count'] == 0

    def test_table_data_pagination(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Int})
        t.insert([{'c1': i} for i in range(10)])
        page1 = bridge.get_table_data('td/t', offset=0, limit=3)
        assert len(page1['rows']) == 3
        assert page1['total_count'] == 10
        page2 = bridge.get_table_data('td/t', offset=5, limit=3)
        assert len(page2['rows']) == 3

    def test_table_data_order_by(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Int})
        t.insert([{'c1': 3}, {'c1': 1}, {'c1': 2}])
        asc = bridge.get_table_data('td/t', order_by='c1', order_desc=False)
        assert [r['c1'] for r in asc['rows']] == [1, 2, 3]
        desc = bridge.get_table_data('td/t', order_by='c1', order_desc=True)
        assert [r['c1'] for r in desc['rows']] == [3, 2, 1]

    def test_table_data_computed_column(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String})
        t.add_computed_column(upper=t.c1.upper())
        t.insert([{'c1': 'hello'}])
        result = bridge.get_table_data('td/t')
        assert result['rows'][0]['upper'] == 'HELLO'

    def test_table_data_nulls(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String, 'c2': pxt.Int})
        t.insert([{'c1': None, 'c2': None}])
        row = bridge.get_table_data('td/t')['rows'][0]
        assert row['c1'] is None
        assert row['c2'] is None

    def test_table_data_json(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Json})
        t.insert([{'c1': {'key': 'value', 'num': 42}}])
        row = bridge.get_table_data('td/t')['rows'][0]
        assert row['c1'] == {'key': 'value', 'num': 42}

    def test_export_csv(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.String, 'c2': pxt.Int})
        t.insert([{'c1': 'hello', 'c2': 1}, {'c1': 'world', 'c2': 2}])
        csv_str = bridge.export_table_csv('ex/t').decode('utf-8')
        lines = csv_str.strip().split('\n')
        assert len(lines) == 3  # header + 2 rows
        assert 'c1' in lines[0]
        assert 'c2' in lines[0]

    def test_export_csv_empty(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        pxt.create_table('ex/t', {'c1': pxt.String})
        lines = bridge.export_table_csv('ex/t').decode('utf-8').strip().split('\n')
        assert len(lines) == 1  # header only

    def test_export_csv_limit(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.Int})
        t.insert([{'c1': i} for i in range(10)])
        lines = bridge.export_table_csv('ex/t', limit=3).decode('utf-8').strip().split('\n')
        assert len(lines) == 4  # header + 3 rows

    def test_export_csv_json_column(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.Json})
        t.insert([{'c1': {'key': 'val'}}])
        csv_str = bridge.export_table_csv('ex/t').decode('utf-8')
        assert 'key' in csv_str

    def test_search_empty_db(self, uses_db: None) -> None:
        result = bridge.search('anything')
        assert result['query'] == 'anything'
        assert result['directories'] == []
        assert result['tables'] == []
        assert result['columns'] == []

    def test_search_finds_dir_table_column(self, uses_db: None) -> None:
        pxt.create_dir('proj')
        pxt.create_table('proj/users', {'email': pxt.String, 'age': pxt.Int})

        r = bridge.search('proj')
        assert len(r['directories']) == 1
        assert r['directories'][0]['path'] == 'proj'

        r = bridge.search('users')
        assert len(r['tables']) == 1
        assert r['tables'][0]['path'] == 'proj/users'

        r = bridge.search('email')
        assert len(r['columns']) == 1
        assert r['columns'][0]['name'] == 'email'
        assert r['columns'][0]['table'] == 'proj/users'

    def test_search_case_insensitive(self, uses_db: None) -> None:
        pxt.create_dir('MyDir')
        assert len(bridge.search('mydir')['directories']) == 1

    def test_search_limit(self, uses_db: None) -> None:
        pxt.create_dir('sl')
        for i in range(5):
            pxt.create_table(f'sl/match_{i}', {'c1': pxt.String})
        assert len(bridge.search('match', limit=3)['tables']) == 3

    def test_pipeline_empty(self, uses_db: None) -> None:
        assert bridge.get_pipeline() == {'nodes': [], 'edges': []}

    def test_pipeline_single_table(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String})
        t.insert([{'c1': 'hello'}])
        result = bridge.get_pipeline()
        assert len(result['nodes']) == 1
        node = result['nodes'][0]
        assert node['path'] == 'pp/t'
        assert node['name'] == 't'
        assert node['is_view'] is False
        assert node['row_count'] == 1
        expected_keys = {
            'path',
            'name',
            'is_view',
            'base',
            'row_count',
            'version',
            'total_errors',
            'columns',
            'indices',
            'versions',
            'computed_count',
            'insertable_count',
            'iterator_type',
        }
        assert expected_keys.issubset(node.keys())
        # Non-computed column has no func info
        cols_by_name = {c['name']: c for c in node['columns']}
        assert cols_by_name['c1']['func_name'] is None
        assert cols_by_name['c1']['func_type'] is None

    def test_pipeline_view_edge(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/base', {'c1': pxt.String})
        pxt.create_view('pp/v', t)
        result = bridge.get_pipeline()
        assert len(result['nodes']) == 2
        assert len(result['edges']) == 1
        edge = result['edges'][0]
        assert edge['source'] == 'pp/base'
        assert edge['target'] == 'pp/v'
        assert edge['type'] == 'view'

    def test_pipeline_computed_columns(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String})
        t.add_computed_column(upper=t.c1.upper())
        node = bridge.get_pipeline()['nodes'][0]
        assert node['computed_count'] == 1
        assert node['insertable_count'] == 1
        cols_by_name = {c['name']: c for c in node['columns']}
        assert cols_by_name['c1']['func_name'] is None
        assert cols_by_name['c1']['func_type'] is None
        assert cols_by_name['upper']['func_name'] == 'upper'
        assert cols_by_name['upper']['func_type'] == 'builtin'

    def test_pipeline_func_name(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String})
        t.add_computed_column(emb=dummy_embed(t.c1))
        node = bridge.get_pipeline()['nodes'][0]
        cols_by_name = {c['name']: c for c in node['columns']}
        assert cols_by_name['emb']['func_name'] == 'dummy_embed'
        assert cols_by_name['emb']['func_type'] == 'custom_udf'

    def test_status(self, uses_db: None) -> None:
        result = bridge.get_status()
        assert result['version'] == pxt.__version__
        assert result['environment'] == 'local'
        assert isinstance(result['total_tables'], int)
        assert isinstance(result['total_errors'], int)
        assert 'home' in result['config']
        assert 'media_dir' in result['config']

    def test_status_table_count(self, uses_db: None) -> None:
        pxt.create_dir('st')
        pxt.create_table('st/t1', {'c1': pxt.String})
        pxt.create_table('st/t2', {'c1': pxt.String})
        assert bridge.get_status()['total_tables'] == 2
