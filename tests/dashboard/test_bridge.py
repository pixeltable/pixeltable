"""Tests for pixeltable.dashboard.bridge — the translation layer between Pixeltable APIs and the dashboard REST API."""

import numpy as np

import pixeltable as pxt
from pixeltable.dashboard import bridge
from pixeltable.functions.video import frame_iterator

from ..utils import get_test_video_files


@pxt.udf
def my_udf(x: int) -> int:
    return x + 1


@pxt.udf
def fail_on_neg(x: int) -> int:
    if x < 0:
        raise ValueError('negative')
    return x


@pxt.udf
def dummy_embed(text: str) -> pxt.Array[(3,), pxt.Float]:
    return np.array([1.0, 2.0, 3.0])


class TestBridge:
    def test_table_metadata_basic(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String, 'c2': pxt.Required[pxt.Int]}, primary_key='c2')
        t.add_computed_column(upper=t.c1.upper())
        t.insert([{'c1': 'hello', 'c2': 1}])

        result = pxt.get_table('md/t').get_metadata()
        assert (
            result['path'],
            result['name'],
            result['is_view'],
            result['is_snapshot'],
            result['base'],
            result['iterator_call'],
        ) == ('md/t', 't', False, False, None, None)
        assert isinstance(result['version'], int)

        cols = result['columns']
        assert {n: c['is_computed'] for n, c in cols.items()} == {'c1': False, 'c2': False, 'upper': True}
        assert cols['c2']['is_primary_key'] is True
        assert cols['upper']['computed_with'] is not None

    def test_table_metadata_view(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/base', {'c1': pxt.String})
        pxt.create_view('md/v', t)
        result = pxt.get_table('md/v').get_metadata()
        assert (result['is_view'], result['base']) == (True, 'md/base')

    def test_table_metadata_indices(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String})
        t.add_embedding_index('c1', embedding=dummy_embed)
        result = pxt.get_table('md/t').get_metadata()
        assert len(result['indices']) > 0
        idx = next(iter(result['indices'].values()))
        assert {'name', 'columns', 'index_type'} <= idx.keys()

    def test_table_data_basic(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String, 'c2': pxt.Int})
        t.insert([{'c1': 'hello', 'c2': 1}, {'c1': 'world', 'c2': 2}])

        result = bridge.get_table_data('td/t')
        assert (result['total_count'], len(result['rows'])) == (2, 2)
        assert {c['name'] for c in result['columns']} == {'c1', 'c2'}
        assert all({'type', 'is_media', 'is_computed', 'is_stored'} <= c.keys() for c in result['columns'])

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
        assert bridge.get_table_data('td/t')['rows'][0] == {'c1': None, 'c2': None}

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
        assert bridge.search('anything') == {'query': 'anything', 'directories': [], 'tables': [], 'columns': []}

    def test_search_finds_dir_table_column(self, uses_db: None) -> None:
        pxt.create_dir('proj')
        pxt.create_table('proj/users', {'email': pxt.String, 'age': pxt.Int})

        assert [d['path'] for d in bridge.search('proj')['directories']] == ['proj']
        assert [t['path'] for t in bridge.search('users')['tables']] == ['proj/users']
        col = bridge.search('email')['columns']
        assert [(c['name'], c['table']) for c in col] == [('email', 'proj/users')]

    def test_search_case_insensitive(self, uses_db: None) -> None:
        pxt.create_dir('MyDir')
        assert len(bridge.search('mydir')['directories']) == 1

    def test_search_limit(self, uses_db: None) -> None:
        pxt.create_dir('sl')
        for i in range(5):
            pxt.create_table(f'sl/match_{i}', {'c1': pxt.String})
        assert len(bridge.search('match', limit=3)['tables']) == 3

    def test_pipeline(self, uses_db: None) -> None:
        assert bridge.get_pipeline() == {'nodes': [], 'edges': []}

        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String})
        t.insert([{'c1': 'hello'}])
        result = bridge.get_pipeline()
        assert len(result['nodes']) == 1
        node = result['nodes'][0]
        assert (node['path'], node['name'], node['is_view'], node['row_count']) == ('pp/t', 't', False, 1)
        expected_keys = {
            'path',
            'name',
            'is_view',
            'base',
            'row_count',
            'version',
            'columns',
            'indices',
            'versions',
            'computed_count',
            'insertable_count',
            'iterator_type',
        }
        assert expected_keys.issubset(node.keys())
        cols_by_name = {c['name']: c for c in node['columns']}
        assert (cols_by_name['c1']['func_name'], cols_by_name['c1']['func_type']) == (None, None)

    def test_pipeline_scoped(self, uses_db: None) -> None:
        # Build a chain root -> mid -> leaf, plus an unrelated standalone table.
        pxt.create_dir('sc')
        root = pxt.create_table('sc/root', {'c': pxt.String})
        mid = pxt.create_view('sc/mid', root)
        pxt.create_view('sc/leaf', mid)
        pxt.create_table('sc/other', {'c': pxt.String})

        # Scoped to mid: includes ancestor (root), self (mid), descendant (leaf); excludes 'other'.
        result = bridge.get_pipeline(tbl_path='sc/mid')
        assert {n['path'] for n in result['nodes']} == {'sc/root', 'sc/mid', 'sc/leaf'}
        assert {(e['source'], e['target']) for e in result['edges']} == {('sc/root', 'sc/mid'), ('sc/mid', 'sc/leaf')}

        # Unknown path returns empty.
        assert bridge.get_pipeline(tbl_path='sc/missing') == {'nodes': [], 'edges': []}

        # No path returns the full catalog.
        assert {n['path'] for n in bridge.get_pipeline()['nodes']} == {'sc/root', 'sc/mid', 'sc/leaf', 'sc/other'}

    def test_pipeline_view_edge(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/base', {'c1': pxt.String})
        pxt.create_view('pp/v', t)
        result = bridge.get_pipeline()
        assert len(result['nodes']) == 2
        assert [(e['source'], e['target'], e['type']) for e in result['edges']] == [('pp/base', 'pp/v', 'view')]

    def test_pipeline_computed_columns(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String, 'c2': pxt.Int})
        t.add_computed_column(upper=t.c1.upper())
        t.add_computed_column(add=t.c2 + t.c1.len())
        t.add_computed_column(add2=2 + t.c1.len())
        t.add_computed_column(add3=t.c1.len() + my_udf(t.c2))
        t.add_computed_column(plus_one=my_udf(t.c1.len()))
        node = bridge.get_pipeline()['nodes'][0]
        assert (node['computed_count'], node['insertable_count']) == (5, 2)
        funcs = {c['name']: (c['func_name'], c['func_type']) for c in node['columns']}
        assert funcs == {
            'c1': (None, None),
            'c2': (None, None),
            'upper': ('upper', 'builtin'),
            'add': ('len', 'builtin'),
            'add2': ('len', 'builtin'),
            'add3': ('len', 'custom_udf'),
            'plus_one': ('my_udf', 'custom_udf'),
        }

    def test_pipeline_snapshot_edge(self, uses_db: None) -> None:
        # Snapshot of an iterator view: validates snapshot edge wiring + version metadata.
        video_path = get_test_video_files()[0]
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.insert([{'video': video_path}])
        view = pxt.create_view('frames', video_t, iterator=frame_iterator(video_t.video, fps=1))
        pxt.create_view('frames_snap', view, is_snapshot=True)

        pipeline = bridge.get_pipeline()
        snap_node = next(n for n in pipeline['nodes'] if n['path'] == 'frames_snap')
        snap_edges = [e for e in pipeline['edges'] if e['target'] == 'frames_snap']

        assert len(snap_edges) == 1
        assert (snap_edges[0]['source'], snap_edges[0]['type'], snap_edges[0]['base_version']) == (
            'frames',
            'snapshot',
            0,
        )
        # Snapshot inherits its base view's iterator_call, so iterator_type is populated.
        # is_view is False (gate is kind == 'view'), even though iterator_type is set — surface
        # this so any future change to the gating logic is caught.
        assert snap_node['iterator_type'] == 'frame_iterator'
        assert snap_node['is_view'] is False
        assert snap_node['base'] == 'frames:0'

    def test_status(self, uses_db: None) -> None:
        result = bridge.get_status()
        assert (result['version'], result['environment']) == (pxt.__version__, 'local')
        assert isinstance(result['total_tables'], int)
        assert isinstance(result['total_errors'], int)
        assert {'home', 'media_dir'} <= result['config'].keys()

    def test_status_table_count(self, uses_db: None) -> None:
        pxt.create_dir('st')
        pxt.create_table('st/t1', {'c1': pxt.String})
        pxt.create_table('st/t2', {'c1': pxt.String})
        assert bridge.get_status()['total_tables'] == 2

    def test_table_data_unstored_column(self, uses_db: None) -> None:
        # Unstored computed columns must not be evaluated by the data view.
        # The expression here would raise on negative inputs; the test asserts that
        # get_table_data succeeds anyway and reports the column as is_stored=False.
        t = pxt.create_table('udata', {'x': pxt.Int})
        t.add_computed_column(plus_one=t.x + 1)
        t.add_computed_column(boom=fail_on_neg(t.x), stored=False)
        t.insert([{'x': 1}, {'x': -1}, {'x': 2}])

        result = bridge.get_table_data('udata')
        storage_by_name = {c['name']: c['is_stored'] for c in result['columns']}
        assert storage_by_name == {'x': True, 'plus_one': True, 'boom': False}
        assert result['rows'] == [{'x': 1, 'plus_one': 2}, {'x': -1, 'plus_one': 0}, {'x': 2, 'plus_one': 3}]

        # Sorting by an unstored column is a no-op (does not raise, does not reorder).
        sorted_result = bridge.get_table_data('udata', order_by='boom', order_desc=True)
        assert [row['x'] for row in sorted_result['rows']] == [1, -1, 2]

        # get_pipeline calls the per-column-error-count helper internally; pre-fix that helper
        # raised on the unstored 'boom' column and the table landed in the error-stub branch.
        node = next(n for n in bridge.get_pipeline()['nodes'] if n['path'] == 'udata')
        assert 'error' not in node

    def test_table_data_sort_gating(self, uses_db: None) -> None:
        # Only stored, B-tree-indexed columns should be reported as sortable. Postgres has no
        # cheap ordering for bool / json / unstored columns, so the bridge skips them.
        pxt.create_dir('s')
        t = pxt.create_table('s/t', {'name': pxt.String, 'flag': pxt.Bool, 'meta': pxt.Json})
        t.add_computed_column(boom=fail_on_neg(t.name.len()), stored=False)
        t.insert([{'name': 'b', 'flag': True, 'meta': {}}, {'name': 'a', 'flag': False, 'meta': {}}])

        result = bridge.get_table_data('s/t')
        sorted_by_name = {c['name']: c['is_sorted'] for c in result['columns']}
        assert sorted_by_name == {'name': True, 'flag': False, 'meta': False, 'boom': False}

        # Sort by an indexed column works.
        sorted_asc = bridge.get_table_data('s/t', order_by='name')
        assert [r['name'] for r in sorted_asc['rows']] == ['a', 'b']

        # Sort by non-indexed columns is a silent no-op (does not raise, does not reorder).
        unsorted = bridge.get_table_data('s/t', order_by='flag', order_desc=True)
        assert [r['name'] for r in unsorted['rows']] == [r['name'] for r in result['rows']]

    def test_table_data_iterator_view(self, uses_db: None) -> None:
        video_path = get_test_video_files()[0]
        pxt.create_dir('iv')
        video_t = pxt.create_table('iv/videos', {'video': pxt.Video})
        video_t.insert([{'video': video_path}])
        pxt.create_view('iv/frames', video_t, iterator=frame_iterator(video_t.video, fps=1))

        result = bridge.get_table_data('iv/frames')
        storage_by_name = {c['name']: c['is_stored'] for c in result['columns']}
        assert storage_by_name == {'pos': False, 'frame': False, 'frame_attrs': True, 'video': True}
        assert all(set(row.keys()) == {'frame_attrs', 'video'} for row in result['rows'])

        # Sort by an unstored column is a silent no-op (does not raise, does not reorder).
        sorted_result = bridge.get_table_data('iv/frames', order_by='frame', order_desc=True)
        assert [r['frame_attrs'] for r in sorted_result['rows']] == [r['frame_attrs'] for r in result['rows']]

        pipeline = bridge.get_pipeline()
        view_node = next(n for n in pipeline['nodes'] if n['path'] == 'iv/frames')
        assert 'error' not in view_node
        assert view_node['is_view'] is True
