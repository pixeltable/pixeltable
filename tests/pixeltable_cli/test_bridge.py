"""Tests for pixeltable_cli.server.bridge - the translation layer between Pixeltable APIs and the dashboard REST API."""

import pathlib
import sys
from textwrap import dedent

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.catalog.model import schema as schema_verbs
from pixeltable.config import Config
from pixeltable.func import Function
from pixeltable.functions.video import frame_iterator
from pixeltable.utils import app_module
from pixeltable.utils.app_module import load_app_module
from pixeltable_cli.server import bridge
from pixeltable_cli.utils import PxtPath

from ..utils import dummy_embedding, get_test_video_files, pxt_raises

pytestmark = pytest.mark.db_roots('local', reason='pxt CLI metadata/data bridge')


@pxt.udf
def my_udf(x: int) -> int:
    return x + 1


@pxt.udf
def fail_on_neg(x: int) -> int:
    if x < 0:
        raise ValueError('negative')
    return x


class TestBridge:
    def test_app_module_non_identifier_name(self, project_env: pathlib.Path) -> None:
        """A file or directory whose name is not a module name is reported, and the message names it."""
        app_file = project_env / '2024 pipeline.py'
        app_file.write_text('import pixeltable as pxt\n\n@pxt.udf\ndef shout(s: str) -> str:\n    return s.upper()\n')
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r"'2024 pipeline' is not a module name"):
            load_app_module(str(app_file), subject='application file')

        (project_env / 'ad gen').mkdir()
        nested = project_env / 'ad gen' / 'app.py'
        nested.write_text('import pixeltable as pxt\n')
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r"'ad gen' is not a module name"):
            load_app_module(str(nested), subject='application file')

        # a udf in a file that is named after the module holding it resolves from its stored reference
        named = project_env / 'pipeline.py'
        named.write_text('import pixeltable as pxt\n\n@pxt.udf\ndef shout(s: str) -> str:\n    return s.upper()\n')
        module = load_app_module(str(named), subject='application file')
        assert Function.from_dict(module.shout.as_dict()) is module.shout

    def test_app_module_outside_project(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loading a file outside the served project is refused, as is loading one when no project is served."""
        served = tmp_path / 'served'
        served.mkdir()
        (served / 'pixeltable.toml').write_text('', encoding='utf-8')
        app_file = tmp_path / 'outside_app.py'
        app_file.write_text('import pixeltable as pxt\n', encoding='utf-8')

        Config.init(reinit=True, project_root=served)
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r'which the file does not sit under') as exc_info:
            load_app_module(str(app_file), subject='application file')
        assert str(served) in exc_info.value.message

        Config.init(reinit=True, project_root=None)
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match=r'there is no project root') as exc_info:
            load_app_module(str(app_file), subject='application file')
        message = exc_info.value.message
        assert 'pixeltable.toml' in message
        assert '[tool.pixeltable]' in message
        assert 'pxt init' in message

    def test_app_module_neighbor_imports(self, project_env: pathlib.Path) -> None:
        """An application file imports the modules of its project by name, in every spelling."""
        (project_env / 'pkg').mkdir()
        (project_env / 'pkg' / 'inner.py').write_text("VALUE = 'first'\n", encoding='utf-8')
        (project_env / 'helpers.py').write_text("TAG = 'first'\n", encoding='utf-8')
        app_file = project_env / 'neighbors_app.py'
        app_file.write_text('import helpers\nfrom pkg.inner import VALUE\nfrom pkg import inner\n', encoding='utf-8')

        module = load_app_module(str(app_file), subject='application file')
        assert (module.helpers.TAG, module.VALUE) == ('first', 'first')
        # 'from pkg import inner' names a module of the package, which the import statement gets as an attribute
        assert module.inner.VALUE == 'first'

    def test_app_module_loading(self, project_env: pathlib.Path) -> None:
        """Loading an application file twice yields one module, imported by the name its project gives it."""
        (project_env / 'ad_gen').mkdir()
        app_file = project_env / 'ad_gen' / 'app.py'
        app_file.write_text("import pixeltable as pxt\n\nVALUE = 'loaded'\n", encoding='utf-8')

        module = load_app_module(str(app_file), subject='application file')
        assert module.VALUE == 'loaded'
        # the directory holding it is a package of the project, and the project root is what imports resolve from
        assert module.__name__ == 'ad_gen.app'
        assert sys.modules['ad_gen.app'] is module
        assert str(project_env) in sys.path

    def test_edited_neighbor_reread(self, project_env: pathlib.Path) -> None:
        """Loading an application file again reads the modules it imports as they now stand."""
        helpers = project_env / 'neighbor_helpers.py'
        helpers.write_text("SUFFIX = 'first'\n", encoding='utf-8')
        app_file = project_env / 'neighbor_app.py'
        app_file.write_text('from neighbor_helpers import SUFFIX\n\nTAG = SUFFIX\n', encoding='utf-8')
        assert load_app_module(str(app_file), subject='application file').TAG == 'first'

        # only the imported module changed; a load that discarded just the entry module would miss this
        helpers.write_text("SUFFIX = 'second'\n", encoding='utf-8')
        assert load_app_module(str(app_file), subject='application file').TAG == 'second'

        # the packages this process runs stay loaded, even with the project holding a checkout of them
        assert sys.modules.get('pixeltable') is pxt

    def test_prohibited_write_names_statement(self, uses_db: None, project_env: pathlib.Path) -> None:
        """The refusal names the line and the statement that wrote to the catalog."""
        direct = project_env / 'direct_write.py'
        direct.write_text(
            "import pixeltable as pxt\n\npxt.create_table('written_directly', {'c': pxt.Int})\n", encoding='utf-8'
        )
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION, match=r'line 3: ') as exc_info:
            load_app_module(str(direct), subject='schema file')
        assert "create_table('written_directly'" in str(exc_info.value)

        # the write happens in a helper, so the named statement is the call
        (project_env / 'writer.py').write_text(
            "import pixeltable as pxt\n\n\ndef make() -> None:\n    pxt.create_table('t2', {'c': pxt.Int})\n",
            encoding='utf-8',
        )
        indirect = project_env / 'indirect_write.py'
        indirect.write_text('from writer import make\n\nmake()\n', encoding='utf-8')
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION, match=r'line 3: make\(\)') as exc_info:
            load_app_module(str(indirect), subject='schema file')

    def test_shadowed_project_modules(self, project_env: pathlib.Path) -> None:
        """A name that collides with an installed distribution is reported; nothing else is."""
        # a module and a package whose names psutil claims
        (project_env / 'psutil.py').write_text('TAG = 1\n', encoding='utf-8')
        (project_env / 'shutil').mkdir()
        (project_env / 'shutil' / 'inner.py').write_text('TAG = 2\n', encoding='utf-8')

        # a name no import reaches elsewhere, and four that are not modules at all
        (project_env / 'ad_gen.py').write_text('TAG = 3\n', encoding='utf-8')
        (project_env / 'psutil.txt').write_text('not a module\n', encoding='utf-8')
        (project_env / 'data').mkdir()  # a directory holding no Python
        (project_env / 'data' / 'rows.csv').write_text('a,b\n', encoding='utf-8')
        (project_env / 'class').mkdir()  # a keyword, so no module path can hold it
        (project_env / 'class' / 'app.py').write_text('TAG = 4\n', encoding='utf-8')

        reported = app_module.shadowed_project_modules()
        assert sorted(w.split(':')[0] for w in reported) == ['psutil.py', 'shutil'], reported
        assert "an import of 'psutil' reads" in next(w for w in reported if w.startswith('psutil.py'))

    def test_unresolvable_udf_reference(self, uses_db: None, project_env: pathlib.Path) -> None:
        """An unresolvable udf reference is reported as an error."""
        (project_env / 'functions.py').write_text(
            dedent(
                """
                import pixeltable as pxt

                @pxt.udf
                def tag(s: str) -> str:
                    return f'{s}!'
                """
            ),
            encoding='utf-8',
        )
        schema_file = project_env / 'app.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                from functions import tag

                TableModel = pxt.model_base()


                class Docs(TableModel, name='docs'):
                    title: pxt.String
                    tagged = tag(title)  # noqa: F821
                """
            ),
            encoding='utf-8',
        )
        module = load_app_module(str(schema_file), subject='schema file')
        bases = app_module.get_model_bases(module)
        assert app_module.check_udf_references(bases) == []

        # the reference names a module this process no longer holds
        sys.modules.pop('functions', None)
        errors = app_module.check_udf_references(bases)
        assert len(errors) == 1, errors
        assert 'functions.tag' in errors[0]

    def test_app_module_catalog_write_refused(self, uses_db: None, project_env: pathlib.Path) -> None:
        """A mutation while an application file loads is refused; a read goes through."""
        t = pxt.create_table('frozen', {'c': pxt.Int})
        t.insert([{'c': 1}])
        refused = r'this application file modifies the catalog while it is imported'

        # DDL
        ddl_file = project_env / 'ddl.py'
        ddl_file.write_text(
            dedent(
                """
                import pixeltable as pxt

                pxt.create_table('made_by_import', {'c': pxt.Int})
                """
            ),
            encoding='utf-8',
        )
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION, match=refused):
            load_app_module(str(ddl_file), subject='application file')
        assert 'made_by_import' not in pxt.list_tables()

        # DML
        dml_file = project_env / 'dml.py'
        dml_file.write_text(
            dedent(
                """
                import pixeltable as pxt

                pxt.get_table('frozen').insert([{'c': 2}])
                """
            ),
            encoding='utf-8',
        )
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION, match=refused):
            load_app_module(str(dml_file), subject='application file')
        assert t.count() == 1

        # reading the catalog while loading is allowed
        read_file = project_env / 'read.py'
        read_file.write_text(
            dedent(
                """
                import pixeltable as pxt

                NUM_ROWS = pxt.get_table('frozen').count()
                """
            ),
            encoding='utf-8',
        )
        assert load_app_module(str(read_file), subject='application file').NUM_ROWS == 1

        # the refusal is scoped to the load: the next mutation goes through
        t.insert([{'c': 3}])
        assert t.count() == 2

    def test_table_metadata_basic(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String | None, 'c2': pxt.Int}, primary_key='c2')
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
        t = pxt.create_table('md/base', {'c1': pxt.String | None})
        pxt.create_view('md/v', t)
        result = pxt.get_table('md/v').get_metadata()
        assert (result['is_view'], result['base']) == (True, 'md/base')

    def test_table_metadata_indices(self, uses_db: None) -> None:
        pxt.create_dir('md')
        t = pxt.create_table('md/t', {'c1': pxt.String | None})
        t.add_embedding_index('c1', embedding=dummy_embedding.using(n=3))
        result = pxt.get_table('md/t').get_metadata()
        assert len(result['indexes']) > 0
        idx = next(iter(result['indexes'].values()))
        assert {'name', 'columns', 'index_type'} <= idx.keys()

    def test_table_data_basic(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String | None, 'c2': pxt.Int | None})
        t.insert([{'c1': 'hello', 'c2': 1}, {'c1': 'world', 'c2': 2}])

        result = bridge.get_table_data('td/t')
        assert (result['total_count'], len(result['rows'])) == (2, 2)
        assert {c['name'] for c in result['columns']} == {'c1', 'c2'}
        assert all({'type', 'is_media', 'is_computed', 'is_stored'} <= c.keys() for c in result['columns'])

    def test_table_data_empty(self, uses_db: None) -> None:
        pxt.create_dir('td')
        pxt.create_table('td/t', {'c1': pxt.String | None})
        result = bridge.get_table_data('td/t')
        assert result['rows'] == []
        assert result['total_count'] == 0

    def test_table_data_pagination(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Int | None})
        t.insert([{'c1': i} for i in range(10)])
        page1 = bridge.get_table_data('td/t', offset=0, limit=3)
        assert len(page1['rows']) == 3
        assert page1['total_count'] == 10
        page2 = bridge.get_table_data('td/t', offset=5, limit=3)
        assert len(page2['rows']) == 3

    def test_table_data_order_by(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Int | None}, has_default_idxs=True)
        t.insert([{'c1': 3}, {'c1': 1}, {'c1': 2}])
        asc = bridge.get_table_data('td/t', order_by='c1', order_desc=False)
        assert [r['c1'] for r in asc['rows']] == [1, 2, 3]
        desc = bridge.get_table_data('td/t', order_by='c1', order_desc=True)
        assert [r['c1'] for r in desc['rows']] == [3, 2, 1]

    def test_table_data_computed_column(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String | None})
        t.add_computed_column(upper=t.c1.upper())
        t.insert([{'c1': 'hello'}])
        result = bridge.get_table_data('td/t')
        assert result['rows'][0]['upper'] == 'HELLO'

    def test_table_data_nulls(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.String | None, 'c2': pxt.Int | None})
        t.insert([{'c1': None, 'c2': None}])
        assert bridge.get_table_data('td/t')['rows'][0] == {'c1': None, 'c2': None}

    def test_table_data_json(self, uses_db: None) -> None:
        pxt.create_dir('td')
        t = pxt.create_table('td/t', {'c1': pxt.Json | None})
        t.insert([{'c1': {'key': 'value', 'num': 42}}])
        row = bridge.get_table_data('td/t')['rows'][0]
        assert row['c1'] == {'key': 'value', 'num': 42}

    def test_export_csv(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.String | None, 'c2': pxt.Int | None})
        t.insert([{'c1': 'hello', 'c2': 1}, {'c1': 'world', 'c2': 2}])
        csv_str = bridge.export_table_csv('ex/t').decode('utf-8')
        lines = csv_str.strip().split('\n')
        assert len(lines) == 3  # header + 2 rows
        assert 'c1' in lines[0]
        assert 'c2' in lines[0]

    def test_export_csv_empty(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        pxt.create_table('ex/t', {'c1': pxt.String | None})
        lines = bridge.export_table_csv('ex/t').decode('utf-8').strip().split('\n')
        assert len(lines) == 1  # header only

    def test_export_csv_limit(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.Int | None})
        t.insert([{'c1': i} for i in range(10)])
        lines = bridge.export_table_csv('ex/t', limit=3).decode('utf-8').strip().split('\n')
        assert len(lines) == 4  # header + 3 rows

    def test_export_csv_json_column(self, uses_db: None) -> None:
        pxt.create_dir('ex')
        t = pxt.create_table('ex/t', {'c1': pxt.Json | None})
        t.insert([{'c1': {'key': 'val'}}])
        csv_str = bridge.export_table_csv('ex/t').decode('utf-8')
        assert 'key' in csv_str

    def test_search_empty_db(self, uses_db: None) -> None:
        assert bridge.search('anything') == {
            'query': 'anything',
            'directories': [],
            'tables': [],
            'columns': [],
            'unavailable': [],
        }

    def test_search_unreachable_catalog(self, uses_db: None) -> None:
        pxt.create_table('users', {'email': pxt.String | None})

        result = bridge.search('users', additional_db_uris=['pxt://nosuch:db'])
        assert [t['path'] for t in result['tables']] == ['users']
        assert [(u['path'], u['kind']) for u in result['unavailable']] == [('pxt://nosuch:db', 'catalog')]
        assert result['unavailable'][0]['error'] != ''

    def test_search_unreadable_table(self, uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A table that can't be opened is reported, not passed off as a result with made-up metadata."""
        pxt.create_table('readable', {'c1': pxt.String | None})
        pxt.create_table('broken', {'c1': pxt.String | None})

        get_table = pxt.get_table

        def get_table_or_fail(path: str) -> pxt.Table:
            if path == 'broken':
                raise RuntimeError('cannot open')
            return get_table(path)

        monkeypatch.setattr(bridge.pxt, 'get_table', get_table_or_fail)

        result = bridge.search('able')
        assert [t['path'] for t in result['tables']] == ['readable']
        assert [(u['path'], u['kind']) for u in result['unavailable']] == [('broken', 'table')]
        assert 'cannot open' in result['unavailable'][0]['error']

    def test_search_dir_table_column(self, uses_db: None) -> None:
        pxt.create_dir('proj')
        pxt.create_table('proj/users', {'email': pxt.String | None, 'age': pxt.Int | None})

        assert [d['path'] for d in bridge.search('proj')['directories']] == ['proj']
        assert [t['path'] for t in bridge.search('users')['tables']] == ['proj/users']
        col = bridge.search('email')['columns']
        assert [(c['name'], c['table']) for c in col] == [('email', 'proj/users')]

    def test_search_case_insensitive(self, uses_db: None) -> None:
        pxt.create_dir('MyDir')
        assert len(bridge.search('mydir')['directories']) == 1

    def test_search_all_matches(self, uses_db: None) -> None:
        pxt.create_dir('sl')
        for i in range(5):
            pxt.create_table(f'sl/match_{i}', {'c1': pxt.String | None})
        assert len(bridge.search('match')['tables']) == 5

    def test_pipeline(self, uses_db: None) -> None:
        assert bridge.get_pipeline() == {'nodes': [], 'edges': []}

        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String | None})
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
        root = pxt.create_table('sc/root', {'c': pxt.String | None})
        mid = pxt.create_view('sc/mid', root)
        pxt.create_view('sc/leaf', mid)
        pxt.create_table('sc/other', {'c': pxt.String | None})

        # Scoped to mid: includes ancestor (root), self (mid), descendant (leaf); excludes 'other'.
        result = bridge.get_pipeline(tbl_path='sc/mid')
        assert {n['path'] for n in result['nodes']} == {'sc/root', 'sc/mid', 'sc/leaf'}
        assert {(e['source'], e['target']) for e in result['edges']} == {('sc/root', 'sc/mid'), ('sc/mid', 'sc/leaf')}

        # Unknown path returns empty.
        assert bridge.get_pipeline(tbl_path='sc/missing') == {'nodes': [], 'edges': []}

        # No path returns the full catalog.
        assert {n['path'] for n in bridge.get_pipeline()['nodes']} == {'sc/root', 'sc/mid', 'sc/leaf', 'sc/other'}
        # Catalog-root aliases are the same full local DAG.
        assert {n['path'] for n in bridge.get_pipeline('')['nodes']} == {'sc/root', 'sc/mid', 'sc/leaf', 'sc/other'}
        assert {n['path'] for n in bridge.get_pipeline('local')['nodes']} == {
            'sc/root',
            'sc/mid',
            'sc/leaf',
            'sc/other',
        }

    def test_pipeline_view_edge(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/base', {'c1': pxt.String | None})
        pxt.create_view('pp/v', t)
        result = bridge.get_pipeline()
        assert len(result['nodes']) == 2
        assert [(e['source'], e['target'], e['type']) for e in result['edges']] == [('pp/base', 'pp/v', 'view')]

    def test_pipeline_computed_columns(self, uses_db: None) -> None:
        pxt.create_dir('pp')
        t = pxt.create_table('pp/t', {'c1': pxt.String | None, 'c2': pxt.Int | None})
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
        video_t = pxt.create_table('videos', {'video': pxt.Video | None})
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
        pxt.create_table('st/t1', {'c1': pxt.String | None})
        pxt.create_table('st/t2', {'c1': pxt.String | None})
        assert bridge.get_status()['total_tables'] == 2

    def test_table_data_unstored_column(self, uses_db: None) -> None:
        # Unstored computed columns must not be evaluated by the data view.
        # The expression here would raise on negative inputs; the test asserts that
        # get_table_data succeeds anyway and reports the column as is_stored=False.
        t = pxt.create_table('udata', {'x': pxt.Int | None})
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
        t = pxt.create_table(
            's/t', {'name': pxt.String | None, 'flag': pxt.Bool | None, 'meta': pxt.Json | None}, has_default_idxs=True
        )
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
        video_t = pxt.create_table('iv/videos', {'video': pxt.Video | None})
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

    def test_schema_update_destructive_refusal(self, uses_db: None, project_env: pathlib.Path) -> None:
        schema_src = dedent(
            """
            from __future__ import annotations

            import pixeltable as pxt

            TableModel = pxt.model_base()


            class Docs(TableModel, name='docs'):
                title: pxt.String
                body: pxt.String
            """
        )
        schema_file = project_env / 'refusal_schema.py'
        schema_file.write_text(schema_src)
        target = PxtPath('refusal')
        schema_verbs.schema_update(str(schema_file), target)

        # the edited schema goes into a module of its own: a process reads a file once, and picks up an edit
        # by starting again
        dropped_file = project_env / 'refusal_schema_v2.py'
        dropped_file.write_text(schema_src.replace('    body: pxt.String\n', ''))

        # dropping a column destroys its data; the refusal tells a CLI user about the flag, not about update_all()
        with pxt_raises(excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE, match='--allow-destructive') as info:
            schema_verbs.schema_update(str(dropped_file), target)
        assert 'update_all()' not in info.value.message
        assert 'body' in pxt.get_table('refusal/docs').columns()
