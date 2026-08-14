"""Tests for 'pxt schema diff', 'pxt schema update' and 'pxt schema prune'."""

import pathlib
from collections.abc import Callable
from textwrap import dedent

import pixeltable as pxt

from ..utils import skip_test_if_not_installed
from .conftest import PxtRunner

SCHEMA_SRC = dedent(
    """
    from __future__ import annotations

    import pixeltable as pxt
    import pixeltable.functions as pxtf

    TableModel = pxt.model_base()


    class Docs(TableModel, name='docs'):
        title: pxt.String
        body: pxt.String | None
        title_upper = pxtf.string.upper(title)


    class TitledDocs(TableModel, name='titled_docs', base=Docs.where(Docs.title != '')):
        headline = Docs.title_upper + '!'
    """
)


class TestSchema:
    def test_update(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('app')

        # create; the tables exist and compute afterwards
        r = cli('schema', 'update', str(schema_file), target)
        assert r.stdout.count('created') == 2
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'title': 'hello', 'body': 'world'}, {'title': '', 'body': 'no title'}])
        titled = pxt.get_table(f'{target}/titled_docs')
        assert titled.select(titled.headline).collect()[0]['headline'] == 'HELLO!'

        # idempotent rerun: exit 0, nothing applied
        r = cli('schema', 'update', str(schema_file), target)
        assert r.returncode == 0
        assert r.stdout.count('created') == 0
        assert 'catalog is up to date' in r.stdout

        # json output: the plan that was applied, which is now empty of changes
        r = cli('schema', 'update', str(schema_file), target, '--json')
        assert r.json['in_agreement']
        assert [t['resolution'] for t in r.json['tables']] == ['up_to_date', 'up_to_date']

        # a model whose kind conflicts with the existing object (table vs view) is an error
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class TitledDocs(TableModel, name='titled_docs'):
                    headline: pxt.String | None
                """
            )
        )
        r = cli('schema', 'update', str(schema_file), target, check=False)
        assert r.returncode == 1
        assert "specifies a table, but 'titled_docs' is a view" in r.stderr
        # the way forward is named in the terms of the surface the caller is using
        assert 'update_all()' not in r.stderr
        assert 'pxt.move()' not in r.stderr

    def test_update_migrates(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('migrate')
        cli('schema', 'update', str(schema_file), target)
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'title': 'hello', 'body': 'world'}])

        # an added column is safe, so it needs no flag, and the existing rows survive
        schema_file.write_text(
            SCHEMA_SRC.replace('body: pxt.String | None', 'body: pxt.String | None\n    author: pxt.String | None')
        )
        r = cli('schema', 'update', str(schema_file), target)
        assert r.returncode == 0
        assert 'updated' in r.stdout
        docs = pxt.get_table(f'{target}/docs')
        assert 'author' in docs.columns()
        assert docs.select(docs.title).collect()['title'] == ['hello']

        r = cli('schema', 'diff', str(schema_file), target)
        assert r.returncode == 0

    def test_update_destructive(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('destructive')
        cli('schema', 'update', str(schema_file), target)

        # dropping a column destroys its data, so it is refused without --allow-destructive
        schema_file.write_text(SCHEMA_SRC.replace('    body: pxt.String | None\n', ''))
        r = cli('schema', 'update', str(schema_file), target, check=False)
        assert r.returncode == 3
        assert 'refusing to apply 1 destructive operation(s)' in r.stderr
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        # -n reports the same plan without applying it
        r = cli('schema', 'update', str(schema_file), target, '-n', check=False)
        assert r.returncode == 2
        assert "column 'body' will be dropped  DESTRUCTIVE" in r.stdout
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        # the refusal is machine-readable too: the plan comes back with the destructive ops marked refused
        r = cli('schema', 'update', str(schema_file), target, '--json', check=False)
        assert r.returncode == 3
        docs = next(t for t in r.json['tables'] if t['path'] == f'{target}/docs')
        assert docs['status'] == 'refused'
        assert [op['status'] for op in docs['ops']] == ['refused']

        r = cli('schema', 'update', str(schema_file), target, '-n', '--json', check=False)
        assert r.returncode == 2
        assert [op['status'] for t in r.json['tables'] for op in t['ops']] == ['skipped']

        # -n applies nothing, even with the drop permitted and confirmed
        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f', '-n', check=False)
        assert r.returncode == 2
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f', '--json')
        assert r.returncode == 0
        docs = next(t for t in r.json['tables'] if t['path'] == f'{target}/docs')
        assert docs['status'] == 'applied'
        assert [op['status'] for op in docs['ops']] == ['applied']
        assert 'body' not in pxt.get_table(f'{target}/docs').columns()

        # rerunning with the same flags reports the same nothing-to-do as an unflagged rerun does
        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f')
        assert r.returncode == 0
        assert 'catalog is up to date' in r.stdout

    def test_diff(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('app')

        # nothing exists yet: every table is a create, and the target itself is left uncreated
        r = cli('schema', 'diff', str(schema_file), target, '--json', check=False)
        assert r.returncode == 2
        assert not r.json['in_agreement']
        assert [(t['path'], t['resolution']) for t in r.json['tables']] == [
            (f'{target}/docs', 'create'),
            (f'{target}/titled_docs', 'create'),
        ]
        # a create subsumes the additions that constitute it
        assert all(t['ops'] == [] for t in r.json['tables'])
        assert r.json['summary'] == {
            'up_to_date': 0,
            'create': 2,
            'update_additive': 0,
            'update_destructive': 0,
            'unsupported': 0,
            'extras': 0,
            'destructive': 0,
        }
        assert target not in pxt.list_dirs(recursive=True)

        cli('schema', 'update', str(schema_file), target)

        # in agreement afterwards
        r = cli('schema', 'diff', str(schema_file), target, '--json')
        assert r.returncode == 0
        assert r.json['in_agreement']
        assert [t['resolution'] for t in r.json['tables']] == ['up_to_date', 'up_to_date']
        assert r.json['summary']['up_to_date'] == 2

        # human-readable form
        r = cli('schema', 'diff', str(schema_file), target)
        assert f'= {target}/docs' in r.stdout
        assert 'Plan: 0 create, 0 update, 2 unchanged, 0 extra  |  0 destructive' in r.stdout

    def test_diff_drift(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('drift')
        cli('schema', 'update', str(schema_file), target)

        # add a column to the model and drop another: one safe op, one destructive
        schema_file.write_text(SCHEMA_SRC.replace('body: pxt.String | None', 'author: pxt.String | None'))
        r = cli('schema', 'diff', str(schema_file), target, '--json', check=False)
        assert r.returncode == 2
        docs = next(t for t in r.json['tables'] if t['path'] == f'{target}/docs')
        assert docs['resolution'] == 'update_destructive'
        assert docs['destructive']
        assert [(op['op'], op['target'], op['name'], op['destructive']) for op in docs['ops']] == [
            ('add', 'column', 'author', False),
            ('drop', 'column', 'body', True),
        ]
        # the added column carries its type, so the plan is actionable without re-reading the schema
        assert next(op for op in docs['ops'] if op['op'] == 'add')['details']['type'] == 'String | None'
        assert r.json['summary']['destructive'] == 1

        r = cli('schema', 'diff', str(schema_file), target, check=False)
        assert "column 'author' will be added  safe" in r.stdout
        assert "column 'body' will be dropped  DESTRUCTIVE" in r.stdout

    def test_diff_extras(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('extras')
        cli('schema', 'update', str(schema_file), target)
        pxt.create_table(f'{target}/scratch', {'x': pxt.Int | None})

        # a table no model declares is reported, but update would not touch it, so the target is still in agreement
        r = cli('schema', 'diff', str(schema_file), target, '--json')
        assert r.returncode == 0
        assert r.json['in_agreement']
        assert r.json['extras'] == [f'{target}/scratch']
        assert r.json['summary']['extras'] == 1

        r = cli('schema', 'diff', str(schema_file), target)
        assert f'! {target}/scratch' in r.stdout
        assert 'extra (not in schema)' in r.stdout

    def test_prune(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('prune')
        cli('schema', 'update', str(schema_file), target)

        # nothing undeclared yet
        r = cli('schema', 'prune', str(schema_file), target, '-f')
        assert r.returncode == 0
        assert 'nothing to prune' in r.stdout

        # a view over an undeclared table, so the drops have to be ordered base-last
        pxt.create_table(f'{target}/scratch', {'x': pxt.Int | None})
        scratch = pxt.get_table(f'{target}/scratch')
        pxt.create_view(f'{target}/scratch_view', scratch.where(scratch.x > 0))

        # -n lists the drops without performing them
        r = cli('schema', 'prune', str(schema_file), target, '-n', check=False)
        assert r.returncode == 2
        assert 'would drop' in r.stdout
        assert pxt.get_table(f'{target}/scratch') is not None

        r = cli('schema', 'prune', str(schema_file), target, '-n', '--json', check=False)
        assert r.returncode == 2
        assert {op['status'] for op in r.json['ops']} == {'skipped'}

        # without -f there is no terminal to confirm at, so the drops are refused
        r = cli('schema', 'prune', str(schema_file), target, check=False)
        assert r.returncode == 3
        assert pxt.get_table(f'{target}/scratch') is not None

        # a refusal is still reported in full under --json, so it needs no output parsing
        r = cli('schema', 'prune', str(schema_file), target, '--json', check=False)
        assert r.returncode == 3
        assert {op['status'] for op in r.json['ops']} == {'refused'}

        r = cli('schema', 'prune', str(schema_file), target, '-f', '--json')
        assert sorted(op['name'] for op in r.json['ops']) == [f'{target}/scratch', f'{target}/scratch_view']
        assert {(op['target'], op['op']) for op in r.json['ops']} == {('table', 'drop')}
        assert {op['status'] for op in r.json['ops']} == {'applied'}
        assert sorted(pxt.list_tables(target)) == [f'{target}/docs', f'{target}/titled_docs']

        # the declared tables are untouched, so the schema and the target still agree
        r = cli('schema', 'diff', str(schema_file), target)
        assert r.returncode == 0

    def test_prune_keeps_tables_with_declared_dependents(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        target = p('keep')

        # 'derived' is declared as a view of 'raw', but only 'derived' is in the schema, so 'raw' reads as an extra
        pxt.create_dir(target)
        raw = pxt.create_table(f'{target}/raw', {'id': pxt.Int | None})
        pxt.create_view(f'{target}/derived', raw.where(raw.id > 0))
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class Derived(TableModel, name='derived'):
                    id: pxt.Int
                """
            )
        )

        r = cli('schema', 'prune', str(schema_file), target, '-f', check=False)
        assert r.returncode == 1
        assert "the following depend on it: 'keep/derived'" in r.stderr
        assert pxt.get_table(f'{target}/raw') is not None

    def test_prune_reports_tables_dropped_before_the_failure(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        target = p('partial')

        # 'gone' prunes cleanly; 'raw' cannot, because the schema keeps the view that depends on it
        pxt.create_dir(target)
        pxt.create_table(f'{target}/gone', {'id': pxt.Int | None})
        raw = pxt.create_table(f'{target}/raw', {'id': pxt.Int | None})
        pxt.create_view(f'{target}/derived', raw.where(raw.id > 0))
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class Derived(TableModel, name='derived'):
                    id: pxt.Int
                """
            )
        )

        r = cli('schema', 'prune', str(schema_file), target, '-f', check=False)
        assert r.returncode == 1
        # the proxy reports fully-qualified paths, the local catalog relative ones
        assert 'The following table(s) were already dropped:' in r.stderr
        assert 'partial/gone' in r.stderr
        assert pxt.get_table(f'{target}/raw') is not None
        assert f'{target}/gone' not in pxt.list_tables(target)

    def test_example(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('sentence_transformers')
        p = make_catalog_path
        target = p('documented')

        # the file the command emits has to be one that actually works
        schema_file = tmp_path / 'example.py'
        schema_file.write_text(cli('schema', 'example', '--brief').stdout)

        r = cli('schema', 'update', str(schema_file), target)
        assert r.stdout.count('created') == 2
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'title': 'hello', 'body': 'world'}, {'title': '', 'body': 'untitled'}])
        titled = pxt.get_table(f'{target}/titled')
        assert titled.select(titled.headline).collect()['headline'] == ['HELLO!']
        assert cli('schema', 'diff', str(schema_file), target).returncode == 0

        # --out writes the same bytes to a file, for either form
        out_file = tmp_path / 'out.py'
        r = cli('schema', 'example', '--brief', '--out', str(out_file))
        assert f'wrote {out_file}' in r.stdout
        assert out_file.read_text() == schema_file.read_text()
        cli('schema', 'example', '--out', str(out_file))
        assert out_file.read_text() == cli('schema', 'example').stdout

        # the full example is the default, and it declares every construct the DSL supports
        full = out_file.read_text()
        assert all(
            construct in full
            for construct in ('pxt.Column(', 'pxt.EmbeddingIndex(', 'iterator=', 'base=', 'pxt.Document')
        )
        # it has to be a file the daemon can import and plan, media types and embedding index included
        full_target = p('full_example')
        r = cli('schema', 'diff', str(out_file), full_target, check=False)
        assert r.returncode == 2, r.stderr  # 2 = changes pending, ie the plan was computed

        # and applying it has to produce working tables, the embedding index included
        r = cli('schema', 'update', str(out_file), full_target)
        assert r.stdout.count('created') == 4
        assert cli('schema', 'diff', str(out_file), full_target).returncode == 0  # nothing left to apply
        docs = pxt.get_table(f'{full_target}/docs')
        docs.insert(
            [
                {'doc_id': 1, 'title': 'bread', 'body': 'Sourdough needs a long, slow fermentation.'},
                {'doc_id': 2, 'title': 'sharks', 'body': 'Great white sharks hunt seals along the coast.'},
                {
                    'doc_id': 3,
                    'title': 'sharks',
                    'body': 'A simple and effective breathing exercise to reduce stress is box breathing',
                },
            ]
        )
        # verify embeddings by running a similarity search
        sim = docs.body.similarity(string='sharks hunting seals near the shore')
        assert docs.order_by(sim, asc=False).select(docs.doc_id).limit(1).collect()['doc_id'] == [2]

        # the file is reachable from wherever an agent lands: the verb list, and every verb's help
        assert 'example' in cli('schema', check=False).stdout
        for verb in ('diff', 'update', 'prune', 'example'):
            assert "'pxt schema example'" in cli('schema', verb, '--help').stdout

    def test_diff_unsupported(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('unsupported')
        cli('schema', 'update', str(schema_file), target)

        # a model declaring a table where a view exists cannot be migrated in place
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class TitledDocs(TableModel, name='titled_docs'):
                    headline: pxt.String | None
                """
            )
        )
        r = cli('schema', 'diff', str(schema_file), target, '--json', check=False)
        assert r.returncode == 2
        tbl = r.json['tables'][0]
        assert tbl['resolution'] == 'unsupported'
        assert all(op['severity'] == 'unsupported' for op in tbl['ops'])
        assert [(op['op'], op['target'], op['name']) for op in tbl['ops']] == [
            ('alter', 'table', 'kind'),
            ('alter', 'table', 'view_filter'),
            ('alter', 'column', 'headline'),
        ]
        assert r.json['summary']['unsupported'] == 1

    def test_update_errors(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)

        # unknown verb
        r = cli('schema', 'doesnotexist', str(schema_file), p('app'), check=False)
        assert r.returncode == 1
        assert 'unknown verb' in r.stderr

        # a usage error is an error, not the exit 2 that means 'changes are pending'
        r = cli('schema', 'diff', str(schema_file), p('app'), '--bogus', check=False)
        assert r.returncode == 1
        assert 'unrecognized arguments' in r.stderr

        # missing schema file
        r = cli('schema', 'update', str(tmp_path / 'nonexistent.py'), p('app'), check=False)
        assert r.returncode == 1
        assert 'not found' in r.stderr

        # schema file without a model base
        no_base = tmp_path / 'no_base.py'
        no_base.write_text('import pixeltable as pxt\n')
        r = cli('schema', 'update', str(no_base), p('app'), check=False)
        assert r.returncode == 1
        assert 'no model_base()' in r.stderr

        # schema file that fails to load
        broken = tmp_path / 'broken.py'
        broken.write_text('raise RuntimeError("boom")\n')
        r = cli('schema', 'update', str(broken), p('app'), check=False)
        assert r.returncode == 1
        assert 'error loading' in r.stderr

    def test_update_relative_path(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        (tmp_path / 'app_schema.py').write_text(SCHEMA_SRC)
        target = p('rel_app')

        # a relative schema path is resolved against the client's cwd, so run the command from that directory
        r = cli('schema', 'update', 'app_schema.py', target, cwd=tmp_path)
        assert r.stdout.count('created') == 2
        assert pxt.get_table(f'{target}/docs') is not None
