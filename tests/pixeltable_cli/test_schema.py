"""Tests for 'pxt schema diff', 'pxt schema update' and 'pxt schema prune'."""

import pathlib
import re
from textwrap import dedent
from typing import Callable

import pytest

import pixeltable as pxt

from ..utils import DatabaseRoot, get_audio_files, get_documents, get_video_files, skip_test_if_not_installed
from .conftest import PxtRunner

# A minimal schema, for the cases that assert on an error message rather than on what Pixeltable can express.
# The breadth -- column types, computed columns, views, iterator views, indexes, media -- comes from the shared
# app corpus in apps/, which test_service.py drives through `pxt service`.
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


def validate_tables(target: str) -> None:
    """Make sure that tables created from models are functional."""
    docs = pxt.get_table(f'{target}/docs')
    docs.insert([{'title': 'alpha', 'body': 'first'}, {'title': '', 'body': 'untitled'}])
    rows = docs.order_by(docs.title).select(docs.title, docs.body, docs.title_upper).collect()
    assert [r['title_upper'] for r in rows] == ['', 'ALPHA']

    # the view holds the rows its predicate admits, plus the column it declares
    titled = pxt.get_table(f'{target}/titled_docs')
    assert titled.select(titled.headline).collect()['headline'] == ['ALPHA!']


def assert_in_agreement(cli: PxtRunner, app: str, target: str, cwd: pathlib.Path | None = None) -> None:
    """Assert that the target holds what the file declares: diff agrees, and no declared table is pending.

    Whatever a command reported about the work it did, this is the reading that says the target converged.
    An undeclared table is not a disagreement, so a target with extras still passes.
    """
    r = cli('schema', 'diff', app, target, '--json', cwd=cwd)
    assert r.returncode == 0, r.stdout
    assert r.json['in_agreement'], r.json
    assert [t['resolution'] for t in r.json['tables']] == ['up_to_date'] * len(r.json['tables']), r.json['tables']


class TestSchema:
    @pytest.mark.db_roots(
        'local',
        'proxy',
        reason='a hosted image holds the project it was built from, and this test writes its udf while running',
    )
    def test_basic(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """The first application of a file: create, use what it created, rerun, and hit a conflict."""
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app.py'
        schema_file.write_text(pathlib.Path(apps('basic.py')).read_text(encoding='utf-8'), encoding='utf-8')
        target = p('app')

        # create; the tables exist and compute afterwards
        r = cli('schema', 'update', str(schema_file), target)
        assert r.stdout.count('created') == 3  # one table and its two views
        docs = pxt.get_table(f'{target}/docs')
        docs.insert(
            [
                {'doc_id': 1, 'title': 'hello', 'body': 'world', 'published': True},
                {'doc_id': 2, 'title': '', 'body': 'no title', 'published': False},
                {'doc_id': 3, 'title': 'a title longer than the limit', 'body': 'long', 'published': False},
            ]
        )
        # computed columns compute, including the unstored one
        row = docs.where(docs.doc_id == 1).select(docs.title_upper, docs.summary, docs.unstored).collect()[0]
        assert row == {'title_upper': 'HELLO', 'summary': 'hello', 'unstored': 'hello'}
        # summary calls a udf the application file defines, which this process resolves from the file alone
        assert docs.where(docs.doc_id == 3).select(docs.summary).collect()['summary'] == ['a title long...']
        # a filtered view holds the rows its predicate admits, a projecting view the columns it selects
        published = pxt.get_table(f'{target}/published')
        assert published.select(published.headline).collect()['headline'] == ['HELLO!']
        titles = pxt.get_table(f'{target}/titles')
        assert set(titles.columns()) == {'doc_id', 't', 'shouted'}
        assert_in_agreement(cli, str(schema_file), target)

        # idempotent rerun: exit 0, nothing applied
        r = cli('schema', 'update', str(schema_file), target)
        assert r.returncode == 0
        assert r.stdout.count('created') == 0
        assert 'catalog is up to date' in r.stdout

        # json output: the plan that was applied, which is now empty of changes
        r = cli('schema', 'update', str(schema_file), target, '--json')
        assert r.json['in_agreement']
        assert [t['resolution'] for t in r.json['tables']] == ['up_to_date'] * 3

        # a model whose kind conflicts with the existing object (table vs view) is an error
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class Published(TableModel, name='published'):
                    headline: pxt.String | None
                """
            )
        )
        r = cli('schema', 'update', str(schema_file), target, check=False)
        assert r.returncode == 1
        assert "specifies a table, but 'published' is a view" in r.stderr
        # the way forward is named in the terms of the surface the caller is using
        assert 'update_all()' not in r.stderr
        assert 'pxt.move()' not in r.stderr

    @pytest.mark.db_roots(
        'local',
        'proxy',
        reason='a hosted image holds the project it was built from, and this test writes its udf while running',
    )
    def test_in_place_edit(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """A second update of a path the daemon already served reads the file as it now stands."""
        p = db_root.make_catalog_path
        target = p('reload')
        declaration = dedent(
            """
            from __future__ import annotations

            import pixeltable as pxt


            @pxt.udf
            def excerpt(text: str) -> str:
                return text[:4]


            TableModel = pxt.model_base()


            class Docs(TableModel, name='docs'):
                title: pxt.String
                summary = excerpt(title)  # noqa: F821
            """
        )
        schema_file = project_dir / 'app.py'
        schema_file.write_text(declaration, encoding='utf-8')
        cli('schema', 'update', str(schema_file), target)
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'title': 'hello world'}])
        assert docs.select(docs.summary).collect()['summary'] == ['hell']

        # the same path, now declaring one more column
        schema_file.write_text(declaration + "    shouted = title + '!'  # noqa: F821\n", encoding='utf-8')
        r = cli('schema', 'update', str(schema_file), target)
        assert 'updated' in r.stdout, r.stdout
        docs = pxt.get_table(f'{target}/docs')
        assert 'shouted' in docs.columns()
        assert docs.select(docs.shouted).collect()['shouted'] == ['hello world!']
        assert_in_agreement(cli, str(schema_file), target)

    def test_evolution(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """Editing the file under data: an added column is applied as it stands, a dropped one is refused."""
        p = db_root.make_catalog_path
        target = p('evolve')
        cli('schema', 'update', apps('basic.py'), target)
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'doc_id': 1, 'title': 'hello', 'body': 'world', 'published': True}])

        # an added column is safe, so it needs no flag, and the existing rows survive
        r = cli('schema', 'update', apps('basic_added_column.py'), target)
        assert r.returncode == 0
        assert 'updated' in r.stdout
        docs = pxt.get_table(f'{target}/docs')
        assert 'author' in docs.columns()
        assert docs.select(docs.title).collect()['title'] == ['hello']
        assert_in_agreement(cli, apps('basic_added_column.py'), target)

        # dropping a column destroys its data, so it is refused without --allow-destructive; this file drops
        # both the column just added and the one holding data
        schema_file = pathlib.Path(apps('basic_dropped_column.py'))
        r = cli('schema', 'update', str(schema_file), target, check=False)
        assert r.returncode == 3
        assert 'refusing to apply 2 destructive operation(s)' in r.stderr
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        # -n reports the same plan without applying it
        r = cli('schema', 'update', str(schema_file), target, '-n', check=False)
        assert r.returncode == 2
        assert "column 'body' will be dropped  DESTRUCTIVE" in r.stdout
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        # the refusal is machine-readable too: the plan comes back with the destructive ops marked refused
        r = cli('schema', 'update', str(schema_file), target, '--json', check=False)
        assert r.returncode == 3
        docs_plan = next(t for t in r.json['tables'] if t['path'] == f'{target}/docs')
        assert docs_plan['status'] == 'refused'
        assert [op['status'] for op in docs_plan['ops']] == ['refused', 'refused']

        r = cli('schema', 'update', str(schema_file), target, '-n', '--json', check=False)
        assert r.returncode == 2
        assert [op['status'] for t in r.json['tables'] for op in t['ops']] == ['skipped', 'skipped']

        # -n applies nothing, even with the drop permitted and confirmed
        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f', '-n', check=False)
        assert r.returncode == 2
        assert 'body' in pxt.get_table(f'{target}/docs').columns()

        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f', '--json')
        assert r.returncode == 0
        docs_plan = next(t for t in r.json['tables'] if t['path'] == f'{target}/docs')
        assert docs_plan['status'] == 'applied'
        assert [op['status'] for op in docs_plan['ops']] == ['applied', 'applied']
        docs = pxt.get_table(f'{target}/docs')
        assert 'body' not in docs.columns() and 'author' not in docs.columns()
        # what the drops did not touch is still there, rows included
        assert docs.select(docs.title, docs.title_upper).collect()[0] == {'title': 'hello', 'title_upper': 'HELLO'}
        assert_in_agreement(cli, str(schema_file), target)

        # rerunning with the same flags reports the same nothing-to-do as an unflagged rerun does
        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f')
        assert r.returncode == 0
        assert 'catalog is up to date' in r.stdout

    def test_diff(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
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

    def test_diff_drift(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('drift')
        cli('schema', 'update', str(schema_file), target)
        validate_tables(target)

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

    def test_diff_replaces_index(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """A named embedding index whose definition changed is replaced."""
        p = db_root.make_catalog_path
        apps('udfs.py')  # the schema below imports its embedding udf
        schema_template = dedent(
            """
            from __future__ import annotations

            import pixeltable as pxt
            from apps.udfs import dummy_embedding
            from pixeltable import EmbeddingIndex

            TableModel = pxt.model_base()


            class Notes(TableModel, name='notes'):
                note_id: pxt.Int
                body: pxt.String

                __indexes__ = [{index}]
            """
        )
        schema_file = project_dir / 'app_schema.py'
        schema_file.write_text(
            schema_template.format(index="EmbeddingIndex(body, embedding=dummy_embedding, name='ix')")
        )
        target = p('index_replace')
        cli('schema', 'update', str(schema_file), target)
        notes = pxt.get_table(f'{target}/notes')
        indexes = notes.get_metadata()['indexes']
        assert set(indexes.keys()) == {'ix'}
        assert indexes['ix']['parameters']['precision'] == 'fp16'

        # the same index name, with different properties
        schema_file.write_text(
            schema_template.format(index="EmbeddingIndex(body, embedding=dummy_embedding, precision='fp32', name='ix')")
        )

        r = cli('schema', 'diff', str(schema_file), target, '--json', check=False)
        assert r.returncode == 2
        tbl = r.json['tables'][0]
        assert tbl['resolution'] == 'update_destructive'
        assert [(op['op'], op['target'], op['name']) for op in tbl['ops']] == [
            ('drop', 'index', 'ix'),
            ('add', 'index', 'ix'),
        ]

        r = cli('schema', 'diff', str(schema_file), target, check=False)
        assert (
            re.search(
                r"index 'ix' on column 'body' will be dropped and re-created from its new definition\s+DESTRUCTIVE",
                r.stdout,
            )
            is not None
        ), r.stdout
        assert re.search(r"EmbeddingIndex 'ix' will be re-created\s+safe", r.stdout) is not None, r.stdout

        r = cli('schema', 'update', str(schema_file), target, '--allow-destructive', '-f')
        assert r.returncode == 0

        # one index of that name remains, built from the new definition, and it answers a similarity query
        notes = pxt.get_table(f'{target}/notes')
        indexes = notes.get_metadata()['indexes']
        assert set(indexes.keys()) == {'ix'}
        assert indexes['ix']['parameters']['precision'] == 'fp32'

        assert_in_agreement(cli, str(schema_file), target)

    def test_iterator_view_indexes(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A schema that declares an iterator view and indexes over what the iterator produces."""
        skip_test_if_not_installed('spacy')  # the view's iterator splits on sentences
        target = db_root.make_catalog_path('app')
        cli('schema', 'update', apps('search.py'), target)

        articles = pxt.get_table(f'{target}/articles')
        articles.insert([{'article_id': 1, 'body': 'One sentence. And a second one.'}])
        assert articles.select(articles.word_count).collect()['word_count'] == [31]

        # the view holds a row per sentence, with the column the iterator produces
        chunks = pxt.get_table(f'{target}/chunks')
        assert chunks.count() == 2
        assert 'text' in chunks.columns()

        # both indexes the models declare exist, and the embedding one answers a similarity query
        indexes = chunks.get_metadata()['indexes']
        assert [ix['index_type'] for ix in indexes.values()] == ['embedding']
        assert [ix['index_type'] for ix in articles.get_metadata()['indexes'].values()] == ['btree']
        sim = chunks.text.similarity(string='sentence')
        assert len(chunks.order_by(sim, asc=False).limit(1).collect()) == 1

        assert_in_agreement(cli, apps('search.py'), target)

    def test_query_udf_column(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A column that calls a query udf reads that udf's table."""
        target = db_root.make_catalog_path('retrieval')
        cli('schema', 'update', apps('retrieval.py'), target)

        pxt.get_table(f'{target}/docs').insert([{'doc_id': 1, 'title': 'alpha'}, {'doc_id': 5, 'title': 'beta'}])
        probe = pxt.get_table(f'{target}/probe')
        probe.insert([{'cutoff': 0}, {'cutoff': 1}])
        assert probe.order_by(probe.cutoff).select(probe.matches).collect()['matches'] == [
            [{'title': 'alpha'}, {'title': 'beta'}],
            [{'title': 'beta'}],
        ]
        assert_in_agreement(cli, apps('retrieval.py'), target)

    def test_media_columns(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A schema with media columns and a view over an iterator that extracts frames from them."""
        target = db_root.make_catalog_path('app')
        cli('schema', 'update', apps('media.py'), target)

        clips = pxt.get_table(f'{target}/clips')
        assert clips.get_metadata()['columns']['video']['type_'] == 'Video'
        clips.insert([{'clip_id': 1, 'video': get_video_files()[0], 'caption': 'a clip'}])

        # the media-typed computed column produced an image, and the iterator a row per frame
        assert clips.get_metadata()['columns']['poster']['type_'] == 'Image | None'
        assert clips.select(path=clips.poster.localpath).collect()[0]['path'].endswith(('.png', '.jpg', '.jpeg'))
        assert pxt.get_table(f'{target}/frames').count() > 0

        # the other media types, and a computed column reading one of them
        recordings = pxt.get_table(f'{target}/recordings')
        columns = recordings.get_metadata()['columns']
        assert (columns['audio']['type_'], columns['transcript']['type_']) == ('Audio', 'Document')
        # a .txt document parses with no optional package, unlike .md and the office formats
        transcript = next(d for d in get_documents() if d.endswith('pxtbrief.txt'))
        recordings.insert([{'recording_id': 1, 'audio': get_audio_files()[0], 'transcript': transcript}])
        codec = recordings.audio_metadata.streams[0].codec_context.name
        assert recordings.select(codec=codec).collect()[0]['codec'] == 'flac'

        assert_in_agreement(cli, apps('media.py'), target)

    def test_routes_not_schema(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """An application file's routes are invisible to the schema: only its models declare tables."""
        target = db_root.make_catalog_path('app')
        cli('schema', 'update', apps('basic.py'), target)
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'doc_id': 1, 'title': 'hello', 'body': 'world', 'published': True}])
        assert docs.select(docs.summary, docs.title_upper).collect()[0] == {'summary': 'hello', 'title_upper': 'HELLO'}

        # the variant adds a route and nothing else, so the schema is already in agreement with it
        assert_in_agreement(cli, apps('basic_added_route.py'), target)

    def test_extras_prune(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """A table the file does not declare: reported as an extra, left alone by update, dropped by prune."""
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('prune')
        cli('schema', 'update', str(schema_file), target)
        assert_in_agreement(cli, str(schema_file), target)
        validate_tables(target)

        # nothing undeclared yet
        r = cli('schema', 'prune', str(schema_file), target, '-f')
        assert r.returncode == 0
        assert 'nothing to prune' in r.stdout

        # a view over an undeclared table, so the drops have to be ordered base-last
        pxt.create_table(f'{target}/scratch', {'x': pxt.Int | None})
        scratch = pxt.get_table(f'{target}/scratch')
        pxt.create_view(f'{target}/scratch_view', scratch.where(scratch.x > 0))

        # a table no model declares is reported, but update would not touch it, so the target is still in agreement
        r = cli('schema', 'diff', str(schema_file), target, '--json')
        assert r.returncode == 0
        assert r.json['in_agreement']
        assert sorted(r.json['extras']) == [f'{target}/scratch', f'{target}/scratch_view']
        assert r.json['summary']['extras'] == 2

        r = cli('schema', 'diff', str(schema_file), target)
        assert f'! {target}/scratch' in r.stdout
        assert 'extra (not in schema)' in r.stdout

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
        assert_in_agreement(cli, str(schema_file), target)

    def test_prune_declared_dependents(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        target = p('keep')

        # 'derived' is declared as a view of 'raw', but only 'derived' is in the schema, so 'raw' reads as an extra
        pxt.create_dir(target)
        raw = pxt.create_table(f'{target}/raw', {'id': pxt.Int | None})
        pxt.create_view(f'{target}/derived', raw.where(raw.id > 0))
        schema_file = project_dir / 'app_schema.py'
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
        assert re.search(r"the following depend on it: '.*keep/derived'", r.stderr) is not None
        assert pxt.get_table(f'{target}/raw') is not None

    def test_prune_reports_dropped(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        target = p('partial')

        # 'gone' prunes cleanly; 'raw' cannot, because the schema keeps the view that depends on it
        pxt.create_dir(target)
        pxt.create_table(f'{target}/gone', {'id': pxt.Int | None})
        raw = pxt.create_table(f'{target}/raw', {'id': pxt.Int | None})
        pxt.create_view(f'{target}/derived', raw.where(raw.id > 0))
        schema_file = project_dir / 'app_schema.py'
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

    @pytest.mark.db_roots(
        'local', 'proxy', reason='Fails with an SSL error, possibly due to excessive data transfer [PXT-1327]'
    )
    def test_example(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        skip_test_if_not_installed('sentence_transformers')
        p = db_root.make_catalog_path
        target = p('documented')

        # the file the command emits has to be one that actually works
        schema_file = project_dir / 'example.py'
        schema_file.write_text(cli('schema', 'example', '--brief').stdout)

        r = cli('schema', 'update', str(schema_file), target)
        assert r.stdout.count('created') == 2
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'title': 'hello', 'body': 'world'}, {'title': '', 'body': 'untitled'}])
        titled = pxt.get_table(f'{target}/titled')
        assert titled.select(titled.headline).collect()['headline'] == ['HELLO!']
        assert_in_agreement(cli, str(schema_file), target)

        # --out writes the same bytes to a file, for either form
        out_file = project_dir / 'out.py'
        r = cli('schema', 'example', '--brief', '--out', str(out_file))
        assert f'wrote {out_file}' in r.stdout
        assert out_file.read_text() == schema_file.read_text()
        cli('schema', 'example', '--out', str(out_file))
        assert out_file.read_text() == cli('schema', 'example').stdout

        # the full example is the default, and it declares every construct the DSL supports
        full = out_file.read_text()
        assert all(
            construct in full
            for construct in ('pxt.Column(', 'pxt.EmbeddingIndex(', 'iterator=', 'base=', 'pxt.Document', '@pxt.udf')
        )
        # it has to be a file the daemon can import and plan, media types and embedding index included
        full_target = p('full_example')
        r = cli('schema', 'diff', str(out_file), full_target, check=False)
        assert r.returncode == 2, r.stderr  # 2 = changes pending, ie the plan was computed

        # and applying it has to produce working tables, the embedding index included
        r = cli('schema', 'update', str(out_file), full_target)
        assert r.stdout.count('created') == 4
        assert_in_agreement(cli, str(out_file), full_target)  # nothing left to apply
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

    def test_diff_unsupported(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)
        target = p('unsupported')
        cli('schema', 'update', str(schema_file), target)
        validate_tables(target)

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

    def test_udfs_in_application_files(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """Computed columns over udfs that an application's own package and its neighbors define."""
        if db_root.id == 'cloud':
            pytest.skip('the runtime of a hosted database does not hold this project')
        p = db_root.make_catalog_path
        # project_dir sits directly under the project root, so it leads every module path below it
        package = project_dir.name

        def write_app(name: str) -> pathlib.Path:
            """An application whose columns call udfs from a sibling module and from a package below it."""
            directory = project_dir / name
            (directory / 'pkg').mkdir(parents=True)
            (directory / 'pkg' / '__init__.py').write_text(f"SUFFIX = '{name}-pkg'\n")
            (directory / 'pkg' / 'inner.py').write_text(
                dedent(
                    """
                    import pixeltable as pxt

                    @pxt.udf
                    def shout(s: str) -> str:
                        return s.upper()
                    """
                )
            )
            (directory / 'helpers.py').write_text(f"TAG = '{name}'\n")
            (directory / 'functions.py').write_text(
                dedent(
                    f"""
                    import pixeltable as pxt
                    from {package}.{name}.helpers import TAG
                    from {package}.{name}.pkg import SUFFIX

                    @pxt.udf
                    def tag(s: str) -> str:
                        return f'{{s}}-{{TAG}}-{{SUFFIX}}'
                    """
                )
            )
            app_file = directory / 'app.py'
            app_file.write_text(
                dedent(
                    f"""
                    from __future__ import annotations

                    import pixeltable as pxt
                    from {package}.{name}.functions import tag
                    from {package}.{name}.pkg.inner import shout

                    TableModel = pxt.model_base()


                    class Docs(TableModel, name='docs'):
                        doc_id = pxt.Column(type=pxt.Int, primary_key=True)
                        title: pxt.String
                        tagged = tag(title)  # noqa: F821
                        shouted = shout(title)  # noqa: F821
                    """
                )
            )
            return app_file

        # two applications of one project declare a udf of the same name, in files of the same name
        for name in ('proj1', 'proj2'):
            app_file = write_app(name)
            cli('schema', 'update', str(app_file), p(name))
            assert_in_agreement(cli, str(app_file), p(name))

        # each application's module path is its own, so each udf reaches its own neighbors
        for name, expected in (('proj1', 'a-proj1-proj1-pkg'), ('proj2', 'a-proj2-proj2-pkg')):
            docs = pxt.get_table(f'{p(name)}/docs')
            docs.insert([{'doc_id': 1, 'title': 'a'}])
            # tagged calls a udf from a sibling module, shouted one from a module of the package below it
            assert docs.select(docs.tagged, docs.shouted).collect()[0] == {'tagged': expected, 'shouted': 'A'}

        # a column stores which udf it calls, not the udf's body, so editing the body changes no schema
        (project_dir / 'proj1' / 'helpers.py').write_text("TAG = 'edited'\n")
        assert_in_agreement(cli, str(project_dir / 'proj1' / 'app.py'), p('proj1'))

    @pytest.mark.db_roots('local', reason='check reads no catalog, so the target axis adds nothing')
    def test_check(
        self,
        cli: PxtRunner,
        apps: Callable[[str], str],
        project_dir: pathlib.Path,
        session_project: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """check validates a file on its own: it imports, declares models, and its udf paths resolve."""
        r = cli('schema', 'check', apps('basic.py'))
        assert r.returncode == 0
        assert 'valid' in r.stdout

        report = cli('schema', 'check', apps('basic.py'), '--json').json
        assert (report['valid'], report['errors'], report['warnings']) == (True, [], [])
        assert report['file'] == apps('basic.py')

        # a file that modifies the catalog while importing
        ddl = project_dir / 'ddl.py'
        ddl.write_text("import pixeltable as pxt\npxt.create_table('t', {'c': pxt.Int})\n", encoding='utf-8')
        r = cli('schema', 'check', str(ddl), check=False)
        assert r.returncode == 1
        assert 'modifies the catalog while it is imported' in r.stderr

        # an empty file
        empty = project_dir / 'empty.py'
        empty.write_text('import pixeltable as pxt\n', encoding='utf-8')
        r = cli('schema', 'check', str(empty), check=False)
        assert r.returncode == 1
        assert 'no model_base() found' in r.stderr

        # non-existent path
        r = cli('schema', 'check', str(project_dir / 'nonexistent.py'), check=False)
        assert r.returncode == 1
        assert 'file not found' in r.stderr

        # a file under a nested project config file
        other_root = tmp_path / 'other_project'
        other_root.mkdir()
        (other_root / 'pixeltable.toml').write_text('', encoding='utf-8')
        outside = other_root / 'app.py'
        outside.write_text(SCHEMA_SRC, encoding='utf-8')
        r = cli('schema', 'check', str(outside), check=False)
        assert r.returncode == 1
        assert f'the project root is {session_project}, which the file does not sit under' in r.stderr
        assert 'has to sit under that root' in r.stderr

        # missing project config file
        no_project = tmp_path / 'loose.py'
        no_project.write_text(SCHEMA_SRC, encoding='utf-8')
        r = cli('schema', 'check', str(no_project), check=False)
        assert r.returncode == 1
        assert 'does not sit under' in r.stderr or 'there is no project root' in r.stderr

        # a shadowed local source file
        shadowed = session_project / 'psutil.py'
        shadowed.write_text('TAG = 1\n', encoding='utf-8')
        try:
            report = cli('schema', 'check', apps('basic.py'), '--json').json
        finally:
            shadowed.unlink()  # the project is shared with every other test in the session
        assert report['valid'], report
        assert len(report['warnings']) == 1, report
        assert "an import of 'psutil' reads" in report['warnings'][0]

    @pytest.mark.db_roots(
        'local', reason='the column is resolved in this process, so the report of a missing file lands here'
    )
    def test_update_errors(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
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
        r = cli('schema', 'update', str(project_dir / 'nonexistent.py'), p('app'), check=False)
        assert r.returncode == 1
        assert 'not found' in r.stderr

        # schema file without a model base
        no_base = project_dir / 'no_base.py'
        no_base.write_text('import pixeltable as pxt\n')
        r = cli('schema', 'update', str(no_base), p('app'), check=False)
        assert r.returncode == 1
        assert 'no model_base()' in r.stderr

        # schema file that fails to load
        broken = project_dir / 'broken.py'
        broken.write_text('raise RuntimeError("boom")\n')
        r = cli('schema', 'update', str(broken), p('app'), check=False)
        assert r.returncode == 1
        assert 'error loading' in r.stderr

        # one that defines a udf before it fails: the udf is registered by the time the failure happens,
        # and the fixed file redefines it, so a load that keeps it would refuse the second one
        halfway = project_dir / 'halfway.py'
        udf_src = dedent(
            """
            @pxt.udf
            def shout(s: str) -> str:
                return s.upper()
            """
        )
        halfway.write_text(SCHEMA_SRC + udf_src + '\nraise RuntimeError("boom")\n')
        r = cli('schema', 'update', str(halfway), p('halfway'), check=False)
        assert r.returncode == 1
        assert 'error loading' in r.stderr
        halfway.write_text(SCHEMA_SRC + udf_src)
        cli('schema', 'update', str(halfway), p('halfway'))

        # a schema file sits at the top of its project, so an import above it names no package
        above = project_dir / 'above.py'
        above.write_text('from .. import something\n')
        r = cli('schema', 'update', str(above), p('app'), check=False)
        assert r.returncode == 1
        assert 'attempted relative import' in r.stderr

    def test_update_relative_path(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        (project_dir / 'app_schema.py').write_text(SCHEMA_SRC)
        target = p('rel_app')

        # a relative schema path is resolved against the client's cwd, so run the command from that directory
        r = cli('schema', 'update', 'app_schema.py', target, cwd=project_dir)
        assert r.stdout.count('created') == 2
        validate_tables(target)

    def test_update_unbound_config_var(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """A schema referencing a value the target has not bound fails before any table is created."""
        p = db_root.make_catalog_path
        schema_file = project_dir / 'app_schema.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                MEDIA_DEST = pxt.ConfigVar('no_such_media_dest', pxt.URI)

                TableModel = pxt.model_base()


                class Docs(TableModel, name='docs'):
                    img: pxt.Image | None
                    thumbnail = pxt.Column(value=img.rotate(90), destination=MEDIA_DEST.value())
                """
            )
        )
        target = p('app')

        # diff shares the plan path with update, so it must report the same thing rather than succeed
        for verb in ('diff', 'update'):
            r = cli('schema', verb, str(schema_file), target, check=False)
            assert r.returncode == 1, verb
            assert 'no_such_media_dest' in r.stderr, verb
            assert 'is not set' in r.stderr, verb

        assert pxt.get_table(f'{target}/docs', if_not_exists='ignore') is None
