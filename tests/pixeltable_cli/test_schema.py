"""Tests for 'pxt schema update'."""

import pathlib
from collections.abc import Callable
from textwrap import dedent

import pixeltable as pxt

from .conftest import PxtRunner

SCHEMA_SRC = dedent(
    """
    import pixeltable as pxt
    import pixeltable.functions as pxtf

    TableModel = pxt.model_base()


    class Docs(TableModel, name='docs'):
        title: pxt.Required[pxt.String]
        body: pxt.String
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

        # idempotent rerun: exit 0, nothing created, both reported as existing
        r = cli('schema', 'update', str(schema_file), target)
        assert r.returncode == 0
        assert r.stdout.count('created') == 0
        assert r.stdout.count('exists') == 2

        # json output
        r = cli('schema', 'update', str(schema_file), target, '--json')
        assert [e['action'] for e in r.json['tables']] == ['exists', 'exists']

        # a model whose kind conflicts with the existing object (table vs view) is an error
        schema_file.write_text(
            dedent(
                """
                import pixeltable as pxt

                TableModel = pxt.model_base()


                class TitledDocs(TableModel, name='titled_docs'):
                    headline: pxt.String
                """
            )
        )
        r = cli('schema', 'update', str(schema_file), target, check=False)
        assert r.returncode == 1
        assert 'is defined as a table' in r.stderr

    def test_update_errors(
        self, cli: PxtRunner, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        p = make_catalog_path
        schema_file = tmp_path / 'app_schema.py'
        schema_file.write_text(SCHEMA_SRC)

        # unknown verb
        r = cli('schema', 'doesnotexist', str(schema_file), p('app'), check=False)
        assert r.returncode == 2
        assert 'unknown verb' in r.stderr

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
