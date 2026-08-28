"""Route declarations against TableModels.

Separate from test_fastapi.py because a module that declares a TableModel needs
`from __future__ import annotations`, which test_fastapi.py cannot have: its response models are pydantic
classes defined inside the test functions, and pydantic cannot resolve stringized annotations from there.
"""

from __future__ import annotations

import json
from typing import Callable

import pytest

import pixeltable as pxt

from ..utils import get_image_files, pxt_raises, skip_test_if_not_installed
from .test_fastapi import add_one, make_test_client


class TestFastAPIModels:
    def test_model_target(self, make_catalog_path: Callable[[str], str]) -> None:
        """Routes can be declared against a model before the table it describes exists."""
        p = make_catalog_path
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        TableModel = pxt.model_base()  # noqa: N806

        class Notes(TableModel, name='notes'):
            note_id = pxt.Column(type=pxt.Int, primary_key=True)
            val: pxt.Int
            incr = add_one(val)  # noqa: F821
            img: pxt.Image | None
            thumb = img.resize(size=(8, 8))  # noqa: F821

        @pxt.query
        def note_thumb(note_id: int) -> pxt.Query:
            return Notes.where(Notes.note_id == note_id).select(Notes.thumb)  # type: ignore[arg-type]

        # a model attribute is a ColumnRefByName at runtime, but a type checker sees the column's declared value
        # type, so these do not satisfy the declared argument types; the ignores go away with the mypy plugin fix
        router = FastAPIRouter()
        router.add_insert_route(
            Notes,
            path='/ins',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[Notes.incr],  # type: ignore[arg-type]
        )
        router.add_compute_route(
            Notes,
            path='/comp',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[Notes.incr],  # type: ignore[arg-type]
        )
        router.add_update_route(
            Notes,
            path='/upd',
            inputs=[Notes.val],  # type: ignore[arg-type]
            outputs=[Notes.note_id, Notes.incr],  # type: ignore[arg-type]
        )
        router.add_delete_route(
            Notes,
            path='/del',
            match_columns=[Notes.note_id],  # type: ignore[arg-type]
        )
        # one query over the model, serving two routes whose responses differ
        router.add_query_route(path='/thumb-json', query=note_thumb)
        router.add_query_route(path='/thumb-file', query=note_thumb, return_fileresponse=True)
        client = make_test_client(router)

        # the definition names the model each route was declared against, before the table exists
        service = router.service_spec(name='notes')
        assert json.loads(json.dumps(service)) == service
        specs = {(spec['method'], spec['path']): spec for spec in service['routes']}
        assert all(spec['model'] == 'notes' for spec in specs.values()), specs
        assert all(spec['table'] is None for spec in specs.values()), specs
        assert specs['POST', '/ins']['inputs'] == ['note_id', 'val']
        assert specs['POST', '/del']['match_columns'] == ['note_id']
        assert specs['POST', '/thumb-file']['return_fileresponse']
        assert specs['POST', '/thumb-json']['query'].endswith('note_thumb')

        # the routes are fully described before the table exists
        schema = client.get('/openapi.json').json()
        assert sorted(path for path in schema['paths'] if not path.startswith('/_pxt/media')) == [
            '/_pxt/jobs/{job_id}',
            '/comp',
            '/del',
            '/ins',
            '/thumb-file',
            '/thumb-json',
            '/upd',
        ]

        TableModel.create_all(p(''))
        router.bind(p(''))

        assert client.post('/ins', json={'note_id': 1, 'val': 10}).json() == {'incr': 11}
        assert client.post('/comp', json={'note_id': 2, 'val': 20}).json() == {'incr': 21}
        assert client.post('/upd', json={'note_id': 1, 'val': 40}).json() == {'note_id': 1, 'incr': 41}
        assert client.post('/del', json={'note_id': 1}).json() == {'num_rows': 1}

        Notes.table.insert([{'note_id': 5, 'val': 50, 'img': get_image_files()[0]}])
        # each route serves the media column the way its own response needs it
        rows = client.post('/thumb-json', json={'note_id': 5}).json()['rows']
        assert len(rows) == 1, rows
        assert '/media/' in rows[0]['thumb'], rows[0]['thumb']
        resp = client.post('/thumb-file', json={'note_id': 5})
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('image/'), resp.headers['content-type']

        # the contract is frozen against the schema seen at bind time
        Notes.table.add_column(extra=pxt.String | None)
        resp = client.post('/ins', json={'note_id': 3, 'val': 30})
        assert resp.status_code == 409, resp.text
        assert 'schema changed' in resp.json()['detail']

    def test_model_target_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        # a router names the service it declares, validated like a Pixeltable identifier
        assert FastAPIRouter(name='docs-api').name == 'docs-api'
        assert FastAPIRouter().name is None
        # name is consumed by FastAPIRouter; APIRouter's own parameters still work
        assert FastAPIRouter(name='ingest', prefix='/v1').prefix == '/v1'
        for bad_name in ('-lead', '_lead', 'has space', 'has.dot', ''):
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='is not a valid service name'):
                FastAPIRouter(name=bad_name)

        TableModel = pxt.model_base()  # noqa: N806

        class Notes(TableModel, name='notes'):
            note_id = pxt.Column(type=pxt.Int, primary_key=True)
            val: pxt.Int

        class BigNotes(TableModel, name='big_notes', base=Notes.where(Notes.val > 10)):
            doubled = Notes.val * 2

        # a route on a prefixed router declares its path relative to the prefix, which is recorded once
        prefixed = FastAPIRouter(name='ingest', prefix='/v1')
        prefixed.add_insert_route(Notes, path='/ins')
        service = prefixed.service_spec()
        assert service['name'] == 'ingest'  # the name the router was constructed with
        assert service['routes'][0]['path'] == '/v1/ins'  # a route records the path as it is served

        router = FastAPIRouter()
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match="unknown column 'nosuchcol'"):
            router.add_delete_route(Notes, path='/e', match_columns=['nosuchcol'])
        # a view cannot be inserted into or deleted from, whether it is named by a model or by a table
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot insert into view'):
            router.add_insert_route(BigNotes, path='/e')
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot delete from view'):
            router.add_delete_route(BigNotes, path='/e')

        # serving a route whose router was never bound names the route and its model
        @pxt.query
        def all_notes() -> pxt.Query:
            return Notes.select(Notes.val)

        router.add_insert_route(Notes, path='/ins')
        router.add_query_route(path='/q', query=all_notes)
        client = make_test_client(router)
        for path, body in (('/ins', {'note_id': 1, 'val': 10}), ('/q', {})):
            resp = client.post(path, json=body)
            assert resp.status_code == 503, resp.text
            assert 'has not been bound' in resp.json()['detail']
            assert '`Notes`' in resp.json()['detail']

        TableModel.create_all(p(''))
        router.bind(p(''))
        assert client.post('/ins', json={'note_id': 1, 'val': 10}).status_code == 200
        assert client.post('/q', json={}).json() == {'rows': [{'val': 10}]}

    def test_view_model_target(self, make_catalog_path: Callable[[str], str]) -> None:
        """A compute route can be declared against a view model before either table exists."""
        p = make_catalog_path
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        TableModel = pxt.model_base()  # noqa: N806

        class Notes(TableModel, name='notes'):
            note_id = pxt.Column(type=pxt.Int, primary_key=True)
            val: pxt.Int

        class BigNotes(TableModel, name='big_notes', base=Notes.where(Notes.val > 10)):
            doubled = Notes.val * 2

        class Halved(TableModel, name='halved', base=Notes.where(Notes.val > 10).select(half=Notes.val / 2)):
            plus = half + 1  # type: ignore[name-defined]  # noqa: F821  # the select() alias, in the body

        router = FastAPIRouter()
        router.add_compute_route(
            BigNotes,
            path='/big',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[BigNotes.doubled],  # type: ignore[arg-type]
        )
        # a view whose base query has a select() list: the columns it projects are its own, alongside the ones
        # its body declares
        router.add_compute_route(
            Halved,
            path='/half',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[Halved.half, Halved.plus],
        )
        client = make_test_client(router)

        # the request takes the base's columns and the response the view's own, before either table exists
        schemas = client.get('/openapi.json').json()['components']['schemas']
        assert sorted(schemas['Body_compute_big_big_post']['properties']) == ['note_id', 'val']
        assert sorted(schemas['BigResponse']['properties']) == ['doubled']
        assert sorted(schemas['Body_compute_half_half_post']['properties']) == ['note_id', 'val']
        assert sorted(schemas['HalfResponse']['properties']) == ['half', 'plus']

        TableModel.create_all(p(''))
        router.bind(p(''))

        # the filter admits this row, so the view computes one
        assert client.post('/big', json={'note_id': 1, 'val': 20}).json() == {'doubled': 40}
        # and drops this one
        assert client.post('/big', json={'note_id': 2, 'val': 5}).json() is None

        assert client.post('/half', json={'note_id': 3, 'val': 20}).json() == {'half': 10.0, 'plus': 11.0}
        assert client.post('/half', json={'note_id': 4, 'val': 5}).json() is None

    def test_bind(self, make_catalog_path: Callable[[str], str]) -> None:
        """bind() resolves model targets, refuses what the tables cannot serve, and rejects a second target."""
        p = make_catalog_path
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        TableModel = pxt.model_base()  # noqa: N806

        class Notes(TableModel, name='notes'):
            note_id = pxt.Column(type=pxt.Int, primary_key=True)
            val: pxt.Int
            note: pxt.String | None

        @pxt.query
        def notes_by_val(min_val: int) -> pxt.Query:
            return Notes.where(Notes.val > min_val).select(Notes.note)  # type: ignore[arg-type]

        router = FastAPIRouter()
        router.add_insert_route(
            Notes,
            path='/ins',
            inputs=[Notes.note_id, Notes.val, Notes.note],  # type: ignore[arg-type]
            outputs=[Notes.note_id],  # type: ignore[arg-type]
        )
        router.add_query_route(path='/q', query=notes_by_val)

        # before the table exists there is nothing to serve, and the refusal names the command that fixes it
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match=r"(?s)'notes'.*does not yet exist.*pxt schema update"):
            router.bind(p(''))

        # an unbound router refuses to start, rather than failing once per request
        app = fastapi.FastAPI()
        app.include_router(router)
        with pxt_raises(pxt.ErrorCode.NOT_BOUND, match="'POST /ins'"), TestClient(app):
            pass

        TableModel.create_all(p(''))
        router.bind(p(''))
        client = make_test_client(router)
        resp = client.post('/ins', json={'note_id': 1, 'val': 10, 'note': 'hi'})
        assert resp.status_code == 200, resp.text
        assert client.post('/q', json={'min_val': 1}).json() == {'rows': [{'note': 'hi'}]}

        # binding is to one place: a second target would silently retarget the routes already being served
        with pxt_raises(pxt.ErrorCode.ALREADY_BOUND, match='already bound'):
            router.bind(p('elsewhere'))

        # any difference from the model stops it binding, whether a route names the column or not: what the
        # routes serve is built from the declaration, and fixing the table is a schema change
        Notes.table.add_column(unrelated=pxt.String | None)
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match=r"(?s)DROPPED.*'unrelated'"):
            router.bind(p(''))
        Notes.table.drop_column('unrelated')
        Notes.table.drop_column('note')
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match=r"(?s)ADDED.*'note'"):
            router.bind(p(''))

    @pytest.mark.skip_cloud(reason='Unclear; re-run once other known issues are fixed')
    def test_bind_mismatches(self, make_catalog_path: Callable[[str], str]) -> None:
        """Every difference between a model and the table it names stops the routes declared against it."""
        p = make_catalog_path
        skip_test_if_not_installed('fastapi')

        from pixeltable.serving import FastAPIRouter

        TableModel = pxt.model_base()  # noqa: N806

        class Notes(TableModel, name='notes'):
            note_id = pxt.Column(type=pxt.Int, primary_key=True)
            val: pxt.Int
            __indexes__ = [pxt.BtreeIndex(val)]  # noqa: F821, RUF012

        class Tags(TableModel, name='tags'):
            tag_id = pxt.Column(type=pxt.Int, primary_key=True)
            label: pxt.String

        notes_router = FastAPIRouter()
        notes_router.add_insert_route(
            Notes,
            path='/notes',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[Notes.note_id],  # type: ignore[arg-type]
        )

        # a view where the model declares a table: the columns line up, so only the kind says it cannot serve,
        # and nothing a schema update does turns one into the other
        src = pxt.create_table(p('src'), {'note_id': pxt.Int, 'val': pxt.Int})
        pxt.create_view(p('notes'), src.where(src.val > 0))
        with pxt_raises(
            pxt.ErrorCode.SCHEMA_MISMATCH, match=r'(?s)kind mismatch.*is a view.*No schema update can reconcile'
        ):
            notes_router.bind(p(''))

        # a table whose key differs from the model's: an insert route would write rows the model cannot address
        pxt.create_dir(p('nopk'))
        pxt.create_table(p('nopk/notes'), {'note_id': pxt.Int, 'val': pxt.Int})
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match="'note_id'"):
            notes_router.bind(p('nopk'))

        # the same declaration against a table that satisfies it serves
        pxt.create_dir(p('ok'))
        TableModel.create_all(p('ok'))
        notes_router.bind(p('ok'))
        assert make_test_client(notes_router).post('/notes', json={'note_id': 1, 'val': 10}).status_code == 200

        router = FastAPIRouter()
        router.add_insert_route(
            Notes,
            path='/notes',
            inputs=[Notes.note_id, Notes.val],  # type: ignore[arg-type]
            outputs=[Notes.note_id],  # type: ignore[arg-type]
        )
        router.add_insert_route(
            Tags,
            path='/tags',
            inputs=[Tags.tag_id, Tags.label],  # type: ignore[arg-type]
            outputs=[Tags.tag_id],  # type: ignore[arg-type]
        )
        router.bind(p('ok'))

        # an index the model declares, dropped from the table: a custom endpoint may rely on it, and no column
        # differs, so nothing but the index says the table no longer matches
        idx_name = next(iter(Notes.table.get_metadata()['indexes']))
        Notes.table.drop_index(idx_name=idx_name)
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match=r'(?s)indexes.*ADDED.*pxt schema update'):
            router.bind(p('ok'))

        # a second table that also drifted: one refusal names both, so one schema update settles them
        Tags.table.add_column(extra=pxt.String | None)
        with pxt_raises(pxt.ErrorCode.SCHEMA_MISMATCH, match=r"(?s)'notes'.*'tags'"):
            router.bind(p('ok'))
