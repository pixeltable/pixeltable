import pathlib
import time
from collections.abc import Callable, Iterator
from textwrap import dedent
from typing import Any

import httpx
import pytest

import pixeltable as pxt

from ..utils import get_audio_files, get_documents, get_video_files, skip_test_if_not_installed
from .conftest import BackgroundPxt, PxtRunner

pytestmark = pytest.mark.local('a local service serves the in-process catalog')

_REQUEST_TIMEOUT = 30.0


@pytest.fixture(autouse=True)
def stop_services(cli: PxtRunner) -> Iterator[None]:
    """Leave nothing running: a service outlives the test that started it, and the next one would see it."""
    yield
    for service in cli('service', 'list', '--json').json:
        cli('service', 'stop', f'{service["base_path"]}/{service["name"]}'.lstrip('/'))


def services(cli: PxtRunner, target: str | None = None) -> dict[str, dict[str, Any]]:
    """What is running, keyed by service name."""
    args = ['service', 'list', '--json'] if target is None else ['service', 'list', target, '--json']
    return {s['name']: s for s in cli(*args).json}


def deploy(cli: PxtRunner, app: str, target: str) -> None:
    """Create the tables the models declare, then serve the file's services against them."""
    cli('schema', 'update', app, target)
    cli('service', 'update', app, target, '-f')


def assert_serving(cli: PxtRunner, app: str, target: str, *names: str) -> dict[str, dict[str, Any]]:
    """Assert that what runs at the target is what the file declares, and that it answers.

    Three independent readings, because each catches what the others miss: the diff agrees (nothing stale is
    deployed), the registry lists exactly these services (nothing is missing or extra), and each endpoint
    serves the paths its own spec claims (a recorded service that is not really serving them fails here).
    """
    r = cli('service', 'diff', app, target, '--json')
    assert r.returncode == 0, r.stdout
    assert r.json['in_agreement'], r.json

    running = services(cli, target)
    assert sorted(running) == sorted(names), running

    for name in names:
        deployment = running[name]
        served = httpx.get(f'{deployment["endpoint"]}/openapi.json', timeout=_REQUEST_TIMEOUT)
        assert served.status_code == 200, served.text
        prefix = deployment['spec']['prefix']
        declared = {f'{prefix}{route["path"]}' for route in deployment['spec']['routes']}
        assert declared <= set(served.json()['paths']), (declared, sorted(served.json()['paths']))
    return running


def _post(endpoint: str, path: str, **body: Any) -> httpx.Response:
    resp = httpx.post(f'{endpoint}{path}', json=body, timeout=_REQUEST_TIMEOUT)
    assert resp.status_code == 200, resp.text
    return resp


def _await_job(job_url: str, timeout: float = 120.0) -> Any:
    """Poll a background job until it stops being pending, and return what it produced."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = httpx.get(job_url, timeout=_REQUEST_TIMEOUT).json()
        if status['status'] != 'pending':
            assert status['status'] == 'done', status
            return status['result']
        time.sleep(0.2)
    raise AssertionError(f'the job at {job_url} was still pending after {timeout:.0f}s')


def assert_not_serving(cli: PxtRunner, *names: str) -> None:
    running = services(cli)
    assert all(name not in running for name in names), running


class TestService:
    def test_deploying_needs_agreeing_config(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """A service inherits the daemon's config values, so a caller resolving them differently cannot deploy."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), make_catalog_path('app')
        cli('schema', 'update', app, target)
        differing = {'OPENAI_API_KEY': 'sk-not-the-one-the-daemon-has'}

        # the plan and the deployment are both refused, and the refusal names the variable and the remedy
        for args in (('diff', app, target), ('update', app, target, '-f')):
            r = cli('service', *args, env_overrides=differing, check=False)
            assert r.returncode == 1, r.stdout
            assert 'OPENAI_API_KEY' in r.stderr
            assert 'pxt daemon restart' in r.stderr
        assert services(cli) == {}, 'a refused update started something'

        # the same commands run for a caller whose environment the daemon shares
        deploy(cli, app, target)
        assert_serving(cli, app, target, 'ingest')

    def test_basic(self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]) -> None:
        """The first deployment: declare, see what is pending, apply it, use it, take it down."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), make_catalog_path('app')
        cli('schema', 'update', app, target)

        # nothing is deployed yet, and diff says so in its exit status
        r = cli('service', 'diff', app, target, '--json', check=False)
        assert r.returncode == 2
        assert [(s['name'], s['resolution']) for s in r.json['services']] == [('ingest', 'create')]
        assert services(cli) == {}

        cli('service', 'update', app, target, '-f')
        running = assert_serving(cli, app, target, 'ingest')
        assert running['ingest']['app_file'] == app
        assert running['ingest']['base_path'] == target
        # list reports what each service serves, in Pixeltable's own terms
        routes = {r['path']: r for r in running['ingest']['spec']['routes']}
        assert routes['/docs']['route_type'] == 'insert'
        assert routes['/docs']['inputs'] == ['doc_id', 'title', 'body', 'published']
        assert routes['/docs/delete']['route_type'] == 'delete'

        endpoint = running['ingest']['endpoint']
        resp = _post(endpoint, '/docs', doc_id=1, title='a long enough title', body=None, published=True)
        assert resp.json() == {'title_upper': 'A LONG ENOUGH TITLE', 'summary': 'a long enoug...'}
        docs = pxt.get_table(f'{target}/docs')
        assert docs.where(docs.doc_id == 1).count() == 1

        # a compute route answers without storing a row
        resp = _post(endpoint, '/preview', doc_id=2, title='unstored', published=False)
        assert resp.json() == {'summary': 'unstored'}
        assert docs.count() == 1

        # an update route identifies the row by its primary key, and a delete route removes it
        resp = _post(endpoint, '/docs/update', doc_id=1, title='renamed')
        assert resp.json() == {'title_upper': 'RENAMED'}
        resp = _post(endpoint, '/docs/delete', doc_id=1)
        assert resp.status_code == 200, resp.text
        assert docs.count() == 0

        # a second update has nothing to do, and leaves the process alone
        pid = running['ingest']['pid']
        r = cli('service', 'update', app, target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['skipped']
        assert services(cli)['ingest']['pid'] == pid

        cli('service', 'stop', 'ingest')
        assert_not_serving(cli, 'ingest')

    def test_iteration(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """Editing the file: an added route is applied by restarting; a changed contract needs a flag."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = make_catalog_path('app')
        deploy(cli, apps('basic.py'), target)
        before = assert_serving(cli, apps('basic.py'), target, 'ingest')['ingest']

        # the variant adds a route: additive, because what is already served keeps being served
        r = cli('service', 'diff', apps('basic_added_route.py'), target, '--json', check=False)
        assert [s['resolution'] for s in r.json['services']] == ['update_additive']
        assert [op['op'] for s in r.json['services'] for op in s['ops']] == ['add']

        # a dry run reports the same and changes nothing
        r = cli('service', 'update', apps('basic_added_route.py'), target, '-n', check=False)
        assert r.returncode == 2
        assert services(cli)['ingest']['pid'] == before['pid']

        r = cli('service', 'update', apps('basic_added_route.py'), target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['applied']
        after = assert_serving(cli, apps('basic_added_route.py'), target, 'ingest')['ingest']
        assert after['pid'] != before['pid'], 'a changed declaration is applied by replacing the process'

        # the added route serves, and so do the routes that were already there
        assert _post(after['endpoint'], '/shout', doc_id=3, title='new route', published=True).json() == {
            'title_upper': 'NEW ROUTE'
        }
        assert _post(after['endpoint'], '/preview', doc_id=4, title='still here', published=True).status_code == 200

        # dropping an output changes a contract callers may be using, so it is refused by default
        r = cli('service', 'update', apps('basic_changed_route.py'), target, '-f', check=False)
        assert r.returncode == 1
        assert '--allow-destructive' in r.stderr
        assert services(cli)['ingest']['pid'] == after['pid']

        cli('service', 'update', apps('basic_changed_route.py'), target, '-f', '--allow-destructive')
        assert_serving(cli, apps('basic_changed_route.py'), target, 'ingest')

    def test_prune(self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]) -> None:
        """A service the file stopped declaring is stopped and forgotten, and can be started again."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = make_catalog_path('app')
        deploy(cli, apps('basic.py'), target)

        # the variant declares the same models under a service of another name, so 'ingest' is an extra
        r = cli('service', 'diff', apps('basic_renamed_service.py'), target, '--json', check=False)
        assert r.json['extras'] == ['ingest']

        r = cli('service', 'prune', apps('basic_renamed_service.py'), target, '-n', check=False)
        assert r.returncode == 2
        assert 'ingest' in services(cli)

        r = cli('service', 'prune', apps('basic_renamed_service.py'), target, '-f', '--json')
        assert [(op['name'], op['status']) for op in r.json] == [('ingest', 'applied')]
        assert_not_serving(cli, 'ingest')

        # stopping is not destructive: declaring it again brings it back
        cli('service', 'update', apps('basic.py'), target, '-f')
        assert_serving(cli, apps('basic.py'), target, 'ingest')

    def test_run_in_the_foreground(
        self,
        cli: PxtRunner,
        cli_bg: Callable[..., BackgroundPxt],
        apps: Callable[[str], str],
        make_catalog_path: Callable[[str], str],
    ) -> None:
        """run serves from the calling process and records nothing; update is the background form."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), make_catalog_path('app')
        cli('schema', 'update', app, target)

        served = cli_bg('service', 'run', app, target)
        served.wait_until_serving()
        assert (
            _post(served.endpoint, '/docs', doc_id=7, title='foreground', body=None, published=True).status_code == 200
        )
        docs = pxt.get_table(f'{target}/docs')
        assert docs.where(docs.doc_id == 7).count() == 1
        assert services(cli) == {}, 'run records nothing'

        served.proc.terminate()
        served.proc.wait(timeout=30)

        # the same file, served the other way
        cli('service', 'update', app, target, '-f')
        assert_serving(cli, app, target, 'ingest')

    def test_blocked_on_the_database(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """A service whose tables do not exist is blocked until the schema is applied."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), make_catalog_path('app')

        r = cli('service', 'diff', app, target, '--json', check=False)
        assert r.returncode == 2
        assert [s['resolution'] for s in r.json['services']] == ['blocked']
        commands = {op['details'].get('command') for s in r.json['services'] for op in s['ops']}
        assert any(c is not None and 'schema update' in c for c in commands), commands

        r = cli('service', 'update', app, target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['refused']
        assert services(cli) == {}

        # applying the schema unblocks the service, and the new tables accept rows
        cli('schema', 'update', app, target)
        docs = pxt.get_table(f'{target}/docs')
        docs.insert([{'doc_id': 1, 'title': 'hello', 'body': 'world', 'published': True}])
        assert docs.select(docs.summary).collect()['summary'] == ['hello']
        assert [
            s['resolution'] for s in cli('service', 'diff', app, target, '--json', check=False).json['services']
        ] == ['create']

        cli('schema', 'update', app, target)
        cli('service', 'update', app, target, '-f')
        assert_serving(cli, app, target, 'ingest')

    def test_inspection(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """list inspects what a service serves, for every service or for one named by its address."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = make_catalog_path('app')
        deploy(cli, apps('media.py'), target)
        assert_serving(cli, apps('media.py'), target, 'clips', 'frames', 'recordings')

        # a listing carries what each service serves, in Pixeltable's terms rather than OpenAPI's: a video
        # arrives as an upload, which a JSON schema would render as an indistinguishable string
        clips = services(cli)['clips']['spec']
        upload = next(r for r in clips['routes'] if r['path'] == '/clips')
        # every field the route accepts is an input; uploadfile_inputs marks the ones that arrive as files
        assert upload['inputs'] == ['clip_id', 'caption', 'video']
        assert upload['uploadfile_inputs'] == ['video']
        poster = next(r for r in clips['routes'] if r['path'] == '/poster')
        assert poster['route_type'] == 'compute'
        assert poster['return_fileresponse']

        # the argument narrows the listing to one service, the way `describe` inspects one table
        assert sorted(services(cli, f'{target}/clips')) == ['clips']
        assert sorted(services(cli, target)) == ['clips', 'frames', 'recordings']

        # the plain rendering shows the routes under each service
        out = cli('service', 'list', f'{target}/clips').stdout
        assert '/clips' in out and 'video (file)' in out, out

    def test_media(self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]) -> None:
        """The routes whose request or response is not JSON: file uploads, a file response, a background job."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('media.py'), make_catalog_path('app')
        deploy(cli, app, target)
        running = assert_serving(cli, app, target, 'clips', 'frames', 'recordings')

        video = get_video_files()[0]
        video_url = pathlib.Path(video).as_uri()
        with open(video, 'rb') as f:
            resp = httpx.post(
                f'{running["clips"]["endpoint"]}/clips',
                data={'clip_id': 1, 'caption': 'a clip'},
                files={'video': ('clip.mp4', f, 'video/mp4')},
                timeout=_REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'clip_id': 1}
        assert pxt.get_table(f'{target}/frames').count() > 0

        # a single media value comes back as the image itself
        resp = _post(running['clips']['endpoint'], '/poster', clip_id=2, video=video_url)
        assert resp.headers['content-type'].startswith('image/'), resp.headers

        # a route over the iterator view answers with a row per frame, media rendered as urls
        rows = _post(running['frames']['endpoint'], '/frames', clip_id=3, video=video_url).json()
        assert len(rows) > 1, rows
        assert all(row['thumb'].startswith('http') for row in rows), rows[0]

        # a background route answers with a job to poll, and the two uploads arrive in one request
        audio = get_audio_files()[0]
        transcript = next(d for d in get_documents() if d.endswith('simple.md'))
        with open(audio, 'rb') as af, open(transcript, 'rb') as tf:
            resp = httpx.post(
                f'{running["recordings"]["endpoint"]}/recordings',
                data={'recording_id': 1},
                files={'audio': ('take.flac', af, 'audio/flac'), 'transcript': ('notes.md', tf, 'text/markdown')},
                timeout=_REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200, resp.text
        result = _await_job(resp.json()['job_url'])
        assert result['recording_id'] == 1, result
        assert result['audio_metadata']['streams'][0]['codec_context']['name'] == 'flac', result
        recordings = pxt.get_table(f'{target}/recordings')
        assert recordings.where(recordings.recording_id == 1).count() == 1

        # stopping one service of a file leaves the others serving
        cli('service', 'stop', f'{target}/frames')
        assert_not_serving(cli, 'frames')
        assert _post(running['clips']['endpoint'], '/poster', clip_id=4, video=video_url).status_code == 200

    def test_search(self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]) -> None:
        """An iterator view and an embedding index over the column the iterator produces."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        skip_test_if_not_installed('spacy')  # the view's iterator splits on sentences
        app, target = apps('search.py'), make_catalog_path('app')
        deploy(cli, app, target)
        endpoint = assert_serving(cli, app, target, 'search')['search']['endpoint']

        body = 'The cat dozed on the sill. Volcanoes reshape a coastline. An espresso machine builds pressure.'
        _post(endpoint, '/articles', article_id=1, body=body)

        # the iterator produced a chunk per sentence, and the index ranks them against a search string
        chunks = pxt.get_table(f'{target}/chunks')
        assert chunks.count() == 3
        sim = chunks.text.similarity(string='volcano')
        assert len(chunks.order_by(sim, asc=False).limit(2).collect()) == 2
        # the query routes run the query the application file declares, against the same index
        sentences = {text.strip() for text in chunks.select(chunks.text).collect()['text']}
        rows = _post(endpoint, '/similar', needle='volcano', limit=2).json()['rows']
        assert len(rows) == 2 and {row['text'].strip() for row in rows} <= sentences, rows
        # one_row=True refuses a query that returns more, so the limit is part of the request
        resp = httpx.get(f'{endpoint}/similar-one', params={'needle': 'volcano', 'limit': 1}, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200, resp.text
        assert resp.json()['text'].strip() in sentences, resp.json()

    def test_custom_app(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """A file supplying its own application declares a service Pixeltable cannot compare or serve."""
        skip_test_if_not_installed('fastapi')
        app, target = apps('custom.py'), make_catalog_path('app')
        cli('schema', 'update', app, target)

        # the file's models produce working tables even though its application cannot be served
        notes = pxt.get_table(f'{target}/notes')
        notes.insert([{'note_id': 1, 'text': 'hello'}])
        assert notes.select(notes.text_upper).collect()['text_upper'] == ['HELLO']

        r = cli('service', 'diff', app, target, '--json', check=False)
        diffs = {s['name']: s for s in r.json['services']}
        assert (diffs['app']['kind'], diffs['app']['resolution']) == ('custom', 'unsupported')
        assert diffs['app']['route_comparison'] == 'unavailable'
        assert diffs['app']['route_detail'] is not None
        # TODO(custom-app): the file declares one service, the application; the router it includes is
        # reported as a service of its own until the custom case is implemented
        assert (diffs['plain']['kind'], diffs['plain']['resolution']) == ('declarative', 'create')
        assert r.json['summary']['unsupported'] == 1

        # nothing in the file is served, and the refusal names the reason
        r = cli('service', 'update', app, target, '-f', check=False)
        assert r.returncode == 1
        assert 'is an application object of its own' in r.stderr
        assert services(cli) == {}

    def test_addressing(
        self, cli: PxtRunner, apps: Callable[[str], str], make_catalog_path: Callable[[str], str]
    ) -> None:
        """One name at two targets: a bare name is ambiguous, an address is not."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app = apps('basic.py')
        first, second = make_catalog_path('one'), make_catalog_path('two')
        deploy(cli, app, first)
        deploy(cli, app, second)
        assert_serving(cli, app, first, 'ingest')
        assert_serving(cli, app, second, 'ingest')

        # a listing narrows to one target
        assert sorted(services(cli, first)) == ['ingest']
        assert len(cli('service', 'list', '--json').json) == 2

        # the same name at two targets cannot be stopped by name alone
        r = cli('service', 'stop', 'ingest', check=False)
        assert r.returncode == 1
        assert 'ambiguous' in r.stderr
        assert f'{first}/ingest' in r.stderr and f'{second}/ingest' in r.stderr

        # the address says which one
        cli('service', 'stop', f'{first}/ingest')
        assert services(cli, first) == {}
        assert_serving(cli, app, second, 'ingest')

    @pytest.mark.local('check reads no catalog, so the target axis adds nothing')
    @pytest.mark.local('drives the local proxy daemon directly, so the target axis adds nothing')
    def test_proxy_daemon_project_handoff(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A running proxy daemon is reused for its own project, and replaced for another."""
        import httpx

        from pixeltable.config import Config
        from pixeltable.service import proxy_daemon

        db = 'test_handoff'
        original = Config.get().project_root
        assert original is not None
        try:
            endpoint = proxy_daemon.start(db, test_mode=True)
            first_pid = proxy_daemon.read_port_lock(db)['pid']

            # the daemon reports the project it was handed
            health = httpx.get(f'{endpoint}/health', timeout=10.0).json()
            assert health['project_root'] == str(original)

            # a caller in the same project reuses it
            assert proxy_daemon.start(db, test_mode=True) == endpoint
            assert proxy_daemon.read_port_lock(db)['pid'] == first_pid

            # a caller in another project gets a daemon for that one instead
            other = tmp_path / 'other_project'
            other.mkdir()
            (other / 'pixeltable.toml').write_text('', encoding='utf-8')
            Config.init(reinit=True, project_root=other)
            proxy_daemon.start(db, test_mode=True)
            assert proxy_daemon.read_port_lock(db)['pid'] != first_pid
            replaced = httpx.get(f'{proxy_daemon.endpoint(db)}/health', timeout=10.0).json()
            assert replaced['project_root'] == str(other)
        finally:
            proxy_daemon.stop(db)
            Config.init(reinit=True, project_root=original)

    def test_check(self, cli: PxtRunner, apps: Callable[[str], str], project_dir: pathlib.Path) -> None:
        """check validates an application file on its own: it imports and declares a service."""
        skip_test_if_not_installed('fastapi')
        r = cli('service', 'check', apps('basic.py'))
        assert r.returncode == 0
        assert 'valid' in r.stdout

        report = cli('service', 'check', apps('basic.py'), '--json').json
        assert (report['valid'], report['errors']) == (True, [])

        # a file that declares models but no service
        no_service = project_dir / 'no_service.py'
        no_service.write_text(
            'from __future__ import annotations\n\nimport pixeltable as pxt\n\n'
            'TableModel = pxt.model_base()\n\n\n'
            "class Docs(TableModel, name='docs'):\n    title: pxt.String\n",
            encoding='utf-8',
        )
        r = cli('service', 'check', str(no_service), check=False)
        assert r.returncode == 1
        assert 'no service found' in r.stderr

        # a router over models from another file, so this one declares no model base
        (project_dir / 'schema.py').write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                TableModel = pxt.model_base()


                class Docs(TableModel, name='docs'):
                    doc_id = pxt.Column(type=pxt.Int, primary_key=True)
                    title: pxt.String
                """
            ),
            encoding='utf-8',
        )
        routes_only = project_dir / 'routes_only.py'
        routes_only.write_text(
            dedent(
                f"""
                from __future__ import annotations

                from pixeltable.serving import FastAPIRouter

                from {project_dir.name}.schema import Docs

                api = FastAPIRouter(name='reader')
                api.add_insert_route(Docs, path='/docs', inputs=[Docs.doc_id, Docs.title])
                """
            ),
            encoding='utf-8',
        )
        report = cli('service', 'check', str(routes_only), '--json').json
        assert (report['valid'], report['errors'], report['warnings']) == (True, [], []), report

    def test_errors(
        self,
        cli: PxtRunner,
        apps: Callable[[str], str],
        make_catalog_path: Callable[[str], str],
        project_dir: pathlib.Path,
    ) -> None:
        """What the verbs do with a file that is missing, unimportable, or declares no service."""
        target = make_catalog_path('app')

        r = cli('service', 'diff', str(project_dir / 'nosuch.py'), target, check=False)
        assert r.returncode == 1
        assert 'not found' in r.stderr

        broken = project_dir / 'broken.py'
        broken.write_text('import pixeltable as pxt\n\nthis is not python\n', encoding='utf-8')
        r = cli('service', 'diff', str(broken), target, check=False)
        assert r.returncode == 1
        assert 'error loading' in r.stderr

        empty = project_dir / 'empty.py'
        empty.write_text('x = 1\n', encoding='utf-8')
        r = cli('service', 'diff', str(empty), target, check=False)
        assert r.returncode == 1
        assert 'no service found' in r.stderr

        # stopping something that is not running is reported, not an error
        r = cli('service', 'stop', 'nosuch', '--json')
        assert [(op['name'], op['status']) for op in r.json] == [('nosuch', 'skipped')]

        # a local service serves the local catalog, so a target naming another one is refused
        for verb in ('diff', 'update'):
            r = cli('service', verb, apps('basic.py'), 'pxt://acme:main/app', check=False)
            assert r.returncode != 0
            assert 'binds its models to the local catalog' in r.stderr, r.stderr
        r = cli('service', 'list', 'pxt://acme:main/app', check=False)
        assert r.returncode != 0
        assert 'binds its models to the local catalog' in r.stderr, r.stderr

    def test_example(self, cli: PxtRunner, make_catalog_path: Callable[[str], str], project_dir: pathlib.Path) -> None:
        """The file `example` writes declares both the tables and the services, and serves."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app_file = project_dir / 'app.py'
        cli('service', 'example', '--out', str(app_file))
        target = make_catalog_path('example')

        deploy(cli, str(app_file), target)
        endpoint = assert_serving(cli, str(app_file), target, 'ingest')['ingest']['endpoint']
        # summary comes from the udf the file defines, computed in the service's own process
        assert _post(endpoint, '/docs', doc_id=1, title='a long enough title', body=None).json() == {
            'title_upper': 'A LONG ENOUGH TITLE',
            'summary': 'a long enoug...',
        }
