import json
import os
import pathlib
import shutil
import socket
import time
from textwrap import dedent
from typing import Any, Callable, Iterator

import httpx
import pytest

import pixeltable as pxt
from pixeltable.config import Config

from ..conftest import SampleFileServer
from ..utils import (
    DatabaseRoot,
    assert_image_bytes,
    fetch_presigned,
    get_audio_files,
    get_documents,
    get_video_files,
    home_bucket_uri,
    skip_test_if_not_installed,
)
from .conftest import (
    BUILD_TIMEOUT,
    BackgroundPxt,
    PxtRunner,
    copy_app_corpus,
    disposable_db_uri,
    read_logs_until,
    write_requirements,
)

_REQUEST_TIMEOUT = 30.0

_SPACY_MODEL = (
    'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/'
    'en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
)


@pytest.fixture(scope='session')
def cloud_db_uri(
    session_cli: PxtRunner, session_project: pathlib.Path, pixeltable_wheel: pathlib.Path
) -> Iterator[str]:
    """A hosted database of this module's own, built from the session's project.

    These tests deploy the project's application files as services, and a service pod runs the image its
    database was built with, so the project reaches the pods only by building one. That is why the database
    the cloud axis normally uses cannot serve here: `pxt db update` would replace the image it runs.

    Session-scoped, since creating a database provisions storage and runs CodeBuild.
    """
    copy_app_corpus(session_project)
    write_requirements(session_project, pixeltable_wheel, 'spacy', _SPACY_MODEL)
    with disposable_db_uri(session_cli, session_project) as uri:
        (session_project / 'pixeltable.toml').write_text(
            f'[[pixeltable.database]]\nname = {json.dumps(uri)}\n', encoding='utf-8'
        )
        # the daemon read the project config when it started
        session_cli('daemon', 'restart', cwd=session_project)
        session_cli('db', 'update', uri, '-f', cwd=session_project, timeout=BUILD_TIMEOUT)
        yield uri


@pytest.fixture
def authenticated_http(db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the caller's API key with every request to a hosted service.

    A local service answers whatever reaches its port. A hosted one sits behind the gateway, which
    authenticates every request and reads the key from X-api-key. Patching the verbs rather than the
    call sites keeps a test's request identical whichever target serves it.
    """
    if db_root.id != 'cloud':
        return
    import httpx

    key = os.environ['PIXELTABLE_API_KEY']
    for verb in ('get', 'post'):
        original = getattr(httpx, verb)

        def _send(url: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            headers = {**(kwargs.pop('headers', None) or {}), 'X-api-key': key}
            return _original(url, *args, headers=headers, **kwargs)

        monkeypatch.setattr(httpx, verb, _send)


@pytest.fixture
def no_hosted_services(db_root: DatabaseRoot) -> Iterator[None]:
    """Leave a hosted database holding no service instances.

    The hosted database outlives every test that runs against it, so a service one test leaves behind is
    still deployed while the next one runs, and a recursive list finds it.
    """
    if db_root.id != 'cloud':
        yield
        return
    from pixeltable.serving.service_manager import get_manager

    def _clear() -> None:
        manager = get_manager(db_root.prefix)
        for instance in manager.list(recursive=True):
            manager.delete(instance)

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def stop_services(cli: PxtRunner) -> Iterator[None]:
    """Leave nothing running: a service outlives the test that started it, and the next one would see it."""
    yield
    for service in cli('service', 'list', '--json').json:
        cli('service', 'stop', f'{service["catalog_path"]}/{service["name"]}'.lstrip('/'))


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
        service = running[name]
        served = httpx.get(f'{service["endpoint"]}/openapi.json', timeout=_REQUEST_TIMEOUT)
        assert served.status_code == 200, served.text
        declared = {route['path'] for route in service['spec']['routes']}
        declared |= set(service['spec']['app_paths'])
        served_paths = set(served.json()['paths'])
        assert declared <= served_paths, (declared, sorted(served_paths))
        assert not any('/_pxt/' in path for path in served_paths), sorted(served_paths)
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


# proxy is excluded because get_manager() hands any non-local path to ServiceManagerProxy, so a
# 'pxt://local:db' target reaches the cloud management API, which knows no org named 'local'.
@pytest.mark.db_roots('local', 'cloud', reason='a proxy-daemon database has no service manager of its own')
@pytest.mark.usefixtures('authenticated_http', 'no_hosted_services')
class TestService:
    def test_config_must_agree(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A service inherits the daemon's config values, so a caller resolving them differently cannot deploy."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), db_root.make_catalog_path('app')
        cli('schema', 'update', app, target)
        differing = {'OPENAI_API_KEY': 'sk-not-the-one-the-daemon-has'}

        # the plan and the service are both refused, and the refusal names the variable and the remedy
        for args in (('diff', app, target), ('update', app, target, '-f')):
            r = cli('service', *args, env_overrides=differing, check=False)
            assert r.returncode == 1, r.stdout
            assert 'OPENAI_API_KEY' in r.stderr
            assert 'pxt daemon restart' in r.stderr
        assert services(cli, target) == {}, 'a refused update started something'

        # the same commands run for a caller whose environment the daemon shares
        deploy(cli, app, target)
        assert_serving(cli, app, target, 'ingest')

    def test_basic(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """The first service: declare, see what is pending, apply it, use it, take it down."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), db_root.make_catalog_path('app')
        cli('schema', 'update', app, target)

        # nothing is deployed yet, and diff says so in its exit status
        r = cli('service', 'diff', app, target, '--json', check=False)
        assert r.returncode == 2
        assert [(s['name'], s['resolution']) for s in r.json['services']] == [('ingest', 'create')]
        assert {op['name'] for op in r.json['services'][0]['ops']} == {
            'POST /docs',
            'POST /preview',
            'POST /docs/update',
            'POST /docs/delete',
        }
        assert services(cli, target) == {}

        cli('service', 'update', app, target, '-f')
        running = assert_serving(cli, app, target, 'ingest')
        assert running['ingest']['app_module'] == 'apps.basic'  # the app corpus sits at <project>/apps
        assert running['ingest']['catalog_path'] == target
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
        assert services(cli, target)['ingest']['pid'] == pid

        cli('service', 'stop', f'{target}/ingest'.lstrip('/'))
        # TODO: assert the instance is listed STOPPED, once a local stop keeps its record like a hosted one does
        # assert services(cli, target)['ingest']['state'] == 'STOPPED'

    def test_iteration(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """Editing the file: an added route is applied by restarting; a changed contract needs a flag."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')
        deploy(cli, apps('basic.py'), target)
        before = assert_serving(cli, apps('basic.py'), target, 'ingest')['ingest']

        # the variant adds a route: additive, because what is already served keeps being served
        r = cli('service', 'diff', apps('basic_added_route.py'), target, '--json', check=False)
        assert [s['resolution'] for s in r.json['services']] == ['update_additive']
        assert [op['op'] for s in r.json['services'] for op in s['ops']] == ['add']

        # a dry run reports the same and changes nothing
        r = cli('service', 'update', apps('basic_added_route.py'), target, '-n', check=False)
        assert r.returncode == 2
        assert services(cli, target)['ingest']['pid'] == before['pid']

        r = cli('service', 'update', apps('basic_added_route.py'), target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['applied']
        after = assert_serving(cli, apps('basic_added_route.py'), target, 'ingest')['ingest']
        assert after['pid'] != before['pid'], 'a changed declaration is applied by replacing the process'
        assert after['port'] == before['port'], 'a restart serves on the port callers were given'
        assert after['endpoint'] == before['endpoint']

        # the added route serves, and so do the routes that were already there
        assert _post(after['endpoint'], '/shout', doc_id=3, title='new route', published=True).json() == {
            'title_upper': 'NEW ROUTE'
        }
        assert _post(after['endpoint'], '/preview', doc_id=4, title='still here', published=True).status_code == 200

        # dropping an output changes a contract callers may be using, so it is refused by default
        r = cli('service', 'update', apps('basic_changed_route.py'), target, '-f', check=False)
        assert r.returncode == 1
        assert '--allow-destructive' in r.stderr
        assert services(cli, target)['ingest']['pid'] == after['pid']

        cli('service', 'update', apps('basic_changed_route.py'), target, '-f', '--allow-destructive')
        assert_serving(cli, apps('basic_changed_route.py'), target, 'ingest')

    def test_custom_app_edits(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """Editing a served application: a path it adds is applied by restarting the service."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')
        # the service takes its name from the module, so iterating on it means editing one file in place
        app_file = pathlib.Path(apps('served_app.py')).with_name('notes_app.py')
        shutil.copy(apps('served_app.py'), app_file)
        deploy(cli, str(app_file), target)

        running = assert_serving(cli, str(app_file), target, 'notes_app')['notes_app']
        assert running['spec']['app_paths'] == ['/notes', '/notes/count']

        # the handlers reach the tables the file's models declare, which the service bound at the target
        assert _post(running['endpoint'], '/notes?note_id=1&text=hello').json() == {'rows': 1}
        counted = httpx.get(f'{running["endpoint"]}/notes/count', timeout=_REQUEST_TIMEOUT)
        assert counted.json() == {'count': 1}, counted.text
        assert pxt.get_table(f'{target}/notes').select().collect()['text_upper'] == ['HELLO']

        # a path the application adds is an addition, as a route added to a router is
        shutil.copy(apps('served_app_added_route.py'), app_file)
        r = cli('service', 'diff', str(app_file), target, '--json', check=False)
        assert [s['resolution'] for s in r.json['services']] == ['update_additive']
        assert [(op['op'], op['name']) for s in r.json['services'] for op in s['ops']] == [('add', '/notes/upper')]

        cli('service', 'update', str(app_file), target, '-f')
        after = assert_serving(cli, str(app_file), target, 'notes_app')['notes_app']
        assert after['pid'] != running['pid'], 'a changed application is applied by replacing the process'
        upper = httpx.get(f'{after["endpoint"]}/notes/upper', timeout=_REQUEST_TIMEOUT)
        assert upper.json() == {'upper': ['HELLO']}, upper.text

    @pytest.mark.db_roots(
        'local', reason='TODO: re-evaluate whether we require a pxt db update for this to work against hosted services'
    )
    def test_source_change(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """An edited udf body restarts the service and the plan names the file; an unimported file does not."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')
        app_file = pathlib.Path(apps('basic.py')).with_name('source_change_app.py')
        shutil.copy(apps('basic.py'), app_file)
        deploy(cli, str(app_file), target)
        before = assert_serving(cli, str(app_file), target, 'ingest')['ingest']

        # a file the application does not import leaves it up to date
        app_file.with_name('unimported_module.py').write_text('unused = 1\n', encoding='utf-8')
        assert cli('service', 'diff', str(app_file), target, '--json').json['in_agreement']

        # change the udf a computed column calls; no route declaration changes with it
        app_file.write_text(
            app_file.read_text(encoding='utf-8').replace(
                "return text if len(text) <= n else f'{text[:n]}...'", 'return text.upper()'
            ),
            encoding='utf-8',
        )
        r = cli('service', 'diff', str(app_file), target, '--json', check=False)
        assert r.returncode == 2
        assert [s['resolution'] for s in r.json['services']] == ['update_additive']
        ops = [op for s in r.json['services'] for op in s['ops']]
        assert [(op['target'], op['op'], op['severity']) for op in ops] == [('project', 'alter', 'additive')]
        assert 'apps/source_change_app.py changed' in ops[0]['description'], ops[0]['description']

        cli('service', 'update', str(app_file), target, '-f')
        after = assert_serving(cli, str(app_file), target, 'ingest')['ingest']
        assert after['pid'] != before['pid'], 'the new source is served by a new process'
        assert _post(after['endpoint'], '/preview', doc_id=1, title='hello', published=True).json() == {
            'summary': 'HELLO'
        }

    def test_prune(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A service the file stopped declaring is stopped and forgotten, and can be started again."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')
        deploy(cli, apps('basic.py'), target)

        # the variant declares the same models under a service of another name, so 'ingest' is an extra
        r = cli('service', 'diff', apps('basic_renamed_service.py'), target, '--json', check=False)
        assert r.json['extras'] == ['ingest']

        r = cli('service', 'prune', apps('basic_renamed_service.py'), target, '-n', check=False)
        assert r.returncode == 2
        assert 'ingest' in services(cli, target)

        r = cli('service', 'prune', apps('basic_renamed_service.py'), target, '-f', '--json')
        assert [(op['name'], op['status']) for op in r.json] == [('ingest', 'applied')]
        assert 'ingest' not in services(cli, target)

        # pruning is not destructive: declaring it again brings it back
        cli('service', 'update', apps('basic.py'), target, '-f')
        assert_serving(cli, apps('basic.py'), target, 'ingest')

    @pytest.mark.db_roots('local', reason='run serves from the calling process, which a hosted pod never does')
    def test_run_foreground(
        self, cli: PxtRunner, cli_bg: Callable[..., BackgroundPxt], apps: Callable[[str], str], db_root: DatabaseRoot
    ) -> None:
        """run serves from the calling process and records nothing; update is the background form."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), db_root.make_catalog_path('app')
        cli('schema', 'update', app, target)

        served = cli_bg('service', 'run', app, target)
        served.wait_until_serving()
        assert (
            _post(served.endpoint, '/docs', doc_id=7, title='foreground', body=None, published=True).status_code == 200
        )
        docs = pxt.get_table(f'{target}/docs')
        assert docs.where(docs.doc_id == 7).count() == 1
        assert services(cli, target) == {}, 'run records nothing'

        served.proc.terminate()
        served.proc.wait(timeout=30)

        # the same file, served the other way
        cli('service', 'update', app, target, '-f')
        assert_serving(cli, app, target, 'ingest')

    def test_blocked_on_database(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A service whose tables do not exist is blocked until the schema is applied."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), db_root.make_catalog_path('app')

        r = cli('service', 'diff', app, target, '--json', check=False)
        assert r.returncode == 2
        assert [s['resolution'] for s in r.json['services']] == ['blocked']
        commands = {op['details'].get('command') for s in r.json['services'] for op in s['ops']}
        assert any(c is not None and 'schema update' in c for c in commands), commands

        r = cli('service', 'update', app, target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['refused']
        assert services(cli, target) == {}

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

    def test_custom_endpoint_model_reference(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot
    ) -> None:
        """A model an application reaches from a handler blocks the service until its table exists."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        # served_app.py declares no route against its model: its handlers reach the table through the model
        app, target = apps('served_app.py'), db_root.make_catalog_path('app')

        r = cli('service', 'diff', app, target, '--json', check=False)
        assert r.returncode == 2
        assert [s['resolution'] for s in r.json['services']] == ['blocked']
        blocked = [op for s in r.json['services'] for op in s['ops'] if op['severity'] == 'blocked']
        assert [op['details']['command'] for op in blocked] == [f'pxt schema update {app} {target}']
        assert "'notes'" in blocked[0]['description'], blocked[0]['description']

        r = cli('service', 'update', app, target, '-f', '--json')
        assert [s['status'] for s in r.json['services']] == ['refused']
        assert services(cli, target) == {}

        # applying the schema unblocks it
        cli('schema', 'update', app, target)
        r = cli('service', 'diff', app, target, '--json', check=False)
        assert [s['resolution'] for s in r.json['services']] == ['create']

    def test_inspection(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """list inspects what a service serves, for every service or for one named by its catalog path."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')
        deploy(cli, apps('media.py'), target)
        assert_serving(cli, apps('media.py'), target, 'clips', 'frames', 'recordings')

        # a listing carries what each service serves, in Pixeltable's terms rather than OpenAPI's: a video
        # arrives as an upload, which a JSON schema would render as an indistinguishable string
        clips = services(cli, target)['clips']['spec']
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

    def test_media(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot, sample_file_server: SampleFileServer
    ) -> None:
        """The routes whose request or response is not JSON: file uploads, a file response, a background job."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('media.py'), db_root.make_catalog_path('app')
        deploy(cli, app, target)
        running = assert_serving(cli, app, target, 'clips', 'frames', 'recordings')

        video = get_video_files()[0]
        with open(video, 'rb') as f:
            resp = httpx.post(
                f'{running["clips"]["endpoint"]}/clips',
                data={'clip_id': 1, 'caption': 'a clip'},
                files={'video': ('clip.mp4', f, 'video/mp4')},
                timeout=_REQUEST_TIMEOUT,
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body['clip_id'] == 1, body
        assert pxt.get_table(f'{target}/frames').count() > 0
        # the persisted poster comes back as a URL: presigned from the home bucket for a hosted table, served by
        # the service for a local one
        if db_root.id == 'cloud':
            assert_image_bytes(fetch_presigned(body['poster'], expires_s=3600, host_suffix='.r2.cloudflarestorage.com'))
        else:
            assert '/media/' in body['poster'], body
            poster = httpx.get(body['poster'], timeout=_REQUEST_TIMEOUT)
            assert poster.status_code == 200, poster.text
            assert_image_bytes(poster.content)

        # a URL the pod can fetch, in place of a local path it cannot read
        video_url = sample_file_server.url(video, db_root)

        # a single media value comes back as the image itself
        resp = _post(running['clips']['endpoint'], '/poster', clip_id=2, video=video_url)
        assert resp.headers['content-type'].startswith('image/'), resp.headers
        assert_image_bytes(resp.content)

        # a route over the iterator view answers with a row per frame, media rendered as urls
        rows = _post(running['frames']['endpoint'], '/frames', clip_id=3, video=video_url).json()
        assert len(rows) > 1, rows
        assert all(row['thumb'].startswith('http') for row in rows), rows[0]
        thumb = httpx.get(rows[0]['thumb'], timeout=_REQUEST_TIMEOUT)
        assert thumb.status_code == 200, thumb.text
        assert_image_bytes(thumb.content)

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
        # the uploaded audio is persisted in the home bucket for a hosted table, in the media dir for a local one
        audio_url = recordings.select(recordings.audio.fileurl).collect()['audio_fileurl'][0]
        expected_prefix = f'{home_bucket_uri(db_root.prefix)}/' if db_root.id == 'cloud' else 'file://'
        assert audio_url.startswith(expected_prefix), audio_url

        # stopping one service of a file leaves the others serving
        cli('service', 'stop', f'{target}/frames')
        listed = services(cli, target)
        # TODO: assert frames is listed STOPPED, once a local stop keeps its record like a hosted one does
        assert listed['clips']['state'] == 'AVAILABLE'
        assert _post(running['clips']['endpoint'], '/poster', clip_id=4, video=video_url).status_code == 200

    def test_search(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """An iterator view and an embedding index over the column the iterator produces."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        skip_test_if_not_installed('spacy')  # the view's iterator splits on sentences
        app, target = apps('search.py'), db_root.make_catalog_path('app')
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

    def test_custom_app(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A file supplying its own application: the router it includes is part of it, not a second service."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('custom.py'), db_root.make_catalog_path('app')
        cli('schema', 'update', app, target)

        r = cli('service', 'diff', app, target, '--json', check=False)
        assert [(s['name'], s['kind'], s['resolution']) for s in r.json['services']] == [('custom', 'custom', 'create')]

        cli('service', 'update', app, target, '-f')
        running = assert_serving(cli, app, target, 'custom')['custom']
        # the spec holds the route declarations of the router, and the paths the application publishes
        assert [(r['method'], r['path'], r['route_type']) for r in running['spec']['routes']] == [
            ('POST', '/notes', 'insert')
        ]
        assert running['spec']['app_paths'] == ['/hand-written']

        # the application serves both the route written by hand and the routes of the router it includes
        endpoint = running['endpoint']
        assert httpx.get(f'{endpoint}/hand-written', timeout=_REQUEST_TIMEOUT).json() == {'written': 'by hand'}
        assert _post(endpoint, '/notes', note_id=1, text='hello').status_code == 200
        assert pxt.get_table(f'{target}/notes').select().collect()['text_upper'] == ['HELLO']

    def test_custom_app_prefixed_router(
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot
    ) -> None:
        """A router included under a prefix is part of the application, and its paths are not app_paths."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('prefixed_app.py'), db_root.make_catalog_path('app')
        cli('schema', 'update', app, target)
        cli('service', 'update', app, target, '-f')

        running = services(cli, target)['prefixed_app']
        # the prefix is part of what the router serves, so the spec records it in the path
        assert [(r['method'], r['path']) for r in running['spec']['routes']] == [('POST', '/v1/notes')]
        assert running['spec']['app_paths'] == ['/hand-written']

        endpoint = running['endpoint']
        assert httpx.get(f'{endpoint}/hand-written', timeout=_REQUEST_TIMEOUT).json() == {'written': 'by hand'}
        assert _post(endpoint, '/v1/notes', note_id=1, text='hello').status_code == 200
        assert pxt.get_table(f'{target}/notes').select().collect()['text_upper'] == ['HELLO']

    def test_addressing(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """One name at two targets: a bare name is ambiguous, a qualified one is not."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app = apps('basic.py')
        first, second = db_root.make_catalog_path('one'), db_root.make_catalog_path('two')
        deploy(cli, app, first)
        deploy(cli, app, second)
        assert_serving(cli, app, first, 'ingest')
        assert_serving(cli, app, second, 'ingest')

        # a listing narrows to one target
        assert sorted(services(cli, first)) == ['ingest']

        if db_root.id != 'cloud':
            # a bare name reaches local services only: a project config may name several databases, so an
            # un-targeted command has no one hosted database to read
            assert len(cli('service', 'list', '--json').json) == 2

            # the same name at two targets cannot be stopped by name alone
            r = cli('service', 'stop', 'ingest', check=False)
            assert r.returncode == 1
            assert 'ambiguous' in r.stderr
            assert f'{first}/ingest' in r.stderr and f'{second}/ingest' in r.stderr

        # the catalog path says which one
        cli('service', 'stop', f'{first}/ingest')
        # TODO: assert first's instance is gone or listed STOPPED, once a local stop keeps its record like
        # a hosted one does
        assert_serving(cli, app, second, 'ingest')

    @pytest.mark.db_roots('cloud', reason='a local service logs to a file, which test_logs_errors checks')
    def test_logs(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """A hosted service's log holds the requests it served."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app = apps('basic.py')
        plain, nested = db_root.make_catalog_path('app'), db_root.make_catalog_path('outer/inner')
        deploy(cli, app, plain)
        deploy(cli, app, nested)
        running = assert_serving(cli, app, plain, 'ingest')
        assert_serving(cli, app, nested, 'ingest')

        # the request line is logged under the router's name, the probes are dropped unless asked for, and the
        # text output is the lines
        _post(running['ingest']['endpoint'], '/preview', doc_id=1, title='logged', published=False)
        records = read_logs_until(cli, 'service', 'logs', f'{plain}/ingest', contains='POST /ingest/preview')
        assert not any('GET /ingest/health' in rec['line'] for rec in records)
        read_logs_until(cli, 'service', 'logs', f'{plain}/ingest', '--include-health', contains='GET /ingest/health')
        assert 'POST /ingest/preview' in cli('service', 'logs', f'{plain}/ingest').stdout

        # two services with the same name at different paths have separate logs
        nested_records = cli('service', 'logs', f'{nested}/ingest', '--json').json
        assert not any('POST /ingest/preview' in rec['line'] for rec in nested_records), nested_records[-5:]
        marker = '/only-nested'
        endpoint = services(cli, nested)['ingest']['endpoint']
        assert httpx.get(f'{endpoint}{marker}', timeout=_REQUEST_TIMEOUT).status_code == 404
        read_logs_until(cli, 'service', 'logs', f'{nested}/ingest', contains=marker)

        # stopping removes the pod; the lines it wrote stay readable
        cli('service', 'stop', f'{plain}/ingest')
        stopped = cli('service', 'logs', f'{plain}/ingest', '--json').json
        assert any('POST /ingest/preview' in rec['line'] for rec in stopped), stopped[-5:]

        # a service that fails during startup leaves its traceback in the log
        failing = db_root.make_catalog_path('failing')
        pxt.create_dir(failing)
        result = cli('service', 'update', apps('failed_startup.py'), failing, '-f', check=False, timeout=600)
        assert result.returncode == 1, result.stdout
        records = read_logs_until(
            cli, 'service', 'logs', f'{failing}/failed_startup', contains='RuntimeError: intentional startup failure'
        )
        assert any('Traceback (most recent call last)' in rec['line'] for rec in records), records[-10:]
        assert records == sorted(records, key=lambda rec: rec['ts_ms'])

    def test_logs_errors(self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot) -> None:
        """What `service logs` refuses: bad options, a name nothing serves, and a local service's file log."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app, target = apps('basic.py'), db_root.make_catalog_path('app')
        deploy(cli, app, target)
        assert_serving(cli, app, target, 'ingest')

        # the daemon validates --since and --tail before resolving the service
        r = cli('service', 'logs', f'{target}/ingest', '--since', 'bogus', check=False)
        assert r.returncode == 1 and 'must be a duration' in r.stderr, r.stderr
        r = cli('service', 'logs', f'{target}/ingest', '--tail', '50000', check=False)
        assert r.returncode == 1 and "'limit' must be <= 10000" in r.stderr, r.stderr
        r = cli('service', 'logs', f'{target}/nosuch', check=False)
        assert r.returncode == 1 and 'No service' in r.stderr, r.stderr

        if db_root.id == 'cloud':
            # a bare name reaches local services only
            r = cli('service', 'logs', 'ingest', check=False)
            assert r.returncode == 1 and "No service 'ingest' is running" in r.stderr, r.stderr
            return

        # a local service logs to a file, and the error names it, for a qualified name and a bare one alike
        log_file = Config.get().home.joinpath('logs', 'services', target, 'ingest.log')
        for name in (f'{target}/ingest', 'ingest'):
            r = cli('service', 'logs', name, check=False)
            assert r.returncode == 1, r.stderr
            assert f'not supported; the log is at {log_file}' in r.stderr, r.stderr
        assert log_file.is_file()

        # the same name at a second target makes the bare name ambiguous
        other = db_root.make_catalog_path('other')
        deploy(cli, app, other)
        r = cli('service', 'logs', 'ingest', check=False)
        assert r.returncode == 1 and 'ambiguous' in r.stderr, r.stderr
        assert f'{target}/ingest' in r.stderr and f'{other}/ingest' in r.stderr

    @pytest.mark.db_roots('local', reason='drives the local proxy daemon directly, so the target axis adds nothing')
    def test_proxy_daemon_project_handoff(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A running proxy daemon is reused for its own project, and replaced for another."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        import httpx

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

    @pytest.mark.db_roots('local', reason='check reads no catalog, so the target axis adds nothing')
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
        assert 'at least one FastAPIRouter' in r.stderr

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
        self, cli: PxtRunner, apps: Callable[[str], str], db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """What the verbs do with a file that is missing, unimportable, or declares no service."""
        target = db_root.make_catalog_path('app')

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
        assert 'at least one FastAPIRouter' in r.stderr

        # stopping something that is not running is reported, not an error
        r = cli('service', 'stop', 'nosuch', '--json')
        assert [(op['name'], op['status']) for op in r.json] == [('nosuch', 'skipped')]

        # a hosted target requires a database
        r = cli('service', 'list', 'pxt://acme', check=False)
        assert r.returncode != 0
        assert 'names no database' in r.stderr, r.stderr

        # 'run' doesn't work for hosted uris
        r = cli('service', 'run', apps('basic.py'), 'pxt://acme:main/app', check=False)
        assert r.returncode != 0
        assert 'serves from this process' in r.stderr, r.stderr

    def test_named_service(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """Name one service of a file that defines two, and pin its port."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        target = db_root.make_catalog_path('app')

        # --port names one port, and two_services.py defines two
        two = project_dir / 'two_services.py'
        two.write_text(
            dedent("""
            import pixeltable as pxt
            from pixeltable.serving import FastAPIRouter

            TableModel = pxt.model_base()


            class Notes(TableModel, name='notes'):
                s: pxt.String


            first = FastAPIRouter(name='first')
            first.add_insert_route(Notes, path='/a', inputs=[Notes.s], outputs=[Notes.s])
            second = FastAPIRouter(name='second')
            second.add_insert_route(Notes, path='/b', inputs=[Notes.s], outputs=[Notes.s])
            """),
            encoding='utf-8',
        )
        cli('schema', 'update', str(two), target)
        r = cli('service', 'update', str(two), target, '-f', '--port', '8123', check=False)
        assert r.returncode == 1
        assert '--port names one port' in r.stderr, r.stderr
        assert services(cli, target) == {}, 'a refused update started nothing'

        r = cli('service', 'update', str(two), target, 'third', '-f', check=False)
        assert r.returncode == 1
        assert "no service named 'third'" in r.stderr, r.stderr
        assert services(cli, target) == {}, 'a refused update started nothing'

        # naming one service leaves the other alone, and makes --port unambiguous
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            free_port = probe.getsockname()[1]
        if db_root.id == 'cloud':
            # naming one service leaves only the hosted rule to refuse --port: a hosted service answers
            # on its own hostname
            r = cli('service', 'update', str(two), target, 'second', '-f', '--port', str(free_port), check=False)
            assert r.returncode == 1
            assert 'not a port' in r.stderr, r.stderr

        port_args = [] if db_root.id == 'cloud' else ['--port', str(free_port)]
        r = cli('service', 'update', str(two), target, 'second', '-f', *port_args, '--json')
        assert [d['name'] for d in r.json['services']] == ['second'], r.json
        running = services(cli, target)
        assert sorted(running) == ['second'], running
        if db_root.id != 'cloud':
            assert running['second']['port'] == free_port
            assert running['second']['endpoint'].endswith(f':{free_port}')

        # diff takes the same name, and reports only that service
        r = cli('service', 'diff', str(two), target, 'second', '--json')
        assert [d['name'] for d in r.json['services']] == ['second'], r.json
        r = cli('service', 'diff', str(two), target, '--json', check=False)
        assert r.returncode == 2, r.stdout
        assert sorted(d['name'] for d in r.json['services']) == ['first', 'second'], r.json
        r = cli('service', 'diff', str(two), target, 'third', check=False)
        assert r.returncode == 1
        assert "no service named 'third'" in r.stderr, r.stderr

    @pytest.mark.db_roots(
        'local', reason='TODO: re-evaluate whether we require a pxt db update for this to work against hosted services'
    )
    def test_example(self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path) -> None:
        """The file `example` writes declares both the tables and the services, and serves."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app_file = project_dir / 'app.py'
        cli('service', 'example', '--out', str(app_file))
        target = db_root.make_catalog_path('example')

        deploy(cli, str(app_file), target)
        endpoint = assert_serving(cli, str(app_file), target, 'ingest')['ingest']['endpoint']
        # summary comes from the udf the file defines, computed in the service's own process
        assert _post(endpoint, '/docs', doc_id=1, title='a long enough title', body=None).json() == {
            'title_upper': 'A LONG ENOUGH TITLE',
            'summary': 'a long enoug...',
        }
