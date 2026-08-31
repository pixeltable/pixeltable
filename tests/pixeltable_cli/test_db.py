"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, so the module is skipped unless the cloud
environment is configured, and it is marked expensive: applying what an entry declares rebuilds an image,
which takes minutes. They run against the session's hosted database, the one the cloud catalog tests use.
"""

import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
import uuid
from typing import Any

import httpx
import pytest

from .conftest import PxtRunner

pytestmark = [
    pytest.mark.remote_api,
    pytest.mark.expensive,
    pytest.mark.db_roots('local', reason='pxt db acts on a hosted database, not on the catalog a test runs against'),
]

# the exit statuses `pxt db diff` and `pxt db update` document
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2

_REQUEST_TIMEOUT = 30.0

_APP_FILE = 'basic.py'  # the corpus file the project holds, and the pod serves


@pytest.fixture(autouse=True)
def hosted_environment() -> None:
    """Skip the test unless the session names a hosted database to act on."""
    if os.environ.get('PXTTEST_CLOUD_DB_URI') is None:
        pytest.skip('PXTTEST_CLOUD_DB_URI is not set.')


@pytest.fixture
def hosted_db() -> str:
    """The hosted database these tests act on, which is the one the cloud catalog tests read."""
    uri = os.environ.get('PXTTEST_CLOUD_DB_URI')
    assert uri is not None  # hosted_environment() skipped the test otherwise
    return uri


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A project of the test's own, holding an application file from the corpus and a lockfile.

    Outside the session's project, so that the archive holds these files and nothing else: the client hands
    the daemon whichever project the working directory establishes.
    """
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'pixeltable.toml').write_text('', encoding='utf-8')
    shutil.copy(pathlib.Path(__file__).parent / 'apps' / _APP_FILE, root / _APP_FILE)
    (root / 'requirements.txt').write_text('pixeltable\n', encoding='utf-8')
    return root


def edit_app(project: pathlib.Path, comment: str) -> None:
    """Append a comment to the project's application file, changing the file and nothing it declares."""
    with open(project / _APP_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\n# {comment}\n')


@pytest.fixture
def current_db(cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> str:
    """A hosted database holding this project: where most scenarios below start."""
    create_project_config(cli, project, hosted_db)
    db_update(cli, project, hosted_db)
    assert_in_agreement(cli, project, hosted_db)
    return hosted_db


def create_project_config(cli: PxtRunner, project: pathlib.Path, db_uri: str, **settings: Any) -> None:
    """Write the project's entry for db_uri with these settings, and hand the daemon the new project."""
    lines = ['[[pixeltable.database]]', f'name = {json.dumps(db_uri)}']
    for key, value in settings.items():
        if isinstance(value, dict):
            lines += [f'{key}.{name} = {json.dumps(bound)}' for name, bound in value.items()]
        else:
            lines.append(f'{key} = {json.dumps(value)}')
    (project / 'pixeltable.toml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # the daemon read the project config when it started
    cli('daemon', 'restart', cwd=project)


def db_diff(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db diff` reports, its exit status under 'returncode'."""
    r = cli('db', 'diff', db_uri, '--json', cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def db_update(cli: PxtRunner, project: pathlib.Path, db_uri: str, *flags: str) -> dict[str, Any]:
    """What `pxt db update` applied, its exit status under 'returncode'."""
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def db_status(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db status` reports."""
    return cli('db', 'status', db_uri, '--json', cwd=project).json


def schema_update(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str) -> None:
    """Create what app_file's models declare at db_uri."""
    cli('schema', 'update', app_file, db_uri, '-f', cwd=project)


def service_update(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str) -> None:
    """Serve what app_file declares at db_uri."""
    cli('service', 'update', app_file, db_uri, '-f', cwd=project)


def service_diff(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str) -> dict[str, Any]:
    """What `pxt service diff` reports for app_file at db_uri, its exit status under 'returncode'."""
    r = cli('service', 'diff', app_file, db_uri, '--json', cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def service_list(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, dict[str, Any]]:
    """The instances `pxt service list` reports at db_uri, keyed by name."""
    return {i['name']: i for i in cli('service', 'list', db_uri, '--json', cwd=project).json}


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


def assert_in_agreement(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> None:
    plan = db_diff(cli, project, db_uri)
    assert plan['in_agreement'], plan['ops']
    assert plan['returncode'] == EXIT_IN_AGREEMENT
    assert plan['ops'] == []


class TestDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A database the control plane does not hold is planned as a create, and the update makes it."""
        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:16]}'
        create_project_config(cli, project, absent)

        plan = db_diff(cli, project, absent)
        assert plan['resolution'] == 'create'
        assert not plan['exists']
        assert plan['state'] is None
        assert plan['ops'] == []
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        try:
            applied = db_update(cli, project, absent)
            assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
            assert cli('db', 'status', absent, '--json', cwd=project).json['state'] == 'AVAILABLE'
            # the database now holds this project, so a second look has nothing to do
            assert_in_agreement(cli, project, absent)
        finally:
            cli('db', 'delete', absent, cwd=project, check=False)

    def test_update_both_artifacts(self, cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> None:
        """A database whose project is not this one needs the image and the archive both."""
        create_project_config(cli, project, hosted_db)

        plan = db_diff(cli, project, hosted_db)
        assert plan['resolution'] == 'update_additive'
        assert [op['name'] for op in get_target_ops(plan, 'image')] == ['image']
        assert [op['name'] for op in get_target_ops(plan, 'archive')] == ['project']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, hosted_db)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert (applied['in_agreement'], applied['returncode']) == (True, EXIT_IN_AGREEMENT)
        assert_in_agreement(cli, project, hosted_db)

    def test_status_list(self, cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> None:
        name = hosted_db.rsplit(':', 1)[-1]
        status = db_status(cli, project, hosted_db)
        assert status['state'] == 'AVAILABLE', status
        listed = cli('db', 'list', 'pxt://pixeltable', '--json', cwd=project).json
        assert name in [entry['db_name'] for entry in listed], listed

    def test_source_edit(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """An edit to a source file moves the archive alone: the environment it runs in is unchanged."""
        edit_app(project, 'an edit that changes no dependency')

        plan = db_diff(cli, project, current_db)
        assert get_target_ops(plan, 'image') == []
        [op] = get_target_ops(plan, 'archive')
        assert op['severity'] == 'additive'
        assert f'{_APP_FILE} changed' in op['description'], op['description']
        assert not plan['summary']['rebuild']

        applied = db_update(cli, project, current_db)
        assert [op['status'] for op in applied['ops']] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_lockfile_edit(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """The lockfile is in both manifests, so editing it moves the image and the archive alike."""
        (project / 'requirements.txt').write_text('pixeltable\ntqdm\n', encoding='utf-8')

        plan = db_diff(cli, project, current_db)
        [image_op] = get_target_ops(plan, 'image')
        [archive_op] = get_target_ops(plan, 'archive')
        assert 'requirements.txt changed' in image_op['description'], image_op['description']
        assert 'requirements.txt changed' in archive_op['description'], archive_op['description']

        applied = db_update(cli, project, current_db)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert_in_agreement(cli, project, current_db)

    def test_excluded_files(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A file the entry excludes is not part of the project, so writing it changes nothing."""
        create_project_config(cli, project, current_db, exclude=['notes/**'])
        assert_in_agreement(cli, project, current_db)

        (project / 'notes').mkdir()
        (project / 'notes' / 'scratch.txt').write_text('not part of the project\n', encoding='utf-8')
        assert_in_agreement(cli, project, current_db)

        edit_app(project, 'an edit to a file the entry selects')
        assert get_target_ops(db_diff(cli, project, current_db), 'archive') != []

    def test_secrets(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PXTTEST_SECRET', 'a value the database holds')
        create_project_config(cli, project, current_db, secrets={'pxttest_secret': 'env:PXTTEST_SECRET'})

        [added] = get_target_ops(db_diff(cli, project, current_db), 'secret')
        assert (added['name'], added['op'], added['destructive']) == ('pxttest_secret', 'add', False)
        assert [op['status'] for op in get_target_ops(db_update(cli, project, current_db), 'secret')] == ['applied']
        assert_in_agreement(cli, project, current_db)

        create_project_config(cli, project, current_db)
        [dropped] = get_target_ops(db_diff(cli, project, current_db), 'secret')
        assert (dropped['op'], dropped['destructive']) == ('drop', True)

        refused = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert refused.returncode == EXIT_ERROR
        assert '--allow-destructive' in refused.stderr, refused.stderr
        assert get_target_ops(db_diff(cli, project, current_db), 'secret') != []

        applied = db_update(cli, project, current_db, '--allow-destructive')
        assert [op['status'] for op in get_target_ops(applied, 'secret')] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_unbound_secret(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A secret naming an environment variable that is not set stops the update."""
        monkeypatch.delenv('PXTTEST_UNSET', raising=False)
        create_project_config(cli, project, current_db, secrets={'pxttest_unset': 'env:PXTTEST_UNSET'})

        r = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR
        assert 'PXTTEST_UNSET' in r.stderr, r.stderr

        # a secret declared as its value names no environment variable, and the file would hold the value
        create_project_config(cli, project, current_db, secrets={'pxttest_literal': 'sk-in-the-file'})
        r = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR
        assert "write 'env:NAME'" in r.stderr, r.stderr

    def test_capacity(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        running_on = db_status(cli, project, current_db)['cpu']
        create_project_config(cli, project, current_db, cpu=running_on + 1)

        [op] = get_target_ops(db_diff(cli, project, current_db), 'capacity')
        assert op['name'] == 'cpu'
        assert not op['destructive']
        assert str(running_on + 1) in op['description'], op['description']

        assert [op['status'] for op in get_target_ops(db_update(cli, project, current_db), 'capacity')] == ['applied']
        assert_in_agreement(cli, project, current_db)

        # taking capacity away is destructive, so it needs the flag that permits it
        create_project_config(cli, project, current_db, cpu=running_on)
        refused = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert refused.returncode == EXIT_ERROR
        assert '--allow-destructive' in refused.stderr, refused.stderr
        assert [
            op['status']
            for op in get_target_ops(db_update(cli, project, current_db, '--allow-destructive'), 'capacity')
        ] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_dry_run(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        edit_app(project, 'an edit no update applies')

        planned = db_update(cli, project, current_db, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']
        assert get_target_ops(db_diff(cli, project, current_db), 'archive') != []

    def test_build_image(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """build-image sends and builds whatever the project holds, in agreement or not."""
        ops = cli('db', 'build-image', current_db, '--json', cwd=project).json
        assert sorted(op['target'] for op in ops) == ['archive', 'image']
        assert all(op['status'] == 'applied' for op in ops), ops
        assert_in_agreement(cli, project, current_db)

    def test_errors(self, cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> None:
        create_project_config(cli, project, hosted_db)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:16]}'
        create_project_config(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr


class TestPod:
    """What a service pod does with the project its database holds."""

    def test_service_diff_before_update(self, cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> None:
        """A database `pxt db update` has not run for can serve nothing, whatever the file declares."""
        create_project_config(cli, project, hosted_db, include_only=['no_such_file'])
        app_file = str(project / _APP_FILE)
        [blocked] = service_diff(cli, project, app_file, hosted_db)['services']
        assert blocked['resolution'] == 'blocked'
        [op] = [op for op in blocked['ops'] if op['target'] == 'project']
        assert f'pxt db update {hosted_db}' in op['description'], op['description']

    def test_pod_serves_project(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, tmp_path: pathlib.Path
    ) -> None:
        app_file = str(project / _APP_FILE)
        schema_update(cli, project, app_file, current_db)
        service_update(cli, project, app_file, current_db)

        unpacked = tmp_path / 'app'
        port = _free_port()
        pod = _run_pod(current_db, unpacked, '--base-path', current_db, '--host', '127.0.0.1', '--port', str(port))
        try:
            _wait_until_serving(f'http://127.0.0.1:{port}')
            assert (unpacked / _APP_FILE).read_text() == (project / _APP_FILE).read_text()
            assert (unpacked / 'requirements.txt').is_file()
            served = httpx.get(f'http://127.0.0.1:{port}/openapi.json', timeout=_REQUEST_TIMEOUT)
            assert '/notes' in served.json()['paths'], served.json()['paths']
        finally:
            pod.terminate()
            pod.wait(timeout=30)

    def test_pod_refuses_digest_mismatch(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, tmp_path: pathlib.Path
    ) -> None:
        """A pod is told which project to run, and runs nothing else."""
        unpacked = tmp_path / 'app'
        digest = hashlib.sha256(b'a digest no project has').hexdigest()
        pod = _run_pod(current_db, unpacked, '--digest', digest, capture=True)
        stderr = pod.communicate(timeout=300)[1]

        assert pod.returncode != 0
        assert digest in stderr, stderr
        assert not unpacked.exists()

    def test_service_diff_blocked(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A hosted service runs the project its database was given, so an edit here is `pxt db update`."""
        app_file = str(project / _APP_FILE)
        schema_update(cli, project, app_file, current_db)
        service_update(cli, project, app_file, current_db)

        edit_app(project, 'an edit the database has not been given')
        [blocked] = service_diff(cli, project, app_file, current_db)['services']
        assert blocked['resolution'] == 'blocked'
        [op] = [op for op in blocked['ops'] if op['target'] == 'project']
        assert f'pxt db update {current_db}' in op['description'], op['description']

        db_update(cli, project, current_db)
        [reconciled] = service_diff(cli, project, app_file, current_db)['services']
        assert reconciled['resolution'] != 'blocked', reconciled['ops']

    def test_service_lifecycle(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """Start, restart on a changed declaration, list, stop and forget a hosted instance."""
        app_file = str(project / _APP_FILE)
        schema_update(cli, project, app_file, current_db)
        service_update(cli, project, app_file, current_db)

        instance = service_list(cli, project, current_db)['ingest']
        assert instance['state'] == 'AVAILABLE', instance
        assert instance['catalog_path'] == current_db

        # a route the file adds is applied by restarting the instance
        edit_app(project, "ingest.add_delete_route(Docs, path='/docs/delete')")
        db_update(cli, project, current_db)
        [added] = service_diff(cli, project, app_file, current_db)['services']
        assert added['resolution'] == 'update_additive', added['ops']
        service_update(cli, project, app_file, current_db)
        assert service_diff(cli, project, app_file, current_db)['in_agreement']

        # stopping keeps the registration, so an update starts it again
        cli('service', 'stop', f'{current_db}/ingest', cwd=project)
        stopped = service_list(cli, project, current_db)['ingest']
        assert stopped['state'] == 'STOPPED', stopped
        assert not service_diff(cli, project, app_file, current_db)['in_agreement']

        # prune forgets what the file does not declare
        cli('service', 'prune', app_file, current_db, '-f', cwd=project)


def _run_pod(db_uri: str, project_dir: pathlib.Path, *flags: str, capture: bool = False) -> subprocess.Popen:
    """Start a service pod for db_uri, serving the 'ingest' service of the app.py the archive holds."""
    argv = (
        'python',
        '-m',
        'pixeltable.serving.pod_runner',
        '--db',
        db_uri,
        '--app-file',
        _APP_FILE,
        '--name',
        'ingest',
        '--project-dir',
        str(project_dir),
        *flags,
    )
    pipe = subprocess.PIPE if capture else None
    return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=pipe, stderr=pipe, text=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _wait_until_serving(endpoint: str, timeout: float = 300.0) -> None:
    """Block until endpoint answers, or fail once timeout seconds have passed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f'{endpoint}/openapi.json', timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise AssertionError(f'nothing was serving on {endpoint} within {timeout:.0f}s')
