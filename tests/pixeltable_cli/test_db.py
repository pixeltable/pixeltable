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
    pytest.mark.local('pxt db acts on a hosted database, not on the catalog the tests run against'),
]

_CLOUD_ENV_VARS = ('PIXELTABLE_API_KEY', 'PIXELTABLE_API_URL', 'PIXELTABLE_CLOUD_HOST')

# the exit statuses `pxt db diff` and `pxt db update` document
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2

_REQUEST_TIMEOUT = 30.0

_APP_FILE = 'basic.py'  # the corpus file the project holds, and the pod serves


@pytest.fixture(autouse=True)
def cloud_environment() -> None:
    """Skip the test unless the environment names a control plane."""
    for name in _CLOUD_ENV_VARS:
        if os.environ.get(name) is None:
            pytest.skip(f'{name} is not set.')


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
def current_db(cli: PxtRunner, project: pathlib.Path, cloud_db_base_uri: str) -> str:
    """A hosted database holding this project: where most scenarios below start."""
    declare(cli, project, cloud_db_base_uri)
    update(cli, project, cloud_db_base_uri)
    assert_in_agreement(cli, project, cloud_db_base_uri)
    return cloud_db_base_uri


def declare(cli: PxtRunner, project: pathlib.Path, db_uri: str, **settings: Any) -> None:
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


def diff(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db diff` reports, its exit status under 'returncode'."""
    r = cli('db', 'diff', db_uri, '--json', cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def update(cli: PxtRunner, project: pathlib.Path, db_uri: str, *flags: str) -> dict[str, Any]:
    """What `pxt db update` applied, its exit status under 'returncode'."""
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def ops_on(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity, secret or placement."""
    return [op for op in plan['ops'] if op['target'] == target]


def assert_in_agreement(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> None:
    plan = diff(cli, project, db_uri)
    assert plan['in_agreement'], plan['ops']
    assert plan['returncode'] == EXIT_IN_AGREEMENT
    assert plan['ops'] == []


class TestDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A database the control plane does not hold is a plan to create it, with nothing to compare."""
        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:16]}'
        declare(cli, project, absent)

        plan = diff(cli, project, absent)
        assert plan['resolution'] == 'create'
        assert not plan['exists']
        assert plan['state'] is None
        assert plan['ops'] == []
        assert plan['returncode'] == EXIT_CHANGES_PENDING

    def test_update_sends_both_artifacts(self, cli: PxtRunner, project: pathlib.Path, cloud_db_base_uri: str) -> None:
        """A database whose project is not this one needs the image and the archive both."""
        declare(cli, project, cloud_db_base_uri)

        plan = diff(cli, project, cloud_db_base_uri)
        assert plan['resolution'] == 'update_additive'
        assert [op['name'] for op in ops_on(plan, 'image')] == ['image']
        assert [op['name'] for op in ops_on(plan, 'archive')] == ['project']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = update(cli, project, cloud_db_base_uri)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert_in_agreement(cli, project, cloud_db_base_uri)

    def test_status_and_list_report_the_database(self, cli: PxtRunner, cloud_db_base_uri: str) -> None:
        name = cloud_db_base_uri.rsplit(':', 1)[-1]
        status = cli('db', 'status', cloud_db_base_uri, '--json').json
        assert status['state'] == 'AVAILABLE', status
        listed = cli('db', 'list', 'pxt://pixeltable', '--json').json
        assert name in [entry['db_name'] for entry in listed], listed

    def test_source_edit_sends_the_project(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """An edit to a source file moves the archive alone: the environment it runs in is unchanged."""
        edit_app(project, 'an edit that changes no dependency')

        plan = diff(cli, project, current_db)
        assert ops_on(plan, 'image') == []
        [op] = ops_on(plan, 'archive')
        assert op['severity'] == 'additive'
        assert f'{_APP_FILE} changed' in op['description'], op['description']
        assert not plan['summary']['rebuild']

        applied = update(cli, project, current_db)
        assert [op['status'] for op in applied['ops']] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_lockfile_edit_rebuilds_the_image(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """The lockfile is in both manifests, so editing it moves the image and the archive alike."""
        (project / 'requirements.txt').write_text('pixeltable\ntqdm\n', encoding='utf-8')

        plan = diff(cli, project, current_db)
        [image_op] = ops_on(plan, 'image')
        [archive_op] = ops_on(plan, 'archive')
        assert 'requirements.txt changed' in image_op['description'], image_op['description']
        assert 'requirements.txt changed' in archive_op['description'], archive_op['description']

        applied = update(cli, project, current_db)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert_in_agreement(cli, project, current_db)

    def test_excluded_files_are_not_sent(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A file the entry excludes is not part of the project, so writing it changes nothing."""
        declare(cli, project, current_db, exclude=['notes/**'])
        assert_in_agreement(cli, project, current_db)

        (project / 'notes').mkdir()
        (project / 'notes' / 'scratch.txt').write_text('not part of the project\n', encoding='utf-8')
        assert_in_agreement(cli, project, current_db)

        edit_app(project, 'an edit to a file the entry selects')
        assert ops_on(diff(cli, project, current_db), 'archive') != []

    def test_secret_is_set_and_deleting_it_is_destructive(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PXTTEST_SECRET', 'a value the database holds')
        declare(cli, project, current_db, secrets={'pxttest_secret': 'env:PXTTEST_SECRET'})

        [added] = ops_on(diff(cli, project, current_db), 'secret')
        assert (added['name'], added['op'], added['destructive']) == ('pxttest_secret', 'add', False)
        assert [op['status'] for op in ops_on(update(cli, project, current_db), 'secret')] == ['applied']
        assert_in_agreement(cli, project, current_db)

        declare(cli, project, current_db)
        [dropped] = ops_on(diff(cli, project, current_db), 'secret')
        assert (dropped['op'], dropped['destructive']) == ('drop', True)

        refused = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert refused.returncode == EXIT_ERROR
        assert '--allow-destructive' in refused.stderr, refused.stderr
        assert ops_on(diff(cli, project, current_db), 'secret') != []

        applied = update(cli, project, current_db, '--allow-destructive')
        assert [op['status'] for op in ops_on(applied, 'secret')] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_unbound_secret_is_refused(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A secret naming an environment variable that is not set stops the update."""
        monkeypatch.delenv('PXTTEST_UNSET', raising=False)
        declare(cli, project, current_db, secrets={'pxttest_unset': 'env:PXTTEST_UNSET'})

        r = cli('db', 'update', current_db, '-f', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR
        assert 'PXTTEST_UNSET' in r.stderr, r.stderr

    def test_capacity_is_applied(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        running_on = cli('db', 'status', current_db, '--json').json['cpu']
        declare(cli, project, current_db, cpu=running_on + 1)

        [op] = ops_on(diff(cli, project, current_db), 'capacity')
        assert op['name'] == 'cpu'
        assert not op['destructive']
        assert str(running_on + 1) in op['description'], op['description']

        assert [op['status'] for op in ops_on(update(cli, project, current_db), 'capacity')] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_placement_cannot_be_changed(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """Where a database runs is fixed when it is created, so a differing entry reports what no update applies."""
        running_in = cli('db', 'status', current_db, '--json').json['region']
        declare(cli, project, current_db, region=f'{running_in}-2')

        plan = diff(cli, project, current_db)
        assert plan['resolution'] == 'unsupported'
        [op] = ops_on(plan, 'placement')
        assert op['severity'] == 'unsupported'
        assert plan['summary']['unsupported'] == 1

        applied = update(cli, project, current_db)
        assert [op['status'] for op in ops_on(applied, 'placement')] == [None]
        assert ops_on(diff(cli, project, current_db), 'placement') != []

    def test_dry_run_applies_nothing(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        edit_app(project, 'an edit no update applies')

        planned = update(cli, project, current_db, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']
        assert ops_on(diff(cli, project, current_db), 'archive') != []

    def test_build_image_does_not_compare_first(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """build-image sends and builds whatever the project holds, in agreement or not."""
        ops = cli('db', 'build-image', current_db, '--json', cwd=project).json
        assert sorted(op['target'] for op in ops) == ['archive', 'image']
        assert all(op['status'] == 'applied' for op in ops), ops
        assert_in_agreement(cli, project, current_db)

    def test_errors(self, cli: PxtRunner, project: pathlib.Path, cloud_db_base_uri: str) -> None:
        declare(cli, project, cloud_db_base_uri)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:16]}'
        declare(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr


class TestPod:
    """What a service pod does with the project its database holds."""

    def test_pod_serves_the_project_the_database_holds(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, tmp_path: pathlib.Path
    ) -> None:
        app_file = str(project / _APP_FILE)
        cli('schema', 'update', app_file, current_db, '-f')
        cli('service', 'update', app_file, current_db, '-f')

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

    def test_pod_refuses_another_project(
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

    def test_service_diff_is_blocked_until_the_database_has_the_project(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str
    ) -> None:
        """A hosted service runs the project its database was given, so an edit here is `pxt db update`."""
        app_file = str(project / _APP_FILE)
        cli('schema', 'update', app_file, current_db, '-f')
        cli('service', 'update', app_file, current_db, '-f')

        edit_app(project, 'an edit the database has not been given')
        [blocked] = cli('service', 'diff', app_file, current_db, '--json', check=False).json['services']
        assert blocked['resolution'] == 'blocked'
        [op] = [op for op in blocked['ops'] if op['target'] == 'project']
        assert f'pxt db update {current_db}' in op['description'], op['description']

        update(cli, project, current_db)
        [reconciled] = cli('service', 'diff', app_file, current_db, '--json', check=False).json['services']
        assert reconciled['resolution'] != 'blocked', reconciled['ops']


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
