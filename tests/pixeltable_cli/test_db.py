"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, so the module is skipped unless the cloud
environment is configured, and it is marked expensive: applying what an entry declares rebuilds an image,
which takes minutes. They run against the session's hosted database, the one the cloud catalog tests use.
"""

import hashlib
import os
import pathlib
import shutil
import socket
import subprocess
import time
import uuid

import httpx
import pytest

from .conftest import PxtRunner
from .hosted import (
    APP_FILE,
    APPLY_TIMEOUT,
    EXIT_CHANGES_PENDING,
    EXIT_ERROR,
    EXIT_IN_AGREEMENT,
    assert_in_agreement,
    create_project_config,
    current_db,
    db_diff,
    db_status,
    db_update,
    edit_app,
    get_target_ops,
    hosted_db,
    project,
    schema_update,
    service_update,
)

__all__ = ['current_db', 'hosted_db', 'project']  # fixtures this module's tests request by name

_REQUEST_TIMEOUT = 30.0


@pytest.fixture(autouse=True)
def hosted_environment() -> None:
    """Skip the test unless the session names a hosted database to act on."""
    if os.environ.get('PXTTEST_CLOUD_DB_URI') is None:
        pytest.skip('PXTTEST_CLOUD_DB_URI is not set.')


pytestmark = [
    pytest.mark.remote_api,
    pytest.mark.expensive,
    pytest.mark.db_roots('local', reason='pxt db acts on a hosted database, not on the catalog a test runs against'),
]


class TestDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:12]}'
        create_project_config(cli, project, absent)

        plan = db_diff(cli, project, absent)
        assert plan['resolution'] == 'create'
        assert not plan['exists']
        assert plan['state'] is None
        assert sorted(op['target'] for op in plan['ops']) == ['archive', 'image']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        try:
            applied = db_update(cli, project, absent)
            assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
            assert db_status(cli, project, absent)['state'] == 'AVAILABLE'
            listed = cli('db', 'list', 'pxt://pixeltable', '--json', cwd=project).json
            assert absent.rsplit(':', 1)[-1] in [entry['db'] for entry in listed], listed
            # the database now holds this project, so a second look has nothing to do
            assert_in_agreement(cli, project, absent)
        finally:
            cli('db', 'delete', absent, cwd=project, check=False)

    def test_source_edit(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """An edit to a source file moves the archive alone: the environment it runs in is unchanged."""
        edit_app(project, 'an edit that changes no dependency')

        # a dry run reports the plan and applies none of it
        planned = db_update(cli, project, current_db, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']

        plan = db_diff(cli, project, current_db)
        assert get_target_ops(plan, 'image') == []
        [op] = get_target_ops(plan, 'archive')
        assert op['severity'] == 'additive'
        assert f'{APP_FILE} changed' in op['description'], op['description']
        assert not plan['summary']['rebuild']

        applied = db_update(cli, project, current_db)
        assert [op['status'] for op in applied['ops']] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_lockfile_edit(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """The lockfile is in both manifests, so editing it moves the image and the archive alike."""
        (project / 'requirements.txt').write_text('pixeltable\ntqdm\n', encoding='utf-8')

        plan = db_diff(cli, project, current_db)
        assert plan['resolution'] == 'update_additive'
        [image_op] = get_target_ops(plan, 'image')
        [archive_op] = get_target_ops(plan, 'archive')
        assert (image_op['name'], archive_op['name']) == ('image', 'project')
        assert 'requirements.txt changed' in image_op['description'], image_op['description']
        assert 'requirements.txt changed' in archive_op['description'], archive_op['description']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, current_db)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert (applied['in_agreement'], applied['returncode']) == (True, EXIT_IN_AGREEMENT)
        assert_in_agreement(cli, project, current_db)

    def test_two_projects(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A database holds one project: whichever was published last, whatever else declares it."""
        other = project.parent / 'other'
        shutil.copytree(project, other)
        edit_app(other, 'the file the other project holds')
        create_project_config(cli, other, current_db)

        # publishing the other project moves the database off this one, and the two swap on every publish
        db_update(cli, other, current_db)
        assert_in_agreement(cli, other, current_db)
        assert get_target_ops(db_diff(cli, project, current_db), 'archive') != []

        db_update(cli, project, current_db)
        assert_in_agreement(cli, project, current_db)
        assert get_target_ops(db_diff(cli, other, current_db), 'archive') != []

    def test_excluded_files(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A file the entry excludes is not part of the project, so writing it changes nothing."""
        # the entry itself is a project file, so the database has to be given the rewritten one
        create_project_config(cli, project, current_db, exclude=['notes/**'])
        db_update(cli, project, current_db)
        assert_in_agreement(cli, project, current_db)

        (project / 'notes').mkdir()
        (project / 'notes' / 'scratch.txt').write_text('not part of the project\n', encoding='utf-8')
        assert_in_agreement(cli, project, current_db)

        edit_app(project, 'an edit to a file the entry selects')
        assert get_target_ops(db_diff(cli, project, current_db), 'archive') != []

    @pytest.mark.skip(
        reason='cpu+1 leaves the pod unschedulable, and the database then holds a rollout no later scenario gets past'
    )
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
        refused = cli('db', 'update', current_db, '-f', cwd=project, check=False, timeout=APPLY_TIMEOUT)
        assert refused.returncode == EXIT_ERROR
        assert '--allow-destructive' in refused.stderr, refused.stderr
        assert [
            op['status']
            for op in get_target_ops(db_update(cli, project, current_db, '--allow-destructive'), 'capacity')
        ] == ['applied']
        assert_in_agreement(cli, project, current_db)

    def test_build_image(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """build-image rebuilds the image whether or not anything changed, and stores what is missing."""
        ops = {
            op['target']: op
            for op in cli('db', 'build-image', current_db, '--json', cwd=project, timeout=APPLY_TIMEOUT).json
        }
        # the database was given this project already, so the store holds its archive
        assert (ops['image']['status'], ops['archive']['status']) == ('applied', 'skipped'), ops
        assert_in_agreement(cli, project, current_db)

    def test_errors(self, cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> None:
        create_project_config(cli, project, hosted_db)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:12]}'
        create_project_config(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr


class TestPodRunner:
    def test_pod_serves_project(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, tmp_path: pathlib.Path
    ) -> None:
        app_file = str(project / APP_FILE)
        schema_update(cli, project, app_file, current_db)
        service_update(cli, project, app_file, current_db)

        unpacked = tmp_path / 'app'
        port = _free_port()
        pod = _run_pod(current_db, unpacked, '--host', '127.0.0.1', '--port', str(port))
        try:
            _wait_until_serving(f'http://127.0.0.1:{port}')
            assert (unpacked / APP_FILE).read_text() == (project / APP_FILE).read_text()
            assert (unpacked / 'requirements.txt').is_file()
            served = httpx.get(f'http://127.0.0.1:{port}/openapi.json', timeout=_REQUEST_TIMEOUT)
            paths = served.json()['paths']
            assert set(paths) == {'/docs', '/docs/update', '/docs/delete', '/preview'}, paths
        finally:
            pod.terminate()
            pod.wait(timeout=30)

    def test_pod_refuses_digest_mismatch(
        self, cli: PxtRunner, project: pathlib.Path, current_db: str, tmp_path: pathlib.Path
    ) -> None:
        """A pod is told which project to run, and runs nothing else."""
        unpacked = tmp_path / 'app'
        digest = hashlib.sha256(b'unknown digest').hexdigest()
        pod = _run_pod(current_db, unpacked, '--digest', digest, capture=True)
        stderr = pod.communicate(timeout=300)[1]

        assert pod.returncode != 0
        assert digest in stderr, stderr
        assert not unpacked.exists()


def _run_pod(db_uri: str, project_dir: pathlib.Path, *flags: str, capture: bool = False) -> subprocess.Popen:
    """Start a service pod for db_uri, serving the 'ingest' service of the app.py the archive holds."""
    argv = (
        'python',
        '-m',
        'pixeltable.serving.pod_runner',
        '--db',
        db_uri,
        '--app-file',
        APP_FILE,
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
