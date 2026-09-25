"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, so the module is skipped unless the cloud
environment is configured, and it is marked expensive: applying what an entry declares rebuilds an image,
which takes minutes. They run against the session's hosted database, the one the cloud catalog tests use.
"""

import pathlib
import shutil
import time
import uuid
from typing import Any, Iterator

import pytest

from pixeltable.service import proxy_daemon
from tests.utils import DatabaseRoot, new_db_uri, skip_test_if_no_config

from .conftest import (
    APPLY_TIMEOUT,
    EXIT_CHANGES_PENDING,
    EXIT_ERROR,
    EXIT_IN_AGREEMENT,
    PxtRunner,
    assert_in_agreement,
    db_diff,
    db_update,
    disposable_db,
    read_logs_until,
)
from .hosted import APP_FILE, create_project_config, edit_app, project

__all__ = ['project']  # fixtures this module's tests request by name

_REQUEST_TIMEOUT = 30.0


def db_status(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What the database at db_uri provides, as `pxt db status` reports it, its resources flattened in."""
    current = cli('db', 'status', db_uri, '--json', cwd=project).json['current']
    return {**(current.get('resources') or {}), **current}


def _list_dbs(cli: PxtRunner, project: pathlib.Path) -> set[str]:
    listed = cli('db', 'list', '--json', cwd=project).json
    return {entry['db'] for entry in listed}


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


@pytest.fixture
def hosted_environment() -> None:
    """Skip the test unless a control plane is configured to create the database against."""
    skip_test_if_no_config('api_key')


@pytest.fixture(scope='module')
def test_db_uri(session_cli: PxtRunner, session_project: pathlib.Path) -> Iterator[str]:
    """A database URI of this module's own, naming nothing until a test creates it, deleted when it ends.

    Module-scoped, unlike the per-test fixture on main: creating a database runs CodeBuild, and the tests
    here only need one to publish to in turn, which costs an archive upload each.
    """
    with disposable_db(session_cli, new_db_uri(), session_project) as uri:
        yield uri


pytestmark = [
    pytest.mark.remote_api,
    pytest.mark.expensive,
    pytest.mark.db_roots('local', reason='pxt db acts on a hosted database, not on the catalog a test runs against'),
]


@pytest.mark.usefixtures('hosted_environment')
class TestDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        db_name = f'pxttest-absent-{uuid.uuid4().hex[:12]}'
        absent = f'pxt://pixeltable:{db_name}'
        create_project_config(cli, project, absent)

        plan = db_diff(cli, project, absent)
        assert plan['resolution'] == 'create'
        assert not plan['exists']
        assert plan['state'] is None
        assert sorted(op['target'] for op in plan['ops']) == ['archive', 'image']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        try:
            status = db_update(cli, project, absent)
            statuses = {op['target']: op['status'] for op in status['ops']}
            # we always have a fresh archive
            assert statuses['archive'] == 'applied', status['ops']
            # we might reuse an image from an earlier test run
            assert statuses['image'] in ('applied', 'skipped'), status['ops']
            assert db_status(cli, project, absent)['state'] == 'AVAILABLE'
            # the pod that just came up has logged its startup, and the probes are dropped unless asked for
            started = 'Connected to Pixeltable database at:'
            records = read_logs_until(cli, 'db', 'logs', absent, contains=started, cwd=project)
            assert records == sorted(records, key=lambda r: r['ts_ms'])
            assert not any('GET /health' in r['line'] for r in records)
            read_logs_until(cli, 'db', 'logs', absent, '--include-health', contains='GET /health', cwd=project)
            assert started in cli('db', 'logs', absent, cwd=project).stdout
            tail = cli('db', 'logs', absent, '--tail', '1', '--json', cwd=project).json
            assert len(tail) == 1
            # More lines may arrive between reads, but the newest cannot precede a line already returned.
            assert tail[0]['ts_ms'] >= records[-1]['ts_ms'], (tail, records[-5:])
            # Let the startup line age out of a short window; a backend ignoring --since would return it.
            time.sleep(2)
            read_started = time.time()
            recent = cli('db', 'logs', absent, '--since', '1s', '--json', cwd=project).json
            assert not any(started in r['line'] for r in recent), recent
            assert all(r['ts_ms'] >= int((read_started - 1) * 1000) for r in recent), recent

            cli('db', 'restart', absent, cwd=project, timeout=APPLY_TIMEOUT)
            assert db_status(cli, project, absent)['state'] == 'AVAILABLE'

            assert db_name in _list_dbs(cli, project)
            # the database now holds this project, so a second look has nothing to do
            assert_in_agreement(cli, project, absent)

            refused = cli('db', 'delete', absent, cwd=project, check=False)
            assert refused.returncode == 3, refused.stderr
            assert db_name in _list_dbs(cli, project)

            cli('db', 'delete', absent, '-f', cwd=project)
            assert db_name not in _list_dbs(cli, project)
        finally:
            cli('db', 'delete', absent, '-f', cwd=project, check=False)

    def test_name_case(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A database is found whatever the case of its name, in the config file or on the command line."""
        db = f'pxttest-absent-{uuid.uuid4().hex[:12]}'
        lower, mixed = f'pxt://pixeltable:{db}', f'pxt://pixeltable:{db.upper()}'

        # db_diff() fails the test on any other exit status, so reaching a plan is the assertion: before,
        # the entry matched only a command spelling it the same way
        create_project_config(cli, project, mixed)
        assert db_diff(cli, project, lower)['resolution'] == 'create'

        create_project_config(cli, project, lower)
        assert db_diff(cli, project, mixed)['resolution'] == 'create'

    def test_source_edit(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        create_project_config(cli, project, test_db_uri)
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        edit_app(project, 'an edit that changes no dependency')

        # a dry run reports the plan and applies none of it
        planned = db_update(cli, project, test_db_uri, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']

        plan = db_diff(cli, project, test_db_uri)
        assert get_target_ops(plan, 'image') == []
        [op] = get_target_ops(plan, 'archive')
        assert op['severity'] == 'additive'
        assert f'{APP_FILE} changed' in op['description'], op['description']
        assert not plan['summary']['rebuild']

        applied = db_update(cli, project, test_db_uri)
        assert [op['status'] for op in applied['ops']] == ['applied']
        assert_in_agreement(cli, project, test_db_uri)

    def test_lockfile_edit(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        create_project_config(cli, project, test_db_uri)
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        # appended, not replaced: the fixture's requirements install the wheel built from this checkout,
        # and dropping that line would rebuild the image against a released pixeltable instead
        requirements = project / 'requirements.txt'
        requirements.write_text(requirements.read_text(encoding='utf-8') + 'tqdm\n', encoding='utf-8')

        plan = db_diff(cli, project, test_db_uri)
        assert plan['resolution'] == 'update_additive'
        [image_op] = get_target_ops(plan, 'image')
        [archive_op] = get_target_ops(plan, 'archive')
        assert (image_op['name'], archive_op['name']) == ('image', 'project')
        assert 'requirements.txt changed' in image_op['description'], image_op['description']
        assert 'requirements.txt changed' in archive_op['description'], archive_op['description']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, test_db_uri)
        statuses = {op['target']: op['status'] for op in applied['ops']}
        # we always have a fresh archive
        assert statuses['archive'] == 'applied', applied['ops']
        # we might reuse an image from an earlier test run
        assert statuses['image'] in ('applied', 'skipped'), applied['ops']
        assert (applied['in_agreement'], applied['returncode']) == (True, EXIT_IN_AGREEMENT)
        assert_in_agreement(cli, project, test_db_uri)

    def test_two_projects(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        """Update a db from two project dirs."""
        create_project_config(cli, project, test_db_uri)
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        other = project.parent / 'other'
        shutil.copytree(project, other)
        edit_app(other, 'the file the other project holds')
        create_project_config(cli, other, test_db_uri)

        # publishing the other project moves the database off this one, and the two swap on every publish
        db_update(cli, other, test_db_uri)
        assert_in_agreement(cli, other, test_db_uri)
        assert get_target_ops(db_diff(cli, project, test_db_uri), 'archive') != []

        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)
        assert get_target_ops(db_diff(cli, other, test_db_uri), 'archive') != []

    def test_excluded_files(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        # the entry itself is a project file, so the database has to be given the rewritten one
        create_project_config(cli, project, test_db_uri, exclude=['notes/**'])
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        (project / 'notes').mkdir()
        (project / 'notes' / 'scratch.txt').write_text('not part of the project\n', encoding='utf-8')
        assert_in_agreement(cli, project, test_db_uri)

        edit_app(project, 'an edit to a file the entry selects')
        assert get_target_ops(db_diff(cli, project, test_db_uri), 'archive') != []

    def test_capacity(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        create_project_config(cli, project, test_db_uri)
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        # memory, not cpu: a node has headroom for another 256Mi, where one more cpu leaves the pod
        # unschedulable and the database then holds a rollout no later scenario gets past
        running_on = db_status(cli, project, test_db_uri)['memory_mb']
        create_project_config(cli, project, test_db_uri, memory_mb=running_on + 256)

        [op] = get_target_ops(db_diff(cli, project, test_db_uri), 'capacity')
        assert op['name'] == 'memory_mb'
        assert not op['destructive']
        assert str(running_on + 256) in op['description'], op['description']

        assert [op['status'] for op in get_target_ops(db_update(cli, project, test_db_uri), 'capacity')] == ['applied']
        assert_in_agreement(cli, project, test_db_uri)

        # taking capacity away is destructive, so it needs the flag that permits it
        create_project_config(cli, project, test_db_uri, memory_mb=running_on)
        refused = cli('db', 'update', test_db_uri, '-f', cwd=project, check=False, timeout=APPLY_TIMEOUT)
        assert refused.returncode == EXIT_ERROR
        assert '--allow-destructive' in refused.stderr, refused.stderr
        assert [
            op['status']
            for op in get_target_ops(db_update(cli, project, test_db_uri, '--allow-destructive'), 'capacity')
        ] == ['applied']
        assert_in_agreement(cli, project, test_db_uri)

    def test_build_image(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        create_project_config(cli, project, test_db_uri)
        db_update(cli, project, test_db_uri)
        assert_in_agreement(cli, project, test_db_uri)

        ops = {
            op['target']: op
            for op in cli('db', 'build-image', test_db_uri, '--json', cwd=project, timeout=APPLY_TIMEOUT).json
        }
        # the database was given this project already, so the store holds its archive
        assert (ops['image']['status'], ops['archive']['status']) == ('applied', 'skipped'), ops
        assert_in_agreement(cli, project, test_db_uri)

    def test_errors(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        create_project_config(cli, project, test_db_uri)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr
        table_uri = cli('db', 'logs', f'{test_db_uri}/table', cwd=project, check=False)
        assert table_uri.returncode == 2 and 'URI must be pxt://org:db' in table_uri.stderr, table_uri.stderr
        # 'db diff' takes a database URI, 'org status' an org URI
        for bad in ('pxt://pixeltable', 'pxt://pixeltable:pxttest/'):
            r = cli('db', 'diff', bad, cwd=project, check=False)
            assert r.returncode == 2 and 'URI must be pxt://org:db' in r.stderr, r.stderr
        org_with_db = cli('org', 'status', test_db_uri, cwd=project, check=False)
        assert org_with_db.returncode == 2 and 'URI must be pxt://org,' in org_with_db.stderr, org_with_db.stderr

        # the daemon validates --since and --tail before reading anything
        r = cli('db', 'logs', test_db_uri, '--since', 'bogus', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR and 'must be a duration' in r.stderr, r.stderr
        r = cli('db', 'logs', test_db_uri, '--tail', '50000', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR and "'limit' must be <= 10000" in r.stderr, r.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:12]}'
        create_project_config(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr


class TestLocalLogs:
    @pytest.mark.db_roots('proxy', reason='a proxy-daemon database logs to a file, which the error names')
    def test_local_logs_error(self, cli: PxtRunner, db_root: DatabaseRoot, proxy_daemon_db: str) -> None:
        r = cli('db', 'logs', db_root.prefix, check=False)
        log_file = proxy_daemon.log_path(proxy_daemon_db)
        assert r.returncode == EXIT_ERROR, r.stderr
        assert f'not supported; the log is at {log_file}' in r.stderr, r.stderr
        assert log_file.is_file()
