"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, and are marked expensive: applying what
an entry declares rebuilds an image, which takes minutes.
"""

import json
import pathlib
import shutil
import uuid
from typing import Any, Iterator

import pytest

from tests.utils import skip_test_if_no_config

from .conftest import PxtRunner, write_requirements

# the exit statuses `pxt db diff` and `pxt db update` document
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2

# an update builds an image, which CodeBuild takes minutes to do
_BUILD_TIMEOUT = 1800.0

_APP_FILE = 'basic.py'  # the corpus file the project holds


@pytest.fixture
def project(tmp_path: pathlib.Path, pixeltable_wheel: pathlib.Path) -> pathlib.Path:
    """A project of the test's own, holding an application file from the corpus and a lockfile.

    Outside the session's project, so that the archive holds these files and nothing else: the client hands
    the daemon whichever project the working directory establishes.
    """
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'pixeltable.toml').write_text('', encoding='utf-8')
    shutil.copy(pathlib.Path(__file__).parent / 'apps' / _APP_FILE, root / _APP_FILE)
    write_requirements(root, pixeltable_wheel)
    return root


@pytest.fixture
def test_db_uri(cli: PxtRunner, project: pathlib.Path) -> Iterator[str]:
    """A database URI of the test's own, naming nothing until the test creates it, deleted when it ends."""
    uri = f'pxt://pixeltable:pxttest-{uuid.uuid4().hex[:12]}'
    try:
        yield uri
    finally:
        cli('db', 'delete', uri, cwd=project, check=False)


def create_project_config(cli: PxtRunner, project: pathlib.Path, db_uri: str, **settings: Any) -> None:
    """Write the project's entry for db_uri with these settings, and hand the daemon the new project."""
    lines = ['[[pixeltable.database]]', f'name = {json.dumps(db_uri)}']
    for key, value in settings.items():
        if isinstance(value, dict):
            lines.extend(f'{key}.{name} = {json.dumps(bound)}' for name, bound in value.items())
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
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False, timeout=_BUILD_TIMEOUT)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def db_status(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db status` reports."""
    return cli('db', 'status', db_uri, '--json', cwd=project).json


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


@pytest.mark.very_expensive
@pytest.mark.db_roots('local', reason='These tests have no catalog operations')
class TestDb:
    def test_db_ops(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        skip_test_if_no_config('api_key')
        create_project_config(cli, project, test_db_uri)

        # `db update`: Check that its first use is planned as a create
        plan = db_diff(cli, project, test_db_uri)
        assert plan['resolution'] == 'create'
        assert not plan['exists']
        assert plan['state'] is None
        assert plan['ops'] == []
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, test_db_uri)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
        assert db_status(cli, project, test_db_uri)['state'] == 'AVAILABLE'

        # `db logs`: the pod that just came up has logged its startup
        records = cli('db', 'logs', test_db_uri, '--json', cwd=project).json
        assert records != [], 'the database pod logged nothing in the last hour'
        assert records == sorted(records, key=lambda r: r['ts_ms'])
        assert not any('GET /health' in r['line'] for r in records)
        too_many = cli('db', 'logs', test_db_uri, '--tail', '50000', cwd=project, check=False)
        assert too_many.returncode == EXIT_ERROR
        assert 'less than or equal to 10000' in too_many.stderr, too_many.stderr

        # `db update`: Check that a second call is planned as an update
        # Every update rebuilds the image, since the database reports no fingerprint to compare
        plan = db_diff(cli, project, test_db_uri)
        assert plan['resolution'] == 'update_additive'
        assert [op['name'] for op in get_target_ops(plan, 'image')] == ['image']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, test_db_uri)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']

        # `db update`: And the next diff plans the same build, since the database reports no fingerprint to compare
        assert get_target_ops(db_diff(cli, project, test_db_uri), 'image') != []

        # Check that the database is listed in the org's databases
        status = db_status(cli, project, test_db_uri)
        assert status['state'] == 'AVAILABLE', status
        listed = cli('db', 'list', 'pxt://pixeltable', '--json', cwd=project).json
        test_db_name = test_db_uri.rsplit(':', 1)[-1]
        assert test_db_name in [entry['db_name'] for entry in listed], listed

        # `db update`: Check that a dry run of an update is planned as an update, but not applied
        planned = db_update(cli, project, test_db_uri, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']
        assert get_target_ops(planned, 'image') != []

        # `db build-image`: Sends whatever the project holds and builds it
        ops = cli('db', 'build-image', test_db_uri, '--json', cwd=project, timeout=_BUILD_TIMEOUT).json
        assert [op['target'] for op in ops] == ['image']
        assert all(op['status'] == 'applied' for op in ops), ops

    def test_db_errors(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        skip_test_if_no_config('api_key')
        create_project_config(cli, project, test_db_uri)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        # test_db_uri is declared but never created, so there is no image to build
        never_built = cli('db', 'build-image', test_db_uri, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr
