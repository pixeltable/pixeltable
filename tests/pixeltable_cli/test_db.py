"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, so the module is skipped unless the cloud
environment is configured, and it is marked expensive: applying what an entry declares rebuilds an image,
which takes minutes. They run against the session's hosted database, the one the cloud catalog tests use.
"""

import json
import pathlib
import shutil
import uuid
from typing import Any

import pytest

from ..utils import DatabaseRoot
from .conftest import PxtRunner, write_requirements

pytestmark = [
    pytest.mark.remote_api,
    pytest.mark.expensive,
    pytest.mark.db_roots('local', reason='pxt db acts on a hosted database, not on the catalog a test runs against'),
]

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
def current_db(cli: PxtRunner, project: pathlib.Path, db_root: DatabaseRoot) -> str:
    """A hosted database running an image built from this project."""
    create_project_config(cli, project, db_root.prefix)
    applied = db_update(cli, project, db_root.prefix)
    assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']
    return db_root.prefix


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
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False, timeout=_BUILD_TIMEOUT)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def db_status(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db status` reports."""
    return cli('db', 'status', db_uri, '--json', cwd=project).json


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


@pytest.mark.db_roots('cloud', reason='Uses the CLI to drive cloud DBs through the control plane')
class TestDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A database the control plane does not hold is planned as a create, and the update makes it."""
        absent = f'pxt://pixeltable:pxttest-gone-{uuid.uuid4().hex[:12]}'
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
            assert db_status(cli, project, absent)['state'] == 'AVAILABLE'
        finally:
            cli('db', 'delete', absent, cwd=project, check=False)

    def test_update_builds_the_image(self, cli: PxtRunner, project: pathlib.Path, db_root: DatabaseRoot) -> None:
        """Every update builds the image, since the database reports no fingerprint to compare."""
        create_project_config(cli, project, db_root.prefix)

        plan = db_diff(cli, project, db_root.prefix)
        assert plan['resolution'] == 'update_additive'
        assert [op['name'] for op in get_target_ops(plan, 'image')] == ['image']
        assert plan['summary']['rebuild']
        assert plan['returncode'] == EXIT_CHANGES_PENDING

        applied = db_update(cli, project, db_root.prefix)
        assert all(op['status'] == 'applied' for op in applied['ops']), applied['ops']

        # and the next diff plans the same build, since there is still nothing to compare
        assert get_target_ops(db_diff(cli, project, db_root.prefix), 'image') != []

    def test_status_list(self, cli: PxtRunner, project: pathlib.Path, db_root: DatabaseRoot) -> None:
        name = db_root.prefix.rsplit(':', 1)[-1]
        status = db_status(cli, project, db_root.prefix)
        assert status['state'] == 'AVAILABLE', status
        listed = cli('db', 'list', 'pxt://pixeltable', '--json', cwd=project).json
        assert name in [entry['db_name'] for entry in listed], listed

    def test_dry_run(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """A dry run reports what an update would do and leaves the database alone."""
        planned = db_update(cli, project, current_db, '-n')
        assert planned['returncode'] == EXIT_CHANGES_PENDING
        assert all(op['status'] is None for op in planned['ops']), planned['ops']
        assert get_target_ops(planned, 'image') != []

    def test_build_image(self, cli: PxtRunner, project: pathlib.Path, current_db: str) -> None:
        """build-image sends whatever the project holds and builds it."""
        ops = cli('db', 'build-image', current_db, '--json', cwd=project, timeout=_BUILD_TIMEOUT).json
        assert [op['target'] for op in ops] == ['image']
        assert all(op['status'] == 'applied' for op in ops), ops

    def test_errors(self, cli: PxtRunner, project: pathlib.Path, db_root: DatabaseRoot) -> None:
        create_project_config(cli, project, db_root.prefix)

        not_a_uri = cli('db', 'diff', 'my_dir', cwd=project, check=False)
        assert 'URI must be pxt://org:db' in not_a_uri.stderr, not_a_uri.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-gone-{uuid.uuid4().hex[:12]}'
        create_project_config(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr
