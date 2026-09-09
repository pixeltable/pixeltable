"""Fixtures and helpers for driving the CLI against a hosted database.

A test that asks for `hosted_db` or `current_db` is skipped unless the session names one to act on.
"""

import json
import os
import pathlib
import shutil
import time
from typing import Any

import pytest

from .conftest import PxtRunner

# the exit statuses `pxt db diff` and `pxt db update` document
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2

# a publish that rebuilds the image waits on CodeBuild, far longer than the default cli timeout allows
APPLY_TIMEOUT = 2400.0

APP_FILE = 'basic.py'  # the corpus file the project holds, and the pod serves

# a database update restarts the services running its project, which redeploys their pods
_SERVICE_RESTART_TIMEOUT = 600.0


@pytest.fixture
def hosted_db() -> str:
    """The hosted database these tests act on, which is the one the cloud catalog tests read."""
    uri = os.environ.get('PXTTEST_CLOUD_DB_URI')
    if uri is None:
        pytest.skip('PXTTEST_CLOUD_DB_URI is not set.')
    return uri


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A test-specific project that is not the session project, pre-loaded with a single app file and requirements."""
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'pixeltable.toml').write_text('', encoding='utf-8')
    shutil.copy(pathlib.Path(__file__).parent / 'apps' / APP_FILE, root / APP_FILE)
    (root / 'requirements.txt').write_text('pixeltable\n', encoding='utf-8')
    return root


@pytest.fixture
def current_db(cli: PxtRunner, project: pathlib.Path, hosted_db: str) -> str:
    """A hosted database holding this project: where most scenarios start."""
    create_project_config(cli, project, hosted_db)
    db_update(cli, project, hosted_db)
    assert_in_agreement(cli, project, hosted_db)
    return hosted_db


def edit_app(project: pathlib.Path, added: str) -> None:
    """Append to the project's application file."""
    with open(project / APP_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\n{added}\n')


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
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False, timeout=APPLY_TIMEOUT)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def db_status(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What the database at db_uri provides, as `pxt db status` reports it."""
    return cli('db', 'status', db_uri, '--json', cwd=project).json['status']


def schema_update(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str) -> None:
    """Create what app_file's models declare at db_uri."""
    cli('schema', 'update', app_file, db_uri, '-f', cwd=project)


def service_update(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str, *flags: str) -> None:
    """Serve what app_file declares at db_uri."""
    cli('service', 'update', app_file, db_uri, '-f', *flags, cwd=project)


def service_diff(cli: PxtRunner, project: pathlib.Path, app_file: str, db_uri: str) -> dict[str, Any]:
    """What `pxt service diff` reports for app_file at db_uri, its exit status under 'returncode'."""
    r = cli('service', 'diff', app_file, db_uri, '--json', cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def service_list(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, dict[str, Any]]:
    """The instances `pxt service list` reports at db_uri, keyed by name."""
    return {i['name']: i for i in cli('service', 'list', db_uri, '--json', cwd=project).json}


def await_service_available(cli: PxtRunner, project: pathlib.Path, db_uri: str, name: str) -> None:
    """Block until the named instance is serving, which it stops doing while its database is updated."""
    deadline = time.monotonic() + _SERVICE_RESTART_TIMEOUT
    while True:
        state = service_list(cli, project, db_uri)[name]['state']
        if state == 'AVAILABLE':
            return
        assert time.monotonic() < deadline, f'{name} is {state} after {_SERVICE_RESTART_TIMEOUT:.0f}s'
        time.sleep(5)


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


def assert_in_agreement(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> None:
    plan = db_diff(cli, project, db_uri)
    assert plan['in_agreement'], plan['ops']
    assert plan['returncode'] == EXIT_IN_AGREEMENT
    assert plan['ops'] == []
