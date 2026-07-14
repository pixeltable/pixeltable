"""conftest for cloud e2e tests: pull pod logs automatically on any test failure."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from .test_cli_e2e import _cloud_env


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: Any) -> Any:
    outcome = yield
    rep = outcome.get_result()
    item._cloud_e2e_rep = rep  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _log_on_failure(request: pytest.FixtureRequest) -> Any:
    yield
    rep = getattr(request.node, '_cloud_e2e_rep', None)
    if rep is None or not rep.failed:
        return
    try:
        resources = request.getfixturevalue('resources')
    except pytest.FixtureLookupError:
        return
    print(f'\n{"─" * 60}')
    print(f'TEST FAILED — status for {resources.db_slug}')
    print(f'{"─" * 60}')
    for cmd, label in [
        (['pxt', 'db', 'status', resources.db_uri, '--json'], 'DB status'),
        (['pxt', 'service', 'status', resources.svc_uri, '--json'], 'service status'),
    ]:
        print(f'\n── {label} ──')
        r = subprocess.run(cmd, capture_output=True, text=True, env=_cloud_env(), timeout=30, check=False)
        output = (r.stdout + r.stderr).strip()
        print(output if output else '(no output)')
    print(f'{"─" * 60}\n')
