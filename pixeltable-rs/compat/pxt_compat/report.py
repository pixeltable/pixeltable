"""pytest plugin that records one outcome per test, with a short reason, as JSON.

Loaded by run.py with `-p pxt_compat.report --compat-report <path>`. Under xdist it runs in the controller, which
receives every worker's reports, and writes the file once at the end of the session.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from pxt_compat import stubs

_MAX_REASON = 300


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--compat-report', default=None, help='write per-test outcomes as JSON to this path')


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption('--compat-report')
    if path and not hasattr(config, 'workerinput'):  # controller (or non-xdist) process only
        config.pluginmanager.register(Recorder(Path(path), config), 'pxt_compat_recorder')


def _crash_message(report: Any) -> str:
    longrepr = report.longrepr
    if longrepr is None:
        return ''
    if isinstance(longrepr, tuple) and len(longrepr) == 3:  # skip: (path, lineno, reason)
        return str(longrepr[2]).removeprefix('Skipped: ')
    crash = getattr(longrepr, 'reprcrash', None)
    if crash is not None and getattr(crash, 'message', None):
        return str(crash.message)
    text = str(longrepr).strip().splitlines()
    return text[-1] if text else ''


def _short(message: str) -> str:
    message = ' '.join(message.split())
    return message if len(message) <= _MAX_REASON else message[: _MAX_REASON - 3] + '...'


def categorize(outcome: str, reason: str, full_text: str) -> str | None:
    """A coarse cause for a non-passing outcome, for summaries."""
    if outcome in ('passed', 'xfailed', 'xpassed'):
        return None
    if outcome == 'skipped':
        return 'skip'
    text = full_text or reason
    if stubs.MARKER in text:
        return 'internal-api'
    if 'UtilsDefinitionError' in text:
        return 'utils-definition'
    if 'pixeltable._testing is not importable' in text:
        return 'testing-hooks'
    if 'Timeout' in reason or '+++++ Timeout +++++' in text:
        return 'timeout'
    if 'crashed' in text and 'worker' in text:
        return 'crash'
    if 'NotImplementedError' in text and 'pixeltable-rs harness' in text:
        return 'harness-helper'
    return 'fail'


class Recorder:
    def __init__(self, path: Path, config: pytest.Config) -> None:
        self.path = path
        self.config = config
        self.tests: dict[str, dict[str, Any]] = {}
        self.collect_errors: dict[str, str] = {}
        self.started = datetime.datetime.now(tz=datetime.timezone.utc)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collect_errors[report.nodeid] = _short(_crash_message(report) or report.longreprtext[-500:])

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        entry = self.tests.setdefault(
            report.nodeid, {'outcome': None, 'reason': '', 'category': None, 'duration': 0.0, '_text': ''}
        )
        entry['duration'] += report.duration
        wasxfail = hasattr(report, 'wasxfail')
        message = _crash_message(report)
        outcome: str | None = None
        if report.when == 'setup':
            if report.failed:
                outcome = 'error'
            elif report.skipped:
                outcome = 'xfailed' if wasxfail else 'skipped'
        elif report.when == 'call':
            if report.passed:
                outcome = 'xpassed' if wasxfail else 'passed'
            elif report.failed:
                outcome = 'failed'
            elif report.skipped:
                outcome = 'xfailed' if wasxfail else 'skipped'
        elif report.when == 'teardown':
            if report.failed and entry['outcome'] in (None, 'passed', 'xpassed', 'skipped', 'xfailed'):
                outcome = 'error'
                message = f'teardown: {message}'
        elif report.failed:  # xdist reports a test whose worker died with when='???'
            outcome = 'failed'
            message = message or 'worker crashed'
        if outcome is None:
            return
        if outcome == 'xfailed' and wasxfail and not message:
            message = str(report.wasxfail)
        entry['outcome'] = outcome
        entry['reason'] = _short(message) if outcome not in ('passed', 'xpassed') else ''
        entry['_text'] = report.longreprtext if report.failed else ''
        entry['category'] = categorize(outcome, entry['reason'], entry['_text'])

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        pxt = sys.modules.get('pixeltable')
        tests = {
            nodeid: {k: (round(v, 3) if k == 'duration' else v) for k, v in entry.items() if not k.startswith('_')}
            for nodeid, entry in sorted(self.tests.items())
            if entry['outcome'] is not None
        }
        data = {
            'meta': {
                'started': self.started.isoformat(),
                'finished': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                'exitstatus': int(exitstatus),
                'args': list(self.config.invocation_params.args),
                'python': sys.executable,
                'pixeltable': getattr(pxt, '__file__', None),
                'pixeltable_version': getattr(pxt, '__version__', None),
                'stubs': stubs.report(),
                'workers': os.environ.get('PYTEST_XDIST_WORKER_COUNT'),
            },
            'collect_errors': dict(sorted(self.collect_errors.items())),
            'tests': tests,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=1, default=str) + '\n', encoding='utf-8')
        tmp.replace(self.path)
