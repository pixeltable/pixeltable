"""`pxt db` against a hosted database.

Every scenario drives the CLI the way a user does: a project with a [[pixeltable.database]] entry, and the
`pxt db` verbs reading and applying it. They need a control plane, so the module is skipped unless the cloud
environment is configured. Applying what an entry declares rebuilds an image, which takes minutes. They run
against the session's hosted database, the one the cloud catalog tests use.
"""

import json
import pathlib
import shutil
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import pytest

import pixeltable_cli.client.commands.db as db_cmd
from pixeltable.service import proxy_daemon
from pixeltable.service.management_protocol import SpanRecord
from pixeltable_cli.client import hosted
from tests.utils import DatabaseRoot, new_db_uri

from .conftest import (
    APPLY_TIMEOUT,
    EXIT_CHANGES_PENDING,
    EXIT_ERROR,
    EXIT_IN_AGREEMENT,
    EXIT_REFUSED,
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


def _span(
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    offset_ms: int,
    status: str = 'UNSET',
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A span as the daemon returns it, starting offset_ms after midnight UTC on 2026-10-15."""
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        start_ms=1_792_022_400_000 + offset_ms,
        duration_ms=12.5,
        status=status,
        attributes=attributes or {},
    ).model_dump(mode='json')


def get_target_ops(plan: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """The plan's operations against one target: image, archive, capacity or secret."""
    return [op for op in plan['ops'] if op['target'] == target]


@pytest.fixture(scope='module')
def test_db_uri(session_cli: PxtRunner, session_project: pathlib.Path) -> Iterator[str]:
    """A database URI of this module's own, naming nothing until a test creates it, deleted when it ends.

    Module-scoped, unlike the per-test fixture on main: creating a database runs CodeBuild, and the tests
    here only need one to publish to in turn, which costs an archive upload each.
    """
    with disposable_db(session_cli, new_db_uri(), session_project) as uri:
        yield uri


pytestmark = pytest.mark.db_roots(
    'local', reason='pxt db acts on a hosted database, not on the catalog a test runs against'
)


@pytest.mark.usefixtures('hosted_environment')
class TestCloudDb:
    def test_create(self, cli: PxtRunner, project: pathlib.Path) -> None:
        db_name = f'pxttest-absent-{uuid.uuid4().hex[:12]}'
        absent = f'pxt://pixeltable:{db_name}'
        undeclared = cli('db', 'diff', absent, cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert f"run 'pxt db diff {absent} --add-config-entry'" in undeclared.stderr, undeclared.stderr

        plan = db_diff(cli, project, absent, '--add-config-entry')
        assert plan['entry_added_to'] == str(project / 'pixeltable.toml')
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

        create_project_config(cli, project, mixed)
        assert db_diff(cli, project, lower)['resolution'] == 'create'

        create_project_config(cli, project, lower)
        assert db_diff(cli, project, mixed)['resolution'] == 'create'

    def test_source_edit(self, cli: PxtRunner, project: pathlib.Path, test_db_uri: str) -> None:
        applied = db_update(cli, project, test_db_uri, '--add-config-entry')
        assert applied['entry_added_to'] == str(project / 'pixeltable.toml')
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
        assert refused.returncode == EXIT_REFUSED
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
        # a usage error exits 1, since 2 means changes pending
        table_uri = cli('db', 'logs', f'{test_db_uri}/table', cwd=project, check=False)
        assert table_uri.returncode == 1 and 'URI must be pxt://org:db' in table_uri.stderr, table_uri.stderr
        # 'db diff' takes a database URI, 'org status' an org URI
        for bad in ('pxt://pixeltable', 'pxt://pixeltable:pxttest/'):
            r = cli('db', 'diff', bad, cwd=project, check=False)
            assert r.returncode == 1 and 'URI must be pxt://org:db' in r.stderr, r.stderr
        org_with_db = cli('org', 'status', test_db_uri, cwd=project, check=False)
        assert org_with_db.returncode == 2 and 'URI must be pxt://org,' in org_with_db.stderr, org_with_db.stderr
        # the daemon refuses an invalid hosted name itself, before any request to the control plane
        not_hosted = cli('db', 'diff', 'pxt://pixeltable:my_db', cwd=project, check=False)
        assert not_hosted.returncode == EXIT_ERROR and "database name 'my_db' is invalid" in not_hosted.stderr
        assert 'Traceback' not in not_hosted.stderr, not_hosted.stderr

        # the daemon validates --since, --tail and --trace before reading anything
        for verb in ('logs', 'spans'):
            r = cli('db', verb, test_db_uri, '--since', 'bogus', cwd=project, check=False)
            assert r.returncode == EXIT_ERROR and 'must be a duration' in r.stderr, r.stderr
            r = cli('db', verb, test_db_uri, '--tail', '50000', cwd=project, check=False)
            assert r.returncode == EXIT_ERROR and "'limit' must be <= 10000" in r.stderr, r.stderr
        r = cli('db', 'spans', test_db_uri, '--trace', 'ABC', cwd=project, check=False)
        assert r.returncode == EXIT_ERROR and 'must be 32 lowercase hex digits' in r.stderr, r.stderr

        undeclared = cli('db', 'diff', 'pxt://pixeltable:pxttest-undeclared', cwd=project, check=False)
        assert undeclared.returncode == EXIT_ERROR
        assert '[[pixeltable.database]]' in undeclared.stderr, undeclared.stderr

        absent = f'pxt://pixeltable:pxttest-absent-{uuid.uuid4().hex[:12]}'
        create_project_config(cli, project, absent)
        never_built = cli('db', 'build-image', absent, cwd=project, check=False)
        assert never_built.returncode == EXIT_ERROR
        assert 'pxt db update' in never_built.stderr, never_built.stderr

    def test_add_config_entry(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """--add-config-entry appends a missing database entry, and leaves an existing one as it is."""
        db_name = f'pxttest-absent-{uuid.uuid4().hex[:12]}'
        absent = f'pxt://pixeltable:{db_name}'

        # after 'pxt init', the entry follows the local one, whose comments stay
        initialized = tmp_path / 'initialized'
        initialized.mkdir()
        cli('init', cwd=initialized)
        config_file = initialized / 'pixeltable.toml'
        written = config_file.read_text(encoding='utf-8')
        plan = db_diff(cli, initialized, absent, '--add-config-entry')
        assert (plan['entry_added_to'], plan['resolution']) == (str(config_file), 'create')
        appended = config_file.read_text(encoding='utf-8')
        assert appended.startswith(written)
        assert tomllib.loads(appended)['pixeltable']['database'] == [{}, {'name': absent}]

        # the entry matches in any case, so a rerun leaves the file as it is
        rerun = db_diff(cli, initialized, f'pxt://pixeltable:{db_name.upper()}', '--add-config-entry')
        assert rerun['entry_added_to'] is None
        assert config_file.read_text(encoding='utf-8') == appended

        # in a pyproject.toml, the entry goes in [tool.pixeltable], or in [tool.pixeltable.pixeltable] if there is one
        for i, (text, expected) in enumerate(
            (
                ('[tool.pixeltable]', {'database': [{'name': absent}]}),  # without a trailing newline
                ('[tool.pixeltable.pixeltable]\n', {'pixeltable': {'database': [{'name': absent}]}}),
            )
        ):
            in_pyproject = tmp_path / f'in_pyproject{i}'
            in_pyproject.mkdir()
            pyproject = in_pyproject / 'pyproject.toml'
            pyproject.write_text(f"[project]\nname = 'app'\n\n{text}", encoding='utf-8')
            assert db_diff(cli, in_pyproject, absent, '--add-config-entry')['entry_added_to'] == str(pyproject)
            assert tomllib.loads(pyproject.read_text(encoding='utf-8'))['tool']['pixeltable'] == expected

        # concurrent commands append the entry once; a plain diff first moves the daemon onto this project, so
        # that the concurrent commands don't each restart it
        concurrent = tmp_path / 'concurrent'
        concurrent.mkdir()
        (concurrent / 'pixeltable.toml').write_text('', encoding='utf-8')
        assert cli('db', 'diff', absent, cwd=concurrent, check=False).returncode == EXIT_ERROR
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(db_diff, cli, concurrent, absent, '--add-config-entry') for _ in range(4)]
        assert sorted(f.result()['entry_added_to'] is None for f in futures) == [False, True, True, True]
        config = tomllib.loads((concurrent / 'pixeltable.toml').read_text(encoding='utf-8'))
        assert config == {'pixeltable': {'database': [{'name': absent}]}}

        # a single [pixeltable.database] table for the database is its entry
        single = tmp_path / 'single'
        single.mkdir()
        (single / 'pixeltable.toml').write_text(f"[pixeltable.database]\nname = '{absent}'\n", encoding='utf-8')
        assert db_diff(cli, single, absent, '--add-config-entry')['entry_added_to'] is None

        # update appends the entry in a dry run, and when it refuses for lack of -f and a TTY
        for flags, returncode in ((('--dry-run',), EXIT_CHANGES_PENDING), ((), EXIT_REFUSED)):
            updated = tmp_path / f'updated{len(flags)}'
            updated.mkdir()
            (updated / 'pixeltable.toml').write_text('', encoding='utf-8')
            r = cli('db', 'update', absent, '--add-config-entry', '--json', *flags, cwd=updated, check=False)
            assert r.returncode == returncode, r.stderr
            assert (r.json['entry_added_to'], r.json['resolution']) == (str(updated / 'pixeltable.toml'), 'create')
            assert db_name not in _list_dbs(cli, updated)


class TestLocalConfigEntry:
    def test_add_config_entry_errors(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """--add-config-entry exits 1 before any request, and leaves the project configuration as it is."""
        uri = 'pxt://acme:main'
        r = cli('db', 'diff', uri, '--add-config-entry', cwd=tmp_path, check=False)
        assert r.returncode == EXIT_ERROR
        assert "pxt db diff: no project configuration here; run 'pxt init' to write one" in r.stderr, r.stderr
        assert list(tmp_path.iterdir()) == []

        r = cli('db', 'update', "pxt://acme:it's", '--add-config-entry', cwd=tmp_path, check=False)
        assert r.returncode == EXIT_ERROR
        assert 'pxt db update: database name "it\'s" is invalid' in r.stderr, r.stderr

        config_file = tmp_path / 'pixeltable.toml'
        for text, error in (
            ('[pixeltable.database]\n', 'declares pixeltable.database as a single table'),
            ('[pixeltable]\ndatabase = [{}]\n', 'declares pixeltable.database as an inline array'),
        ):
            config_file.write_text(text, encoding='utf-8')
            r = cli('db', 'diff', uri, '--add-config-entry', cwd=tmp_path, check=False)
            assert r.returncode == EXIT_ERROR and error in r.stderr, r.stderr
            assert config_file.read_text(encoding='utf-8') == text


class TestLocalLogs:
    @pytest.mark.db_roots('proxy', reason='a proxy-daemon database logs to a file, which the error names')
    def test_local_logs_error(self, cli: PxtRunner, db_root: DatabaseRoot, proxy_daemon_db: str) -> None:
        r = cli('db', 'logs', db_root.prefix, check=False)
        log_file = proxy_daemon.log_path(proxy_daemon_db)
        assert r.returncode == EXIT_ERROR, r.stderr
        assert f'writes its log to {log_file}' in r.stderr, r.stderr
        assert 'Pixeltable Cloud database' in r.stderr and 'pxt new' in r.stderr, r.stderr
        assert "'pxt db update pxt://org:db'" in r.stderr, r.stderr
        assert log_file.is_file()
        # a local database writes its spans to the span_dump file, if one is set
        r = cli('db', 'spans', db_root.prefix, check=False)
        assert r.returncode == EXIT_ERROR, r.stderr
        assert 'not supported. Set OTEL_SPAN_DUMP' in r.stderr, r.stderr

    def test_spans_text(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """The text output groups the spans by trace and indents each span under its parent."""
        trace_a, trace_b = 'a' * 32, 'b' * 32
        spans = [
            # trace a starts first, so it is printed first
            _span(trace_a, '01', None, 'POST /ingest', 0),
            # trace b starts before the rest of trace a, so it is printed second
            _span(trace_b, '11', None, 'GET /search', 5),
            # a daemon request or a served route is named by its operation attribute
            _span(trace_a, '02', '01', 'pixeltable.route', 10, attributes={'pxt.route': 'POST /ingest/docs'}),
            _span(trace_a, '03', '02', 'pixeltable.rpc', 20, status='ERROR', attributes={'pxt.method': 'Table.insert'}),
            _span(trace_a, '04', '01', 'commit', 30),
            # its parent was not sampled, so it sits at depth 0
            _span(trace_b, '12', '99', 'orphan', 40),
        ]
        monkeypatch.setattr(hosted, 'get_request', lambda _path, _params: {'spans': spans})

        db_cmd.run(['spans', 'pxt://acme:main'])
        assert capsys.readouterr().out.splitlines() == [
            f'trace {trace_a}',
            '2026-10-15T00:00:00.000+00:00  12.5 ms  POST /ingest',
            '  2026-10-15T00:00:00.010+00:00  12.5 ms  pixeltable.route POST /ingest/docs',
            '    2026-10-15T00:00:00.020+00:00  12.5 ms  pixeltable.rpc Table.insert  ERROR',
            '  2026-10-15T00:00:00.030+00:00  12.5 ms  commit',
            f'trace {trace_b}',
            '2026-10-15T00:00:00.005+00:00  12.5 ms  GET /search',
            '2026-10-15T00:00:00.040+00:00  12.5 ms  orphan',
        ]

        db_cmd.run(['spans', 'pxt://acme:main', '--json'])
        assert json.loads(capsys.readouterr().out) == spans

        monkeypatch.setattr(hosted, 'get_request', lambda _path, _params: {'spans': []})
        db_cmd.run(['spans', 'pxt://acme:main', '--since', '10m'])
        assert capsys.readouterr().out == 'No spans in the last 10m.\n'
