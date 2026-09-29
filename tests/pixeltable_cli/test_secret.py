"""Tests for `pxt secret`."""

import http.server
import json
import os
import pathlib
import subprocess
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ..utils import CLOUD_DB_ROOT_URIS
from .conftest import PxtResult, PxtRunner

_RUN_TIMEOUT_SECS = 180.0


@dataclass
class FakeControlPlane:
    """A stand-in for the cloud management API, holding the request bodies it was sent."""

    url: str
    received_requests: list[dict[str, Any]]
    # the body of the answer to each operation_type; any other operation is answered with {}
    responses: dict[str, dict[str, Any]] = field(default_factory=dict)


@pytest.fixture
def fake_control_plane() -> Iterator[FakeControlPlane]:
    """A control plane on localhost: it remembers every request it was sent, and answers from its responses."""
    received_requests: list[dict[str, Any]] = []
    responses: dict[str, dict[str, Any]] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received_requests.append(request)
            body = json.dumps(responses.get(request['operation_type'], {})).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield FakeControlPlane(f'http://127.0.0.1:{server.server_address[1]}', received_requests, responses)
    finally:
        server.shutdown()
        server.server_close()


def _pxt_secret(port: int, cwd: pathlib.Path, control_plane: FakeControlPlane, *args: str) -> PxtResult:
    """Run `pxt secret`. The daemon it starts inherits this environment and talks to the fake control plane."""
    r = subprocess.run(
        ['pxt', 'secret', *args],
        env={
            **os.environ,
            'PXT_PORT': str(port),
            'PIXELTABLE_API_URL': control_plane.url,
            'PIXELTABLE_API_KEY': 'test-key',
        },
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=_RUN_TIMEOUT_SECS,
    )
    return PxtResult(r.returncode, r.stdout, r.stderr)


class TestSecret:
    def test_set_list_delete(
        self, daemon_port: int, tmp_path: pathlib.Path, fake_control_plane: FakeControlPlane
    ) -> None:
        def pxt_secret(*args: str) -> PxtResult:
            # any cwd will do, as long as it's not the repo root or its subdir
            return _pxt_secret(daemon_port, tmp_path, fake_control_plane, *args)

        def sent() -> list[dict[str, Any]]:
            requests = list(fake_control_plane.received_requests)
            fake_control_plane.received_requests.clear()
            return requests

        # set
        r = pxt_secret('set', 'pxt://acme:main', 'OPENAI_API_KEY=test=value', 'CUSTOM_TOKEN=custom-value', '--json')
        assert r.returncode == 0, r.stderr
        assert r.json == [
            {'key': 'CUSTOM_TOKEN', 'scope': 'pxt://acme:main'},
            {'key': 'OPENAI_API_KEY', 'scope': 'pxt://acme:main'},
        ]
        assert sent() == [
            {
                'operation_type': 'set_secret',
                'org': 'acme',
                'db': 'main',
                'key': 'OPENAI_API_KEY',
                'value': 'test=value',
            },
            {
                'operation_type': 'set_secret',
                'org': 'acme',
                'db': 'main',
                'key': 'CUSTOM_TOKEN',
                'value': 'custom-value',
            },
        ]
        r = pxt_secret('set', 'pxt://acme', 'ORG_KEY=org-value')
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == ['KEY      SCOPE', 'ORG_KEY  pxt://acme']
        assert sent() == [
            {'operation_type': 'set_secret', 'org': 'acme', 'db': None, 'key': 'ORG_KEY', 'value': 'org-value'}
        ]

        # delete
        r = pxt_secret('delete', 'pxt://acme', 'OLD_KEY', 'OLD_TOKEN')
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == ['KEY        SCOPE', 'OLD_KEY    pxt://acme', 'OLD_TOKEN  pxt://acme']
        assert sent() == [
            {'operation_type': 'delete_secret', 'org': 'acme', 'db': None, 'key': 'OLD_KEY'},
            {'operation_type': 'delete_secret', 'org': 'acme', 'db': None, 'key': 'OLD_TOKEN'},
        ]
        r = pxt_secret('delete', 'pxt://acme:main', 'OLD_KEY', '--json')
        assert r.returncode == 0, r.stderr
        assert r.json == [{'key': 'OLD_KEY', 'scope': 'pxt://acme:main'}]
        assert sent() == [{'operation_type': 'delete_secret', 'org': 'acme', 'db': 'main', 'key': 'OLD_KEY'}]

        # list: the whole org, with an override; the fake's audit field is ignored
        fake_control_plane.responses['list_all_secrets'] = {
            'org': 'acme',
            'secrets': [
                {'key': 'SHARED_KEY', 'db': 'main', 'audit': {'updated_at': '2026-09-01T00:00:00Z'}},
                {'key': 'SHARED_KEY', 'db': None},
                {'key': 'OTHER_KEY', 'db': 'dev'},
                {'key': 'DB_KEY', 'db': 'main'},
                {'key': 'ORG_KEY', 'db': None},
            ],
        }
        r = pxt_secret('list')
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == [
            'KEY         SCOPE            NOTE',
            'ORG_KEY     pxt://acme',
            'SHARED_KEY  pxt://acme',
            'OTHER_KEY   pxt://acme:dev',
            'DB_KEY      pxt://acme:main',
            'SHARED_KEY  pxt://acme:main  overrides an organization secret',
        ]
        r = pxt_secret('list', 'pxt://acme')
        assert r.returncode == 0, r.stderr
        assert sent() == [
            {'operation_type': 'list_all_secrets', 'org': None, 'db': None},
            {'operation_type': 'list_all_secrets', 'org': 'acme', 'db': None},
        ]

        # list: one database; the server's answer to db='main' has the org's secrets and main's
        fake_control_plane.responses['list_all_secrets'] = {
            'org': 'acme',
            'secrets': [
                {'key': 'SHARED_KEY', 'db': 'main'},
                {'key': 'SHARED_KEY', 'db': None},
                {'key': 'DB_KEY', 'db': 'main'},
                {'key': 'ORG_KEY', 'db': None},
            ],
        }
        r = pxt_secret('list', 'pxt://acme:main', '--json')
        assert r.returncode == 0, r.stderr
        assert r.json == [
            {'key': 'ORG_KEY', 'scope': 'pxt://acme'},
            {'key': 'SHARED_KEY', 'scope': 'pxt://acme'},
            {'key': 'DB_KEY', 'scope': 'pxt://acme:main'},
            {'key': 'SHARED_KEY', 'scope': 'pxt://acme:main', 'overrides_org': True},
        ]
        assert sent() == [{'operation_type': 'list_all_secrets', 'org': 'acme', 'db': 'main'}]

        # list: no overrides; NOTE stays in the header
        fake_control_plane.responses['list_all_secrets'] = {'org': 'acme', 'secrets': [{'key': 'ORG_KEY', 'db': None}]}
        r = pxt_secret('list')
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == ['KEY      SCOPE       NOTE', 'ORG_KEY  pxt://acme']

        # list: nothing set
        fake_control_plane.responses['list_all_secrets'] = {'org': 'acme', 'secrets': []}
        r = pxt_secret('list')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == 'No secrets for pxt://acme.'
        r = pxt_secret('list', 'pxt://acme:main')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == 'No secrets for pxt://acme:main.'
        r = pxt_secret('list', 'pxt://acme', '--json')
        assert r.returncode == 0, r.stderr
        assert r.json == []
        sent()

        # invalid arguments are refused before any request
        for key in ('PIXELTABLE_HOME', 'PIXELTABLE_DB', 'PIXELTABLE_VAR_FOO', 'pixeltable_home', 'Pixeltable_Db'):
            r = pxt_secret('set', 'pxt://acme:main', f'{key}=test-value')
            assert r.returncode != 0
            assert 'is reserved' in r.stderr, r.stderr
        for assignment in ('NO_VALUE', '=value'):
            r = pxt_secret('set', 'pxt://acme', assignment)
            assert r.returncode == 2
            assert f'expected KEY=VALUE, got {assignment!r}' in r.stderr, r.stderr
        for args in (
            ('list', 'acme'),
            ('list', 'pxt://acme:main/tbl'),
            ('set', 'pxt://acme:main/tbl', 'KEY=value'),
            ('delete', 'acme', 'KEY'),
        ):
            r = pxt_secret(*args)
            assert r.returncode == 2, args
            assert 'URI must be pxt://org or pxt://org:db' in r.stderr, r.stderr
        assert sent() == []

    @pytest.mark.usefixtures('hosted_environment')
    def test_cloud_list(self, session_cli: PxtRunner) -> None:
        db_uri = CLOUD_DB_ROOT_URIS['cloud-cli']
        other_db_uri = CLOUD_DB_ROOT_URIS['cloud']
        org_uri = db_uri.rsplit(':', maxsplit=1)[0]
        assert other_db_uri.rsplit(':', maxsplit=1)[0] == org_uri
        run_id = uuid.uuid4().hex[:8].upper()
        shared, org_only, db_only = f'PXTTEST_SHARED_{run_id}', f'PXTTEST_ORG_{run_id}', f'PXTTEST_DB_{run_id}'
        other_db_only = f'PXTTEST_OTHER_DB_{run_id}'
        try:
            session_cli('secret', 'set', org_uri, f'{shared}=org-value', f'{org_only}=org-value')
            session_cli('secret', 'set', db_uri, f'{shared}=db-value', f'{db_only}=db-value')
            session_cli('secret', 'set', other_db_uri, f'{other_db_only}=other-db-value')
            org_rows = [{'key': org_only, 'scope': org_uri}, {'key': shared, 'scope': org_uri}]
            db_rows = [{'key': db_only, 'scope': db_uri}, {'key': shared, 'scope': db_uri, 'overrides_org': True}]
            other_db_rows = [{'key': other_db_only, 'scope': other_db_uri}]
            # databases sort by name, and 'pxttest' sorts before 'pxttest-cli'
            whole_org = org_rows + other_db_rows + db_rows
            for args, expected in (((), whole_org), ((org_uri,), whole_org), ((db_uri,), org_rows + db_rows)):
                rows = session_cli('secret', 'list', *args, '--json').json
                assert [row for row in rows if run_id in row['key']] == expected, args
        finally:
            session_cli('secret', 'delete', org_uri, shared, org_only, check=False)
            session_cli('secret', 'delete', db_uri, shared, db_only, check=False)
            session_cli('secret', 'delete', other_db_uri, other_db_only, check=False)
