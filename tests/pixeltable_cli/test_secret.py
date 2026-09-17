"""Tests for `pxt secret set`."""

import http.server
import json
import os
import pathlib
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from .conftest import PxtResult

_RUN_TIMEOUT_SECS = 180.0


@dataclass
class FakeControlPlane:
    """A stand-in for the cloud management API, holding the request bodies it was sent."""

    url: str
    received_requests: list[dict[str, Any]]


@pytest.fixture
def fake_control_plane() -> Iterator[FakeControlPlane]:
    """A control plane on localhost: it remembers every request it was sent, and answers each with {}."""
    received_requests: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received_requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield FakeControlPlane(f'http://127.0.0.1:{server.server_address[1]}', received_requests)
    finally:
        server.shutdown()
        server.server_close()


def _pxt_secret(port: int, cwd: pathlib.Path, control_plane: FakeControlPlane, *args: str) -> PxtResult:
    """Run `pxt secret`. The daemon it starts inherits this environment, and talk to the fake control plane."""
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
    def test_set(self, daemon_port: int, tmp_path: pathlib.Path, fake_control_plane: FakeControlPlane) -> None:
        r = _pxt_secret(
            daemon_port,
            tmp_path,  # any path will do, as long as it's not the repo root or its subdir
            fake_control_plane,
            'set',
            'pxt://acme:main',
            'OPENAI_API_KEY=test=value',
            'CUSTOM_TOKEN=custom-value',
            '--json',
        )
        assert r.returncode == 0, r.stderr
        assert r.json == ['CUSTOM_TOKEN', 'OPENAI_API_KEY']
        expected_requests = [
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
        assert fake_control_plane.received_requests == expected_requests

        # reserved prefix
        for key in ('PIXELTABLE_HOME', 'PIXELTABLE_DB', 'PIXELTABLE_VAR_FOO', 'pixeltable_home', 'Pixeltable_Db'):
            r = _pxt_secret(daemon_port, tmp_path, fake_control_plane, 'set', 'pxt://acme:main', f'{key}=test-value')
            assert r.returncode != 0
            assert 'is reserved' in r.stderr, r.stderr
        assert fake_control_plane.received_requests == expected_requests
