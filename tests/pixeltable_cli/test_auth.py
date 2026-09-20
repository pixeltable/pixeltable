"""Tests for `pxt login`, `pxt logout`, `pxt whoami`, `pxt key` and `pxt org create`.

The session commands read and write the daemon's own cache, so a prepared file in it stands in for a
sign-in. The commands that reach the control plane talk to a stub of it, served by a daemon this
module starts with `PIXELTABLE_API_URL` set to it.

The stub also stands in for the sign-in service: it advertises itself as its own OIDC issuer, so
discovery sends every device-flow request back to it.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Iterator

import pytest

from pixeltable import exceptions as excs
from pixeltable.service import auth

from .conftest import PxtResult, PxtRunner

pytestmark = pytest.mark.db_roots('local', reason='the CLI surface under test never reaches a catalog')

_API_URL = 'http://127.0.0.1:{port}'
_A_KEY = 'sk-pxt-test-auth'
_ORG = 'acme'

# what the sign-in service answers when a device code is asked for; interval 0 keeps polling instant
_DEVICE = {
    'device_code': 'dev-code',
    'user_code': 'ABCD-EFGH',
    'verification_uri_complete': 'https://signin.example.com/device?user_code=ABCD-EFGH',
    'expires_in': 300,
    'interval': 0,
}

_PENDING = (400, {'error': 'authorization_pending', 'error_description': 'not yet'})
_SLOW_DOWN = (400, {'error': 'slow_down', 'error_description': 'too fast'})
_DENIED = (400, {'error': 'access_denied', 'error_description': 'refused'})
_REJECTED = (400, {'error': 'invalid_grant', 'error_description': 'token already used'})


def _claims(**claims: Any) -> str:
    """A token whose middle segment decodes to claims. Nothing here verifies a signature."""
    import base64

    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b'=').decode()
    return f'header.{body}.signature'


@dataclass
class ControlPlane:
    """A stub of Pixeltable Cloud: the management API, and the sign-in service behind it.

    One server for both: it advertises itself as its own sign-in issuer, so PIXELTABLE_API_URL is
    the only address a test sets. It replies from `answers` and `tokens`, and records requests in
    `seen` and `token_seen`.
    """

    port: int
    answers: dict[str, Any] = field(default_factory=dict)
    seen: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    client_id: str = 'client_01TEST'
    device: dict[str, Any] = field(default_factory=lambda: dict(_DEVICE))
    tokens: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    token_seen: list[dict[str, str]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return _API_URL.format(port=self.port)

    def last(self, operation: str) -> dict[str, Any]:
        """The most recent request for an operation, which must have been made."""
        matching = [r for r in self.seen if r.get('operation_type') == operation]
        assert len(matching) > 0, f'no {operation} request; saw {[r.get("operation_type") for r in self.seen]}'
        return matching[-1]

    def grant(self, **overrides: Any) -> dict[str, Any]:
        """A token response, with an access token this test can date as it likes."""
        granted = {
            'access_token': _claims(sid='session_01TEST', exp=time.time() + 3600),
            'refresh_token': 'refresh-1',
            'user': {'email': 'you@example.com'},
            'organization_id': 'org_01TEST',
        }
        granted.update(overrides)
        return granted

    def next_token(self, fields: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.token_seen.append(fields)
        return self.tokens.pop(0) if len(self.tokens) > 0 else (200, self.grant())


def _serve(plane: ControlPlane) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, answer: Any) -> None:
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == '/.well-known/pixeltable-auth':
                self._reply(200, {'client_id': plane.client_id, 'workos_api': plane.url})
            else:
                self._reply(404, {})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            if self.path.startswith('/user_management/'):
                fields = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode()).items()}
                if self.path.endswith('/authorize/device'):
                    self._reply(200, plane.device)
                else:
                    self._reply(*plane.next_token(fields))
                return
            request = json.loads(body or b'{}')
            plane.seen.append(request)
            self._reply(plane.status, plane.answers.get(request.get('operation_type'), {}))

        def log_message(self, *_args: Any) -> None:
            pass

    server = HTTPServer(('127.0.0.1', plane.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def control_plane() -> Iterator[ControlPlane]:
    plane = ControlPlane(port=_free_port())
    server = _serve(plane)
    try:
        yield plane
    finally:
        server.shutdown()


@pytest.fixture(scope='module')
def cloud_cli(
    control_plane: ControlPlane, tmp_path_factory: pytest.TempPathFactory, session_project: pathlib.Path
) -> Iterator[PxtRunner]:
    """A CLI runner whose daemon reaches the stub, and signs in and out against it.

    A daemon of its own: the one serving the suite took its environment at spawn, and both the
    address of the management API and the cache the session lands in are read there.
    """
    port = _free_port()
    env = {**os.environ, 'PXT_PORT': str(port), 'BROWSER': 'true', 'PIXELTABLE_API_URL': control_plane.url}
    env.pop('PIXELTABLE_API_KEY', None)
    log_path = tmp_path_factory.mktemp('auth-daemon') / 'daemon.log'
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'pixeltable_cli.server.daemon', '--project-root', str(session_project)],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )

    def run(*args: str, check: bool = True, timeout: float = 120.0) -> PxtResult:
        r = subprocess.run(
            ['pxt', *args], capture_output=True, text=True, env=env, cwd=session_project, timeout=timeout, check=False
        )
        result = PxtResult(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        assert not check or r.returncode == 0, f'pxt {" ".join(args)} failed ({r.returncode}): {r.stderr}'
        return result

    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'auth daemon exited early: {log_path.read_text(errors="replace")[-800:]}')
        if run('health', check=False).returncode == 0:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f'auth daemon did not come up: {log_path.read_text(errors="replace")[-800:]}')
    try:
        yield run
    finally:
        run('daemon', 'stop', '-f', check=False)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(autouse=True)
def signed_in(cloud_cli: PxtRunner, control_plane: ControlPlane) -> Callable[..., None]:
    """Signed in before each test, and a way to sign in again with a different grant.

    Signing in is how a session comes to exist; where it is kept afterwards is the credential
    module's own business.
    """
    control_plane.answers.setdefault('list_orgs', {'orgs': []})

    def _sign_in(**grant: Any) -> None:
        control_plane.tokens[:] = [(200, control_plane.grant(**grant))]
        control_plane.device = dict(_DEVICE)
        cloud_cli('login')
        control_plane.tokens.clear()

    _sign_in()
    return _sign_in


class TestWhoami:
    def test_whoami(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        r = cloud_cli('whoami')
        assert 'you@example.com' in r.stdout
        assert control_plane.url in r.stdout

    def test_whoami_no_session(self, cloud_cli: PxtRunner) -> None:
        cloud_cli('logout')

        r = cloud_cli('whoami', check=False)

        assert r.returncode == 1
        assert 'Not signed in' in r.stderr
        assert 'pxt login' in r.stderr

    def test_whoami_organization(self, cloud_cli: PxtRunner, signed_in: Callable[..., None]) -> None:
        signed_in(organization_id='org_01ACME')

        assert 'Organization: org_01ACME' in cloud_cli('whoami').stdout

    def test_whoami_no_organization(self, cloud_cli: PxtRunner, signed_in: Callable[..., None]) -> None:
        """An account that has not finished onboarding has none, and is told so at sign-in."""
        signed_in(organization_id='')

        assert 'Organization: (none)' in cloud_cli('whoami').stdout

    def test_whoami_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        r = cloud_cli('whoami', '--json')

        assert r.json['email'] == 'you@example.com'
        assert r.json['api_url'] == control_plane.url
        assert r.json['using'] == 'session'
        assert r.json['accepted']

    def test_whoami_rejected(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A cached session says nothing about whether it still works, so whoami asks."""
        control_plane.status = 401
        try:
            r = cloud_cli('whoami', check=False)
        finally:
            control_plane.status = 200

        assert r.returncode == 1
        assert 'not accepted' in r.stderr

    def test_whoami_offline(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """--offline reports the cached session without asking, for a machine with no network."""
        control_plane.status = 401
        try:
            r = cloud_cli('whoami', '--offline')
        finally:
            control_plane.status = 200

        assert 'you@example.com' in r.stdout


class TestLogout:
    def test_logout(self, cloud_cli: PxtRunner) -> None:
        r = cloud_cli('logout')
        assert 'Signed out' in r.stdout

        assert cloud_cli('whoami', check=False).returncode == 1

    def test_logout_no_session(self, cloud_cli: PxtRunner) -> None:
        cloud_cli('logout')

        assert 'Not signed in' in cloud_cli('logout').stdout

    def test_logout_browser(self, cloud_cli: PxtRunner) -> None:
        """Signing out of one and not the other leaves you signed in as someone you did not choose."""
        assert 'Signing out of the browser' in cloud_cli('logout').stdout

    @pytest.mark.parametrize('token', ['', 'not-a-jwt', 'h.%%%.s'])
    def test_logout_no_session_id(self, cloud_cli: PxtRunner, signed_in: Callable[..., None], token: str) -> None:
        """A token with no readable sid points at no browser session; this device is cleared anyway."""
        signed_in(access_token=token or _claims(exp=time.time() + 60))

        r = cloud_cli('logout')

        assert 'Signed out' in r.stdout
        assert 'Signing out of the browser' not in r.stdout


def _key(name: str, key_type: str = 'runtime', grants: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    return {'name': name, 'key_type': key_type, 'grants': grants or [], 'created_at': '2026-09-18T00:00:00Z', **extra}


class TestKey:
    def test_key_list(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['list_keys'] = {
            'keys': [_key('ci', key_type='user'), _key('app', grants=['access:pxt://acme:main/services/ingest'])]
        }

        r = cloud_cli('key', 'list')

        assert 'ci' in r.stdout
        assert 'acts as you' in r.stdout
        assert 'app' in r.stdout
        assert 'main:' in r.stdout

    def test_key_list_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['list_keys'] = {'keys': [_key('ci', key_type='user')]}

        r = cloud_cli('key', 'list', '--json')

        assert [k['name'] for k in r.json['keys']] == ['ci']

    def test_key_list_empty(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['list_keys'] = {'keys': []}

        assert 'No keys.' in cloud_cli('key', 'list').stdout

    def test_key_create(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """The secret is printed once, by create, and cannot be retrieved afterwards."""
        control_plane.answers['create_key'] = {
            'key': _key('app', grants=['access:pxt://acme:main/services/ingest'], api_key='sk-secret-once')
        }

        r = cloud_cli('key', 'create', 'app', '--grant', 'access:pxt://acme:main/services/ingest')

        sent = control_plane.last('create_key')
        assert sent['name'] == 'app'
        assert sent['grants'] == ['access:pxt://acme:main/services/ingest']
        assert 'sk-secret-once' in r.stdout

    def test_key_create_without_grants(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """No grant asks for a key that acts as you; the control plane is told by an empty list."""
        control_plane.answers['create_key'] = {'key': _key('ci', key_type='user', api_key='sk-user-key')}

        r = cloud_cli('key', 'create', 'ci')

        assert control_plane.last('create_key')['grants'] == []
        assert 'acts as you' in r.stdout

    def test_key_create_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_key'] = {'key': _key('app', api_key='sk-secret-once')}

        r = cloud_cli('key', 'create', 'app', '--json')

        assert r.json['name'] == 'app'
        assert r.json['api_key'] == 'sk-secret-once'

    def test_key_create_repeated_grants(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """Repeated and comma-joined flags both flatten, and a duplicate is sent once."""
        control_plane.answers['create_key'] = {'key': _key('app')}

        cloud_cli(
            'key',
            'create',
            'app',
            '--grant',
            'access:pxt://acme:main/services,manage:pxt://acme:main/services',
            '--grant',
            'access:pxt://acme:main/services',
        )

        assert control_plane.last('create_key')['grants'] == [
            'access:pxt://acme:main/services',
            'manage:pxt://acme:main/services',
        ]

    def test_key_update(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['update_key'] = {'key': _key('app', grants=['access:pxt://acme:db2/services'])}

        r = cloud_cli(
            'key',
            'update',
            'app',
            '--grant',
            'access:pxt://acme:db2/services',
            '--revoke',
            'access:pxt://acme:main/services',
        )

        sent = control_plane.last('update_key')
        assert sent['allow'] == ['access:pxt://acme:db2/services']
        assert sent['revoke'] == ['access:pxt://acme:main/services']
        assert 'db2:' in r.stdout

    def test_key_update_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['update_key'] = {'key': _key('app', grants=['access:pxt://acme:db2/services'])}

        r = cloud_cli('key', 'update', 'app', '--grant', 'access:pxt://acme:db2/services', '--json')

        assert r.json['name'] == 'app'

    def test_key_update_no_args(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        before = len(control_plane.seen)

        r = cloud_cli('key', 'update', 'app', check=False)

        assert r.returncode == 2
        assert 'nothing to do' in r.stderr
        assert len(control_plane.seen) == before

    def test_key_delete(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['delete_key'] = {'name': 'app'}

        r = cloud_cli('key', 'delete', 'app')

        assert control_plane.last('delete_key')['name'] == 'app'
        assert r.stdout.strip() == 'app'

    def test_key_delete_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['delete_key'] = {'name': 'app'}

        r = cloud_cli('key', 'delete', 'app', '--json')

        assert r.json == {'name': 'app'}

    @pytest.mark.parametrize(
        'grant',
        ['nonsense', 'read:pxt://acme:main', 'access:acme:main', 'access:pxt://acme', 'access:pxt://acme:main/tables'],
    )
    def test_key_malformed_grant(self, cloud_cli: PxtRunner, control_plane: ControlPlane, grant: str) -> None:
        """Only the shape is checked here, so a typo is answered next to the flag that caused it."""
        before = len(control_plane.seen)

        r = cloud_cli('key', 'create', 'app', '--grant', grant, check=False)

        assert r.returncode == 2
        assert '--grant takes' in r.stderr
        assert len(control_plane.seen) == before


class TestOrgCreate:
    def test_org_create(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_org'] = {'org': _ORG, 'default_db': 'main'}

        r = cloud_cli('org', 'create', _ORG, '--name', 'Acme Inc', '--location', 'aws/us-east-1')

        sent = control_plane.last('create_org')
        assert sent['org'] == _ORG
        assert sent['display_name'] == 'Acme Inc'
        assert sent['location'] == 'aws/us-east-1'
        assert _ORG in r.stdout
        assert 'main' in r.stdout

    def test_org_create_defaults(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A display name and a location are optional, and reach the control plane as null when omitted."""
        control_plane.answers['create_org'] = {'org': _ORG, 'default_db': 'main'}

        cloud_cli('org', 'create', _ORG)

        sent = control_plane.last('create_org')
        assert sent['display_name'] is None
        assert sent['location'] is None

    def test_org_create_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_org'] = {'org': _ORG, 'default_db': 'main'}

        r = cloud_cli('org', 'create', _ORG, '--json')

        assert r.json['org'] == _ORG


class TestLogin:
    """`pxt login` end to end: the client polls, the daemon exchanges, the session lands in its cache."""

    @pytest.fixture(autouse=True)
    def _fresh(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        cloud_cli('logout')
        control_plane.tokens.clear()
        control_plane.token_seen.clear()
        control_plane.device = dict(_DEVICE)

    def test_login(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A pending code is the normal answer while the browser is still open."""
        control_plane.tokens[:] = [_PENDING, (200, control_plane.grant())]

        r = cloud_cli('login')

        assert 'Signed in as you@example.com' in r.stdout
        assert 'ABCD-EFGH' in r.stderr
        assert len(control_plane.token_seen) == 2

    def test_login_no_client_secret(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A distributed binary has none to send, which is why this grant was chosen."""
        cloud_cli('login')

        sent = control_plane.token_seen[0]
        assert 'client_secret' not in sent
        assert sent['grant_type'] == 'urn:ietf:params:oauth:grant-type:device_code'
        assert sent['device_code'] == 'dev-code'

    def test_login_slow_down(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """Backing off is not failing: the next poll still gets the token."""
        control_plane.tokens[:] = [_SLOW_DOWN, (200, control_plane.grant())]

        r = cloud_cli('login')

        assert 'Signed in' in r.stdout
        assert len(control_plane.token_seen) == 2

    def test_login_code_expired(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """The code stops working when the server says so, not when our own timeout runs out."""
        control_plane.device = {**_DEVICE, 'expires_in': 0}

        r = cloud_cli('login', check=False)

        assert r.returncode == 1
        assert 'expired' in r.stderr
        assert len(control_plane.token_seen) == 0

    def test_login_refused(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.tokens[:] = [_DENIED]

        r = cloud_cli('login', check=False)

        assert r.returncode == 1
        assert 'refused in the browser' in r.stderr

    def test_login_failure_output(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """One line saying what the sign-in service refused, and no traceback."""
        control_plane.tokens[:] = [_DENIED]

        r = cloud_cli('login', check=False)

        assert 'Traceback' not in r.stderr
        assert len(r.stderr.strip().splitlines()) <= 3

    def test_login_json(self, cloud_cli: PxtRunner) -> None:
        """The code and the link are progress, so a caller parsing stdout must not see them."""
        r = cloud_cli('login', '--json')

        assert r.json['email'] == 'you@example.com'
        assert r.json['organization_id'] == 'org_01TEST'
        assert 'ABCD-EFGH' not in r.stdout

    def test_login_outlives_the_command(self, cloud_cli: PxtRunner) -> None:
        """The sign-in is for the machine, not for the process that ran it."""
        cloud_cli('login')

        assert 'you@example.com' in cloud_cli('whoami').stdout

    def test_renewal_rotated_token(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """WorkOS rotates on every renewal, so the next one has to present the token it just issued."""
        # a token already past its expiry, so the next command renews before sending anything
        control_plane.tokens[:] = [(200, control_plane.grant(access_token=_claims(exp=time.time() - 1)))]
        cloud_cli('login')
        control_plane.tokens[:] = [(200, control_plane.grant(refresh_token='refresh-2'))]

        cloud_cli('whoami')

        renewal = control_plane.token_seen[-1]
        assert renewal['grant_type'] == 'refresh_token'
        assert renewal['refresh_token'] == 'refresh-1'

    def test_renewal_rejected(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A refresh token the server no longer honors sends you back to the browser."""
        control_plane.tokens[:] = [(200, control_plane.grant(access_token=_claims(exp=time.time() - 1)))]
        cloud_cli('login')
        control_plane.tokens[:] = [_REJECTED]

        r = cloud_cli('whoami', check=False)

        assert r.returncode == 1
        assert 'rejected' in r.stderr


class TestDiscovery:
    """Resolving where to sign in, which the control plane answers."""

    @pytest.fixture
    def plane(self) -> Iterator[ControlPlane]:
        """A stub on a port of its own, so its URL misses the process-wide endpoint cache."""
        plane = ControlPlane(port=_free_port())
        server = _serve(plane)
        try:
            yield plane
        finally:
            server.shutdown()

    def test_sign_in_config(self, plane: ControlPlane) -> None:
        resolved = auth.sign_in_config(plane.url)

        assert resolved.client_id == plane.client_id
        assert resolved.workos_api == plane.url
        assert resolved.url('/user_management/authenticate') == f'{plane.url}/user_management/authenticate'

    def test_control_plane_without_a_client(self, plane: ControlPlane) -> None:
        plane.client_id = ''

        with pytest.raises(excs.Error, match='did not say which sign-in client'):
            auth.sign_in_config(plane.url)
