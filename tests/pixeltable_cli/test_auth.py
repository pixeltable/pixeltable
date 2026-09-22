"""Tests for `pxt login`, `pxt logout`, `pxt whoami`, `pxt org create`, and what `pxt key` refuses.

The session commands read and write the daemon's own cache, so a prepared file in it stands in for a
sign-in. The commands that reach the control plane talk to a stub of it, served by a daemon this
module starts with `PIXELTABLE_API_URL` set to it. The stub also plays the sign-in service, which is
what lets a test script an approval, a rotation or a refusal that real WorkOS will not perform.

What a stub cannot check is whether the control plane stores what it reports, so `pxt key` is
exercised against a real one in test_key.py, and only its client-side refusals remain here.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Iterator

import pytest

from pixeltable import exceptions as excs
from pixeltable.service import auth, session_cache
from pixeltable.utils import cloud_utils
from pixeltable_cli.client.commands import login
from pixeltable_cli.server import routes
from pixeltable_cli.server.router import Request

from ..utils import pxt_raises
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
    # the status and body of the discovery document; None answers with client_id and this stub's address,
    # and a bytes body is sent as it is
    discovery: tuple[int, Any] | None = None
    device: dict[str, Any] = field(default_factory=lambda: dict(_DEVICE))
    tokens: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    token_seen: list[dict[str, str]] = field(default_factory=list)
    token_delay_s: float = 0.0  # how long the token endpoint takes to answer

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
        time.sleep(self.token_delay_s)
        return self.tokens.pop(0) if len(self.tokens) > 0 else (200, self.grant())


def _serve(plane: ControlPlane) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, answer: Any) -> None:
            payload = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == '/.well-known/pixeltable-auth':
                self._reply(*(plane.discovery or (200, {'client_id': plane.client_id, 'workos_api': plane.url})))
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


@pytest.fixture
def fresh_plane() -> Iterator[ControlPlane]:
    """A stub on a port of its own, so its URL misses the process-wide endpoint cache."""
    plane = ControlPlane(port=_free_port())
    server = _serve(plane)
    try:
        yield plane
    finally:
        server.shutdown()


@pytest.fixture(scope='module')
def auth_daemon_port() -> int:
    return _free_port()


def _post_to_daemon(port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST body to the daemon on port the way the CLI does, and return the status and the answer."""
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope='module')
def cloud_cli(
    control_plane: ControlPlane,
    auth_daemon_port: int,
    tmp_path_factory: pytest.TempPathFactory,
    session_project: pathlib.Path,
) -> Iterator[PxtRunner]:
    """A CLI runner whose daemon reaches the stub, and signs in and out against it.

    A daemon of its own: the one serving the suite took its environment at spawn, and both the
    address of the management API and the cache the session lands in are read there.
    """
    port = auth_daemon_port
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
        """An account that has not finished onboarding has none, and is told how to create one."""
        signed_in(organization_id='')

        assert 'No organization yet: create one with `pxt org create NAME`' in cloud_cli('whoami').stdout

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

    def test_whoami_outage(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """An unreachable control plane says nothing about the credential, so it is reported as an outage."""
        control_plane.status = 503
        try:
            r = cloud_cli('whoami', check=False)
        finally:
            control_plane.status = 200

        assert r.returncode == 1
        assert '503' in r.stderr
        assert 'not accepted' not in r.stderr

    def test_whoami_scoped_credential(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A 403 refuses the operation, not the credential, as it does for a key that its grants limit."""
        control_plane.status = 403
        try:
            r = cloud_cli('whoami')
            answer = cloud_cli('whoami', '--json').json
        finally:
            control_plane.status = 200

        assert 'is valid but is not permitted to list_orgs' in r.stdout
        assert 'pxt login' not in r.stdout + r.stderr
        assert answer['accepted']
        assert 'not permitted' in answer['note']

    def test_whoami_json_no_credential(self, cloud_cli: PxtRunner) -> None:
        cloud_cli('logout')

        r = cloud_cli('whoami', '--json', check=False)

        assert r.returncode == 1
        assert r.json['using'] == 'none'
        assert not r.json['accepted']


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

    def test_logout_clears_when_sign_in_service_is_down(
        self, fresh_plane: ControlPlane, private_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forgetting the session takes no network; the browser's sign-out does, and is reported as not done."""
        fresh_plane.discovery = (503, {'error': 'sign-in is not configured for this environment'})
        monkeypatch.setenv('PIXELTABLE_API_URL', fresh_plane.url)
        expires_at = time.time() + 3600
        token = _claims(sid='session_01TEST', exp=expires_at)
        session_cache.save(fresh_plane.url, session_cache.Session(access_token=token, expires_at=expires_at))

        answer = routes.logout(Request(query={}, body_bytes=b'{}'))

        assert answer.signed_out
        assert session_cache.load(fresh_plane.url) is None
        assert answer.browser_logout_url == ''
        assert 'browser could not be signed out' in answer.warning

    def test_logout_without_browser(
        self, control_plane: ControlPlane, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With no browser to open, the sign-out link is printed for the user to open."""
        url = f'{control_plane.url}/user_management/sessions/logout?session_id=session_01TEST'
        answer = {'signed_out': True, 'browser_logout_url': url, 'warning': ''}
        monkeypatch.setattr(login, 'post_request', lambda _path, _body: answer)
        monkeypatch.setattr(webbrowser, 'open', lambda _url: False)

        login.run_logout([])

        assert url in capsys.readouterr().err


def _key(name: str, key_type: str = 'runtime', grants: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    return {'name': name, 'key_type': key_type, 'grants': grants or [], 'created_at': '2026-09-18T00:00:00Z', **extra}


class TestKey:
    def test_key_list_empty(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['list_keys'] = {'keys': []}

        assert 'No keys.' in cloud_cli('key', 'list').stdout

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

    def test_key_list_creator(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """Every member of the organization sees every key, so each one says who created it."""
        control_plane.answers['list_keys'] = {
            'keys': [
                _key('ci', key_type='user', created_by='ada@example.com', created_at='2026-09-18T00:00:00+00:00'),
                _key('app', grants=['access:pxt://acme:main/services/ingest'], created_by='bob@example.com'),
                _key('old', key_type='user'),
            ]
        }

        r = cloud_cli('key', 'list')

        assert 'ci  (acts as ada@example.com:' in r.stdout
        assert 'app  (created by bob@example.com)' in r.stdout
        assert 'old  (acts as its creator:' in r.stdout
        listed = cloud_cli('key', 'list', '--json').json['keys']
        assert listed[0]['created_at'] == '2026-09-18T00:00:00+00:00'

    def test_key_create_creator(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_key'] = {
            'key': _key('ci', key_type='user', created_by='you@example.com', api_key='sk-pxt-created')
        }

        r = cloud_cli('key', 'create', 'ci')

        assert 'ci  (acts as you@example.com:' in r.stdout
        assert 'sk-pxt-created' in r.stdout

    def test_key_update_no_args(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        before = len(control_plane.seen)

        r = cloud_cli('key', 'update', 'app', check=False)

        assert r.returncode == 2
        assert 'nothing to do' in r.stderr
        assert len(control_plane.seen) == before

    def test_key_create_nested_service_grant(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A service is addressed by base_path/name, so its path may have several components."""
        control_plane.answers['create_key'] = {'key': _key('app')}

        cloud_cli('key', 'create', 'app', '--grant', 'access:pxt://acme:main/services/a/b/ingest')

        assert control_plane.last('create_key')['grants'] == ['access:pxt://acme:main/services/a/b/ingest']

    @pytest.mark.parametrize(
        'grant',
        [
            'nonsense',
            'read:pxt://acme:main',
            'access:acme:main',
            'access:pxt://acme',
            'access:pxt://acme:main/tables',
            'access:pxt://acme:main/',
            'access:pxt://acme:main/services/',
            'access:pxt://acme:main/services//ingest',
            'access:pxt://acme:main/services/a/../ingest',
            'access:pxt://acme:main/catalog',
            'manage:pxt://acme:main',
            'access:pxt://acme:main/services/ingest:v1',
        ],
    )
    def test_key_malformed_grant(self, cloud_cli: PxtRunner, control_plane: ControlPlane, grant: str) -> None:
        """A grant the control plane would refuse is answered next to the flag that caused it."""
        before = len(control_plane.seen)

        r = cloud_cli('key', 'create', 'app', '--grant', grant, check=False)

        assert r.returncode == 2
        assert '--grant takes' in r.stderr
        assert len(control_plane.seen) == before


_CREATED_ORG = {'org_id': 'org_01NEW', 'org': _ORG, 'default_db': 'main'}


class TestOrgCreate:
    def test_org_create(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_org'] = dict(_CREATED_ORG)

        r = cloud_cli('org', 'create', _ORG, '--name', 'Acme Inc', '--location', 'aws/us-east-1')

        sent = control_plane.last('create_org')
        assert sent['org'] == _ORG
        assert sent['display_name'] == 'Acme Inc'
        assert sent['location'] == 'aws/us-east-1'
        assert _ORG in r.stdout
        assert 'main' in r.stdout

    def test_org_create_defaults(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A display name and a location are optional, and reach the control plane as null when omitted."""
        control_plane.answers['create_org'] = dict(_CREATED_ORG)

        cloud_cli('org', 'create', _ORG)

        sent = control_plane.last('create_org')
        assert sent['display_name'] is None
        assert sent['location'] is None

    def test_org_create_json(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        control_plane.answers['create_org'] = dict(_CREATED_ORG)

        r = cloud_cli('org', 'create', _ORG, '--json')

        assert r.json['org'] == _ORG

    def test_org_create_switches_session(
        self, cloud_cli: PxtRunner, control_plane: ControlPlane, signed_in: Callable[..., None]
    ) -> None:
        """The control plane takes the organization from the token, so the session is renewed for the new one."""
        signed_in(organization_id='')
        control_plane.answers['create_org'] = dict(_CREATED_ORG)
        control_plane.token_seen.clear()
        control_plane.tokens[:] = [(200, control_plane.grant(organization_id='org_01NEW', refresh_token='refresh-2'))]

        r = cloud_cli('org', 'create', _ORG)

        assert control_plane.token_seen == [
            {
                'grant_type': 'refresh_token',
                'refresh_token': 'refresh-1',
                'client_id': control_plane.client_id,
                'organization_id': 'org_01NEW',
            }
        ]
        assert f'session now uses {_ORG}' in r.stdout
        assert 'Organization: org_01NEW' in cloud_cli('whoami').stdout
        assert len(control_plane.token_seen) == 1

    def test_org_create_switch_refused(
        self, cloud_cli: PxtRunner, control_plane: ControlPlane, signed_in: Callable[..., None]
    ) -> None:
        """The organization exists either way, so a refused renewal is a warning and not a failure."""
        control_plane.answers['create_org'] = dict(_CREATED_ORG)
        signed_in(organization_id='')
        control_plane.tokens[:] = [_REJECTED]

        r = cloud_cli('org', 'create', _ORG)

        assert _ORG in r.stdout
        assert 'pxt login' in r.stderr

        signed_in(organization_id='')
        control_plane.tokens[:] = [_REJECTED]

        r = cloud_cli('org', 'create', _ORG, '--json')

        assert r.json['org'] == _ORG
        assert 'pxt login' in r.stderr

    def test_org_create_with_api_key(
        self, fresh_plane: ControlPlane, private_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An API key belongs to one organization, so creating another one leaves the key where it was."""
        monkeypatch.setenv('PIXELTABLE_API_URL', fresh_plane.url)
        monkeypatch.setenv('PIXELTABLE_API_KEY', _A_KEY)
        fresh_plane.answers['create_org'] = dict(_CREATED_ORG)

        answer = routes.create_org(Request(query={}, body_bytes=json.dumps({'org': _ORG}).encode()))

        assert 'stays bound to its own organization' in answer.warning
        assert answer.session_organization_id == ''
        assert fresh_plane.token_seen == []


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

    def test_login_deadline_inside_interval(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A code that expires before the next poll is due is reported as expired, without that poll."""
        control_plane.device = {**_DEVICE, 'interval': 2, 'expires_in': 1}

        r = cloud_cli('login', check=False)

        assert r.returncode == 1
        assert 'expired before it was confirmed' in r.stderr
        assert len(control_plane.token_seen) == 0

    @pytest.mark.parametrize(
        ('answer', 'reason'),
        [
            ({'error': 'expired_token', 'error_description': 'too late'}, 'the code expired before it was confirmed'),
            (
                {'error': 'invalid_client', 'error_description': 'unknown client'},
                'the sign-in failed (invalid_client: unknown client)',
            ),
        ],
    )
    def test_login_failure_reason(
        self, cloud_cli: PxtRunner, control_plane: ControlPlane, answer: dict[str, str], reason: str
    ) -> None:
        control_plane.tokens[:] = [(400, answer)]

        r = cloud_cli('login', check=False)

        assert r.returncode == 1
        assert reason in r.stderr

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

    def test_login_no_organization(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """Every hosted command needs an organization, so the sign-in says how to create the first one."""
        control_plane.tokens[:] = [(200, control_plane.grant(organization_id=''))]

        r = cloud_cli('login')

        assert 'No organization yet: create one with `pxt org create NAME`' in r.stdout

    def test_login_json(self, cloud_cli: PxtRunner) -> None:
        """The code and the link are progress, so a caller parsing stdout must not see them."""
        r = cloud_cli('login', '--json')

        assert r.json['email'] == 'you@example.com'
        assert r.json['organization_id'] == 'org_01TEST'
        assert 'ABCD-EFGH' not in r.stdout

    def test_login_poll_refuses_unknown_code(
        self, cloud_cli: PxtRunner, control_plane: ControlPlane, auth_daemon_port: int
    ) -> None:
        """Redeeming a code this daemon did not issue would sign this machine in as whoever approved it."""
        status, answer = _post_to_daemon(
            auth_daemon_port,
            '/api/login/poll',
            {'client_id': control_plane.client_id, 'device_code': 'someone-elses-code'},
        )

        assert status == 422
        assert 'not issued by this daemon' in answer['detail']
        assert control_plane.token_seen == []
        assert 'Not signed in' in cloud_cli('whoami', check=False).stderr

    def test_login_outlives_the_command(self, cloud_cli: PxtRunner) -> None:
        """The sign-in is for the machine, not for the process that ran it."""
        cloud_cli('login')

        assert 'you@example.com' in cloud_cli('whoami').stdout

    def test_renewal_rotated_token(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """WorkOS rotates on every renewal, so the next one has to present the token it just issued."""
        # tokens already past their expiry, so each command renews before sending anything
        spent = _claims(exp=time.time() - 1)
        control_plane.tokens[:] = [(200, control_plane.grant(access_token=spent))]
        cloud_cli('login')
        control_plane.tokens[:] = [
            (200, control_plane.grant(access_token=spent, refresh_token='refresh-2')),
            (200, control_plane.grant(refresh_token='refresh-3')),
        ]

        cloud_cli('whoami')
        cloud_cli('whoami')

        first, second = control_plane.token_seen[-2:]
        assert first == {
            'grant_type': 'refresh_token',
            'refresh_token': 'refresh-1',
            'client_id': control_plane.client_id,
            'organization_id': 'org_01TEST',
        }
        assert second['refresh_token'] == 'refresh-2'
        assert second['organization_id'] == 'org_01TEST'

    def test_renewal_outage(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A failing sign-in service decides nothing about the session, which renews once the service is back.

        Its answer is not read as an OAuth error, even one that looks like a refusal.
        """
        control_plane.tokens[:] = [(200, control_plane.grant(access_token=_claims(exp=time.time() - 1)))]
        cloud_cli('login')
        control_plane.tokens[:] = [(503, {'error': 'invalid_grant', 'error_description': 'upstream is down'})]

        r = cloud_cli('whoami', check=False)

        assert r.returncode == 1
        assert 'HTTP 503' in r.stderr
        assert 'pxt login' not in r.stderr

        control_plane.tokens[:] = [(200, control_plane.grant())]
        assert 'you@example.com' in cloud_cli('whoami').stdout

    def test_renewal_rejected(self, cloud_cli: PxtRunner, control_plane: ControlPlane) -> None:
        """A refresh token the server no longer honors sends you back to the browser, and is discarded."""
        control_plane.tokens[:] = [(200, control_plane.grant(access_token=_claims(exp=time.time() - 1)))]
        cloud_cli('login')
        control_plane.tokens[:] = [_REJECTED]

        r = cloud_cli('whoami', check=False)

        assert r.returncode == 1
        assert 'rejected' in r.stderr
        assert 'Not signed in' in cloud_cli('whoami', check=False).stderr


# Renew the session of the control plane at argv[1] once the file at argv[3] exists, having created the one
# at argv[2] to say it is ready.
_RENEW_IN_A_PROCESS = """\
import pathlib
import sys
import time

from pixeltable.service import auth

pathlib.Path(sys.argv[2]).touch()
while not pathlib.Path(sys.argv[3]).exists():
    time.sleep(0.01)
print(auth.access_token(sys.argv[1]))
"""


class TestRenewal:
    """Renewing a spent session, which every process sharing the Pixeltable home does from one file."""

    _CALLERS = 4

    @staticmethod
    def _renew_in_threads(api_url: str, workdir: pathlib.Path) -> list[str]:
        barrier = threading.Barrier(TestRenewal._CALLERS)

        def renew(_i: int) -> str:
            barrier.wait(timeout=30)
            return str(auth.access_token(api_url))

        with ThreadPoolExecutor(max_workers=TestRenewal._CALLERS) as pool:
            return list(pool.map(renew, range(TestRenewal._CALLERS)))

    @staticmethod
    def _renew_in_processes(api_url: str, workdir: pathlib.Path) -> list[str]:
        go = workdir / 'go'
        ready = [workdir / f'ready-{i}' for i in range(TestRenewal._CALLERS)]
        procs = [
            subprocess.Popen(
                [sys.executable, '-c', _RENEW_IN_A_PROCESS, api_url, str(r), str(go)],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for r in ready
        ]
        try:
            deadline = time.time() + 120
            while not all(r.exists() for r in ready):
                exited = [p for p in procs if p.poll() is not None]
                assert len(exited) == 0, [p.communicate() for p in exited]
                assert time.time() < deadline, 'the renewing processes did not start'
                time.sleep(0.05)
            go.touch()
            outputs = [p.communicate(timeout=60) for p in procs]
        finally:
            for p in procs:
                p.kill()
        assert all(p.returncode == 0 for p in procs), [stderr for _stdout, stderr in outputs]
        return [stdout.strip().splitlines()[-1] for stdout, _stderr in outputs]

    @pytest.mark.parametrize('callers', ['threads', 'processes'])
    def test_concurrent_renewals_refresh_once(
        self, fresh_plane: ControlPlane, private_home: pathlib.Path, tmp_path: pathlib.Path, callers: str
    ) -> None:
        """Callers that share a spent session wait for one renewal instead of each spending the refresh token.

        WorkOS refuses a refresh token it has already rotated, so a second renewal would read as a sign-out.
        """
        spent_at = time.time() - 1
        session_cache.save(
            fresh_plane.url,
            session_cache.Session(
                access_token=_claims(exp=spent_at),
                expires_at=spent_at,
                refresh_token='refresh-1',
                client_id=fresh_plane.client_id,
            ),
        )
        renewed = fresh_plane.grant(refresh_token='refresh-2')
        fresh_plane.tokens[:] = [(200, renewed)]
        # long enough for every caller to arrive while the first renewal is in flight
        fresh_plane.token_delay_s = 0.3

        renew = self._renew_in_threads if callers == 'threads' else self._renew_in_processes
        tokens = renew(fresh_plane.url, tmp_path)

        assert len(fresh_plane.token_seen) == 1
        assert tokens == [renewed['access_token']] * self._CALLERS
        assert session_cache.load(fresh_plane.url).refresh_token == 'refresh-2'


class TestHomeBucket:
    """The home bucket's calls to the control plane send the credential a management call sends."""

    @pytest.mark.parametrize(
        ('status', 'code', 'message'),
        [
            (
                401,
                excs.ErrorCode.PROVIDER_AUTH_ERROR,
                'API key from the PIXELTABLE_API_KEY environment variable was rejected',
            ),
            (403, excs.ErrorCode.INSUFFICIENT_PRIVILEGES, 'is valid but is not permitted to get_bucket_credentials'),
        ],
    )
    def test_refused_credential(
        self,
        fresh_plane: ControlPlane,
        private_home: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        code: excs.ErrorCode,
        message: str,
    ) -> None:
        """A refused credential is reported as such, and is not retried as an unreachable control plane is."""
        monkeypatch.setenv('PIXELTABLE_API_URL', fresh_plane.url)
        monkeypatch.setenv('PIXELTABLE_API_KEY', _A_KEY)
        fresh_plane.status = status

        with pxt_raises(code, match=message):
            cloud_utils.get_bucket_credentials('acme', 'main', 'home')


class TestDiscovery:
    """Resolving where to sign in, which the control plane answers."""

    def test_sign_in_config(self, fresh_plane: ControlPlane) -> None:
        resolved = auth.sign_in_config(fresh_plane.url)

        assert resolved.client_id == fresh_plane.client_id
        assert resolved.workos_api == fresh_plane.url
        assert resolved.url('/user_management/authenticate') == f'{fresh_plane.url}/user_management/authenticate'

    def test_control_plane_without_client(self, fresh_plane: ControlPlane) -> None:
        fresh_plane.client_id = ''

        with pxt_raises(excs.ErrorCode.INTERNAL_ERROR, match='did not say which sign-in client'):
            auth.sign_in_config(fresh_plane.url)

    def test_sign_in_not_configured(self, fresh_plane: ControlPlane) -> None:
        """The control plane's own reason is the actionable part."""
        fresh_plane.discovery = (503, {'error': 'sign-in is not configured for this environment'})

        with pxt_raises(
            excs.ErrorCode.PROVIDER_ERROR, match='cannot sign you in: sign-in is not configured for this environment'
        ):
            auth.sign_in_config(fresh_plane.url)

    @pytest.mark.parametrize('body', [b'', b'<html></html>', b'null'])
    def test_control_plane_without_login(self, fresh_plane: ControlPlane, body: bytes) -> None:
        """A control plane that predates `pxt login` answers the discovery path with an empty 200."""
        fresh_plane.discovery = (200, body)

        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION, match='does not support `pxt login` yet'):
            auth.sign_in_config(fresh_plane.url)
