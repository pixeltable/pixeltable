"""Signing in with a device code, and keeping that session alive across commands."""

import base64
import contextlib
import email.message
import io
import json
import pathlib
import time
import types
import urllib.error
from typing import Any

import pytest

from pixeltable.config import Config
from pixeltable.service import auth, credentials

_API = 'https://api.pixeltable.com'
_CLIENT = 'client_01TEST'
_DEVICE = {
    'device_code': 'dev-code',
    'user_code': 'ABCD-EFGH',
    'verification_uri_complete': 'https://signin.example.com/device?user_code=ABCD-EFGH',
    'expires_in': 300,
    'interval': 5,
}
_GRANTED = {
    'access_token': 'new-access',
    'refresh_token': 'new-refresh',
    'user': {'email': 'you@example.com'},
    'organization_id': 'org_01TEST',
}


@pytest.fixture(autouse=True)
def _home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Config caches home at first access, so each test needs its own instance to get its own file.
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    Config.init(reinit=True)
    # Polling is the one place this code sleeps; no test should wait on it.
    monkeypatch.setattr(auth.time, 'sleep', lambda _s: None)
    # Per-process state, so each test starts having authorized nothing.
    auth._authorized.clear()


def _session(**kw: Any) -> credentials.Session:
    base = {
        'access_token': 'old-access',
        'expires_at': time.time() + 3600,
        'refresh_token': 'old-refresh',
        'client_id': _CLIENT,
        'logged_in_at': time.time(),
    }
    return credentials.Session(**{**base, **kw})


def _oauth_error(code: str) -> auth.AuthError:
    return auth.AuthError(f'{code}: because', code=code)


class _Calls:
    """Answers each POST in turn, recording what was asked for."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.seen: list[tuple[str, dict[str, str]]] = []

    def __call__(self, path: str, fields: dict[str, str]) -> dict[str, Any]:
        self.seen.append((path, fields))
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return answer


def _posting(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> _Calls:
    calls = _Calls(*answers)
    monkeypatch.setattr(auth, '_post_form', calls)
    monkeypatch.setattr(auth, 'client_id_for', lambda _url: _CLIENT)
    return calls


class TestDeviceLogin:
    def test_it_polls_until_the_code_is_approved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pending code is the normal case, not a failure: the user is still in the browser."""
        calls = _posting(monkeypatch, _DEVICE, _oauth_error('authorization_pending'), _GRANTED)

        session = auth.device_login(_API, open_browser=False)

        assert session.access_token == 'new-access'
        assert [path for path, _ in calls.seen] == [auth._DEVICE_AUTH_PATH, auth._TOKEN_PATH, auth._TOKEN_PATH]

    def test_it_asks_for_the_device_grant_and_sends_no_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A distributed binary has no client secret to send, which is why this flow was chosen."""
        calls = _posting(monkeypatch, _DEVICE, _GRANTED)

        auth.device_login(_API, open_browser=False)

        _, fields = calls.seen[1]
        assert fields['grant_type'] == 'urn:ietf:params:oauth:grant-type:device_code'
        assert fields['device_code'] == 'dev-code'
        assert not any('secret' in key for key in fields)

    def test_the_session_it_saves_can_renew_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _posting(monkeypatch, _DEVICE, _GRANTED)

        auth.device_login(_API, open_browser=False)

        saved = credentials.load(_API)
        assert saved is not None
        assert (saved.refresh_token, saved.client_id) == ('new-refresh', _CLIENT)
        assert saved.can_refresh()

    def test_it_records_the_organization_the_token_carries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Signing in is not the same as having somewhere to work; `pxt login` says so separately."""
        _posting(monkeypatch, _DEVICE, _GRANTED)

        assert auth.device_login(_API, open_browser=False).organization_id == 'org_01TEST'

    def test_an_account_with_no_organization_still_signs_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _posting(monkeypatch, _DEVICE, {**_GRANTED, 'organization_id': None})

        assert auth.device_login(_API, open_browser=False).organization_id == ''

    def test_it_polls_at_the_interval_workos_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        waits: list[float] = []
        monkeypatch.setattr(auth.time, 'sleep', lambda s: waits.append(s))
        _posting(monkeypatch, {**_DEVICE, 'interval': 9}, _GRANTED)

        auth.device_login(_API, open_browser=False)

        assert waits == [9]

    def test_slow_down_backs_off_rather_than_failing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        waits: list[float] = []
        monkeypatch.setattr(auth.time, 'sleep', lambda s: waits.append(s))
        _posting(monkeypatch, _DEVICE, _oauth_error('slow_down'), _GRANTED)

        auth.device_login(_API, open_browser=False)

        assert waits[1] > waits[0]

    def test_progress_does_not_pollute_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """`pxt login --json` promises a parseable document; the code and link are progress."""
        _posting(monkeypatch, _DEVICE, _GRANTED)

        auth.device_login(_API, open_browser=False)

        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'ABCD-EFGH' in captured.err

    def test_it_backs_off_by_five_seconds_when_told_to_slow_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RFC 8628 says five; less keeps polling faster than the server allows."""
        waits: list[float] = []
        monkeypatch.setattr(auth.time, 'sleep', lambda s: waits.append(s))
        _posting(monkeypatch, {**_DEVICE, 'interval': 5}, _oauth_error('slow_down'), _GRANTED)

        auth.device_login(_API, open_browser=False)

        assert waits == [5, 10]

    def test_a_non_object_answer_is_an_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """json.loads happily returns a list; callers here index it as a mapping."""
        body = contextlib.nullcontext(types.SimpleNamespace(read=lambda: b'[]'))
        monkeypatch.setattr(auth.urllib.request, 'urlopen', lambda *a, **k: body)

        with pytest.raises(auth.AuthError, match='not an object'):
            auth.auth_config('https://api.example.com')

    def test_a_refusal_in_the_browser_is_reported_as_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _posting(monkeypatch, _DEVICE, _oauth_error('access_denied'))

        with pytest.raises(auth.AuthError, match='refused'):
            auth.device_login(_API, open_browser=False)

    def test_an_unexpected_error_is_not_mistaken_for_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Polling forever on a real failure is worse than stopping: nothing tells the user."""
        _posting(monkeypatch, _DEVICE, _oauth_error('invalid_client'))

        with pytest.raises(auth.AuthError, match='invalid_client'):
            auth.device_login(_API, open_browser=False)

    def test_it_honours_the_deadline_workos_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The code stops working when WorkOS says so, not when our own timeout runs out."""
        calls = _posting(monkeypatch, {**_DEVICE, 'expires_in': 0})

        with pytest.raises(auth.AuthError, match='expired'):
            auth.device_login(_API, open_browser=False)
        assert [path for path, _ in calls.seen] == [auth._DEVICE_AUTH_PATH]

    def test_an_environment_that_names_no_client_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, '_request', lambda *a, **k: {'login_url': 'https://x.example.com'})

        with pytest.raises(auth.AuthError, match='which sign-in client'):
            auth.device_login(_API, open_browser=False)


class TestRefresh:
    def test_it_presents_the_refresh_token_and_no_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _posting(monkeypatch, _GRANTED)

        auth.refresh(_API, _session())

        _, fields = calls.seen[0]
        assert fields == {'grant_type': 'refresh_token', 'refresh_token': 'old-refresh', 'client_id': _CLIENT}

    def test_the_rotated_token_is_persisted_before_it_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WorkOS rotates on every refresh; keeping the spent one fails the next renewal instead."""
        _posting(monkeypatch, _GRANTED)

        renewed = auth.refresh(_API, _session())

        assert renewed.refresh_token == 'new-refresh'
        saved = credentials.load(_API)
        assert saved is not None and saved.refresh_token == 'new-refresh'

    def test_renewing_does_not_move_the_sign_in_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise a session in daily use would never require a browser again."""
        _posting(monkeypatch, _GRANTED)
        signed_in_at = time.time() - 300

        assert auth.refresh(_API, _session(logged_in_at=signed_in_at)).logged_in_at == signed_in_at

    def test_it_can_scope_the_token_to_an_organization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """How a session signed in before its organization existed picks it up."""
        calls = _posting(monkeypatch, {**_GRANTED, 'organization_id': 'org_01NEW'})
        credentials.save(_API, _session())

        assert auth.authorize_org(_API, 'org_01NEW').organization_id == 'org_01NEW'
        assert calls.seen[0][1]['organization_id'] == 'org_01NEW'

    def test_a_plain_renewal_asks_for_no_organization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WorkOS keeps the one the session already had; naming it here could only narrow it."""
        calls = _posting(monkeypatch, _GRANTED)

        auth.refresh(_API, _session())

        assert 'organization_id' not in calls.seen[0][1]

    def test_a_session_with_nothing_renewable_is_reported(self) -> None:
        with pytest.raises(auth.AuthError, match='cannot be renewed'):
            auth.refresh(_API, _session(refresh_token=None))

    def test_a_refused_refresh_is_reported_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _posting(monkeypatch, _oauth_error('invalid_grant'))

        with pytest.raises(auth.AuthError, match='invalid_grant'):
            auth.refresh(_API, _session())


class TestAccessToken:
    def test_no_session_is_not_an_error(self) -> None:
        """The caller falls back to an API key, so this must not raise."""
        assert auth.access_token(_API) is None

    def test_a_live_token_is_returned_without_a_network_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, '_post_form', lambda *a, **k: pytest.fail('renewed a live token'))
        credentials.save(_API, _session())

        assert auth.access_token(_API) == 'old-access'

    def test_an_expired_token_is_refreshed_transparently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The token TTL is invisible: the command gets a token, not an error."""
        _posting(monkeypatch, _GRANTED)
        credentials.save(_API, _session(expires_at=time.time() - 1))

        assert auth.access_token(_API) == 'new-access'

    def test_a_token_dying_inside_the_skew_is_refreshed_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _posting(monkeypatch, _GRANTED)
        credentials.save(_API, _session(expires_at=time.time() + credentials.EXPIRY_SKEW_S / 2))

        assert auth.access_token(_API) == 'new-access'

    def test_a_sign_in_past_the_deadline_is_refused_rather_than_renewed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cached token is still live here. Sending it anyway is what the deadline prevents."""
        monkeypatch.setattr(auth, '_post_form', lambda *a, **k: pytest.fail('renewed an expired sign-in'))
        credentials.save(
            _API, _session(logged_in_at=time.time() - credentials.MAX_SESSION_AGE_S - 1, expires_at=time.time() + 3600)
        )

        with pytest.raises(auth.AuthError, match='expired'):
            auth.access_token(_API)

    def test_work_already_under_way_is_not_cut_off_at_the_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bulk ingest that began inside the hour reconnects after it; failing there loses work."""
        _posting(monkeypatch, _GRANTED)
        credentials.save(_API, _session())

        assert auth.access_token(_API) == 'old-access'  # authorizes this process
        credentials.save(_API, _session(logged_in_at=time.time() - credentials.MAX_SESSION_AGE_S - 1))

        assert auth.access_token(_API) == 'old-access'  # still serving, deadline notwithstanding

    def test_each_environment_refreshes_its_own_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A prod token presented to a sandbox is the failure this keying exists to prevent."""
        _posting(monkeypatch, _GRANTED)
        credentials.save(_API, _session(expires_at=time.time() - 1))
        credentials.save('https://api.dev.pxt.run', _session(access_token='dev-live'))

        assert auth.access_token(_API) == 'new-access'
        assert auth.access_token('https://api.dev.pxt.run') == 'dev-live'


class TestErrorsAreLegible:
    """An OAuth server reports failures in the body, not the status line."""

    def _http_error(self, body: bytes, status: int = 403) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            'https://api.workos.com/user_management/authenticate',
            status,
            'Forbidden',
            email.message.Message(),
            io.BytesIO(body),
        )

    def _raising(self, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
        monkeypatch.setattr(auth.urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(error))

    def test_the_oauth_error_reaches_the_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HTTP 403` alone hides the diagnosis; the description is the whole message."""
        body = b'{"error":"invalid_client","error_description":"Application not found."}'
        self._raising(monkeypatch, self._http_error(body))

        with pytest.raises(auth.AuthError, match='Application not found'):
            auth._post_form('/x', {})

    def test_the_error_code_is_kept_separately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Polling switches on the code, so it cannot only be formatted into the message."""
        self._raising(monkeypatch, self._http_error(b'{"error":"authorization_pending"}'))

        with pytest.raises(auth.AuthError) as e:
            auth._post_form('/x', {})
        assert e.value.code == 'authorization_pending'

    def test_a_non_json_body_still_says_something(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._raising(monkeypatch, self._http_error(b'<html>gateway timeout</html>', status=504))

        with pytest.raises(auth.AuthError, match='504'):
            auth._post_form('/x', {})

    def test_a_200_that_is_not_json_is_not_a_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An environment too old to serve discovery answers 200 with nothing in it."""

        empty = contextlib.nullcontext(types.SimpleNamespace(read=lambda: b''))
        monkeypatch.setattr(auth.urllib.request, 'urlopen', lambda *a, **k: empty)

        with pytest.raises(auth.AuthError, match='did not answer with JSON'):
            auth.auth_config('https://api.example.com')

    def test_an_unreachable_host_is_not_a_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._raising(monkeypatch, OSError('name resolution failed'))

        with pytest.raises(auth.AuthError, match='could not reach'):
            auth._post_form('/x', {})


class TestTokenExpiry:
    def test_the_expiry_comes_from_the_token_itself(self) -> None:
        exp = int(time.time()) + 1234
        claims = base64.urlsafe_b64encode(json.dumps({'exp': exp}).encode()).rstrip(b'=').decode()

        assert auth._expiry_from(f'h.{claims}.s') == pytest.approx(exp)

    @pytest.mark.parametrize('token', ['', 'not-a-jwt', 'h.%%%.s', 'h.e30.s'])
    def test_an_unreadable_token_falls_back_rather_than_failing(self, token: str) -> None:
        """A token we cannot read is one to renew early, not one to refuse."""
        assert auth._expiry_from(token, default_s=60) == pytest.approx(time.time() + 60, abs=2)
