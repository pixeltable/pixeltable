"""Keeping a session alive across commands, which is what makes an API key optional."""

import base64
import email.message
import io
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pytest

from pixeltable.config import Config
from pixeltable.service import auth, credentials

_API = 'https://api.pixeltable.com'
_LOGIN = 'https://acme.app.pixeltable.com'


@pytest.fixture(autouse=True)
def _home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Config caches home at first access, so each test needs its own instance to get its own file.
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    Config.init(reinit=True)


def _session(**kw: Any) -> credentials.Session:
    base = {
        'access_token': 'old-access',
        'expires_at': time.time() + 3600,
        'sealed_session': 'old-sealed',
        'login_url': _LOGIN,
        'logged_in_at': time.time(),
    }
    return credentials.Session(**{**base, **kw})


def _stub(monkeypatch: pytest.MonkeyPatch, payload: dict, calls: list | None = None) -> None:
    def _get(url: str, headers: dict | None = None, data: bytes | None = None) -> dict:
        if calls is not None:
            calls.append((url, headers or {}))
        return payload

    monkeypatch.setattr(auth, '_get_json', _get)


class TestAccessToken:
    def test_no_session_is_not_an_error(self) -> None:
        """The caller falls back to an API key, so this must not raise."""
        assert auth.access_token(_API) is None

    def test_a_live_token_is_returned_without_a_network_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, '_get_json', lambda *a, **k: pytest.fail('renewed a live token'))
        credentials.save(_API, _session())

        assert auth.access_token(_API) == 'old-access'

    def test_an_expired_token_is_refreshed_transparently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The token TTL is invisible: the command gets a token, not an error."""
        _stub(monkeypatch, {'token': 'new-access', 'session': 'new-sealed'})
        credentials.save(_API, _session(expires_at=time.time() - 1))

        assert auth.access_token(_API) == 'new-access'

    def test_a_token_dying_inside_the_skew_is_refreshed_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub(monkeypatch, {'token': 'new-access'})
        credentials.save(_API, _session(expires_at=time.time() + credentials.EXPIRY_SKEW_S / 2))

        assert auth.access_token(_API) == 'new-access'

    def test_a_sign_in_past_the_deadline_is_refused_rather_than_renewed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cached token is still live here. Sending it anyway is what the deadline prevents."""
        monkeypatch.setattr(auth, '_get_json', lambda *a, **k: pytest.fail('renewed an expired sign-in'))
        credentials.save(
            _API, _session(logged_in_at=time.time() - credentials.MAX_SESSION_AGE_S - 1, expires_at=time.time() + 3600)
        )

        with pytest.raises(auth.AuthError) as e:
            auth.access_token(_API)
        assert 'expired' in str(e.value)


class TestRefresh:
    def test_the_rotated_pair_is_persisted_before_it_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A spent refresh token left on disk kills the session on the next command."""
        _stub(monkeypatch, {'token': 'new-access', 'session': 'new-sealed'})
        credentials.save(_API, _session(expires_at=time.time() - 1))

        auth.access_token(_API)
        stored = credentials.load(_API)

        assert (stored.access_token, stored.sealed_session) == ('new-access', 'new-sealed')

    def test_renewing_does_not_move_the_sign_in_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise a session in daily use would never require a browser again."""
        _stub(monkeypatch, {'token': 'new-access', 'session': 'new-sealed'})
        signed_in_at = time.time() - 300
        renewed = auth.refresh(_API, _session(logged_in_at=signed_in_at, expires_at=time.time() - 1))

        assert renewed.logged_in_at == signed_in_at

    def test_a_response_without_a_rotated_session_keeps_the_old_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A response that omits it means unchanged; dropping it would end the session early."""
        _stub(monkeypatch, {'token': 'new-access'})
        credentials.save(_API, _session(expires_at=time.time() - 1))

        auth.access_token(_API)

        assert credentials.load(_API).sealed_session == 'old-sealed'

    def test_it_presents_the_sealed_session_to_the_dashboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In a header, not a URL: the sealed session is a credential."""
        calls: list = []
        _stub(monkeypatch, {'token': 'a'}, calls)
        credentials.save(_API, _session(expires_at=time.time() - 1))

        auth.access_token(_API)
        url, headers = calls[0]

        assert url == f'{_LOGIN}/api/auth/cli/token'
        assert headers['Authorization'] == 'Bearer old-sealed'

    def test_a_refused_refresh_is_reported_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Looking signed-out would send the caller to an API key instead of saying 'sign in again'."""
        refused = auth.AuthError('session expired', code='session_expired')
        monkeypatch.setattr(auth, '_get_json', lambda *a, **k: (_ for _ in ()).throw(refused))
        credentials.save(_API, _session(expires_at=time.time() - 1))

        with pytest.raises(auth.AuthError, match='session expired'):
            auth.access_token(_API)


class TestErrorsAreLegible:
    """An OAuth server reports failures in the body, not the status line."""

    def _http_error(self, body: bytes, status: int = 403) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            'https://issuer/oauth2/token', status, 'Forbidden', email.message.Message(), io.BytesIO(body)
        )

    def test_the_oauth_error_reaches_the_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HTTP 403` alone hides the diagnosis; the description is the whole message."""
        body = b'{"error":"invalid_client","error_description":"Application not found."}'
        monkeypatch.setattr(
            auth.urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(self._http_error(body))
        )

        with pytest.raises(auth.AuthError, match='invalid_client: Application not found'):
            auth._get_json('https://issuer/oauth2/token')

    def test_the_error_code_is_kept_separately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Polling branches on it, so it cannot only exist inside the formatted message."""
        body = b'{"error":"authorization_pending","error_description":"Still waiting."}'
        monkeypatch.setattr(
            auth.urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(self._http_error(body))
        )

        with pytest.raises(auth.AuthError) as exc:
            auth._get_json('https://issuer/oauth2/token')

        assert exc.value.code == 'authorization_pending'

    def test_a_non_json_body_still_says_something(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            auth.urllib.request,
            'urlopen',
            lambda *a, **k: (_ for _ in ()).throw(self._http_error(b'<html>502</html>', 502)),
        )

        with pytest.raises(auth.AuthError, match='HTTP 502'):
            auth._get_json('https://issuer/oauth2/token')

    def test_an_unreachable_host_is_not_a_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth.urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(OSError('no route')))

        with pytest.raises(auth.AuthError, match='could not reach'):
            auth._get_json('https://issuer/oauth2/token')

    def test_a_session_with_nothing_renewable_is_reported(self) -> None:
        credentials.save(_API, _session(expires_at=time.time() - 1, sealed_session=None))

        with pytest.raises(auth.AuthError, match='cannot be renewed'):
            auth.access_token(_API)

    def test_each_environment_refreshes_its_own_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub(monkeypatch, {'token': 'prod-new'})
        credentials.save(_API, _session(expires_at=time.time() - 1))
        credentials.save('https://api.dev.pxt.run', _session(access_token='dev-live'))

        assert auth.access_token(_API) == 'prod-new'
        assert auth.access_token('https://api.dev.pxt.run') == 'dev-live'


class TestBrowserLogin:
    """The handoff from browser to CLI, where the state parameter is the whole defence."""

    def _served(self, monkeypatch: pytest.MonkeyPatch, reply: dict, opened: list) -> None:
        """Run browser_login with the browser replaced by something that answers the callback."""

        def _open(target: str) -> bool:
            opened.append(target)
            query = urllib.parse.parse_qs(urllib.parse.urlparse(target).query)
            callback = query['callback'][0]
            payload = dict(reply)
            if payload.pop('echo_state', False):
                payload['state'] = query['state'][0]
            urllib.request.urlopen(f'{callback}?{urllib.parse.urlencode(payload)}', timeout=5).read()
            return True

        monkeypatch.setattr(auth, 'login_url_for', lambda url: _LOGIN)
        monkeypatch.setattr('webbrowser.open', _open)

    def test_a_matching_reply_is_saved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list = []
        self._served(monkeypatch, {'echo_state': True, 'session': 'sealed-1', 'email': 'a@b.c'}, opened)
        monkeypatch.setattr(auth, 'refresh', lambda url, s: s)

        session = auth.browser_login(_API)

        assert (session.sealed_session, session.email, session.login_url) == ('sealed-1', 'a@b.c', _LOGIN)

    def test_the_callback_is_on_loopback_and_carries_high_entropy_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list = []
        self._served(monkeypatch, {'echo_state': True, 'session': 's', 'access_token': 'tok'}, opened)

        auth.browser_login(_API)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)

        assert query['callback'][0].startswith('http://127.0.0.1:')
        assert len(query['state'][0]) >= 32

    def test_a_reply_with_the_wrong_state_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Another local process racing the port must not be able to plant a session."""
        opened: list = []
        self._served(monkeypatch, {'state': 'not-the-one', 'session': 'attacker'}, opened)

        with pytest.raises(auth.AuthError, match='did not match'):
            auth.browser_login(_API)

        assert credentials.load(_API) is None, 'a mismatched reply must save nothing'

    def test_a_reply_with_no_state_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list = []
        self._served(monkeypatch, {'session': 'attacker'}, opened)

        with pytest.raises(auth.AuthError, match='did not match'):
            auth.browser_login(_API)

    def test_a_reply_with_no_session_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list = []
        self._served(monkeypatch, {'echo_state': True, 'error': 'you denied it'}, opened)

        with pytest.raises(auth.AuthError, match='you denied it'):
            auth.browser_login(_API)


class TestTokenExpiry:
    def test_the_expiry_comes_from_the_token_itself(self) -> None:
        exp = int(time.time()) + 900
        claims = base64.urlsafe_b64encode(json.dumps({'exp': exp}).encode()).decode().rstrip('=')

        assert auth._expiry_from(f'header.{claims}.sig') == pytest.approx(exp, abs=1)

    @pytest.mark.parametrize('token', ['', 'not-a-jwt', 'a.!!!.c', 'a.eyJubyI6ImV4cCJ9.c'])
    def test_an_unreadable_token_falls_back_rather_than_failing(self, token: str) -> None:
        """A token we cannot parse still works; only the renewal schedule is guessed."""
        assert auth._expiry_from(token) == pytest.approx(time.time() + 300, abs=2)
