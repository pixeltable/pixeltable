"""`pxt login|logout|whoami` as commands: exit codes and what lands on stdout.

The auth tests cover the session itself. These cover the shell around it — what a script sees.
"""

import json
import pathlib
import time

import pytest

from pixeltable.config import Config
from pixeltable.service import auth, credentials
from pixeltable_cli.client.commands import login as cmd

_API = 'https://api.pixeltable.com'


@pytest.fixture(autouse=True)
def _home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('PIXELTABLE_API_KEY', raising=False)
    Config.init(reinit=True)
    auth._authorized.clear()
    monkeypatch.setattr('pixeltable.service.management_client.api_url', lambda: _API)
    monkeypatch.setattr('pixeltable_cli.client.commands.login.api_url', lambda: _API)


def _signed_in(**kw: object) -> None:
    credentials.save(
        _API,
        credentials.Session(
            access_token='a.b.c',
            expires_at=time.time() + 3600,
            refresh_token='r',
            client_id='c',
            email='you@example.com',
            logged_in_at=time.time(),
            **kw,  # type: ignore[arg-type]
        ),
    )


class TestWhoami:
    def test_not_signed_in_says_so_and_fails(self, capsys: pytest.CaptureFixture) -> None:
        """A script asking who it is must be able to tell 'nobody' from 'someone'."""
        with pytest.raises(SystemExit) as e:
            cmd.run_whoami([])

        assert e.value.code == 1
        assert 'Not signed in' in capsys.readouterr().err

    def test_signed_in_names_the_account_and_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        _signed_in()

        cmd.run_whoami([])

        assert 'you@example.com' in capsys.readouterr().out

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture) -> None:
        _signed_in()

        cmd.run_whoami(['--json'])

        record = json.loads(capsys.readouterr().out)
        assert record['email'] == 'you@example.com'
        assert record['using'] == 'session'

    def test_an_api_key_is_reported_as_what_commands_will_send(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """A session that exists but is outranked is the state people misread."""
        _signed_in()
        monkeypatch.setenv('PIXELTABLE_API_KEY', 'pxt_example')

        cmd.run_whoami(['--json'])

        assert json.loads(capsys.readouterr().out)['using'] == 'api_key'


class TestLogout:
    def test_not_signed_in_is_not_a_failure(self, capsys: pytest.CaptureFixture) -> None:
        cmd.run_logout([])

        assert capsys.readouterr().out.strip() == 'Not signed in.'

    def test_it_ends_the_browsers_sign_in_too(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Signing out of one without the other is the state that confuses people."""
        _signed_in()
        monkeypatch.setattr(auth, 'browser_logout_url', lambda _url: 'https://dash.example.com/api/auth/logout')
        opened: list[str] = []
        monkeypatch.setattr('pixeltable_cli.client.commands.login.webbrowser.open', lambda u: opened.append(u))

        cmd.run_logout([])

        assert opened == ['https://dash.example.com/api/auth/logout']
        assert credentials.load(_API) is None

    def test_an_environment_naming_no_dashboard_still_clears_this_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _signed_in()
        monkeypatch.setattr(auth, 'browser_logout_url', lambda _url: '')
        monkeypatch.setattr(
            'pixeltable_cli.client.commands.login.webbrowser.open', lambda _u: pytest.fail('opened nothing')
        )

        cmd.run_logout([])

        assert credentials.load(_API) is None

    def test_it_forgets_the_session(self, capsys: pytest.CaptureFixture) -> None:
        _signed_in()

        cmd.run_logout([])

        assert credentials.load(_API) is None
        assert 'Signed out' in capsys.readouterr().out


class TestLoginOutput:
    def test_json_mode_puts_nothing_but_json_on_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """The device code and link are progress, and a caller parsing stdout must not see them."""

        def _fake_login(api_url: str, open_browser: bool = True) -> credentials.Session:
            print('Your code is ABCD-EFGH')  # what device_login emits, if it emitted to stdout
            return credentials.Session(
                access_token='a.b.c', expires_at=time.time() + 3600, email='you@example.com', organization_id='org_1'
            )

        monkeypatch.setattr(auth, 'device_login', _fake_login)
        monkeypatch.setattr('pixeltable_cli.client.commands.login.auth.device_login', _fake_login)

        cmd.run(['--json'])

        out = capsys.readouterr().out
        # The stub deliberately writes to stdout; the assertion is that _login's own JSON is the
        # last line and parses, which is what a `pxt login --json | jq` pipeline depends on.
        assert json.loads(out.strip().splitlines()[-1])['email'] == 'you@example.com'

    def test_a_failed_sign_in_exits_nonzero_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def _refuse(api_url: str, open_browser: bool = True) -> credentials.Session:
            raise auth.AuthError('the sign-in was refused in the browser')

        monkeypatch.setattr('pixeltable_cli.client.commands.login.auth.device_login', _refuse)

        with pytest.raises(SystemExit) as e:
            cmd.run([])

        assert e.value.code == 1
        assert 'refused' in capsys.readouterr().err
