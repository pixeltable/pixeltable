"""The on-disk session cache: what it keeps apart, what it refuses to leak, what it survives."""

import json
import os
import pathlib
import stat
import time
from typing import Any

import pytest

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.service import management_client, session_cache

from .utils import pxt_raises

_PROD = 'https://api.pixeltable.com'
_DEV = 'https://api.dev.pxt.run'


@pytest.fixture(autouse=True)
def _home(private_home: pathlib.Path) -> pathlib.Path:
    """Each test gets a cache file of its own, and no configured API key."""
    return private_home


def _session(**kw: Any) -> session_cache.Session:
    base = {'access_token': 'at', 'expires_at': time.time() + 3600, 'refresh_token': 'rt', 'client_id': 'client_01TEST'}
    return session_cache.Session(**{**base, **kw})


def _cache_file() -> pathlib.Path:
    return Config.get().home / 'auth' / 'sessions.json'


def _write_cache_file(content: bytes) -> None:
    """Replace the cache file with content, private to this user so that only the content is at fault."""
    path = _cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o600)


class TestRoundTrip:
    def test_no_session(self) -> None:
        assert session_cache.load(_PROD) is None

    def test_round_trip(self) -> None:
        session_cache.save(_PROD, _session(email='a@b.c'))
        loaded = session_cache.load(_PROD)

        assert loaded is not None
        assert (loaded.access_token, loaded.refresh_token, loaded.email) == ('at', 'rt', 'a@b.c')

    def test_keying_by_control_plane(self) -> None:
        """A prod token presented to a sandbox is the failure this keying exists to prevent."""
        session_cache.save(_PROD, _session(access_token='prod-token'))
        session_cache.save(_DEV, _session(access_token='dev-token'))

        assert session_cache.load(_PROD).access_token == 'prod-token'
        assert session_cache.load(_DEV).access_token == 'dev-token'

    def test_replace_on_second_save(self) -> None:
        session_cache.save(_PROD, _session(access_token='first'))
        session_cache.save(_PROD, _session(access_token='second'))

        assert session_cache.load(_PROD).access_token == 'second'


class TestExpiry:
    def test_usable(self) -> None:
        assert _session(expires_at=time.time() + 3600).is_usable()

    def test_expired(self) -> None:
        assert not _session(expires_at=time.time() - 1).is_usable()

    def test_skew(self) -> None:
        """It would likely expire in flight, and a 401 mid-command is worse than refreshing."""
        assert not _session(expires_at=time.time() + session_cache.EXPIRY_SKEW_S / 2).is_usable()

    def test_expires_in(self) -> None:
        now = time.time()
        assert _session(expires_at=now + 120).expires_in(now) == pytest.approx(120, abs=1)

    def test_can_refresh(self) -> None:
        assert not _session(refresh_token=None).can_refresh()
        assert not _session(client_id='').can_refresh()
        assert _session().can_refresh()


class TestClear:
    def test_clear_one(self) -> None:
        session_cache.save(_PROD, _session())
        session_cache.save(_DEV, _session())

        assert session_cache.clear(_PROD) is True
        assert session_cache.load(_PROD) is None
        assert session_cache.load(_DEV) is not None

    def test_clear_all(self) -> None:
        session_cache.save(_PROD, _session())
        session_cache.save(_DEV, _session())

        assert session_cache.clear() is True
        assert session_cache.load(_PROD) is None
        assert session_cache.load(_DEV) is None

    def test_clear_absent(self) -> None:
        assert session_cache.clear(_PROD) is False


class TestFileSafety:
    @pytest.mark.skipif(os.name != 'posix', reason='Windows has no POSIX permissions')
    def test_private_on_posix(self) -> None:
        session_cache.save(_PROD, _session())

        assert stat.S_IMODE(_cache_file().stat().st_mode) == 0o600
        assert stat.S_IMODE(_cache_file().parent.stat().st_mode) == 0o700

    @pytest.mark.skipif(os.name != 'posix', reason='Windows has no POSIX permissions')
    def test_refuses_readable_by_others(self) -> None:
        """A token another user could have copied is not sent, and signing out and in again replaces it.

        Signing out still reads it, for the session id that signs the browser out.
        """
        session_cache.save(_PROD, _session())
        _cache_file().chmod(0o644)

        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='other users'):
            session_cache.load(_PROD)
        assert session_cache.load_for_sign_out(_PROD).access_token == 'at'
        assert session_cache.clear() is True

        session_cache.save(_PROD, _session())
        assert stat.S_IMODE(_cache_file().stat().st_mode) == 0o600
        assert session_cache.load(_PROD) is not None

    @pytest.mark.skipif(os.name != 'posix', reason='Windows has no POSIX permissions')
    def test_readable_by_others_keeps_no_session(self) -> None:
        """Signing in or out of one control plane carries no other session out of a file other users can read.

        Another user could have copied those tokens; rewritten into a private file, they would be usable again.
        """
        session_cache.save(_PROD, _session())
        session_cache.save(_DEV, _session(access_token='dev'))
        _cache_file().chmod(0o644)

        session_cache.save(_PROD, _session(access_token='new'))
        assert stat.S_IMODE(_cache_file().stat().st_mode) == 0o600
        assert session_cache.load(_PROD).access_token == 'new'
        assert session_cache.load(_DEV) is None

        session_cache.save(_DEV, _session(access_token='dev'))
        _cache_file().chmod(0o644)
        assert session_cache.clear(_PROD) is True
        assert not _cache_file().exists()

    @pytest.mark.parametrize('content', [b'{not json', b'["a list"]', b'', b'\xff\xfe\x00'])
    def test_corrupt_file(self, content: bytes) -> None:
        """An unreadable file is reported, a new sign-in replaces it, and signing out deletes it."""
        _write_cache_file(content)

        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='is unreadable'):
            session_cache.load(_PROD)
        assert session_cache.load_for_sign_out(_PROD) is None

        session_cache.save(_PROD, _session(access_token='new'))
        assert session_cache.load(_PROD).access_token == 'new'

        _write_cache_file(content)
        assert session_cache.clear(_PROD) is True
        assert not _cache_file().exists()
        assert session_cache.load(_PROD) is None

    @pytest.mark.skipif(os.name != 'posix', reason='Windows has no POSIX permissions')
    @pytest.mark.skipif(os.name == 'posix' and os.geteuid() == 0, reason='root can read a file whatever its mode')
    def test_unreadable_file(self) -> None:
        """A file its owner cannot read is reported like a corrupt one.

        A new sign-in replaces it, and signing out deletes it.
        """
        session_cache.save(_PROD, _session())
        _cache_file().chmod(0o000)

        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='is unreadable'):
            session_cache.load(_PROD)
        assert session_cache.load_for_sign_out(_PROD) is None

        session_cache.save(_PROD, _session(access_token='new'))
        assert session_cache.load(_PROD).access_token == 'new'

        _cache_file().chmod(0o000)
        assert session_cache.clear(_PROD) is True
        assert not _cache_file().exists()

    @pytest.mark.parametrize(
        'record',
        [
            'not a record',
            {'access_token': 'at'},
            {'access_token': 'at', 'expires_at': 'tomorrow'},
            {'access_token': 'at', 'expires_at': True},
            {'access_token': 7, 'expires_at': 0},
            {'access_token': 'at', 'expires_at': 0, 'refresh_token': 7},
            {'access_token': 'at', 'expires_at': 0, 'email': ['a@b.c']},
        ],
    )
    def test_invalid_record(self, record: Any) -> None:
        """A record that is not a session is reported like an unreadable file, and leaves the others readable.

        Signing out removes it, and a new sign-in replaces it.
        """
        _write_cache_file(json.dumps({_PROD: record, _DEV: {'access_token': 'dev', 'expires_at': 0}}).encode())

        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='is unreadable'):
            session_cache.load(_PROD)
        assert session_cache.load_for_sign_out(_PROD) is None
        assert session_cache.load(_DEV).access_token == 'dev'

        assert session_cache.clear(_PROD) is True
        assert session_cache.load(_PROD) is None

        _write_cache_file(json.dumps({_PROD: record}).encode())
        session_cache.save(_PROD, _session(access_token='new'))
        assert session_cache.load(_PROD).access_token == 'new'

    def test_windows_mode_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows reports a writable file as 0o666, and before Python 3.13 has no os.fchmod()."""
        monkeypatch.setattr(session_cache, '_POSIX', False)
        monkeypatch.delattr(os, 'fchmod', raising=False)

        session_cache.save(_PROD, _session(access_token='first'))
        _cache_file().chmod(0o666)
        assert session_cache.load(_PROD).access_token == 'first'

        session_cache.save(_PROD, _session(access_token='second'))
        _cache_file().chmod(0o666)
        assert session_cache.load(_PROD).access_token == 'second'

        assert session_cache.clear(_PROD) is True
        assert session_cache.load(_PROD) is None


class TestCredentialChoice:
    """Which credential a command sends, and where it came from."""

    def test_api_key_outranks_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting a key is the explicit choice, and what CI runs on."""
        monkeypatch.setenv('PIXELTABLE_API_KEY', 'sk-test')
        session_cache.save(management_client.api_url(), _session())

        cred = management_client.configured_credential()

        assert cred is not None
        assert (cred.kind, cred.value) == ('api_key', 'sk-test')
        assert 'PIXELTABLE_API_KEY' in cred.source

    def test_api_key_from_config_file(self) -> None:
        """The source says "the Pixeltable config file" rather than its path, which is under the Pixeltable home."""
        Config.get().config_file.write_text('[pixeltable]\napi_key = "sk-file"\n', encoding='utf-8')
        Config.init(reinit=True, project_root=Config.get().project_root)

        cred = management_client.configured_credential()

        assert cred is not None
        assert (cred.kind, cred.value, cred.source) == ('api_key', 'sk-file', 'api_key in the Pixeltable config file')

    def test_session_without_key(self) -> None:
        session_cache.save(management_client.api_url(), _session(access_token='session-token'))

        cred = management_client.configured_credential()

        assert cred is not None
        assert (cred.kind, cred.value) == ('session', 'session-token')

    def test_session_with_wrong_types(self) -> None:
        """A string in expires_at is reported as an unreadable sign-in, with the way out, and not as a TypeError."""
        record = {'access_token': 'at', 'expires_at': 'tomorrow', 'refresh_token': 'rt', 'client_id': 'client_01TEST'}
        _write_cache_file(json.dumps({management_client.api_url(): record}).encode())

        with pxt_raises(
            excs.ErrorCode.MISSING_CREDENTIALS, match=r'is unreadable, so it was not used\. Run `pxt logout`, then'
        ):
            management_client.resolve('reach Pixeltable Cloud')

    def test_neither(self) -> None:
        assert management_client.configured_credential() is None

    def test_no_credential(self) -> None:
        """The error says what the credential was for, and both ways to provide one, without a home path."""
        with pxt_raises(
            excs.ErrorCode.MISSING_CREDENTIALS,
            match=r'API key or sign-in is required to reach the home bucket\. Run `pxt login`, or set an API key .*'
            r'in the `\[pixeltable\]` section of the Pixeltable config file\.',
        ) as info:
            management_client.resolve('reach the home bucket')
        assert str(Config.get().home) not in info.value.message
