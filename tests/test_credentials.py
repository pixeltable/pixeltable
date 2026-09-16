"""The on-disk session cache: what it keeps apart, what it refuses to leak, what it survives."""

import json
import os
import stat
import time

import pytest

from pixeltable.config import Config

from pixeltable.service import credentials

_PROD = 'https://api.pixeltable.com'
_DEV = 'https://api.dev.pxt.run'


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    # Config caches home at first access, so each test needs its own instance to get its own file.
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    Config.init(reinit=True)


def _session(**kw) -> credentials.Session:
    base = {
        'access_token': 'at',
        'expires_at': time.time() + 3600,
        'sealed_session': 'sealed',
        'login_url': 'https://acme.app.pixeltable.com',
    }
    return credentials.Session(**{**base, **kw})


class TestRoundTrip:
    def test_not_signed_in_is_none_not_an_error(self) -> None:
        assert credentials.load(_PROD) is None

    def test_a_saved_session_comes_back(self) -> None:
        credentials.save(_PROD, _session(email='a@b.c'))
        loaded = credentials.load(_PROD)

        assert loaded is not None
        assert (loaded.access_token, loaded.sealed_session, loaded.email) == ('at', 'sealed', 'a@b.c')

    def test_environments_do_not_share_a_session(self) -> None:
        """A prod token presented to a sandbox is the failure this keying exists to prevent."""
        credentials.save(_PROD, _session(access_token='prod-token'))
        credentials.save(_DEV, _session(access_token='dev-token'))

        assert credentials.load(_PROD).access_token == 'prod-token'
        assert credentials.load(_DEV).access_token == 'dev-token'
        assert credentials.signed_in() == sorted([_PROD, _DEV])

    def test_signing_in_again_replaces_rather_than_accumulates(self) -> None:
        credentials.save(_PROD, _session(access_token='first'))
        credentials.save(_PROD, _session(access_token='second'))

        assert credentials.load(_PROD).access_token == 'second'
        assert credentials.signed_in() == [_PROD]


class TestExpiry:
    def test_a_live_token_is_usable(self) -> None:
        assert _session(expires_at=time.time() + 3600).is_usable()

    def test_an_expired_token_is_not(self) -> None:
        assert not _session(expires_at=time.time() - 1).is_usable()

    def test_a_token_dying_inside_the_skew_is_refused_early(self) -> None:
        """It would likely expire in flight, and a 401 mid-command is worse than refreshing."""
        assert not _session(expires_at=time.time() + credentials.EXPIRY_SKEW_S / 2).is_usable()

    def test_expires_in_is_reported_without_the_skew(self) -> None:
        now = time.time()
        assert _session(expires_at=now + 120).expires_in(now) == pytest.approx(120, abs=1)

    def test_a_session_without_a_sealed_session_cannot_renew(self) -> None:
        assert not _session(sealed_session=None).can_refresh()
        assert not _session(login_url='').can_refresh()
        assert _session().can_refresh()


class TestOnDisk:
    def test_the_file_is_not_readable_by_anyone_else(self) -> None:
        credentials.save(_PROD, _session())
        mode = stat.S_IMODE(os.stat(credentials._path()).st_mode)

        assert mode == 0o600, f'bearer tokens in a {oct(mode)} file'

    def test_the_directory_is_not_traversable_by_anyone_else(self) -> None:
        credentials.save(_PROD, _session())
        mode = stat.S_IMODE(os.stat(credentials._path().parent).st_mode)

        assert mode == 0o700, oct(mode)

    def test_a_corrupt_cache_reads_as_signed_out(self) -> None:
        """Truncated JSON must not make every command fail; it means sign in again."""
        credentials.save(_PROD, _session())
        credentials._path().write_text('{"sessions": {"htt', encoding='utf-8')

        assert credentials.load(_PROD) is None
        assert credentials.signed_in() == []

    def test_an_unknown_field_from_a_newer_version_is_ignored(self) -> None:
        credentials.save(_PROD, _session())
        raw = json.loads(credentials._path().read_text(encoding='utf-8'))
        raw['sessions'][_PROD]['invented_later'] = 'x'
        credentials._path().write_text(json.dumps(raw), encoding='utf-8')

        assert credentials.load(_PROD).access_token == 'at'


class TestClear:
    def test_logging_out_of_one_leaves_the_others(self) -> None:
        credentials.save(_PROD, _session())
        credentials.save(_DEV, _session())

        assert credentials.clear(_PROD) is True
        assert credentials.signed_in() == [_DEV]

    def test_logging_out_everywhere(self) -> None:
        credentials.save(_PROD, _session())
        credentials.save(_DEV, _session())

        assert credentials.clear() is True
        assert credentials.signed_in() == []

    def test_logging_out_when_not_signed_in_is_not_an_error(self) -> None:
        assert credentials.clear(_PROD) is False
