"""The on-disk session cache: what it keeps apart, what it refuses to leak, what it survives."""

import pathlib
import time
from typing import Any

import pytest

from pixeltable.config import Config
from pixeltable.service import session_cache

_PROD = 'https://api.pixeltable.com'
_DEV = 'https://api.dev.pxt.run'


@pytest.fixture(autouse=True)
def _home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Config caches home at first access, so each test needs its own instance to get its own file.
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    Config.init(reinit=True)


def _session(**kw: Any) -> session_cache.Session:
    base = {'access_token': 'at', 'expires_at': time.time() + 3600, 'refresh_token': 'rt', 'client_id': 'client_01TEST'}
    return session_cache.Session(**{**base, **kw})


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
