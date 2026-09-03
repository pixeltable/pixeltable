"""Route surface in catalog-only mode vs a full daemon."""

import importlib
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from pixeltable.config import Config
from pixeltable_cli.models import Method
from pixeltable_cli.server import daemon, http_server, routes as routes_module

# enumerated independently of routes.CATALOG_ROUTES: a change to the boundary should take two edits
_CATALOG_ROUTES: set[tuple[Method, str]] = {
    ('GET', '/api/health'),
    ('GET', '/api/status'),
    ('GET', '/api/config'),
    ('GET', '/api/dirs'),
    ('GET', '/api/tables/rows'),
    ('GET', '/api/tables/row'),
    ('GET', '/api/tables/count'),
    ('GET', '/api/tables/errors'),
    ('GET', '/api/tables/history'),
    ('GET', '/api/tables/describe'),
    ('GET', '/api/columns'),
    ('GET', '/api/indexes'),
    ('POST', '/api/tables/drop'),
    ('POST', '/api/tables/revert'),
    ('POST', '/api/dirs/drop'),
    ('POST', '/api/move'),
    ('POST', '/api/schema/diff'),
    ('POST', '/api/schema/prune'),
    ('POST', '/api/schema/update'),
    ('GET', '/api/dashboard/search'),
    ('GET', '/api/dashboard/tables/meta'),
    ('GET', '/api/dashboard/tables/pipeline'),
    ('GET', '/api/dashboard/pipeline'),
    ('GET', '/api/dashboard/tables/data'),
    ('GET', '/api/dashboard/tables/export'),
}


def _reload(catalog_only: bool) -> Iterator[None]:
    Config.init({'cli_server.catalog_only': catalog_only}, reinit=True)
    try:
        importlib.reload(routes_module)
        yield
    finally:
        Config.init({}, reinit=True)


@pytest.fixture
def catalog_only_routes() -> Iterator[None]:
    yield from _reload(catalog_only=True)
    importlib.reload(routes_module)  # restore module state for the rest of the suite


@pytest.fixture
def full_routes() -> Iterator[None]:
    yield from _reload(catalog_only=False)
    importlib.reload(routes_module)


def _served() -> set[tuple[Method, str]]:
    return set(routes_module.router._routes)


class TestCatalogOnlyRouteSurface:
    def test_catalog_only_serves_the_catalog_and_nothing_else(self, catalog_only_routes: None) -> None:
        served = _served()
        assert served == _CATALOG_ROUTES, (
            f'served in a hosted pod but not in the allow-list: {sorted(served - _CATALOG_ROUTES)}; '
            f'in the allow-list but not served: {sorted(_CATALOG_ROUTES - served)}'
        )

    def test_every_route_is_served_when_not_catalog_only(self, full_routes: None) -> None:
        assert _served() > _CATALOG_ROUTES, 'a full daemon serves more than the catalog'

    def test_catalog_only_is_read_from_config(self) -> None:
        try:
            Config.init({'cli_server.catalog_only': True}, reinit=True)
            assert routes_module.in_catalog_only_mode()
            Config.init({'cli_server.catalog_only': False}, reinit=True)
            assert not routes_module.in_catalog_only_mode()
            Config.init({}, reinit=True)
            assert not routes_module.in_catalog_only_mode()
        finally:
            Config.init({}, reinit=True)


class TestFixedAddressMode:
    """A configured cli_server host/port makes the daemon a plain foreground server, as a pod needs."""

    @staticmethod
    def _configured(monkeypatch: pytest.MonkeyPatch, host: str | None, port: int | None) -> None:
        # env vars, as the pod is given them: daemon.main reinitialises Config, which drops overrides
        monkeypatch.delenv('CLI_SERVER_HOST', raising=False)
        monkeypatch.delenv('CLI_SERVER_PORT', raising=False)
        if host is not None:
            monkeypatch.setenv('CLI_SERVER_HOST', host)
        if port is not None:
            monkeypatch.setenv('CLI_SERVER_PORT', str(port))

    def test_bind_defaults_to_loopback(self) -> None:
        server = http_server.bind(0)
        try:
            assert server.server_address[0] == '127.0.0.1'
        finally:
            server.server_close()

    def test_host_skips_the_pidfile_handshake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._configured(monkeypatch, '0.0.0.0', 0)
        with (
            patch('pixeltable_cli.server.daemon.run') as run,
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
            patch('pixeltable_cli.server.daemon.is_running') as is_running,
        ):
            daemon.main([])
        Config.init({}, reinit=True)
        bind.assert_called_once_with(0, '0.0.0.0')
        run.assert_called_once()
        write_pidfile.assert_not_called()
        is_running.assert_not_called()

    def test_port_alone_selects_fixed_address_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._configured(monkeypatch, None, 0)
        with (
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main([])
        Config.init({}, reinit=True)
        bind.assert_called_once_with(0, '127.0.0.1')
        write_pidfile.assert_not_called()

    def test_without_cli_server_config_the_local_path_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._configured(monkeypatch, None, None)
        with (
            patch('pixeltable_cli.server.daemon.get_port', return_value=0),
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main([])
        Config.init({}, reinit=True)
        write_pidfile.assert_called_once()
