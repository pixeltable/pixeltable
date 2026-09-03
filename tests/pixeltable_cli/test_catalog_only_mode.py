"""Route surface in catalog-only mode (cli_server.catalog_only) vs a full daemon.

The allow-list is read at import time, so each case reloads the module under a config override.
"""

import importlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from pixeltable.config import Config
from pixeltable_cli.server import daemon, http_server, routes as routes_module

# Reachable in both modes; must match routes.CATALOG_ROUTES, enumerated here independently.
_CATALOG_ROUTES = {
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

# Control-plane proxy plus the stateful working directory.
_MANAGEMENT_ROUTES = {
    ('GET', '/api/cwd'),
    ('POST', '/api/cwd'),
    ('GET', '/api/orgs'),
    ('GET', '/api/org'),
    ('GET', '/api/dbs'),
    ('POST', '/api/dbs'),
    ('GET', '/api/db'),
    ('POST', '/api/db/delete'),
    ('POST', '/api/db/start'),
    ('POST', '/api/db/stop'),
    ('POST', '/api/db/update'),
    ('GET', '/api/db/upload-url'),
    ('POST', '/api/db/update-runtime'),
    ('GET', '/api/services'),
    ('POST', '/api/services'),
    ('GET', '/api/service'),
    ('POST', '/api/service/delete'),
    ('POST', '/api/service/start'),
    ('POST', '/api/service/stop'),
    ('POST', '/api/service/update'),
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


def _unreachable(routes: set[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(r for r in routes if routes_module.router.match(*r) is None)


class TestCatalogOnlyRouteSurface:
    def test_catalog_only_serves_the_catalog_and_nothing_else(self, catalog_only_routes: None) -> None:
        reachable = sorted(r for r in _MANAGEMENT_ROUTES if routes_module.router.match(*r) is not None)
        assert not reachable, f'reachable in a hosted pod: {reachable}'
        assert not _unreachable(_CATALOG_ROUTES)

    def test_every_route_is_served_when_not_catalog_only(self, full_routes: None) -> None:
        assert not _unreachable(_CATALOG_ROUTES | _MANAGEMENT_ROUTES)

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
    def _configured(host: str | None, port: int | None) -> None:
        overrides: dict[str, Any] = {}
        if host is not None:
            overrides['cli_server.host'] = host
        if port is not None:
            overrides['cli_server.port'] = port
        Config.init(overrides, reinit=True)

    def test_bind_defaults_to_loopback(self) -> None:
        server = http_server.bind(0)
        try:
            assert server.server_address[0] == '127.0.0.1'
        finally:
            server.server_close()

    def test_host_skips_the_pidfile_handshake(self) -> None:
        self._configured('0.0.0.0', 0)
        with (
            patch('pixeltable_cli.server.daemon.run') as run,
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
            patch('pixeltable_cli.server.daemon.is_running') as is_running,
        ):
            daemon.main()
        Config.init({}, reinit=True)
        bind.assert_called_once_with(0, '0.0.0.0')
        run.assert_called_once()
        write_pidfile.assert_not_called()
        is_running.assert_not_called()

    def test_port_alone_selects_fixed_address_mode(self) -> None:
        self._configured(None, 0)
        with (
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main()
        Config.init({}, reinit=True)
        bind.assert_called_once_with(0, '127.0.0.1')
        write_pidfile.assert_not_called()

    def test_without_cli_server_config_the_local_path_is_unchanged(self) -> None:
        self._configured(None, None)
        with (
            patch('pixeltable_cli.server.daemon.get_port', return_value=0),
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main()
        Config.init({}, reinit=True)
        write_pidfile.assert_called_once()
