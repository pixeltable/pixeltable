"""Route surface in catalog-only mode (PIXELTABLE_SERVER_MODE=catalog_only) vs a full daemon.

The allow-list is read at import time, so each case reloads the module under a patched environment.
"""

import importlib
import os
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

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
    env = {**os.environ}
    if catalog_only:
        env['PIXELTABLE_SERVER_MODE'] = routes_module.CATALOG_ONLY_MODE
    else:
        env.pop('PIXELTABLE_SERVER_MODE', None)
    with patch.dict(os.environ, env, clear=True):
        importlib.reload(routes_module)
        yield


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

    def test_mode_is_read_from_config(self) -> None:
        with patch.dict(os.environ, {'PIXELTABLE_SERVER_MODE': routes_module.CATALOG_ONLY_MODE}):
            assert routes_module.in_catalog_only_mode()
        with patch.dict(os.environ, {'PIXELTABLE_SERVER_MODE': 'something_else'}):
            assert not routes_module.in_catalog_only_mode()
        env = {k: v for k, v in os.environ.items() if k != 'PIXELTABLE_SERVER_MODE'}
        with patch.dict(os.environ, env, clear=True):
            assert not routes_module.in_catalog_only_mode()


class TestFixedAddressMode:
    """A configured daemon_host/daemon_port makes the daemon a plain foreground server, as a pod needs."""

    @staticmethod
    def _configured(host: str | None, port: int | None) -> Any:
        """Stand in for Config so neither the environment nor a developer's config.toml decides the branch."""
        return patch(
            'pixeltable_cli.server.daemon.Config.get',
            return_value=SimpleNamespace(get_string_value=lambda key: host, get_int_value=lambda key: port),
        )

    def test_bind_defaults_to_loopback(self) -> None:
        server = http_server.bind(0)
        try:
            assert server.server_address[0] == '127.0.0.1'
        finally:
            server.server_close()

    def test_daemon_host_skips_the_pidfile_handshake(self) -> None:
        # A pod has no peer daemon to arbitrate with, so main() must not write a pidfile or probe for one.
        with (
            self._configured('0.0.0.0', 0),
            patch('pixeltable_cli.server.daemon.run') as run,
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
            patch('pixeltable_cli.server.daemon.is_running') as is_running,
        ):
            daemon.main()
        bind.assert_called_once_with(0, '0.0.0.0')
        run.assert_called_once()
        write_pidfile.assert_not_called()
        is_running.assert_not_called()

    def test_daemon_port_alone_selects_fixed_address_mode(self) -> None:
        # Matches proxy_daemon: either half of the pair is enough, and the host falls back to loopback.
        with (
            self._configured(None, 0),
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon.bind') as bind,
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main()
        bind.assert_called_once_with(0, '127.0.0.1')
        write_pidfile.assert_not_called()

    def test_without_daemon_config_the_local_path_is_unchanged(self) -> None:
        with (
            self._configured(None, None),
            patch.dict(os.environ, {'PXT_PORT': '0'}),
            patch('pixeltable_cli.server.daemon.run'),
            patch('pixeltable_cli.server.daemon._write_pidfile') as write_pidfile,
        ):
            daemon.main()
        write_pidfile.assert_called_once()
