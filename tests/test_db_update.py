import socket
from pathlib import Path
from typing import Any

import pytest

from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.service import proxy_client
from pixeltable.service.db import db_update
from pixeltable.service.db_md import DatabaseStatus
from pixeltable.service.management_protocol import (
    DatabaseReport,
    GetDbRequest,
    GetDbResponse,
    UpdateDbRequest,
    UpdateDbResponse,
)
from pixeltable_cli.types import DbPlan


class TestDbUpdate:
    """`db_update()` against a control plane that settles the database at once."""

    @pytest.fixture(autouse=True)
    def _no_proxy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tunnel check honors HTTPS_PROXY, so a developer's setting would reroute these tests."""
        for name in ('HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy'):
            monkeypatch.delenv(name, raising=False)

    @staticmethod
    def _apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DbPlan:
        (tmp_path / 'pixeltable.toml').write_text('[[pixeltable.database]]\nname = "pxt://acme:main"\n')
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'requirements.txt').write_text('pandas\n')
        Config.init(reinit=True, project_root=tmp_path)
        pending = DbPlan(db_uri='pxt://acme:main', exists=True, state='AVAILABLE', resolution='update_additive')
        requested: list[UpdateDbRequest] = []

        def api_call(request: Any) -> dict[str, Any]:
            if isinstance(request, UpdateDbRequest):
                requested.append(request)
                return UpdateDbResponse(plan=pending, report=DatabaseReport()).model_dump(mode='json')
            assert isinstance(request, GetDbRequest)
            # the database settles on what it was last asked for
            current = DatabaseStatus(resources=requested[-1].target, state='AVAILABLE')
            return GetDbResponse(report=DatabaseReport(current=current)).model_dump(mode='json')

        monkeypatch.setattr('pixeltable.service.db.management_client.api_call', api_call)
        return db_update('pxt://acme:main')

    def test_unreachable_tunnel_is_a_warning(
        self, init_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control plane answers on 443, so a network that blocks the tunnel's port still reaches AVAILABLE;
        the update says so, rather than leaving it to the first command that uses the tables."""
        host, port = Env.get().proxy_endpoint('acme', 'main')
        timeouts: list[float | None] = []

        def drop(address: tuple[str, int], timeout: float | None = None) -> socket.socket:
            assert address == (host, port)
            timeouts.append(timeout)
            raise TimeoutError('timed out')

        monkeypatch.setattr(socket, 'create_connection', drop)
        plan = self._apply(tmp_path, monkeypatch)

        assert plan.status == 'applied'
        assert plan.state == 'AVAILABLE'
        assert len(plan.warnings) == 1
        warning = plan.warnings[0]
        assert warning.startswith('pxt://acme:main is AVAILABLE, but commands that use its tables'), warning
        assert f'the TCP connection to {host}:{port} timed out' in warning
        assert f'allow outbound TCP port {port} to *.{host.partition(".")[2]}' in warning
        # the same tries as a command that uses the tables, so the warning is about what that command would meet
        assert timeouts == [proxy_client._CONNECT_TIMEOUT] * proxy_client._CONNECT_TRIES

    def test_reachable_tunnel_is_not_a_warning(
        self, init_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[socket.socket] = []

        def accept(address: tuple[str, int], timeout: float | None = None) -> socket.socket:
            ours, theirs = socket.socketpair()
            opened.extend((ours, theirs))
            return ours

        monkeypatch.setattr(socket, 'create_connection', accept)
        try:
            plan = self._apply(tmp_path, monkeypatch)
            assert plan.warnings == []
            assert opened[0].fileno() == -1  # the check closed its connection
        finally:
            for sock in opened:
                sock.close()
