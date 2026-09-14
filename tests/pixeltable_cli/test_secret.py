"""`pxt secret set`: the management request the daemon builds from what the client posts."""

import json
from typing import Any

import pytest

from pixeltable import exceptions as excs
from pixeltable.service import management_client
from pixeltable.service.management_protocol import SetSecretRequest
from pixeltable_cli.client.commands import secret as secret_cmd
from pixeltable_cli.server import router as server_router, routes as server_routes


def _forwarded(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Any]:
    """Run one `pxt secret` command into the route it posts to, and return the requests the route forwarded."""
    sent: list[Any] = []

    def api_call(request: Any) -> dict[str, Any]:
        sent.append(request)
        return {}

    def post_request(path: str, body: dict[str, Any]) -> dict[str, Any]:
        handler = server_routes.router.match('POST', path)
        assert handler is not None, path
        handler(server_router.Request(query={}, body_bytes=json.dumps(body).encode()))
        return {}

    monkeypatch.setattr(management_client, 'api_call', api_call)
    monkeypatch.setattr(secret_cmd, 'post_request', post_request)
    secret_cmd.run(argv)
    return sent


class TestSecret:
    def test_set(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        argv = [
            'set',
            'pxt://acme:main',
            'OPENAI_API_KEY=test=value',
            'PIXELTABLE_SECRET_CUSTOM=custom-value',
            '--json',
        ]
        assert _forwarded(monkeypatch, argv) == [
            SetSecretRequest(org='acme', db='main', key='OPENAI_API_KEY', value='test=value'),
            SetSecretRequest(org='acme', db='main', key='PIXELTABLE_SECRET_CUSTOM', value='custom-value'),
        ]
        captured = capsys.readouterr()
        assert json.loads(captured.out) == ['OPENAI_API_KEY', 'PIXELTABLE_SECRET_CUSTOM']
        assert captured.err == ''

        prohibited_keys = [
            'PIXELTABLE_HOME',
            'PIXELTABLE_DB',
            'PIXELTABLE_VAR_FOO',
            'PIXELTABLE_UNKNOWN',
            'PIXELTABLE_SECRET',
            'pixeltable_home',
            'Pixeltable_Db',
        ]
        for key in prohibited_keys:
            with pytest.raises(excs.RequestError, match='reserved for Pixeltable configuration'):
                _forwarded(monkeypatch, ['set', 'pxt://acme:main', f'{key}=test-value'])
