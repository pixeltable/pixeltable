from pathlib import Path

import pytest

from pixeltable.serving.pod_runner import _add_gateway_openapi_security
from tests.utils import skip_test_if_not_installed


class TestHostedOpenapi:
    def test_pod_runner_ignores_proxy_headers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('uvicorn')
        import uvicorn

        from pixeltable.serving import pod_runner

        fingerprint_file = tmp_path / 'fingerprint.json'
        fingerprint_file.write_text('{}')
        monkeypatch.setattr(pod_runner.ProjectFingerprint, 'model_validate_json', lambda value: object())
        monkeypatch.setattr(pod_runner.Config, 'init', lambda **kwargs: None)
        monkeypatch.setattr(pod_runner, 'create_app', lambda *args: (object(), None))
        monkeypatch.setattr(pod_runner, '_add_gateway_openapi_security', lambda app: None)
        monkeypatch.setattr(pod_runner, 'report_instance_fingerprint', lambda *args: None)
        received: dict[str, object] = {}
        monkeypatch.setattr(uvicorn, 'run', lambda app, **kwargs: received.update(kwargs))

        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)

        assert received['proxy_headers'] is False
        assert received['root_path'] == '/nested/svc'

    def test_gateway_security_only_on_hosted_schema(self) -> None:
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.security import HTTPBasic
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()
        basic = HTTPBasic()

        @app.get('/private', dependencies=[fastapi.Depends(basic)])
        def private() -> dict[str, bool]:
            return {'ok': True}

        @app.post('/write')
        def write() -> dict[str, bool]:
            return {'ok': True}

        @app.get('/health')
        def health() -> dict[str, bool]:
            return {'ok': True}

        local_schema = app.openapi()
        assert local_schema['paths']['/private']['get']['security'] == [{'HTTPBasic': []}]
        assert 'security' not in local_schema['paths']['/write']['post']

        _add_gateway_openapi_security(app)
        with TestClient(app, root_path='/nested/service') as client:
            hosted_schema = client.get('/openapi.json').json()

        assert hosted_schema['servers'] == [{'url': '/nested/service'}]
        assert hosted_schema['components']['securitySchemes'] == {
            'HTTPBasic': {'type': 'http', 'scheme': 'basic'},
            'PixeltableGatewayApiKey': {'type': 'apiKey', 'in': 'header', 'name': 'X-api-key'},
        }
        # the route's own Authorization-based scheme and the gateway key are both required
        assert hosted_schema['paths']['/private']['get']['security'] == [
            {'PixeltableGatewayApiKey': [], 'HTTPBasic': []}
        ]
        assert hosted_schema['paths']['/write']['post']['security'] == [{'PixeltableGatewayApiKey': []}]
        assert 'security' not in hosted_schema['paths']['/health']['get']
        assert app.openapi() is app.openapi()
        assert 'PixeltableGatewayApiKey' not in local_schema['components']['securitySchemes']
