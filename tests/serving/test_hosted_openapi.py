from pixeltable.serving.pod_runner import _add_gateway_openapi_security
from tests.utils import skip_test_if_not_installed


def test_gateway_security_only_on_hosted_schema() -> None:
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
    assert hosted_schema['components']['securitySchemes']['PixeltableGatewayBearer'] == {
        'type': 'http',
        'scheme': 'bearer',
    }
    assert hosted_schema['components']['securitySchemes']['PixeltableGatewayApiKey'] == {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-api-key',
    }
    assert hosted_schema['paths']['/private']['get']['security'] == [
        {'PixeltableGatewayBearer': [], 'HTTPBasic': []},
        {'PixeltableGatewayApiKey': [], 'HTTPBasic': []},
    ]
    assert hosted_schema['paths']['/write']['post']['security'] == [
        {'PixeltableGatewayBearer': []},
        {'PixeltableGatewayApiKey': []},
    ]
    assert 'security' not in hosted_schema['paths']['/health']['get']
    assert app.openapi() is app.openapi()
    assert 'PixeltableGatewayBearer' not in local_schema['components']['securitySchemes']
