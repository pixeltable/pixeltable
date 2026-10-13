from pathlib import Path
from unittest import mock

import pytest

from tests.utils import skip_test_if_not_installed


class TestPodRunner:
    def test_server_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('uvicorn')
        import uvicorn

        from pixeltable.serving import pod_runner

        fingerprint_file = tmp_path / 'fingerprint.json'
        fingerprint_file.write_text('{}')
        monkeypatch.setattr(pod_runner.ProjectFingerprint, 'model_validate_json', lambda value: object())
        monkeypatch.setattr(pod_runner.Config, 'init', lambda **kwargs: None)
        router = mock.Mock()
        monkeypatch.setattr(pod_runner, 'create_app', lambda *args: (object(), None, [router]))
        monkeypatch.setattr(pod_runner, '_add_gateway_openapi_security', lambda app: None)
        monkeypatch.setattr(pod_runner, 'report_instance_fingerprint', lambda *args: None)
        served: list[pod_runner._DrainingServer] = []
        monkeypatch.setattr(pod_runner._DrainingServer, 'run', lambda self, sockets=None: served.append(self))

        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)

        config: uvicorn.Config = served[0].config
        assert config.proxy_headers is False
        assert config.root_path == '/nested/svc'
        # without the pod variables, job polls are not forwarded and SIGTERM does not wait for jobs
        router._configure_job_forwarding.assert_not_called()
        assert served[0].drain_s == 0.0

        monkeypatch.setenv('PIXELTABLE_SERVE_INSTANCE_HOST', '10.0.0.7')
        monkeypatch.setenv('PIXELTABLE_SERVE_DRAIN_S', '285')
        monkeypatch.setenv('PIXELTABLE_DB_CONNECT_STR', 'postgresql://u:p@db/pxt')
        keys: list[bytes] = []
        for service_name in ('svc', 'svc', 'other'):
            router.reset_mock()
            pod_runner._serve(
                'pxt://org:db', 'app.py', service_name, 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False
            )
            addr, key, root_path = router._configure_job_forwarding.call_args.args
            assert (addr, root_path) == ('10.0.0.7:8000', f'/nested/{service_name}')
            keys.append(key)
        assert served[-1].drain_s == 285.0
        # each pod of a service derives the same key, and another service a different one
        assert keys[0] == keys[1] != keys[2]
