from pathlib import Path
from unittest import mock

import pytest

from tests.utils import skip_test_if_not_installed


class TestPodRunner:
    def test_server_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('uvicorn')
        import uvicorn

        from pixeltable.serving import pod_runner

        # _serve() reads a fingerprint file, whose content does not matter here
        fingerprint_file = tmp_path / 'fingerprint.json'
        fingerprint_file.write_text('{}')
        # no project, no config file and no control plane; the mock router records its _configure_job_forwarding() call
        monkeypatch.setattr(pod_runner.ProjectFingerprint, 'model_validate_json', lambda value: object())
        monkeypatch.setattr(pod_runner.Config, 'init', lambda **kwargs: None)
        router = mock.Mock()
        monkeypatch.setattr(pod_runner, 'create_app', lambda *args: (object(), None, [router]))
        monkeypatch.setattr(pod_runner, '_add_gateway_openapi_security', lambda app: None)
        monkeypatch.setattr(pod_runner, 'report_instance_fingerprint', lambda *args: None)
        # record each server instead of running it
        served: list[pod_runner._DrainingServer] = []
        monkeypatch.setattr(pod_runner._DrainingServer, 'run', lambda self, sockets=None: served.append(self))

        # service svc at base path nested, without the variables of a hosted pod
        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)

        config: uvicorn.Config = served[0].config
        # the pod trusts no X-Forwarded-* headers, and root_path puts the gateway's stripped prefix back into its URLs
        assert config.proxy_headers is False
        assert config.root_path == '/nested/svc'
        # SIGTERM waits for the background jobs
        assert served[0].drain_s > 0
        # without the pod variables, job polls are not forwarded
        router._configure_job_forwarding.assert_not_called()

        # a hosted pod: its IP, and the db credential, the secret for its job ids
        monkeypatch.setenv('PIXELTABLE_SERVE_INSTANCE_HOST', '10.0.0.7')
        monkeypatch.setenv('PIXELTABLE_DB_CONNECT_STR', 'postgresql://u:p@db/pxt')
        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)
        # the other pods forward polls to <pod IP>:<port>, under the service's root path
        router._configure_job_forwarding.assert_called_once_with(
            '10.0.0.7:8000', 'postgresql://u:p@db/pxt', '/nested/svc'
        )
