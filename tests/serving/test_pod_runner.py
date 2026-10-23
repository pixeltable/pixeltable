from pathlib import Path
from unittest import mock

import pytest

import pixeltable as pxt
from tests.utils import pxt_raises, skip_test_if_not_installed


class TestPodRunner:
    def test_server_config(
        self, init_env: None, private_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # init_env sets up the Env that skip_test_if_not_installed() consults.
        # private_home precedes monkeypatch, so that Config.init is no longer patched when private_home restores Config
        skip_test_if_not_installed('fastapi', 'uvicorn')
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

        # a hosted pod: its IP, and the database's API key, the secret for its job ids
        monkeypatch.setenv('PXTCLOUD_POD_IP', '10.0.0.7')
        monkeypatch.setenv('PIXELTABLE_API_KEY', 'pxt-worker-key')
        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)
        # the other pods forward polls to <pod IP>:<port>, under the service's root path
        router._configure_job_forwarding.assert_called_once_with('10.0.0.7:8000', 'pxt-worker-key', '/nested/svc')

        # an IPv6 pod address gets brackets in <host>:<port>
        router.reset_mock()
        monkeypatch.setenv('PXTCLOUD_POD_IP', 'fd00::7')
        pod_runner._serve('pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False)
        assert router._configure_job_forwarding.call_args.args[0] == '[fd00::7]:8000'

        # a hosted pod without the API key has no secret for its job ids
        monkeypatch.delenv('PIXELTABLE_API_KEY')
        with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match='a pod needs PIXELTABLE_API_KEY'):
            pod_runner._serve(
                'pxt://org:db', 'app.py', 'svc', 'nested', tmp_path, fingerprint_file, '0.0.0.0', 8000, False
            )
