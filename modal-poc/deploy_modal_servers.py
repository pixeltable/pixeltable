"""Deploy one Modal Server per entry of [tool.pixeltable.udf] to the active Modal workspace, without Pixeltable Cloud.

Writes .modal_env.sh, which exports the PXT_UDF_SERVERS and PXT_UDF_SERVERS_HEADERS that ModalEndpoints reads.
README.md says how to run it.
"""

import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.exec.expr_eval.modal_client import UDF_SECRET_HEADER_NAME
from pixeltable.serving.modal_server import PORT, UDF_PATH_VAR, UDF_SECRET_VAR

APP_NAME = 'pxt-udf-modal-poc'
PROJECT_ROOT = Path(__file__).resolve().parent


class PxtUdfServer:
    """ModalServer without the project download: the image already holds this checkout's pixeltable."""

    server: Any

    @modal.enter()
    def enter(self) -> None:
        from pixeltable.serving.udf_server import UdfServer

        self.server = UdfServer(os.environ[UDF_PATH_VAR], os.environ.get('MODAL_TASK_ID'), os.environ[UDF_SECRET_VAR])
        self.server.start(PORT)


def server_name(setting_path: str) -> str:
    digest = hashlib.sha256(setting_path.encode()).hexdigest()[:8]
    return f'{setting_path.rsplit(".", 1)[-1]}-{digest}'


def proxy_token() -> dict[str, str]:
    """Return the proxy token that calls the Servers, creating it on the first run."""
    path = PROJECT_ROOT / '.modal_proxy_token.json'
    if not path.is_file():
        out = subprocess.run(
            ['modal', 'workspace', 'proxy-tokens', 'create', '--name', APP_NAME, '--json'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        path.touch(mode=0o600)
        path.write_text(out, encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    Config.init(reinit=True, project_root=PROJECT_ROOT)
    udf_configs = Env.get().udf_configs()
    if not udf_configs:
        sys.exit('pyproject.toml has no [tool.pixeltable.udf] entries')
    token = proxy_token()
    udf_secret = secrets.token_urlsafe(32)

    # pixeltable comes from add_local_python_source() below, and modal from Modal's own image setup
    requirements = subprocess.run(
        [
            *('uv', 'export', '--frozen', '--no-hashes', '--no-emit-workspace'),
            *('--no-emit-package', 'pixeltable', '--no-emit-package', 'modal'),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp_dir:
        req_file = Path(tmp_dir) / 'requirements.txt'
        req_file.write_text(requirements, encoding='utf-8')
        image = (
            modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
            .apt_install('ffmpeg', 'git')
            .uv_pip_install(requirements=[str(req_file)], extra_options='--no-deps')
            .env({'HF_HOME': '/cache/hf'})
            .add_local_python_source('pixeltable', 'pixeltable_cli')
        )
        hf_cache = modal.Volume.from_name('pxt-hf-cache', create_if_missing=True)

        app = modal.App(APP_NAME)
        servers: dict[str, modal.Server] = {}
        for cfg in udf_configs.values():
            r = cfg.resources
            servers[cfg.setting_path] = app.server(
                name=server_name(cfg.setting_path),
                image=image,
                serialized=True,
                volumes={'/cache': hf_cache},
                env={UDF_PATH_VAR: cfg.setting_path, UDF_SECRET_VAR: udf_secret},
                port=PORT,
                h2_enabled=True,
                # Modal counts physical cores, each 2 vCPUs
                cpu=None if r.cpu is None else r.cpu / 2,
                memory=r.memory_mb,
                gpu=r.gpu,
                target_concurrency=1,
                max_containers=r.max_containers,
                scaledown_window=60,
                exit_grace_period=r.timeout_seconds,
                startup_timeout=60,
            )(PxtUdfServer)
        with modal.enable_output():
            app.deploy(name=APP_NAME, strategy='recreate')

    urls = {path: url for path, server in servers.items() if (url := server.get_url()) is not None}
    headers = {
        'Modal-Key': token['Modal-Key'],
        'Modal-Secret': token['Modal-Secret'],
        UDF_SECRET_HEADER_NAME: udf_secret,
    }
    env_file = PROJECT_ROOT / '.modal_env.sh'
    env_file.touch(mode=0o600)
    env_file.write_text(
        f'export PXT_UDF_SERVERS={shlex.quote(json.dumps(urls))}\n'
        f'export PXT_UDF_SERVERS_HEADERS={shlex.quote(json.dumps(headers))}\n',
        encoding='utf-8',
    )
    for path, url in urls.items():
        print(f'{path} -> {url}')
    print(f'Wrote {env_file.name}; run: source {env_file.name}')


if __name__ == '__main__':
    main()
