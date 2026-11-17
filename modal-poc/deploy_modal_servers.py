"""Deploy PR #1970's UdfServer for each entry in [tool.pixeltable.udf] directly to Modal.

Use this script to demo or test against your own Modal workspace (via PXT_UDF_SERVERS and
PXT_UDF_SERVERS_HEADERS) without waiting for the Pixeltable Cloud control plane.

Usage:
    uv run --extra modal-cli python deploy_modal_servers.py
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.exec.expr_eval.modal_client import UDF_SECRET_HEADER_NAME

APP_NAME = 'pxt-udf-modal-poc'
PORT = 8000
ROUTING_REGION = 'us-east'
SCALEDOWN_SECONDS = 60


class PxtUdfServer:
    """Starts Pixeltable's UdfServer for one UDF inside a Modal Server container."""

    server: Any

    @modal.enter()
    def enter(self) -> None:
        from pixeltable.serving.udf_server import UdfServer

        self.server = UdfServer(
            os.environ['PXT_UDF_PATH'], os.environ.get('MODAL_TASK_ID'), os.environ['PXT_UDF_SECRET']
        )
        self.server.start(PORT)


def _server_name(setting_path: str) -> str:
    digest = hashlib.sha256(setting_path.encode()).hexdigest()[:8]
    return f'{setting_path.rsplit(".", 1)[-1]}-{digest}'


def _export_requirements(project_root: Path) -> str:
    result = subprocess.run(
        [
            'uv',
            'export',
            '--frozen',
            '--no-hashes',
            '--no-emit-workspace',
            '--no-emit-package',
            'pixeltable',
            '--no-emit-package',
            'modal',
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _load_or_create_proxy_token(project_root: Path) -> dict[str, Any]:
    token_path = project_root / '.modal_proxy_token.json'
    if token_path.is_file():
        return json.loads(token_path.read_text(encoding='utf-8'))

    print('Creating a Modal proxy token in the active workspace...')
    modal_bin = Path(sys.executable).parent / 'modal'
    cmd = [str(modal_bin if modal_bin.is_file() else 'modal'), 'workspace', 'proxy-tokens', 'create']
    cmd += ['--name', APP_NAME, '--json']
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=True)
    token_path.write_text(result.stdout, encoding='utf-8')
    os.chmod(token_path, 0o600)
    print(f'Saved proxy token to {token_path.name} (chmod 600).')
    return json.loads(result.stdout)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    Config.init(reinit=True, project_root=project_root)

    proxy_token = _load_or_create_proxy_token(project_root)
    udf_configs = Env.get().udf_configs()
    if not udf_configs:
        sys.exit('No [tool.pixeltable.udf] entries found in pyproject.toml')

    print(f'Discovered {len(udf_configs)} UDFs in [tool.pixeltable.udf]:')
    for cfg in udf_configs.values():
        r = cfg.resources
        print(
            f'  - {cfg.setting_path}: cpu={r.cpu}, memory_mb={r.memory_mb}, '
            f'gpu={r.gpu}, max_containers={r.max_containers}'
        )

    udf_secret = secrets.token_urlsafe(32)
    requirements = _export_requirements(project_root)

    with tempfile.TemporaryDirectory() as tmp_dir:
        req_file = Path(tmp_dir) / 'requirements.txt'
        req_file.write_text(requirements, encoding='utf-8')

        py_ver = f'{sys.version_info.major}.{sys.version_info.minor}'
        image = (
            modal.Image.debian_slim(python_version=py_ver)
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
            # Keep compute_region unset so GPU containers schedule across available US regions
            # instead of waiting on a single zone's GPU queue.
            servers[cfg.setting_path] = app.server(
                name=_server_name(cfg.setting_path),
                image=image,
                serialized=True,
                volumes={'/cache': hf_cache},
                env={'PXT_UDF_PATH': cfg.setting_path, 'PXT_UDF_SECRET': udf_secret},
                port=PORT,
                h2_enabled=True,
                routing_region=ROUTING_REGION,
                cpu=None if r.cpu is None else r.cpu / 2,
                memory=r.memory_mb,
                gpu=r.gpu,
                target_concurrency=1,
                max_containers=r.max_containers,
                scaledown_window=SCALEDOWN_SECONDS,
                exit_grace_period=r.timeout_seconds,
                startup_timeout=60,
            )(PxtUdfServer)

        print(f'\nDeploying Modal App {APP_NAME!r} (Python {py_ver})...')
        start = time.monotonic()
        with modal.enable_output():
            app.deploy(name=APP_NAME, strategy='recreate')
        elapsed = time.monotonic() - start
        print(f'\nDeployed {APP_NAME!r} in {elapsed:.1f}s')

    server_urls = {path: url for path, srv in servers.items() if (url := srv.get_url()) is not None}
    headers = {
        'Modal-Key': proxy_token['Modal-Key'],
        'Modal-Secret': proxy_token['Modal-Secret'],
        UDF_SECRET_HEADER_NAME: udf_secret,
    }

    env_payload = {
        'app_name': APP_NAME,
        'deployed_seconds': round(elapsed, 2),
        'PXT_UDF_SERVERS': server_urls,
        'PXT_UDF_SERVERS_HEADERS': headers,
    }
    env_json_path = project_root / '.modal_env.json'
    env_json_path.write_text(json.dumps(env_payload, indent=2) + '\n', encoding='utf-8')
    os.chmod(env_json_path, 0o600)

    env_sh_path = project_root / '.modal_env.sh'
    env_sh_path.write_text(
        f'export PXT_UDF_SERVERS={shlex.quote(json.dumps(server_urls))}\n'
        f'export PXT_UDF_SERVERS_HEADERS={shlex.quote(json.dumps(headers))}\n',
        encoding='utf-8',
    )
    os.chmod(env_sh_path, 0o600)

    print('\nServer URLs:')
    for path, url in server_urls.items():
        print(f'  {path} -> {url}')
    print(f'\nWrote {env_json_path.name} and {env_sh_path.name} (chmod 600).')


if __name__ == '__main__':
    main()
