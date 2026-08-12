"""Cloud e2e test for the R2 PartSink media path.

Proves that hosted-db media inserts travel out of band through the db's R2 home bucket
(proxy_client.R2PartSink uploads to uploads/<request-uuid>/, the RPC carries only object keys,
and proxy_dispatch._prefetch_remote_parts localizes them before dispatch) instead of inline
in the proxy RPC.

Sequence: create a db, set the daemon-side secrets (PIXELTABLE_DAEMON_ORG/_DB and the control
plane URL/key the daemon needs to mint R2 credentials), stop/start the db so the secrets sync
into the pod (set_secret writes WorkOS Vault only; the k8s pxt-user-secrets Secret refreshes
on db start), then update-runtime with a bundle built from a pinned pixeltable git ref so the
daemon runs this branch's protocol.

The tests run in sequence against one hosted database and must stay on a single xdist worker;
they are deselected by default and from every CI tier via the cloud_e2e marker.

Run:
    PIXELTABLE_API_KEY=sk_... pytest tests/cloud/test_r2_media_e2e.py -v -s -m cloud_e2e

Required env:
    PIXELTABLE_API_KEY        (must be bound to PIXELTABLE_TEST_ORG)

Optional env:
    PIXELTABLE_API_URL        (default: https://dev-internal-api.pixeltable.com)
    PIXELTABLE_CLOUD_HOST     (default: dev.pxt.run)
    PIXELTABLE_TEST_ORG       (default: pixeltable)
    PIXELTABLE_RUNTIME_SPEC   pip requirement the runtime bundle pins pixeltable to; set this to a
                              git ref of the branch under test (default: the main repo's main branch)

Pass --keep-cloud-resources to leave the database in place after the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

from .test_cli_e2e import _cloud_env, _pxt, _pxt_json, _sdk

pytestmark = [pytest.mark.cloud_e2e, pytest.mark.xdist_group('cloud_e2e')]

_ORG = os.environ.get('PIXELTABLE_TEST_ORG', 'pixeltable')
_API_KEY = os.environ.get('PIXELTABLE_API_KEY', '')
_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://dev-internal-api.pixeltable.com')
_RUNTIME_SPEC = os.environ.get(
    'PIXELTABLE_RUNTIME_SPEC', 'pixeltable[serve] @ git+https://github.com/pixeltable/pixeltable@main'
)

_FILE_IMAGE = Path(__file__).parents[1] / 'data' / 'images' / 'sewing-threads-smaller.jpg'


class Resources(NamedTuple):
    org: str
    db: str
    db_uri: str
    svc_uri: str  # no service is deployed; present only for the conftest failure-logging hook
    media_table_uri: str
    scalar_table_uri: str


@pytest.fixture(scope='module', autouse=True)
def require_api_key() -> None:
    """Skip every test in this module unless a key for the hosted deployment is available."""
    if _API_KEY == '':
        pytest.skip('PIXELTABLE_API_KEY not set')


@pytest.fixture(scope='module')
def sample_app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Minimal runtime-only project for update-runtime: pinned pixeltable ref, no services.

    python_version is pinned to the deploying interpreter so the server-side image build cannot
    diverge from what uv.lock was resolved for (see TECH_DEBT.md on update-runtime).
    """
    app_dir = tmp_path_factory.mktemp('r2_media_app')
    py = f'{sys.version_info.major}.{sys.version_info.minor}'
    (app_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "r2-media-e2e-app"\n'
        'version = "0.1.0"\n'
        f'requires-python = ">={py}"\n'
        f'dependencies = ["{_RUNTIME_SPEC}"]\n'
    )
    (app_dir / 'pixeltable.toml').write_text(f'[pixeltable.database]\npython_version = "{py}"\n')
    lock = subprocess.run(['uv', 'lock'], cwd=app_dir, capture_output=True, text=True, check=False)
    if lock.returncode != 0:
        raise RuntimeError(f'uv lock failed for r2 media sample app:\n{lock.stdout}\n{lock.stderr}')
    return app_dir


@pytest.fixture(scope='module')
def resources(request: pytest.FixtureRequest) -> Iterator[Resources]:
    # A daemon left over from an earlier session holds the api url and cloud host it started with, so
    # restart it to pick up the values in _cloud_env()
    subprocess.run(['pxt', 'daemon', 'restart'], capture_output=True, env=_cloud_env(), check=False)

    run_id = uuid.uuid4().hex[:8]
    db = f'r2media-e2e-{run_id}'
    db_uri = f'pxt://{_ORG}:{db}'
    r = Resources(
        org=_ORG,
        db=db,
        db_uri=db_uri,
        svc_uri=f'{db_uri}/services/none',
        media_table_uri=f'{db_uri}/media_items',
        scalar_table_uri=f'{db_uri}/scalar_items',
    )
    try:
        yield r
    finally:
        if request.config.getoption('--keep-cloud-resources') or request.session.testsfailed > 0:
            print(f'\n[cleanup skipped, resources left for inspection: {db_uri}]', flush=True)
        else:
            _pxt('db', 'delete', db_uri, '--json', check=False)


class TestR2MediaE2E:
    def test_db_create(self, resources: Resources) -> None:
        _pxt('db', 'create', resources.db_uri)
        out = _pxt_json('db', 'status', resources.db_uri)
        assert 'AVAILABLE' in out
        assert resources.db in out

    def test_set_daemon_secrets(self, resources: Resources) -> None:
        """Register the daemon-side env vars as db-scoped secrets via the management API.

        The daemon needs its own control plane URL + key to mint R2 credentials for localizing
        uploads (proxy_dispatch reads daemon_org/daemon_db from Config, i.e. from these env vars).
        """
        secrets = {
            'PIXELTABLE_DAEMON_ORG': resources.org,
            'PIXELTABLE_DAEMON_DB': resources.db,
            'PIXELTABLE_API_URL': _API_URL,
            'PIXELTABLE_API_KEY': _API_KEY,
        }
        code = f"""
            import pixeltable as pxt
            pxt.init()
            from pixeltable.service import management_client
            from pixeltable.service.management_protocol import ListSecretsRequest, SetSecretRequest
            for key, value in {secrets!r}.items():
                req = SetSecretRequest(org='{resources.org}', db='{resources.db}', key=key, value=value)
                management_client.api_call(req)
            listed = management_client.api_call(ListSecretsRequest(org='{resources.org}', db='{resources.db}'))
            print('secret_keys:', sorted(listed['keys']))
        """
        out = _sdk(code)
        assert f'secret_keys: {sorted(secrets)}' in out, f'daemon secrets not registered:\n{out}'

    def test_db_restart_syncs_secrets(self, resources: Resources) -> None:
        """Stop/start the db so the vault secrets sync into the pod.

        set_secret only writes WorkOS Vault; the k8s pxt-user-secrets Secret the daemon pod
        envFroms is refreshed on db start (and the update-runtime pod restart then re-reads it).
        """
        out = _pxt_json('db', 'stop', resources.db_uri)
        assert 'STOPPED' in out
        out = _pxt_json('db', 'start', resources.db_uri)
        assert 'AVAILABLE' in out

    def test_db_update_runtime(self, resources: Resources, sample_app: Path) -> None:
        out = _pxt('db', 'update-runtime', resources.db_uri, '--json', cwd=sample_app, timeout=1800)
        json_line = next((line for line in out.splitlines() if line.startswith('{')), None)
        assert json_line is not None, f'No JSON in update-runtime output:\n{out}'
        assert json.loads(json_line).get('state') == 'AVAILABLE', f'update-runtime did not return AVAILABLE:\n{out}'
        status = json.loads(_pxt_json('db', 'status', resources.db_uri))
        assert status.get('last_build_state') == 'ACTIVE', (
            f'Runtime build did not succeed (last_build_state={status.get("last_build_state")!r}, '
            f'error={status.get("last_build_error")!r})'
        )

    def test_media_insert_roundtrip(self, resources: Resources) -> None:
        """Assertion 1: file-backed and in-memory images round-trip through the hosted table."""
        assert _FILE_IMAGE.is_file()
        code = f"""
            import pixeltable as pxt
            from PIL import Image
            pxt.init()
            expected = Image.open('{_FILE_IMAGE}').size
            t = pxt.create_table(
                '{resources.media_table_uri}',
                {{'id': pxt.Required[pxt.Int], 'img': pxt.Image}},
                primary_key='id',
                if_exists='replace_force',
            )
            status = t.insert([
                {{'id': 1, 'img': '{_FILE_IMAGE}'}},
                {{'id': 2, 'img': Image.new('RGB', (64, 48), (0, 128, 255))}},
            ])
            print('rows:', status.num_rows)
            sizes = {{r['id']: r['img'].size for r in t.select(t.id, t.img).collect()}}
            print('file_ok:', sizes[1] == expected)
            print('mem_ok:', sizes[2] == (64, 48))
        """
        # Retry: proxy gateway may lag behind AVAILABLE state after the runtime update
        for attempt in range(4):
            out = _sdk(code)
            if 'mem_ok: True' in out:
                break
            if attempt < 3:
                time.sleep(20)
        assert 'rows: 2' in out, f'media insert failed:\n{out}'
        assert 'file_ok: True' in out, f'file-backed image did not round-trip:\n{out}'
        assert 'mem_ok: True' in out, f'in-memory image did not round-trip:\n{out}'

    def test_uploads_listed_in_home_bucket(self, resources: Resources) -> None:
        """Assertion 2: the insert's media parts landed under uploads/ in the R2 home bucket.

        The db is created fresh by this run, so every uploads/ object is ours: the single
        two-image insert must have produced (at least) two objects, still unexpired.
        """
        code = f"""
            import pixeltable as pxt
            pxt.init()
            from pixeltable.utils.object_stores import ObjectOps
            store = ObjectOps.get_store('pxtfs://{resources.org}:{resources.db}/home/uploads/', False)
            keys = store.list_objects(return_uri=False, n_max=100)
            for key in sorted(keys):
                print('upload_key:', key)
        """
        out = _sdk(code)
        keys = [ln.removeprefix('upload_key:').strip() for ln in out.splitlines() if ln.startswith('upload_key:')]
        assert len(keys) >= 2, f'expected the 2 inserted media parts under uploads/, got {keys}:\n{out}'
        assert all(k.startswith('uploads/') for k in keys), f'non-uploads key in listing: {keys}'

    def test_stored_cells_not_upload_keys(self, resources: Resources) -> None:
        """Assertion 3: persisted cell values are not uploads/ keys (uploads expire via lifecycle)."""
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.media_table_uri}')
            for r in t.select(url=t.img.fileurl).collect():
                print('stored_url:', r['url'])
        """
        out = _sdk(code)
        urls = [ln.removeprefix('stored_url:').strip() for ln in out.splitlines() if ln.startswith('stored_url:')]
        assert len(urls) == 2, f'expected 2 stored media urls:\n{out}'
        assert all(len(u) > 0 and 'uploads/' not in u for u in urls), f'stored cell is an uploads/ key: {urls}'

    def test_scalar_insert_skips_control_plane(self, resources: Resources) -> None:
        """Assertion 4: scalar-only requests never mint bucket credentials.

        R2PartSink builds its store lazily on the first media part; with get_bucket_credentials
        poisoned in the client process, create_table + insert can only succeed if no request
        touched the control plane for R2 credentials.
        """
        code = f"""
            import pixeltable as pxt
            pxt.init()
            import pixeltable.utils.cloud_utils as cloud_utils
            import pixeltable.utils.pxt_store as pxt_store
            def _fail(*args, **kwargs):
                raise AssertionError('scalar-only request minted bucket credentials')
            cloud_utils.get_bucket_credentials = _fail
            pxt_store.get_bucket_credentials = _fail
            t = pxt.create_table(
                '{resources.scalar_table_uri}',
                {{'id': pxt.Required[pxt.Int], 'name': pxt.String}},
                primary_key='id',
                if_exists='replace_force',
            )
            status = t.insert([{{'id': i, 'name': f'n{{i}}'}} for i in range(3)])
            print('scalar_rows:', status.num_rows)
        """
        out = _sdk(code)
        assert 'scalar_rows: 3' in out, f'scalar-only insert hit the control plane (or failed):\n{out}'
