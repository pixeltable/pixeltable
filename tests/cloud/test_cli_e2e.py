"""Cloud e2e test.

Covers the full DB + service lifecycle: create, probe, update-runtime, stop/start, delete.
SDK probes run inline after each major phase to cross-check state via the proxy.

The tests run in sequence against one hosted database and must stay on a single xdist worker;
they are deselected by default and from every CI tier via the cloud_e2e marker.

Run:
    PIXELTABLE_API_KEY=sk_... pytest tests/cloud/test_cli_e2e.py -v -s -m cloud_e2e

Required env:
    PIXELTABLE_API_KEY

Optional env:
    PIXELTABLE_API_URL        (default: https://dev-internal-api.pixeltable.com)
    PIXELTABLE_CLOUD_HOST     (default: dev.pxt.run)

Pass --keep-cloud-resources to leave the database and service in place after the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest
import requests

# xdist_group keeps the whole sequence on one worker: the tests share a single hosted database and
# run in order
pytestmark = [pytest.mark.cloud_e2e, pytest.mark.xdist_group('cloud_e2e')]

_SAMPLE_APP = Path(__file__).parent / 'sample_app'

_ORG = 'pixeltable'

_API_KEY = os.environ.get('PIXELTABLE_API_KEY', '')
_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://dev-internal-api.pixeltable.com')
_CLOUD_HOST = os.environ.get('PIXELTABLE_CLOUD_HOST', 'dev.pxt.run')


# ── helpers ──────────────────────────────────────────────────────────────────


def _cloud_env() -> dict[str, str]:
    e = os.environ.copy()
    e['PIXELTABLE_API_KEY'] = _API_KEY
    e['PIXELTABLE_API_URL'] = _API_URL
    e['PIXELTABLE_CLOUD_HOST'] = _CLOUD_HOST
    return e


def _pxt(*args: str, cwd: Path | None = None, check: bool = True, timeout: int = 900) -> str:
    r = subprocess.run(
        ['pxt', *args], capture_output=True, text=True, env=_cloud_env(), cwd=cwd, timeout=timeout, check=False
    )
    out = r.stdout + r.stderr
    if check and r.returncode != 0:
        raise AssertionError(f'pxt {" ".join(args)} failed (rc={r.returncode}):\n{out}')
    return out


def _pxt_json(*args: str, cwd: Path | None = None) -> str:
    return _pxt(*args, '--json', cwd=cwd)


def _sdk(code: str) -> str:
    """Run a Python snippet in a subprocess with cloud env."""
    r = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=_cloud_env(),
        timeout=120,
        check=False,
    )
    return r.stdout + r.stderr


def _post(url: str, json: dict, *, retries: int = 10, delay: float = 5.0) -> requests.Response:
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=json, headers={'X-api-key': _API_KEY}, timeout=15)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    assert resp is not None
    return resp


def _get(url: str, params: dict | None = None, *, retries: int = 10, delay: float = 5.0) -> requests.Response:
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers={'X-api-key': _API_KEY}, timeout=15)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    assert resp is not None
    return resp


def _wait_for_state(resource_type: str, uri: str, desired: str, *, timeout: int = 180, poll_interval: int = 5) -> str:
    """Poll `pxt <resource_type> status <uri> --json` until <desired> appears in output.

    Fails immediately if any worker pod is in CrashLoopBackOff and not recovering.
    """
    deadline = time.time() + timeout
    out = ''
    while time.time() < deadline:
        out = _pxt_json(resource_type, 'status', uri)
        if desired in out:
            return out
        # Detect terminal crash state: all workers crashing, none ready
        try:
            data = json.loads(out.strip())
            workers = data.get('workers', [])
            if workers:
                crash_pods = [w for w in workers if w.get('status') == 'CrashLoopBackOff']
                ready_pods = [w for w in workers if w.get('ready', 0) > 0]
                if crash_pods and not ready_pods and len(crash_pods) == len(workers):
                    raise AssertionError(
                        f'All {len(crash_pods)} worker pod(s) in CrashLoopBackOff; '
                        f'service will not reach {desired}.\nStatus:\n{out}'
                    )
        except (json.JSONDecodeError, AttributeError):
            pass
        time.sleep(poll_interval)
    raise AssertionError(f'{resource_type} {uri} did not reach {desired} within {timeout}s.\nLast status:\n{out}')


# ── resources fixture ─────────────────────────────────────────────────────────


_SVC_NAME = 'e2e-svc'  # must match [[pixeltable.service]] name in sample_app/pixeltable.toml


class Resources(NamedTuple):
    org: str
    db: str
    svc_name: str
    db_uri: str
    svc_uri: str
    table_uri: str  # pxt://org:db/e2e_items


@pytest.fixture(scope='module', autouse=True)
def require_api_key() -> None:
    """Skip every test in this module unless a key for the hosted deployment is available."""
    if _API_KEY == '':
        pytest.skip('PIXELTABLE_API_KEY not set')


@pytest.fixture(scope='module')
def resources(request: pytest.FixtureRequest) -> Iterator[Resources]:
    # A daemon left over from an earlier session holds the api url and cloud host it started with, so
    # restart it to pick up the values in _cloud_env()
    subprocess.run(['pxt', 'daemon', 'restart'], capture_output=True, env=_cloud_env(), check=False)

    # The sample app's uv.lock is gitignored and is never generated by the bundle build (that only packages
    # an existing lock). Regenerate it fresh from the pyproject on every run, always, even if one exists,
    # so a stale/absent lock can't deploy the wrong pixeltable or fail the server-side `uv sync --frozen`.
    (_SAMPLE_APP / 'uv.lock').unlink(missing_ok=True)
    lock = subprocess.run(['uv', 'lock'], cwd=_SAMPLE_APP, capture_output=True, text=True, check=False)
    if lock.returncode != 0:
        raise RuntimeError(f'uv lock failed for sample app:\n{lock.stdout}\n{lock.stderr}')

    run_id = uuid.uuid4().hex[:8]
    db = f'clitest-e2e-{run_id}'
    db_uri = f'pxt://{_ORG}:{db}'
    svc_uri = f'{db_uri}/services/{_SVC_NAME}'
    table_uri = f'{db_uri}/e2e_items'

    r = Resources(org=_ORG, db=db, svc_name=_SVC_NAME, db_uri=db_uri, svc_uri=svc_uri, table_uri=table_uri)
    try:
        yield r
    finally:
        if request.config.getoption('--keep-cloud-resources') or request.session.testsfailed > 0:
            print(f'\n[cleanup skipped, resources left for inspection: {db_uri}]', flush=True)
        else:
            _pxt('service', 'delete', svc_uri, '--json', check=False)
            _pxt('db', 'delete', db_uri, '--json', check=False)


@pytest.fixture(scope='module')
def svc_base(resources: Resources) -> str:
    """Root URL of the deployed service, as reported by `pxt service status`; routes hang off it."""
    out = _pxt_json('service', 'status', resources.svc_uri)
    json_line = next((line for line in out.splitlines() if line.startswith('{')), None)
    assert json_line is not None, f'no JSON in service status output:\n{out}'
    endpoint = json.loads(json_line).get('endpoint')
    assert endpoint is not None, f'service status reports no endpoint:\n{out}'
    return endpoint.rstrip('/')


# ── tests ─────────────────────────────────────────────────────────────────────


class TestCloudE2E:
    def test_org_list(self, resources: Resources) -> None:
        out = _pxt_json('org', 'list')
        assert resources.org in out

    def test_org_status(self, resources: Resources) -> None:
        out = _pxt_json('org', 'status', f'pxt://{resources.org}')
        assert resources.org in out

    def test_db_create(self, resources: Resources) -> None:
        _pxt('db', 'create', resources.db_uri)
        out = _pxt_json('db', 'status', resources.db_uri)
        assert 'AVAILABLE' in out
        assert resources.db in out

    def test_db_list(self, resources: Resources) -> None:
        out = _pxt_json('db', 'list', f'pxt://{resources.org}')
        assert resources.db in out

    def test_db_status(self, resources: Resources) -> None:
        out = _pxt_json('db', 'status', resources.db_uri)
        assert resources.db in out

    def test_db_update(self, resources: Resources) -> None:
        _pxt('db', 'update', resources.db_uri, '--workers', '2')
        out = _pxt_json('db', 'status', resources.db_uri)
        assert resources.db in out

    def test_sdk_create_table(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.create_table(
                '{resources.table_uri}',
                {{'id': pxt.Int, 'name': pxt.String}},
                primary_key='id',
                if_exists='ignore',
            )
            t.add_computed_column(name_upper=t.name.upper(), if_exists='ignore')
            status = t.insert([{{'id': i, 'name': f'item_{{i}}'}} for i in range(5)])
            print('rows:', status.num_rows)
        """
        # Retry: proxy gateway may lag behind AVAILABLE state
        for attempt in range(4):
            out = _sdk(code)
            if 'rows: 5' in out:
                break
            if attempt < 3:
                time.sleep(20)
        assert 'rows: 5' in out, f'SDK table create failed:\n{out}'

    def test_sdk_list_tables(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            tables = pxt.list_tables('{resources.db_uri}')
            print('tables:', [t for t in tables])
        """
        out = _sdk(code)
        assert 'e2e_items' in out, f'list_tables did not show e2e_items:\n{out}'

    def test_sdk_table_count(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            print('count:', t.count())
        """
        out = _sdk(code)
        assert 'count: 5' in out, f'table count wrong:\n{out}'

    def test_sdk_read_rows(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            rows = t.select(t.id, t.name).order_by(t.id).collect()
            for r in rows:
                print(r['id'], r['name'])
        """
        out = _sdk(code)
        for i in range(5):
            assert f'{i} item_{i}' in out, f'row {i} missing:\n{out}'

    def test_sdk_media_over_tunnel(self, resources: Resources) -> None:
        """Image round-trip over the TLS tunnel: insert a generated image, then read it back.

        Reading the image cell forces TunnelTransport.fetch() to pull /media/<ref> over the tunnel
        (the daemon serves it there, not via a direct HTTP GET). Regression guard for the media-localization
        gap where the tunnel transport produced no media_url and any media result would fail to localize.
        """
        media_uri = f'{resources.db_uri}/media_test'
        code = f"""
            import pixeltable as pxt
            from PIL import Image
            pxt.init()
            t = pxt.create_table(
                '{media_uri}',
                {{'id': pxt.Int, 'img': pxt.Image}},
                primary_key='id',
                if_exists='replace_force',
            )
            t.insert([{{'id': 1, 'img': Image.new('RGB', (64, 48), (255, 0, 0))}}])
            row = t.select(t.img).collect()[0]
            print('img_size:', row['img'].size)
        """
        # Retry: proxy gateway may lag behind AVAILABLE state
        for attempt in range(4):
            out = _sdk(code)
            if 'img_size: (64, 48)' in out:
                break
            if attempt < 3:
                time.sleep(20)
        assert 'img_size: (64, 48)' in out, f'media-over-tunnel round-trip failed:\n{out}'

    def test_sdk_concurrent_inserts(self, resources: Resources) -> None:
        """Two threads insert into the same hosted table at once, exercising the tunnel connection pool.

        The pre-pool single-connection client would interleave bytes on one socket and corrupt the stream;
        with the pool each thread borrows its own tunnel connection and both inserts land.
        """
        conc_uri = f'{resources.db_uri}/concurrent_test'
        code = f"""
            import concurrent.futures
            import pixeltable as pxt
            pxt.init()
            pxt.create_table(
                '{conc_uri}',
                {{'id': pxt.Int, 'name': pxt.String}},
                primary_key='id',
                if_exists='replace_force',
            )
            def _insert(base):
                t = pxt.get_table('{conc_uri}')
                return t.insert([{{'id': base + i, 'name': f'n{{base + i}}'}} for i in range(10)]).num_rows
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                inserted = sum(ex.map(_insert, [0, 100]))
            print('inserted:', inserted)
            print('count:', pxt.get_table('{conc_uri}').count())
        """
        # Retry: proxy gateway may lag behind AVAILABLE state
        for attempt in range(4):
            out = _sdk(code)
            if 'count: 20' in out:
                break
            if attempt < 3:
                time.sleep(20)
        assert 'inserted: 20' in out, f'concurrent inserts did not all land:\n{out}'
        assert 'count: 20' in out, f'concurrent-insert final count wrong:\n{out}'

    def test_service_create(self, resources: Resources) -> None:
        out = _pxt_json(
            'service', 'create', resources.svc_name, '--base-uri', resources.db_uri, '--workers', '1', cwd=_SAMPLE_APP
        )
        assert resources.svc_name in out

    def test_service_list(self, resources: Resources) -> None:
        out = _pxt_json('service', 'list', resources.db_uri)
        assert resources.svc_name in out

    def test_service_status(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert resources.svc_name in out

    def test_service_available(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert 'AVAILABLE' in out

    def test_route_insert(self, svc_base: str) -> None:
        resp = _post(f'{svc_base}/insert', {'id': 9000, 'name': 'lifecycle_probe'})
        assert resp.status_code == 200
        assert 'name_upper' in resp.text

    def test_sdk_verify_insert(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            rows = t.where(t.id == 9000).collect()
            print('rows:', len(rows))
        """
        out = _sdk(code)
        assert 'rows: 1' in out, f'SDK did not see id=9000 inserted via service:\n{out}'

    def test_route_compute(self, svc_base: str) -> None:
        resp = _post(f'{svc_base}/compute', {'id': 9001, 'name': 'compute_probe'})
        assert resp.status_code == 200
        assert 'COMPUTE_PROBE' in resp.text

    def test_route_update(self, svc_base: str) -> None:
        resp = _post(f'{svc_base}/update', {'id': 9000, 'name': 'updated_probe'})
        assert resp.status_code == 200
        assert 'UPDATED_PROBE' in resp.text

    def test_route_delete(self, svc_base: str) -> None:
        resp = _post(f'{svc_base}/delete', {'id': 9000})
        assert resp.status_code == 200
        assert 'num_rows' in resp.text

    def test_sdk_verify_compute_delete(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            print('9001:', len(t.where(t.id == 9001).collect()))
            print('9000:', len(t.where(t.id == 9000).collect()))
        """
        out = _sdk(code)
        assert '9001: 1' in out, f'id=9001 not persisted:\n{out}'
        assert '9000: 0' in out, f'id=9000 not deleted:\n{out}'

    def test_db_update_runtime(self, resources: Resources) -> None:
        out = _pxt('db', 'update-runtime', resources.db_uri, '--json', cwd=_SAMPLE_APP, timeout=1200)
        # stdout + stderr are combined by _pxt; find the JSON line (starts with '{')
        json_line = next((line for line in out.splitlines() if line.startswith('{')), None)
        assert json_line is not None, f'No JSON in update-runtime output:\n{out}'
        data = json.loads(json_line)
        assert data.get('state') == 'AVAILABLE', f'update-runtime did not return AVAILABLE:\n{out}'
        # update-runtime output may not include last_build_state; always verify via db status
        status_out = _pxt_json('db', 'status', resources.db_uri)
        status = json.loads(status_out)
        assert status.get('last_build_state') == 'ACTIVE', (
            f'Runtime build did not succeed (last_build_state={status.get("last_build_state")!r}, '
            f'error={status.get("last_build_error")!r})'
        )

    def test_service_update_add_query_route(self, resources: Resources) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_toml = Path(tmp) / 'pixeltable.toml'
            shutil.copy(_SAMPLE_APP / 'pixeltable.toml', tmp_toml)
            with open(tmp_toml, 'a', encoding='utf-8') as f:
                f.write(
                    '\n[[pixeltable.service.routes]]\n'
                    'type    = "query"\n'
                    'path    = "/find"\n'
                    'query   = "udfs:find_by_id"\n'
                    'inputs  = ["item_id"]\n'
                    'one_row = true\n'
                    'method  = "get"\n'
                )
            _pxt_json(
                'service', 'update', resources.svc_uri, '--workers', '2', '--config', str(tmp_toml), cwd=_SAMPLE_APP
            )
        out = _wait_for_state('service', resources.svc_uri, 'AVAILABLE', timeout=180)
        assert 'AVAILABLE' in out

    def test_route_query(self, svc_base: str) -> None:
        # Rolling update race: after service update the old pod (without /find) may still
        # serve while the new pod is pulling the update-runtime image. Give it up to 3 min.
        resp = _get(f'{svc_base}/find', params={'item_id': 1}, retries=24, delay=8.0)
        assert resp.status_code == 200, f'/find returned {resp.status_code}: {resp.text[:300]}'
        assert 'item_1' in resp.text

    def test_service_stop(self, resources: Resources) -> None:
        out = _pxt_json('service', 'stop', resources.svc_uri)
        assert 'STOPPED' in out

    def test_db_stop(self, resources: Resources) -> None:
        out = _pxt_json('db', 'stop', resources.db_uri)
        assert 'STOPPED' in out

    def test_service_status_stopped(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert 'STOPPED' in out

    def test_db_status_stopped(self, resources: Resources) -> None:
        out = _pxt_json('db', 'status', resources.db_uri)
        assert 'STOPPED' in out

    def test_db_start(self, resources: Resources) -> None:
        out = _pxt_json('db', 'start', resources.db_uri)
        assert 'AVAILABLE' in out

    def test_service_start(self, resources: Resources) -> None:
        _pxt_json('service', 'start', resources.svc_uri)
        out = _wait_for_state('service', resources.svc_uri, 'AVAILABLE', timeout=120)
        assert 'AVAILABLE' in out

    def test_sdk_persistence_after_restart(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            all_ids = sorted(r['id'] for r in t.select(t.id).collect())
            print('orig_ok:', all(i in all_ids for i in range(5)))
            print('9001_ok:', 9001 in all_ids)
            print('9000_gone:', 9000 not in all_ids)
            print('count:', len(all_ids))
        """
        out = _sdk(code)
        assert 'orig_ok: True' in out, f'original rows missing after restart:\n{out}'
        assert '9001_ok: True' in out, f'id=9001 missing after restart:\n{out}'
        assert '9000_gone: True' in out, f'id=9000 reappeared after restart:\n{out}'

    def test_route_insert_after_restart(self, svc_base: str) -> None:
        resp = _post(f'{svc_base}/insert', {'id': 9002, 'name': 'post_restart'})
        assert resp.status_code == 200
        assert 'name_upper' in resp.text

    def test_sdk_verify_insert_after_restart(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            print('9002:', len(t.where(t.id == 9002).collect()))
        """
        out = _sdk(code)
        assert '9002: 1' in out, f'id=9002 not found after restart:\n{out}'

    def test_service_delete(self, resources: Resources) -> None:
        _pxt('service', 'delete', resources.svc_uri, '--json')

    def test_db_delete(self, resources: Resources) -> None:
        _pxt('db', 'delete', resources.db_uri, '--json')
