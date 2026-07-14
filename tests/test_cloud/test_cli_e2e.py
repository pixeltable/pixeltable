"""Cloud e2e test — Python equivalent of local/test_cli_e2e.sh.

Covers the full DB + service lifecycle: create, probe, update-runtime, stop/start, delete.
SDK probes run inline after each major phase to cross-check state via the proxy.

Run:
    PXT_E2E_ENABLED=1 PIXELTABLE_API_KEY=sk_... pytest tests/test_cloud/test_cli_e2e.py -v -s

Required env:
    PIXELTABLE_API_KEY

Optional env:
    PIXELTABLE_API_URL        (default: https://dev-internal-api.pixeltable.com)
    PIXELTABLE_CLOUD_HOST     (default: dev.pxt.run)
    PXT_E2E_SVC_DOMAIN        (default: svc.<CLOUD_HOST>)
    PXT_E2E_ORG_SLUG          (default: pixeltable)
    PXT_E2E_SKIP_CLEANUP      (set to 1 to leave resources for inspection)
    PXT_E2E_SKIP_UPDATE_RUNTIME (set to 1 to skip the ~10 min CodeBuild step)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest
import requests

pytestmark = pytest.mark.cloud_e2e

_SAMPLE_APP = Path(__file__).parent / 'sample_app'

_API_KEY = os.environ.get('PIXELTABLE_API_KEY', '')
_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://dev-internal-api.pixeltable.com')
_CLOUD_HOST = os.environ.get('PIXELTABLE_CLOUD_HOST', 'dev.pxt.run')
_SVC_DOMAIN = os.environ.get('PXT_E2E_SVC_DOMAIN', f'svc.{_CLOUD_HOST}')
_ORG_SLUG = os.environ.get('PXT_E2E_ORG_SLUG', 'pixeltable')
_SKIP_CLEANUP = os.environ.get('PXT_E2E_SKIP_CLEANUP', '0') == '1'
_SKIP_UPDATE_RUNTIME = os.environ.get('PXT_E2E_SKIP_UPDATE_RUNTIME', '0') == '1'


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


class Resources(NamedTuple):
    org_slug: str
    db_slug: str
    svc_name: str
    db_uri: str
    svc_uri: str
    svc_base: str
    table_uri: str  # pxt://org:db/e2e_items
    toml_dir: Path  # temp dir holding pixeltable.toml for service config


def _write_initial_toml(toml_dir: Path, svc_name: str) -> None:
    """4-route TOML (no query route yet — added by service update later)."""
    (toml_dir / 'pixeltable.toml').write_text(
        f'[[pixeltable.service]]\n'
        f'name = "{svc_name}"\n\n'
        '[[pixeltable.service.routes]]\n'
        'path    = "/insert"\ntype    = "insert"\ntable   = "e2e_items"\n'
        'inputs  = ["id", "name"]\noutputs = ["id", "name", "name_upper"]\n\n'
        '[[pixeltable.service.routes]]\n'
        'path    = "/compute"\ntype    = "compute"\ntable   = "e2e_items"\n'
        'inputs  = ["id", "name"]\noutputs = ["name_upper"]\n\n'
        '[[pixeltable.service.routes]]\n'
        'path    = "/update"\ntype    = "update"\ntable   = "e2e_items"\n'
        'inputs  = ["name"]\noutputs = ["id", "name", "name_upper"]\n\n'
        '[[pixeltable.service.routes]]\n'
        'path          = "/delete"\ntype          = "delete"\ntable         = "e2e_items"\n'
        'match_columns = ["id"]\n'
    )


def _write_full_toml(toml_dir: Path, svc_name: str) -> None:
    """5-route TOML with the query route added (used for service update)."""
    _write_initial_toml(toml_dir, svc_name)
    with open(toml_dir / 'pixeltable.toml', 'a', encoding='utf-8') as f:
        f.write(
            '\n[[pixeltable.service.routes]]\n'
            'path    = "/find"\ntype    = "query"\nquery   = "udfs:find_by_id"\n'
            'inputs  = ["item_id"]\none_row = true\nmethod  = "get"\n'
        )


@pytest.fixture(scope='module')
def resources(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Iterator[Resources]:
    if not _API_KEY:
        pytest.skip('PIXELTABLE_API_KEY not set')

    run_id = uuid.uuid4().hex[:8]
    db_slug = f'clitest-e2e-{run_id}'
    svc_name = f'svc-e2e-{run_id}'
    db_uri = f'pxt://{_ORG_SLUG}:{db_slug}'
    svc_uri = f'{db_uri}/services/{svc_name}'
    svc_base = f'https://{_ORG_SLUG}-{db_slug}.{_SVC_DOMAIN}/{svc_name}'
    table_uri = f'{db_uri}/e2e_items'
    toml_dir = tmp_path_factory.mktemp('cloud_e2e_toml')

    r = Resources(
        org_slug=_ORG_SLUG,
        db_slug=db_slug,
        svc_name=svc_name,
        db_uri=db_uri,
        svc_uri=svc_uri,
        svc_base=svc_base,
        table_uri=table_uri,
        toml_dir=toml_dir,
    )
    try:
        yield r
    finally:
        if _SKIP_CLEANUP or request.session.testsfailed > 0:
            print(f'\n[cleanup skipped — resources left for inspection: {db_uri}]', flush=True)
        else:
            _pxt('service', 'delete', svc_uri, '--json', check=False)
            _pxt('db', 'delete', db_uri, '--json', check=False)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestCloudE2E:
    # ── 0. daemon restart ────────────────────────────────────────────────────

    def test_0_daemon_restart(self) -> None:
        subprocess.run(['pxt', 'daemon', 'restart'], capture_output=True, env=_cloud_env(), check=False)

    # ── 1. help smoke tests ───────────────────────────────────────────────────

    def test_1a_db_help(self) -> None:
        out = _pxt('db', '--help', check=False)
        assert 'create' in out
        assert 'list' in out
        assert 'update' in out
        assert 'update-runtime' in out
        assert 'status' in out

    def test_1b_db_update_help(self) -> None:
        out = _pxt('db', 'update', '--help', check=False)
        assert '--workers' in out
        assert '--cpu' in out

    def test_1c_service_help(self) -> None:
        out = _pxt('service', '--help', check=False)
        assert 'create' in out
        assert 'update' in out
        assert 'stop' in out
        assert 'start' in out
        assert 'status' in out

    def test_1d_service_update_help(self) -> None:
        out = _pxt('service', 'update', '--help', check=False)
        assert '--workers' in out

    def test_1e_org_help(self) -> None:
        out = _pxt('org', '--help', check=False)
        assert 'list' in out
        assert 'status' in out

    # ── 2. org list / status ──────────────────────────────────────────────────

    def test_2a_org_list(self, resources: Resources) -> None:
        out = _pxt_json('org', 'list')
        assert '[' in out or '{' in out

    def test_2b_org_status(self, resources: Resources) -> None:
        out = _pxt_json('org', 'status', f'pxt://{resources.org_slug}')
        assert resources.org_slug in out

    # ── 3. db create ─────────────────────────────────────────────────────────

    def test_3_db_create(self, resources: Resources) -> None:
        _pxt('db', 'create', resources.db_uri)
        out = _pxt_json('db', 'status', resources.db_uri)
        assert 'AVAILABLE' in out
        assert resources.db_slug in out

    # ── 4. db list / status ───────────────────────────────────────────────────

    def test_4a_db_list(self, resources: Resources) -> None:
        out = _pxt_json('db', 'list', f'pxt://{resources.org_slug}')
        assert resources.db_slug in out

    def test_4b_db_status(self, resources: Resources) -> None:
        out = _pxt_json('db', 'status', resources.db_uri)
        assert resources.db_slug in out

    # ── 5. db update ──────────────────────────────────────────────────────────

    def test_5_db_update(self, resources: Resources) -> None:
        _pxt('db', 'update', resources.db_uri, '--workers', '2')
        out = _pxt_json('db', 'status', resources.db_uri)
        assert resources.db_slug in out

    # ── 6. SDK: create table + insert rows ───────────────────────────────────

    def test_6_sdk_create_table(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.create_table(
                '{resources.table_uri}',
                {{'id': pxt.Required[pxt.Int], 'name': pxt.String}},
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

    def test_6b_sdk_list_tables(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            tables = pxt.list_tables('{resources.db_uri}')
            print('tables:', [t for t in tables])
        """
        out = _sdk(code)
        assert 'e2e_items' in out, f'list_tables did not show e2e_items:\n{out}'

    def test_6c_sdk_get_table_count(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            print('count:', t.count())
        """
        out = _sdk(code)
        assert 'count: 5' in out, f'table count wrong:\n{out}'

    def test_6d_sdk_read_rows(self, resources: Resources) -> None:
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

    # ── 7. service create ─────────────────────────────────────────────────────

    def test_7_service_create(self, resources: Resources) -> None:
        _write_initial_toml(resources.toml_dir, resources.svc_name)
        out = _pxt_json(
            'service',
            'create',
            resources.svc_name,
            '--base-uri',
            resources.db_uri,
            '--workers',
            '1',
            cwd=resources.toml_dir,
        )
        assert resources.svc_name in out

    # ── 8. service list / status ──────────────────────────────────────────────

    def test_8a_service_list(self, resources: Resources) -> None:
        out = _pxt_json('service', 'list', resources.db_uri)
        assert resources.svc_name in out

    def test_8b_service_status(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert resources.svc_name in out

    def test_8c_service_available(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert 'AVAILABLE' in out

    # ── 9-10. probe /insert ───────────────────────────────────────────────────

    def test_9_probe_insert(self, resources: Resources) -> None:
        resp = _post(f'{resources.svc_base}/insert', {'id': 9000, 'name': 'lifecycle_probe'})
        assert resp.status_code == 200
        assert 'name_upper' in resp.text

    def test_10_sdk_cross_check_insert(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            rows = t.where(t.id == 9000).collect()
            print('rows:', len(rows))
        """
        out = _sdk(code)
        assert 'rows: 1' in out, f'SDK did not see id=9000 inserted via service:\n{out}'

    # ── 11. probe /compute /update /delete ────────────────────────────────────

    def test_11a_probe_compute(self, resources: Resources) -> None:
        resp = _post(f'{resources.svc_base}/compute', {'id': 9001, 'name': 'compute_probe'})
        assert resp.status_code == 200
        assert 'COMPUTE_PROBE' in resp.text

    def test_11b_probe_update(self, resources: Resources) -> None:
        resp = _post(f'{resources.svc_base}/update', {'id': 9000, 'name': 'updated_probe'})
        assert resp.status_code == 200
        assert 'UPDATED_PROBE' in resp.text

    def test_11c_probe_delete(self, resources: Resources) -> None:
        resp = _post(f'{resources.svc_base}/delete', {'id': 9000})
        assert resp.status_code == 200
        assert 'num_rows' in resp.text

    def test_11d_sdk_verify_compute_delete(self, resources: Resources) -> None:
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

    # ── 12. db update-runtime (~10 min CodeBuild) ─────────────────────────────

    def test_12_update_runtime(self, resources: Resources) -> None:
        if _SKIP_UPDATE_RUNTIME:
            pytest.skip('PXT_E2E_SKIP_UPDATE_RUNTIME=1')
        out = _pxt('db', 'update-runtime', resources.db_uri, '--json', cwd=_SAMPLE_APP, timeout=1200)
        # stdout + stderr are combined by _pxt; find the JSON line (starts with '{')
        json_line = next((line for line in out.splitlines() if line.startswith('{')), None)
        assert json_line is not None, f'No JSON in update-runtime output:\n{out}'
        data = json.loads(json_line)
        assert data.get('state') == 'AVAILABLE', f'update-runtime did not return AVAILABLE:\n{out}'
        # last_build_state is only present once the backend supports it; when present, must be ACTIVE
        if data.get('last_build_state') is not None:
            assert data.get('last_build_state') == 'ACTIVE', (
                f'Runtime build did not succeed (last_build_state={data.get("last_build_state")!r}, '
                f'error={data.get("last_build_error")!r}):\n{out}'
            )

    # ── 13. service update: add query route ───────────────────────────────────

    def test_13_service_update_add_query_route(self, resources: Resources) -> None:
        if _SKIP_UPDATE_RUNTIME:
            pytest.skip('query route requires update-runtime image with udfs.py')
        _write_full_toml(resources.toml_dir, resources.svc_name)
        _pxt_json(
            'service',
            'update',
            resources.svc_uri,
            '--workers',
            '2',
            '--config',
            str(resources.toml_dir / 'pixeltable.toml'),
            cwd=resources.toml_dir,
        )
        out = _wait_for_state('service', resources.svc_uri, 'AVAILABLE', timeout=180)
        assert 'AVAILABLE' in out

    def test_13b_probe_query_route(self, resources: Resources) -> None:
        if _SKIP_UPDATE_RUNTIME:
            pytest.skip('query route requires update-runtime image with udfs.py')
        # Rolling update race: after service update the old pod (without /find) may still
        # serve while the new pod is pulling the update-runtime image. Give it up to 3 min.
        resp = _get(f'{resources.svc_base}/find', params={'item_id': 1}, retries=24, delay=8.0)
        assert resp.status_code == 200, f'/find returned {resp.status_code}: {resp.text[:300]}'
        assert 'item_1' in resp.text

    # ── 14-16. stop service + DB, verify ─────────────────────────────────────

    def test_14_service_stop(self, resources: Resources) -> None:
        out = _pxt_json('service', 'stop', resources.svc_uri)
        assert 'STOPPED' in out

    def test_15_db_stop(self, resources: Resources) -> None:
        out = _pxt_json('db', 'stop', resources.db_uri)
        assert 'STOPPED' in out

    def test_16a_service_status_stopped(self, resources: Resources) -> None:
        out = _pxt_json('service', 'status', resources.svc_uri)
        assert 'STOPPED' in out

    def test_16b_db_status_stopped(self, resources: Resources) -> None:
        out = _pxt_json('db', 'status', resources.db_uri)
        assert 'STOPPED' in out

    # ── 17-18. start DB + service ─────────────────────────────────────────────

    def test_17_db_start(self, resources: Resources) -> None:
        out = _pxt_json('db', 'start', resources.db_uri)
        assert 'AVAILABLE' in out

    def test_18_service_start(self, resources: Resources) -> None:
        _pxt_json('service', 'start', resources.svc_uri)
        out = _wait_for_state('service', resources.svc_uri, 'AVAILABLE', timeout=120)
        assert 'AVAILABLE' in out

    # ── 19. persistence after stop/start ─────────────────────────────────────

    def test_19_sdk_persistence(self, resources: Resources) -> None:
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

    # ── 20. probe /insert post-restart ────────────────────────────────────────

    def test_20_probe_insert_post_restart(self, resources: Resources) -> None:
        resp = _post(f'{resources.svc_base}/insert', {'id': 9002, 'name': 'post_restart'})
        assert resp.status_code == 200
        assert 'name_upper' in resp.text

    def test_20b_sdk_verify_post_restart_insert(self, resources: Resources) -> None:
        code = f"""
            import pixeltable as pxt
            pxt.init()
            t = pxt.get_table('{resources.table_uri}')
            print('9002:', len(t.where(t.id == 9002).collect()))
        """
        out = _sdk(code)
        assert '9002: 1' in out, f'id=9002 not found after restart:\n{out}'

    # ── 21-22. delete service + DB ────────────────────────────────────────────

    def test_21_service_delete(self, resources: Resources) -> None:
        _pxt('service', 'delete', resources.svc_uri, '--json')

    def test_22_db_delete(self, resources: Resources) -> None:
        _pxt('db', 'delete', resources.db_uri, '--json')
