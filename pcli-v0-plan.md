# pcli v0 — scaffolding + `pcli ls`

Implementation plan for the first slice: get a working `pcli ls` end-to-end,
plus the daemon process that the CLI and (eventually) the dashboard both
share.

## Architecture target

Single always-on FastAPI daemon process. Serves both:
- pcli HTTP API (`/pcli/v0/*`).
- Dashboard (currently served by an in-process stdlib `ThreadingHTTPServer`
  in a daemon thread; to be migrated to FastAPI routes in this same daemon
  process).

End state: no more in-process dashboard thread inside arbitrary pxt
processes. Pxt processes that need dashboard functionality just talk to the
daemon over HTTP.

Port: 22089 (existing dashboard port; the daemon takes ownership).

Process model: one daemon per machine, started on demand by either:
- `pcli` clients (probe-or-spawn).
- `import pixeltable` (probe-or-spawn from `Env` initialization).

Whichever triggers it first wins; both detect via the existing
`/api/pixeltable-health` probe.

## Packaging

Single package. The `pcli` subtree lives under the `pixeltable` package
(e.g., `pixeltable/cli/`), or as a sibling top-level `pcli/` inside the same
wheel. No advantage to splitting into a separate distributable for the AI
use case — pcli depends heavily on pxt internals; they'd version-lock anyway.
Decision: top-level `pcli/` inside the same wheel (cleaner import paths,
clearer ownership).

## Directory layout

```
pcli/
├── __init__.py
├── models.py             # pydantic models (server only)
├── server/
│   ├── __init__.py
│   ├── app.py            # FastAPI app construction; mounts all routers
│   ├── daemon.py         # `python -m pcli.server.daemon` entry point
│   ├── probe.py          # shared "is the daemon running?" / spawn helper
│   └── routes/
│       ├── __init__.py
│       ├── health.py     # /api/pixeltable-health + /pcli/v0/health
│       └── ls.py         # /pcli/v0/ls
└── client/
    ├── __init__.py
    ├── main.py           # pcli console script entry; argv dispatch
    ├── http.py           # httpx wrapper, X-Cwd header
    └── commands/
        ├── __init__.py
        └── ls.py
```

Dashboard routes get added under `pcli/server/routes/dashboard/` in a later
phase (see Phasing below).

## Server side

### `pcli/server/app.py`

```python
from fastapi import FastAPI
from .routes import health, ls

def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)  # internal-only; no /docs
    app.include_router(health.router)
    app.include_router(ls.router)
    return app
```

### `pcli/server/daemon.py`

Entry point: `python -m pcli.server.daemon`. Binds 127.0.0.1:22089, runs
uvicorn with the FastAPI app.

```python
import uvicorn
from .app import create_app

PORT = 22089

def main():
    uvicorn.run(create_app(), host='127.0.0.1', port=PORT, log_config=None)

if __name__ == '__main__':
    main()
```

Logs: uvicorn stdout/stderr → `~/.pixeltable/logs/pcli-daemon.log` (handled
by the caller redirecting stdio at spawn time).

### `pcli/models.py` (server only)

```python
from typing import Literal
from pydantic import BaseModel

class LsEntry(BaseModel):
    path: str
    kind: Literal['table', 'view', 'dir']
    num_rows: int | None = None
    num_cols: int | None = None
    last_version: int | None = None
    flags: str = ''

class LsRequest(BaseModel):
    path: str = '/'
    tree: bool = False
    long: bool = False
    no_counts: bool = False

class LsResponse(BaseModel):
    entries: list[LsEntry]
    tree: dict | None = None
```

### `pcli/server/routes/health.py`

```python
from fastapi import APIRouter
import pixeltable
import os, datetime

router = APIRouter()

# kept for compat with existing dashboard probe
@router.get('/api/pixeltable-health')
def dashboard_health():
    return {'status': 'ok'}

@router.get('/pcli/v0/health')
def pcli_health():
    return {
        'ok': True,
        'pxt_version': pixeltable.__version__,
        'pid': os.getpid(),
        'started_at': datetime.datetime.utcnow().isoformat() + 'Z',
    }
```

### `pcli/server/routes/ls.py`

```python
from fastapi import APIRouter, Header
import pixeltable as pxt
from pcli.models import LsRequest, LsResponse, LsEntry

router = APIRouter()

@router.post('/pcli/v0/ls', response_model=LsResponse)
def ls(req: LsRequest, x_cwd: str | None = Header(default=None)):
    entries: list[LsEntry] = []
    # flat listing of req.path; populate fields conditionally on req.long
    for name in pxt.list_dirs(req.path):
        entries.append(LsEntry(path=name, kind='dir'))
    for name in pxt.list_tables(req.path):
        t = pxt.get_table(name)
        entries.append(LsEntry(
            path=name,
            kind='view' if t.is_view else 'table',
            num_rows=None if req.no_counts else t.count(),
            num_cols=len(t.schema) if req.long else None,
        ))
    if req.tree:
        # walk recursively; build nested dict
        ...
    return LsResponse(entries=entries)
```

### `pcli/server/probe.py`

Shared between client (decides whether to spawn) and Env init (same
decision). Single source of truth for "is the daemon running?"

```python
import urllib.request, urllib.error, json

HEALTH_URL = 'http://localhost:22089/api/pixeltable-health'

def is_running(timeout: float = 0.3) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return json.loads(r.read()).get('status') == 'ok'
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False

def spawn_detached():
    import subprocess, os
    log_path = os.path.expanduser('~/.pixeltable/logs/pcli-daemon.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, 'a')
    subprocess.Popen(
        ['python', '-m', 'pcli.server.daemon'],
        start_new_session=True, stdout=log, stderr=log,
    )

def wait_for_health(timeout: float = 10.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return
        time.sleep(0.1)
    raise RuntimeError('daemon did not come up')
```

Used by `Env` init (replaces the current `DashboardHarness.start()`) — same
probe, same spawn, but spawns the external daemon process instead of a
thread.

## Client side

### `pcli/client/main.py`

```python
import sys

def main():
    if len(sys.argv) < 2:
        print('usage: pcli <command> [...]', file=sys.stderr); sys.exit(2)
    cmd = sys.argv[1]
    if cmd == 'ls':
        from .commands import ls
        ls.run(sys.argv[2:])
    elif cmd == 'health':
        from .commands import health
        health.run(sys.argv[2:])
    else:
        print(f'unknown command: {cmd}', file=sys.stderr); sys.exit(2)
```

Manual argv parsing; no click; lazy imports.

### `pcli/client/http.py`

```python
def post(path: str, body: dict) -> dict:
    import httpx, os
    from pcli.server.probe import is_running, spawn_detached, wait_for_health
    if not is_running():
        spawn_detached()
        wait_for_health()
    r = httpx.post(f'http://localhost:22089{path}', json=body,
                   headers={'X-Cwd': os.getcwd()}, timeout=30)
    r.raise_for_status()
    return r.json()

def get(path: str) -> dict:
    # same shape, no body
    ...
```

Note: client imports `pcli.server.probe`, but probe.py only imports stdlib
+ `subprocess` — no pydantic, no fastapi, no pxt. Safe for client cold-start.

### `pcli/client/commands/ls.py`

```python
import argparse, json
from ..http import post

def run(argv):
    ap = argparse.ArgumentParser(prog='pcli ls')
    ap.add_argument('path', nargs='?', default='/')
    ap.add_argument('--tree', action='store_true')
    ap.add_argument('-l', '--long', action='store_true')
    ap.add_argument('--no-counts', action='store_true')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    resp = post('/pcli/v0/ls', {
        'path': args.path, 'tree': args.tree, 'long': args.long,
        'no_counts': args.no_counts,
    })

    if args.as_json:
        print(json.dumps(resp))
        return

    for e in resp['entries']:
        cols = [e['path'], e['kind']]
        if e.get('num_rows') is not None:
            cols.append(str(e['num_rows']))
        if args.long:
            cols.append(str(e.get('num_cols') or ''))
            cols.append(e.get('flags', ''))
        print('\t'.join(cols))
```

## Changes to existing pixeltable code

### `pixeltable/env.py` (replace `DashboardHarness` thread with external spawn)

Today: `Env._initialize` does `self._dashboard_harness = DashboardHarness(port);
self._dashboard_harness.start()`. The harness starts the dashboard server in a
daemon thread.

Replacement: call `pcli.server.probe.is_running()` and
`pcli.server.probe.spawn_detached()` instead. Drops the in-thread server and
all watchdog complexity.

Whether to keep the `DashboardHarness` class as a shim (with new logic) or
delete it: delete. Net negative LOC.

### `pixeltable/dashboard/server.py`, `harness.py`

In phase 1 (v0): leave as-is. The daemon's `/api/pixeltable-health` handler
satisfies the existing probe so the old harness code is now dead inside pxt
processes (probe returns 'pixeltable', they attach as watchdog — but with
the harness removed, this path is gone).

In phase 2 (separate task): port dashboard routes from `dashboard/server.py`
into FastAPI routes under `pcli/server/routes/dashboard/`. Delete the stdlib
server. The bridge module (`dashboard/bridge.py`) presumably contains the
real logic; it gets reused.

## Phasing

**Phase 1 (v0 scope — this task):**
1. Scaffold `pcli/` tree.
2. Build the FastAPI daemon (app, daemon entry point, health route).
3. Implement `/pcli/v0/ls` server-side + `pcli ls` client-side.
4. Replace `Env`'s dashboard-thread spawn with external-daemon spawn.
5. Verify dashboard UI still works through the FastAPI daemon's
   `/api/pixeltable-health` (probe-only; dashboard requests still served by
   whatever path the existing static files use — TBD, see open questions).

**Phase 2 (separate task):**
- Port dashboard endpoints to FastAPI routes inside the same daemon.
- Static file serving via `StaticFiles`.
- Delete `pixeltable/dashboard/{server,harness}.py`.

**Phase 3 (separate task):**
- More pcli commands per the main design doc (describe, head, errors,
  endpoints, logs, lint, etc.).

## Test structure

Pytest-based, designed to **mix CLI invocations with Python API calls in the
same test**. Common pattern: create state via API, validate via CLI; or
mutate via CLI, assert via API.

### Fixtures (in `tests/pcli/conftest.py`)

```python
import json, subprocess
from dataclasses import dataclass
import pytest
from pcli.server.probe import is_running, spawn_detached, wait_for_health

@pytest.fixture(scope='session')
def daemon():
    """Ensure the pcli daemon is running for the test session."""
    if not is_running():
        spawn_detached()
        wait_for_health(timeout=15)
    yield
    # Don't kill: leave the daemon running across test sessions for speed.
    # Tests must be tolerant of pre-existing daemon state.

@dataclass
class PcliResult:
    returncode: int
    stdout: str
    stderr: str
    @property
    def json(self):
        return json.loads(self.stdout)

@pytest.fixture
def pcli(daemon):
    def _run(*args: str, check: bool = True) -> PcliResult:
        r = subprocess.run(['pcli', *args], capture_output=True, text=True)
        if check and r.returncode != 0:
            raise AssertionError(f'pcli {args} failed: {r.stderr}')
        return PcliResult(r.returncode, r.stdout, r.stderr)
    return _run
```

### Test shape

```python
def test_ls_reflects_api_state(uses_db, pcli):
    # set up via API
    pxt.create_table('foo', {'x': pxt.Int})
    pxt.create_dir('subdir')

    # validate via CLI
    out = pcli('ls', '--json').json
    paths = {e['path'] for e in out['entries']}
    assert 'foo' in paths
    assert 'subdir' in paths

    # mutate via API
    pxt.drop_table('foo')

    # validate via CLI again
    out = pcli('ls', '--json').json
    paths = {e['path'] for e in out['entries']}
    assert 'foo' not in paths
```

### Cross-process state consistency

The pytest process and the daemon process are separate. Both connect to the
same postgres instance (same `~/.pixeltable/pgdata`), so the underlying
catalog is shared. But each process has its own `Env`, plan cache, and
catalog metadata caches.

**Critical decision for v0:** the daemon's CLI handlers must query catalog
metadata fresh on each request (no caching across requests for
`list_tables`, `get_table`, etc.). Otherwise tests like the one above will
see stale CLI output after `pxt.drop_table('foo')` in the pytest process.

If pxt has cross-process catalog invalidation today (via postgres notify or
version checks), we get this for free. If not, the daemon either:
- Re-reads catalog state on every request (simplest, fine for v0).
- Subscribes to a postgres notify channel for invalidation (cleaner but
  more work; deferred).

Verify behavior with the smoke test above; if it fails due to staleness,
add a force-refresh call in the handler.

### Cold-start measurement

A separate test that times `pcli ls` repeatedly (after warm-up) and asserts
p50 < 100ms. Skipped in CI by default (timing-sensitive); run locally with
`pytest -m perf`.

```python
@pytest.mark.perf
def test_pcli_cold_start(uses_db, pcli):
    # warm
    pcli('ls', '--json')
    # measure
    import time
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        pcli('ls', '--json')
        samples.append(time.perf_counter() - t0)
    samples.sort()
    p50 = samples[len(samples)//2]
    assert p50 < 0.1, f'p50={p50*1000:.0f}ms'
```

## Open questions

1. **Dashboard UI in phase 1.** The existing dashboard is served from
   `pixeltable/dashboard/static/`. If we drop the in-process server in
   phase 1, what serves the static UI? Two options:
   - (a) Phase 1 also adds a minimal static-file route to the daemon
     (`StaticFiles(directory=...)`); dashboard UI keeps working.
   - (b) Phase 1 breaks dashboard temporarily; phase 2 restores it via
     proper FastAPI route migration. Less attractive — dashboard is a
     visible feature.
   I'd recommend (a): one extra line in the daemon to mount
   `StaticFiles`. Cheap to do as part of v0.
2. **Probe-and-spawn from `Env` init**: synchronous (block until daemon up)
   or fire-and-forget? Synchronous means `import pixeltable` waits up to
   ~10s for the daemon to start. Fire-and-forget means dashboard might not
   be reachable for a few seconds after `import pixeltable`. Today's
   threading harness is fire-and-forget-ish (thread starts in background).
   I'd match that: fire-and-forget, log a warning if it doesn't come up
   within 10s.
3. **What about pxt processes started before the daemon convention exists
   in the codebase?** Old in-thread dashboard code is still there in phase 1
   (just disconnected from Env). Should we delete the dead code in phase 1
   or wait for phase 2? Delete in phase 1 — keeping dead code around is
   confusing.
4. **Cross-process catalog freshness.** Daemon serves CLI requests against
   the shared postgres but holds its own per-process catalog cache. Tests
   that mutate via Python API in the pytest process and then validate via
   CLI need fresh state. Plan: handlers re-read catalog metadata each
   request. Verify via the smoke test above; add notify-based invalidation
   later if perf matters.

## Decisions taken

- Daemon is a FastAPI process owning port 22089.
- Dashboard moves into the daemon (phase 2 migrates routes; phase 1 keeps
  the dashboard UI reachable via a `StaticFiles` mount on the daemon).
- Single package; `pcli/` lives in the pixeltable wheel.
- Pydantic stays server-side only.
- Probe-and-spawn shared between client and `Env` init via
  `pcli.server.probe`.
- Daemon runs forever (no idle timeout in v0).
- Daemon stdio → `~/.pixeltable/logs/pcli-daemon.log`.
