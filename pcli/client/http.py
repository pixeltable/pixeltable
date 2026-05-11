import os
import sys
from typing import Any


def _check(r: Any) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json().get('detail') or r.text
        except Exception:
            detail = r.text
        print(f'pcli: {r.status_code} {detail}', file=sys.stderr)
        sys.exit(1)
    return r.json()


def get(path: str) -> Any:
    import httpx
    from pcli.probe import ensure_running
    r = httpx.get(f'{ensure_running()}{path}', headers={'X-Cwd': os.getcwd()}, timeout=30)
    return _check(r)


def post(path: str, body: dict) -> Any:
    import httpx
    from pcli.probe import ensure_running
    r = httpx.post(f'{ensure_running()}{path}', json=body, headers={'X-Cwd': os.getcwd()}, timeout=30)
    return _check(r)
