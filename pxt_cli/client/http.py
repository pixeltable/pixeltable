"""Stdlib-only HTTP client.

Keeping this dependency-free shaves ~80-100ms off every pcli invocation
(no httpx/anyio/h11/idna/certifi imports). probe.py already uses urllib
for health probes; this extends the same pattern to the request path.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from pxt_cli.probe import ensure_running


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pcli: {e}', file=sys.stderr)
        sys.exit(1)
    url = f'{base}{path}'
    headers = {'X-Cwd': os.getcwd()}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        try:
            detail = (json.loads(e.read() or b'null') or {}).get('detail') or e.reason
        except (ValueError, AttributeError):
            detail = e.reason
        print(f'pcli: {e.code} {detail}', file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f'pcli: cannot reach daemon at {url}: {e.reason}', file=sys.stderr)
        sys.exit(1)


def get(path: str) -> Any:
    return _request('GET', path)


def post(path: str, body: dict[str, Any]) -> Any:
    return _request('POST', path, body)
