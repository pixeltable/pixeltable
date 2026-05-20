"""Stdlib-only HTTP client.

Keeping this dependency-free shaves ~80-100ms off every pxt invocation
(no httpx/anyio/h11/idna/certifi imports). probe.py already uses urllib
for health probes; this extends the same pattern to the request path.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pxt_cli.probe import ensure_running


def _request(method: str, path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)

    url = f'{base}{path}'
    if params is not None:
        # Drop unset values so the daemon sees its default; coerce bool to '1'/'0' (FastAPI
        # accepts both, but '1'/'0' is what the rest of the CLI sends).
        filtered = {k: ('1' if v is True else '0' if v is False else v) for k, v in params.items() if v is not None}
        if len(filtered) > 0:
            # doseq=True expands list values into repeated params (?pk=a&pk=b), which is
            # how FastAPI's Query(...) list-typed parameters expect to receive them.
            url += '?' + urllib.parse.urlencode(filtered, doseq=True)

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
        print(f'pxt: {e.code} {detail}', file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f'pxt: cannot reach daemon at {url}: {e.reason}', file=sys.stderr)
        sys.exit(1)


def get(path: str, params: dict[str, Any] | None = None) -> Any:
    return _request('GET', path, params=params)


def post(path: str, body: dict[str, Any]) -> Any:
    return _request('POST', path, body=body)


def quote_path(path: str) -> str:
    """Validate and URL-encode a pxt path. Slashes are preserved (FastAPI's path
    converter matches them as part of the path parameter); other special characters get
    percent-encoded. Bad shapes ('.' separator, leading/trailing '/', '//') exit 2 with a
    clear message before any network round-trip."""
    if '.' in path:
        _exit_bad_path(f"pxt paths use '/' as the separator; got {path!r}")
    if path.startswith('/'):
        _exit_bad_path(f"pxt paths are relative; drop the leading '/' (use '' for root). Got {path!r}")
    if path.endswith('/'):
        _exit_bad_path(f"pxt paths must not end with '/'; got {path!r}")
    if '//' in path:
        _exit_bad_path(f"pxt paths must not contain empty components ('//'); got {path!r}")
    return urllib.parse.quote(path, safe='/')


def _exit_bad_path(message: str) -> None:
    print(f'pxt: {message}', file=sys.stderr)
    sys.exit(2)
