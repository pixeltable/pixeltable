"""Stdlib-only HTTP client.

Keeping this dependency-free reduces startup time of the client.
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pixeltable_cli.client.utils import ensure_running
from pixeltable_cli.utils import validate_path_shape


def _request(method: str, path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)

    url = f'{base}{path}'
    if params is not None:
        # Drop unset values so the daemon sees its default; coerce bool to '1'/'0' to
        # match the server's query_bool parser.
        filtered = {k: ('1' if v is True else '0' if v is False else v) for k, v in params.items() if v is not None}
        if len(filtered) > 0:
            # doseq=True expands list values into repeated params (?pk=a&pk=b).
            url += '?' + urllib.parse.urlencode(filtered, doseq=True)

    headers: dict[str, str] = {}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    # No timeout: localhost call, legitimate operations have no defensible upper bound.
    try:
        with urllib.request.urlopen(req) as r:
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
    """Validate and URL-encode a pxt path. Slashes are preserved (the router's path
    converter matches them as part of the path parameter); other special characters get
    percent-encoded. Bad shapes ('.' separator, leading/trailing '/', '//') exit 2 with a
    clear message before any network round-trip."""
    err = validate_path_shape(path)
    if err is not None:
        print(f'pxt: {err}', file=sys.stderr)
        sys.exit(2)
    return urllib.parse.quote(path, safe='/')
