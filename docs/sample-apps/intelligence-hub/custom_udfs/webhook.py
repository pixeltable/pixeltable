"""Generic HTTP webhook UDFs.

These connect Pixeltable to any HTTP endpoint -- Zapier, n8n, Make,
or your own services.  Use as computed columns to automatically push
data whenever new rows are inserted.
"""

import json as _json

import requests

import pixeltable as pxt


@pxt.udf
def post(url: str, payload: pxt.Json, headers: pxt.Json | None = None) -> pxt.Json:
    """POST JSON to any URL and return the response.

    Example:
        >>> t.add_computed_column(
        ...     hook=webhook.post('https://hooks.example.com/trigger', t.result_json)
        ... )
    """
    h = {'Content-Type': 'application/json'}
    if headers:
        h.update(headers)
    resp = requests.post(url, data=_json.dumps(payload), headers=h, timeout=30)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {'status': resp.status_code, 'text': resp.text}


@pxt.udf
def get(url: str, headers: pxt.Json | None = None) -> pxt.Json:
    """GET JSON from any URL.

    Useful for pulling data from REST APIs as a computed column.
    """
    h = {}
    if headers:
        h.update(headers)
    resp = requests.get(url, headers=h, timeout=30)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {'status': resp.status_code, 'text': resp.text}
