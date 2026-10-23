"""Call every route of the Lead Enrichment & Outreach API.

Usage:
    python client_demo.py                             # local service (pxt service run app.py leads --port 8000)
    python client_demo.py https://<your-service-url>  # hosted service; set PIXELTABLE_API_KEY first

Uses only the Python standard library. The API key, if any, is read from the
PIXELTABLE_API_KEY environment variable and sent as the X-api-key header.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
import uuid
from typing import Any

BASE = (sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000').rstrip('/')
API_KEY = os.environ.get('PIXELTABLE_API_KEY')


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = dict(extra or {})
    if API_KEY:
        h['X-api-key'] = API_KEY
    return h


def get_json(path: str, params: dict[str, str] | None = None) -> Any:
    qs = f'?{urllib.parse.urlencode(params)}' if params else ''
    req = urllib.request.Request(f'{BASE}{path}{qs}', headers=_headers())
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode())


def get_bytes(path: str, params: dict[str, str]) -> tuple[bytes, str]:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f'{BASE}{path}?{qs}', headers=_headers())
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read(), resp.headers.get('Content-Type', '')


def post_json(path: str, payload: dict[str, object]) -> Any:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{BASE}{path}', data=data, headers=_headers({'Content-Type': 'application/json'}), method='POST'
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    print(
        'preview (stateless compute)',
        post_json(
            '/preview',
            {
                'lead_id': str(uuid.uuid4()),
                'full_name': 'Pierre Brunelle',
                'domain': 'pixeltable.com',
                'role': 'Founder & CEO',
                'stage': 'new',
            },
        ),
    )
    inserted = post_json(
        '/leads',
        {
            'lead_id': str(uuid.uuid4()),
            'full_name': 'Pierre Brunelle',
            'domain': 'pixeltable.com',
            'role': 'Founder & CEO',
            'stage': 'new',
        },
    )
    print('insert', inserted)
    print('all', get_json('/leads/all', {'limit': '10'}))
    print('by-verdict', get_json('/leads/by-verdict', {'verdict': 'valid'}))
    print('by-tier', get_json('/leads/by-tier', {'tier': 'priority', 'stage': 'new'}))
    print('by-domain', get_json('/leads/by-domain', {'domain': 'pixeltable.com'}))
    audio_bytes, ctype = get_bytes('/leads/voice-note', {'lead_id': inserted['lead_id']})
    print('voice-note bytes', len(audio_bytes), 'content-type', ctype)
    print('stage', post_json('/leads/stage', {'lead_id': inserted['lead_id'], 'stage': 'queued'}))
    print('delete', post_json('/leads/delete', {'lead_id': inserted['lead_id']}))


if __name__ == '__main__':
    main()
