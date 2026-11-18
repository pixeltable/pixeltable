"""Run the demo end to end in its own database, .pxt_home, and stop it again.

Creates the tables, starts the service, uploads two test videos, queries every route, saves an annotated frame of each
video to output_media/, and prints a summary of the Modal calls. The UDF servers are the ones in PXT_UDF_SERVERS
(source .modal_env.sh first), or else the local database's in Pixeltable Cloud. README.md says how to run it.
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
PXT_HOME = PROJECT_ROOT / '.pxt_home'
OUTPUT_DIR = PROJECT_ROOT / 'output_media'
VIDEO_DIR = PROJECT_ROOT.parent / 'tests/data/videos'
VIDEOS = {1: 'bangkok_half_res.mp4', 2: "filename with spaces and '_half_res.mp4"}
QUERIES = ('busy city street with cars and traffic', 'a cat relaxing on a couch')
# the line ModalServerClient.call() logs at DEBUG
MODAL_CALL_RE = re.compile(
    r'Modal call of (?P<udf>[^:]+): (?P<rows>\d+) rows in (?P<total_s>[0-9.]+)s, '
    r'UDF (?P<udf_s>[0-9.]+)s, container (?P<container>\S+)'
)

ENV = {k: v for k, v in os.environ.items() if k != 'PIXELTABLE_DB_CONNECT_STR'}
ENV.update(PIXELTABLE_HOME=str(PXT_HOME), PXT_PORT='22195')


def pxt(*args: str) -> str:
    print(f'$ pxt {" ".join(args)}')
    return subprocess.run(['pxt', *args], cwd=PROJECT_ROOT, env=ENV, check=True, capture_output=True, text=True).stdout


def stop_all() -> None:
    for args in (['service', 'stop', 'demo/demo'], ['daemon', 'stop']):
        subprocess.run(['pxt', *args], cwd=PROJECT_ROOT, env=ENV, check=False, capture_output=True)
    # `pxt daemon stop` leaves the embedded Postgres server running
    venv = Path(sys.executable).parent.parent
    pg_ctl = next(venv.glob('lib/python3.*/site-packages/pixeltable_pgserver/pginstall/bin/pg_ctl'), None)
    if pg_ctl is not None and (PXT_HOME / 'pgdata/postmaster.pid').is_file():
        subprocess.run([pg_ctl, '-D', PXT_HOME / 'pgdata', 'stop', '-m', 'fast'], check=False, capture_output=True)


def most_detections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: len(row['detections']['boxes']))


def modal_calls() -> list[dict[str, Any]]:
    calls = []
    for log_file in sorted(PXT_HOME.rglob('*.log')):
        for m in MODAL_CALL_RE.finditer(log_file.read_text(encoding='utf-8', errors='replace')):
            calls.append(
                {
                    'udf': m['udf'],
                    'rows': int(m['rows']),
                    'total_s': float(m['total_s']),
                    'udf_s': float(m['udf_s']),
                    'container': m['container'],
                }
            )
    return calls


def print_call_summary(calls: list[dict[str, Any]]) -> None:
    by_udf: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for call in calls:
        by_udf[call['udf']].append(call)
    print(f'\n{"UDF":60} {"calls":>5} {"rows":>5} {"first (s)":>9} {"fastest (s)":>11}')
    for udf, udf_calls in sorted(by_udf.items()):
        times = [c['total_s'] for c in udf_calls]
        rows = sum(c['rows'] for c in udf_calls)
        print(f'{udf:60} {len(udf_calls):>5} {rows:>5} {times[0]:>9.2f} {min(times):>11.2f}')


def main() -> None:
    stop_all()
    shutil.rmtree(PXT_HOME, ignore_errors=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    try:
        pxt('schema', 'update', 'app.py', 'demo')
        pxt('service', 'update', 'app.py', 'demo', '-f')
        service = next(s for s in json.loads(pxt('service', 'list', 'demo', '--json')) if s['name'] == 'demo')
        url = service['endpoint'].rstrip('/')
        print(f'Service at {url}')

        with httpx.Client(base_url=url, timeout=600.0) as client:
            # the first insert starts the containers; the second finds them warm
            for video_id, name in VIDEOS.items():
                start = time.monotonic()
                with (VIDEO_DIR / name).open('rb') as f:
                    files = {'video': (name, f, 'video/mp4')}
                    client.post('/videos', data={'video_id': str(video_id)}, files=files).raise_for_status()
                elapsed = round(time.monotonic() - start, 2)
                results[f'insert_{video_id}_s'] = elapsed
                print(f'Inserted video {video_id} in {elapsed} s')

            for text in QUERIES:
                rows = client.post('/similar', json={'text': text}).raise_for_status().json()['rows']
                results[f'similar: {text}'] = rows
                for row in rows:
                    print(f'{text!r}: video {row["video_id"]} pos {row["pos"]} sim {row["sim"]:.3f} {row["caption"]!r}')

            client.post('/with-excerpt', json={'video_id': 1}).raise_for_status()
            for video_id in VIDEOS:
                rows = client.post('/frames', json={'video_id': video_id}).raise_for_status().json()['rows']
                media = most_detections(rows)['annotated']
                path = OUTPUT_DIR / f'video_{video_id}_annotated.png'
                path.write_bytes(client.get(urljoin(f'{url}/', media.lstrip('/'))).raise_for_status().content)
                print(f'{len(rows)} frames of video {video_id}; saved {path.relative_to(PROJECT_ROOT)}')

        results['modal_calls'] = calls = modal_calls()
        print_call_summary(calls)
        (PROJECT_ROOT / 'demo_results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    finally:
        stop_all()


if __name__ == '__main__':
    main()
