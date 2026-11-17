"""End-to-end live demo runner for the Modal UDF integration.

Starts an isolated local Pixeltable daemon and FastAPI service against either:
- the direct Modal deployment from deploy_modal_servers.py (.modal_env.json), or
- the Pixeltable Cloud control plane (when .modal_env.json is absent).

Uploads two test videos over HTTP, queries /frames, /similar, and /with-excerpt,
exports annotated frame images with DETR bounding boxes, prints Modal container
call metrics from the logs, and cleanly stops all local processes.
"""

from __future__ import annotations

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
PXT_PORT = '22195'
VIDEO_DIR = PROJECT_ROOT.parent / 'tests/data/videos'

MODAL_LOG_RE = re.compile(
    r'Modal call of (?P<udf>[^:]+): (?P<rows>\d+) rows in (?P<total_s>[0-9.]+)s, '
    r'UDF (?P<udf_s>[0-9.]+)s, container (?P<container>\S+)'
)


def _stop_all(env: dict[str, str], pxt_bin: str) -> None:
    for cmd in ([pxt_bin, 'service', 'stop', 'demo/demo'], [pxt_bin, 'daemon', 'stop']):
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False, capture_output=True, text=True)

    venv_dir = Path(sys.executable).resolve().parent.parent
    pg_ctl = next(venv_dir.rglob('pg_ctl'), None)
    pgdata = PXT_HOME / 'pgdata'
    if pg_ctl is not None and (pgdata / 'postmaster.pid').is_file():
        subprocess.run(
            [str(pg_ctl), '-D', str(pgdata), 'stop', '-m', 'fast'], check=False, capture_output=True, text=True
        )


def _run_pxt(args: list[str], env: dict[str, str], pxt_bin: str) -> str:
    cmd = [pxt_bin, *args]
    print(f'$ {" ".join(cmd)}')
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip(), file=sys.stderr)
    return res.stdout


def _pick_frame(rows: list[dict[str, Any]], *, preferred_label: str | None = None) -> dict[str, Any]:
    if preferred_label is not None:
        matched = [r for r in rows if preferred_label in r.get('detections', {}).get('label_text', [])]
        if matched:
            return matched[0]
    with_boxes = [r for r in rows if r.get('detections', {}).get('boxes')]
    if with_boxes:
        return max(with_boxes, key=lambda r: len(r['detections']['boxes']))
    return rows[0]


def _print_call_summary(modal_calls: list[dict[str, Any]]) -> None:
    if not modal_calls:
        print('\nNo Modal UDF call log lines found.')
        return
    by_udf: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for call in modal_calls:
        by_udf[call['udf']].append(call)

    print('\n=== Modal UDF Call Summary ===')
    print(f'{"UDF":55} {"Calls":>5} {"Rows":>5} {"Cold(s)":>8} {"WarmMin(s)":>10} {"Sample Container"}')
    for udf, calls in sorted(by_udf.items()):
        totals = [c['total_s'] for c in calls]
        rows_sum = sum(c['rows'] for c in calls)
        print(f'{udf:55} {len(calls):>5} {rows_sum:>5} {totals[0]:>8.2f} {min(totals):>10.2f} {calls[0]["container"]}')


def main() -> None:
    bin_dir = Path(sys.executable).resolve().parent
    pxt_bin = str(bin_dir / 'pxt' if (bin_dir / 'pxt').is_file() else 'pxt')

    env = {k: v for k, v in os.environ.items() if k != 'PIXELTABLE_DB_CONNECT_STR'}
    env.update({'PIXELTABLE_HOME': str(PXT_HOME), 'PXT_PORT': PXT_PORT})

    modal_env_path = PROJECT_ROOT / '.modal_env.json'
    if modal_env_path.is_file():
        modal_env = json.loads(modal_env_path.read_text(encoding='utf-8'))
        env['PXT_UDF_SERVERS'] = json.dumps(modal_env['PXT_UDF_SERVERS'])
        env['PXT_UDF_SERVERS_HEADERS'] = json.dumps(modal_env['PXT_UDF_SERVERS_HEADERS'])

    _stop_all(env, pxt_bin)
    if PXT_HOME.exists():
        shutil.rmtree(PXT_HOME)
    PXT_HOME.mkdir(parents=True, exist_ok=True)

    output_media_dir = PROJECT_ROOT / 'output_media'
    output_media_dir.mkdir(exist_ok=True)

    results: dict[str, Any] = {'steps': {}}
    try:
        _run_pxt(['init'], env, pxt_bin)
        _run_pxt(['schema', 'update', 'app.py', 'demo'], env, pxt_bin)
        _run_pxt(['service', 'update', 'app.py', 'demo', '-f'], env, pxt_bin)

        svc_list = json.loads(_run_pxt(['service', 'list', 'demo', '--json'], env, pxt_bin))
        svc = next(s for s in svc_list if s['name'] == 'demo')
        base_url = svc['endpoint'].rstrip('/')
        results['service'] = svc
        print(f'\nService live at: {base_url}\n')

        with httpx.Client(timeout=600.0) as client:
            v1_path = VIDEO_DIR / 'bangkok_half_res.mp4'
            print(f'Uploading Video 1 (cold start): {v1_path.name}...')
            t0 = time.monotonic()
            with v1_path.open('rb') as f:
                r1 = client.post(
                    f'{base_url}/videos', data={'video_id': '1'}, files={'video': (v1_path.name, f, 'video/mp4')}
                )
            r1.raise_for_status()
            t_v1 = round(time.monotonic() - t0, 2)
            results['steps']['insert_video_1_cold'] = {'elapsed_s': t_v1, 'response': r1.json()}
            print(f'Video 1 ingested in {t_v1}s: duration={r1.json().get("duration")}')

            v2_path = VIDEO_DIR / "filename with spaces and '_half_res.mp4"
            print(f'Uploading Video 2 (warm containers): {v2_path.name}...')
            t0 = time.monotonic()
            with v2_path.open('rb') as f:
                r2 = client.post(
                    f'{base_url}/videos', data={'video_id': '2'}, files={'video': (v2_path.name, f, 'video/mp4')}
                )
            r2.raise_for_status()
            t_v2 = round(time.monotonic() - t0, 2)
            results['steps']['insert_video_2_warm'] = {'elapsed_s': t_v2, 'response': r2.json()}
            print(f'Video 2 ingested in {t_v2}s: duration={r2.json().get("duration")}')

            f1_rows = client.post(f'{base_url}/frames', json={'video_id': 1}).raise_for_status().json()['rows']
            f2_rows = client.post(f'{base_url}/frames', json={'video_id': 2}).raise_for_status().json()['rows']
            results['steps']['frames_counts'] = {'video_1': len(f1_rows), 'video_2': len(f2_rows)}

            for key, query_text in (
                ('similar_traffic', 'busy city street with cars and traffic'),
                ('similar_cat', 'a cat relaxing on a couch'),
            ):
                t0 = time.monotonic()
                resp = client.post(f'{base_url}/similar', json={'text': query_text, 'limit': 3}).raise_for_status()
                rows = resp.json()['rows']
                elapsed = round(time.monotonic() - t0, 3)
                results['steps'][key] = {'query': query_text, 'elapsed_s': elapsed, 'rows': rows}
                top = rows[0] if rows else {}
                print(
                    f'Query {query_text!r} ({elapsed}s) -> video_id={top.get("video_id")} '
                    f'pos={top.get("pos")} sim={top.get("sim"):.4f} caption={top.get("caption")!r}'
                )

            t0 = time.monotonic()
            exc_rows = client.post(f'{base_url}/with-excerpt', json={'video_id': 1}).raise_for_status().json()['rows']
            results['steps']['query_with_excerpt'] = {'elapsed_s': round(time.monotonic() - t0, 3), 'rows': exc_rows}

            best_v1 = _pick_frame(f1_rows)
            best_v2 = _pick_frame(f2_rows, preferred_label='cat')
            for filename, media_rel in (
                ('bangkok_annotated_frame.png', best_v1['annotated']),
                ('cat_annotated_frame.png', best_v2['annotated']),
            ):
                img_bytes = client.get(urljoin(f'{base_url}/', media_rel.lstrip('/'))).raise_for_status().content
                (output_media_dir / filename).write_bytes(img_bytes)
                print(f'Saved {output_media_dir.name}/{filename} ({len(img_bytes)} bytes)')

        modal_calls: list[dict[str, Any]] = []
        raw_lines: list[str] = []
        for log_file in sorted(PXT_HOME.rglob('*.log')):
            for line in log_file.read_text(encoding='utf-8', errors='replace').splitlines():
                if 'Modal call of' in line:
                    raw_lines.append(line)
                    if (m := MODAL_LOG_RE.search(line)) is not None:
                        modal_calls.append(
                            {
                                'udf': m.group('udf'),
                                'rows': int(m.group('rows')),
                                'total_s': float(m.group('total_s')),
                                'udf_s': float(m.group('udf_s')),
                                'container': m.group('container'),
                            }
                        )
        results['modal_calls'] = modal_calls
        results['raw_modal_log_lines'] = raw_lines
        _print_call_summary(modal_calls)

        results_path = PROJECT_ROOT / 'demo_results.json'
        results_path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f'\nWrote {results_path.name} ({len(modal_calls)} Modal calls).')
    finally:
        print('\nStopping local Pixeltable service, daemon, and Postgres...')
        _stop_all(env, pxt_bin)


if __name__ == '__main__':
    main()
