"""Runs the storage benchmarks (`storage_bench.py`), each case in its own process, and prints a table.

    python run_storage.py [CASE ...] [--rows N] [--python PY] [--out results.json]
    python run_storage.py [CASE ...] --builds before=/path/to/pkg after= [--repeat 3] [--out results.json]
    python run_storage.py --compare before.json after.json [more.json ...]

Run it from any directory except the repository root (whose `pixeltable/` is upstream's package). `--python` selects
the interpreter (default: the pixeltable-rs venv). `--builds` runs every case once per build per repeat, alternating
builds so that they see the same machine load, and reports the median of the repeats; a build is a directory put first
on `PYTHONPATH` (holding a `pixeltable` package), or empty for the interpreter's own. `--compare` prints the metrics of
earlier runs side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = HERE.parent / '.venv' / 'bin' / 'python'
sys.path.insert(0, str(HERE))
from storage_bench import CASES  # noqa: E402


def run_case(case: str, rows: int, python: str, pythonpath: str = '') -> dict:
    env = dict(os.environ)
    if pythonpath:
        env['PYTHONPATH'] = pythonpath + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    t = time.perf_counter()
    proc = subprocess.run(
        [python, str(HERE / 'storage_bench.py'), case, '--rows', str(rows)], capture_output=True, text=True, cwd='/', env=env
    )
    wall = time.perf_counter() - t
    if proc.returncode != 0:
        print(f'{case}: FAILED\n{proc.stderr[-2000:]}', file=sys.stderr)
        return {'error': proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'failed'}
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f'{case} [{pythonpath or "default"}] ({wall:.1f} s): {out}', file=sys.stderr)
    return out


def median_results(runs: list[dict]) -> dict:
    ok = [r for r in runs if 'error' not in r]
    if not ok:
        return runs[0] if runs else {}
    return {m: round(statistics.median(r[m] for r in ok if m in r), 3) for m in ok[0]}


def table(runs: list[tuple[str, dict]]) -> str:
    header = ['case', 'metric'] + [name for name, _ in runs]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    cases: list[str] = []
    for _, r in runs:
        for c in r.get('results', {}):
            if c not in cases:
                cases.append(c)
    for c in cases:
        metrics: list[str] = []
        for _, r in runs:
            for m in r.get('results', {}).get(c, {}):
                if m not in metrics:
                    metrics.append(m)
        for m in metrics:
            vals = [str(r.get('results', {}).get(c, {}).get(m, '')) for _, r in runs]
            lines.append(f'| {c} | {m} | ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*')
    ap.add_argument('--rows', type=int, default=100_000)
    ap.add_argument('--python', default=str(DEFAULT_PYTHON))
    ap.add_argument('--out')
    ap.add_argument('--label', default='')
    ap.add_argument('--builds', nargs='+', metavar='NAME=PYTHONPATH')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--compare', nargs='+')
    args = ap.parse_args()
    if args.compare:
        runs: list[tuple[str, dict]] = []
        for p in args.compare:
            d = json.loads(Path(p).read_text())
            runs.extend(d['builds'].items() if 'builds' in d else [(d.get('label') or Path(p).stem, d)])
        print(table(runs))
        return
    cases = args.cases or list(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        ap.error(f'unknown cases: {unknown}; known: {sorted(CASES)}')
    builds = [tuple(b.split('=', 1)) if '=' in b else (b, '') for b in (args.builds or [args.label or 'result='])]
    raw: dict[str, dict[str, list[dict]]] = {name: {c: [] for c in cases} for name, _ in builds}
    for _ in range(args.repeat):
        for c in cases:
            for name, path in builds:
                raw[name][c].append(run_case(c, args.rows, args.python, path))
    outs = [
        (name, {'label': name, 'rows': args.rows, 'repeat': args.repeat, 'results': {c: median_results(raw[name][c]) for c in cases}})
        for name, _ in builds
    ]
    if args.out:
        data = outs[0][1] if len(outs) == 1 else {'rows': args.rows, 'repeat': args.repeat, 'builds': dict(outs)}
        Path(args.out).write_text(json.dumps(data, indent=2) + '\n')
    print(table(outs))


if __name__ == '__main__':
    main()
