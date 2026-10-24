"""Runs the storage benchmarks (`storage_bench.py`), each case in its own process, and prints a table.

    python run_storage.py [CASE ...] [--rows N] [--python PY] [--out results.json]
    python run_storage.py --compare before.json after.json [more.json ...]

Run it from any directory except the repository root (whose `pixeltable/` is upstream's package). `--python` selects
the interpreter (default: the pixeltable-rs venv); `--compare` prints the metrics of earlier runs side by side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = HERE.parent / '.venv' / 'bin' / 'python'
sys.path.insert(0, str(HERE))
from storage_bench import CASES  # noqa: E402


def run(cases: list[str], rows: int, python: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for case in cases:
        t = time.perf_counter()
        proc = subprocess.run(
            [python, str(HERE / 'storage_bench.py'), case, '--rows', str(rows)],
            capture_output=True,
            text=True,
            cwd='/',
        )
        wall = time.perf_counter() - t
        if proc.returncode != 0:
            print(f'{case}: FAILED\n{proc.stderr[-2000:]}', file=sys.stderr)
            results[case] = {'error': proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'failed'}
            continue
        results[case] = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f'{case} ({wall:.1f} s): {results[case]}', file=sys.stderr)
    return results


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
    ap.add_argument('--compare', nargs='+')
    args = ap.parse_args()
    if args.compare:
        runs = [(json.loads(Path(p).read_text()).get('label') or Path(p).stem, json.loads(Path(p).read_text())) for p in args.compare]
        print(table(runs))
        return
    cases = args.cases or list(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        ap.error(f'unknown cases: {unknown}; known: {sorted(CASES)}')
    out = {'label': args.label, 'rows': args.rows, 'python': args.python, 'results': run(cases, args.rows, args.python)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(table([(args.label or 'result', out)]))


if __name__ == '__main__':
    main()
