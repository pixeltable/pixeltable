"""Runs the pixeltable-rs benchmarks: one fresh process (and $PIXELTABLE_HOME) per benchmark.

    .venv/bin/python bench/run.py                       # all benchmarks, with this interpreter's pixeltable
    .venv/bin/python bench/run.py sync_sleep_insert ... # a selection
    .venv/bin/python bench/run.py --json out.json --compare bench/results/baseline.json
    .venv/bin/python bench/run.py --build before=/path/to/python --build upstream=... --build after=.venv/bin/python

Prints a table (seconds; with --compare also the baseline and the speedup) and writes the results as JSON. With
`--build NAME=PYTHON` (repeatable) every benchmark runs once per build, the builds one after the other so that they
see the same machine load, and the table shows each build and the speedup of the last build over each other one.
Extra environment for every benchmark process can be given with --env KEY=VALUE (e.g. PIXELTABLE_UDF_THREADS=1).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import benchmarks  # noqa: E402


def run(name: str, python: str, extra_env: dict[str, str], timeout: float) -> dict[str, float] | str:
    home = tempfile.mkdtemp(prefix='pxtrs-bench-')
    env = dict(os.environ)
    env.update(
        {
            'PIXELTABLE_HOME': home,
            'PYTHONPATH': str(BENCH_DIR),
            # BLAS threads would compete with UDF threads in numpy_matmul; keep it single-threaded for every build
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
        }
    )
    env.update(extra_env)
    try:
        proc = subprocess.run(
            [python, '-c', 'import benchmarks; benchmarks.main()', name],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 'timeout'
    finally:
        shutil.rmtree(home, ignore_errors=True)
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ['?'])[-1]
        return f'error: {tail[:120]}'
    return json.loads(proc.stdout.strip().splitlines()[-1])


def fmt(v: float | str | None) -> str:
    return f'{v:>10.3f}' if isinstance(v, (int, float)) else f'{(v or "-")[:10]:>10}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', help=f'benchmarks to run (default: all): {", ".join(benchmarks.BENCHMARKS)}')
    parser.add_argument('--json', default=str(BENCH_DIR / 'results' / 'latest.json'), help='where to write results')
    parser.add_argument('--compare', help='a results JSON file to compare against')
    parser.add_argument('--build', action='append', default=[], help='NAME=PYTHON: run every benchmark with this interpreter')
    parser.add_argument('--env', action='append', default=[], help='KEY=VALUE for the benchmark processes')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()

    names = args.names or list(benchmarks.BENCHMARKS)
    unknown = [n for n in names if n not in benchmarks.BENCHMARKS]
    if unknown:
        parser.error(f'unknown benchmarks: {", ".join(unknown)}')
    extra_env = dict(kv.split('=', 1) for kv in args.env)
    builds = dict(b.split('=', 1) for b in args.build) or {'this': sys.executable}
    base = json.loads(Path(args.compare).read_text())['results'] if args.compare else {}
    multi = len(builds) > 1
    last = list(builds)[-1]

    results: dict[str, dict[str, float | str]] = {b: {} for b in builds}
    descriptions: dict[str, str] = {}
    width = 18
    if multi:
        header = f'{"benchmark":<{width}}' + ''.join(f' {b[:10]:>10}' for b in builds)
        header += ''.join(f' {("vs " + b)[:12]:>12}' for b in builds if b != last)
    else:
        header = f'{"benchmark":<{width}} {"seconds":>10}' + (f' {"baseline":>10} {"speedup":>9}' if base else '')
    print(header)
    print('-' * len(header))
    for name in names:
        outcomes = {b: run(name, py, extra_env, args.timeout) for b, py in builds.items()}
        keys: list[str] = []
        for r in outcomes.values():
            for k in r if isinstance(r, dict) else []:
                if k not in keys:
                    keys.append(k)
        if not keys:
            keys = [name]
        for k in keys:
            descriptions[k] = benchmarks.describe(name)
            vals = {}
            for b, r in outcomes.items():
                vals[b] = r.get(k, 'missing') if isinstance(r, dict) else r
                results[b][k] = vals[b]
            line = f'{k:<{width}}' + ''.join(f' {fmt(vals[b])}' for b in builds)
            if multi:
                for b in builds:
                    if b == last:
                        continue
                    v, ref = vals[last], vals[b]
                    ok = isinstance(v, (int, float)) and isinstance(ref, (int, float)) and v
                    line += f' {(f"{ref / v:.1f}x" if ok else "-"):>12}'
            else:
                b = base.get(k)
                v = vals[last]
                if isinstance(b, (int, float)) and isinstance(v, (int, float)):
                    line += f' {b:>10.3f} {b / v:>8.1f}x'
            print(line, flush=True)

    out = {
        'results': results[last],
        'builds': results if multi else None,
        'descriptions': descriptions,
        'env': extra_env,
        'machine': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cpus': os.cpu_count(),
            'load': os.getloadavg(),
        },
        'commit': subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=BENCH_DIR, capture_output=True, text=True
        ).stdout.strip(),
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nresults: {path}')


if __name__ == '__main__':
    main()
