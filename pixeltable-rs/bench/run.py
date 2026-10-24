"""Runs the pixeltable-rs benchmarks: one fresh process (and $PIXELTABLE_HOME) per benchmark.

    .venv/bin/python bench/run.py                       # all benchmarks
    .venv/bin/python bench/run.py sync_sleep_insert ... # a selection
    .venv/bin/python bench/run.py --json out.json --compare bench/results/baseline.json

Prints a table (seconds; with --compare also the baseline and the speedup) and writes the results as JSON.
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


def run(name: str, extra_env: dict[str, str], timeout: float) -> dict[str, float] | str:
    home = tempfile.mkdtemp(prefix='pxtrs-bench-')
    env = dict(os.environ)
    env.update(
        {
            'PIXELTABLE_HOME': home,
            'PYTHONPATH': str(BENCH_DIR),
            # BLAS threads would compete with UDF threads in numpy_matmul; keep it single-threaded for both builds
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
        }
    )
    env.update(extra_env)
    try:
        proc = subprocess.run(
            [sys.executable, '-c', 'import benchmarks; benchmarks.main()', name],
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
        return f'error: {tail}'
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', help=f'benchmarks to run (default: all): {", ".join(benchmarks.BENCHMARKS)}')
    parser.add_argument('--json', default=str(BENCH_DIR / 'results' / 'latest.json'), help='where to write results')
    parser.add_argument('--compare', help='a results JSON file to compare against')
    parser.add_argument('--env', action='append', default=[], help='KEY=VALUE for the benchmark processes')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()

    names = args.names or list(benchmarks.BENCHMARKS)
    unknown = [n for n in names if n not in benchmarks.BENCHMARKS]
    if unknown:
        parser.error(f'unknown benchmarks: {", ".join(unknown)}')
    extra_env = dict(kv.split('=', 1) for kv in args.env)
    base = json.loads(Path(args.compare).read_text())['results'] if args.compare else {}

    results: dict[str, float | str] = {}
    descriptions: dict[str, str] = {}
    width = 22
    header = f'{"benchmark":<{width}} {"seconds":>10}'
    if base:
        header += f' {"baseline":>10} {"speedup":>9}'
    print(header)
    print('-' * len(header))
    for name in names:
        r = run(name, extra_env, args.timeout)
        if isinstance(r, str):
            results[name] = r
            print(f'{name:<{width}} {r}')
            continue
        for k, v in r.items():
            results[k] = v
            descriptions[k] = benchmarks.describe(name)
            line = f'{k:<{width}} {v:>10.3f}'
            b = base.get(k)
            if isinstance(b, (int, float)):
                line += f' {b:>10.3f} {b / v:>8.1f}x'
            print(line, flush=True)

    out = {
        'results': results,
        'descriptions': descriptions,
        'env': extra_env,
        'machine': {'platform': platform.platform(), 'python': platform.python_version(), 'cpus': os.cpu_count()},
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
