#!/usr/bin/env python3
"""Run the copied upstream tests against pixeltable-rs and summarize the outcome.

Writes per-test outcomes (with a short failure reason and a coarse category) to compat/results/latest.json and prints
a table per file. pytest's own output goes to compat/results/pytest.log unless --stream is given.

Usage (extra pytest arguments go after --):
  python run.py                          all selected files, 4 xdist workers
  python run.py test_dirs test_table     a subset of files (test_ prefix and .py suffix optional)
  python run.py -k 'create and not view' pytest -k expression
  python run.py --update-passing         also rewrite PASSING.txt from this run
  python run.py --check                  exit 1 if any test in PASSING.txt no longer passes (regression gate)
  python run.py -- -x --lf               extra pytest arguments after --

Exit status: with --check, 1 if there are regressions, else 0. Otherwise pytest's status when pytest itself failed
(usage error, internal error, no tests collected), else 0.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pxt_compat.selection import (
    COMPAT_DIR,
    COMPAT_TESTS,
    PASSING_FILE,
    PUBLIC_API_FILE,
    RESULTS_DIR,
    SELECTED_TESTS,
    UPSTREAM_ROOT,
    VENV_DIR,
)

OUTCOMES = ('passed', 'failed', 'error', 'skipped', 'xfailed', 'xpassed')


def test_file_of(nodeid: str) -> str:
    return nodeid.split('::', 1)[0]


def function_id(nodeid: str) -> str:
    """The nodeid without its parametrization suffix."""
    return nodeid.split('[', 1)[0]


def normalize_file_arg(arg: str) -> str:
    name = Path(arg).name.removesuffix('.py')
    if not name.startswith('test_'):
        name = f'test_{name}'
    if name not in SELECTED_TESTS:
        raise SystemExit(f'error: {arg!r} is not a selected test file (see pxt_compat/selection.py)')
    return name


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    # never let the upstream checkout (its pixeltable/ and tests/ packages) onto sys.path; make the harness package
    # importable in every xdist worker
    banned = {UPSTREAM_ROOT.resolve(), (UPSTREAM_ROOT / 'tests').resolve(), COMPAT_DIR.resolve()}
    kept = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and Path(p).resolve() not in banned]
    env['PYTHONPATH'] = os.pathsep.join([str(COMPAT_DIR), *kept])
    env.pop('PYTHONSAFEPATH', None)
    return env


def run_pytest(args: argparse.Namespace, files: list[str], results_path: Path, log_path: Path) -> int:
    python = args.python or str(VENV_DIR / 'bin' / 'python')
    cmd = [
        python,
        '-m',
        'pytest',
        '-c',
        str(COMPAT_DIR / 'pytest.ini'),
        '--rootdir',
        str(COMPAT_DIR),
        '-p',
        'pxt_compat.report',
        '--compat-report',
        str(results_path),
        '-p',
        'no:cacheprovider',
        '-rN',
    ]
    if args.workers > 0:
        cmd += ['-n', str(args.workers), '--max-worker-restart', '100000', '-o', 'timeout_method=thread']
    else:
        cmd += ['-p', 'no:xdist']
    if args.timeout is not None:
        cmd += ['--timeout', str(args.timeout)]
    if args.k:
        cmd += ['-k', args.k]
    if args.m:
        cmd += ['-m', args.m]
    cmd += [f'tests/{name}.py' for name in files]
    cmd += args.pytest_args
    env = build_env()
    print(f'running: {" ".join(cmd)}', flush=True)
    if results_path.exists():
        results_path.unlink()
    if args.stream:
        proc = subprocess.run(cmd, cwd=COMPAT_DIR, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', encoding='utf-8') as log:
            proc = subprocess.run(cmd, cwd=COMPAT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        print(f'pytest output: {log_path}')
    return proc.returncode


def summarize(data: dict[str, Any], files: list[str], public: set[str]) -> None:
    tests: dict[str, dict[str, Any]] = data['tests']
    per_file: dict[str, collections.Counter[str]] = {f'tests/{name}.py': collections.Counter() for name in files}
    public_per_file: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for nodeid, entry in tests.items():
        f = test_file_of(nodeid)
        per_file.setdefault(f, collections.Counter())[entry['outcome']] += 1
        if function_id(nodeid) in public:
            public_per_file[f][entry['outcome']] += 1

    header = (
        f'{"file":34} {"total":>6} {"pass":>6} {"fail":>6} {"error":>6} {"skip":>6} {"xfail":>6}  '
        f'{"public pass/total":>17}'
    )
    print()
    print(header)
    print('-' * len(header))
    totals: collections.Counter[str] = collections.Counter()
    public_totals: collections.Counter[str] = collections.Counter()
    for f, counts in per_file.items():
        total = sum(counts.values())
        totals.update(counts)
        pc = public_per_file.get(f, collections.Counter())
        public_totals.update(pc)
        xfail = counts['xfailed'] + counts['xpassed']
        note = '  COLLECTION ERROR' if f in data.get('collect_errors', {}) else ''
        pub = f'{pc["passed"]}/{sum(pc.values())}'
        print(
            f'{f:34} {total:6} {counts["passed"]:6} {counts["failed"]:6} {counts["error"]:6} {counts["skipped"]:6} '
            f'{xfail:6}  {pub:>17}{note}'
        )
    print('-' * len(header))
    total = sum(totals.values())
    pub = f'{public_totals["passed"]}/{sum(public_totals.values())}'
    print(
        f'{"TOTAL":34} {total:6} {totals["passed"]:6} {totals["failed"]:6} {totals["error"]:6} {totals["skipped"]:6} '
        f'{totals["xfailed"] + totals["xpassed"]:6}  {pub:>17}'
    )

    categories = collections.Counter(e['category'] for e in tests.values() if e['category'] not in (None, 'skip'))
    if categories:
        print('\nnon-passing by cause: ' + ', '.join(f'{c}={n}' for c, n in categories.most_common()))
    reasons = collections.Counter(
        e['reason'][:140] for e in tests.values() if e['outcome'] in ('failed', 'error') and e['reason']
    )
    if reasons:
        print('\nmost common failure reasons:')
        for reason, n in reasons.most_common(12):
            print(f'  {n:5}  {reason}')
    if data.get('collect_errors'):
        print('\ncollection errors:')
        for nodeid, reason in data['collect_errors'].items():
            print(f'  {nodeid}: {reason}')
    stub_report = data.get('meta', {}).get('stubs', {})
    if stub_report.get('modules'):
        print('\nstubbed modules: ' + ', '.join(stub_report['modules']))
    for module, names in stub_report.get('attributes', {}).items():
        print(f'stubbed names in {module}: {", ".join(names)}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help='subset of selected test files (default: all)')
    parser.add_argument('-k', help='pytest -k expression')
    parser.add_argument('-m', help='pytest -m expression (replaces the default marker filter)')
    parser.add_argument('-n', '--workers', type=int, default=4, help='xdist workers (0 = no xdist; default 4)')
    parser.add_argument('--timeout', type=float, default=None, help='per-test timeout in seconds (default: pytest.ini)')
    parser.add_argument('--python', help='interpreter to run pytest with (default: pixeltable-rs/.venv/bin/python)')
    parser.add_argument('--results', type=Path, default=RESULTS_DIR / 'latest.json', help='where to write results')
    parser.add_argument('--stream', action='store_true', help="show pytest's output instead of logging it")
    parser.add_argument('--update-passing', action='store_true', help='rewrite PASSING.txt from this run')
    parser.add_argument('--check', action='store_true', help='exit 1 if a test listed in PASSING.txt no longer passes')
    parser.add_argument('--no-run', action='store_true', help='do not run pytest; summarize/check existing results')
    argv = sys.argv[1:]
    extra: list[str] = []
    if '--' in argv:
        extra = argv[argv.index('--') + 1 :]
        argv = argv[: argv.index('--')]
    args = parser.parse_args(argv)
    args.pytest_args = extra

    files = [normalize_file_arg(f) for f in args.files] or list(SELECTED_TESTS)
    missing = [f for f in files if not (COMPAT_TESTS / f'{f}.py').is_file()]
    if missing:
        print(f'error: not synced: {", ".join(missing)}; run sync_tests.py', file=sys.stderr)
        return 2
    results_path: Path = args.results.resolve()
    log_path = results_path.with_name(results_path.stem + '.pytest.log')

    returncode = 0
    if not args.no_run:
        returncode = run_pytest(args, files, results_path, log_path)
    if not results_path.is_file():
        print(f'error: pytest wrote no results (exit status {returncode}); see {log_path}', file=sys.stderr)
        if log_path.is_file():
            tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines(True)[-40:]
            print(''.join(tail), file=sys.stderr)
        return returncode or 2
    data = json.loads(results_path.read_text(encoding='utf-8'))
    public = set(read_lines(PUBLIC_API_FILE))
    summarize(data, files, public)
    print(f'\nresults: {results_path}')

    tests: dict[str, dict[str, Any]] = data['tests']
    passing_now = {nodeid for nodeid, e in tests.items() if e['outcome'] == 'passed'}
    run_files = {f'tests/{name}.py' for name in files}
    filtered = bool(args.k or args.m or args.pytest_args)
    previously_passing = set(read_lines(PASSING_FILE))

    exit_status = 0 if returncode in (0, 1) else returncode
    if args.check:
        in_scope = {
            nodeid
            for nodeid in previously_passing
            if test_file_of(nodeid) in run_files and (nodeid in tests or not filtered)
        }
        regressions = sorted(nodeid for nodeid in in_scope if nodeid not in passing_now)
        if regressions:
            print(f'\nREGRESSIONS: {len(regressions)} test(s) in PASSING.txt no longer pass:')
            for nodeid in regressions:
                entry = tests.get(nodeid)
                status = f'{entry["outcome"]}: {entry["reason"]}' if entry else 'not run / not collected'
                print(f'  {nodeid}  ({status})')
            exit_status = 1
        else:
            print(f'\ncheck: OK ({len(in_scope)} previously passing tests still pass)')
            exit_status = 0
        newly = sorted(passing_now - previously_passing)
        if newly:
            print(f'{len(newly)} test(s) pass that are not in PASSING.txt (use --update-passing)')

    if args.update_passing:
        ran = set(tests)
        kept = {
            nodeid
            for nodeid in previously_passing
            if not (test_file_of(nodeid) in run_files and (nodeid in ran or not filtered))
        }
        updated = sorted(kept | passing_now)
        PASSING_FILE.write_text(''.join(f'{n}\n' for n in updated), encoding='utf-8')
        added, removed = len(set(updated) - previously_passing), len(previously_passing - set(updated))
        print(f'PASSING.txt: {len(updated)} tests (+{added}, -{removed})')

    return exit_status


if __name__ == '__main__':
    sys.exit(main())
