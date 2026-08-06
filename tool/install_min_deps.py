#!/usr/bin/env python3
"""
Install the minimum supported version of each of Pixeltable's dependencies.

Reads `[project].dependencies` from pyproject.toml, rewrites each lower bound into an exact `==`
pin, and hands the resulting requirements to pip in a single invocation so that pip's resolver sees
them all at once. A pin that conflicts with a transitive constraint therefore surfaces as a
resolution error rather than being silently upgraded away from the minimum.

The `[project.optional-dependencies]` extras are opt-in via `--extras`; the `otel` extra requires
`opentelemetry-instrumentation-pixeltable`, which lives in `packages/` and is not published to PyPI
yet, so pip cannot install it by name. The `[dependency-groups]` (dev, test, ...) are never
installed: they are development tooling, not part of what a user's environment has to satisfy.

Usage:
    python tool/install_min_deps.py                       # required deps only
    python tool/install_min_deps.py --extras all          # required deps + every extra
    python tool/install_min_deps.py --extras serve        # required deps + selected extras
    python tool/install_min_deps.py -n                    # print the pinned requirements only
    python tool/install_min_deps.py -o reqs.txt -n        # write a requirements file
    python tool/install_min_deps.py -- --no-cache-dir     # pass extra args through to pip
"""

import argparse
import importlib
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from packaging.requirements import Requirement

DEFAULT_PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'


def bootstrap(module: str) -> Any:
    """Import `module`, pip-installing it first if it isn't present.

    This script is meant to run in an environment that has nothing but pip installed, so it cannot
    assume its own helpers are available.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    print(f'Installing {module} (needed to parse dependency metadata) ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', module], check=True)
    return importlib.import_module(module)


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file, using whichever parser this interpreter has available."""
    parser: Any
    for module in ('tomllib', 'tomli'):  # tomllib is stdlib as of 3.11
        try:
            parser = importlib.import_module(module)
        except ImportError:
            continue
        with path.open('rb') as fp:
            return parser.load(fp)
    parser = bootstrap('toml')
    return parser.load(path)


def min_version_req(req: 'Requirement') -> tuple['Requirement', Optional[str]]:
    """
    Rewrite a requirement so that it pins the minimum version its specifier permits.

    Returns the (possibly rewritten) requirement, along with a warning message if the requirement
    could not be pinned and was therefore left as-is.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    if any(spec.operator in ('==', '===') for spec in req.specifier):
        return req, None  # already an exact pin

    # `>=x` and `~=x` both admit x itself as their smallest version; `>x` does not name any
    # installable version, so it cannot be turned into a pin.
    lower_bounds = [Version(spec.version) for spec in req.specifier if spec.operator in ('>=', '~=')]
    if len(lower_bounds) == 0:
        reason = 'no lower version bound' if len(req.specifier) == 0 else f'unpinnable specifier {req.specifier}'
        return req, f'{req.name}: {reason}; installing whatever pip resolves'

    # Among several lower bounds, the largest is the binding one.
    minimum = max(lower_bounds)
    if not req.specifier.contains(minimum, prereleases=True):
        # Eg. `>=1.0,!=1.0`: the nominal minimum is excluded by another clause of the specifier.
        return req, f'{req.name}: {minimum} is excluded by {req.specifier}; installing whatever pip resolves'

    extras = f'[{",".join(sorted(req.extras))}]' if len(req.extras) > 0 else ''
    marker = f'; {req.marker}' if req.marker is not None else ''
    return Requirement(f'{req.name}{extras}=={minimum}{marker}'), None


def select_dependencies(project: dict[str, Any], extras_arg: str) -> tuple[list[str], list[str]]:
    """
    Gather the dependency strings to pin, per the `--extras` selection.

    Returns the dependency strings along with the names of the extras that contributed to them.
    """
    optional: dict[str, list[str]] = project.get('optional-dependencies', {})
    if extras_arg == 'all':
        selected = sorted(optional)
    elif extras_arg == 'none':
        selected = []
    else:
        selected = [name.strip() for name in extras_arg.split(',') if len(name.strip()) > 0]
        unknown = [name for name in selected if name not in optional]
        if len(unknown) > 0:
            raise SystemExit(f'Unknown extra(s): {", ".join(unknown)}. Available: {", ".join(sorted(optional))}')

    deps: list[str] = list(project['dependencies'])
    for name in selected:
        deps.extend(optional[name])
    return deps, selected


def merge_requirements(reqs: list['Requirement']) -> tuple[list['Requirement'], list[str]]:
    """
    Combine requirements that name the same package, so that each package is pinned exactly once.

    A package listed in both `dependencies` and an extra (or in two extras) would otherwise reach
    pip as two separate `==` pins, which pip rejects if they disagree. Intersecting the specifiers
    first means the pin we compute is the smallest version that satisfies *every* listing of that
    package. Requirements are only merged when their environment markers match, so conditional
    listings (eg. one per Python version) stay independent.

    Returns the merged requirements along with a note for each package that was listed more than
    once.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    merged: dict[tuple[str, str], Requirement] = {}
    notes: list[str] = []
    for req in reqs:
        key = (canonicalize_name(req.name), str(req.marker) if req.marker is not None else '')
        prev = merged.get(key)
        if prev is None:
            merged[key] = req
            continue
        specifier = prev.specifier & req.specifier
        extras = f'[{",".join(sorted(prev.extras | req.extras))}]' if len(prev.extras | req.extras) > 0 else ''
        marker = f'; {req.marker}' if req.marker is not None else ''
        merged[key] = Requirement(f'{prev.name}{extras}{specifier}{marker}')
        if str(prev) != str(req):  # an exact repeat of the same listing needs no explanation
            notes.append(f'{prev.name}: listed more than once ({prev} + {req}); using {merged[key]}')
    return list(merged.values()), notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pyproject', type=Path, default=DEFAULT_PYPROJECT, help='path to pyproject.toml')
    parser.add_argument('-n', '--dry-run', action='store_true', help='print the pip command without running it')
    parser.add_argument('-o', '--output', type=Path, help='also write the pinned requirements to this file')
    parser.add_argument('--python', default=sys.executable, help='interpreter to install into (default: this one)')
    parser.add_argument(
        '--extras',
        default='none',
        metavar='all|none|NAME,...',
        help='which optional-dependency extras to include (default: none). Dependency groups are never included.',
    )
    parser.add_argument('pip_args', nargs='*', help='additional arguments for `pip install` (pass after `--`)')
    args = parser.parse_args()

    bootstrap('packaging')
    from packaging.requirements import Requirement

    dependencies, extras = select_dependencies(load_toml(args.pyproject)['project'], args.extras)
    requirements, notes = merge_requirements([Requirement(dep) for dep in dependencies])

    pinned: list[str] = []
    warnings: list[str] = notes
    for req in requirements:
        pinned_req, warning = min_version_req(req)
        pinned.append(str(pinned_req))
        if warning is not None:
            warnings.append(warning)

    included = f'required + extras: {", ".join(extras)}' if len(extras) > 0 else 'required only'
    print(f'{len(pinned)} dependencies from {args.pyproject} ({included}):')
    for requirement in pinned:
        print(f'  {requirement}')
    for warning in warnings:
        print(f'WARNING: {warning}', file=sys.stderr)

    if args.output is not None:
        args.output.write_text('\n'.join(pinned) + '\n')
        print(f'Wrote {args.output}')

    cmd = [args.python, '-m', 'pip', 'install', *args.pip_args, *pinned]
    if args.dry_run:
        print('\nWould run:')
        print(shlex.join(cmd))  # quoted: environment markers contain both spaces and semicolons
        return 0
    print()
    sys.stdout.flush()  # keep our output ahead of pip's, which writes to the same stream
    return subprocess.run(cmd, check=False).returncode  # let the caller see pip's exit code


if __name__ == '__main__':
    sys.exit(main())
