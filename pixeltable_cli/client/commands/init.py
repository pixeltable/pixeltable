"""'pxt init': write the project configuration, making a directory a project root."""

import json
import pathlib
import sys

from pixeltable_cli.utils import PROJECT_CONFIG_FILE, find_project_root

from ..parser import Parser

EPILOG = """\
Examples:
  pxt init                          # write the project configuration here, making this the project root
  pxt init --json                   # the same, machine-readable

What it writes:
  pixeltable.toml, holding one entry per database the project uses:

    [[pixeltable.database]]         # the local database
    vars.media_dest = 's3://bucket/prefix'
    secrets.openai_api_key = '...'

  vars and secrets bind the config vars a schema declares. A hosted database is a second entry,
  named by its uri, which also carries what goes into its runtime image ('pxt db update-runtime').
  In a directory that already holds a pyproject.toml, the same entry is appended there as
  [[tool.pixeltable.database]] rather than writing a second file.
  A directory that already holds a project configuration is reported, and left as it is.

The project root:
  The directory holding the project configuration. Every local module path is relative to it:
  ~/proj/ad_gen/app.py is imported as 'ad_gen.app', and a udf defined in it is recorded as
  'ad_gen.app.<name>'. That recorded path is how a later process -- the daemon, a serving worker,
  a hosted runtime -- finds the udf again, so each directory name along the way has to be a Python
  identifier, and 'pxt schema' and 'pxt service' refuse a file that sits under no project root.

Exit codes:
  0  the directory is a project root
  1  error: the project configuration could not be written
  3  refused: a project root already sits above this directory"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 3

_PYPROJECT = 'pyproject.toml'

# what a fresh project configuration holds: the local database, and the bindings that go on it
_DATABASE_ENTRY = """\
{header}
# The local database. vars and secrets bind the config vars a schema declares; a hosted database is
# a second entry, named by its uri.
# vars.media_dest = 's3://bucket/prefix'
# secrets.openai_api_key = '...'
"""
_PROJECT_CONFIG = f"""\
# Pixeltable project configuration. This file makes its directory the project root, which every
# local module path is relative to.

{_DATABASE_ENTRY.format(header='[[pixeltable.database]]')}"""
_PYPROJECT_CONFIG = f"""
{_DATABASE_ENTRY.format(header='[[tool.pixeltable.database]]')}"""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt init', epilog=EPILOG, usage_exit_code=EXIT_ERROR)
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    root = pathlib.Path.cwd()
    existing = _find_project_config(root)
    if existing is not None and existing.parent == root:
        _report(root, existing, created=False, as_json=args.as_json)
        return
    if existing is not None:
        print(
            f'pxt init: {existing.parent} already holds a project configuration ({existing.name}), and '
            f'{root} sits under it.\nA project has one root, which every module path under it is relative '
            f'to: work under {existing.parent}, or remove {existing.name} to make this directory a root.',
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSED)

    config_file = _write_project_config(root)
    _report(root, config_file, created=True, as_json=args.as_json)


def _find_project_config(start: pathlib.Path) -> pathlib.Path | None:
    """The configuration file of the nearest project root at or above start. Returns None if there is none."""
    root = find_project_root(start)
    if root is None:
        return None
    config_file = root / PROJECT_CONFIG_FILE
    return config_file if config_file.is_file() else root / _PYPROJECT


def _write_project_config(root: pathlib.Path) -> pathlib.Path:
    """Write the project configuration in root, and return the file holding it."""
    pyproject = root / _PYPROJECT
    try:
        if pyproject.is_file():
            with open(pyproject, 'a', encoding='utf-8') as fp:
                fp.write(_PYPROJECT_CONFIG)
            return pyproject
        config_file = root / PROJECT_CONFIG_FILE
        config_file.write_text(_PROJECT_CONFIG, encoding='utf-8')
        return config_file
    except OSError as e:
        print(f'pxt init: the project configuration in {root} could not be written: {e}', file=sys.stderr)
        sys.exit(EXIT_ERROR)


def _report(root: pathlib.Path, config_file: pathlib.Path, *, created: bool, as_json: bool) -> None:
    """Report the root, the file configuring it, and any directory whose name a module path cannot hold."""
    unusable = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.isidentifier() and any(d.glob('*.py'))
    )
    if as_json:
        print(
            json.dumps(
                {
                    'project_root': str(root),
                    'config_file': str(config_file),
                    'created': created,
                    'unusable_dirs': unusable,
                },
                indent=2,
            )
        )
        return
    if created:
        print(f'project root: {root}\nwrote {config_file.name}')
    else:
        print(f'project root: {root}\nalready configured by {config_file.name}')
    for name in unusable:
        print(
            f"pxt init: '{name}' holds Python files, but its name is not a Python identifier, so nothing "
            'under it can be imported; rename the directory to use it in this project.',
            file=sys.stderr,
        )
