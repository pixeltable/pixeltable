"""'pxt init': write the project configuration, making a directory a project root."""

import json
import pathlib
import sys

from pixeltable_cli.utils import _resolve_pixeltable_home, project_root_refusal

from ..parser import Parser
from ..project_config import find_project_config, write_project_config
from ..utils import EXIT_ERROR, EXIT_REFUSED

EPILOG = """\
Examples:
  pxt init                          # write the project configuration here, making this the project root
  pxt init --json                   # the same, machine-readable

What it writes:
  pixeltable.toml, holding one entry per database the project uses:

    [[pixeltable.database]]         # the local database
    vars.media_dest = 's3://bucket/prefix'

  vars supply the values of the config vars a schema declares. A hosted database is a second
  entry, identified by its uri, which also records what goes into its image ('pxt db update').
  'pxt db diff <uri> --add-config-entry' appends that entry.
  In a directory with a pyproject.toml, the same entry is appended there as
  [[tool.pixeltable.database]] rather than writing a second file.
  A directory that already has a project configuration is reported, and left as it is.

The project root:
  The directory holding the project configuration. Every local module path is relative to it:
  ~/proj/ad_gen/app.py is imported as 'ad_gen.app', and a udf defined in it is recorded as
  'ad_gen.app.<name>'. That recorded path is how a later process -- the daemon, a serving worker,
  a hosted runtime -- finds the udf again, so each directory name along the way has to be a Python
  identifier, and 'pxt schema' and 'pxt service' refuse a file that sits under no project root.

Exit codes:
  0  the directory is a project root
  1  error: the project configuration could not be written
  3  refused: a project root already sits above this directory, or the project root would be your home
     directory, a directory above it, or the Pixeltable home"""

# the shared set has no name for this one: init reports whether the directory is a project root
EXIT_OK = 0


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt init', epilog=EPILOG, usage_exit_code=EXIT_ERROR)
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    root = pathlib.Path.cwd()
    try:
        existing = find_project_config(root)
    except RuntimeError as e:
        print(f'pxt init: {e}', file=sys.stderr)
        sys.exit(EXIT_ERROR)
    project_root = (root if existing is None else existing.parent).resolve()
    refusal = project_root_refusal(project_root, pathlib.Path(_resolve_pixeltable_home()))
    if refusal is not None:
        if existing is None:
            print(
                f'pxt init: {root} cannot be a project root: {refusal}.\n'
                'Make a directory for the project, and run pxt init there.',
                file=sys.stderr,
            )
        else:
            print(
                f'pxt init: {existing} makes {existing.parent} a project root, but {refusal}.\n'
                f'Remove the project configuration from {existing}, then run pxt init in a directory made for '
                'the project.',
                file=sys.stderr,
            )
        sys.exit(EXIT_REFUSED)
    if existing is not None and existing.parent == root:
        _report(root, existing, created=False, as_json=args.as_json)
        return
    if existing is not None:
        print(
            f'pxt init: {existing.parent} already has a project configuration ({existing.name}), and '
            f'{root} sits under it.\nA project has one root, which every module path under it is relative '
            f'to: work under {existing.parent}, or remove {existing.name} to make this directory a root.',
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSED)

    try:
        config_file = write_project_config(root)
    except RuntimeError as e:
        print(f'pxt init: {e}', file=sys.stderr)
        sys.exit(EXIT_ERROR)
    _report(root, config_file, created=True, as_json=args.as_json)


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
            f"pxt init: '{name}' contains Python files, but its name is not a Python identifier, so nothing "
            'under it can be imported; rename the directory to use it in this project.',
            file=sys.stderr,
        )
