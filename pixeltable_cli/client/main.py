import difflib
import importlib
import importlib.metadata
import re
import sys
from typing import Callable

# Single source of truth for both the top-level help message and shell-mode dispatch.
# Each key names a module under pixeltable_cli.client.commands.* exposing run(argv: list[str]) -> None.
COMMANDS: dict[str, str] = {
    'cd': 'set or clear the working directory prepended to relative paths',
    'columns': 'list columns across tables (optionally one)',
    'computed': "list computed columns (alias for 'columns --computed')",
    'config': 'show resolved configuration: every documented setting, its value, and its source',
    'count': 'count rows in a table',
    'daemon': 'control the daemon (start/stop/restart/status)',
    'dashboard': 'print and open the dashboard URL',
    'db': 'manage hosted databases (diff/update/list/status/logs/start/stop/restart/retry/build-image/delete)',
    'describe': "show a table's schema and metadata",
    'drop': "drop a table or view (use 'drop-dir' for directories)",
    'drop-dir': "remove a directory (use 'drop' for tables/views)",
    'errors': 'list rows where a computed column failed',
    'feedback': 'send a report to the Pixeltable team, after showing it to you (preview by default)',
    'functions': 'list functions, aggregates and iterators with their signatures',
    'get': 'look up a single table row by primary key',
    'health': 'show daemon info',
    'history': "show a table's version timeline",
    'idxs': 'list indexes across tables (optionally one)',
    'init': 'write the project configuration here, making this the project root',
    'key': 'manage API keys and scoped runtime keys (list/create/update/delete)',
    'localproxy': 'manage local proxy daemons (create/start/stop/delete)',
    'login': 'sign in to Pixeltable Cloud in a browser (no API key needed)',
    'logout': 'forget the cached session or trial',
    'ls': 'list catalog entries',
    'mv': 'move a table/view/dir to a different directory',
    'new': 'get a free Pixeltable Cloud database, no account needed',
    'org': 'manage organizations (create/list/status)',
    'pwd': 'print the working directory',
    'recompute': 'recompute one or more computed columns of a table',
    'rename': 'rename a table/view/dir in place',
    'revert': 'undo the last op(s) on a table',
    'rows': 'peek the first N rows of a table',
    'schema': 'reconcile a directory with a class-based schema file (diff/update/prune/check/example)',
    'secret': "manage an org's or a database's secrets (list/set/delete)",
    'service': (
        'run the services defined in an application file '
        '(diff/update/run/prune/stop/restart/retry/list/logs/check/example)'
    ),
    'shell': 'interactive REPL (avoids per-command Python startup)',
    'skills': 'install the Pixeltable skill for coding agents in this directory (install)',
    'status': 'show daemon/runtime state',
    'usage': "show this month's usage of your organization against its plan",
    'whoami': 'show who this machine is signed in as',
}


def _resolve(cmd: str) -> Callable[[list[str]], None]:
    mod = importlib.import_module(f'pixeltable_cli.client.commands.{cmd.replace("-", "_")}')
    return mod.run


def _print_help() -> None:
    sys.stdout.write('usage: pxt <command> [args...]\n\ncommands:\n')
    width = max(len(c) for c in COMMANDS)
    for cmd, help_text in COMMANDS.items():
        sys.stdout.write(f'  {cmd.ljust(width)}  {help_text}\n')
    sys.stdout.write("\nUse 'pxt <command> --help' for subcommand options.\n")


def _suggestions(cmd: str) -> list[str]:
    """Commands an unknown command may have meant: each command with a subcommand of that name, or a close name."""
    # a help line lists its command's subcommands in parentheses, as in 'manage hosted databases (diff/update/...)'
    verb = re.compile(rf'[(/]{re.escape(cmd)}[/)]')
    with_verb = [f'{name} {cmd}' for name, help_text in COMMANDS.items() if verb.search(help_text)]
    return with_verb or difflib.get_close_matches(cmd, COMMANDS, n=3)


def dispatch(cmd: str, argv: list[str]) -> None:
    if cmd not in COMMANDS:
        print(f'pxt: unknown command: {cmd}', file=sys.stderr)
        for suggestion in _suggestions(cmd):
            print(f'did you mean: pxt {suggestion}', file=sys.stderr)
        sys.exit(2)
    from pixeltable_cli.client import utils

    utils.invocation = [cmd, *argv]
    _resolve(cmd)(argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        _print_help()
        sys.exit(0)
    if sys.argv[1] == '--version':
        # importlib.metadata is stdlib; avoids importing pixeltable just to read its version
        print(f'pxt {importlib.metadata.version("pixeltable")}')
        sys.exit(0)
    dispatch(sys.argv[1], sys.argv[2:])


if __name__ == '__main__':
    main()
