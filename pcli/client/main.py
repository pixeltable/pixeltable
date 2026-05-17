import importlib
import sys
from collections.abc import Callable

# Single source of truth for both the top-level help message and shell-mode dispatch.
# Each key names a module under pcli.client.commands.* exposing run(argv: list[str]) -> None.
COMMANDS: dict[str, str] = {
    'health': 'show daemon info',
    'ls': 'list catalog entries',
    'describe': "show a table's schema and metadata",
    'errors': 'list rows where a computed column failed',
    'history': "show a table's version timeline",
    'columns': 'list columns across tables (optionally one)',
    'idxs': 'list indexes across tables (optionally one)',
    'rows': 'peek the first N rows of a table',
    'get': 'look up a single row by primary key',
    'count': 'count rows in a table',
    'status': 'show daemon/runtime state',
    'env': 'show pixeltable env vars and active config file',
    'computed': "list computed columns (alias for 'columns --computed')",
    'drop': "drop a table or view (use 'rm' for directories)",
    'rm': "remove a directory (use 'drop' for tables/views)",
    'rename': 'rename a table/view/dir in place',
    'mv': 'move a table/view/dir to a different directory',
    'revert': 'undo the last op(s) on a table',
    'shell': 'interactive REPL (avoids per-command Python startup)',
}


def _resolve(cmd: str) -> Callable[[list[str]], None]:
    mod = importlib.import_module(f'pcli.client.commands.{cmd}')
    return mod.run


def _print_help() -> None:
    sys.stdout.write('usage: pcli <command> [args...]\n\ncommands:\n')
    width = max(len(c) for c in COMMANDS)
    for cmd, help_text in COMMANDS.items():
        sys.stdout.write(f'  {cmd.ljust(width)}  {help_text}\n')
    sys.stdout.write("\nUse 'pcli <command> --help' for subcommand options.\n")


def dispatch(cmd: str, argv: list[str]) -> None:
    if cmd not in COMMANDS:
        print(f'pcli: unknown command: {cmd}', file=sys.stderr)
        sys.exit(2)
    _resolve(cmd)(argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        _print_help()
        sys.exit(0 if len(sys.argv) >= 2 else 2)
    dispatch(sys.argv[1], sys.argv[2:])


if __name__ == '__main__':
    main()
