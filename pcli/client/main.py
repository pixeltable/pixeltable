import importlib
import sys
from collections.abc import Callable

# (module_name, short help) — single source of truth for both the top-level help
# message and the shell-mode dispatcher. Modules under pcli.client.commands.* must
# expose `run(argv: list[str]) -> None`.
COMMANDS: dict[str, tuple[str, str]] = {
    'health': ('health', 'show daemon info'),
    'ls': ('ls', 'list catalog entries'),
    'describe': ('describe', "show a table's schema and metadata"),
    'errors': ('errors', 'list rows where a computed column failed'),
    'history': ('history', "show a table's version timeline"),
    'columns': ('columns', 'list columns across tables (optionally one)'),
    'idxs': ('idxs', 'list indexes across tables (optionally one)'),
    'rows': ('rows', 'peek the first N rows of a table'),
    'get': ('get', 'look up a single row by primary key'),
    'count': ('count', 'count rows in a table'),
    'status': ('status', 'show daemon/runtime state'),
    'env': ('env', 'show pixeltable env vars and active config file'),
    'computed': ('computed', "list computed columns (alias for 'columns --computed')"),
    'drop': ('drop', "drop a table or view (use 'rm' for directories)"),
    'rm': ('rm', "remove a directory (use 'drop' for tables/views)"),
    'rename': ('rename', 'rename a table/view/dir in place'),
    'mv': ('mv', 'move a table/view/dir to a different directory'),
    'revert': ('revert', 'undo the last op(s) on a table'),
    'shell': ('shell', 'interactive REPL (avoids per-command Python startup)'),
}


def _resolve(cmd: str) -> Callable[[list[str]], None]:
    module_name, _ = COMMANDS[cmd]
    mod = importlib.import_module(f'pcli.client.commands.{module_name}')
    return mod.run


def _print_help() -> None:
    sys.stdout.write('usage: pcli <command> [args...]\n\ncommands:\n')
    width = max(len(c) for c in COMMANDS)
    for cmd, (_, help_text) in COMMANDS.items():
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
