from __future__ import annotations

import importlib
import sys

DB_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    'create': ('create_db', 'create a cloud-hosted database'),
    'list': ('list_dbs', 'list cloud-hosted databases for an org'),
    'delete': ('delete_db', 'delete a cloud-hosted database'),
    'start': ('start_db', 'start (wake) a stopped cloud-hosted database'),
    'stop': ('stop_db', 'stop (sleep) a running cloud-hosted database'),
    'update-runtime': ('update_runtime', 'trigger a runtime rebuild for a cloud-hosted database'),
}


def _print_help() -> None:
    sys.stdout.write('usage: pxt db <subcommand> [args...]\n\nsubcommands:\n')
    width = max(len(k) for k in DB_SUBCOMMANDS)
    for sub, (_, help_text) in DB_SUBCOMMANDS.items():
        sys.stdout.write(f'  {sub.ljust(width)}  {help_text}\n')
    sys.stdout.write("\nUse 'pxt db <subcommand> --help' for subcommand options.\n")


def run(argv: list[str]) -> None:
    if not argv or argv[0] in ('-h', '--help'):
        _print_help()
        sys.exit(0)
    subcmd = argv[0]
    if subcmd not in DB_SUBCOMMANDS:
        print(f'pxt db: unknown subcommand: {subcmd!r}', file=sys.stderr)
        print("Use 'pxt db --help' for a list of subcommands.", file=sys.stderr)
        sys.exit(2)
    module_name, _ = DB_SUBCOMMANDS[subcmd]
    mod = importlib.import_module(f'pixeltable_cli.client.commands.{module_name}')
    mod.run(argv[1:])
