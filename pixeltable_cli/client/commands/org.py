from __future__ import annotations

import importlib
import sys

ORG_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    'list': ('list_orgs', 'list organizations accessible to the current API key')
}


def _print_help() -> None:
    sys.stdout.write('usage: pxt org <subcommand> [args...]\n\nsubcommands:\n')
    width = max(len(k) for k in ORG_SUBCOMMANDS)
    for sub, (_, help_text) in ORG_SUBCOMMANDS.items():
        sys.stdout.write(f'  {sub.ljust(width)}  {help_text}\n')
    sys.stdout.write("\nUse 'pxt org <subcommand> --help' for subcommand options.\n")


def run(argv: list[str]) -> None:
    if not argv or argv[0] in ('-h', '--help'):
        _print_help()
        sys.exit(0)
    subcmd = argv[0]
    if subcmd not in ORG_SUBCOMMANDS:
        print(f'pxt org: unknown subcommand: {subcmd!r}', file=sys.stderr)
        print("Use 'pxt org --help' for a list of subcommands.", file=sys.stderr)
        sys.exit(2)
    module_name, _ = ORG_SUBCOMMANDS[subcmd]
    mod = importlib.import_module(f'pixeltable_cli.client.commands.{module_name}')
    mod.run(argv[1:])
