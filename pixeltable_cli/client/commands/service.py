from __future__ import annotations

import importlib
import sys

SERVICE_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    'create': ('create_service', 'create a service from a table in a cloud-hosted database'),
    'list': ('list_services', 'list services in a cloud-hosted database'),
    'start': ('start_service', 'start a service'),
    'stop': ('stop_service', 'stop a service'),
    'delete': ('delete_service', 'delete a service'),
}


def _print_help() -> None:
    sys.stdout.write('usage: pxt service <subcommand> [args...]\n\nsubcommands:\n')
    width = max(len(k) for k in SERVICE_SUBCOMMANDS)
    for sub, (_, help_text) in SERVICE_SUBCOMMANDS.items():
        sys.stdout.write(f'  {sub.ljust(width)}  {help_text}\n')
    sys.stdout.write("\nUse 'pxt service <subcommand> --help' for subcommand options.\n")


def run(argv: list[str]) -> None:
    if not argv or argv[0] in ('-h', '--help'):
        _print_help()
        sys.exit(0)
    subcmd = argv[0]
    if subcmd not in SERVICE_SUBCOMMANDS:
        print(f'pxt service: unknown subcommand: {subcmd!r}', file=sys.stderr)
        print("Use 'pxt service --help' for a list of subcommands.", file=sys.stderr)
        sys.exit(2)
    module_name, _ = SERVICE_SUBCOMMANDS[subcmd]
    mod = importlib.import_module(f'pixeltable_cli.client.commands.{module_name}')
    mod.run(argv[1:])
