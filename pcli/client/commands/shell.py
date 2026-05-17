"""Interactive REPL.

Each line is parsed as a pcli command and dispatched in-process, amortizing
Python interpreter + module import cost across the session.
"""

import shlex
import sys
import traceback

from ..parser import Parser

EPILOG = """\
Inside the shell:
  <command> [args...]      run a pcli command (e.g. `ls -l`)
  help                     list available commands
  exit | quit | Ctrl-D     leave the shell"""

_PROMPT = 'pcli> '


def run(argv: list[str]) -> None:
    ap = Parser(prog='pcli shell', epilog=EPILOG)
    ap.parse_args(argv)

    # Lazy: only touch readline if it's available (POSIX); Windows users get a
    # no-history fallback without crashing.
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    from ..main import COMMANDS, dispatch  # local import to avoid circular at module load

    while True:
        try:
            line = input(_PROMPT)
        except EOFError:
            sys.stdout.write('\n')
            return
        except KeyboardInterrupt:
            sys.stdout.write('\n')
            continue

        line = line.strip()
        if not line:
            continue
        if line in ('exit', 'quit'):
            return
        if line in ('help', '?'):
            _print_help(COMMANDS)
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f'pcli: parse error: {e}', file=sys.stderr)
            continue
        cmd, args = parts[0], parts[1:]
        if cmd == 'shell':
            print('pcli: already in shell', file=sys.stderr)
            continue
        # Subcommands call sys.exit() on argparse errors / failed RPCs; intercept so
        # one bad command doesn't kill the session.
        try:
            dispatch(cmd, args)
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()


def _print_help(commands: dict[str, str]) -> None:
    width = max(len(c) for c in commands)
    for cmd, help_text in commands.items():
        if cmd == 'shell':
            continue
        print(f'  {cmd.ljust(width)}  {help_text}')
    print(f'  {"exit".ljust(width)}  leave the shell')
