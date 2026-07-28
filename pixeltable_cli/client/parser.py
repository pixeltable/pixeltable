import argparse
import sys
from typing import NoReturn


class Parser(argparse.ArgumentParser):
    """ArgumentParser that appends the epilog (examples) to stderr on error."""

    # status to exit with on a usage error; overridable where 2 already carries another meaning
    usage_exit_code: int

    def __init__(self, *args: object, usage_exit_code: int = 2, **kwargs: object) -> None:
        kwargs.setdefault('formatter_class', argparse.RawDescriptionHelpFormatter)
        self.usage_exit_code = usage_exit_code
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f'\n{self.prog}: error: {message}\n')
        if self.epilog is not None:
            sys.stderr.write(f'\n{self.epilog}\n')
        sys.exit(self.usage_exit_code)


def parse_cols(arg: str | None, parser: Parser) -> list[str] | None:
    """Parse a comma-separated --cols value, rejecting empty tokens so 'a,' or 'a,,b'
    don't sneak a '' through to the server and produce a misleading 'unknown columns: '."""
    if arg is None:
        return None
    parts = [c.strip() for c in arg.split(',')]
    if any(p == '' for p in parts):
        parser.error(f'--cols token must not be empty; got {arg!r}')
    return parts
