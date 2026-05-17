import argparse
import sys
from typing import NoReturn


class Parser(argparse.ArgumentParser):
    """ArgumentParser that appends the epilog (examples) to stderr on error."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault('formatter_class', argparse.RawDescriptionHelpFormatter)
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f'\n{self.prog}: error: {message}\n')
        if self.epilog is not None:
            sys.stderr.write(f'\n{self.epilog}\n')
        sys.exit(2)
