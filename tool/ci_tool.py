#!/usr/bin/env python3
"""
CI Tool for Pixeltable

Command-line utility for CI/CD operations.
"""

import argparse
import json
import sys
from typing import Literal, NamedTuple, NoReturn


class MatrixConfig(NamedTuple):
    display_name_prefix: str
    test_category: Literal['py', 'ipynb', 'lint', 'random-ops']
    os: str
    python_version: str
    uv_options: str = ''
    extra_env: str = ''

    @property
    def display_name(self) -> str:
        return f'{self.display_name_prefix}, {self.os}, {self.python_version}'


BASIC_PLATFORMS = ('ubuntu-24.04', 'macos-15', 'windows-2022')
EXPENSIVE_PLATFORMS = ('ubuntu-x64-t4',)
ALTERNATIVE_PLATFORMS = ('ubuntu-24.04-arm', 'macos-15-intel')


def generate_matrix(args: argparse.Namespace) -> None:
    """Generate test matrix configuration."""
    output_file = args.output_file
    trigger = args.trigger
    force_all = args.force_all
    print('Generating test matrix configuration.')
    print('Output file : ', output_file)
    print('Triggered on: ', trigger)
    print('Force all   : ', force_all)
    print()

    # Special configs that are always run
    configs = [
        MatrixConfig('minimal', 'py', 'ubuntu-24.04', '3.10', uv_options='--no-dev'),  # Minimal test (no dev deps)
        MatrixConfig('notebooks', 'ipynb', 'ubuntu-24.04', '3.10'),  # Notebook tests
        MatrixConfig('static-checks', 'lint', 'ubuntu-24.04', '3.10'),  # Linting, type checking, etc.
        MatrixConfig('random-ops', 'random-ops', 'ubuntu-24.04', '3.10'),  # Random operations tests
    ]

    # Full test suite on basic platforms on Python 3.10
    configs.extend(MatrixConfig('full', 'py', os, '3.10') for os in BASIC_PLATFORMS)

    if force_all or trigger != 'pull_request':
        # Full test suite on basic platforms on Python 3.13
        configs.extend(MatrixConfig('full', 'py', os, '3.13') for os in BASIC_PLATFORMS)

        # Full test suite on Ubuntu on intermediate Python versions
        configs.extend(MatrixConfig('full', 'py', 'ubuntu-24.04', py) for py in ('3.11', '3.12'))

        # Minimal test on Python 3.13
        configs.append(MatrixConfig('minimal', 'py', 'ubuntu-24.04', '3.13', uv_options='--no-dev'))

        # Minimal tests on alternative platforms (we don't run the full suite on these, since dev dependencies
        # can be hit-or-miss)
        configs.extend(MatrixConfig('minimal', 'py', os, '3.10', uv_options='--no-dev') for os in ALTERNATIVE_PLATFORMS)

    if force_all or trigger == 'schedule':
        # Expensive tests on special hardware on Python 3.10
        configs.extend(MatrixConfig('full', 'py', os, '3.10') for os in EXPENSIVE_PLATFORMS)

    configs.sort(key=lambda cfg: cfg.display_name)

    matrix = {'include': [cfg._asdict() | {'display_name': cfg.display_name} for cfg in configs]}

    print(json.dumps(matrix, indent=4).replace('_', '-'))
    output = f'matrix={json.dumps(matrix).replace("_", "-")}\n'
    with open(output_file, 'a', encoding='utf8') as fp:
        fp.write(output)


def main() -> NoReturn:
    """Main entry point for the CI tool."""
    parser = argparse.ArgumentParser(
        description='CI/CD utilities for Pixeltable', formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # generate-matrix subcommand
    matrix_parser = subparsers.add_parser('generate-matrix', help='Generate test matrix configuration')
    matrix_parser.add_argument('output_file', help='Output file for the test matrix')
    matrix_parser.add_argument('trigger', help='CI trigger type')
    matrix_parser.add_argument('--force-all', action='store_true', help='Force generation of all configurations')
    matrix_parser.set_defaults(func=generate_matrix)

    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)

    sys.exit(0)


if __name__ == '__main__':
    main()
