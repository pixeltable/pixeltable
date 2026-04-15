#!/usr/bin/env python3
"""
CI Tool for Pixeltable

Command-line utility for CI/CD operations.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Literal, NamedTuple, NoReturn

DEFAULT_PYTEST = "-m 'not expensive and not very_expensive and not benchmark'"
EXPENSIVE_PYTEST = "-m 'not very_expensive and not benchmark'"
VERY_EXPENSIVE_PYTEST = "-m 'not benchmark'"

MAIN_PLATFORM = 'ubuntu-24.04'
BASIC_PLATFORMS = ('macos-15', 'windows-2022')
EXPENSIVE_PLATFORMS = ('ubuntu-small-t4',)
ALTERNATIVE_PLATFORMS = ('ubuntu-24.04-arm', 'macos-15-intel')

COCKROACH_TEST_MODULES = ('table', 'index')


class MatrixConfig(NamedTuple):
    display_name_prefix: str
    test_category: Literal['py', 'ipynb', 'lint', 'random-ops']
    os: str
    python_version: str
    uv_options: str = ''
    pytest_options: str = DEFAULT_PYTEST
    pre_test_cmd: str = ''  # Extra bash command to be run just before tests

    @property
    def display_name(self) -> str:
        return f'{self.display_name_prefix}, {self.os}, {self.python_version}'

    @property
    def matrix_entry(self) -> dict[str, str]:
        return {
            'display-name': self.display_name,
            'test-category': self.test_category,
            'os': self.os,
            'python-version': self.python_version,
            'uv-options': self.uv_options,
            'pytest-options': self.pytest_options,
            'pre-test-cmd': self.pre_test_cmd,
        }


def new_bucket_addr() -> str:
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    bucket_uuid = uuid.uuid4().hex
    return f's3://pxt-test/pytest-media-dest/{date_str}/{bucket_uuid}'


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
        MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.10', uv_options='--no-dev'),  # Minimal test (no dev deps)
        MatrixConfig('notebooks', 'ipynb', MAIN_PLATFORM, '3.10'),  # Notebook tests
        MatrixConfig('static-checks', 'lint', MAIN_PLATFORM, '3.10'),  # Linting, type checking, etc.
        MatrixConfig('random-ops', 'random-ops', MAIN_PLATFORM, '3.10', uv_options='--no-dev'),  # Random operations
    ]

    # Standard configs that are always run
    configs.extend(MatrixConfig('standard', 'py', os, '3.10') for os in BASIC_PLATFORMS)

    # All other configs are dependent on the CI scenario. There are three basic scenarios:
    # 1. During a PR, we run a limited set of tests: MAIN_PLATFORM (Ubuntu) identically to the standard configs, and
    #    nothing additional.
    # 2. In merge queue or on a workflow dispatch, we include 'expensive' tests on MAIN_PLATFORM, and also run a suite
    #    of other jobs providing broader test coverage.
    # 3. On a scheduled run, or if "Run on all platforms" is checked during a workflow dispatch, then in addition to
    #    the above, we also run the 'very_expensive' tests on MAIN_PLATFORM and the basic tests on EXPENSIVE_PLATFORMS.
    # In 2 and 3, we also upgrade the Ubuntu runner to 'ubuntu-medium' (instead of free tier).

    if trigger == 'pull_request':
        # Just the standard tests on MAIN_PLATFORM.
        configs.append(MatrixConfig('standard', 'py', MAIN_PLATFORM, '3.10'))

    else:
        if force_all or trigger == 'schedule':
            # Standard + expensive + very_expensive tests on MAIN_PLATFORM; upgrade to 'ubuntu-medium'.
            configs.append(
                MatrixConfig('standard++', 'py', 'ubuntu-medium', '3.10', pytest_options=VERY_EXPENSIVE_PYTEST)
            )

            # Expensive platforms (e.g., GPU runners).
            configs.extend(MatrixConfig('standard', 'py', os, '3.10') for os in EXPENSIVE_PLATFORMS)

        else:
            # Standard + expensive (but not very_expensive) tests on MAIN_PLATFORM; upgrade to 'ubuntu-medium'.
            configs.append(MatrixConfig('standard+', 'py', 'ubuntu-medium', '3.10', pytest_options=EXPENSIVE_PYTEST))

        # The remaining configs run on all non-PR triggers.

        # Standard test suite on main & basic platforms on Python 3.14
        configs.extend(MatrixConfig('standard', 'py', os, '3.14') for os in (MAIN_PLATFORM, *BASIC_PLATFORMS))

        # Standard test suite on Ubuntu on intermediate Python versions
        configs.extend(MatrixConfig('standard', 'py', MAIN_PLATFORM, py) for py in ('3.11', '3.12', '3.13'))

        # Minimal tests on Python 3.14
        configs.append(MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.14', uv_options='--no-dev'))

        # Minimal tests on alternative platforms (we don't run the standard suite on these, since dev dependencies
        # can be hit-or-miss)
        configs.extend(MatrixConfig('minimal', 'py', os, '3.10', uv_options='--no-dev') for os in ALTERNATIVE_PLATFORMS)

        if os.environ.get('PXTTEST_COCKROACH_DB_CONNECT_STR'):
            configs.extend(
                MatrixConfig(
                    f'cockroach-{module}',
                    'py',
                    MAIN_PLATFORM,
                    '3.10',
                    pytest_options=f'--reruns 2 -m cockroachdb tests/test_{module}.py',
                    pre_test_cmd='export PIXELTABLE_DB_CONNECT_STR="$PXTTEST_COCKROACH_DB_CONNECT_STR"',
                )
                for module in COCKROACH_TEST_MODULES
            )

        # Minimal tests with S3 media destination. We use a unique bucket name that incorporates today's date, so that
        # different test runs don't interfere with each other and any stale data is easy to clean up.
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            configs.append(
                MatrixConfig(
                    's3-output-dest',
                    'py',
                    MAIN_PLATFORM,
                    '3.10',
                    uv_options='--no-dev --group storage-sdks',
                    pre_test_cmd=f'export PIXELTABLE_OUTPUT_MEDIA_DEST={new_bucket_addr()}',
                )
            )

    configs.sort(key=lambda cfg: cfg.display_name)

    matrix = {'include': [cfg.matrix_entry for cfg in configs]}

    print(json.dumps(matrix, indent=4))
    output = f'matrix={json.dumps(matrix)}\n'
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
