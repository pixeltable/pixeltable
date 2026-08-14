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

# cloud_e2e is excluded from every tier: it provisions hosted databases and services against a live
# Pixeltable cloud deployment, and is run on demand rather than from the matrix
DEFAULT_PYTEST = "-m 'not expensive and not very_expensive and not benchmark and not cloud_e2e'"
EXPENSIVE_PYTEST = "-m 'not very_expensive and not benchmark and not cloud_e2e'"
VERY_EXPENSIVE_PYTEST = "-m 'not benchmark and not cloud_e2e'"

# Note: in addition to these pytest filters, the tests that actually run are implicitly filtered by
# skip_test_if_not_installed() and the install configuration.

# The core-functionality test modules that PR checks run on every push. This mirrors the `slimpytest` target in the
# Makefile.
SLIM_TESTS = ' '.join(
    f'tests/test_{name}.py'
    for name in (
        'alter_column',
        'catalog',
        'dirs',
        'env',
        'exprs',
        'function',
        'index',
        'operational_table',
        'snapshot',
        'table',
        'table_model',
        'types',
        'view',
    )
)

MAIN_PLATFORM = 'ubuntu-24.04'
BASIC_PLATFORMS = ('macos-15', 'windows-2025')
EXPENSIVE_PLATFORMS = ('ubuntu-small-t4',)
ALTERNATIVE_PLATFORMS = ('ubuntu-24.04-arm', 'macos-15-intel')


class MatrixConfig(NamedTuple):
    display_name_prefix: str
    test_category: Literal['py', 'ipynb', 'lint', 'random-ops', 'otel']
    os: str
    python_version: str
    uv_options: str = ''
    pytest_options: str = DEFAULT_PYTEST
    pre_test_cmd: str = ''  # Extra bash command to be run just before tests
    build_dashboard: bool = True  # Whether this config runs tests that need the dashboard SPA bundle

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
            'build-dashboard': str(self.build_dashboard).lower(),
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

    # Run on every trigger: static checks, plus the tests under a no-dev-dependencies install, once with the default
    # resolution and once with required deps pinned to their minimum versions.
    configs = [
        MatrixConfig('static-checks', 'lint', MAIN_PLATFORM, '3.11'),
        MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.11', uv_options='--no-dev'),
        MatrixConfig('minimum-deps', 'py', MAIN_PLATFORM, '3.11', uv_options='--no-dev --resolution lowest-direct'),
    ]

    if trigger == 'pull_request':
        # On every push to a PR we run only the slim tests. It is strictly a subset of what the merge queue runs.
        configs.extend(
            MatrixConfig(
                'slim', 'py', platform, '3.11', pytest_options=f'{DEFAULT_PYTEST} {SLIM_TESTS}', build_dashboard=False
            )
            for platform in (MAIN_PLATFORM, *BASIC_PLATFORMS)
        )

    else:
        # A non-PR trigger: merge queue, workflow dispatch, or schedule.

        configs.extend(MatrixConfig('standard', 'py', os, '3.11') for os in BASIC_PLATFORMS)

        configs.append(MatrixConfig('random-ops', 'random-ops', MAIN_PLATFORM, '3.11', uv_options='--no-dev'))
        configs.append(MatrixConfig('otel', 'otel', MAIN_PLATFORM, '3.11', uv_options='--no-dev --extra otel'))

        # force_all is set by the "Run on all platforms" checkbox on a workflow dispatch.
        if force_all or trigger == 'schedule':
            configs.append(
                MatrixConfig('standard++', 'py', 'ubuntu-large', '3.11', pytest_options=VERY_EXPENSIVE_PYTEST)
            )
            configs.append(MatrixConfig('notebooks++', 'ipynb', 'ubuntu-large', '3.11'))

            configs.extend(MatrixConfig('standard', 'py', os, '3.11') for os in EXPENSIVE_PLATFORMS)

        else:
            configs.append(MatrixConfig('standard+', 'py', 'ubuntu-large', '3.11', pytest_options=EXPENSIVE_PYTEST))
            # Non-HF notebooks. HF-dependent notebooks are gated behind --include-expensive, which only the
            # scheduled run passes (see NB_TEST_OPTS in pytest.yml), so they are excluded here.
            configs.append(MatrixConfig('notebooks+', 'ipynb', 'ubuntu-large', '3.11'))

        # Standard test suite on main & basic platforms on Python 3.14
        configs.extend(MatrixConfig('standard', 'py', os, '3.14') for os in (MAIN_PLATFORM, *BASIC_PLATFORMS))

        # Standard test suite on Ubuntu on intermediate Python versions
        configs.extend(MatrixConfig('standard', 'py', MAIN_PLATFORM, py) for py in ('3.11', '3.12', '3.13'))

        # Minimal tests on Python 3.14
        configs.append(MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.14', uv_options='--no-dev'))

        # Minimal tests on alternative platforms (we don't run the standard suite on these, since dev dependencies
        # can be hit-or-miss)
        configs.extend(MatrixConfig('minimal', 'py', os, '3.11', uv_options='--no-dev') for os in ALTERNATIVE_PLATFORMS)

        # Minimal tests with S3 media destination. We use a unique bucket name that incorporates today's date, so that
        # different test runs don't interfere with each other and any stale data is easy to clean up.
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            configs.append(
                MatrixConfig(
                    's3-output-dest',
                    'py',
                    MAIN_PLATFORM,
                    '3.11',
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
