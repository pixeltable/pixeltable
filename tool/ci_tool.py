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

VERY_EXPENSIVE_MARKERS = 'not resource_intensive and not benchmark'
EXPENSIVE_MARKERS = f'{VERY_EXPENSIVE_MARKERS} and not very_expensive'
DEFAULT_MARKERS = f'{EXPENSIVE_MARKERS} and not expensive'
RESOURCE_INTENSIVE_MARKERS = 'resource_intensive'

# Note: in addition to these pytest marker expressions, the tests that actually run are implicitly filtered by
# skip_test_if_not_installed() and the install configuration.

MAIN_PLATFORM = 'ubuntu-24.04'
BASIC_PLATFORMS = ('macos-15', 'windows-2025')
ALL_PLATFORMS = (MAIN_PLATFORM, *BASIC_PLATFORMS)
EXPENSIVE_PLATFORMS = ('ubuntu-small-t4',)
ALTERNATIVE_PLATFORMS = ('ubuntu-24.04-arm', 'macos-15-intel')

# The GitHub event names that pytest.yml is triggered on
TRIGGERS = ('pull_request', 'merge_group', 'schedule', 'workflow_dispatch')

DEFAULT_XDIST_WORKERS = 'logical'  # one worker per vCPU; -n auto would be 2 on windows-2025 (physical cores)
LARGE_RUNNER_XDIST_WORKERS = '2'


class MatrixConfig(NamedTuple):
    display_name_prefix: str
    test_category: Literal['py', 'ipynb', 'lint', 'random-ops', 'otel']
    os: str
    python_version: str
    uv_options: str = ''
    pytest_markers: str = DEFAULT_MARKERS
    xdist_workers: str = DEFAULT_XDIST_WORKERS
    pre_test_cmd: str = ''  # Extra bash command to be run just before tests
    random_ops_seconds: int = 1800

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
            'pytest-markers': self.pytest_markers,
            'xdist-workers': self.xdist_workers,
            'pre-test-cmd': self.pre_test_cmd,
            'random-ops-seconds': str(self.random_ops_seconds),
        }


def new_bucket_addr() -> str:
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    bucket_uuid = uuid.uuid4().hex
    return f's3://pxt-test/pytest-media-dest/{date_str}/{bucket_uuid}'


def build_configs(trigger: str, force_all: bool, has_aws_credentials: bool) -> list[MatrixConfig]:
    """Return the matrix configurations to run for the given trigger."""
    assert trigger in TRIGGERS, trigger

    # Run on every trigger: static checks, plus the tests under a no-dev-dependencies install on the oldest and newest
    # Python, and once with required deps pinned to their minimum versions.
    configs = [
        MatrixConfig('static-checks', 'lint', MAIN_PLATFORM, '3.11'),
        MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.11', uv_options='--no-dev'),
        MatrixConfig('minimal', 'py', MAIN_PLATFORM, '3.15', uv_options='--no-dev'),
        MatrixConfig('minimum-deps', 'py', MAIN_PLATFORM, '3.11', uv_options='--no-dev --resolution lowest-direct'),
    ]

    if trigger == 'pull_request':
        # On every push to a PR, the minimal tests on the basic platforms as well, and a shorter random-ops run than the
        # merge queue's
        configs.extend(
            MatrixConfig('minimal', 'py', platform, '3.11', uv_options='--no-dev') for platform in BASIC_PLATFORMS
        )
        configs.append(
            MatrixConfig(
                'random-ops', 'random-ops', MAIN_PLATFORM, '3.11', uv_options='--no-dev', random_ops_seconds=600
            )
        )

    else:
        # A non-PR trigger: merge queue, workflow dispatch, or schedule.

        configs.extend(MatrixConfig('standard', 'py', platform, '3.11') for platform in BASIC_PLATFORMS)

        configs.append(MatrixConfig('random-ops', 'random-ops', MAIN_PLATFORM, '3.11', uv_options='--no-dev'))
        configs.append(
            MatrixConfig('otel', 'otel', MAIN_PLATFORM, '3.11', uv_options='--no-dev --extra otel --extra serve')
        )

        if trigger == 'merge_group':
            configs.append(MatrixConfig('standard', 'py', MAIN_PLATFORM, '3.11'))
            # Non-HF notebooks. HF-dependent notebooks are gated behind --include-expensive, which only the
            # weekly run passes (see NB_TEST_OPTS in pytest.yml), so they are excluded here.
            configs.append(MatrixConfig('notebooks+', 'ipynb', 'ubuntu-large', '3.11'))

        else:
            # force_all is set by the "Run on all platforms" checkbox on a workflow dispatch, and by the weekly
            # scheduled run.
            if force_all:
                configs.append(
                    MatrixConfig('standard++', 'py', MAIN_PLATFORM, '3.11', pytest_markers=VERY_EXPENSIVE_MARKERS)
                )
                configs.append(MatrixConfig('notebooks++', 'ipynb', 'ubuntu-large', '3.11'))
                configs.extend(MatrixConfig('standard', 'py', platform, '3.11') for platform in EXPENSIVE_PLATFORMS)
            else:
                configs.append(MatrixConfig('standard+', 'py', MAIN_PLATFORM, '3.11', pytest_markers=EXPENSIVE_MARKERS))

            configs.append(
                MatrixConfig(
                    'resource-intensive',
                    'py',
                    'ubuntu-large',
                    '3.11',
                    pytest_markers=RESOURCE_INTENSIVE_MARKERS,
                    xdist_workers=LARGE_RUNNER_XDIST_WORKERS,
                    pre_test_cmd='./scripts/log-memory.sh 15 &',
                )
            )

        # Standard test suite on main & basic platforms on Python 3.15
        configs.extend(MatrixConfig('standard', 'py', platform, '3.15') for platform in ALL_PLATFORMS)

        # Standard test suite on Ubuntu on intermediate Python versions
        configs.extend(MatrixConfig('standard', 'py', MAIN_PLATFORM, py) for py in ('3.12', '3.13', '3.14'))

        # Minimal tests on alternative platforms (we don't run the standard suite on these, since dev dependencies
        # can be hit-or-miss)
        configs.extend(
            MatrixConfig('minimal', 'py', platform, '3.11', uv_options='--no-dev') for platform in ALTERNATIVE_PLATFORMS
        )

        # Minimal tests with S3 media destination. We use a unique bucket name that incorporates today's date, so that
        # different test runs don't interfere with each other and any stale data is easy to clean up.
        if has_aws_credentials:
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

    if trigger in ('pull_request', 'merge_group'):
        configs = [cfg._replace(pytest_markers=f'{cfg.pytest_markers} and not skip_merge_queue') for cfg in configs]

    configs.sort(key=lambda cfg: cfg.display_name)
    return configs


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

    configs = build_configs(trigger, force_all, has_aws_credentials=bool(os.environ.get('AWS_ACCESS_KEY_ID')))
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
    matrix_parser.add_argument('trigger', choices=TRIGGERS, help='CI trigger type')
    matrix_parser.add_argument('--force-all', action='store_true', help='Force generation of all configurations')
    matrix_parser.set_defaults(func=generate_matrix)

    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)

    sys.exit(0)


if __name__ == '__main__':
    main()
