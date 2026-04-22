"""Smoke test for the perftest tool. Runs with a single request and checks that it completes."""

import subprocess

import pytest

from ..utils import skip_test_if_no_client, skip_test_if_not_installed


@pytest.mark.remote_api
class TestPerftestTool:
    def test_perftest_tool(self) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('openai')
        subprocess.run(
            ('python', '-m', 'tool.perftest_providers', '--provider', 'openai', '--n', '1', '--t', '10'), check=True
        )
