import subprocess

import pytest

from ..utils import skip_test_if_not_installed


@pytest.mark.remote_api
class TestPerftestTool:
    def test_perftest_tool(self) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_not_installed('google.genai')
        subprocess.run(
            ('python', '-m', 'tool.perftest_providers', '--provider', 'openai', '--n', '1', '--t', '10'), check=True
        )
