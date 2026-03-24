import subprocess

import pytest

from ..utils import skip_test_if_not_installed


@pytest.mark.remote_api
@pytest.mark.expensive
class TestPerftestTool:
    def test_perftest_tool(self) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_not_installed('google.genai')
        subprocess.run(
            ('python', 'tool/perftest_providers.py', '--provider', 'openai', '--n', '1', '--t', '10'), check=True
        )
