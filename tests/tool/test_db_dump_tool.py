import platform
import subprocess
import sysconfig

import pytest

from ..utils import skip_test_if_not_installed


@pytest.mark.skipif(platform.system() == 'Windows', reason='Tool is not supported on Windows')
@pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Tool is not supported on Linux ARM')
@pytest.mark.very_expensive
class TestDbDumpTool:
    def test_db_dump_tool(self) -> None:
        skip_test_if_not_installed('transformers')
        # A generous timeout to allow for a large HF download
        subprocess.run(('python', 'tool/create_test_db_dump.py'), check=True, timeout=900)
