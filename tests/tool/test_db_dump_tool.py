import platform
import subprocess

import pytest

from ..utils import skip_test_if_not_installed


@pytest.mark.skipif(platform.system() == 'Windows', reason='Tool is not supported on Windows')
class TestDbDumpTool:

    def test_db_dump_tool(self) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        subprocess.run('python pixeltable/tool/create_test_db_dump.py'.split(' '), check=True)
