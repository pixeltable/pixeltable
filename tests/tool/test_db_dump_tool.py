import platform
import subprocess
import sys

import pytest

from ..utils import skip_test_if_not_installed


@pytest.mark.skipif(platform.system() == 'Windows', reason='Tool is not supported on Windows')
@pytest.mark.skipif(sys.version_info >= (3, 10), reason='Runs only on Python 3.9 (due to pickling issue)')
class TestDbDumpTool:

    def test_db_dump_tool(self) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        skip_test_if_not_installed('toml')
        subprocess.run('python pixeltable/tool/create_test_db_dump.py'.split(' '), check=True)
