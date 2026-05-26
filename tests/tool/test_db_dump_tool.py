import platform
import subprocess
import sys
import sysconfig

import pytest

from pixeltable.env import Env

from ..utils import skip_test_if_not_installed


@pytest.mark.skipif(platform.system() == 'Windows', reason='Tool is not supported on Windows')
@pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Tool is not supported on Linux ARM')
@pytest.mark.skipif(sys.version_info >= (3, 11), reason='Runs only on Python 3.10 (due to pickling issue)')
class TestDbDumpTool:
    def test_db_dump_tool(self) -> None:
        if Env.get().is_using_cockroachdb:
            pytest.skip('Tool reinitializes the local pixeltable DB; not supported on CockroachDB')
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('label_studio_sdk')
        subprocess.run(('python', 'tool/create_test_db_dump.py'), check=True)
