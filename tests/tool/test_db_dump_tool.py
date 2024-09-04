import subprocess

from ..utils import skip_test_if_not_installed


class TestDbDumpTool:

    def test_db_dump_tool(self) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        skip_test_if_not_installed('toml')
        subprocess.run('python pixeltable/tool/create_test_db_dump.py'.split(' '), check=True)
