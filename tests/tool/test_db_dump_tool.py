import subprocess


class TestDbDumpTool:

    def test_db_dump_tool(self) -> None:
        subprocess.run('python pixeltable/tool/create_test_db_dump.py'.split(' '), check=True)
