import json
import pathlib
import sys
import textwrap

import pytest

from pixeltable_cli.client.commands import functions


class TestFunctions:
    def test_lists_type_methods(self, capsys: pytest.CaptureFixture[str]) -> None:
        functions.run(['--on', 'Json'])
        lines = capsys.readouterr().out.splitlines()
        assert any(line.startswith('udf       pxt.Json.len() -> pxt.Int  # ') for line in lines)

        functions.run(['video', '--kind', 'iterator', '--json'])
        records = json.loads(capsys.readouterr().out)
        assert all(record['kind'] == 'iterator' for record in records)
        assert 'pixeltable.functions.video.frame_iterator' in [record['path'] for record in records]

        functions.run(['--search', 'no function mentions this'])
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == 'pxt functions: no function matches\n'

    def test_lists_project_module(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        udf_src = textwrap.dedent(
            """
            import pixeltable as pxt


            @pxt.udf
            def shout(text: str) -> str:
                return text.upper()
            """
        )
        monkeypatch.setattr(sys, 'path', list(sys.path))

        # outside a project, a module in the working directory
        (tmp_path / 'cli_functions_app.py').write_text(udf_src)
        monkeypatch.chdir(tmp_path)
        functions.run(['cli_functions_app'])
        assert capsys.readouterr().out == 'udf       cli_functions_app.shout(text: pxt.String) -> pxt.String\n'

        # in a project, the module path from the project root, from any of its directories
        project = tmp_path / 'project'
        (project / 'api').mkdir(parents=True)
        (project / 'pixeltable.toml').write_text('')
        (project / 'cli_functions_project_app.py').write_text(udf_src)
        monkeypatch.chdir(project / 'api')
        functions.run(['cli_functions_project_app'])
        assert capsys.readouterr().out == 'udf       cli_functions_project_app.shout(text: pxt.String) -> pxt.String\n'

    def test_unknown_module_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            functions.run(['no_such_module'])
        assert exc_info.value.code == 1
        assert capsys.readouterr().err == "pxt functions: No module named 'no_such_module'\n"
