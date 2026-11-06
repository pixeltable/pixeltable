import json
import pathlib
import sys
import textwrap

import pytest

from pixeltable_cli.client.commands import functions


class TestFunctions:
    def test_filters(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # pxt functions unloads the modules under the working directory, which would include the test modules
        monkeypatch.chdir(tmp_path)
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

    def test_project_module(
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
        path = list(sys.path)

        # outside a project, a module in the working directory
        (tmp_path / 'cli_functions_app.py').write_text(udf_src)
        monkeypatch.chdir(tmp_path)
        functions.run(['cli_functions_app'])
        assert capsys.readouterr().out == 'udf       cli_functions_app.shout(text: pxt.String) -> pxt.String\n'
        assert sys.path == path

        # pxt shell runs every command in one process: the next command reads the edited module
        (tmp_path / 'cli_functions_app.py').write_text(udf_src.replace('shout', 'whisper'))
        functions.run(['cli_functions_app'])
        assert capsys.readouterr().out == 'udf       cli_functions_app.whisper(text: pxt.String) -> pxt.String\n'

        # in a project, the module path from the project root, from any of its directories
        project = tmp_path / 'project'
        (project / 'api').mkdir(parents=True)
        (project / 'pixeltable.toml').write_text('')
        (project / 'cli_functions_project_app.py').write_text(udf_src)
        monkeypatch.chdir(project / 'api')
        functions.run(['cli_functions_project_app'])
        assert capsys.readouterr().out == 'udf       cli_functions_project_app.shout(text: pxt.String) -> pxt.String\n'

    def test_module_errors(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        for module in ('no_such_module', 'no_such_module.sub'):
            with pytest.raises(SystemExit) as exc_info:
                functions.run([module])
            assert exc_info.value.code == 1
            assert capsys.readouterr().err == f"pxt functions: No module named '{module}'\n"

        # a module that fails to import
        (tmp_path / 'cli_functions_broken.py').write_text("raise RuntimeError('no config')\n")
        with pytest.raises(SystemExit) as exc_info:
            functions.run(['cli_functions_broken'])
        assert exc_info.value.code == 1
        assert capsys.readouterr().err == 'pxt functions: error loading cli_functions_broken: RuntimeError: no config\n'

        # with --json, the error is also a JSON document on stdout
        with pytest.raises(SystemExit) as exc_info:
            functions.run(['no_such_module', '--json'])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {
            'error_code': 'INVALID_ARGUMENT',
            'message': "No module named 'no_such_module'",
            'retryable': False,
        }
        assert captured.err == "pxt functions: No module named 'no_such_module'\n"

    def test_malformed_project_config(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_file = tmp_path / 'pixeltable.toml'
        config_file.write_text('[[')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            functions.run(['app'])
        assert exc_info.value.code == 1
        assert capsys.readouterr().err.startswith(f'pxt functions: {config_file} cannot be parsed: ')

        with pytest.raises(SystemExit) as exc_info:
            functions.run(['app', '--json'])
        assert exc_info.value.code == 1
        doc = json.loads(capsys.readouterr().out)
        assert (doc['error_code'], doc['retryable']) == ('INVALID_CONFIGURATION', False)
        assert doc['message'].startswith(f'{config_file} cannot be parsed: ')
