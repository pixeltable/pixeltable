"""Tests for 'pxt init'."""

import pathlib
import tomllib

import pytest

from .conftest import PxtRunner

pytestmark = pytest.mark.local('writes a file on the filesystem, independent of any catalog')


class TestInit:
    def test_writes_the_project_configuration(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A fresh directory gets a pixeltable.toml, and running it again reports what is already there."""
        r = cli('init', cwd=tmp_path)
        assert f'project root: {tmp_path}' in r.stdout
        assert 'wrote pixeltable.toml' in r.stdout

        # what it wrote is the local database, and nothing else
        written = tomllib.loads((tmp_path / 'pixeltable.toml').read_text())
        assert written == {'pixeltable': {'database': [{}]}}, written

        r = cli('init', cwd=tmp_path)
        assert r.returncode == 0
        assert 'already configured by pixeltable.toml' in r.stdout

        r = cli('init', '--json', cwd=tmp_path)
        assert r.json == {
            'project_root': str(tmp_path),
            'config_file': str(tmp_path / 'pixeltable.toml'),
            'created': False,
            'unusable_dirs': [],
        }

    def test_the_written_configuration_binds_a_var(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """The commented binding in the file it writes is the form that actually binds a var."""
        cli('init', cwd=tmp_path)
        config_file = tmp_path / 'pixeltable.toml'
        text = config_file.read_text()
        assert "# vars.media_dest = 's3://bucket/prefix'" in text
        config_file.write_text(text.replace('# vars.media_dest', 'vars.media_dest'))
        entries = tomllib.loads(config_file.read_text())['pixeltable']['database']
        assert entries == [{'vars': {'media_dest': 's3://bucket/prefix'}}], entries

    def test_pyproject_gets_the_configuration(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A directory that already has a pyproject.toml is configured in it, rather than beside it."""
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[project]\nname = "proj"\n', encoding='utf-8')

        r = cli('init', cwd=tmp_path)
        assert 'wrote pyproject.toml' in r.stdout
        assert not (tmp_path / 'pixeltable.toml').exists()
        written = tomllib.loads(pyproject.read_text())
        assert written == {'project': {'name': 'proj'}, 'tool': {'pixeltable': {'database': [{}]}}}, written
        assert cli('init', cwd=tmp_path).stdout.count('already configured by pyproject.toml') == 1

    def test_a_root_above_is_refused(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A directory under a project root is refused: a project has one root, and nesting is not offered."""
        (tmp_path / 'pixeltable.toml').write_text('', encoding='utf-8')
        nested = tmp_path / 'ad_gen'
        nested.mkdir()

        r = cli('init', cwd=nested, check=False)
        assert r.returncode == 3
        assert 'already holds a project configuration' in r.stderr
        assert str(tmp_path) in r.stderr
        assert not (nested / 'pixeltable.toml').exists()

    def test_reports_a_directory_no_module_path_can_use(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A directory whose name is not an identifier is reported when it holds Python files."""
        (tmp_path / 'ad gen').mkdir()
        (tmp_path / 'ad gen' / 'app.py').write_text('', encoding='utf-8')
        (tmp_path / 'plain').mkdir()
        (tmp_path / 'plain' / 'app.py').write_text('', encoding='utf-8')
        # a directory with no Python files under it is nobody's module path
        (tmp_path / 'raw data').mkdir()

        r = cli('init', cwd=tmp_path)
        assert r.returncode == 0
        assert "'ad gen' holds Python files, but its name is not a Python identifier" in r.stderr
        assert 'plain' not in r.stderr
        assert 'raw data' not in r.stderr
        assert cli('init', '--json', cwd=tmp_path).json['unusable_dirs'] == ['ad gen']


class TestProjectHandoff:
    def test_daemon_serves_the_project_the_client_stands_in(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A command establishes which project the daemon serves; one that establishes none takes it as it is."""
        project = tmp_path / 'proj'
        project.mkdir()
        cli('init', cwd=project)

        served = cli('status', '--json', cwd=project).json
        assert served['project_root'] == str(project)

        # a directory that marks no project asks for nothing, so the daemon keeps serving the one it has
        outside = tmp_path / 'outside'
        outside.mkdir()
        unchanged = cli('status', '--json', cwd=outside).json
        assert unchanged['project_root'] == str(project)
        assert unchanged['pid'] == served['pid']

        # a second project takes the daemon over, which is a restart
        second = tmp_path / 'second'
        second.mkdir()
        cli('init', cwd=second)
        moved = cli('status', '--json', cwd=second).json
        assert moved['project_root'] == str(second)
        assert moved['pid'] != served['pid']
