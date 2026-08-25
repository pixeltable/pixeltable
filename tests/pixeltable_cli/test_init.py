"""Tests for 'pxt init'."""

import pathlib

import pytest

from .conftest import PxtRunner

pytestmark = pytest.mark.local('marks a directory on the filesystem, independent of any catalog')


class TestInit:
    def test_marks_a_directory(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A fresh directory gets a pixeltable.toml, and marking it again reports what is already there."""
        r = cli('init', cwd=tmp_path)
        assert f'project root: {tmp_path}' in r.stdout
        assert 'wrote pixeltable.toml' in r.stdout
        assert (tmp_path / 'pixeltable.toml').is_file()

        r = cli('init', cwd=tmp_path)
        assert r.returncode == 0
        assert 'already marked by pixeltable.toml' in r.stdout

        r = cli('init', '--json', cwd=tmp_path)
        assert r.json == {
            'project_root': str(tmp_path),
            'marker': str(tmp_path / 'pixeltable.toml'),
            'created': False,
            'unusable_dirs': [],
        }

    def test_pyproject_gets_a_section(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A directory that already has a pyproject.toml is marked in it, rather than beside it."""
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[project]\nname = "proj"\n', encoding='utf-8')

        r = cli('init', cwd=tmp_path)
        assert 'wrote pyproject.toml' in r.stdout
        assert '[tool.pixeltable]' in pyproject.read_text()
        assert not (tmp_path / 'pixeltable.toml').exists()
        assert cli('init', cwd=tmp_path).stdout.count('already marked by pyproject.toml') == 1

    def test_a_root_above_is_refused(self, cli: PxtRunner, tmp_path: pathlib.Path) -> None:
        """A directory under a project root is refused, and -f makes it a root of its own."""
        (tmp_path / 'pixeltable.toml').write_text('', encoding='utf-8')
        nested = tmp_path / 'ad_gen'
        nested.mkdir()

        r = cli('init', cwd=nested, check=False)
        assert r.returncode == 3
        assert 'is already a project root' in r.stderr
        assert '-f' in r.stderr
        assert not (nested / 'pixeltable.toml').exists()

        r = cli('init', '-f', cwd=nested)
        assert (nested / 'pixeltable.toml').is_file()

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
