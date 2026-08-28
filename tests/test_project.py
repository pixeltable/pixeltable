"""The project's file selection and its fingerprint."""

import pathlib

import pytest

import pixeltable as pxt
from pixeltable.config import DatabaseConfig
from pixeltable.utils.project import project_fingerprint, selected_files


class TestProject:
    @pytest.fixture
    def project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """A project holding a module, a lockfile, a gitignored artifact and an ignored cache."""
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / '.gitignore').write_text('build/\n*.log\n')
        (tmp_path / 'run.log').write_text('noise\n')
        (tmp_path / 'build').mkdir()
        (tmp_path / 'build' / 'out.bin').write_text('artifact\n')
        return tmp_path

    def _names(self, root: pathlib.Path, config: DatabaseConfig | None = None) -> list[str]:
        return sorted(p.relative_to(root).as_posix() for p in selected_files(root, config))

    def test_selection_honors_gitignore(self, project: pathlib.Path) -> None:
        assert self._names(project) == ['.gitignore', 'app.py', 'uv.lock']

    def test_selection_applies_patterns(self, project: pathlib.Path) -> None:
        assert self._names(project, DatabaseConfig(exclude=['*.py'])) == ['.gitignore', 'uv.lock']
        assert self._names(project, DatabaseConfig(include=['run.log'])) == [
            '.gitignore',
            'app.py',
            'run.log',
            'uv.lock',
        ]
        # include_only replaces the selection, and the lockfile is selected whatever the patterns say
        assert self._names(project, DatabaseConfig(include_only=['app.py'])) == ['app.py', 'uv.lock']

    def test_selection_refuses_include_only_with_include(self, project: pathlib.Path) -> None:
        with pytest.raises(pxt.Error, match='include_only'):
            selected_files(project, DatabaseConfig(include_only=['app.py'], exclude=['*.log']))

    def test_fingerprint_covers_file_contents(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        assert set(before.files) == {'.gitignore', 'app.py', 'uv.lock'}
        assert before.restart_needed(before) is False

        (project / 'app.py').write_text('x = 2\n')
        after = project_fingerprint(project, None)
        assert after.restart_needed(before)
        assert after.rebuild_needed(before)
        assert after.changes(before) == ['app.py changed']

    def test_fingerprint_reports_added_and_removed_files(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'helper.py').write_text('y = 1\n')
        (project / 'app.py').unlink()
        assert project_fingerprint(project, None).changes(before) == ['helper.py added', 'app.py removed']

    def test_fingerprint_covers_the_lockfile(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'uv.lock').write_text('version = 2\n')
        assert project_fingerprint(project, None).changes(before) == ['uv.lock changed']

    def test_bindings_restart_without_rebuilding(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        after = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://two'}))
        assert after.restart_needed(before)
        assert after.rebuild_needed(before) is False
        assert after.changes(before) == ['var dest changed']

        with_secret = project_fingerprint(project, DatabaseConfig(secrets={'openai': 'env:OPENAI_API_KEY'}))
        assert with_secret.restart_needed(before)
        assert with_secret.rebuild_needed(before) is False
        assert with_secret.changes(before) == ['var dest changed', 'secret openai changed']

    def test_environment_inputs_rebuild(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(python_version='3.11'))
        after = project_fingerprint(project, DatabaseConfig(python_version='3.12'))
        assert after.rebuild_needed(before)
        assert after.changes(before) == ['python_version 3.11 -> 3.12']

        with_deps = project_fingerprint(project, DatabaseConfig(python_version='3.11', system_dependencies=['ffmpeg']))
        assert with_deps.rebuild_needed(before)
        assert with_deps.changes(before) == ['system_dependencies changed']
