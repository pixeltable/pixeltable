"""The project's file selection and its fingerprint."""

import pathlib

import pytest

from pixeltable import exceptions as excs
from pixeltable.config import DatabaseConfig
from pixeltable.utils.project import _archive_files, project_fingerprint

from .utils import pxt_raises


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
        return sorted(p.relative_to(root).as_posix() for p in _archive_files(root, config))

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
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='include_only'):
            _archive_files(project, DatabaseConfig(include_only=['app.py'], exclude=['*.log']))

    def test_fingerprint_covers_file_contents(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        assert set(before.files) == {'.gitignore', 'app.py', 'uv.lock'}
        assert before.restart_needed(before) is False

        (project / 'app.py').write_text('x = 2\n')
        after = project_fingerprint(project, None)
        assert after.restart_needed(before)
        # a source edit is shipped as an archive and does not touch the environment the image holds
        assert after.archive_needed(before)
        assert after.image_needed(before) is False
        assert after.changes(before) == ['app.py changed']

    def test_fingerprint_reports_added_and_removed_files(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'helper.py').write_text('y = 1\n')
        (project / 'app.py').unlink()
        assert project_fingerprint(project, None).changes(before) == ['helper.py added', 'app.py removed']

    def test_fingerprint_covers_the_lockfile(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'uv.lock').write_text('version = 2\n')
        after = project_fingerprint(project, None)
        assert after.changes(before) == ['uv.lock changed']
        # the lockfile is a project file and the declaration of the environment, so it moves both artifacts
        assert after.image_needed(before)
        assert after.archive_needed(before)
        assert after.changes(before, 'image') == ['uv.lock changed']
        assert after.changes(before, 'archive') == ['uv.lock changed']

    def test_publish_compares_the_files_an_application_loaded(self, project: pathlib.Path) -> None:
        """A published project holds every selected file; an application loads a part of it."""
        published = project_fingerprint(project, None)
        loaded = published.model_copy(update={'files': {'app.py': published.files['app.py']}})

        # the two are scoped differently, so they are not equal, and the database is still up to date
        assert loaded.restart_needed(published)
        assert loaded.publish_needed(published) is False

        # a file the application never loaded moves the project, and asks nothing of this application
        (project / 'other.py').write_text('y = 1\n')
        assert loaded.publish_needed(project_fingerprint(project, None)) is False

        # a file it did load, edited here and not yet given to the database, asks for a publish
        (project / 'app.py').write_text('x = 2\n')
        edited = project_fingerprint(project, None)
        loaded = edited.model_copy(update={'files': {'app.py': edited.files['app.py']}})
        assert loaded.publish_needed(published)
        assert loaded.changes(published, 'published') == ['app.py changed']

    def test_bindings_restart_without_shipping_anything(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        after = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://two'}))
        assert after.restart_needed(before)
        assert after.image_needed(before) is False
        assert after.archive_needed(before) is False
        assert after.changes(before) == ['var dest changed']

        with_secret = project_fingerprint(project, DatabaseConfig(secrets={'openai': 'env:OPENAI_API_KEY'}))
        assert with_secret.restart_needed(before)
        assert with_secret.image_needed(before) is False
        assert with_secret.changes(before) == ['var dest changed', 'secret openai changed']

    def test_environment_inputs_rebuild(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(python_version='3.11'))
        after = project_fingerprint(project, DatabaseConfig(python_version='3.12'))
        assert after.image_needed(before)
        # the files are untouched, so the archive the pods fetch stays as it is
        assert after.archive_needed(before) is False
        assert after.changes(before) == ['python_version 3.11 -> 3.12']
        assert after.changes(before, 'archive') == []

        with_deps = project_fingerprint(project, DatabaseConfig(python_version='3.11', system_dependencies=['ffmpeg']))
        assert with_deps.image_needed(before)
        assert with_deps.changes(before, 'image') == ['system_dependencies changed']

    def test_digests_name_the_two_artifacts(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        # a changed binding restarts a service without shipping either artifact
        rebound = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://two'}))
        assert rebound.archive_digest() == before.archive_digest()
        assert rebound.image_digest() == before.image_digest()

        (project / 'app.py').write_text('x = 2\n')
        edited = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        assert edited.archive_digest() != before.archive_digest()
        # one image serves every project declaring its environment, whatever their sources
        assert edited.image_digest() == before.image_digest()

        (project / 'app.py').write_text('x = 1\n')
        reverted = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        assert reverted.archive_digest() == before.archive_digest()

        with_deps = project_fingerprint(project, DatabaseConfig(system_dependencies=['ffmpeg']))
        assert with_deps.image_digest() != before.image_digest()
        assert with_deps.archive_digest() == before.archive_digest()
