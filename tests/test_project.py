"""The project's file selection and its fingerprint."""

import pathlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from pixeltable import exceptions as excs
from pixeltable.config import DatabaseConfig
from pixeltable.utils import project as project_mod
from pixeltable.utils.project import ProjectPart, _archive_files, loaded_fingerprint, project_fingerprint

from .utils import pxt_raises

ARCHIVE, IMAGE, BINDINGS = ProjectPart.ARCHIVE, ProjectPart.IMAGE, ProjectPart.BINDINGS


def _some(files: dict[str, str], *paths: str) -> dict[str, str]:
    return {path: files[path] for path in paths}


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

    def test_gitignore(self, project: pathlib.Path) -> None:
        assert self._names(project) == ['.gitignore', 'app.py', 'uv.lock']

    def test_venv(self, project: pathlib.Path) -> None:
        """A virtual environment is dropped even when nothing ignores it."""
        for name, marker in (('.venv', 'pyvenv.cfg'), ('env', 'conda-meta/history')):
            venv = project / name
            (venv / 'lib').mkdir(parents=True)
            (venv / 'lib' / 'pkg.py').write_text('z = 1\n')
            (venv / marker).parent.mkdir(parents=True, exist_ok=True)
            (venv / marker).write_text('')
        assert self._names(project) == ['.gitignore', 'app.py', 'uv.lock']

    def test_patterns(self, project: pathlib.Path) -> None:
        assert self._names(project, DatabaseConfig(exclude=['*.py'])) == ['.gitignore', 'uv.lock']
        assert self._names(project, DatabaseConfig(include=['run.log'])) == [
            '.gitignore',
            'app.py',
            'run.log',
            'uv.lock',
        ]
        # include_only replaces the selection, and the lockfile is selected whatever the patterns say
        assert self._names(project, DatabaseConfig(include_only=['app.py'])) == ['app.py', 'uv.lock']

    def test_project_config_always_selected(self, project: pathlib.Path) -> None:
        """A pod reads its database entry out of the archive, so no pattern can drop the file holding it."""
        (project / 'pixeltable.toml').write_text("[[pixeltable.database]]\nname = 'pxt://acme:main'\n")
        assert 'pixeltable.toml' in self._names(project, DatabaseConfig(include_only=['app.py']))
        assert 'pixeltable.toml' in self._names(project, DatabaseConfig(exclude=['*.toml']))

    def test_include_only_with_include(self, project: pathlib.Path) -> None:
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='include_only'):
            _archive_files(project, DatabaseConfig(include_only=['app.py'], exclude=['*.log']))

    def test_file_contents(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        assert set(before.files) == {'.gitignore', 'app.py', 'uv.lock'}
        assert before.compare(before) == set()

        (project / 'app.py').write_text('x = 2\n')
        after = project_fingerprint(project, None)
        # a source edit is sent as an archive and does not touch the environment the image holds
        assert after.compare(before) == {ARCHIVE}
        assert after.changes(before) == ['app.py changed']

    def test_added_and_removed_files(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'helper.py').write_text('y = 1\n')
        (project / 'app.py').unlink()
        assert project_fingerprint(project, None).changes(before) == ['helper.py added', 'app.py removed']

    def test_lockfile(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, None)
        (project / 'uv.lock').write_text('version = 2\n')
        after = project_fingerprint(project, None)
        # the lockfile is a project file and the declaration of the environment, so it moves both artifacts
        assert after.compare(before) == {IMAGE, ARCHIVE}
        assert after.changes(before) == ['uv.lock changed']
        assert after.changes(before, {IMAGE}) == ['uv.lock changed']
        assert after.changes(before, {ARCHIVE}) == ['uv.lock changed']

    def test_loaded_files(self, project: pathlib.Path) -> None:
        """A published project holds every selected file; an application loads a part of it."""
        published = project_fingerprint(project, None)
        # what loaded_fingerprint() produces: the modules the application imported, plus the lockfile
        loaded = published.model_copy(update={'files': _some(published.files, 'app.py', 'uv.lock')})

        # the two name different files, so one holds what the other does not; the loaded ones agree
        assert loaded.compare(published) == {ARCHIVE}
        assert loaded.compare(published, own_files_only=True) == set()

        # a file the application never loaded moves the project, and asks nothing of this application
        (project / 'other.py').write_text('y = 1\n')
        assert loaded.compare(project_fingerprint(project, None), own_files_only=True) == set()

        # a file it did load, edited here and not yet given to the database, asks for a publish
        (project / 'app.py').write_text('x = 2\n')
        edited = project_fingerprint(project, None)
        loaded = edited.model_copy(update={'files': _some(edited.files, 'app.py', 'uv.lock')})
        assert loaded.compare(published, own_files_only=True) == {ARCHIVE}
        assert loaded.changes(published, {ARCHIVE}, own_files_only=True) == ['app.py changed']

    def test_environment_files_are_not_loaded_files(self, project: pathlib.Path) -> None:
        """A project holding its own virtualenv fingerprints the same whatever it has loaded from it.

        Each process imports a different set of the environment's packages, so counting them would make two
        fingerprints of one application differ and restart the service for nothing.
        """
        installed = project / '.venv' / 'lib' / 'python3.11' / 'site-packages'
        installed.mkdir(parents=True)
        (installed / 'vendored.py').write_text('z = 1\n')

        before = loaded_fingerprint(project, None)
        module = ModuleType('vendored')
        module.__file__ = str(installed / 'vendored.py')
        with patch.dict(sys.modules, {'vendored': module}), patch.object(project_mod, '_ENV_DIRS', (installed,)):
            assert loaded_fingerprint(project, None) == before

    def test_bindings(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        after = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://two'}))
        # a binding is held by neither artifact, and a process that read the old one is stale
        assert after.compare(before) == {BINDINGS}
        assert after.changes(before) == ['var dest changed']

        with_secret = project_fingerprint(project, DatabaseConfig(secrets={'openai': 'env:OPENAI_API_KEY'}))
        assert with_secret.compare(before) == {BINDINGS}
        assert with_secret.changes(before) == ['var dest changed', 'secret openai changed']

    def test_environment(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(python_version='3.11'))
        after = project_fingerprint(project, DatabaseConfig(python_version='3.12'))
        # the files are untouched, so the archive the pods fetch stays as it is
        assert after.compare(before) == {IMAGE}
        assert after.changes(before) == ['python_version 3.11 -> 3.12']
        assert after.changes(before, {ARCHIVE}) == []

        with_deps = project_fingerprint(project, DatabaseConfig(python_version='3.11', system_dependencies=['ffmpeg']))
        assert with_deps.compare(before) == {IMAGE}
        assert with_deps.changes(before, {IMAGE}) == ['system_dependencies changed']

    def test_digests(self, project: pathlib.Path) -> None:
        before = project_fingerprint(project, DatabaseConfig(vars={'dest': 's3://one'}))
        # a changed binding restarts a service without sending either artifact
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
