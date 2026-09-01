"""Tests for the project archive: what a pod fetches and an image build installs from."""

from __future__ import annotations

import logging
import tarfile
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pixeltable import exceptions as excs
from pixeltable.catalog import Path as PxtPath
from pixeltable.config import Config, DatabaseConfig
from pixeltable.utils.project import create_project_archive

from ..utils import pxt_raises


def local_entry() -> DatabaseConfig | None:
    """The project's entry for the local database, which selects the files an archive holds."""
    return Config.get().get_database_config(PxtPath.parse('', allow_empty_path=True))


class TestProjectArchive:
    def test_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every project file is packaged under project/, alongside the build's metadata.json."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        # newline='\n' so the write isn't translated to CRLF on Windows (the content assertion below is exact)
        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n', newline='\n')
        (tmp_path / 'subdir').mkdir()
        (tmp_path / 'subdir' / 'helper.py').write_text('# helper\n')

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            assert sorted(tar.getnames()) == ['metadata.json', 'project/subdir/helper.py', 'project/udfs.py']
            with tar.extractfile(tar.getmember('project/udfs.py')) as f:
                assert f.read().decode() == 'import pixeltable as pxt\n'

    def test_include_exclude(self, tmp_path: Path) -> None:
        (tmp_path / 'a_include.txt').write_text('# included by default')
        (tmp_path / 'b_exclude.txt').write_text('# excluded explicitly')
        (tmp_path / 'a_include.py').write_text('# exclude-then-include')
        (tmp_path / 'a_exclude.py').write_text('# excluded')

        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [[pixeltable.database]]
                exclude = ["*.py", "b_exclude.txt"]
                include = ["a_include.py"]
            """)
        )

        Config.init(reinit=True, project_root=tmp_path)

        archive_path = create_project_archive(tmp_path, local_entry())

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/a_include.txt' in members
            assert 'project/a_include.py' in members
            assert 'project/pixeltable.toml' in members
            assert 'project/a_exclude.py' not in members
            assert 'project/b_exclude.txt' not in members

    def test_gitignore_respected(self, tmp_path: Path) -> None:
        """Files matching .gitignore patterns are excluded from project/."""
        (tmp_path / '.gitignore').write_text('__pycache__/\n*.pyc\n.env\n')
        (tmp_path / 'app.py').write_text('# app')
        (tmp_path / '__pycache__').mkdir()
        (tmp_path / '__pycache__' / 'app.cpython-311.pyc').write_bytes(b'\x00')
        (tmp_path / '.env').write_text('SECRET=abc')

        Config.init(reinit=True)

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/app.py' in members
            assert not any('__pycache__' in m for m in members)
            assert 'project/.env' not in members

    def test_uv_lock_included(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """uv.lock in the project dir is included under project/ for server-side uv sync."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "app"\n')

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            assert 'project/uv.lock' in tar.getnames()
            assert 'project/pyproject.toml' in tar.getnames()
            # no root-level requirements.txt or runtime_config.json
            assert 'requirements.txt' not in tar.getnames()
            assert 'runtime_config.json' not in tar.getnames()

    def test_single_table_config_form(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """[pixeltable.database] written as one table, which a project predating the array form uses."""
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database]
                system_dependencies = ["ffmpeg", "libpq"]
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        entry = local_entry()
        assert entry is not None
        assert entry.system_dependencies == ['ffmpeg', 'libpq']

    def test_is_valid_bz2_tar(self, tmp_path: Path) -> None:
        """The output file is a valid bz2 tarball."""
        archive_path = create_project_archive(tmp_path)
        assert tarfile.is_tarfile(archive_path)
        assert archive_path.suffix == '.bz2'

    def test_no_lockfile(self, init_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
        """A project with no lockfile draws a warning and is packaged anyway; a conda environment is not a lockfile."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)
        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n')
        monkeypatch.setenv('CONDA_PREFIX', str(tmp_path / 'envs' / 'pxt'))

        # the console logger writes to the stream it was built with, which no capture fixture owns
        monkeypatch.setattr(logging.getLogger('pixeltable'), 'propagate', True)
        with caplog.at_level(logging.WARNING, logger='pixeltable'):
            archive_path = create_project_archive(tmp_path)
        assert 'No dependency lockfile' in caplog.text

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
        assert 'project/udfs.py' in members
        assert all('conda' not in member for member in members)

    def test_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A project directory that does not exist, and a project config file Config cannot use."""
        with pytest.raises(FileNotFoundError, match='does not exist'):
            create_project_archive(Path('/nonexistent/path/xyz'))

        # Config reads the project's config file, so an unusable entry is refused before deploy sees it
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [[pixeltable.database]]
                include = "not-a-list"
            """)
        )
        monkeypatch.chdir(tmp_path)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'Invalid `DatabaseConfig`'):
            Config.init(reinit=True)
