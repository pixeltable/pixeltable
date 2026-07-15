"""Tests for runtime bundle packaging (pixeltable/serving/deploy.py)."""

from __future__ import annotations

import json
import sys
import tarfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from pixeltable import exceptions as excs, metadata
from pixeltable.config import Config
from pixeltable.serving.deploy import build_db_runtime_bundle

from ..utils import pxt_raises

pytestmark = pytest.mark.local('runtime bundle packaging')


class TestDbRuntimeBundle:
    def test_bundle_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bundle always contains metadata.json and project/; nothing else at root."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n')
        (tmp_path / 'subdir').mkdir()
        (tmp_path / 'subdir' / 'helper.py').write_text('# helper\n')

        bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()

            # Required top-level files
            assert 'metadata.json' in members
            assert 'project/udfs.py' in members
            assert 'project/subdir/helper.py' in members

            # Nothing else at root
            root_files = [m for m in members if '/' not in m]
            assert root_files == ['metadata.json']

            # metadata.json has pxt_md_version and python_version
            with tar.extractfile(tar.getmember('metadata.json')) as f:
                meta = json.loads(f.read())
                assert meta['pxt_md_version'] == metadata.VERSION
                assert meta['python_version'] == f'{sys.version_info.major}.{sys.version_info.minor}'

            # project file content preserved
            with tar.extractfile(tar.getmember('project/udfs.py')) as f:
                assert f.read().decode() == 'import pixeltable as pxt\n'

    def test_bundle_include_exclude(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """include= limits which project files are bundled; exclude= removes matched files."""
        (tmp_path / 'a_include.py').write_text('# included')
        (tmp_path / 'a_exclude.py').write_text('# excluded')
        (tmp_path / 'b_exclude.txt').write_text('# excluded (not in include)')

        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database]
                include = ["*.py", "*.toml"]
                exclude = ["a_exclude.py"]
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/a_include.py' in members
            assert 'project/pixeltable.toml' in members
            assert 'project/a_exclude.py' not in members
            assert 'project/b_exclude.txt' not in members

    def test_bundle_gitignore_respected(self, tmp_path: Path) -> None:
        """Files matching .gitignore patterns are excluded from project/."""
        (tmp_path / '.gitignore').write_text('__pycache__/\n*.pyc\n.env\n')
        (tmp_path / 'app.py').write_text('# app')
        (tmp_path / '__pycache__').mkdir()
        (tmp_path / '__pycache__' / 'app.cpython-311.pyc').write_bytes(b'\x00')
        (tmp_path / '.env').write_text('SECRET=abc')

        bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/app.py' in members
            assert not any('__pycache__' in m for m in members)
            assert 'project/.env' not in members

    def test_bundle_uv_lock_included(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """uv.lock in the project dir is included under project/ for server-side uv sync."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "app"\n')

        bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            assert 'project/uv.lock' in tar.getnames()
            assert 'project/pyproject.toml' in tar.getnames()
            # no root-level requirements.txt or runtime_config.json
            assert 'requirements.txt' not in tar.getnames()
            assert 'runtime_config.json' not in tar.getnames()

    def test_bundle_system_dependencies_in_metadata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """system_dependencies from pixeltable.toml are validated and written to metadata.json."""
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database]
                system_dependencies = ["ffmpeg", "libpq"]
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        with patch('pixeltable.serving.deploy._validate_system_dependencies') as mock_validate:
            bundle_path = build_db_runtime_bundle(tmp_path)
            mock_validate.assert_called_once_with(['ffmpeg', 'libpq'])

        with tarfile.open(bundle_path, 'r:bz2') as tar, tar.extractfile(tar.getmember('metadata.json')) as f:
            meta = json.loads(f.read())
            assert meta['system_dependencies'] == ['ffmpeg', 'libpq']

    def test_bundle_no_system_dependencies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No system_dependencies → metadata.json has no system_dependencies key."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar, tar.extractfile(tar.getmember('metadata.json')) as f:
            meta = json.loads(f.read())
            assert 'system_dependencies' not in meta

    def test_bundle_is_valid_bz2_tar(self, tmp_path: Path) -> None:
        """The output file is a valid bz2 tarball."""
        bundle_path = build_db_runtime_bundle(tmp_path)
        assert tarfile.is_tarfile(bundle_path)
        assert bundle_path.suffix == '.bz2'

    def test_bundle_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error paths in build_db_runtime_bundle()."""
        with pytest.raises(FileNotFoundError, match='does not exist'):
            build_db_runtime_bundle(Path('/nonexistent/path/xyz'))

        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database]
                include = "not-a-list"
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'Invalid \[pixeltable\.database\]'):
            build_db_runtime_bundle(tmp_path)
