"""Tests for runtime bundle packaging (pixeltable/serving/deploy.py)."""

from __future__ import annotations

import json
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

_FAKE_CONDA_YAML = b'name: test\ndependencies:\n  - python=3.11\n  - pip:\n    - fastapi==0.120.0\n'
_FAKE_REQUIREMENTS = b'fastapi==0.120.0\nhttpx==0.27.0\n'


class TestDbRuntimeBundle:
    def test_bundle_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build a runtime bundle and verify its structure and file contents."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n')
        (tmp_path / 'subdir').mkdir()
        (tmp_path / 'subdir' / 'helper.py').write_text('# helper\n')

        with (
            patch('pixeltable.serving.deploy._export_conda_env', return_value=_FAKE_CONDA_YAML),
            patch('pixeltable.serving.deploy._export_uv_requirements', return_value=_FAKE_REQUIREMENTS),
            patch('pixeltable.serving.deploy._detect_pxt_version', return_value='0.7.0'),
        ):
            bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()

            # Required top-level files
            assert 'metadata.json' in members
            assert 'environment.yml' in members
            assert 'requirements.txt' in members
            assert 'runtime_config.json' in members

            # Project files under project/
            assert 'project/udfs.py' in members
            assert 'project/subdir/helper.py' in members

            # metadata.json
            with tar.extractfile(tar.getmember('metadata.json')) as f:
                content = json.loads(f.read())
                assert content['pxt_md_version'] == metadata.VERSION

            # requirements.txt content preserved exactly
            with tar.extractfile(tar.getmember('requirements.txt')) as f:
                assert f.read() == _FAKE_REQUIREMENTS

            # environment.yml content preserved exactly
            with tar.extractfile(tar.getmember('environment.yml')) as f:
                assert f.read() == _FAKE_CONDA_YAML

            # runtime_config.json written for stable version
            with tar.extractfile(tar.getmember('runtime_config.json')) as f:
                cfg = json.loads(f.read())
                assert cfg['pixeltable_source']['version'] == '0.7.0'

            # project file content preserved
            with tar.extractfile(tar.getmember('project/udfs.py')) as f:
                assert f.read().decode() == 'import pixeltable as pxt\n'

    def test_bundle_include_exclude(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """include= limits which project files are bundled; exclude= removes matched files."""
        (tmp_path / 'a_include.py').write_text('# included by include pattern')
        (tmp_path / 'a_exclude.py').write_text('# excluded by exclude pattern')
        (tmp_path / 'b_exclude.txt').write_text('# excluded because not in include')

        config_path = tmp_path / 'pixeltable.toml'
        config_path.write_text(
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
            assert 'project/pixeltable.toml' in members  # *.toml pattern
            assert 'project/a_exclude.py' not in members  # explicitly excluded
            assert 'project/b_exclude.txt' not in members  # not in include pattern

            # file content preserved
            with tar.extractfile(tar.getmember('project/a_include.py')) as f:
                assert f.read().decode() == '# included by include pattern'

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

    def test_bundle_pixeltable_source_from_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit [pixeltable.database.pixeltable_source] config takes precedence over detected version."""
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database.pixeltable_source]
                git    = "https://github.com/pixeltable/pixeltable"
                branch = "main"
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        # Even if a stable version is detected, the TOML source wins
        with patch('pixeltable.serving.deploy._detect_pxt_version', return_value='9.9.9'):
            bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            with tar.extractfile(tar.getmember('runtime_config.json')) as f:
                cfg = json.loads(f.read())
                assert cfg['pixeltable_source']['git'] == 'https://github.com/pixeltable/pixeltable'
                assert cfg['pixeltable_source']['branch'] == 'main'
                assert 'version' not in cfg['pixeltable_source']

    def test_bundle_no_runtime_config_for_dev_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A local dev version does not produce runtime_config.json — not installable from PyPI."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        with patch('pixeltable.serving.deploy._detect_pxt_version', return_value='0.0.1.dev1600+abc1234'):
            bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            assert 'runtime_config.json' not in tar.getnames()

    def test_bundle_no_conda_no_uv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no conda env and no uv.lock, only metadata.json and project/ files are present."""
        monkeypatch.chdir(tmp_path)
        Config.init({}, reinit=True)

        (tmp_path / 'app.py').write_text('# app')

        with (
            patch('pixeltable.serving.deploy._export_conda_env', return_value=None),
            patch('pixeltable.serving.deploy._export_uv_requirements', return_value=None),
            patch('pixeltable.serving.deploy._detect_pxt_version', return_value=None),
        ):
            bundle_path = build_db_runtime_bundle(tmp_path)

        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'metadata.json' in members
            assert 'project/app.py' in members
            assert 'environment.yml' not in members
            assert 'requirements.txt' not in members
            assert 'runtime_config.json' not in members

    def test_bundle_is_valid_bz2_tar(self, tmp_path: Path) -> None:
        """The output file is a valid bz2 tarball."""
        bundle_path = build_db_runtime_bundle(tmp_path)
        assert tarfile.is_tarfile(bundle_path)
        assert bundle_path.suffix == '.bz2'

    def test_bundle_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error paths in build_db_runtime_bundle()."""
        # Nonexistent project directory
        with pytest.raises(FileNotFoundError, match='does not exist'):
            build_db_runtime_bundle(Path('/nonexistent/path/xyz'))

        # Invalid [pixeltable.database] configuration (include must be a list, not a string)
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
