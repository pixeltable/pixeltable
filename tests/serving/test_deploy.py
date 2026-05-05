"""Tests for deployment bundling (pixeltable/serving/deploy.py)."""

from __future__ import annotations

import tarfile
import textwrap
from pathlib import Path

import pytest

from pixeltable import config
from pixeltable.serving.deploy import build_deploy_bundle


class TestDeploy:
    def test_deploy_bundle(self, uses_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build a deployment bundle from a TOML config and verify its contents."""
        config_path = tmp_path / 'pixeltable.toml'
        config_path.write_text(
            textwrap.dedent(
                """\
                [[deployment]]
                name = "test-deploy"
                include = ["*.toml", "a*.txt"]
                exclude = ["a_exclude.txt"]
                """
            )
        )
        (tmp_path / 'a_include.txt').write_text("I'm a random artifact in the project folder.")
        (tmp_path / 'a_exclude.txt').write_text("I'm explicitly excluded.")
        (tmp_path / 'b_exclude.txt').write_text("I'm excluded because I'm not in the includes.")

        monkeypatch.chdir(tmp_path)  # cwd until end of test

        config.Config.init({}, reinit=True)  # pick up the new configuration

        bundle_path = build_deploy_bundle('test-deploy')

        # Extract the bundle and verify contents
        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'environment.yml' in members
            assert 'project/pixeltable.toml' in members
            assert 'project/a_include.txt' in members
            assert 'project/a_exclude.txt' not in members
            assert 'project/b_exclude.txt' not in members

            # Verify the contents of a_include.txt
            a_include_member = tar.getmember('project/a_include.txt')
            with tar.extractfile(a_include_member) as f:
                content = f.read().decode('utf-8')
                assert content == "I'm a random artifact in the project folder.", content

            # Verify the contents of environment.yml
            env_member = tar.getmember('environment.yml')
            with tar.extractfile(env_member) as f:
                content = f.read().decode('utf-8')
                assert 'pixeltable==' in content, content
