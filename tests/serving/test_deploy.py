"""Tests for deployment bundling (pixeltable/serving/deploy.py)."""

from __future__ import annotations

import json
import tarfile
import textwrap
from pathlib import Path

import pytest
import toml

import pixeltable as pxt
from pixeltable import metadata
from pixeltable.config import Config
from pixeltable.serving.deploy import build_deploy_bundle


class TestDeploy:
    def test_deploy_bundle(self, uses_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build a deploy bundle from a TOML config and verify its contents."""
        _ = pxt.create_table('table1', {'id': pxt.Int, 'name': pxt.String})
        _ = pxt.create_table('table2', {'id': pxt.Int, 'value': pxt.Float})

        config_path = tmp_path / 'pixeltable.toml'
        config_contents = textwrap.dedent(
            """\
            [[environment]]
            name = "test-deploy"
            include = ["*.toml", "a*.txt"]
            exclude = ["a_exclude.txt"]
            services = ["myservice1", "myservice2"]

            [[service]]
            name = "myservice1"

            [[service.routes]]
            type = "insert"
            table = "table1"
            path = "/insert"

            [[service]]
            name = "myservice2"

            [[service.routes]]
            type = "insert"
            table = "table2"
            path = "/insert"
            """
        )
        config_path.write_text(config_contents)
        (tmp_path / 'a_include.txt').write_text("I'm a random artifact in the project folder.")
        (tmp_path / 'a_exclude.txt').write_text("I'm explicitly excluded.")
        (tmp_path / 'b_exclude.txt').write_text("I'm excluded because I'm not in the includes.")

        monkeypatch.chdir(tmp_path)  # cwd until end of test

        Config.init({}, reinit=True)  # pick up the new configuration

        bundle_path = build_deploy_bundle('test-deploy')

        # Extract the bundle and verify contents
        with tarfile.open(bundle_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'config.toml' in members
            assert 'metadata.json' in members
            assert 'conda-env.yml' in members
            assert 'project/pixeltable.toml' in members
            assert 'project/a_include.txt' in members
            assert 'project/a_exclude.txt' not in members
            assert 'project/b_exclude.txt' not in members

            # Verify the contents of a_include.txt
            a_include_member = tar.getmember('project/a_include.txt')
            with tar.extractfile(a_include_member) as f:
                text = f.read().decode('utf-8')
                assert text == "I'm a random artifact in the project folder.", text

            # Verify the contents of config.toml
            config_member = tar.getmember('config.toml')
            with tar.extractfile(config_member) as f:
                content = toml.loads(f.read().decode('utf-8'))
                assert content['environment'][0]['name'] == 'test-deploy'
                assert len(content['service']) == 2
                assert content['service'][0]['name'] == 'myservice1'
                assert content['service'][0]['routes'][0]['table'] == 'table1'
                assert content['service'][1]['name'] == 'myservice2'
                assert content['service'][1]['routes'][0]['table'] == 'table2'

            # Verify the contents of metadata.json
            metadata_member = tar.getmember('metadata.json')
            with tar.extractfile(metadata_member) as f:
                content = json.loads(f.read().decode('utf-8'))
                assert content['pxt_version'] == pxt.__version__
                assert content['pxt_md_version'] == metadata.VERSION
                assert len(content['tables_md']) == 2  # 2 tables referenced in services
                assert len(content['tables_md'][0]) == 3  # TableVersionMd structure

            # Verify the contents of environment.yml
            env_member = tar.getmember('conda-env.yml')
            with tar.extractfile(env_member) as f:
                text = f.read().decode('utf-8')
                assert 'pixeltable==' in text, text
