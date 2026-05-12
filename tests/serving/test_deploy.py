"""Tests for deployment bundling (pixeltable/serving/deploy.py)."""

from __future__ import annotations

import json
import sys
import tarfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import toml

import pixeltable as pxt
from pixeltable import exceptions as excs, metadata
from pixeltable.config import Config
from pixeltable.serving.deploy import build_deploy_bundle

from ..utils import pxt_raises, skip_test_if_not_installed


class TestDeploy:
    def test_deploy_bundle(self, uses_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build a deploy bundle from a TOML config and verify its contents."""
        skip_test_if_not_installed('fastapi')
        from fastapi import FastAPI

        from pixeltable.serving import FastAPIRouter

        _ = pxt.create_table('table1', {'id': pxt.Int, 'name': pxt.String})
        pxt.create_dir('dir1')
        _ = pxt.create_table('dir1.table2', {'id': pxt.Int, 'value': pxt.Float})
        tbl3 = pxt.create_table('dir1.table3', {'id': pxt.Int, 'description': pxt.String})

        app = FastAPI(title='Pixeltable Test Service', version='0.42')
        router = FastAPIRouter()
        router.add_compute_route(tbl3, path='/compute3')
        app.include_router(router)

        # Monkeypatch a new module with the test service so that config can find it
        test_service_module = MagicMock()
        test_service_module.test_service = app
        monkeypatch.setitem(sys.modules, 'pxttest', test_service_module)

        config_path = tmp_path / 'pixeltable.toml'
        config_contents = textwrap.dedent(
            """\
            [[deployment]]
            name = "deploy-svc1"
            service = "myservice1"
            env = "prod"
            include = ["*.toml", "a*.txt"]
            exclude = ["a_exclude.txt"]

            [[deployment]]
            name = "deploy-svc2"
            service = "myservice2"
            env = "prod"

            [[deployment]]
            name = "deploy-code"
            service = "pxttest:test_service"
            env = "prod"

            [[service]]
            name = "myservice1"

            [[service.routes]]
            type = "compute"
            table = "table1"
            path = "/compute1"

            [[service]]
            name = "myservice2"

            [[service.routes]]
            type = "compute"
            table = "dir1.table2"
            path = "/compute2"
            """
        )
        config_path.write_text(config_contents)
        (tmp_path / 'a_include.txt').write_text("I'm a random artifact in the project folder.")
        (tmp_path / 'a_exclude.txt').write_text("I'm explicitly excluded.")
        (tmp_path / 'b_exclude.txt').write_text("I'm excluded because I'm not in the includes.")

        monkeypatch.chdir(tmp_path)  # cwd until end of test

        Config.init({}, reinit=True)  # pick up the new configuration

        # deploy-svc1: TOML-defined service with file include/exclude
        bundle_path = build_deploy_bundle('deploy-svc1')
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
                assert content['deployment'][0]['name'] == 'deploy-svc1'
                assert len(content['service']) == 1
                assert content['service'][0]['name'] == 'myservice1'
                assert content['service'][0]['routes'][0]['table'] == 'table1'

            # Verify the contents of metadata.json
            metadata_member = tar.getmember('metadata.json')
            with tar.extractfile(metadata_member) as f:
                content = json.loads(f.read().decode('utf-8'))
                assert content['pxt_version'] == pxt.__version__
                assert content['pxt_md_version'] == metadata.VERSION
                assert len(content['tables_md']) == 1
                assert len(content['tables_md'][0]) == 3  # TableVersionMd structure

            # Verify the contents of conda-env.yml
            env_member = tar.getmember('conda-env.yml')
            with tar.extractfile(env_member) as f:
                text = f.read().decode('utf-8')
                assert 'pixeltable==' in text, text

        # deploy-svc2: second TOML-defined service
        bundle_path = build_deploy_bundle('deploy-svc2')
        with tarfile.open(bundle_path, 'r:bz2') as tar:
            config_member = tar.getmember('config.toml')
            with tar.extractfile(config_member) as f:
                content = toml.loads(f.read().decode('utf-8'))
                assert content['deployment'][0]['name'] == 'deploy-svc2'
                assert len(content['service']) == 1
                assert content['service'][0]['name'] == 'myservice2'
                assert content['service'][0]['routes'][0]['table'] == 'dir1.table2'
            metadata_member = tar.getmember('metadata.json')
            with tar.extractfile(metadata_member) as f:
                content = json.loads(f.read().decode('utf-8'))
                assert len(content['tables_md']) == 1

        # deploy-code: code-defined service (pxttest:test_service)
        bundle_path = build_deploy_bundle('deploy-code')
        with tarfile.open(bundle_path, 'r:bz2') as tar:
            config_member = tar.getmember('config.toml')
            with tar.extractfile(config_member) as f:
                content = toml.loads(f.read().decode('utf-8'))
                assert content['deployment'][0]['name'] == 'deploy-code'
                assert 'service' not in content or len(content.get('service', [])) == 0
            metadata_member = tar.getmember('metadata.json')
            with tar.extractfile(metadata_member) as f:
                content = json.loads(f.read().decode('utf-8'))
                assert len(content['tables_md']) == 1

    def test_deploy_bundle_errors(self, uses_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error paths in build_deploy_bundle()."""
        skip_test_if_not_installed('fastapi')
        from fastapi import FastAPI

        from pixeltable.serving import FastAPIRouter

        config_path = tmp_path / 'pixeltable.toml'
        monkeypatch.chdir(tmp_path)

        # Invalid deployment name
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "123bad"
            service = "x"
            env = "prod"
            """)
        )
        with pytest.raises(pxt.Error, match='not a valid Pixeltable identifier'):
            Config.init({}, reinit=True)

        # No deployments configured
        config_path.write_text('')
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.DEPLOYMENT_NOT_FOUND, match='No deployments found'):
            build_deploy_bundle('nonexistent')

        # Deployment name not found among configured deployments
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "other-env"
            service = "x"
            env = "prod"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.DEPLOYMENT_NOT_FOUND, match="Deployment 'nonexistent' not found"):
            build_deploy_bundle('nonexistent')

        # Service referenced by deployment not found (no services configured)
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "missing-service"
            env = "prod"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.SERVICE_NOT_FOUND, match='No services found'):
            build_deploy_bundle('my-env')

        # Service referenced by deployment not found (different service configured)
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "missing-service"
            env = "prod"

            [[service]]
            name = "other-service"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.SERVICE_NOT_FOUND, match="Service 'missing-service' not found"):
            build_deploy_bundle('my-env')

        # TOML-defined service: route type is not 'compute' (e.g. 'insert')
        _ = pxt.create_table('deploy_err_toml_tbl', {'id': pxt.Int})
        for route_type in ('insert', 'update', 'delete'):
            config_path.write_text(
                textwrap.dedent(f"""\
                [[deployment]]
                name = "my-env"
                service = "my-service"
                env = "prod"

                [[service]]
                name = "my-service"

                [[service.routes]]
                type = "{route_type}"
                table = "deploy_err_toml_tbl"
                path = "/invalid"
                """)
            )
            Config.init({}, reinit=True)
            with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match="only 'compute' routes are supported"):
                build_deploy_bundle('my-env')

        # Table referenced in route does not exist
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "my-service"
            env = "prod"

            [[service]]
            name = "my-service"

            [[service.routes]]
            type = "compute"
            table = "no_such_table"
            path = "/compute"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.PATH_NOT_FOUND, match='no_such_table'):
            build_deploy_bundle('my-env')

        # Code-defined service: module cannot be imported
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "no_such_module:app"
            env = "prod"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='Could not import module'):
            build_deploy_bundle('my-env')

        # Code-defined service: module exists but attribute does not
        skip_test_if_not_installed('fastapi')
        test_module = MagicMock(spec=[])  # spec=[] means no attributes
        monkeypatch.setitem(sys.modules, 'pxttest_noattr', test_module)
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "pxttest_noattr:missing_app"
            env = "prod"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='has no attribute'):
            build_deploy_bundle('my-env')

        # Code-defined service: attribute exists but is not a FastAPI app
        test_module_bad = MagicMock()
        test_module_bad.not_a_fastapi = 'just a string'
        monkeypatch.setitem(sys.modules, 'pxttest_bad', test_module_bad)
        config_path.write_text(
            textwrap.dedent("""\
            [[deployment]]
            name = "my-env"
            service = "pxttest_bad:not_a_fastapi"
            env = "prod"
            """)
        )
        Config.init({}, reinit=True)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='is not a FastAPI app'):
            build_deploy_bundle('my-env')

        # Code-defined service: route is not a 'compute' route (e.g. 'insert')
        t = pxt.create_table('deploy_err_tbl', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        for route_type in ('insert', 'update', 'delete'):
            insert_app = FastAPI()
            insert_router = FastAPIRouter()
            add_route_fn = getattr(insert_router, f'add_{route_type}_route')
            add_route_fn(t, path='/invalid')
            insert_app.include_router(insert_router)
            test_module = MagicMock()
            test_module.app = insert_app
            monkeypatch.setitem(sys.modules, 'pxttest', test_module)
            config_path.write_text(
                textwrap.dedent("""\
                [[deployment]]
                name = "my-env"
                service = "pxttest:app"
                env = "prod"
                """)
            )
            Config.init({}, reinit=True)
            with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match="only 'compute' routes are supported"):
                build_deploy_bundle('my-env')
