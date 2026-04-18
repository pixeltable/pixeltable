"""Tests for TOML-based service configuration and app creation."""

import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
import toml

import pixeltable as pxt
from tests.utils import skip_test_if_not_installed


class TestConfig:
    def test_load_valid_config(self, uses_db: None) -> None:
        """Load a valid TOML config, create an app, and exercise the routes via TestClient."""
        skip_test_if_not_installed('fastapi')
        from fastapi.testclient import TestClient

        from pixeltable.serving._config import create_app_from_config, load_app_config

        pxt.create_dir('test_config')
        t = pxt.create_table('test_config.items', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        t.add_computed_column(name_upper=t.name.upper())

        config_dict = {
            'service': {'title': 'Test Service', 'port': 9999},
            'routes': [
                {
                    'type': 'insert',
                    'table': 'test_config.items',
                    'path': '/insert',
                    'inputs': ['id', 'name'],
                    'outputs': ['id', 'name', 'name_upper'],
                },
                {'type': 'delete', 'table': 'test_config.items', 'path': '/delete'},
            ],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
            toml.dump(config_dict, f)
            config_path = f.name

        try:
            config = load_app_config(config_path)
            assert config.service.title == 'Test Service'
            assert config.service.port == 9999
            assert len(config.routes) == 2

            app = create_app_from_config(config)
            client = TestClient(app)

            # insert
            resp = client.post('/insert', json={'id': 1, 'name': 'alice'})
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data['id'] == 1
            assert data['name'] == 'alice'
            assert data['name_upper'] == 'ALICE'

            # delete
            resp = client.post('/delete', json={'id': 1})
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'num_rows': 1}

            # delete again: 0 rows
            resp = client.post('/delete', json={'id': 1})
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'num_rows': 0}
        finally:
            os.unlink(config_path)

    def test_query_route_from_config(self, uses_db: None) -> None:
        """Query route resolves a dotted-path reference to a @pxt.query function."""
        skip_test_if_not_installed('fastapi')
        from fastapi.testclient import TestClient

        from pixeltable.serving._config import create_app_from_config, load_app_config

        pxt.create_dir('test_config')
        t = pxt.create_table('test_config.docs', {'id': pxt.Required[pxt.Int], 'text': pxt.String}, primary_key='id')
        t.insert([{'id': 1, 'text': 'hello'}, {'id': 2, 'text': 'world'}])

        # define a query function in a temporary module
        import types

        query_mod = types.ModuleType('_test_query_mod')
        # we need to exec in the module's namespace so @pxt.query sees the right globals
        exec(
            """
import pixeltable as pxt

t = pxt.get_table('test_config.docs')

@pxt.query
def search(min_id: int) -> pxt.Query:
    return t.where(t.id >= min_id).select(t.id, t.text).order_by(t.id)
""",
            query_mod.__dict__,
        )
        import sys

        sys.modules['_test_query_mod'] = query_mod

        try:
            config_dict = {
                'modules': ['_test_query_mod'],
                'routes': [{'type': 'query', 'path': '/search', 'query': '_test_query_mod.search'}],
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
                toml.dump(config_dict, f)
                config_path = f.name

            config = load_app_config(config_path)
            app = create_app_from_config(config)
            client = TestClient(app)

            resp = client.post('/search', json={'min_id': 2})
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'rows': [{'id': 2, 'text': 'world'}]}
        finally:
            os.unlink(config_path)
            del sys.modules['_test_query_mod']

    def test_validation_errors(self) -> None:
        """Invalid TOML configs produce clear pxt.Error messages."""
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving._config import load_app_config

        cases: list[tuple[dict[str, Any], str]] = [
            # missing routes
            ({}, 'routes'),
            # unknown route type (match on field name + invalid value to avoid coupling to Pydantic's exact phrasing)
            ({'routes': [{'type': 'update', 'path': '/x'}]}, r'type.*update|update.*type'),
            # insert missing table
            ({'routes': [{'type': 'insert', 'path': '/x'}]}, 'table'),
            # query missing query
            ({'routes': [{'type': 'query', 'path': '/x'}]}, 'query'),
            # delete missing table
            ({'routes': [{'type': 'delete', 'path': '/x'}]}, 'table'),
        ]

        for config_dict, expected_substring in cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
                toml.dump(config_dict, f)
                config_path = f.name
            try:
                with pytest.raises(pxt.Error, match=expected_substring):
                    load_app_config(config_path)
            finally:
                os.unlink(config_path)

    def test_create_app_errors(self) -> None:
        """create_app_from_config surfaces clear errors for module/query resolution failures."""
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving._config import AppConfig, QueryRouteConfig, ServiceConfig, create_app_from_config

        def _query_app(query: str) -> AppConfig:
            return AppConfig(service=ServiceConfig(), routes=[QueryRouteConfig(type='query', path='/x', query=query)])

        # query reference without a dot
        with pytest.raises(pxt.Error, match='invalid query reference'):
            create_app_from_config(_query_app('noseparator'))

        # query module not importable
        with pytest.raises(pxt.Error, match='could not import module'):
            create_app_from_config(_query_app('definitely_not_a_real_module_xyz.search'))

        # query module exists but attribute is missing
        with pytest.raises(pxt.Error, match='has no attribute'):
            create_app_from_config(_query_app('os.this_attr_does_not_exist'))

        # query resolves to something that isn't a @pxt.query
        with pytest.raises(pxt.Error, match=r'expected a @pxt\.query'):
            create_app_from_config(_query_app('os.getcwd'))

        # `modules` entry that fails to import
        bad_modules_app = AppConfig(
            service=ServiceConfig(),
            modules=['definitely_not_a_real_module_xyz'],
            routes=[QueryRouteConfig(type='query', path='/x', query='os.getcwd')],
        )
        with pytest.raises(pxt.Error, match='listed in `modules`'):
            create_app_from_config(bad_modules_app)

    def test_cli_serve_config(self) -> None:
        """`pxt serve config <path>` loads TOML and applies --port override."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        config_dict = {
            'service': {'title': 'CLI Test', 'host': '127.0.0.1', 'port': 7777},
            'routes': [{'type': 'insert', 'table': 'dummy.t', 'path': '/insert'}],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
            toml.dump(config_dict, f)
            config_path = f.name

        try:
            with (
                patch('pixeltable.serving._config.load_app_config') as mock_load,
                patch('pixeltable.serving._config.create_app_from_config') as mock_create,
                patch('uvicorn.run') as mock_run,
            ):
                from pixeltable.serving._config import AppConfig, ServiceConfig

                mock_load.return_value = AppConfig(
                    service=ServiceConfig(title='CLI Test', host='127.0.0.1', port=7777), routes=[]
                )
                mock_create.return_value = 'fake_app'

                from pixeltable.cli import main

                with patch('sys.argv', ['pxt', 'serve', 'config', config_path, '--port', '9999']):
                    main()

                mock_load.assert_called_once_with(config_path)
                passed_config = mock_create.call_args.args[0]
                assert passed_config.service.port == 9999
                assert passed_config.service.host == '127.0.0.1'
                assert passed_config.service.title == 'CLI Test'
                mock_run.assert_called_once_with('fake_app', host='127.0.0.1', port=9999)
        finally:
            os.unlink(config_path)

    def test_cli_serve_insert(self) -> None:
        """`pxt serve insert ...` builds a single-route AppConfig from cmdline args."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        with (
            patch('pixeltable.serving._config.create_app_from_config') as mock_create,
            patch('uvicorn.run') as mock_run,
        ):
            from pixeltable.serving._config import InsertRouteConfig

            mock_create.return_value = 'fake_app'
            from pixeltable.cli import main

            argv = [
                'pxt', 'serve', 'insert',
                '--table', 'd.items', '--path', '/insert',
                '--inputs', 'id', 'name',
                '--outputs', 'id', 'name', 'name_upper',
                '--background',
                '--port', '9000',
            ]  # fmt: skip
            with patch('sys.argv', argv):
                main()

            passed_config = mock_create.call_args.args[0]
            assert len(passed_config.routes) == 1
            route = passed_config.routes[0]
            assert isinstance(route, InsertRouteConfig)
            assert route.table == 'd.items'
            assert route.path == '/insert'
            assert route.inputs == ['id', 'name']
            assert route.outputs == ['id', 'name', 'name_upper']
            assert route.background is True
            assert route.return_fileresponse is False
            assert passed_config.service.port == 9000
            mock_run.assert_called_once_with('fake_app', host='0.0.0.0', port=9000)

    def test_cli_serve_delete(self) -> None:
        """`pxt serve delete ...` builds a single delete-route AppConfig from cmdline args."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        with patch('pixeltable.serving._config.create_app_from_config') as mock_create, patch('uvicorn.run'):
            from pixeltable.serving._config import DeleteRouteConfig

            mock_create.return_value = 'fake_app'
            from pixeltable.cli import main

            argv = [
                'pxt', 'serve', 'delete',
                '--table', 'd.items', '--path', '/delete',
                '--match-columns', 'id',
            ]  # fmt: skip
            with patch('sys.argv', argv):
                main()

            route = mock_create.call_args.args[0].routes[0]
            assert isinstance(route, DeleteRouteConfig)
            assert route.table == 'd.items'
            assert route.path == '/delete'
            assert route.match_columns == ['id']
            assert route.background is False

    def test_cli_serve_query(self) -> None:
        """`pxt serve query ...` builds a single query-route AppConfig from cmdline args."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        with patch('pixeltable.serving._config.create_app_from_config') as mock_create, patch('uvicorn.run'):
            from pixeltable.serving._config import QueryRouteConfig

            mock_create.return_value = 'fake_app'
            from pixeltable.cli import main

            argv = [
                'pxt', 'serve', 'query',
                '--query', 'mymod.search', '--path', '/search',
                '--inputs', 'min_id',
                '--method', 'get',
                '--one-row',
                '--host', '127.0.0.1',
            ]  # fmt: skip
            with patch('sys.argv', argv):
                main()

            passed_config = mock_create.call_args.args[0]
            route = passed_config.routes[0]
            assert isinstance(route, QueryRouteConfig)
            assert route.query == 'mymod.search'
            assert route.path == '/search'
            assert route.inputs == ['min_id']
            assert route.method == 'get'
            assert route.one_row is True
            assert passed_config.service.host == '127.0.0.1'
