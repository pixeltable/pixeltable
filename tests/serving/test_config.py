"""Tests for TOML-based service configuration and app creation."""

import os
import tempfile
from typing import Any

import pytest
import toml

import pixeltable as pxt
from pixeltable.serving._config import (
    AppConfig,
    QueryRouteConfig,
    ServiceConfig,
    create_app_from_config,
    load_app_config,
)
from tests.utils import skip_test_if_not_installed


class TestConfig:
    def test_load_valid_config(self, uses_db: None) -> None:
        """Load a valid TOML config, create an app, and exercise the routes via TestClient."""
        skip_test_if_not_installed('fastapi')
        from fastapi.testclient import TestClient

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
