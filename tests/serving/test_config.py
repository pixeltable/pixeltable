"""Tests for TOML-based service configuration and app creation."""

import os
import pathlib
import sys
import tempfile
import textwrap
import types
from typing import Any

import pytest
import sqlalchemy as sql
import toml

import pixeltable as pxt
from pixeltable import config
from pixeltable.serving._config import create_service_from_config, lookup_service_config
from tests.serving.test_fastapi import assert_sqlite_row, make_sqlite_target
from tests.utils import skip_test_if_not_installed


class TestConfig:
    def test_load_valid_config(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Load a valid TOML config, create an app, and exercise the routes via TestClient."""
        skip_test_if_not_installed('fastapi')
        from fastapi.testclient import TestClient

        pxt.create_dir('test_config')
        t = pxt.create_table('test_config.items', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        t.add_computed_column(name_upper=t.name.upper())

        # sqlite target for the export_sql route
        db_connect = make_sqlite_target(
            tmp_path / 'export.db', 'items_out', {'id': sql.Integer, 'name': sql.VARCHAR, 'name_upper': sql.VARCHAR}
        )

        file_contents = textwrap.dedent(
            f"""
            [[service]]
            name = "test-service"
            port = 9999

            [[service.routes]]
            type = "insert"
            table = "test_config/items"
            path = "/insert"
            inputs = ["id", "name"]
            outputs = ["id", "name", "name_upper"]

            [[service.routes]]
            type = "insert"
            table = "test_config/items"
            path = "/insert-export"
            inputs = ["id", "name"]
            outputs = ["id", "name", "name_upper"]

            [service.routes.export_sql]
            db_connect = '{db_connect}'
            table = "items_out"

            [[service.routes]]
            type = "update"
            table = "test_config/items"
            path = "/update"
            inputs = ["name"]
            outputs = ["id", "name", "name_upper"]

            [[service.routes]]
            type = "delete"
            table = "test_config/items"
            path = "/delete"
            """
        ).strip()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as fp:
            fp.write(file_contents)
            config_path = fp.name

        try:
            config.Config.get().init({}, additional_config_files=[config_path], reinit=True)
            services = config.Config.get().get_value('service', list[config.ServiceConfig])
            assert len(services) == 1
            cfg = services[0]
            assert cfg.name == 'test-service'
            assert cfg.port == 9999
            assert len(cfg.routes) == 4

            app = create_service_from_config(cfg)
            client = TestClient(app)

            # insert
            resp = client.post('/insert', json={'id': 1, 'name': 'alice'})
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data['id'] == 1
            assert data['name'] == 'alice'
            assert data['name_upper'] == 'ALICE'

            # insert with export_sql: row should also land in the sqlite target
            resp = client.post('/insert-export', json={'id': 2, 'name': 'carol'})
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'id': 2, 'name': 'carol', 'name_upper': 'CAROL'}
            assert_sqlite_row(db_connect, 'items_out', {'id': 2}, {'id': 2, 'name': 'carol', 'name_upper': 'CAROL'})

            # update: change the name, verify cascade of name_upper
            resp = client.post('/update', json={'id': 1, 'name': 'bob'})
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data['id'] == 1
            assert data['name'] == 'bob'
            assert data['name_upper'] == 'BOB'

            # update: missing row -> 404
            resp = client.post('/update', json={'id': 999, 'name': 'x'})
            assert resp.status_code == 404, resp.text

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

        query_mod = types.ModuleType('_test_query_mod')
        # we need to exec in the module's namespace so @pxt.query sees the right globals

        exec(
            textwrap.dedent("""
                import pixeltable as pxt

                t = pxt.get_table('test_config.docs')

                @pxt.query
                def search(min_id: int) -> pxt.Query:
                    return t.where(t.id >= min_id).select(t.id, t.text).order_by(t.id)
            """),
            query_mod.__dict__,
        )

        sys.modules['_test_query_mod'] = query_mod

        config_dict = {
            'service': [
                {
                    'name': 'query-service',
                    'modules': ['_test_query_mod'],
                    'routes': [{'type': 'query', 'path': '/search', 'query': '_test_query_mod.search'}],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
            toml.dump(config_dict, f)
            config_path = f.name

        try:
            config.Config.get().init({}, additional_config_files=[config_path], reinit=True)
            cfg = lookup_service_config('query-service')
            app = create_service_from_config(cfg)
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

        basic_cases: list[tuple[dict[str, Any], str]] = [
            # unknown route type (match on field name + invalid value to avoid coupling to Pydantic's exact phrasing)
            ({'routes': [{'type': 'notarealtype', 'path': '/x'}]}, r'type.*notarealtype|notarealtype.*type'),
            # insert missing table
            ({'routes': [{'type': 'insert', 'path': '/x'}]}, 'table'),
            # update missing table
            ({'routes': [{'type': 'update', 'path': '/x'}]}, 'table'),
            # query missing query
            ({'routes': [{'type': 'query', 'path': '/x'}]}, 'query'),
            # delete missing table
            ({'routes': [{'type': 'delete', 'path': '/x'}]}, 'table'),
            # extra/unknown key rejected
            ({'routes': [{'type': 'insert', 'table': 'd.t', 'path': '/x', 'typo_key': 'val'}]}, 'typo_key'),
            # path missing leading slash
            ({'routes': [{'type': 'insert', 'table': 'd.t', 'path': 'no-slash'}]}, 'path'),
            # prefix missing leading slash
            ({'prefix': 'api', 'routes': [{'type': 'insert', 'table': 'd.t', 'path': '/x'}]}, 'prefix'),
            # export_sql: unknown nested key
            (
                {
                    'routes': [
                        {
                            'type': 'insert',
                            'table': 'd.t',
                            'path': '/x',
                            'export_sql': {'db_connect': 'sqlite:///x', 'table': 'y', 'typo_key': 'z'},
                        }
                    ]
                },
                'typo_key',
            ),
            # export_sql: invalid method
            (
                {
                    'routes': [
                        {
                            'type': 'insert',
                            'table': 'd.t',
                            'path': '/x',
                            'export_sql': {'db_connect': 'sqlite:///x', 'table': 'y', 'method': 'bogus'},
                        }
                    ]
                },
                "input_value='bogus'",
            ),
            # export_sql: missing table
            (
                {
                    'routes': [
                        {'type': 'insert', 'table': 'd.t', 'path': '/x', 'export_sql': {'db_connect': 'sqlite:///x'}}
                    ]
                },
                r'export_sql\.[^\s]*table',
            ),
        ]

        # Each of the basic_cases is wrapped in a {'service': [...]} block and given the name 'test-service'. Other
        # test cases that don't fit this pattern are subsequently appended.
        cases = [
            ({'service': [{'name': 'test-service'} | config_dict]}, expected_string)
            for config_dict, expected_string in basic_cases
        ]
        cases.append(
            ({'service': [{'name': 'test-service'}, {'name': 'test-service'}]}, 'Duplicate `ServiceConfig` name')
        )

        for config_dict, expected_substring in cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8') as f:
                toml.dump(config_dict, f)
                config_path = f.name
            try:
                print(config_dict)
                with pytest.raises(pxt.Error, match=expected_substring):
                    config.Config.get().init({}, additional_config_files=[config_path], reinit=True)
                    lookup_service_config('test-service')
            finally:
                os.unlink(config_path)

    def test_create_app_errors(self) -> None:
        """create_app_from_config surfaces clear errors for module/query resolution failures."""
        skip_test_if_not_installed('fastapi')

        def _query_app(query: str) -> config.ServiceConfig:
            return config.ServiceConfig(
                name='test', routes=[config.QueryRouteConfig(type='query', path='/x', query=query)]
            )

        # query reference without a dot
        with pytest.raises(pxt.Error, match='invalid query reference'):
            create_service_from_config(_query_app('noseparator'))

        # query module not importable
        with pytest.raises(pxt.Error, match='could not import module'):
            create_service_from_config(_query_app('definitely_not_a_real_module_xyz.search'))

        # query module exists but attribute is missing
        with pytest.raises(pxt.Error, match='has no attribute'):
            create_service_from_config(_query_app('os.this_attr_does_not_exist'))

        # query resolves to something that isn't a @pxt.query
        with pytest.raises(pxt.Error, match=r'expected a @pxt\.query'):
            create_service_from_config(_query_app('os.getcwd'))

        # `modules` entry that fails to import
        bad_modules_app = config.ServiceConfig(
            name='test',
            modules=['definitely_not_a_real_module_xyz'],
            routes=[config.QueryRouteConfig(type='query', path='/x', query='os.getcwd')],
        )
        with pytest.raises(pxt.Error, match='listed in `modules`'):
            create_service_from_config(bad_modules_app)
