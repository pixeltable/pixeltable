"""Tests for the pxt CLI (pixeltable/cli.py).

All CLI tests live here so they are easy to find as new subcommands are added.
Tests that exercise the underlying serving machinery (route resolution, FastAPI
app creation, HTTP semantics) belong in tests/serving/.
"""

from __future__ import annotations

import errno as errno_mod
import json
import pathlib
from unittest.mock import patch

import pytest
import toml

import pixeltable as pxt
from pixeltable.cli import main as cli_main
from tests.utils import skip_test_if_not_installed


def _run_cli(
    argv: list[str],
    capsys: pytest.CaptureFixture,
    *,
    exit_code: int = 0,
    stdout: str | list[str] | None = None,
    stderr: str | list[str] | None = None,
) -> None:
    """Invoke cli_main, assert exit code, and optionally assert stdout/stderr substrings."""
    actual_exit_code: int | str | None = 0
    with patch('sys.argv', argv):
        try:
            cli_main()
        except SystemExit as e:
            actual_exit_code = e.code
    assert actual_exit_code == exit_code
    if stdout is not None or stderr is not None:
        captured = capsys.readouterr()
        if stdout is not None:
            needles = [stdout] if isinstance(stdout, str) else stdout
            for needle in needles:
                assert needle in captured.out
        if stderr is not None:
            needles = [stderr] if isinstance(stderr, str) else stderr
            for needle in needles:
                assert needle in captured.err


class TestCLI:
    def test_arg_parsing(self, capsys: pytest.CaptureFixture) -> None:
        _run_cli(['pxt'], capsys, stdout='usage:')
        _run_cli(['pxt', '--version'], capsys, stdout=pxt.__version__)
        _run_cli(['pxt', 'serve', 'insert'], capsys, exit_code=2, stderr=['Examples:', '--table'])
        _run_cli(['pxt', 'serve', 'delete'], capsys, exit_code=2, stderr='Examples:')
        _run_cli(['pxt', 'serve', 'query'], capsys, exit_code=2, stderr='Examples:')
        _run_cli(['pxt', 'serve', 'config'], capsys, exit_code=2, stderr='Examples:')

    def test_dry_run(self, capsys: pytest.CaptureFixture, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('fastapi')

        # insert plain
        _run_cli(
            ['pxt', 'serve', 'insert', '--table', 'd.items', '--path', '/ins', '--dry-run'],
            capsys,
            stdout=['[insert]', '/ins'],
        )

        # --prefix appears in dry-run output
        _run_cli(
            ['pxt', 'serve', 'insert', '--table', 'd.items', '--path', '/ins', '--dry-run', '--prefix', '/api'],
            capsys,
            stdout='Prefix:',
        )

        # insert --json
        _run_cli(['pxt', 'serve', 'insert', '--table', 'd.items', '--path', '/ins', '--dry-run', '--json'], capsys)
        data = json.loads(capsys.readouterr().out)
        assert data['routes'][0]['type'] == 'insert'
        assert data['routes'][0]['path'] == '/ins'
        assert data['routes'][0]['table'] == 'd.items'

        # delete plain
        _run_cli(
            ['pxt', 'serve', 'delete', '--table', 'd.items', '--path', '/del', '--dry-run'],
            capsys,
            stdout=['[delete]', '/del'],
        )

        # query plain
        _run_cli(
            ['pxt', 'serve', 'query', '--query', 'mymod.fn', '--path', '/search', '--dry-run'],
            capsys,
            stdout=['[query]', '/search'],
        )

        # config plain and --json (from TOML file)
        config_path = tmp_path / 'service.toml'
        config_path.write_text(
            toml.dumps(
                {
                    'service': {'title': 'Dry Run Svc', 'port': 9999},
                    'routes': [{'type': 'insert', 'table': 'd.t', 'path': '/ins'}],
                }
            )
        )

        _run_cli(['pxt', 'serve', 'config', str(config_path), '--dry-run'], capsys, stdout=['Dry Run Svc', '[insert]'])

        _run_cli(['pxt', 'serve', 'config', str(config_path), '--dry-run', '--json'], capsys)
        data = json.loads(capsys.readouterr().out)
        assert data['service']['title'] == 'Dry Run Svc'
        assert data['service']['port'] == 9999
        assert data['routes'][0]['type'] == 'insert'

    def test_serve_routes(self, tmp_path: pathlib.Path) -> None:
        """CLI args are correctly wired into AppConfig and RouteConfig objects."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        from pixeltable.serving._config import (
            AppConfig,
            DeleteRouteConfig,
            InsertRouteConfig,
            QueryRouteConfig,
            ServiceConfig,
        )

        with (
            patch('pixeltable.serving._config.load_app_config') as mock_load,
            patch('pixeltable.serving._config.create_app_from_config') as mock_create,
            patch('uvicorn.run') as mock_run,
        ):
            mock_create.return_value = 'fake_app'

            # config: TOML load + --port override
            config_path = tmp_path / 'service.toml'
            config_path.write_text(
                toml.dumps({'service': {'title': 'CLI Test', 'host': '127.0.0.1', 'port': 7777}, 'routes': []})
            )
            mock_load.return_value = AppConfig(
                service=ServiceConfig(title='CLI Test', host='127.0.0.1', port=7777), routes=[]
            )
            with patch('sys.argv', ['pxt', 'serve', 'config', str(config_path), '--port', '9999']):
                cli_main()
            mock_load.assert_called_once_with(str(config_path))
            cfg = mock_create.call_args.args[0]
            assert cfg.service.port == 9999
            assert cfg.service.host == '127.0.0.1'
            assert cfg.service.title == 'CLI Test'
            mock_run.assert_called_once_with('fake_app', host='127.0.0.1', port=9999)

            mock_create.reset_mock()
            mock_run.reset_mock()

            # insert
            argv = [
                'pxt', 'serve', 'insert',
                '--table', 'd.items', '--path', '/insert',
                '--inputs', 'id', 'name',
                '--outputs', 'id', 'name', 'name_upper',
                '--background', '--port', '9000',
            ]  # fmt: skip
            with patch('sys.argv', argv):
                cli_main()
            route = mock_create.call_args.args[0].routes[0]
            assert isinstance(route, InsertRouteConfig)
            assert route.table == 'd.items'
            assert route.inputs == ['id', 'name']
            assert route.outputs == ['id', 'name', 'name_upper']
            assert route.background is True
            assert route.return_fileresponse is False
            mock_run.assert_called_once_with('fake_app', host='0.0.0.0', port=9000)

            mock_create.reset_mock()
            mock_run.reset_mock()

            # delete
            with patch(
                'sys.argv',
                ['pxt', 'serve', 'delete', '--table', 'd.items', '--path', '/delete', '--match-columns', 'id'],
            ):
                cli_main()
            route = mock_create.call_args.args[0].routes[0]
            assert isinstance(route, DeleteRouteConfig)
            assert route.table == 'd.items'
            assert route.match_columns == ['id']
            assert route.background is False

            mock_create.reset_mock()

            # query
            argv = [
                'pxt', 'serve', 'query',
                '--query', 'mymod.search', '--path', '/search',
                '--inputs', 'min_id', '--method', 'get', '--one-row',
                '--host', '127.0.0.1',
            ]  # fmt: skip
            with patch('sys.argv', argv):
                cli_main()
            cfg = mock_create.call_args.args[0]
            route = cfg.routes[0]
            assert isinstance(route, QueryRouteConfig)
            assert route.query == 'mymod.search'
            assert route.inputs == ['min_id']
            assert route.method == 'get'
            assert route.one_row is True
            assert cfg.service.host == '127.0.0.1'

    def test_serve_output(self, capsys: pytest.CaptureFixture) -> None:
        """--json startup record and EADDRINUSE error output (plain and JSON)."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')

        with patch('pixeltable.serving._config.create_app_from_config') as mock_create, patch('uvicorn.run'):
            mock_create.return_value = 'fake_app'

            # --json startup record
            _run_cli(['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins', '--port', '8080', '--json'], capsys)
            data = json.loads(capsys.readouterr().out)
            assert data['status'] == 'starting'
            assert data['port'] == 8080
            assert 'url' in data and 'docs_url' in data
            assert data['routes'] == 1

        eaddrinuse = OSError(errno_mod.EADDRINUSE, 'Address already in use')

        with (
            patch('pixeltable.serving._config.create_app_from_config') as mock_create,
            patch('uvicorn.run', side_effect=eaddrinuse),
        ):
            mock_create.return_value = 'fake_app'

            # EADDRINUSE plain
            _run_cli(
                ['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins', '--port', '8080'],
                capsys,
                exit_code=1,
                stderr='port 8080 is already in use',
            )

            # EADDRINUSE --json: caller parses the structured error record
            _run_cli(
                ['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins', '--port', '8080', '--json'],
                capsys,
                exit_code=1,
            )
            data = json.loads(capsys.readouterr().err)
            assert data['status'] == 'error'
            assert data['code'] == 'EADDRINUSE'
            assert data['port'] == 8080

        # pxt.Error plain: _emit_error writes 'pxt: error: ...' to stderr
        with patch('pixeltable.serving._config.create_app_from_config', side_effect=pxt.Error('bad config')):
            _run_cli(
                ['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins'],
                capsys,
                exit_code=1,
                stderr='pxt: error: bad config',
            )

        # pxt.Error --json: _emit_error writes a JSON error record to stderr
        with patch('pixeltable.serving._config.create_app_from_config', side_effect=pxt.Error('bad config')):
            _run_cli(['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins', '--json'], capsys, exit_code=1)
            data = json.loads(capsys.readouterr().err)
            assert data['status'] == 'error'
            assert data['message'] == 'bad config'

        # uvicorn not installed: pxt.Error with install hint
        with (
            patch('pixeltable.serving._config.create_app_from_config', return_value='fake_app'),
            patch.dict('sys.modules', {'uvicorn': None}),
        ):
            _run_cli(
                ['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins'],
                capsys,
                exit_code=1,
                stderr='fastapi[standard]',
            )

        # IPv6 host: URL is bracket-formatted in --json startup record
        with patch('pixeltable.serving._config.create_app_from_config', return_value='fake_app'), patch('uvicorn.run'):
            _run_cli(['pxt', 'serve', 'insert', '--table', 'd.t', '--path', '/ins', '--host', '::1', '--json'], capsys)
            data = json.loads(capsys.readouterr().out)
            assert data['url'] == 'http://[::1]:8000'
