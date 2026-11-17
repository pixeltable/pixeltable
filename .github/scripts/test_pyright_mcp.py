"""Tests for pyright_mcp.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'

The navigation tests start a real pyright-langserver and are skipped when it is not on PATH.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyright_mcp as mcp

STORE = """\
class Store:
    def insert(self, row: dict) -> None:
        pass


class Log:
    def insert(self, line: str) -> None:
        pass
"""

USER = """\
from store import Log, Store


def save(store: Store) -> None:
    store.insert({'a': 1})


def record(log: Log) -> None:
    log.insert('x')
    [].insert(0, 1)
"""


def call(nav: mcp.Navigator, name: str, **args: Any) -> dict[str, Any]:
    response = mcp.handle(
        nav, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}}
    )
    assert response is not None
    result: dict[str, Any] = response['result']
    return result


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve()
        (self.root / 'store.py').write_text(STORE)
        # No language server starts unless a tool needs one.
        self.nav = mcp.Navigator(self.root, ['false'])

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_initialize_and_list(self) -> None:
        init = mcp.handle(
            self.nav, {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': 'x'}}
        )
        assert init is not None
        self.assertEqual(init['result']['protocolVersion'], 'x')
        listed = mcp.handle(self.nav, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
        assert listed is not None
        names = {t['name'] for t in listed['result']['tools']}
        self.assertIn('find_references', names)
        # Navigation only: no tool may edit files or run code.
        self.assertFalse(names & {'edit_file', 'rename_symbol', 'execute'})

    def test_notification_gets_no_response(self) -> None:
        self.assertIsNone(mcp.handle(self.nav, {'jsonrpc': '2.0', 'method': 'notifications/initialized'}))

    def test_unknown_method(self) -> None:
        response = mcp.handle(self.nav, {'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list'})
        assert response is not None
        self.assertEqual(response['error']['code'], -32601)

    def test_input_errors_are_tool_errors(self) -> None:
        cases: list[tuple[str, dict[str, Any], str]] = [
            ('find_references', {'path': '../etc/passwd', 'line': 1, 'symbol': 'x'}, 'outside the workspace'),
            ('find_references', {'path': 'missing.py', 'line': 1, 'symbol': 'x'}, 'not a file'),
            ('find_references', {'path': 'store.py', 'line': 99, 'symbol': 'insert'}, 'does not exist'),
            ('find_references', {'path': 'store.py', 'line': 1, 'symbol': 'insert'}, 'does not occur'),
            ('find_references', {'path': 'store.py'}, 'missing'),
            ('delete_everything', {}, 'unknown tool'),
        ]
        for name, args, message in cases:
            with self.subTest(name=name, args=args):
                result = call(self.nav, name, **args)
                self.assertTrue(result.get('isError'))
                self.assertIn(message, result['content'][0]['text'])

    def test_serve_skips_bad_lines(self) -> None:
        stdin = ['\n', 'not json\n', json.dumps({'jsonrpc': '2.0', 'id': 7, 'method': 'ping'}) + '\n']
        out: list[str] = []

        class Sink:
            def write(self, s: str) -> None:
                out.append(s)

            def flush(self) -> None:
                pass

        mcp.serve(self.nav, iter(stdin), Sink())  # type: ignore[arg-type]
        self.assertEqual([json.loads(line) for line in out], [{'jsonrpc': '2.0', 'id': 7, 'result': {}}])


@unittest.skipUnless(shutil.which('pyright-langserver'), 'pyright-langserver is not installed')
class NavigationTest(unittest.TestCase):
    dir: tempfile.TemporaryDirectory[str]
    root: Path
    nav: mcp.Navigator

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.dir.name).resolve()
        (cls.root / 'store.py').write_text(STORE)
        (cls.root / 'user.py').write_text(USER)
        cls.nav = mcp.Navigator(cls.root, ['pyright-langserver', '--stdio'])

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.nav._client is not None:
            cls.nav._client.proc.kill()
        cls.dir.cleanup()

    def text(self, name: str, **args: Any) -> str:
        result = call(self.nav, name, **args)
        self.assertFalse(result.get('isError'), result)
        return str(result['content'][0]['text'])

    def test_references_resolve_by_type(self) -> None:
        # Store.insert has one caller. Log.insert and list.insert share the name and must not appear.
        text = self.text('find_references', path='store.py', line=2, symbol='insert')
        self.assertIn('user.py:5:', text)
        self.assertNotIn('user.py:9:', text)
        self.assertNotIn('user.py:10:', text)

    def test_incoming_calls(self) -> None:
        text = self.text('incoming_calls', path='store.py', line=7, symbol='Log.insert')
        self.assertIn('record', text)
        self.assertNotIn('save', text)

    def test_definition_and_symbols(self) -> None:
        self.assertIn('store.py:2', self.text('find_definition', path='user.py', line=5, symbol='insert'))
        self.assertIn('class Store', self.text('document_symbols', path='store.py'))
        self.assertIn('store.py:1', self.text('workspace_symbols', query='Store'))


if __name__ == '__main__':
    unittest.main()
