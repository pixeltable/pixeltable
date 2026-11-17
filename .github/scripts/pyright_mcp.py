"""A read-only MCP server that answers code navigation questions with pyright.

The risk-assessment agent uses it to find the callers of changed code, which text search cannot do reliably for common
method names such as `insert` or `update`. It speaks MCP (newline-delimited JSON-RPC) on stdin/stdout and drives
`pyright-langserver --stdio` as a Language Server Protocol client. It uses only the standard library, and it has no
tool that edits files or runs code: pyright analyzes source statically.

Usage: python3 pyright_mcp.py --root <workspace> [--langserver <pyright-langserver command>]

Every tool takes paths relative to the workspace root and 1-based line numbers. A position is given as a line plus
the symbol name on that line, because the agent reads line numbers, not columns.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import IO, Any

REQUEST_TIMEOUT_S = 300.0
STARTUP_TIMEOUT_S = 120.0
MAX_RESULTS = 200
WORKSPACE_LOADED = re.compile(r'(Found \d+ source files?|No source files found)')

# Symbol kinds, from the LSP specification.
SYMBOL_KINDS = {
    1: 'file',
    2: 'module',
    3: 'namespace',
    4: 'package',
    5: 'class',
    6: 'method',
    7: 'property',
    8: 'field',
    9: 'constructor',
    10: 'enum',
    11: 'interface',
    12: 'function',
    13: 'variable',
    14: 'constant',
    22: 'enum member',
    26: 'type parameter',
}


class ToolError(Exception):
    """An error caused by the tool's input. It is returned to the agent as a tool result with isError set."""


class LspClient:
    """A minimal LSP client: one language server process, requests answered by a reader thread."""

    def __init__(self, command: list[str], root: Path) -> None:
        self.root = root
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=root
        )
        self.next_id = 0
        self.lock = threading.Lock()
        self.pending: dict[int, tuple[threading.Event, list[Any]]] = {}
        self.opened: set[str] = set()
        # Pyright logs "Found <n> source files" once it has enumerated the workspace. A references request sent earlier
        # searches only the files already open and silently returns too few results.
        self.workspace_loaded = threading.Event()
        threading.Thread(target=self._read_loop, daemon=True).start()
        root_uri = root.as_uri()
        self.request(
            'initialize',
            {
                'processId': os.getpid(),
                'rootUri': root_uri,
                'workspaceFolders': [{'uri': root_uri, 'name': root.name}],
                'capabilities': {
                    'textDocument': {
                        'definition': {},
                        'references': {},
                        'hover': {'contentFormat': ['plaintext', 'markdown']},
                        'callHierarchy': {},
                        'documentSymbol': {'hierarchicalDocumentSymbolSupport': True},
                    },
                    'workspace': {'symbol': {}, 'configuration': True, 'workspaceFolders': True},
                },
            },
        )
        self.notify('initialized', {})
        if not self.workspace_loaded.wait(STARTUP_TIMEOUT_S):
            raise ToolError(f'pyright did not load the workspace within {STARTUP_TIMEOUT_S:.0f}s')

    def _send(self, message: dict[str, Any]) -> None:
        body = json.dumps(message).encode()
        assert self.proc.stdin is not None
        with self.lock:
            self.proc.stdin.write(f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            self.proc.stdin.flush()

    def _read_loop(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        while True:
            length = None
            while True:
                header = stdout.readline()
                if not header:
                    self._fail_pending('the language server exited')
                    return
                header_text = header.decode('ascii', errors='replace').strip()
                if not header_text:
                    break
                if header_text.lower().startswith('content-length:'):
                    length = int(header_text.split(':', 1)[1])
            if length is None:
                continue
            message = json.loads(stdout.read(length))
            if 'method' in message and 'id' in message:
                self._answer_server_request(message)
            elif 'id' in message:
                entry = self.pending.pop(message['id'], None)
                if entry is not None:
                    entry[1].append(message)
                    entry[0].set()
            elif message.get('method') == 'window/logMessage' and WORKSPACE_LOADED.match(
                message['params'].get('message', '')
            ):
                self.workspace_loaded.set()
            # Other notifications (diagnostics, progress) are not needed.

    def _answer_server_request(self, message: dict[str, Any]) -> None:
        result: Any = None
        if message['method'] == 'workspace/configuration':
            # One answer per requested section. Name this interpreter, so that pyright never looks for one in the
            # workspace. Pyright reads the workspace's pyrightconfig.json for everything else.
            result = [
                {'pythonPath': sys.executable} if item.get('section') == 'python' else None
                for item in message['params'].get('items', [])
            ]
        elif message['method'] == 'workspace/workspaceFolders':
            result = [{'uri': self.root.as_uri(), 'name': self.root.name}]
        self._send({'jsonrpc': '2.0', 'id': message['id'], 'result': result})

    def _fail_pending(self, reason: str) -> None:
        for event, box in list(self.pending.values()):
            box.append({'error': {'message': reason}})
            event.set()
        self.pending.clear()

    def notify(self, method: str, params: Any) -> None:
        self._send({'jsonrpc': '2.0', 'method': method, 'params': params})

    def request(self, method: str, params: Any) -> Any:
        event = threading.Event()
        box: list[Any] = []
        with self.lock:
            self.next_id += 1
            request_id = self.next_id
        self.pending[request_id] = (event, box)
        self._send({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
        if not event.wait(REQUEST_TIMEOUT_S):
            self.pending.pop(request_id, None)
            raise ToolError(f'{method} timed out after {REQUEST_TIMEOUT_S:.0f}s')
        response = box[0]
        if 'error' in response:
            raise ToolError(f'{method} failed: {response["error"].get("message")}')
        return response.get('result')

    def open(self, path: Path) -> str:
        uri = path.as_uri()
        if uri not in self.opened:
            text = path.read_text(encoding='utf-8', errors='replace')
            self.notify(
                'textDocument/didOpen',
                {'textDocument': {'uri': uri, 'languageId': 'python', 'version': 1, 'text': text}},
            )
            self.opened.add(uri)
        return uri


class Navigator:
    """Implements the MCP tools on top of an LspClient."""

    def __init__(self, root: Path, langserver: list[str]) -> None:
        self.root = root
        self.langserver = langserver
        self._client: LspClient | None = None

    @property
    def client(self) -> LspClient:
        # Start the language server on first use, so that a run that never navigates does not pay for it.
        if self._client is None:
            self._client = LspClient(self.langserver, self.root)
        return self._client

    def resolve(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if not path.is_relative_to(self.root):
            raise ToolError(f'{rel_path} is outside the workspace')
        if not path.is_file():
            raise ToolError(f'{rel_path} is not a file in the workspace')
        return path

    def position(self, rel_path: str, line: int, symbol: str) -> tuple[str, dict[str, int]]:
        path = self.resolve(rel_path)
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        if not 1 <= line <= len(lines):
            raise ToolError(f'{rel_path} has {len(lines)} lines; line {line} does not exist')
        name = symbol.rsplit('.', 1)[-1]
        match = re.search(rf'(?<![\w]){re.escape(name)}(?![\w])', lines[line - 1])
        if match is None:
            raise ToolError(f'{name!r} does not occur on {rel_path}:{line}: {lines[line - 1].strip()}')
        return self.client.open(path), {'line': line - 1, 'character': match.start()}

    def location(self, uri: str, line0: int) -> str:
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(uri).path))
        try:
            shown = str(path.relative_to(self.root))
        except ValueError:
            shown = f'{path} (outside the workspace)'
        return f'{shown}:{line0 + 1}'

    def source_line(self, uri: str, line0: int) -> str:
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(uri).path))
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return ''
        return lines[line0].strip() if 0 <= line0 < len(lines) else ''

    def format_locations(self, locations: list[dict[str, Any]], what: str) -> str:
        if not locations:
            return f'No {what} found.'
        rows = sorted({(loc['uri'], loc['range']['start']['line']) for loc in locations}, key=lambda r: (r[0], r[1]))
        out = [f'{len(rows)} {what}:']
        out.extend(f'{self.location(uri, line0)}: {self.source_line(uri, line0)}' for uri, line0 in rows[:MAX_RESULTS])
        if len(rows) > MAX_RESULTS:
            out.append(f'... and {len(rows) - MAX_RESULTS} more')
        return '\n'.join(out)

    def find_definition(self, path: str, line: int, symbol: str) -> str:
        uri, pos = self.position(path, line, symbol)
        result = self.client.request('textDocument/definition', {'textDocument': {'uri': uri}, 'position': pos})
        locations = result if isinstance(result, list) else [result] if result else []
        locations = [
            {'uri': loc.get('targetUri', loc.get('uri')), 'range': loc.get('targetRange', loc.get('range'))}
            for loc in locations
        ]
        return self.format_locations(locations, 'definitions')

    def find_references(self, path: str, line: int, symbol: str, include_declaration: bool = False) -> str:
        uri, pos = self.position(path, line, symbol)
        result = self.client.request(
            'textDocument/references',
            {'textDocument': {'uri': uri}, 'position': pos, 'context': {'includeDeclaration': include_declaration}},
        )
        return self.format_locations(result or [], 'references')

    def incoming_calls(self, path: str, line: int, symbol: str) -> str:
        uri, pos = self.position(path, line, symbol)
        items = self.client.request(
            'textDocument/prepareCallHierarchy', {'textDocument': {'uri': uri}, 'position': pos}
        )
        if not items:
            return f'{symbol} at {path}:{line} is not a function or method that pyright can trace.'
        out: list[str] = []
        for item in items:
            calls = self.client.request('callHierarchy/incomingCalls', {'item': item}) or []
            out.append(f'{len(calls)} callers of {item["name"]}:')
            for call in calls[:MAX_RESULTS]:
                caller = call['from']
                at = ', '.join(str(n) for n in sorted({r['start']['line'] + 1 for r in call.get('fromRanges', [])}))
                where = self.location(caller['uri'], caller['selectionRange']['start']['line'])
                detail = f' ({caller["detail"]})' if caller.get('detail') else ''
                out.append(f'{caller["name"]}{detail} defined at {where}, calls on lines {at}')
            if len(calls) > MAX_RESULTS:
                out.append(f'... and {len(calls) - MAX_RESULTS} more')
        return '\n'.join(out)

    def hover(self, path: str, line: int, symbol: str) -> str:
        uri, pos = self.position(path, line, symbol)
        result = self.client.request('textDocument/hover', {'textDocument': {'uri': uri}, 'position': pos})
        if not result:
            return 'No type information.'
        contents = result.get('contents')
        if isinstance(contents, dict):
            return str(contents.get('value', ''))
        if isinstance(contents, list):
            return '\n'.join(c if isinstance(c, str) else str(c.get('value', '')) for c in contents)
        return str(contents)

    def document_symbols(self, path: str) -> str:
        uri = self.client.open(self.resolve(path))
        result = self.client.request('textDocument/documentSymbol', {'textDocument': {'uri': uri}}) or []
        out: list[str] = []

        def walk(symbols: list[dict[str, Any]], depth: int) -> None:
            for s in symbols:
                rng = s.get('range') or s['location']['range']
                kind = SYMBOL_KINDS.get(s.get('kind', 0), 'symbol')
                if kind in ('variable', 'constant', 'field', 'property') and depth > 1:
                    continue
                out.append(
                    f'{"  " * depth}{kind} {s["name"]}: lines {rng["start"]["line"] + 1}-{rng["end"]["line"] + 1}'
                )
                walk(s.get('children', []), depth + 1)

        walk(result, 0)
        return '\n'.join(out) or 'No symbols.'

    def workspace_symbols(self, query: str) -> str:
        result = self.client.request('workspace/symbol', {'query': query}) or []
        rows = []
        for s in result:
            if 'range' not in s.get('location', {}):
                continue
            kind = SYMBOL_KINDS.get(s.get('kind', 0), 'symbol')
            container = f'{s["containerName"]}.' if s.get('containerName') else ''
            where = self.location(s['location']['uri'], s['location']['range']['start']['line'])
            rows.append(f'{kind} {container}{s["name"]}: {where}')
        if not rows:
            return f'No symbols match {query!r}.'
        extra = [f'... and {len(rows) - MAX_RESULTS} more'] if len(rows) > MAX_RESULTS else []
        return '\n'.join([f'{len(rows)} symbols:', *rows[:MAX_RESULTS], *extra])


POSITION_PROPERTIES = {
    'path': {'type': 'string', 'description': 'File path relative to the workspace root.'},
    'line': {'type': 'integer', 'description': '1-based line number where the symbol occurs.'},
    'symbol': {'type': 'string', 'description': 'The name on that line, such as insert or TableVersion.insert.'},
}

TOOLS = [
    {
        'name': 'find_references',
        'description': (
            'Find every reference to the symbol at a position, resolved by type, not by name. Use it for callers of a '
            'changed function, method, class or attribute. Give the line of its definition or of any use.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                **POSITION_PROPERTIES,
                'include_declaration': {'type': 'boolean', 'description': 'Also list the declaration.'},
            },
            'required': ['path', 'line', 'symbol'],
        },
    },
    {
        'name': 'incoming_calls',
        'description': (
            'List the functions and methods that call the function or method at a position, with the lines of the '
            'calls. Call it again on a caller to walk up the call hierarchy.'
        ),
        'inputSchema': {'type': 'object', 'properties': POSITION_PROPERTIES, 'required': ['path', 'line', 'symbol']},
    },
    {
        'name': 'find_definition',
        'description': 'Find where the symbol at a position is defined.',
        'inputSchema': {'type': 'object', 'properties': POSITION_PROPERTIES, 'required': ['path', 'line', 'symbol']},
    },
    {
        'name': 'hover',
        'description': 'Show the inferred type and signature of the symbol at a position.',
        'inputSchema': {'type': 'object', 'properties': POSITION_PROPERTIES, 'required': ['path', 'line', 'symbol']},
    },
    {
        'name': 'document_symbols',
        'description': 'List the classes, functions and methods in a file with their line ranges.',
        'inputSchema': {'type': 'object', 'properties': {'path': POSITION_PROPERTIES['path']}, 'required': ['path']},
    },
    {
        'name': 'workspace_symbols',
        'description': 'Find classes, functions and methods in the workspace by name.',
        'inputSchema': {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'A name or part of a name.'}},
            'required': ['query'],
        },
    },
]


def handle(nav: Navigator, message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get('method')
    if 'id' not in message:
        return None  # A notification, such as notifications/initialized.
    response: dict[str, Any] = {'jsonrpc': '2.0', 'id': message['id']}
    if method == 'initialize':
        response['result'] = {
            'protocolVersion': message.get('params', {}).get('protocolVersion', '2025-06-18'),
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'pyright-nav', 'version': '1.0.0'},
        }
    elif method == 'ping':
        response['result'] = {}
    elif method == 'tools/list':
        response['result'] = {'tools': TOOLS}
    elif method == 'tools/call':
        params = message.get('params', {})
        name, args = params.get('name'), params.get('arguments') or {}
        try:
            if name not in {t['name'] for t in TOOLS}:
                raise ToolError(f'unknown tool {name}')
            text = getattr(nav, name)(**args)
            response['result'] = {'content': [{'type': 'text', 'text': text}]}
        except (ToolError, TypeError, OSError) as e:
            response['result'] = {'content': [{'type': 'text', 'text': str(e)}], 'isError': True}
    else:
        response['error'] = {'code': -32601, 'message': f'method not found: {method}'}
    return response


def serve(nav: Navigator, stdin: IO[str], stdout: IO[str]) -> None:
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        response = handle(nav, message)
        if response is not None:
            stdout.write(json.dumps(response) + '\n')
            stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', required=True, help='workspace root that pyright analyzes')
    parser.add_argument('--langserver', default='pyright-langserver --stdio', help='language server command line')
    args = parser.parse_args()
    serve(Navigator(Path(args.root).resolve(), args.langserver.split()), sys.stdin, sys.stdout)


if __name__ == '__main__':
    main()
