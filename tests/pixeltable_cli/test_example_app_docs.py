"""The documented application file has to stay the one 'pxt service example' writes.

README.md, the quickstart, and skill.md all reproduce `_EXAMPLE_APP`. When the template gained a
required column the documented `curl` kept its old body, so the first request a new user made
returned 422. These tests compare the documented snippets against the template itself, and check
that every documented request body carries the route's required inputs.
"""

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pixeltable_cli.client.commands.service import _EXAMPLE_APP

_REPO_ROOT = Path(__file__).parents[2]

# Every file that reproduces the generated application file.
_MIRRORS = [Path('README.md'), Path('docs/release/overview/quick-start.mdx'), Path('docs/release/skill.md')]

_PYTHON_FENCE = re.compile(r'```python\n(.*?)```', re.DOTALL)
_CURL_BODY = re.compile(r"-d '(\{.*?\})'", re.DOTALL)


def _shape(src: str) -> dict[str, Any]:
    """The parts of an application file a reader copies: its udfs, its columns, and its routes."""
    tree = ast.parse(src)
    udfs: list[str] = []
    columns: list[tuple[str, str]] = []
    routes: list[tuple[str, Any, tuple[str, ...], tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            udfs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    columns.append(('annotation', stmt.target.id))
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    columns.append(('computed', stmt.targets[0].id))
        elif isinstance(node, ast.Call) and getattr(node.func, 'attr', '').startswith('add_'):
            kwargs = {
                kw.arg: tuple(ast.unparse(e).split('.')[-1] for e in kw.value.elts)
                if isinstance(kw.value, ast.List)
                else ast.unparse(kw.value)
                for kw in node.keywords
            }
            routes.append((node.func.attr, kwargs.get('path'), kwargs.get('inputs', ()), kwargs.get('outputs', ())))
    return {'udfs': udfs, 'columns': columns, 'routes': routes}


def _nullable_columns() -> set[str]:
    """Columns the template annotates `| None`. A request may omit these; omitting the rest is a 422."""
    nullable: set[str] = set()
    for node in ast.walk(ast.parse(_EXAMPLE_APP)):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and 'None' in ast.unparse(stmt.annotation)
            ):
                nullable.add(stmt.target.id)
    return nullable


def _documented_app(path: Path) -> str:
    text = (_REPO_ROOT / path).read_text()
    blocks = [b for b in _PYTHON_FENCE.findall(text) if 'model_base' in b]
    assert len(blocks) == 1, f'{path}: expected one application-file snippet, found {len(blocks)}'
    return textwrap.dedent(blocks[0])


@pytest.mark.parametrize('path', _MIRRORS, ids=lambda p: p.name)
class TestExampleAppDocs:
    def test_snippet_matches_template(self, path: Path) -> None:
        assert _shape(_documented_app(path)) == _shape(_EXAMPLE_APP)

    def test_request_bodies_carry_required_inputs(self, path: Path) -> None:
        """A documented curl that omits a required route input returns 422 against the real app."""
        template = _shape(_EXAMPLE_APP)
        required = {name for kind, name in template['columns'] if kind == 'annotation'} - _nullable_columns()
        insert_routes = [r for r in template['routes'] if r[0] == 'add_insert_route']
        assert insert_routes, 'the template no longer defines an insert route'
        needed = required & set(insert_routes[0][2])

        bodies = _CURL_BODY.findall((_REPO_ROOT / path).read_text())
        for body in bodies:
            keys = set(json.loads(body))
            assert needed <= keys, f'{path}: request body {sorted(keys)} is missing {sorted(needed - keys)}'
