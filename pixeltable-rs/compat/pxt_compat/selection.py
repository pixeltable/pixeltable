"""The upstream test files the harness runs, and where everything lives.

Shared by sync_tests.py, check_unmodified.py, run.py and classify.py.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

COMPAT_DIR = Path(__file__).resolve().parents[1]
RS_DIR = COMPAT_DIR.parent  # pixeltable-rs/
UPSTREAM_ROOT = RS_DIR.parent  # the upstream Pixeltable checkout
UPSTREAM_TESTS = UPSTREAM_ROOT / 'tests'
COMPAT_TESTS = COMPAT_DIR / 'tests'
MANIFEST = COMPAT_DIR / 'manifest.txt'
RESULTS_DIR = COMPAT_DIR / 'results'
PASSING_FILE = COMPAT_DIR / 'PASSING.txt'
PUBLIC_API_FILE = COMPAT_DIR / 'public_api_tests.txt'
VENV_DIR = RS_DIR / '.venv'

# Upstream test modules run by the harness, copied byte for byte.
SELECTED_TESTS: list[str] = [
    'test_dirs',
    'test_table',
    'test_query',
    'test_exprs',
    'test_view',
    'test_snapshot',
    'test_function',
    'test_alter_column',
    'test_primary_key_index',
    'test_uuid_type',
    'test_sample',
    'test_component_view',
    'test_iterator',
    'test_types',
    'test_array_type',
    'test_pixeltable',
    'test_conftest',
    'test_history',
]

# Files in compat/tests/ that the harness owns; they replace their upstream namesakes.
HARNESS_FILES = frozenset({'conftest.py', 'utils.py'})

# Test data stays in the upstream checkout and is reached through these symlinks: (link, target), both relative
# to COMPAT_DIR. The test files address it as tests/data/..., docs/resources/... relative to the project root, and
# as TESTS_DIR / 'data'.
DATA_LINKS: list[tuple[str, str]] = [
    ('tests/data', '../../../tests/data'),
    ('docs/resources', '../../../docs/resources'),
]

_SCRIPT_REF = re.compile(r"""['"]tests/([A-Za-z0-9_]+\.py)['"]""")


def selected_test_files() -> list[str]:
    return [f'{name}.py' for name in SELECTED_TESTS]


def _local_imports(path: Path) -> set[str]:
    """Names of the modules in upstream tests/ that the module at path imports, plus scripts it references by path."""
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module is None:
                    found.update(alias.name for alias in node.names)
                else:
                    found.add(node.module.split('.')[0])
            elif node.level == 0 and node.module is not None and node.module.split('.')[0] == 'tests':
                parts = node.module.split('.')
                if len(parts) > 1:
                    found.add(parts[1])
                else:
                    found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                if parts[0] == 'tests' and len(parts) > 1:
                    found.add(parts[1])
    found.update(Path(m).stem for m in _SCRIPT_REF.findall(src))
    return found


def required_files() -> list[str]:
    """The selected test files plus every upstream tests/ module they need (transitively), as paths relative to
    tests/. conftest.py and utils.py are replaced by the harness and never copied."""
    result: set[str] = {'__init__.py'}
    todo = [UPSTREAM_TESTS / f for f in selected_test_files()]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(UPSTREAM_TESTS).as_posix()
        if rel not in HARNESS_FILES:
            result.add(rel)
        for name in _local_imports(path):
            if f'{name}.py' in HARNESS_FILES:
                continue
            module_file = UPSTREAM_TESTS / f'{name}.py'
            package_init = UPSTREAM_TESTS / name / '__init__.py'
            if module_file.is_file():
                todo.append(module_file)
            elif package_init.is_file():
                todo.extend(sorted((UPSTREAM_TESTS / name).rglob('*.py')))
    return sorted(result)


def manifest_files() -> list[str]:
    """The files sync_tests.py copied, as recorded in the manifest."""
    if not MANIFEST.is_file():
        return []
    return [line.split()[1] for line in MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
