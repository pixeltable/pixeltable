#!/usr/bin/env python3
"""Statically classify the selected test functions by whether they use only Pixeltable's public API.

A test is in the public-API target set unless it (or a helper it calls, transitively within its module):
- uses a name imported from an internal module (pixeltable.env, exprs, catalog, func, exec, index, utils, runtime,
  config, metadata, iterators.base, ...), e.g. `Env`, `exprs.Literal`, `ColumnRef`, `FunctionRegistry`;
- reaches an internal module through the package, e.g. `pxt.exec.CachePrefetchNode`, `pxt.func.FunctionRegistry`;
- accesses a private attribute (`x._attr`; `self._attr`, dunders, `_repr_html_` and names the test module defines
  itself are fine);
- accesses an attribute that only internal objects expose (`col_type`, `signature`, `serialize`, `subexprs`, ...);
- requests a fixture that needs internals (`test_tbl_exprs`, `img_tbl_exprs`, `multi_img_tbl_exprs`).
Tests that the default marker filter deselects (expensive, very_expensive, remote_api, benchmark, otel, a skip mark, or
a Hugging Face model fixture) are not in the target set either.

Public here means: the pixeltable package's exported names, pixeltable.functions.*, pixeltable.io, pixeltable.iterators
(except iterators.base), pixeltable.exceptions, pixeltable.types, pixeltable.type_system (its classes are public objects
in pixeltable-rs, per PLAN.md) and pixeltable.func.Batch (the documented annotation for batched UDFs). Annotations are
ignored, since the stubs make them evaluate harmlessly.

Writes compat/public_api_tests.txt (function-level node ids, without parametrization) and prints per-file counts.
Usage: python classify.py [--explain]   (--explain lists every excluded test with its reasons)
"""

from __future__ import annotations

import argparse
import ast
import collections
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pxt_compat.selection import COMPAT_TESTS, PUBLIC_API_FILE, SELECTED_TESTS

INTERNAL_MODULE_PREFIXES = (
    'pixeltable.env',
    'pixeltable.exprs',
    'pixeltable.catalog',
    'pixeltable.func',
    'pixeltable.exec',
    'pixeltable.plan',
    'pixeltable.index',
    'pixeltable.metadata',
    'pixeltable.utils',
    'pixeltable.runtime',
    'pixeltable.config',
    'pixeltable.store',
    'pixeltable.service',
    'pixeltable.globals',
    'pixeltable._query',
    'pixeltable.query_clauses',
    'pixeltable.iterators.base',
    'pixeltable.telemetry',
)
# public names that live in otherwise internal modules
PUBLIC_EXCEPTIONS = {('pixeltable.func', 'Batch')}
# attributes of the pixeltable package that are internal modules (upstream leaves them out of __all__)
INTERNAL_PXT_ATTRS = {
    'catalog',
    'env',
    'exec',
    'exprs',
    'func',
    'globals',
    'index',
    'metadata',
    'plan',
    'utils',
    'runtime',
    'store',
    'config',
    'service',
    'query_clauses',
    'telemetry',
}
# attribute names that only internal objects expose (heuristic: unambiguous in the selected files)
INTERNAL_ATTRS = {
    'col_type',
    'signature',
    'signatures',
    'py_fn',
    'py_fns',
    'subexprs',
    'serialize',
    'deserialize',
    'display_name',
    'self_path',
    'module_fns',
    'get_resource_pool_info',
    'default_input_media_dest',
    'default_output_media_dest',
    'is_proxy_daemon',
    'required_parameters',
    'decorated_callable',
    'arity',
    'call_output_schema',
    'tbl_version',
    'store_tbl',
}
ALLOWED_PRIVATE_ATTRS = {'_repr_html_'}
INTERNAL_FIXTURES = {'test_tbl_exprs', 'img_tbl_exprs', 'multi_img_tbl_exprs'}
HF_FIXTURES = {'clip_embed', 'e5_embed', 'all_mpnet_embed'}
DESELECTING_MARKS = {'expensive', 'very_expensive', 'remote_api', 'benchmark', 'otel', 'skip'}


def is_internal_module(name: str) -> bool:
    return any(name == p or name.startswith(p + '.') for p in INTERNAL_MODULE_PREFIXES)


@dataclass
class Unit:
    """A function, method or class whose body may use internals."""

    name: str
    reasons: set[str] = field(default_factory=set)
    deps: set[str] = field(default_factory=set)  # names of other units in the module


class ModuleAnalysis:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        self.internal_names: dict[str, str] = {}  # local name -> what it refers to
        self.pxt_aliases: set[str] = set()  # names bound to the pixeltable package
        self.module_defs: set[str] = set()  # names defined at module level in the test file
        self.units: dict[str, Unit] = {}
        self._collect_imports()
        self._collect_units()

    def _collect_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'pixeltable':
                        self.pxt_aliases.add(alias.asname or 'pixeltable')
                    elif alias.name.startswith('pixeltable.') and alias.asname and is_internal_module(alias.name):
                        self.internal_names[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module != 'pixeltable' and not node.module.startswith('pixeltable.'):
                    continue
                for alias in node.names:
                    local = alias.asname or alias.name
                    target = f'{node.module}.{alias.name}'
                    if (node.module, alias.name) in PUBLIC_EXCEPTIONS:
                        continue
                    if is_internal_module(node.module) or (node.module == 'pixeltable' and is_internal_module(target)):
                        self.internal_names[local] = target
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.module_defs.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.module_defs.add(t.id)

    def _collect_units(self) -> None:
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.units[node.name] = self._analyze(node.name, node, None)
            elif isinstance(node, ast.ClassDef):
                class_unit = self._analyze(node.name, node, node.name, skip_methods=True)
                self.units[node.name] = class_unit
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qual = f'{node.name}.{item.name}'
                        self.units[qual] = self._analyze(qual, item, node.name)
                        # pytest calls a test class's methods one at a time; code that uses any other class (an
                        # iterator, an aggregator) may have any of its methods called implicitly
                        if not node.name.startswith('Test'):
                            class_unit.deps.add(qual)

    def _analyze(self, name: str, node: ast.AST, class_name: str | None, skip_methods: bool = False) -> Unit:
        unit = Unit(name)
        annotation_nodes: set[int] = set()
        for sub in ast.walk(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = sub.args
                for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                    if a is not None and a.annotation is not None:
                        annotation_nodes.update(id(n) for n in ast.walk(a.annotation))
                if sub.returns is not None:
                    annotation_nodes.update(id(n) for n in ast.walk(sub.returns))
            elif isinstance(sub, ast.AnnAssign):
                annotation_nodes.update(id(n) for n in ast.walk(sub.annotation))
        skipped: set[int] = set()
        if skip_methods:
            assert isinstance(node, ast.ClassDef)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    skipped.update(id(n) for n in ast.walk(item))
                    # decorators of methods run at class creation
                    for d in item.decorator_list:
                        skipped.difference_update(id(n) for n in ast.walk(d))

        for sub in ast.walk(node):
            if id(sub) in annotation_nodes or id(sub) in skipped:
                continue
            if isinstance(sub, ast.Name):
                if sub.id in self.internal_names:
                    unit.reasons.add(f'uses {self.internal_names[sub.id]}')
                elif sub.id in self.module_defs and sub.id != name:
                    unit.deps.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                base = sub.value
                if isinstance(base, ast.Name) and base.id in self.pxt_aliases and sub.attr in INTERNAL_PXT_ATTRS:
                    unit.reasons.add(f'uses pixeltable.{sub.attr}')
                elif sub.attr.startswith('_') and not (sub.attr.startswith('__') and sub.attr.endswith('__')):
                    is_self = isinstance(base, ast.Name) and base.id in ('self', 'cls')
                    if is_self and class_name is not None:
                        unit.deps.add(f'{class_name}.{sub.attr}')
                    elif not is_self and sub.attr not in ALLOWED_PRIVATE_ATTRS and sub.attr not in self.module_defs:
                        unit.reasons.add(f'private attribute .{sub.attr}')
                elif sub.attr in INTERNAL_ATTRS:
                    unit.reasons.add(f'internal attribute .{sub.attr}')
                if isinstance(base, ast.Name) and base.id in ('self', 'cls') and class_name is not None:
                    unit.deps.add(f'{class_name}.{sub.attr}')
                elif isinstance(base, ast.Name) and base.id in self.module_defs:
                    unit.deps.add(f'{base.id}.{sub.attr}')
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in [*node.args.args, *node.args.kwonlyargs]:
                if a.arg in INTERNAL_FIXTURES:
                    unit.reasons.add(f'fixture {a.arg} needs internals')
        return unit

    def closure_reasons(self, unit_name: str) -> set[str]:
        reasons: set[str] = set()
        seen: set[str] = set()
        todo = [unit_name]
        while todo:
            n = todo.pop()
            if n in seen or n not in self.units:
                continue
            seen.add(n)
            u = self.units[n]
            via = '' if n == unit_name else f' (via {n})'
            reasons.update(r + via for r in u.reasons)
            todo.extend(u.deps)
        return reasons


def _mark_names(decorators: list[ast.expr]) -> set[str]:
    names = set()
    for d in decorators:
        target = d.func if isinstance(d, ast.Call) else d
        text = ast.unparse(target)
        if text.startswith('pytest.mark.'):
            names.add(text.removeprefix('pytest.mark.'))
    return names


def _pytestmark(body: list[ast.stmt]) -> set[str]:
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pytestmark' for t in node.targets):
            values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            return _mark_names(values)
    return set()


def classify_file(name: str) -> list[tuple[str, set[str]]]:
    """(function node id, reasons) for every test function in the file; no reasons = public-API only."""
    path = COMPAT_TESTS / f'{name}.py'
    analysis = ModuleAnalysis(path)
    module_marks = _pytestmark(analysis.tree.body)
    out: list[tuple[str, set[str]]] = []
    classes = {n.name: n for n in analysis.tree.body if isinstance(n, ast.ClassDef)}

    def methods_of(cls: ast.ClassDef) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
        result: dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for base in cls.bases:  # test classes inheriting from another class in the same module
            if isinstance(base, ast.Name) and base.id in classes:
                result.update((m, v) for m, v in methods_of(classes[base.id]))
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[item.name] = (cls.name, item)
        return list(result.items())

    def deselected(marks: set[str], fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        reasons = {f'deselected by mark {m}' for m in marks & DESELECTING_MARKS}
        hf = {a.arg for a in fn.args.args} & HF_FIXTURES
        if hf:
            reasons.add(f'deselected: Hugging Face fixture {", ".join(sorted(hf))}')
        return reasons

    for node in analysis.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test'):
            reasons = analysis.closure_reasons(node.name)
            reasons |= deselected(module_marks | _mark_names(node.decorator_list), node)
            out.append((f'tests/{name}.py::{node.name}', reasons))
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            class_marks = module_marks | _mark_names(node.decorator_list) | _pytestmark(node.body)
            for method_name, (owner, fn) in methods_of(node):
                if not method_name.startswith('test'):
                    continue
                reasons = analysis.closure_reasons(f'{owner}.{method_name}')
                reasons |= deselected(class_marks | _mark_names(fn.decorator_list), fn)
                out.append((f'tests/{name}.py::{node.name}::{method_name}', reasons))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--explain', action='store_true', help='list excluded tests with their reasons')
    args = parser.parse_args()

    public: list[str] = []
    print(f'{"file":34} {"tests":>6} {"public":>7} {"internal":>9} {"deselected":>11}')
    totals = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    for name in SELECTED_TESTS:
        results = classify_file(name)
        n_public = n_internal = n_deselected = 0
        for nodeid, reasons in results:
            if not reasons:
                public.append(nodeid)
                n_public += 1
                continue
            if any(r.startswith('deselected') for r in reasons):
                n_deselected += 1
            else:
                n_internal += 1
                reason_counts.update({r.split(' (via ')[0] for r in reasons})
            if args.explain:
                print(f'  {nodeid}: {"; ".join(sorted(reasons))}')
        totals.update(tests=len(results), public=n_public, internal=n_internal, deselected=n_deselected)
        print(f'tests/{name + ".py":28} {len(results):6} {n_public:7} {n_internal:9} {n_deselected:11}')
    print(f'{"TOTAL":34} {totals["tests"]:6} {totals["public"]:7} {totals["internal"]:9} {totals["deselected"]:11}')
    print('\nmost common reasons for exclusion (internal):')
    for reason, n in reason_counts.most_common(15):
        print(f'  {n:5}  {reason}')
    PUBLIC_API_FILE.write_text(''.join(f'{n}\n' for n in sorted(public)), encoding='utf-8')
    print(f'\nwrote {PUBLIC_API_FILE} ({len(public)} test functions)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
