"""Stand-ins for the parts of the upstream Pixeltable package that pixeltable-rs does not provide.

The upstream test modules import internal modules at the top (`from pixeltable.env import Env`,
`from pixeltable import exprs, catalog, func`, `import pixeltable.type_system as ts`, ...). pixeltable-rs ships only
the public API, so without help those imports would fail and take every test in the file down with them. This module
makes them succeed with placeholders, so that only a test that actually uses an internal fails, with an error that
says why.

Two mechanisms, both limited to names that the copied test files import (computed statically by `scan_imports()`):

1. Missing modules. `StubFinder` is appended to the end of `sys.meta_path`, so every real finder runs first and a
   module that pixeltable-rs provides is never shadowed. Only when no finder can locate an allow-listed
   `pixeltable.*` module does it create a stub module whose every public attribute is a placeholder.
2. Missing names in modules that do exist. For `from pixeltable.x import Name` at the top of a test file, where
   `pixeltable.x` exists but lacks `Name` (and Name is not a module in the upstream source tree),
   `install_attribute_fallbacks()` gives the real module a module-level `__getattr__` (PEP 562) that answers with a
   placeholder for exactly those allow-listed names. Real attributes are untouched, and any other missing name still
   raises AttributeError.

A placeholder is a class (so it works as a base class, in annotations, in `X | None` and as `X[...]`) whose
attributes are further placeholders. Calling it, instantiating it, or using it in isinstance()/issubclass() raises
`InternalAPIUnavailable`, which derives from BaseException so that an `except Exception` in the code under test
cannot swallow it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Sequence

MARKER = 'not available in pixeltable-rs'

# Upstream modules that are part of the public API: pixeltable.functions.*, io, iterators, exceptions, types,
# type_system (whose classes the pixeltable-rs plan makes public). Everything else under pixeltable.* is internal.
_PUBLIC_PREFIXES = (
    'pixeltable.functions',
    'pixeltable.io',
    'pixeltable.iterators',
    'pixeltable.exceptions',
    'pixeltable.types',
    'pixeltable.type_system',
)


_INTERNAL_PREFIXES = ('pixeltable.iterators.base',)  # legacy iterator base class, not part of the public API


def is_public_module(name: str) -> bool:
    def under(prefixes: tuple[str, ...]) -> bool:
        return any(name == p or name.startswith(p + '.') for p in prefixes)

    return under(_PUBLIC_PREFIXES) and not under(_INTERNAL_PREFIXES)


def _describe(qualname: str) -> str:
    kind = 'public API' if is_public_module(qualname) else 'internal API'
    return f'{kind} `{qualname}` is {MARKER} (stubbed by the compat harness)'


class InternalAPIUnavailable(BaseException):
    """Raised when a test uses an upstream module or name that pixeltable-rs does not provide."""


class _PlaceholderMeta(type):
    _pxt_qualname: str

    def __getattr__(cls, name: str) -> Any:
        if name.startswith('_'):
            # keep introspection (pytest, typing, inspect, pickle) away from placeholders
            raise AttributeError(f'{_describe(cls._pxt_qualname)}; no attribute {name!r}')
        return make_placeholder(f'{cls._pxt_qualname}.{name}')

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        raise InternalAPIUnavailable(_describe(cls._pxt_qualname))

    def __getitem__(cls, item: Any) -> Any:
        return make_placeholder(f'{cls._pxt_qualname}[...]')

    def __instancecheck__(cls, instance: Any) -> bool:
        raise InternalAPIUnavailable(_describe(cls._pxt_qualname))

    def __subclasscheck__(cls, subclass: Any) -> bool:
        raise InternalAPIUnavailable(_describe(cls._pxt_qualname))

    def __iter__(cls) -> Any:
        raise InternalAPIUnavailable(_describe(cls._pxt_qualname))

    def __bool__(cls) -> bool:
        return True  # defined because __len__ is: truth tests must not raise

    def __repr__(cls) -> str:
        return f'<placeholder for {cls._pxt_qualname} ({MARKER})>'


def _unavailable_operator(name: str) -> Any:
    def op(cls: Any, *args: Any) -> Any:
        raise InternalAPIUnavailable(_describe(cls._pxt_qualname))

    op.__name__ = name
    return op


# Arithmetic, ordering, containment and conversions on a placeholder (`'a' * BtreeIndex.MAX_STRING_LEN`) raise the
# stub error. Left alone: `|` (for `X | None` annotations), `==`/hash (identity, so placeholders work as dict keys),
# and truth tests.
for _op in (
    'add',
    'mul',
    'sub',
    'truediv',
    'floordiv',
    'mod',
    'pow',
    'matmul',
    'lshift',
    'rshift',
    'and',
    'xor',
    'divmod',
):
    setattr(_PlaceholderMeta, f'__{_op}__', _unavailable_operator(f'__{_op}__'))
    setattr(_PlaceholderMeta, f'__r{_op}__', _unavailable_operator(f'__r{_op}__'))
for _op in (
    'neg',
    'pos',
    'abs',
    'invert',
    'index',
    'int',
    'float',
    'complex',
    'len',
    'lt',
    'le',
    'gt',
    'ge',
    'contains',
):
    setattr(_PlaceholderMeta, f'__{_op}__', _unavailable_operator(f'__{_op}__'))


def make_placeholder(qualname: str) -> Any:
    name = qualname.rsplit('.', 1)[-1]
    module = qualname.rsplit('.', 1)[0] if '.' in qualname else qualname
    return _PlaceholderMeta(name, (), {'_pxt_qualname': qualname, '__module__': module, '__qualname__': name})


class StubModule(types.ModuleType):
    """A module that pixeltable-rs lacks; every public attribute is a placeholder."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name.startswith('_'):
            raise AttributeError(f'{_describe(self.__name__)}; no attribute {name!r}')
        value = make_placeholder(f'{self.__name__}.{name}')
        setattr(self, name, value)
        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise InternalAPIUnavailable(_describe(self.__name__))

    def __repr__(self) -> str:
        return f'<stub module {self.__name__!r} ({MARKER})>'


_stubbed_modules: dict[str, str] = {}  # module name -> 'public' | 'internal'
_stubbed_attrs: dict[str, list[str]] = {}  # real module name -> names answered by placeholders


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return StubModule(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        module.__path__ = []  # a package, so that stubbed submodules can be imported under it
        _stubbed_modules[module.__name__] = 'public' if is_public_module(module.__name__) else 'internal'
        parent_name, _, child = module.__name__.rpartition('.')
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, child):
            setattr(parent, child, module)


class StubFinder(importlib.abc.MetaPathFinder):
    """Last-resort finder for the allow-listed pixeltable.* modules that no real finder can locate."""

    def __init__(self, module_names: Iterable[str]) -> None:
        self.module_names = frozenset(module_names)

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self.module_names:
            return None
        parent_name, _, child = fullname.rpartition('.')
        parent = sys.modules.get(parent_name)
        if parent is not None and not isinstance(parent, StubModule) and child in parent.__dict__:
            return None  # a real attribute of that name exists; the import system would overwrite it with the stub
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)


class ImportScan:
    """The pixeltable.* imports found in a set of test files."""

    def __init__(self) -> None:
        self.modules: set[str] = set()  # every pixeltable.* module imported anywhere, with its parents
        self.from_names: dict[str, set[str]] = {}  # module -> non-module names imported from it at the top of a file


def _is_upstream_module(name: str, upstream_root: Path | None) -> bool:
    """Whether name (pixeltable.x.y) is a module in the upstream source tree; read from the files, never imported."""
    if upstream_root is None:
        return False
    path = upstream_root.joinpath(*name.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def scan_imports(files: Iterable[Path], upstream_root: Path | None = None) -> ImportScan:
    """Collect the pixeltable.* modules and names that files import. For `from pixeltable.x import y`, y counts as a
    module when the upstream checkout at upstream_root has it as one (pixeltable/x/y.py or pixeltable/x/y/)."""
    scan = ImportScan()

    def add_module(name: str) -> None:
        parts = name.split('.')
        for i in range(2, len(parts) + 1):
            scan.modules.add('.'.join(parts[:i]))

    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        top_level = set(map(id, tree.body))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('pixeltable.'):
                        add_module(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                if node.module != 'pixeltable' and not node.module.startswith('pixeltable.'):
                    continue
                if node.module != 'pixeltable':
                    add_module(node.module)
                for alias in node.names:
                    candidate = f'{node.module}.{alias.name}'
                    if _is_upstream_module(candidate, upstream_root):
                        add_module(candidate)
                    elif id(node) in top_level:
                        scan.from_names.setdefault(node.module, set()).add(alias.name)
    return scan


def install(scan: ImportScan) -> StubFinder:
    """Register the stub finder (once) for the modules in scan."""
    for finder in sys.meta_path:
        if isinstance(finder, StubFinder):
            return finder
    finder = StubFinder(scan.modules)
    sys.meta_path.append(finder)
    return finder


def install_attribute_fallbacks(scan: ImportScan) -> None:
    """For `from pixeltable.x import Name` at the top of a test file: if pixeltable.x is a real module that lacks
    Name (and Name is not a submodule), let the module answer Name with a placeholder."""
    for module_name, names in sorted(scan.from_names.items()):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if isinstance(module, StubModule):
            continue
        missing = set()
        for name in names:
            if hasattr(module, name):
                continue
            try:
                importlib.import_module(f'{module_name}.{name}')
            except ImportError:
                missing.add(name)
            else:
                continue
        if not missing:
            continue
        _add_fallback(module, missing)


def _add_fallback(module: types.ModuleType, names: set[str]) -> None:
    previous = module.__dict__.get('__getattr__')
    allowed = frozenset(names) | frozenset(_stubbed_attrs.get(module.__name__, ()))
    _stubbed_attrs[module.__name__] = sorted(allowed)
    module_name = module.__name__

    def module_getattr(name: str) -> Any:
        if previous is not None:
            try:
                return previous(name)
            except AttributeError:
                if name not in allowed:
                    raise
        if name in allowed:
            return make_placeholder(f'{module_name}.{name}')
        raise AttributeError(f'module {module_name!r} has no attribute {name!r}')

    module.__getattr__ = module_getattr  # type: ignore[method-assign]


def report() -> dict[str, Any]:
    """What has been stubbed in this process so far."""
    return {'modules': dict(sorted(_stubbed_modules.items())), 'attributes': dict(sorted(_stubbed_attrs.items()))}
