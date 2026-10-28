from __future__ import annotations

import collections
import importlib
import types
from collections.abc import Collection

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.types import FunctionKind

from .function import Function
from .iterator import GeneratingFunction

_FUNCTIONS_MODULE = 'pixeltable.functions'


class FunctionRegistry:
    """
    A central, in-process registry of module-level Functions and of the methods/properties registered for each
    column type.
    """

    _instance: FunctionRegistry | None = None

    module_fns: dict[str, Function]
    type_methods: dict[ts.ColumnType.Type, dict[str, Function]]

    @classmethod
    def get(cls) -> FunctionRegistry:
        if cls._instance is None:
            cls._instance = FunctionRegistry()
        return cls._instance

    def __init__(self) -> None:
        self.module_fns: dict[str, Function] = {}  # fqn -> Function
        self.type_methods: dict[ts.ColumnType.Type, dict[str, Function]] = {}

    def register_function(self, fqn: str, fn: Function) -> None:
        if fqn in self.module_fns:
            raise excs.AlreadyExistsError(
                excs.ErrorCode.FUNCTION_ALREADY_EXISTS, f'A UDF with that name already exists: {fqn}'
            )
        self.module_fns[fqn] = fn
        if fn.is_method or fn.is_property:
            base_type = fn.signatures[0].parameters_by_pos[0].col_type.type_enum
            if base_type not in self.type_methods:
                self.type_methods[base_type] = {}
            if fn.name in self.type_methods[base_type]:
                raise excs.AlreadyExistsError(
                    excs.ErrorCode.FUNCTION_ALREADY_EXISTS, f'Duplicate method name for type {base_type}: {fn.name}'
                )
            self.type_methods[base_type][fn.name] = fn

    def deregister_module(self, module_name: str) -> None:
        """Drop every function registered for module_name."""
        self.deregister_functions([fqn for fqn in self.module_fns if fqn.startswith(f'{module_name}.')])

    def deregister_functions(self, fqns: Collection[str]) -> None:
        """Drop the named functions, and any type method for which they are registered."""
        fns = {id(self.module_fns.pop(fqn)) for fqn in fqns}
        for methods in self.type_methods.values():
            for name in [name for name, fn in methods.items() if id(fn) in fns]:
                del methods[name]

    def get_type_methods(self, base_type: ts.ColumnType.Type) -> list[Function]:
        """
        Get a list of all methods (and properties) for a given base type.
        """
        return list(self._get_type_methods(base_type).values())

    def lookup_type_method(self, base_type: ts.ColumnType.Type, name: str) -> Function | None:
        """
        Look up a method (or property) by name for a given base type. If no such method is registered, return None.
        """
        return self._get_type_methods(base_type).get(name)

    def _get_type_methods(self, base_type: ts.ColumnType.Type) -> dict[str, Function]:
        """
        Return the methods (and properties) of a given base type, keyed by name. An exact match on the base type
        takes precedence; an Int base additionally matches methods registered for Float.
        """
        methods = self.type_methods.get(base_type, {})
        if base_type == ts.ColumnType.Type.INT:
            return {**self.type_methods.get(ts.ColumnType.Type.FLOAT, {}), **methods}
        return methods

    def list_functions(
        self, module: str | None, on: ts.ColumnType | None, search: str | None, kind: FunctionKind | None
    ) -> list[tuple[str, Function | GeneratingFunction]]:
        """Return the functions that match all given filters, as (import path, function) pairs sorted by path."""
        # importing pixeltable.functions registers the type methods
        root = self._resolve_module(module)
        entries: list[tuple[str, Function | GeneratingFunction]]
        if on is None:
            entries = self._exported_functions(root)
        else:
            entries = [
                (fn.self_path, fn)
                for fn in self.get_type_methods(on.type_enum)
                if fn.self_path is not None and fn.self_path.startswith(f'{root.__name__}.')
            ]
        if kind is not None:
            entries = [(path, fn) for path, fn in entries if fn.kind == kind]
        if search is not None:
            search = search.lower()
            entries = [
                (path, fn)
                for path, fn in entries
                if search in path.lower() or ((doc := fn.comment()) is not None and search in doc.lower())
            ]
        return sorted(entries, key=lambda entry: entry[0])

    def _resolve_module(self, module: str | None) -> types.ModuleType:
        """Import module, a submodule of pixeltable.functions (such as 'video') or a module path."""
        functions_module = importlib.import_module(_FUNCTIONS_MODULE)
        if module is None:
            return functions_module
        submodule = getattr(functions_module, module, None)
        if isinstance(submodule, types.ModuleType):
            return submodule
        try:
            return importlib.import_module(module)
        except ModuleNotFoundError as e:
            # module is missing if e.name is module or one of its parents, such as 'a' for 'a.b'
            if e.name is None or (module != e.name and not module.startswith(f'{e.name}.')):
                raise
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'No module named {module!r}') from e

    def _exported_functions(self, root: types.ModuleType) -> list[tuple[str, Function | GeneratingFunction]]:
        """Return the functions defined in root and its submodules, each under its shortest exported path."""
        entries: list[tuple[str, Function | GeneratingFunction]] = []
        seen: set[int] = set()
        modules = collections.deque([root])
        while len(modules) > 0:
            mod = modules.popleft()
            names = getattr(mod, '__all__', None)
            if names is None:
                names = [name for name in vars(mod) if not name.startswith('_')]
            for name in names:
                obj = getattr(mod, name, None)
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith(f'{mod.__name__}.'):
                        modules.append(obj)
                    continue
                if not isinstance(obj, (Function, GeneratingFunction)) or id(obj) in seen:
                    continue
                defining_path = obj.fqn if isinstance(obj, GeneratingFunction) else obj.self_path
                # an object imported from outside root belongs to the listing of its own module
                if defining_path is None or not defining_path.startswith(f'{root.__name__}.'):
                    continue
                seen.add(id(obj))
                entries.append((f'{mod.__name__}.{name}', obj))
        return entries
