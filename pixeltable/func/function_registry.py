from __future__ import annotations

import importlib
import sys
from collections.abc import Collection
from types import ModuleType

from pixeltable import exceptions as excs, type_system as ts

from .function import Function


class FunctionRegistry:
    """
    A central, in-process registry of module-level Functions and of the methods/properties registered for each
    column type.
    """

    _instance: FunctionRegistry | None = None

    module_fns: dict[str, Function]
    module_names: dict[str, str]  # fqn -> name of the module that defines the function
    type_methods: dict[ts.ColumnType.Type, dict[str, Function]]

    @classmethod
    def get(cls) -> FunctionRegistry:
        if cls._instance is None:
            cls._instance = FunctionRegistry()
        return cls._instance

    def __init__(self) -> None:
        self.module_fns: dict[str, Function] = {}  # fqn -> Function
        self.module_names = {}
        self.type_methods: dict[ts.ColumnType.Type, dict[str, Function]] = {}

    def register_function(self, fqn: str, fn: Function, module_name: str) -> None:
        if fqn in self.module_fns:
            raise excs.AlreadyExistsError(
                excs.ErrorCode.FUNCTION_ALREADY_EXISTS, f'A UDF with that name already exists: {fqn}'
            )
        self.module_fns[fqn] = fn
        self.module_names[fqn] = module_name
        if fn.is_method or fn.is_property:
            base_type = fn.signatures[0].parameters_by_pos[0].col_type.type_enum
            if base_type not in self.type_methods:
                self.type_methods[base_type] = {}
            if fn.name in self.type_methods[base_type]:
                raise excs.AlreadyExistsError(
                    excs.ErrorCode.FUNCTION_ALREADY_EXISTS, f'Duplicate method name for type {base_type}: {fn.name}'
                )
            self.type_methods[base_type][fn.name] = fn

    def import_module(self, name: str) -> ModuleType:
        """Import the module name; if that raises, drop the functions of the modules that failed to import."""
        registered = set(self.module_fns)
        try:
            return importlib.import_module(name)
        except BaseException:
            # Python removes a failed module from sys.modules but leaves its functions registered; a module that
            # imported along the way, such as a parent package or a dependency, stays in sys.modules with its functions
            new_fqns = set(self.module_fns) - registered
            self.deregister_functions([fqn for fqn in new_fqns if self.module_names[fqn] not in sys.modules])
            raise

    def deregister_module(self, module_name: str) -> None:
        """Drop every function registered for module_name."""
        self.deregister_functions([fqn for fqn, name in self.module_names.items() if name == module_name])

    def deregister_functions(self, fqns: Collection[str]) -> None:
        """Drop the named functions, and any type method for which they are registered."""
        fns = {id(self.module_fns.pop(fqn)) for fqn in fqns}
        for fqn in fqns:
            del self.module_names[fqn]
        for methods in self.type_methods.values():
            for name in [name for name, fn in methods.items() if id(fn) in fns]:
                del methods[name]

    def list_functions(self) -> list[Function]:
        return list(self.module_fns.values())

    def get_type_methods(self, base_type: ts.ColumnType.Type) -> list[Function]:
        """
        Get a list of all methods (and properties) registered for a given base type.
        """
        if base_type in self.type_methods:
            return list(self.type_methods[base_type].values())
        return []

    def lookup_type_method(self, base_type: ts.ColumnType.Type, name: str) -> Function | None:
        """
        Look up a method (or property) by name for a given base type. If no such method is registered, return None.

        An exact match on the base type takes precedence; an Int base additionally matches methods registered
        for Float.
        """
        if base_type in self.type_methods and name in self.type_methods[base_type]:
            return self.type_methods[base_type][name]
        if base_type == ts.ColumnType.Type.INT:
            return self.lookup_type_method(ts.ColumnType.Type.FLOAT, name)
        return None
