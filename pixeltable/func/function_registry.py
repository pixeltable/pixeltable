from __future__ import annotations

from pixeltable import exceptions as excs, type_system as ts

from .function import Function


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
