from __future__ import annotations

import collections
import html
import importlib
import inspect
import types

import pandas as pd

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.types import FunctionInfo, FunctionKind

from .aggregate_function import AggregateFunction
from .function import Function
from .function_registry import FunctionRegistry
from .iterator import GeneratingFunction
from .query_template_function import QueryTemplateFunction
from .signature import Signature

_FUNCTIONS_MODULE = 'pixeltable.functions'


class FunctionList(list[FunctionInfo]):
    """A list of functions, as returned by [`pxt.list_functions()`][pixeltable.list_functions].

    Printed, it shows one function per line; in a notebook, it renders as a table.
    """

    @classmethod
    def collect(
        cls, module: str | None, on: ts.ColumnType | None, search: str | None, kind: FunctionKind | None
    ) -> FunctionList:
        """Collect the functions that match all given filters."""
        # importing pixeltable.functions registers the type methods
        root = cls._resolve_module(module)
        entries: list[tuple[str, Function | GeneratingFunction]]
        if on is None:
            entries = cls._exported_functions(root)
        else:
            entries = [
                (fn.self_path, fn)
                for fn in cls._type_methods(on)
                if fn.self_path is not None and fn.self_path.startswith(f'{root.__name__}.')
            ]
        infos = [cls._info(path, fn) for path, fn in entries]
        if kind is not None:
            infos = [info for info in infos if info['kind'] == kind]
        if search is not None:
            needle = search.lower()
            infos = [
                info
                for info in infos
                if needle in info['path'].lower() or (info['doc'] is not None and needle in info['doc'].lower())
            ]
        return cls(sorted(infos, key=lambda info: info['path']))

    @classmethod
    def _resolve_module(cls, module: str | None) -> types.ModuleType:
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
            if e.name != module:
                raise
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'No module named {module!r}') from e

    @classmethod
    def _exported_functions(cls, root: types.ModuleType) -> list[tuple[str, Function | GeneratingFunction]]:
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

    @classmethod
    def _type_methods(cls, on: ts.ColumnType) -> list[Function]:
        """Return the methods and properties of on; an Int also has the Float ones, as in lookup_type_method()."""
        registry = FunctionRegistry.get()
        fns = registry.get_type_methods(on.type_enum)
        if on.type_enum == ts.ColumnType.Type.INT:
            names = {fn.name for fn in fns}
            fns += [fn for fn in registry.get_type_methods(ts.ColumnType.Type.FLOAT) if fn.name not in names]
        return fns

    @classmethod
    def _info(cls, path: str, fn: Function | GeneratingFunction) -> FunctionInfo:
        kind: FunctionKind
        if isinstance(fn, GeneratingFunction):
            kind = 'iterator'
            signatures = [fn.signature]
            # not inspect.getdoc(), which falls back to a base class's docstring
            own_doc = fn.decorated_callable.__doc__
            doc = None if own_doc is None else inspect.cleandoc(own_doc)
            output_schema = fn.fixed_output_schema
            if output_schema is None:
                outputs = None
                returns = '(columns depend on the arguments)'
            else:
                outputs = {name: cls._type_str(t) for name, t in output_schema.items()}
                returns = '{' + ', '.join(f'{name}: {t}' for name, t in outputs.items()) + '}'
        else:
            if isinstance(fn, AggregateFunction):
                kind = 'aggregate'
            elif isinstance(fn, QueryTemplateFunction):
                kind = 'query'
            else:
                kind = 'udf'
            signatures = fn.signatures
            doc = fn.comment()
            outputs = None
            returns = signatures[0].return_str(pretty_print_json=True)

        return FunctionInfo(
            path=path,
            name=fn.name,
            kind=kind,
            call=f'{cls._call_str(path, fn, signatures[0])} -> {returns}',
            signatures=[cls._signature_str(s) for s in signatures],
            summary=None if doc is None else ' '.join(doc.split('\n\n', 1)[0].split()),
            doc=doc,
            outputs=outputs,
        )

    @classmethod
    def _call_str(cls, path: str, fn: Function | GeneratingFunction, sig: Signature) -> str:
        """Format a call of fn with sig, without its return type."""
        if isinstance(fn, Function) and (fn.is_method or fn.is_property):
            receiver_type = sig.parameters_by_pos[0].col_type
            assert receiver_type is not None
            # as Python's docs write str.upper()
            receiver = f'pxt.{str(receiver_type.copy(nullable=False)).split("[", 1)[0]}'
            if fn.is_property:
                return f'{receiver}.{fn.name}'
            return f'{receiver}.{fn.name}({sig.params_str(with_defaults=True, as_method=True)})'
        return f'{path.removeprefix(f"{_FUNCTIONS_MODULE}.")}({sig.params_str(with_defaults=True)})'

    @classmethod
    def _signature_str(cls, sig: Signature) -> str:
        return f'({sig.params_str(with_defaults=True)}) -> {sig.return_str(pretty_print_json=True)}'

    @classmethod
    def _type_str(cls, col_type: ts.ColumnType) -> str:
        """Format col_type as a signature does, with a JSON type declared by a TypedDict as the TypedDict's name."""
        if isinstance(col_type, ts.JsonType) and col_type.pretty_print_name is not None:
            return col_type.pretty_print_name
        return f'pxt.{col_type}'

    def __repr__(self) -> str:
        return '\n'.join(self._line(info) for info in self)

    def _line(self, info: FunctionInfo) -> str:
        line = f'{info["kind"]:<9} {info["call"]}'
        if len(info['signatures']) > 1:
            line += f'  [{len(info["signatures"])} signatures]'
        if info['summary'] is not None:
            line += f'  # {info["summary"]}'
        return line

    def _repr_html_(self) -> str:
        rows = ''.join(
            f'<tr><td>{info["kind"]}</td><td><code>{html.escape(info["call"])}</code></td>'
            f'<td>{"" if info["summary"] is None else html.escape(info["summary"])}</td></tr>'
            for info in self
        )
        return f'<table><tr><th>kind</th><th>call</th><th>summary</th></tr>{rows}</table>'

    def to_pandas(self) -> pd.DataFrame:
        """Return the functions as a DataFrame with one row per function and one column per field."""
        return pd.DataFrame(list(self), columns=list(FunctionInfo.__annotations__))
