"""Models defined in a TypeScript schema file, through the @pixeltable/sdk package.

Node evaluates the file and prints each model as data: its columns, in order, with every computed value as a tree of
name-based nodes. This module replays that data through the Python DSL, so the result is the model a Python class body
with the same definitions would produce, and the CLI handles it like one.
"""

from __future__ import annotations

import datetime
import functools
import json
import operator
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, cast

from typing_extensions import TypeForm

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.config import Config
from pixeltable.func.symbols import resolve_symbol

from .base import model_base
from .definition import Column, ModelBody, TableModelMeta, define_model

SUFFIXES = frozenset(('.ts', '.tsx', '.mts'))

_EXPORT_VERSION = 1
_NODE_TIMEOUT_S = 120
_STDERR_TAIL_LINES = 20

# runs from the project root, which resolves the package from its node_modules
_EXPORT_SCRIPT = (
    "const { exportModels } = await import('@pixeltable/sdk/export');"
    'process.stdout.write(JSON.stringify(await exportModels(process.argv[1])));'
)

# the type each name in a type spec stands for, as a class body would annotate it
_TYPE_FORMS: dict[str, TypeForm] = {
    'string': pxt.String,
    'int': pxt.Int,
    'float': pxt.Float,
    'bool': pxt.Bool,
    'timestamp': pxt.Timestamp,
    'date': pxt.Date,
    'uuid': pxt.UUID,
    'binary': pxt.Binary,
    'json': pxt.Json,
    'image': pxt.Image,
    'video': pxt.Video,
    'audio': pxt.Audio,
    'document': pxt.Document,
}

_BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    'eq': operator.eq,
    'ne': operator.ne,
    'lt': operator.lt,
    'lte': operator.le,
    'gt': operator.gt,
    'gte': operator.ge,
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': operator.truediv,
    'mod': operator.mod,
    'floorDiv': operator.floordiv,
    'pow': operator.pow,
    'concat': operator.add,
}

# associative: more than two operands fold left
_VARIADIC_OPS: dict[str, Callable[[Any, Any], Any]] = {'and': operator.and_, 'or': operator.or_}

_UNARY_OPS: dict[str, Callable[[Any], Any]] = {
    'neg': operator.neg,
    'not': operator.invert,
    'abs': abs,
    'isNull': lambda e: e == None,
    'isNotNull': lambda e: e != None,
}

# Python's names for the column options a spec can set
_COLUMN_OPTIONS = {
    'primaryKey': 'primary_key',
    'stored': 'stored',
    'mediaValidation': 'media_validation',
    'customMetadata': 'custom_metadata',
    'comment': 'comment',
    'destination': 'destination',
}

_MODEL_OPTIONS = {
    'comment': 'comment',
    'customMetadata': 'custom_metadata',
    'mediaValidation': 'media_validation',
    'hasDefaultIdxs': 'has_default_idxs',
}


def is_typescript_file(file: str) -> bool:
    return Path(file).suffix in SUFFIXES


def model_bases(app_file: str) -> list[TableModelMeta]:
    """The model base holding every model app_file exports."""
    path = Path(app_file).resolve()
    root = Config.get().project_root or path.parent
    return [models_from_export(export_models(path, root), module=path.stem)]


def models_from_export(exported: Any, module: str) -> TableModelMeta:
    """A fresh model base holding one model per spec in exported, the output of the package's exportModels()."""
    if not isinstance(exported, dict) or exported.get('v') != _EXPORT_VERSION:
        raise _invalid(f'unsupported model export format (expected version {_EXPORT_VERSION})')
    base = model_base()
    for spec in exported.get('models', []):
        _define(base, spec, module)
    return base


def export_models(app_file: Path, cwd: Path) -> Any:
    node = shutil.which('node')
    if node is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'{app_file.name} is a TypeScript schema file, which needs Node.js 22.18 or later on PATH',
        )
    try:
        proc = subprocess.run(
            [
                node,
                '--disable-warning=ExperimentalWarning',
                '--input-type=module',
                '--eval',
                _EXPORT_SCRIPT,
                str(app_file),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_NODE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise _invalid(f'{app_file.name}: Node.js did not finish evaluating it within {_NODE_TIMEOUT_S}s') from e
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-_STDERR_TAIL_LINES:])
        raise _invalid(f'{app_file.name}: Node.js failed to evaluate it:\n{tail}')
    try:
        return json.loads(proc.stdout)
    except ValueError as e:
        raise _invalid(f'{app_file.name}: Node.js printed something other than the model export') from e


def _define(base: TableModelMeta, spec: Any, module: str) -> None:
    if not isinstance(spec, dict) or not isinstance(spec.get('cls'), str) or not isinstance(spec.get('name'), str):
        raise _invalid(f'malformed model spec: {spec!r}')
    cls_name = spec['cls']
    if len(spec.get('indexes', [])) > 0:
        raise _invalid(f'model `{cls_name}`: indexes are not supported in TypeScript schemas yet')
    if spec.get('base') is not None or spec.get('iterator') is not None:
        raise _invalid(f'model `{cls_name}`: views are not supported in TypeScript schemas yet')
    options = spec.get('options', {})
    table_kwargs: dict[str, Any] = {'name': spec['name']}
    for key, value in options.items():
        if key not in _MODEL_OPTIONS:
            raise _invalid(f'model `{cls_name}`: unknown option {key!r}')
        table_kwargs[_MODEL_OPTIONS[key]] = value

    def body(model_body: ModelBody) -> None:
        for entry in spec.get('columns', []):
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                raise _invalid(f'model `{cls_name}`: malformed column entry {entry!r}')
            col_name, col_spec = entry
            try:
                value = _column(col_spec, model_body)
            except excs.Error as e:
                # no traceback points into the schema file, so the message carries the model and column; augmenting
                # in place keeps the error's type and code
                e.args = (f'model `{cls_name}`, column {col_name!r}: {e.message}',)
                raise
            except Exception as e:
                # the DSL signals a misuse (an operator on the wrong types, say) with a plain Python exception
                raise _invalid(f'model `{cls_name}`, column {col_name!r}: {e}') from e
            model_body.define(col_name, value)

    define_model(base, cls_name, module, body, **table_kwargs)


def _column(col_spec: Any, model_body: ModelBody) -> Column | exprs.Expr:
    if not isinstance(col_spec, dict):
        raise _invalid(f'malformed column spec {col_spec!r}')
    options: dict[str, Any] = {}
    for key, value in col_spec.items():
        if key in ('type', 'value'):
            continue
        if key not in _COLUMN_OPTIONS:
            raise _invalid(f'unknown column option {key!r}')
        options[_COLUMN_OPTIONS[key]] = value
    if 'type' in col_spec:
        return Column(type=type_form(col_spec['type']), **options)
    if 'value' not in col_spec:
        raise _invalid('a column spec needs a type or a value')
    value = resolve(col_spec['value'], model_body.ref)
    if len(options) == 0:
        return value
    return Column(value=value, **options)


def type_form(type_spec: Any) -> TypeForm:
    if not isinstance(type_spec, dict) or type_spec.get('type') not in _TYPE_FORMS:
        raise _invalid(f'unsupported column type {type_spec!r}')
    form = _TYPE_FORMS[type_spec['type']]
    # a union of type forms is one too; mypy types the expression as UnionType
    return cast(TypeForm, form | None) if type_spec.get('nullable', False) else form


def column_type(type_spec: Any) -> ts.ColumnType:
    return ts.ColumnType.normalize_type(type_form(type_spec), allow_builtin_types=False)


def resolve(node: Any, ref: Callable[[str], exprs.Expr | None]) -> Any:
    """The Python value or expression that node denotes; ref() looks up a column by name.

    A column is reached only through ref(), and a method only through the function registry, never through
    getattr(), which would also reach an expression's internals.
    """
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if isinstance(node, list):
        return [resolve(n, ref) for n in node]
    if not isinstance(node, dict):
        raise _invalid(f'malformed expression node {node!r}')
    if 'col' in node:
        name = node['col']
        col = ref(name) if isinstance(name, str) else None
        if col is None:
            raise _invalid(f'unknown column {name!r}')
        return col
    if 'lit' in node:
        return _literal(node)
    if 'op' in node:
        return _operator(node['op'], [resolve(arg, ref) for arg in node.get('args', [])])
    if 'method' in node or 'prop' in node:
        return _method(node, ref)
    if 'call' in node:
        fn = _function(node['call'])
        return fn(*resolve(node.get('args', []), ref), **_kwargs(node, ref))
    if 'cast' in node:
        return _on(node, ref).astype(column_type(node['cast']))
    if 'get' in node:
        return _on(node, ref)[node['get']]
    if 'at' in node:
        return _on(node, ref)[node['at']]
    if 'slice' in node:
        start, stop, step = (node['slice'] + [None, None, None])[:3]
        return _on(node, ref)[slice(start, stop, step)]
    if 'isIn' in node:
        return _on(node, ref).isin(resolve(node['isIn'], ref))
    raise _invalid(f'unknown expression node {node!r}')


def _on(node: dict, ref: Callable[[str], exprs.Expr | None]) -> exprs.Expr:
    base = resolve(node.get('on'), ref)
    if not isinstance(base, exprs.Expr):
        raise _invalid(f'{node!r} applies to {base!r}, which is not an expression')
    return base


def _kwargs(node: dict, ref: Callable[[str], exprs.Expr | None]) -> dict[str, Any]:
    kwargs = node.get('kwargs', {})
    if not isinstance(kwargs, dict):
        raise _invalid(f'malformed kwargs in {node!r}')
    return {key: resolve(value, ref) for key, value in kwargs.items()}


def _literal(node: dict) -> exprs.Expr:
    value = node['lit']
    if isinstance(value, dict) and '$pxt' in value:
        tag, v = value['$pxt'], value.get('v')
        if tag == 'timestamp':
            value = datetime.datetime.fromisoformat(v)
        elif tag == 'date':
            value = datetime.date.fromisoformat(v)
        elif tag == 'uuid':
            value = uuid.UUID(v)
        elif tag == 'int':
            value = int(v)
        else:
            raise _invalid(f'unknown value tag {tag!r}')
    type_spec = node.get('type')
    if type_spec is None:
        return exprs.Literal(value)
    col_type = column_type(type_spec if isinstance(type_spec, dict) else {'type': type_spec})
    return exprs.Literal(value, col_type=col_type)


def _operator(op: Any, args: list[Any]) -> Any:
    if op in _BINARY_OPS:
        if len(args) != 2:
            raise _invalid(f'{op} takes 2 operands, got {len(args)}')
        return _BINARY_OPS[op](*args)
    if op in _VARIADIC_OPS:
        if len(args) < 2:
            raise _invalid(f'{op} takes 2 or more operands, got {len(args)}')
        return functools.reduce(_VARIADIC_OPS[op], args)
    if op in _UNARY_OPS:
        if len(args) != 1:
            raise _invalid(f'{op} takes 1 operand, got {len(args)}')
        return _UNARY_OPS[op](args[0])
    if op == 'round':
        if len(args) not in (1, 2):
            raise _invalid(f'round takes 1 or 2 operands, got {len(args)}')
        return round(*args)
    raise _invalid(f'unknown operator {op!r}')


def _method(node: dict, ref: Callable[[str], exprs.Expr | None]) -> exprs.Expr:
    is_prop = 'prop' in node
    name = node['prop'] if is_prop else node['method']
    base = _on(node, ref)
    if not isinstance(name, str):
        raise _invalid(f'malformed method name {name!r}')
    try:
        method_ref = exprs.MethodRef(base, name)
    except AttributeError as e:
        raise _invalid(f'{base!s} (of type {base.col_type}) has no method or property {name!r}') from e
    # is_property is None on a method declared without it
    if bool(method_ref.fn.is_property) != is_prop:
        kind = 'a property' if method_ref.fn.is_property else 'a method'
        raise _invalid(f'{name!r} is {kind} of {base.col_type}')
    if is_prop:
        return method_ref.fn(method_ref.base_expr)
    return method_ref(*resolve(node.get('args', []), ref), **_kwargs(node, ref))


def _function(path: Any) -> func.Function:
    if not isinstance(path, str) or any(not el.isidentifier() or el.startswith('_') for el in path.split('.')):
        raise _invalid(f'invalid function path {path!r}')
    try:
        fn = resolve_symbol(path)
    except (ImportError, AttributeError):
        fn = None
    if not isinstance(fn, func.Function):
        raise _invalid(f'{path!r} is not a Pixeltable function')
    return fn


def _invalid(msg: str) -> excs.Error:
    return excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, msg)
