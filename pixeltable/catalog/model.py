from __future__ import annotations
import __future__

import dataclasses
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, MutableMapping, NamedTuple, TypedDict

from pixeltable import catalog, exceptions as excs, exprs, func, index, type_system as ts
from pixeltable.env import Env
from pixeltable.query_clauses import SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .globals import MediaValidation, is_valid_identifier
from .table import Table
from .table_version_handle import TableVersionHandle

if TYPE_CHECKING:
    import pixeltable as pxt

# Table methods exposed as class-level operations on the model.
FORWARDED_TABLE_METHODS: frozenset[str] = frozenset(
    (
        'batch_update',
        'collect',
        'count',
        'cursor',
        'delete',
        'describe',
        'distinct',
        'get_metadata',
        'get_versions',
        'group_by',
        'head',
        'history',
        'insert',
        'join',
        'limit',
        'list_views',
        'order_by',
        'recompute_columns',
        'sample',
        'select',
        'show',
        'tail',
        'update',
        'where',
    )
)

# Sanity check to guard against drift in the SDK surface.
for method in FORWARDED_TABLE_METHODS:
    assert hasattr(Table, method), method


@dataclass(frozen=True)
class Column:
    """A column specification used in a TableModel or ViewModel definition."""

    type: type | None = None
    value: Any = None
    primary_key: bool | None = None
    stored: bool | None = None
    media_validation: Literal['on_read', 'on_write'] | None = None
    destination: str | Path | None = None
    custom_metadata: Any = None
    comment: str | None = None

    def to_column_spec(self) -> ColumnSpec:
        column_spec: ColumnSpec = {}
        if self.type is not None:
            column_spec['type'] = self.type
        if self.value is not None:
            column_spec['value'] = self.value
        if self.primary_key is not None:
            column_spec['primary_key'] = self.primary_key
        if self.stored is not None:
            column_spec['stored'] = self.stored
        if self.media_validation is not None:
            column_spec['media_validation'] = self.media_validation
        if self.destination is not None:
            column_spec['destination'] = self.destination
        if self.custom_metadata is not None:
            column_spec['custom_metadata'] = self.custom_metadata
        if self.comment is not None:
            column_spec['comment'] = self.comment
        return column_spec


@dataclass(frozen=True)
class EmbeddingIndex:
    """An embedding index specification used in a TableModel or ViewModel definition."""

    column: Any
    embedding: func.Function | None = None
    string_embed: func.Function | None = None
    image_embed: func.Function | None = None
    audio_embed: func.Function | None = None
    video_embed: func.Function | None = None
    document_embed: func.Function | None = None
    metric: Literal['cosine', 'ip', 'l2'] = 'cosine'
    precision: Literal['fp16', 'fp32'] = 'fp16'


class TableSpec(TypedDict):
    """Table specification from a TableModel or ViewModel."""

    name: str
    display_name: str
    base: ModelQuery | None
    iterator: func.GeneratingFunctionCall | None
    create_default_idxs: bool
    media_validation: MediaValidation
    comment: str | None
    custom_metadata: Any


def _col_type_from_spec(column_spec: ColumnSpec) -> ts.ColumnType:
    """The ColumnType that a column defined by `column_spec` will have."""
    if 'type' in column_spec:
        return ts.ColumnType.normalize_type(column_spec['type'], nullable_default=True, allow_builtin_types=False)
    assert 'value' in column_spec
    return column_spec['value'].col_type


class ModelColumnRef(exprs.Expr):
    """
    A placeholder for a ColumnRef instance, which gets substituted with an actual ColumnRef during
    Table creation or binding.
    """

    name: str

    def __init__(self, name: str, col_type: ts.ColumnType | None = None) -> None:
        super().__init__(col_type if col_type is not None else ts.InvalidType())
        self.name = name
        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'ModelColumnRef({self.name!r})'

    def __str__(self) -> str:
        # Render as a bare column name, identically to the `ColumnRef` this placeholder stands in for.
        return self.name

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('name', self.name)]

    def _equals(self, other: ModelColumnRef) -> bool:
        return self.name == other.name

    def eval(self, data_row: exprs.data_row.DataRow, row_builder: exprs.row_builder.RowBuilder) -> None:
        raise AssertionError('It should never be possible to observe a placeholder in an execution context.')

    def __getattr__(self, item: str) -> Any:
        if item in ('errortype', 'errormsg', 'fileurl', 'localpath'):
            prop = exprs.ColumnPropertyRef.Property[item.upper()]
            return exprs.ColumnPropertyRef(self, prop)  # type: ignore[arg-type]
        return super().__getattr__(item)

    # Placeholders are serialized so that a pre-substitution model can be shipped to whichever catalog creates
    # the table (e.g. a proxied catalog). We hook the standard `_as_dict()`/`_from_dict()` (not `as_dict()`/
    # `from_dict()`) so that `Expr.from_dict()` reconstructs placeholders nested inside value expressions via its
    # generic `_classname` dispatch; the registration below makes that dispatch resolve this class. `name` is
    # what substitution matches on; `col_type` keeps the enclosing expression well-typed until substituted away.

    def _as_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'col_type': self.col_type.as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, _components: list[exprs.Expr], _tbl_versions: Any = None) -> ModelColumnRef:
        return cls(d['name'], ts.ColumnType.from_dict(d['col_type']))


# `Expr.from_dict()` resolves expression classes by name from the `pixeltable.exprs` namespace. Register
# `ModelColumnRef` there so value expressions carrying placeholders can be deserialized by whichever
# catalog creates the table (e.g. a proxied catalog's daemon), even though the class lives in `catalog.model`.
exprs.ModelColumnRef = ModelColumnRef  # type: ignore[attr-defined]


@dataclasses.dataclass
class ModelQuery:
    """
    A placeholder query used in ViewModel definitions,
    which gets substituted with an actual Query during Table creation or binding.
    """

    from_clause: type[TableModelMeta]
    select_clause: tuple[tuple[Any, ...], dict[str, Any]] | None
    where_clause: exprs.Expr | None
    group_by_clause: list[exprs.Expr] | None
    grouping_tbl: type[TableModelMeta] | None
    order_by_clause: list[tuple[exprs.Expr, bool]] | None
    limit_val: exprs.Expr | None
    offset_val: exprs.Expr | None
    sample_clause: SampleClause | None

    def __init__(
        self,
        from_clause: type[TableModelMeta],
        select_clause: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
        where_clause: exprs.Expr | None = None,
        group_by_clause: list[exprs.Expr] | None = None,
        grouping_tbl: type[TableModelMeta] | None = None,
        order_by_clause: list[tuple[exprs.Expr, bool]] | None = None,
        limit_val: exprs.Expr | None = None,
        offset_val: exprs.Expr | None = None,
        sample_clause: SampleClause | None = None,
    ) -> None:
        self.from_clause = from_clause
        self.select_clause = select_clause
        self.where_clause = where_clause
        self.group_by_clause = group_by_clause
        self.grouping_tbl = grouping_tbl
        self.order_by_clause = order_by_clause
        self.limit_val = limit_val
        self.offset_val = offset_val
        self.sample_clause = sample_clause

    def select(self, *items: Any, **named_items: Any) -> ModelQuery:
        if self.select_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`select()` list already specified in `ViewModel` base query.'
            )
        for name in named_items:
            if not is_valid_identifier(name):
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid name: {name}')
        if len(items) + len(named_items) == 0:
            return self
        return dataclasses.replace(self, select_clause=(items, named_items))

    def where(self, pred: exprs.Expr) -> ModelQuery:
        if self.where_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`where()` clause already specified in `ViewModel` base query.'
            )
        return dataclasses.replace(self, where_clause=pred)

    def group_by(self, *grouping_items: exprs.Expr) -> ModelQuery:
        if self.group_by_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`group_by()` clause already specified in `ViewModel` base query.'
            )
        return dataclasses.replace(self, group_by_clause=list(grouping_items))

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> ModelQuery:
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend((e.copy(), asc) for e in expr_list)
        return dataclasses.replace(self, order_by_clause=order_by_clause)

    def limit(self, n: int, offset: int | None = None) -> ModelQuery:
        if self.limit_val is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`limit()` clause already specified in `ViewModel` base query.'
            )
        limit_val = exprs.Expr.from_object(n)
        offset_val = exprs.Expr.from_object(offset) if offset is not None else None
        return dataclasses.replace(self, limit_val=limit_val, offset_val=offset_val)

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> ModelQuery:
        if self.sample_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`sample()` clause already specified in `ViewModel` base query.'
            )
        stratify_exprs: list[exprs.Expr] = []
        if stratify_by is not None:
            if isinstance(stratify_by, exprs.Expr):
                stratify_by = [stratify_by]
            stratify_exprs = list(stratify_by)
        sample_clause = SampleClause(None, n, n_per_stratum, fraction, seed, stratify_exprs)
        return dataclasses.replace(self, sample_clause=sample_clause)

    def _bind(self, binding_root: str) -> 'pxt.Query':
        tbl: Table = self.from_clause._bind(binding_root)  # type: ignore[arg-type]
        subst_dict: dict[exprs.Expr, exprs.Expr] = {}
        for col_name in tbl.columns():
            subst_dict[ModelColumnRef(col_name)] = getattr(tbl, col_name)

        q: pxt.Query
        if self.select_clause is None:
            q = tbl.select()
        else:
            items, named_items = self.select_clause
            items = [expr.substitute(subst_dict) for expr in items]
            named_items = {name: expr.substitute(subst_dict) for name, expr in named_items.items()}
            q = tbl.select(*items, **named_items)

        if self.where_clause is not None:
            where_clause = self.where_clause.substitute(subst_dict)
            q = q.where(where_clause)

        if self.group_by_clause is not None:
            group_by_clause = [expr.substitute(subst_dict) for expr in self.group_by_clause]
            q = q.group_by(*group_by_clause)

        if self.grouping_tbl is not None:
            grouping_tbl = self.grouping_tbl._bind(binding_root)  # type: ignore[arg-type]
            q = q.group_by(grouping_tbl)

        if self.order_by_clause is not None:
            order_by_clause = [(expr.substitute(subst_dict), asc) for (expr, asc) in self.order_by_clause]
            for expr, asc in order_by_clause:
                q = q.order_by(expr, asc=asc)

        if self.limit_val is not None:
            limit_val = self.limit_val.substitute(subst_dict)
            offset_val = self.offset_val.substitute(subst_dict) if self.offset_val is not None else None
            q = q.limit(limit_val, offset=offset_val)  # type: ignore[arg-type]

        if self.sample_clause is not None:
            q = q.sample(
                n=self.sample_clause.n,
                n_per_stratum=self.sample_clause.n_per_stratum,
                fraction=self.sample_clause.fraction,
                seed=self.sample_clause.seed,
                stratify_by=[expr.substitute(subst_dict) for expr in self.sample_clause.stratify_exprs],
            )

        return q


class _AnnotationRecorder(dict):
    """
    Used to override the default behavior of a class namespace's `__annotations__` dict, so that we can register
    bare annotations promptly as placeholder columns in the class namespace, in the order they are declared.
    """

    namespace: _ModelNamespace

    def __init__(self, namespace: _ModelNamespace) -> None:
        super().__init__()
        self.namespace = namespace

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            # Register the type annotation in the namespace
            self.namespace.set_col_type(key, value)
        super().__setitem__(key, value)


class _ModelNamespace(dict):
    """
    Class namespace that manages placeholder column references, ensuring that all declarations (bare annotations,
    computed column expressions, Column and EmbeddingIndex specifications) are registered promptly and in the exact
    order of declaration.
    """

    table_spec: TableSpec
    known_cols: dict[str, ColumnSpec]
    known_idxs: dict[str, EmbeddingIndex]

    # Names that are produced by the base query or iterator; these cannot be redefined in the model.
    reserved_cols: dict[str, Literal['base query', 'iterator']]

    # The scope in which the class body is defined; used to evaluate stringized type annotations (see
    # `set_col_type`). Populated from the defining frame in `TableModelMeta.__prepare__`.
    eval_globals: dict[str, Any]
    eval_locals: dict[str, Any]

    def __init__(self, table_spec: TableSpec, eval_globals: dict[str, Any], eval_locals: dict[str, Any]) -> None:
        super().__init__()

        self.table_spec = table_spec
        self.known_cols = {}
        self.known_idxs = {}
        self.reserved_cols = {}
        self.eval_globals = eval_globals
        self.eval_locals = eval_locals

        # Pre-seed __annotations__ so the compiler routes bare annotations through
        # our recorder rather than a plain dict it would otherwise create.
        self['__annotations__'] = _AnnotationRecorder(self)

    def __setitem__(self, key: str, value: Any) -> None:
        if key.startswith('__') and key.endswith('__'):
            # "Dunder" methods and attributes are not table columns.
            super().__setitem__(key, value)
        elif not is_valid_identifier(key):
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Invalid column name: {key!r}')
        else:
            self.set_col_value(key, value)

    def add_reserved_column_ref(
        self, name: str, col_type: ts.ColumnType, kind: Literal['base query', 'iterator']
    ) -> None:
        """Add `name` as a reserved column (it is resolvable in the class body, and its symbol cannot be reused,
        but it does not have a ColumnSpec and will not be included in the list of columns for the view to create).
        """
        self.reserved_cols[name] = kind
        super().__setitem__(name, ModelColumnRef(name, col_type))

    def _check_reserved(self, name: str) -> None:
        if name in self.reserved_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{name!r} is already defined by the {self.reserved_cols[name]}; it cannot be redeclared.',
            )

    def set_col_value(self, name: str, value: Any) -> None:
        self._check_reserved(name)
        if isinstance(value, EmbeddingIndex):
            if name in self.known_cols or name in self.known_idxs:
                raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Index {name!r}: duplicate definition.')
            self.known_idxs[name] = value

        else:
            if name in self.known_cols or name in self.known_idxs:
                raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Column {name!r}: duplicate definition.')
            spec: ColumnSpec
            if isinstance(value, Column):
                spec = value.to_column_spec()
                if ('type' in spec) == ('value' in spec):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'Column specification for {name!r} must define `type` or `value`, but not both',
                    )
            else:
                # Computed column expression.
                expr = exprs.Expr.from_object(value)
                if expr is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'Column {name!r}: invalid value (not a literal or expression recognized by Pixeltable).',
                    )
                spec = {'value': expr}
            self.known_cols[name] = spec
            # Add the column to the namespace so that it can be referenced in subsequent expressions in the class body.
            super().__setitem__(name, ModelColumnRef(name, _col_type_from_spec(spec)))

    def set_col_type(self, name: str, type_: Any) -> None:
        self._check_reserved(name)
        if isinstance(type_, str):
            # Under `from __future__ import annotations` (PEP 563) -- and mandatory on Python 3.14+, where
            # PEP 649 otherwise defers annotation evaluation entirely -- annotations arrive as strings. Evaluate
            # the string in the scope where the model class is defined to recover the actual type.
            try:
                type_ = eval(type_, self.eval_globals, self.eval_locals)
            except Exception as exc:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Could not resolve the type annotation {type_!r} for column {name!r}: {exc}',
                ) from exc
        type_ = ts.ColumnType.normalize_type(type_, nullable_default=True, allow_builtin_types=False)
        if name in self.known_idxs:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Cannot set a type annotation for index {name!r}.')
        if name in self.known_cols:
            # We previously processed this column via `set_col_value()`. Sanity check the type.
            if _col_type_from_spec(self.known_cols[name]) != type_:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'Conflicting type annotation for column {name!r}.'
                )
            return
        # Bare annotation (`col: SomeType`): record the spec and make the name referenceable in the body.
        self.known_cols[name] = {'type': type_}  # type: ignore[typeddict-item]
        super().__setitem__(name, ModelColumnRef(name, type_))


class TableModelMeta(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    __table_spec__: TableSpec
    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, EmbeddingIndex]
    __bound_table__: Table | None

    _binding_root: str | None

    @classmethod
    def __prepare__(  # type: ignore[override]
        mcs,  # noqa: N804  # Neither mypy nor ruff seems to understand metaclasses.
        cls_name: str,
        bases: tuple[type, ...],
        /,
        name: str,
        base: 'TableModelMeta | ModelQuery | None' = None,
        iterator: func.GeneratingFunctionCall | None = None,
        create_default_idxs: bool = True,
        media_validation: Literal['on_read', 'on_write'] = 'on_write',
        comment: str | None = None,
        custom_metadata: Any = None,
    ) -> MutableMapping[str, object]:
        if len(bases) == 0:
            # This is a model_base() class. No special processing.
            return super().__prepare__(cls_name, bases)
        elif len(bases) > 1 or '__registered_models__' not in bases[0].__dict__:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                'Pixeltable schemas must be direct subclasses of a model_base(). '
                '(Use `pxt.model_base()` to create one.)',
            )
        else:
            display_name = f'model `{cls_name}`'

            # Validate table name
            tbl_name = name
            if not isinstance(tbl_name, str) or not is_valid_identifier(tbl_name, allow_hyphens=True):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'{display_name}: `name` must be a valid Pixeltable identifier.'
                )
            # scoped to this model_base() so a long-lived process can re-load a schema without colliding with a
            # previous load; duplicate names within one schema are still caught
            base_models = bases[0].__registered_models__  # type: ignore[attr-defined]
            if tbl_name in base_models:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'{display_name} has name {tbl_name!r}, but that name was '
                    f'previously used by `{base_models[tbl_name].__name__}`.',
                )

            # Validate base
            if base is not None:
                if isinstance(base, ModelQuery):
                    pass
                elif isinstance(base, TableModelMeta):
                    base = base.select()  # convert to a ModelQuery
                else:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `base` must be a valid base table reference '
                        f'(another Pixeltable model, or a query over a model).',
                    )
                assert isinstance(base, ModelQuery)
                base_model = base.from_clause
                if len(base_model.__bases__) == 0 or base_model.__bases__[0] is not bases[0]:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `base` must reference a Pixeltable model with the same '
                        f'`model_base()` as `{cls_name}`.',
                    )

            # Validate iterator
            if iterator is not None:
                if base is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `iterator` can only be specified together with a `base`.',
                    )
                if not isinstance(iterator, func.GeneratingFunctionCall):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `iterator` must be a valid iterator reference.',
                    )

            media_validation_ = MediaValidation.validated(media_validation, '`media_validation`')

            # Capture the scope in which the class body is being defined, so that stringized type annotations
            # (see `_ModelNamespace.set_col_type`) can be evaluated. `sys._getframe(1)` is the frame executing
            # the `class ...:` statement (`__build_class__` is a C function and creates no frame).
            caller = sys._getframe(1)

            # On Python 3.14+, annotations are not evaluated eagerly (PEP 649), so the model's column annotations
            # would be dropped and body references to them would raise `NameError` *before* we ever reach
            # `__new__`. `from __future__ import annotations` restores the eager (stringized) behavior the model
            # relies on. Detect its absence here -- before the body runs -- and fail with an actionable message.
            future_annotations = bool(caller.f_code.co_flags & __future__.annotations.compiler_flag)
            if sys.version_info >= (3, 14) and not future_annotations:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'{display_name}: On Python 3.14+, you must use `from __future__ import annotations` '
                    'in your module in order to declare a TableModel.',
                )

            namespace = _ModelNamespace(
                {
                    'name': tbl_name,
                    'display_name': display_name,
                    'base': base,
                    'iterator': iterator,
                    'create_default_idxs': create_default_idxs,
                    'media_validation': media_validation_,
                    'comment': comment,
                    'custom_metadata': custom_metadata,
                },
                eval_globals=caller.f_globals,
                eval_locals=caller.f_locals,
            )

            if base is not None and base.select_clause is not None:
                # Make the select list's named columns referenceable in the body.
                for col_name, expr in base.select_clause[1].items():
                    assert is_valid_identifier(col_name)  # since it must be a Python symbol
                    namespace.add_reserved_column_ref(col_name, expr.col_type, 'base query')

            if iterator is not None:
                # Likewise for the iterator's outputs: referenceable, but created by the iterator.
                for col_name, output in iterator.outputs.items():
                    assert is_valid_identifier(col_name)
                    namespace.add_reserved_column_ref(col_name, output.col_type, 'iterator')

            return namespace

    def __new__(
        mcs, cls_name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMeta:
        if len(bases) == 0:
            # This is a model_base(); no special processing.
            return super().__new__(mcs, cls_name, bases, namespace)

        assert isinstance(namespace, _ModelNamespace)

        if len(namespace.known_cols) == 0 and namespace.table_spec['base'] is None:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, 'Empty table schema not allowed.')

        # "normalize" the namespace to a plain dict; at this point, we're done with the special namespace treatment
        namespace_dict = dict(namespace)
        namespace_dict['__table_spec__'] = namespace.table_spec
        namespace_dict['__columns__'] = namespace.known_cols
        namespace_dict['__indexes__'] = namespace.known_idxs
        namespace_dict['__bound_table__'] = None
        namespace_dict['_binding_root'] = None

        cls = super().__new__(mcs, cls_name, bases, namespace_dict)
        assert hasattr(bases[0], '__registered_models__')  # This was checked in __prepare__()
        bases[0].__registered_models__[namespace.table_spec['name']] = cls
        return cls

    def _resolve_tbl(cls, binding_root: str, if_not_exists: Literal['error', 'ignore']) -> Table | None:
        import pixeltable as pxt

        if cls._binding_root is not None and binding_root != cls._binding_root:
            raise excs.RequestError(
                excs.ErrorCode.ALREADY_BOUND,
                f'Cannot bind `{cls.__name__}` at {binding_root!r}: it is already bound at {cls._binding_root!r}.',
            )

        bound_path = f'{binding_root}{cls.__table_spec__["name"]}'
        return pxt.get_table(bound_path, if_not_exists=if_not_exists)

    @property
    def is_bound(cls) -> bool:
        return cls._binding_root is not None

    @classmethod
    def _normalize_binding_root(cls, binding_root: str) -> str:
        if binding_root.endswith('/'):
            binding_root = binding_root[:-1]
        _ = catalog.Path.parse(binding_root, allow_empty_path=True)  # validate
        if len(binding_root) > 0:
            binding_root += '/'
        return binding_root

    def _bind(cls, binding_root: str = '') -> pxt.Table:
        binding_root = cls._normalize_binding_root(binding_root)

        tbl = cls._resolve_tbl(binding_root, if_not_exists='error')

        if cls.is_bound:
            return tbl

        else:
            col_refs = {col_name: getattr(tbl, col_name) for col_name in tbl.columns()}

            # Table ops succeeded; now update the class.
            for col_name, col_ref in col_refs.items():
                setattr(cls, col_name, col_ref)
            cls._binding_root = binding_root
            return tbl

    def _create(cls, binding_root: str = '') -> tuple[Table, bool]:
        """Returns the table and whether it was created now (False if it already existed)."""
        binding_root = cls._normalize_binding_root(binding_root)

        if cls.is_bound:
            tbl = cls._resolve_tbl(binding_root, if_not_exists='error')
            assert tbl is not None
            return tbl, False

        table_spec: TableSpec = cls.__table_spec__

        # Bind the base query to an actual Query over the (already-existing) base table. This happens client-side,
        # outside any transaction; the resulting Query references real columns and so is serializable to whichever
        # catalog owns the table being created.
        base: pxt.Query | None = None
        if table_spec['base'] is not None:
            base = table_spec['base']._bind(binding_root)

        # The model's own column specs, with `type` annotations resolved to ColumnTypes (so they're serializable
        # for a proxied catalog). Computed `value` expressions still carry `ModelColumnRef`s referencing
        # sibling and base columns; those are substituted by the catalog that owns the table (create_from_model).
        columns: dict[str, ColumnSpec] = {}
        for name, col_spec in cls.__columns__.items():
            spec = col_spec.copy()
            if 'type' in spec:
                spec['type'] = ts.ColumnType.normalize_type(  # type: ignore[typeddict-item]
                    spec['type'], nullable_default=True, allow_builtin_types=False
                )
            columns[name] = spec

        bound_path = f'{binding_root}{table_spec["name"]}'
        tbl_path = catalog.Path.parse(bound_path)

        cat = get_runtime().get_catalog(tbl_path)
        tbl, was_created = cat.create_from_model(
            path=tbl_path,
            columns=columns,
            display_name=table_spec['display_name'],
            create_default_idxs=table_spec['create_default_idxs'],
            media_validation=table_spec['media_validation'],
            comment=table_spec['comment'],
            custom_metadata=table_spec['custom_metadata'],
            iterator=table_spec['iterator'],
            base=base,
            embedding_idxs=cls.__indexes__,
        )

        if was_created:
            Env.get().console_logger.info(f'Created {tbl._path()!r} from {table_spec["display_name"]}.')

        return cls._bind(binding_root), was_created

    def __getattr__(cls, item: str) -> Any:
        if item in FORWARDED_TABLE_METHODS:
            if not cls.is_bound and hasattr(ModelQuery, item):
                # This model is not bound to a table, but the desired operation is accessible via a placeholder query.
                return getattr(ModelQuery(cls), item)  # type: ignore[arg-type]
            else:
                try:
                    return getattr(cls.table, item)
                except excs.RequestError as exc:
                    raise AttributeError(f'{item}(): {exc}') from exc
        return super().__getattribute__(item)

    @property
    def table(cls) -> Table:
        """The underlying [`Table`][pixeltable.Table] this model is bound to."""
        if not cls.is_bound:
            raise excs.RequestError(
                excs.ErrorCode.NOT_BOUND,
                f'`{cls.__name__}` is not yet bound to an actual table. You must first call '
                f'`{cls.__name__}.bind()`, `{cls.__name__}.create()`, `pxt.bind_all()`, or `pxt.create_all()`.',
            )
        return cls._resolve_tbl(cls._binding_root, if_not_exists='error')


def prepare_model(
    tbl_handle: TableVersionHandle,
    columns: dict[str, ColumnSpec],
    display_name: str,
    media_validation: MediaValidation,
    iterator: func.GeneratingFunctionCall | None,
    base: 'pxt.Query | None',
    embedding_idxs: dict[str, EmbeddingIndex],
) -> tuple[
    func.GeneratingFunctionCall | None, list[catalog.Column], list[tuple[catalog.Column, str | None, index.IndexBase]]
]:
    """
    Given model declarations in the form of columns, base, iterator, and embedding_idx specifications, along with
    the relevant metadata, assembles lists of additional columns and additional indices to be created in the table.
    The outputs will be fully resolved (ModelColumnRefs replaced with actual ColumnRefs and EmbeddingIndex
    dataclass instances replaced with actual instances of index.IndexBase).

    Returns: a tuple of (rebound iterator, additional columns, additional idxs).
    """

    # View columns always go in a specific order:
    # - iterator columns first
    # - then columns from the base query's select_list
    #     (but not if it's a select(*): then just inherit the base table's columns)
    # - finally, the view's additional_columns.

    # Create a counter to track column ids.
    next_col_id = itertools.count()

    # A registry of visible columns of the table (base table/query columns, iterator columns,
    # and additional columns).
    visible_cols: dict[str, catalog.Column] = {}

    # A substitution dictionary resolving ModelColumnRefs to actual ColumnRefs; we'll build this up incrementally
    # as we process the model's columns.
    subst_dict: dict[exprs.Expr, exprs.Expr] = {}

    # First the iterator columns, if present.
    if iterator is not None:
        # Rebind the iterator, resolving its argument references against the base table.
        assert base is not None
        base_tbl_subst_dict: dict[exprs.Expr, exprs.Expr] = {
            ModelColumnRef(col.name): exprs.ColumnRef(col.column_version_md()) for col in base._first_tbl.columns()
        }
        subst_args = [arg.substitute(base_tbl_subst_dict) for arg in iterator.args]
        subst_kwargs = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.kwargs.items()}
        subst_bound_args = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.bound_args.items()}
        iterator = func.GeneratingFunctionCall(
            iterator.it, subst_args, subst_kwargs, subst_bound_args, iterator.outputs, iterator.validation_error
        )
        # Build substitutions for the iterator's output columns.
        for name, output in iterator.outputs.items():
            catalog_col = catalog.Column.create(name, {'type': output.col_type, 'stored': output.is_stored})  # type: ignore[arg-type]
            catalog_col.id = next(next_col_id)
            catalog_col.tbl_handle = tbl_handle
            visible_cols[name] = catalog_col
            subst_dict[ModelColumnRef(name)] = exprs.ColumnRef(
                catalog_col.column_version_md(), perform_validation=(media_validation == 'on_read')
            )

    if base is not None:
        # Build substitutions for the base table/query's columns.
        if base.select_list is None:
            # select(*): all visible columns from the base table
            for col in base._first_tbl.columns():
                # Iterator column names take precedence over base table column names in the model namespace, so
                # only update the substitution dicts if the name isn't already present.
                if col.name not in visible_cols:
                    visible_cols[col.name] = col
                    ref = exprs.ColumnRef(col.column_version_md())
                    subst_dict[ModelColumnRef(col.name)] = ref
        else:
            # explicit select list: new columns will be created that represent the selected expressions.
            for expr, select_name in base.select_list:
                col_name: str | None
                if select_name is not None:
                    # The select list has an explicit name for this expression as a kwarg; use it.
                    col_name = select_name
                elif isinstance(expr, exprs.ColumnRef):
                    # It's an unnamed column reference; use the name of the referenced column as a fallback.
                    col_name = expr.column_md.name
                else:
                    # It's a compound expression with no explicit name. A name will be assigned when the table
                    # is created, but it's anonymous to the TableModel.
                    # TODO: Revisit this behavior. Should we be allowing unnamed compound expressions in the
                    #     first place?
                    col_name = None

                # Increment the `id` whether or not this column is visible to the model, to ensure we have
                # ids that are consistent at table creation time.
                id = next(next_col_id)
                if col_name is not None:
                    # Column names that arrived via an explicit select list take precedence over iterator column
                    # names in the model namespace, so here we always update the dicts.
                    catalog_col = catalog.Column.create(col_name, expr.col_type)
                    catalog_col.id = id
                    catalog_col.tbl_handle = tbl_handle
                    visible_cols[col_name] = catalog_col
                    subst_dict[ModelColumnRef(col_name)] = exprs.ColumnRef(
                        catalog_col.column_version_md(), perform_validation=(media_validation == 'on_read')
                    )

    # Process any additional columns specified in the view model body.
    additional_cols: list[catalog.Column] = []
    for name, spec in columns.items():
        subst_spec = spec.copy()
        if 'value' in subst_spec:
            subst_spec['value'] = subst_spec['value'].substitute(subst_dict)
            residual_placeholders = list(subst_spec['value'].subexprs(ModelColumnRef))
            if len(residual_placeholders) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} in {display_name} references columns that are not in '
                    f"the model's scope: {[c.name for c in residual_placeholders]}",
                )
        catalog_col = catalog.Column.create(name, subst_spec)
        catalog_col.tbl_handle = tbl_handle
        catalog_col.id = next(next_col_id)
        additional_cols.append(catalog_col)
        visible_cols[name] = catalog_col
        subst_dict[ModelColumnRef(name, catalog_col.col_type)] = exprs.ColumnRef(
            catalog_col.column_version_md(),
            perform_validation=subst_spec.get('media_validation', media_validation) == 'on_read',
        )

    # Resolve each declared embedding index against the model's visible columns.
    resolved_idxs: list[tuple[catalog.Column, str | None, index.IndexBase]] = []
    for idx_name, idx_spec in embedding_idxs.items():
        if not isinstance(idx_spec.column, ModelColumnRef):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'Embedding index {idx_name!r} in {display_name} has an invalid column reference.',
            )
        col_name = idx_spec.column.name
        if col_name not in visible_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'Embedding index {idx_name!r} in {display_name} references unknown column {col_name!r}.',
            )
        idx = index.EmbeddingIndex(
            metric=idx_spec.metric,
            precision=idx_spec.precision,
            embed=idx_spec.embedding,
            string_embed=idx_spec.string_embed,
            image_embed=idx_spec.image_embed,
            audio_embed=idx_spec.audio_embed,
            video_embed=idx_spec.video_embed,
            document_embed=idx_spec.document_embed,
            column=visible_cols[col_name],
        )
        resolved_idxs.append((visible_cols[col_name], idx_name, idx))

    return iterator, additional_cols, resolved_idxs


class ValidationResults(NamedTuple):
    model_cls: TableModelMeta
    tbl_exists: bool
    model_kind: str  # 'table' or 'view'
    existing_kind: str | None  # None if the table does not yet exist
    model_iterator: str | None  # display string of the model's iterator call, or None if it has no iterator
    existing_iterator: str | None  # None if the table does not yet exist or has no iterator
    new_columns: list[str]
    dropped_columns: list[str]
    new_indexes: list[str]
    dropped_indexes: list[str]

    @property
    def has_changes(self) -> bool:
        return self.has_fatal_changes or self.has_destructive_changes or self.has_benign_changes

    @property
    def has_fatal_changes(self) -> bool:
        return self.tbl_exists and (
            self.model_kind != self.existing_kind or self.model_iterator != self.existing_iterator
        )

    @property
    def has_destructive_changes(self) -> bool:
        return self.tbl_exists and (len(self.dropped_columns) > 0 or len(self.dropped_indexes) > 0)

    @property
    def has_benign_changes(self) -> bool:
        return not self.tbl_exists or len(self.new_columns) > 0 or len(self.new_indexes) > 0


def validate_models(registered_models: dict[str, TableModelMeta], binding_root: str) -> dict[str, ValidationResults]:
    """
    Analyze each registered model against the current catalog state, summarizing the schema changes that creating
    the models would entail, along with any incompatibilities with an already-existing table of the same name.
    This is purely informational: it neither modifies the catalog nor raises on incompatibilities.
    """
    binding_root = TableModelMeta._normalize_binding_root(binding_root)
    results: dict[str, ValidationResults] = {}

    for name, model in registered_models.items():
        model_cols = set(model.__columns__.keys())
        model_idxs = set(model.__indexes__.keys())
        # A model with no base is a table, otherwise a view.
        model_kind = 'table' if model.__table_spec__['base'] is None else 'view'
        # The model's iterator still carries `ModelColumnRef` placeholders, but those render identically to the
        # `ColumnRef`s in a stored (bound) iterator call, so the display strings are directly comparable.
        iterator = model.__table_spec__['iterator']
        model_iterator = None if iterator is None else iterator.display_str()

        existing = model._resolve_tbl(binding_root, if_not_exists='ignore')
        if existing is None:
            results[name] = ValidationResults(
                model_cls=model,
                tbl_exists=False,
                new_columns=sorted(model_cols),
                dropped_columns=[],
                new_indexes=sorted(model_idxs),
                dropped_indexes=[],
                model_kind=model_kind,
                existing_kind=None,
                model_iterator=model_iterator,
                existing_iterator=None,
            )
            continue

        existing_md = existing.get_metadata()
        # Restrict the existing columns to those defined in this table (i.e. not inherited from a base) and not
        # produced by an iterator, so that they line up with the model's own declared columns.
        existing_cols = {
            col_name
            for col_name, col_md in existing_md['columns'].items()
            if col_md['defined_in'] == existing_md['name'] and not col_md['is_iterator_col']
        }
        existing_idxs = set(idx_name for idx_name, info in existing_md['indices'].items() if info['index_type'] == 'embedding')

        # TODO: validate table properties (comment, custom_metadata, media_validation, primary_key, etc.)
        # TODO: validate base table query

        results[name] = ValidationResults(
            model_cls=model,
            tbl_exists=True,
            new_columns=sorted(model_cols - existing_cols),
            dropped_columns=sorted(existing_cols - model_cols),
            new_indexes=sorted(model_idxs - existing_idxs),
            dropped_indexes=sorted(existing_idxs - model_idxs),
            model_kind=model_kind,
            existing_kind=existing_md['kind'],
            model_iterator=model_iterator,
            existing_iterator=existing_md['iterator_call'],
        )

    return results


def _format_diff(name: str, r: ValidationResults) -> list[str]:
    """Human-readable lines describing how the model named `name` differs from the current catalog state."""
    if not r.tbl_exists:
        return [f'{r.model_kind.capitalize()} {name!r} (from model `{r.model_cls.__name__}`) does not yet exist.']

    detail: list[str] = []
    if r.model_kind != r.existing_kind:
        detail.append(
            f'  kind mismatch (FATAL): `{r.model_cls.__name__}` specifies a {r.model_kind}, '
            f'but {name!r} is a {r.existing_kind}'
        )
    if r.model_iterator != r.existing_iterator:
        detail.append('  iterator mismatch (FATAL):')
        detail.append(f'    model iterator   : {r.model_iterator}')
        detail.append(f'    existing iterator: {r.existing_iterator}')
    if len(r.new_columns) > 0:
        detail.append('  the following columns are new to the model, and will be ADDED:')
        for col_name in r.new_columns:
            detail.append(f'    {col_name!r} = {r.model_cls.__columns__[col_name]}')
    if len(r.dropped_columns) > 0:
        detail.append('  the following columns are no longer in the model, and will be DROPPED:')
        for col_name in r.dropped_columns:
            detail.append(f'    {col_name!r}')
    if len(r.new_indexes) > 0:
        detail.append('  the following indexes are new to the model, and will be ADDED:')
        for idx_name in r.new_indexes:
            detail.append(f'    {idx_name!r} = {r.model_cls.__indexes__[idx_name]}')
    if len(r.dropped_indexes) > 0:
        detail.append('  the following indexes are no longer in the model, and will be DROPPED:')
        for idx_name in r.dropped_indexes:
            detail.append(f'    {idx_name!r}')

    if len(detail) == 0:
        return []

    return [f'{r.model_kind.capitalize()} {name!r} (from model `{r.model_cls.__name__}`) has differences:', *detail]


def model_base(cls_name: str = 'TableModel') -> type[TableModelMeta]:
    # mypy fundamentally does not understand metaclasses.
    cls = TableModelMeta(cls_name, (), {}, name='')
    registered_models: dict[str, TableModelMeta] = {}
    cls.__registered_models__ = registered_models  # type: ignore[attr-defined]

    def _bind_all(binding_root: str = '') -> None:
        for model in registered_models.values():
            model._bind(binding_root)

    def _create_all(binding_root: str = '') -> tuple[list[str], list[str]]:
        """Returns (created, existing): absolute paths of tables created now and those that already exist."""
        created: list[str] = []
        existed: list[str] = []

        # Gather validation results for all models before creating anything. For now these are informational only;
        # we neither act on them nor raise in response.
        _ = validate_models(registered_models, binding_root)

        for model in registered_models.values():
            tbl, was_created = model._create(binding_root)
            (created if was_created else existed).append(str(tbl._path()))

        return created, existed

    def _diff_all(binding_root: str = '') -> None:
        results = validate_models(registered_models, binding_root)
        lines: list[str] = []
        for name, r in results.items():
            lines.extend(_format_diff(name, r))
        Env.get().console_logger.info('\n'.join(lines) if len(lines) > 0 else 'Catalog is up to date.')

    cls.bind_all = _bind_all  # type: ignore[attr-defined]
    cls.create_all = _create_all  # type: ignore[attr-defined]
    cls.diff_all = _diff_all  # type: ignore[attr-defined]

    return cls  # type: ignore[return-value]
