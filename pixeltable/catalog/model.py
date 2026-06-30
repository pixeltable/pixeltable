from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, MutableMapping, NamedTuple, TypedDict

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.env import Env
from pixeltable.query_clauses import SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .globals import MediaValidation, is_valid_identifier
from .table import Table

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
        'sync',
        'tail',
        'unlink_external_stores',
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
    base: _PlaceholderQuery | None
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


class _PlaceholderColumnRef(exprs.Expr):
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
        return f'_PlaceholderColumnRef({self.name!r})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('name', self.name)]

    def _equals(self, other: _PlaceholderColumnRef) -> bool:
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
    def _from_dict(cls, d: dict, _components: list[exprs.Expr], _tbl_versions: Any = None) -> _PlaceholderColumnRef:
        return cls(d['name'], ts.ColumnType.from_dict(d['col_type']))


# `Expr.from_dict()` resolves expression classes by name from the `pixeltable.exprs` namespace. Register
# `_PlaceholderColumnRef` there so value expressions carrying placeholders can be deserialized by whichever
# catalog creates the table (e.g. a proxied catalog's daemon), even though the class lives in `catalog.model`.
exprs._PlaceholderColumnRef = _PlaceholderColumnRef  # type: ignore[attr-defined]


@dataclasses.dataclass
class _PlaceholderQuery:
    """
    A placeholder query used in ViewModel definitions,
    which gets substituted with an actual Query during Table creation or binding.
    """

    from_clause: type[TableModelMetaclass]
    select_clause: tuple[tuple[Any, ...], dict[str, Any]] | None
    where_clause: exprs.Expr | None
    group_by_clause: list[exprs.Expr] | None
    grouping_tbl: type[TableModelMetaclass] | None
    order_by_clause: list[tuple[exprs.Expr, bool]] | None
    limit_val: exprs.Expr | None
    offset_val: exprs.Expr | None
    sample_clause: SampleClause | None

    def __init__(
        self,
        from_clause: type[TableModelMetaclass],
        select_clause: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
        where_clause: exprs.Expr | None = None,
        group_by_clause: list[exprs.Expr] | None = None,
        grouping_tbl: type[TableModelMetaclass] | None = None,
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

    def select(self, *items: Any, **named_items: Any) -> _PlaceholderQuery:
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

    def where(self, pred: exprs.Expr) -> _PlaceholderQuery:
        if self.where_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`where()` clause already specified in `ViewModel` base query.'
            )
        return dataclasses.replace(self, where_clause=pred)

    def group_by(self, *grouping_items: exprs.Expr) -> _PlaceholderQuery:
        if self.group_by_clause is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`group_by()` clause already specified in `ViewModel` base query.'
            )
        return dataclasses.replace(self, group_by_clause=list(grouping_items))

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> _PlaceholderQuery:
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend((e.copy(), asc) for e in expr_list)
        return dataclasses.replace(self, order_by_clause=order_by_clause)

    def limit(self, n: int, offset: int | None = None) -> _PlaceholderQuery:
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
    ) -> _PlaceholderQuery:
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

    def bind(self, binding_root: str) -> 'pxt.Query':
        tbl: Table = self.from_clause.bind(binding_root)  # type: ignore[arg-type]
        subst_dict: dict[exprs.Expr, exprs.Expr] = {}
        for col_name in tbl.columns():
            subst_dict[_PlaceholderColumnRef(col_name)] = getattr(tbl, col_name)

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
            grouping_tbl = self.grouping_tbl.bind(binding_root)  # type: ignore[arg-type]
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

    def __init__(self, table_spec: TableSpec) -> None:
        super().__init__()

        self.table_spec = table_spec
        self.known_cols = {}
        self.known_idxs = {}
        self.reserved_cols = {}

        # Pre-seed __annotations__ so the compiler routes bare annotations through
        # our recorder rather than a plain dict it would otherwise create.
        super().__setitem__('__annotations__', _AnnotationRecorder(self))
        super().__setitem__('_binding_root', None)

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            # Replace the value with a _PlaceholderColumnRef or EmbeddingIndex
            value = self.set_col_value(key, value)
        super().__setitem__(key, value)

    def register_reference(
        self, name: str, placeholder: _PlaceholderColumnRef, kind: Literal['base query', 'iterator']
    ) -> None:
        """Make `name` resolvable in the class body without registering it as a column to create.

        Used for a view's base-query columns and iterator outputs: the model can reference them, but they are
        created by the select list / iterator, not as additional columns. The name is reserved so that an
        explicit (re)definition of the same name is rejected.
        """
        self.reserved_cols[name] = kind
        super().__setitem__(name, placeholder)

    def _check_reserved(self, name: str) -> None:
        if name in self.reserved_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{name!r} is already defined by the {self.reserved_cols[name]}; it cannot be redeclared.',
            )

    def set_col_value(self, name: str, value: Any) -> EmbeddingIndex | _PlaceholderColumnRef:
        self._check_reserved(name)
        if isinstance(value, EmbeddingIndex):
            if name in self.known_cols or name in self.known_idxs:
                raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Index {name!r}: duplicate definition.')
            self.known_idxs[name] = value
            return value

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
            return _PlaceholderColumnRef(name, _col_type_from_spec(spec))

    def set_col_type(self, name: str, type_: Any) -> None:
        self._check_reserved(name)
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
        self.known_cols[name] = {'type': type_}
        super().__setitem__(name, _PlaceholderColumnRef(name, type_))


class TableModelMetaclass(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    registered_models: ClassVar[dict[str, TableModelMetaclass]] = {}  # table name -> model

    __table_spec__: TableSpec
    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, EmbeddingIndex]

    _binding_root: str | None

    @classmethod
    def __prepare__(  # type: ignore[override]
        mcs,  # noqa: N804  # Neither mypy nor ruff seems to understand metaclasses.
        cls_name: str,
        bases: tuple[type, ...],
        /,
        name: str,
        base: 'TableModel | ViewModel | _PlaceholderQuery | None' = None,
        iterator: func.GeneratingFunctionCall | None = None,
        create_default_idxs: bool = True,
        media_validation: Literal['on_read', 'on_write'] = 'on_write',
        comment: str | None = None,
        custom_metadata: Any = None,
    ) -> MutableMapping[str, object]:
        if len(bases) == 0:
            # This is the TableModel or ViewModel base class itself; no additional processing.
            return super().__prepare__(cls_name, bases)
        elif len(bases) > 1 or bases[0] not in (TableModel, ViewModel):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                'Pixeltable schemas must subclass exactly one of `TableModel`, `ViewModel`.',
            )
        else:
            display_name = f'{bases[0].__name__} `{cls_name}`'

            # Validate table name
            tbl_name = name
            if not isinstance(tbl_name, str) or not is_valid_identifier(tbl_name, allow_hyphens=True):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'{display_name}: `name` must be a valid Pixeltable identifier.'
                )
            if tbl_name in mcs.registered_models:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'{display_name} has name {tbl_name!r}, but that name was '
                    f'previously used by `{mcs.registered_models[tbl_name].__name__}`.',
                )

            # Validate base
            if base is not None:
                if bases[0] is TableModel:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'`base` not allowed for a `TableModel`; `{cls_name}` must subclass `ViewModel` instead.',
                    )
                if isinstance(base, _PlaceholderQuery):
                    pass
                elif isinstance(base, TableModelMetaclass):
                    base = base.select()  # convert to a _PlaceholderQuery
                else:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `base` must be a valid base table reference '
                        f'(another `TableModel` or `ViewModel`, or a query over a model).',
                    )
                assert isinstance(base, _PlaceholderQuery)
            elif bases[0] is ViewModel:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{display_name} must specify a `base`.')

            # Validate iterator
            if iterator is not None:
                if bases[0] is TableModel:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'`iterator` not allowed for a `TableModel`; `{cls_name}` must subclass `ViewModel` instead.',
                    )
                if not isinstance(iterator, func.GeneratingFunctionCall):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `iterator` must be a valid iterator reference.',
                    )

            media_validation_ = MediaValidation.validated(media_validation, '`media_validation`')

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
                }
            )

            if base is not None and base.select_clause is not None:
                # Make the select list's named columns referenceable in the body. They are created by the base
                # query, not as additional columns, so register them as references rather than `known_cols`.
                for col_name, expr in base.select_clause[1].items():
                    assert is_valid_identifier(col_name)  # since it must be a Python symbol
                    namespace.register_reference(col_name, _PlaceholderColumnRef(col_name, expr.col_type), 'base query')

            if iterator is not None:
                # Likewise for the iterator's outputs: referenceable, but created by the iterator.
                for col_name, output in iterator.outputs.items():
                    assert is_valid_identifier(col_name)
                    namespace.register_reference(col_name, _PlaceholderColumnRef(col_name, output.col_type), 'iterator')

            return namespace

    def __new__(
        mcs, cls_name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        if len(bases) == 0:
            # This is the TableModel or ViewModel base class itself; no additional processing.
            return super().__new__(mcs, cls_name, bases, namespace)

        assert isinstance(namespace, _ModelNamespace)

        if len(namespace.known_cols) == 0 and bases[0] is TableModel:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, 'Empty `TableModel` not allowed.')

        namespace['__table_spec__'] = namespace.table_spec
        namespace['__columns__'] = namespace.known_cols
        namespace['__indexes__'] = namespace.known_idxs

        # Remove the direct index references from the namespace; unlike columns,
        # they are not part of the table's namespace.
        for idx_name in namespace.known_idxs:
            namespace.pop(idx_name)

        # "normalize" the namespace to a plain dict; at this point, we're done with the special namespace treatment
        namespace = dict(namespace)

        cls = super().__new__(mcs, cls_name, bases, namespace)
        mcs.registered_models[namespace['__table_spec__']['name']] = cls
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

    class ValidationResults(NamedTuple):
        new_columns: list[str]
        deleted_columns: list[str]
        altered_columns: list[str]
        new_indices: list[str]
        deleted_indices: list[str]
        altered_indices: list[str]

        def has_changes(self) -> bool:
            return len(self.new_columns) > 0 or len(self.deleted_columns) > 0 or len(self.altered_columns) > 0

    @classmethod
    def _normalize_binding_root(cls, binding_root: str) -> str:
        if binding_root.endswith('/'):
            binding_root = binding_root[:-1]
        _ = catalog.Path.parse(binding_root, allow_empty_path=True)  # validate
        if len(binding_root) > 0:
            binding_root += '/'
        return binding_root

    def bind(cls, binding_root: str = '') -> pxt.Table:
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

    def create(cls, binding_root: str = '') -> Table:
        binding_root = cls._normalize_binding_root(binding_root)

        if cls.is_bound:
            return cls._resolve_tbl(binding_root, if_not_exists='error')

        table_spec: TableSpec = cls.__table_spec__
        indexes: dict[str, EmbeddingIndex] = cls.__indexes__

        # Bind the base query to an actual Query over the (already-existing) base table. This happens client-side,
        # outside any transaction; the resulting Query references real columns and so is serializable to whichever
        # catalog owns the table being created.
        base: pxt.Query | None = None
        if table_spec['base'] is not None:
            base = table_spec['base'].bind(binding_root)

        # The model's own column specs, with `type` annotations resolved to ColumnTypes (so they're serializable
        # for a proxied catalog). Computed `value` expressions still carry `_PlaceholderColumnRef`s referencing
        # sibling and base columns; those are substituted by the catalog that owns the table (create_from_model).
        columns: dict[str, ColumnSpec] = {}
        for name, col_spec in cls.__columns__.items():
            spec = col_spec.copy()
            if 'type' in spec:
                spec['type'] = ts.ColumnType.normalize_type(spec['type'], nullable_default=True, allow_builtin_types=False)
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
        )

        if was_created:
            # Add any declared embedding indexes.
            # TODO: Bring this under the transaction umbrella of create_from_model().
            for idx_name, idx_spec in indexes.items():
                col_ref = idx_spec.column
                if isinstance(col_ref, str):
                    col_ref = getattr(tbl, col_ref)
                elif isinstance(col_ref, _PlaceholderColumnRef):
                    col_ref = getattr(tbl, col_ref.name)
                kwargs = dataclasses.asdict(idx_spec)
                kwargs['column'] = col_ref
                kwargs['idx_name'] = idx_name
                tbl.add_embedding_index(**kwargs)

            Env.get().console_logger.info(f'Created {tbl._path()!r} from {table_spec["display_name"]}.')

        return cls.bind(binding_root)

    def __getattr__(cls, item: str) -> Any:
        if item in FORWARDED_TABLE_METHODS:
            if not cls.is_bound and hasattr(_PlaceholderQuery, item):
                # This model is not bound to a table, but the desired operation is accessible via a placeholder query.
                return getattr(_PlaceholderQuery(cls), item)  # type: ignore[arg-type]
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


# `name` will be ignored for the base classes, but is required to conform to the TableModelMetaclass signature
class TableModel(metaclass=TableModelMetaclass, name=''):
    """
    Base class for declarative Pixeltable table models.
    """


class ViewModel(metaclass=TableModelMetaclass, name=''):
    """
    Base class for declarative Pixeltable view models.
    """


def bind_all(binding_root: str = '') -> None:
    for model in TableModelMetaclass.registered_models.values():
        model.bind(binding_root)


def create_all(binding_root: str = '') -> None:
    for model in TableModelMetaclass.registered_models.values():
        model.create(binding_root)
