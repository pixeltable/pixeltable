from __future__ import annotations

import dataclasses
import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, MutableMapping, NamedTuple, TypedDict

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.catalog.table_path import TableVersionPath
from pixeltable.env import Env
from pixeltable.query_clauses import SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .catalog import retry_loop
from .globals import IfExistsParam, MediaValidation, is_valid_identifier
from .table import Table
from .table_version import TableVersionKey
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
    column: Any
    embedding: func.Function | None = None
    string_embed: func.Function | None = None
    image_embed: func.Function | None = None
    audio_embed: func.Function | None = None
    video_embed: func.Function | None = None
    document_embed: func.Function | None = None
    metric: Literal['cosine', 'ip', 'l2'] = 'cosine'
    precision: Literal['fp16', 'fp32'] = 'fp16'


class _PlaceholderColumnRef(exprs.Expr):
    """
    A placeholder column reference used in TableModel definitions,
    which gets substituted with an actual ColumnRef during Table creation.
    """

    tbl_name: str
    name: str
    column_spec: ColumnSpec

    def __init__(self, name: str, column_spec: ColumnSpec | None = None) -> None:
        col_type: ts.ColumnType
        if column_spec is None:
            col_type = ts.InvalidType()
        elif 'type' in column_spec:
            type_ = column_spec['type']
            col_type = ts.ColumnType.normalize_type(type_, nullable_default=True, allow_builtin_types=False)
        else:
            assert 'value' in column_spec
            col_type = column_spec['value'].col_type

        super().__init__(col_type)

        self.name = name
        self.column_spec = column_spec
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

    def as_dict(self) -> dict[str, Any]:
        raise AssertionError('It should never be possible to serialize a placeholder.')


@dataclasses.dataclass
class _PlaceholderQuery:
    """
    A placeholder query object that can be used in ViewModel definitions to reference a base table or query
    that is not yet defined at the time of class body execution.
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
            placeholder = _PlaceholderColumnRef(col_name, {'type': ts.InvalidType()})  # type: ignore[arg-type]
            subst_dict[placeholder] = getattr(tbl, col_name)

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
    namespace: _ModelNamespace

    def __init__(self, namespace: _ModelNamespace) -> None:
        super().__init__()
        self.namespace = namespace

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            self.namespace.set_col_type(key, value)
        super().__setitem__(key, value)


class TableSpec(TypedDict):
    name: str
    display_name: str
    base: _PlaceholderQuery | None
    iterator: func.GeneratingFunctionCall | None
    create_default_idxs: bool
    media_validation: MediaValidation
    comment: str | None
    custom_metadata: Any


class _ModelNamespace(dict):
    """Class namespace that records the source order of every name bound in the body,
    including bare annotations (which never write to the namespace itself)."""

    table_spec: TableSpec
    known_cols: dict[str, _PlaceholderColumnRef]
    known_idxs: dict[str, EmbeddingIndex]

    def __init__(self, table_spec: TableSpec) -> None:
        super().__init__()

        self.table_spec = table_spec
        self.known_cols = {}
        self.known_idxs = {}

        # Pre-seed __annotations__ so the compiler routes bare annotations through
        # our recorder rather than a plain dict it would otherwise create.
        super().__setitem__('__annotations__', _AnnotationRecorder(self))
        super().__setitem__('_binding_root', None)

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            # Replace the value with a _PlaceholderColumnRef or EmbeddingIndex
            value = self.set_col_value(key, value)
        super().__setitem__(key, value)

    def set_col_value(self, name: str, value: Any) -> EmbeddingIndex | _PlaceholderColumnRef:
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
            col_ref = _PlaceholderColumnRef(name, spec)
            self.known_cols[name] = col_ref
            return col_ref

    def set_col_type(self, name: str, type_: Any) -> _PlaceholderColumnRef:
        if name in self.known_idxs:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'Cannot set a type annotation for index {name!r}.')
        if name in self.known_cols:
            # We previously processed this column via `set_col_value()`. Sanity check the type.
            existing_col_ref = self.known_cols[name]
            if existing_col_ref.col_type != type_:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'Conflicting type annotation for column {name!r}.'
                )
            return existing_col_ref
        else:
            col_ref = _PlaceholderColumnRef(name, {'type': type_})
            self.known_cols[name] = col_ref
            super().__setitem__(name, col_ref)
            return col_ref


class TableModelMetaclass(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    registered_models: ClassVar[dict[str, TableModelMetaclass]] = {}  # table name -> model

    __table_spec__: TableSpec
    __columns__: dict[str, _PlaceholderColumnRef]
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
                    base = base.select()
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
                # Pre-populate the namespace with named elements of the select list, appropriately typed.
                for col_name, expr in base.select_clause[1].items():
                    assert is_valid_identifier(col_name)  # since it must be a Python symbol
                    namespace[col_name] = _PlaceholderColumnRef(col_name, {'value': expr})

            if iterator is not None:
                # Pre-populate the namespace with the iterator's outputs, appropriately typed.
                for col_name, output in iterator.outputs.items():
                    assert is_valid_identifier(col_name)
                    namespace[col_name] = _PlaceholderColumnRef(col_name, {'type': output.col_type})  # type: ignore[arg-type]

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

    def validate_model(cls, existing_tbl: Table) -> ValidationResults:
        existing_md = existing_tbl.get_metadata()
        model_kind = 'view' if issubclass(cls, ViewModel) else 'table'
        if model_kind != existing_md['kind']:
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                f'{cls.__table_spec__["display_name"]} is defined as a {model_kind}, '
                f'but the existing table {existing_md["path"]!r} is a {existing_md["kind"]}.',
            )

        # TODO: inspect columns and indices and populate the ValidationResults accordingly.
        return cls.ValidationResults([], [], [], [], [], [])

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

        existing_tbl = cls._resolve_tbl(binding_root, if_not_exists='ignore')
        if existing_tbl is not None:
            # TODO: Schema validation / schema merge
            return cls.bind(binding_root)

        table_spec: TableSpec = cls.__table_spec__
        columns: dict[str, _PlaceholderColumnRef] = cls.__columns__
        indexes: dict[str, EmbeddingIndex] = cls.__indexes__

        catalog_columns: list[catalog.Column] = []
        subst_dict: dict[exprs.Expr, exprs.Expr] = {}

        placeholder_base = table_spec['base']
        iterator = table_spec['iterator']

        initial_col_id = 0
        base: pxt.Query | None = None
        if placeholder_base is not None:
            base = placeholder_base.bind(binding_root)
            if base.select_list is None:
                # select(*): put all visible columns from the base table into scope
                for col in base._first_tbl.columns():
                    subst_dict[_PlaceholderColumnRef(col.name)] = exprs.ColumnRef(col.column_version_md())
            else:
                initial_col_id = len(base.select_list)
                for expr, name in base.select_list:
                    if name is not None:
                        subst_dict[_PlaceholderColumnRef(name)] = expr
                for expr, name in base.select_list:
                    if name is None and isinstance(expr, exprs.ColumnRef):
                        subst_dict[_PlaceholderColumnRef(expr.column_md.name)] = expr

        tbl_id = uuid.uuid4()
        tbl_handle = TableVersionHandle(TableVersionKey(tbl_id, None))
        next_col_id = itertools.count(initial_col_id)

        if iterator is not None:
            subst_args = [arg.substitute(subst_dict) for arg in iterator.args]
            subst_kwargs = {k: v.substitute(subst_dict) for k, v in iterator.kwargs.items()}
            subst_bound_args = {k: v.substitute(subst_dict) for k, v in iterator.bound_args.items()}
            iterator = func.GeneratingFunctionCall(
                iterator.it, subst_args, subst_kwargs, subst_bound_args, iterator.outputs, iterator.validation_error
            )
            for name, output in iterator.outputs.items():
                catalog_col = catalog.Column.create(name, {'type': output.col_type, 'stored': output.is_stored})  # type: ignore[arg-type]
                catalog_col.tbl_handle = tbl_handle
                catalog_col.id = next(next_col_id)
                subst_dict[_PlaceholderColumnRef(name)] = exprs.ColumnRef(
                    catalog_col.column_version_md(), perform_validation=(table_spec['media_validation'] == 'on_read')
                )

        for name, placeholder in columns.items():
            subst_spec: ColumnSpec = placeholder.column_spec.copy()
            if 'value' in subst_spec:
                subst_spec['value'] = subst_spec['value'].substitute(subst_dict)
                residual_placeholders = list(subst_spec['value'].subexprs(_PlaceholderColumnRef))
                if len(residual_placeholders) > 0:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'Column {name!r} in `{cls.__name__}` references columns that are not in '
                        f"the model's scope: {[c.name for c in residual_placeholders]}",
                    )
            catalog_col = catalog.Column.create(name, subst_spec)
            catalog_col.tbl_handle = tbl_handle
            catalog_col.id = next(next_col_id)
            catalog_columns.append(catalog_col)
            subst_dict[placeholder] = exprs.ColumnRef(
                catalog_col.column_version_md(),
                perform_validation=subst_spec.get('media_validation', table_spec['media_validation']) == 'on_read',
            )

        cat = get_runtime().catalog
        bound_path = f'{binding_root}{table_spec["name"]}'
        tbl_path = catalog.Path.parse(bound_path)
        base_tvp: TableVersionPath | None = None

        if issubclass(cls, TableModel):
            create_fn = retry_loop(for_write=True)(
                lambda: cat._create_table(
                    path=tbl_path,
                    columns=catalog_columns,
                    if_exists=IfExistsParam.ERROR,
                    primary_key=None,
                    comment=table_spec['comment'],
                    custom_metadata=table_spec['custom_metadata'],
                    media_validation=table_spec['media_validation'],
                    create_default_idxs=table_spec['create_default_idxs'],
                    is_versioned=True,
                    tbl_id=tbl_id,
                )
            )

        else:
            assert issubclass(cls, ViewModel)

            base_tvp = base._first_tbl

            create_fn = retry_loop(for_write=True)(
                lambda: cat._create_view(
                    path=tbl_path,
                    base_path=base_tvp,
                    select_list=base.select_list,
                    where=base.where_clause,
                    sample_clause=base.sample_clause,
                    additional_columns=catalog_columns,
                    is_snapshot=False,
                    create_default_idxs=table_spec['create_default_idxs'],
                    iterator=iterator,
                    comment=table_spec['comment'],
                    custom_metadata=table_spec['custom_metadata'],
                    media_validation=table_spec['media_validation'],
                    if_exists=IfExistsParam.ERROR,
                    tbl_id=tbl_id,
                )
            )

        cat._roll_forward_ids.clear()
        tbl_id_, _ = create_fn()
        assert tbl_id == tbl_id_
        if base_tvp is not None and base_tvp.is_mutable():
            # invalidate base's TableVersion instance, so that it gets reloaded with the new mutable view
            cat._clear_tv_cache(base_tvp.tbl_version.key)
        cat._roll_forward()

        get_fn = retry_loop(read_tbl_ids=[tbl_id])(lambda: cat.get_table_by_id(tbl_id))
        tbl = get_fn()

        # Now add any declared indexes.
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

        Env.get().console_logger.info(f'Created {tbl._path()!r} from {cls.__table_spec__["display_name"]}.')

        return cls.bind(binding_root)  # strip trailing slash

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
