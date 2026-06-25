from __future__ import annotations

import dataclasses
import itertools
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.catalog.table_path import TableVersionPath
from pixeltable.query_clauses import FromClause, SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .catalog import retry_loop
from .globals import MediaValidation, is_valid_identifier
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
            return exprs.ColumnPropertyRef(self, prop)
        return super().__getattr__(item)

    def as_dict(self) -> dict[str, Any]:
        raise AssertionError('It should never be possible to serialize a placeholder.')


@dataclass
class _PlaceholderColumnSpec:
    column_spec: ColumnSpec


class _PlaceholderFactory:
    """
    Class for creating placeholder column objects that can be used in TableModel definitions.

    This allows users to reference column names in computed expressions without needing to define them
    a priori as actual columns.
    """

    _column_ctx: _ColumnCtx | None

    def __init__(self) -> None:
        self.set_column_ctx(None)

    def __getattr__(self, key: str) -> _PlaceholderColumnRef:
        if not isinstance(key, str) or not is_valid_identifier(key):
            raise AttributeError(f'Invalid column name: {key}')
        if self._column_ctx is None:
            raise AttributeError(
                f'Cannot reference abstract column {key!r} outside of a `TableModel` or `ViewModel` definition'
            )
        return getattr(self._column_ctx, key)

    def __hasattr__(self, key: str) -> bool:
        return isinstance(key, str) and is_valid_identifier(key)

    def __call__(
        self,
        *,
        type: type | None = None,
        value: exprs.Expr | None = None,
        primary_key: bool | None = None,
        stored: bool | None = None,
        media_validation: Literal['on_read', 'on_write'] | None = None,
        destination: str | Path | None = None,
        custom_metadata: Any = None,
        comment: str | None = None,
    ) -> _PlaceholderColumnSpec:
        column_spec: ColumnSpec = {}
        if type is not None:
            column_spec['type'] = type
        if value is not None:
            column_spec['value'] = value
        if primary_key is not None:
            column_spec['primary_key'] = primary_key
        if stored is not None:
            column_spec['stored'] = stored
        if media_validation is not None:
            column_spec['media_validation'] = media_validation
        if destination is not None:
            column_spec['destination'] = destination
        if custom_metadata is not None:
            column_spec['custom_metadata'] = custom_metadata
        if comment is not None:
            column_spec['comment'] = comment
        return _PlaceholderColumnSpec(column_spec)

    def set_column_ctx(self, column_ctx: _ColumnCtx | None) -> None:
        self._column_ctx = column_ctx


Column: _PlaceholderFactory = _PlaceholderFactory()


@dataclasses.dataclass
class _PlaceholderQuery:
    """
    A placeholder query object that can be used in ViewModel definitions to reference a base table or query
    that is not yet defined at the time of class body execution.
    """

    from_clause: type[TableModelMetaclass]
    select_list: list[tuple[exprs.Expr, str | None]] | None
    where_clause: exprs.Expr | None
    group_by_clause: list[exprs.Expr] | None
    order_by_clause: list[tuple[exprs.Expr, bool]] | None
    limit_val: exprs.Expr | None
    offset_val: exprs.Expr | None
    sample_clause: SampleClause | None

    def __init__(
        self,
        from_clause: type[TableModelMetaclass],
        select_list: list[tuple[exprs.Expr, str | None]] | None = None,
        where_clause: exprs.Expr | None = None,
        group_by_clause: list[exprs.Expr] | None = None,
        order_by_clause: list[tuple[exprs.Expr, bool]] | None = None,
        limit_val: exprs.Expr | None = None,
        offset_val: exprs.Expr | None = None,
        sample_clause: SampleClause | None = None,
    ) -> None:
        self.from_clause = from_clause
        self.select_list = select_list
        self.where_clause = where_clause
        self.group_by_clause = group_by_clause
        self.order_by_clause = order_by_clause
        self.limit_val = limit_val
        self.offset_val = offset_val
        self.sample_clause = sample_clause

    def select(self, *items: Any, **named_items: Any) -> _PlaceholderQuery:
        if self.select_list is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, '`select()` list already specified in `ViewModel` base query.'
            )
        for name in named_items:
            if not is_valid_identifier(name):
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid name: {name}')
        select_list = [(expr, None) for expr in items] + [(expr, k) for (k, expr) in named_items.items()]
        if len(select_list) == 0:
            return self
        return dataclasses.replace(self, select_list=select_list)

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

    def bind(self) -> 'pxt.Query':
        import pixeltable as pxt

        tbl = self.from_clause.bind()
        subst_dict = exprs.ExprDict[exprs.ColumnRef]()
        for col_name in tbl.columns():
            placeholder = _PlaceholderColumnRef(col_name, {'type': ts.InvalidType()})
            subst_dict[placeholder] = getattr(tbl, col_name)

        select_list = (
            [(expr.substitute(subst_dict), alias) for (expr, alias) in self.select_list]
            if self.select_list is not None
            else None
        )
        where_clause = self.where_clause.substitute(subst_dict) if self.where_clause is not None else None
        group_by_clause = (
            [expr.substitute(subst_dict) for expr in self.group_by_clause] if self.group_by_clause is not None else None
        )
        order_by_clause = (
            [(expr.substitute(subst_dict), asc) for (expr, asc) in self.order_by_clause]
            if self.order_by_clause is not None
            else None
        )
        limit_val = self.limit_val.substitute(subst_dict) if self.limit_val is not None else None
        offset_val = self.offset_val.substitute(subst_dict) if self.offset_val is not None else None
        sample_clause = (
            SampleClause(
                None,
                self.sample_clause.n,
                self.sample_clause.n_per_stratum,
                self.sample_clause.fraction,
                self.sample_clause.seed,
                [expr.substitute(subst_dict) for expr in self.sample_clause.stratify_by],
            )
            if self.sample_clause is not None
            else None
        )

        return pxt.Query(
            FromClause([tbl._tbl_version_path]),
            select_list,
            where_clause,
            group_by_clause,
            None,
            order_by_clause,
            limit_val,
            offset_val,
            sample_clause,
        )


class _ColumnCtx:
    tbl_name: str
    known_cols: dict[str, _PlaceholderColumnRef]
    known_idxs: dict[str, EmbeddingIndex]

    def __init__(self, tbl_name: str) -> None:
        self.tbl_name = tbl_name
        self.known_cols = {}
        self.known_idxs = {}

    def __hasattr__(self, key: str) -> bool:
        return key in self.known_cols

    def __getattr__(self, item: str) -> _PlaceholderColumnRef:
        if item not in self.known_cols:
            raise AttributeError(f'Column {item!r} is not defined yet')
        return self.known_cols[item]

    def set_col_value(self, name: str, value: Any) -> EmbeddingIndex | _PlaceholderColumnRef:
        if isinstance(value, EmbeddingIndex):
            self.known_idxs[name] = value
            return value
        else:
            assert name not in self.known_cols  # `value` always gets set before `type` if both are defined
            spec: ColumnSpec
            if isinstance(value, _PlaceholderColumnSpec):
                spec = value.column_spec
                if ('type' in spec) == ('value' in spec):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'Column specification for {name!r} must define `type` or `value`, but not both',
                    )
            else:
                # Computed column expression.
                expr = exprs.Expr.from_object(value)
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
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Type annotation for column {name!r} conflicts with the `type=` argument in `Column()`',
                )
            return existing_col_ref
        else:
            col_ref = _PlaceholderColumnRef(name, {'type': type_})
            self.known_cols[name] = col_ref
            return col_ref


class _AnnotationRecorder(dict):
    namespace: _ModelNamespace

    def __init__(self, namespace: _ModelNamespace) -> None:
        super().__init__()
        self.namespace = namespace

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            self.namespace.set_col_type(key, value)
        super().__setitem__(key, value)


class _ModelNamespace(dict):
    """Class namespace that records the source order of every name bound in the body,
    including bare annotations (which never write to the namespace itself)."""

    column_ctx: _ColumnCtx
    base: _PlaceholderQuery | None
    iterator: func.GeneratingFunctionCall | None

    def __init__(
        self, cls_name: str, tbl_name: str, base: _PlaceholderQuery | None, iterator: func.GeneratingFunctionCall | None
    ) -> None:
        super().__init__()
        self.column_ctx = _ColumnCtx(tbl_name)
        self.base = base
        self.iterator = iterator
        # Pre-seed __annotations__ so the compiler routes bare annotations through
        # our recorder rather than a plain dict it would otherwise create.
        super().__setitem__('__annotations__', _AnnotationRecorder(self))
        super().__setitem__(cls_name, self.column_ctx)
        super().__setitem__('_is_bound', False)
        Column.set_column_ctx(self.column_ctx)
        self._finalizer = weakref.finalize(self, lambda: self.cleanup_column_ctx())

    def cleanup_column_ctx(self) -> None:
        if Column._column_ctx is self.column_ctx:
            Column.set_column_ctx(None)

    def __setitem__(self, key: str, value: Any) -> None:
        if not key.startswith('_'):
            # Replace the value with a _PlaceholderColumnRef or EmbeddingIndex
            value = self.column_ctx.set_col_value(key, value)

        super().__setitem__(key, value)

    def set_col_type(self, key: str, type_: Any) -> None:
        col_ref = self.column_ctx.set_col_type(key, type_)
        super().__setitem__(key, col_ref)


@dataclass(frozen=True)
class EmbeddingIndex:
    column: str | exprs.Expr
    embedding: func.Function | None = None
    string_embed: func.Function | None = None
    image_embed: func.Function | None = None
    audio_embed: func.Function | None = None
    video_embed: func.Function | None = None
    document_embed: func.Function | None = None
    metric: Literal['cosine', 'ip', 'l2'] = 'cosine'
    precision: Literal['fp16', 'fp32'] = 'fp16'


class TableModelMetaclass(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    _registered_models: dict[str, TableModelMetaclass] = {}  # table name -> model

    __columns__: dict[str, _PlaceholderColumnRef]
    __indexes__: dict[str, EmbeddingIndex]
    __table_name__: str
    __base_table__: _PlaceholderQuery | 'pxt.Query' | None
    __iterator__: func.GeneratingFunctionCall | None

    _is_bound: bool

    @classmethod
    def __prepare__(mcs, cls_name: str, bases: tuple[type, ...], **kwargs: Any) -> dict[str, Any]:
        if len(bases) == 0:
            # This is the TableModel or ViewModel base class itself; no additional processing.
            return super().__prepare__(cls_name, bases, **kwargs)
        elif len(bases) > 1 or bases[0] not in (TableModel, ViewModel):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'Pixeltable schemas must subclass exactly one of `TableModel`, `ViewModel`.',
            )
        else:
            display_name = f'{bases[0].__name__} `{cls_name}`'

            # Validate table name
            if 'name' not in kwargs:
                raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'{display_name} must specify a `name`.')
            tbl_name = kwargs['name']
            if not isinstance(tbl_name, str) or not is_valid_identifier(tbl_name, allow_hyphens=True):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'{display_name}: `name` must be a valid Pixeltable identifier.'
                )
            if tbl_name in mcs._registered_models:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'{display_name} has name {tbl_name!r}, but that name was '
                    f'previously used by `{mcs._registered_models[tbl_name].__name__}`.',
                )

            # Validate base
            base: _PlaceholderQuery | None = None
            if 'base' in kwargs:
                if bases[0] is TableModel:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'`base` not allowed for a `TableModel`; `{cls_name}` must subclass `ViewModel` instead.',
                    )
                if isinstance(kwargs['base'], _PlaceholderQuery):
                    base = kwargs['base']
                elif isinstance(kwargs['base'], TableModelMetaclass):
                    base = kwargs['base'].select()
                else:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'{display_name}: `base` must be a valid base table reference '
                        f'(another `TableModel` or `ViewModel`, or a query over a model).',
                    )
            elif bases[0] is ViewModel:
                raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'{display_name} must specify a `base`.')

            # Validate iterator
            iterator: func.GeneratingFunctionCall | None = None
            if 'iterator' in kwargs:
                if bases[0] is TableModel:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'`iterator` not allowed for a `TableModel`; `{cls_name}` must subclass `ViewModel` instead.',
                    )
                if isinstance(kwargs['iterator'], func.GeneratingFunctionCall):
                    iterator = kwargs['iterator']
                else:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA, f'{display_name}: `iterator` must be a valid iterator reference.'
                    )

            namespace = _ModelNamespace(cls_name=cls_name, tbl_name=tbl_name, base=base, iterator=iterator)

            if base is not None and base.select_list is not None:
                # Pre-populate the namespace with named elements of the select list, appropriately typed.
                for expr, name in base.select_list:
                    if name is not None:
                        assert is_valid_identifier(name)  # since it must be a Python symbol
                        namespace[name] = _PlaceholderColumnRef(name, {'value': expr})

            if iterator is not None:
                # Pre-populate the namespace with the iterator's outputs, appropriately typed.
                for name, output in iterator.outputs.items():
                    assert is_valid_identifier(name)
                    namespace[name] = _PlaceholderColumnRef(name, {'type': output.col_type})

            return namespace

    def __new__(
        mcs, cls_name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        if len(bases) == 0:
            # This is the TableModel or ViewModel base class itself; no additional processing.
            return super().__new__(mcs, cls_name, bases, namespace)

        assert isinstance(namespace, _ModelNamespace)

        # Remove the _ColumnCtx object; it's no longer needed, and we need to "normalize" the namespace
        column_ctx = namespace.pop(cls_name)
        assert isinstance(column_ctx, _ColumnCtx)

        if len(column_ctx.known_cols) == 0 and bases[0] is TableModel:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, 'Empty `TableModel` not allowed.')

        # Process declarations. We need to process bare annotations (such as `col1: int`) as well as
        #    values (such as `col2 = Column.col1 + 1` or `idx0 = EmbeddingIndex(...)`).

        namespace['__table_name__'] = column_ctx.tbl_name
        namespace['__base_table__'] = namespace.base
        namespace['__iterator__'] = namespace.iterator
        namespace['__columns__'] = column_ctx.known_cols
        namespace['__indexes__'] = column_ctx.known_idxs

        for idx_name in column_ctx.known_idxs:
            namespace.pop(idx_name)

        cls = super().__new__(mcs, cls_name, bases, namespace)
        mcs._registered_models[cls.__table_name__] = cls
        return cls

    def _resolve_tbl(cls) -> Table:
        import pixeltable as pxt

        return pxt.get_table(cls.__table_name__)

    def _resolve_column(cls, col_name: str) -> exprs.ColumnRef:
        return getattr(cls._resolve_tbl(), col_name)

    def create(cls, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error') -> Table:
        tbl_media_validation = 'on_write'  # TODO: allow configuring this at the table level

        base: _PlaceholderQuery | pxt.Query | None = cls.__base_table__
        iterator: func.GeneratingFunctionCall | None = cls.__iterator__
        columns: dict[str, _PlaceholderColumnRef] = cls.__columns__
        indexes: dict[str, EmbeddingIndex] = cls.__indexes__

        catalog_columns: list[catalog.Column] = []
        subst_dict = exprs.ExprDict[exprs.ColumnRef]()

        initial_col_id = 0
        if base is not None:
            if isinstance(base, _PlaceholderQuery):
                base = base.bind()
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
                catalog_col = catalog.Column.create(name, {'type': output.col_type, 'stored': output.is_stored})
                catalog_col.tbl_handle = tbl_handle
                catalog_col.id = next(next_col_id)
                subst_dict[_PlaceholderColumnRef(name)] = exprs.ColumnRef(
                    catalog_col.column_version_md(), perform_validation=(tbl_media_validation == 'on_read')
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
                perform_validation=subst_spec.get('media_validation', tbl_media_validation) == 'on_read',
            )

        # Create the table with its non-computed columns
        # initial_schema = {col_name: col_spec for col_name, col_spec in columns.items() if col_spec.get('value') is None}

        cat = get_runtime().catalog
        tbl_path = catalog.Path.parse(cls.__table_name__)
        base_tvp: TableVersionPath | None = None

        if issubclass(cls, TableModel):
            create_fn = retry_loop(for_write=True)(
                lambda: cat._create_table(
                    path=tbl_path,
                    columns=catalog_columns,
                    if_exists=if_exists,
                    primary_key=None,
                    comment=None,
                    custom_metadata=None,
                    media_validation=MediaValidation.ON_WRITE,
                    create_default_idxs=True,
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
                    create_default_idxs=True,
                    iterator=iterator,
                    comment=None,
                    custom_metadata=None,
                    media_validation=MediaValidation.ON_WRITE,
                    if_exists=if_exists,
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

        return cls.bind()

    def bind(cls) -> pxt.Table:
        if cls.is_bound:
            return cls._resolve_tbl()
        else:
            tbl = cls._resolve_tbl()
            # TODO: Validation
            for col_name in tbl.columns():
                setattr(cls, col_name, getattr(tbl, col_name))
            cls._is_bound = True
            return tbl

    @property
    def is_bound(cls) -> bool:
        return cls._is_bound

    def __getattr__(cls, item: str) -> Any:
        if item in FORWARDED_TABLE_METHODS:
            return getattr(cls._resolve_tbl(), item)
        return super().__getattribute__(item)

    @property
    def table(cls) -> Table:
        """The underlying [`Table`][pixeltable.Table] this model is bound to.

        Use this to access the full table and query API with static type information; e.g.,
        `MyModel.table.where(...).order_by(...).collect()`.
        """
        return cls._resolve_tbl()

    def select(cls, *items: Any, **named_items: Any) -> _PlaceholderQuery | pxt.Query:
        if cls._is_bound:
            return cls.table.select(*items, **named_items)
        else:
            return _PlaceholderQuery(cls).select(*items, **named_items)

    def where(cls, pred: exprs.Expr) -> _PlaceholderQuery | pxt.Query:
        if cls._is_bound:
            return cls.table.where(pred)
        else:
            return _PlaceholderQuery(cls).where(pred)


class TableModel(metaclass=TableModelMetaclass):
    """
    Base class for declarative Pixeltable table models.
    """


class ViewModel(metaclass=TableModelMetaclass):
    """
    Base class for declarative Pixeltable view models.
    """


def create_all() -> None:
    for model in TableModelMetaclass._registered_models.values():
        model.create()
