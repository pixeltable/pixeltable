# ruff: noqa: N804  # Neither mypy nor ruff seems to understand metaclasses.

from __future__ import annotations
import __future__

import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, MutableMapping, Sequence, TypedDict
from uuid import UUID

from typing_extensions import TypeForm

from pixeltable import catalog, exceptions as excs, exprs, func, index, type_system as ts
from pixeltable.env import Env
from pixeltable.exprs import ColumnRefByName
from pixeltable.query_clauses import SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .globals import MediaValidation, is_valid_identifier
from .table import Table
from .table_metadata import ColumnMetadata, TableMetadata
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


@dataclasses.dataclass(frozen=True)
class Column:
    """A column specification used in a TableModel or ViewModel definition."""

    type: TypeForm | None = None
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


@dataclasses.dataclass(frozen=True)
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
    name: str | None = None

    def as_fn_call(self) -> exprs.FunctionCall:
        # Static resolution of the embedding function as a FunctionCall.
        assert isinstance(self.column, exprs.ColumnRefByName)
        col_type = self.column.col_type
        if col_type.is_string_type() and self.string_embed is not None:
            return self.string_embed(self.column)
        elif col_type.is_image_type() and self.image_embed is not None:
            return self.image_embed(self.column)
        elif col_type.is_audio_type() and self.audio_embed is not None:
            return self.audio_embed(self.column)
        elif col_type.is_video_type() and self.video_embed is not None:
            return self.video_embed(self.column)
        elif col_type.is_document_type() and self.document_embed is not None:
            return self.document_embed(self.column)
        elif self.embedding is not None:
            return self.embedding(self.column)
        else:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'EmbeddingIndex has no embedding function defined for type: {col_type}'
            )

    def __repr__(self) -> str:
        embeds = [
            f'{name}={fn}'
            for name, fn in (
                ('embedding', self.embedding),
                ('string_embed', self.string_embed),
                ('image_embed', self.image_embed),
                ('audio_embed', self.audio_embed),
                ('video_embed', self.video_embed),
                ('document_embed', self.document_embed),
            )
            if fn is not None
        ]
        parts = [f'column={self.column}', *embeds]
        # Only surface metric/precision when they deviate from their defaults.
        if self.metric != 'cosine':
            parts.append(f'metric={self.metric!r}')
        if self.precision != 'fp16':
            parts.append(f'precision={self.precision!r}')
        if self.name is not None:
            parts.append(f'name={self.name!r}')
        return f'EmbeddingIndex({", ".join(parts)})'


@dataclasses.dataclass(frozen=True)
class BtreeIndex:
    """A B-tree index specification used in a TableModel or ViewModel definition."""

    column: Any

    def __repr__(self) -> str:
        return f'BtreeIndex(column={self.column})'


# An index specification declared as a class attribute in a TableModel or ViewModel definition.
IndexDeclaration = EmbeddingIndex | BtreeIndex


class TableSpec(TypedDict):
    """Table specification from a TableModel or ViewModel."""

    name: str
    display_name: str
    base: ModelQuery | None
    iterator: func.GeneratingFunctionCall | None
    has_default_idxs: bool
    media_validation: MediaValidation
    comment: str | None
    custom_metadata: Any


def _col_type_from_spec(column_spec: ColumnSpec) -> ts.ColumnType:
    """The ColumnType that a column defined by `column_spec` will have."""
    if 'type' in column_spec:
        return ts.ColumnType.normalize_type(column_spec['type'], allow_builtin_types=False)
    assert 'value' in column_spec
    return column_spec['value'].col_type


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

    def _bind(self, catalog_dir: str) -> 'pxt.Query':
        tbl: Table = self.from_clause._bind(catalog_dir)  # type: ignore[arg-type]
        subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
        for col_name in tbl.columns():
            subst_dict[ColumnRefByName(col_name)] = getattr(tbl, col_name)

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
            grouping_tbl = self.grouping_tbl._bind(catalog_dir)  # type: ignore[arg-type]
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
    computed column expressions, Column and index specifications) are registered promptly and in the exact
    order of declaration.
    """

    table_spec: TableSpec
    known_cols: dict[str, ColumnSpec]

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
        super().__setitem__(name, ColumnRefByName(name, col_type))

    def _check_reserved(self, name: str) -> None:
        if name in self.reserved_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{name!r} is already defined by the {self.reserved_cols[name]}; it cannot be redeclared.',
            )

    def set_col_value(self, name: str, value: Any) -> None:
        self._check_reserved(name)
        if name in self.known_cols:
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
        super().__setitem__(name, exprs.ColumnRefByName(name, _col_type_from_spec(spec)))

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
        type_ = ts.ColumnType.normalize_type(type_, allow_builtin_types=False)
        if name in self.known_cols:
            # We previously processed this column via `set_col_value()`. Sanity check the type.
            if _col_type_from_spec(self.known_cols[name]) != type_:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'Conflicting type annotation for column {name!r}.'
                )
            return
        # Bare annotation (`col: SomeType`): record the spec and make the name referenceable in the body.
        self.known_cols[name] = {'type': type_}  # type: ignore[typeddict-item]
        super().__setitem__(name, exprs.ColumnRefByName(name, type_))


class TableModelMeta(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    __table_spec__: TableSpec
    __columns__: dict[str, ColumnSpec]
    __indexes__: list[IndexDeclaration]
    __bound_table__: Table | None

    _catalog_dir: str | None

    @classmethod
    def __prepare__(  # type: ignore[override]
        mcs,
        cls_name: str,
        bases: tuple[type, ...],
        /,
        name: str,
        base: 'TableModelMeta | ModelQuery | None' = None,
        iterator: func.GeneratingFunctionCall | None = None,
        has_default_idxs: bool = False,
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
                if base.select_clause is not None:
                    # Validate the select list.
                    items, _ = base.select_clause
                    for item in items:
                        if not isinstance(item, exprs.ColumnRefByName):
                            raise excs.RequestError(
                                excs.ErrorCode.INVALID_ARGUMENT,
                                f'{display_name}: `base` select() list may contain only direct column references '
                                f'or named expressions, but contains an anonymous compound expression: {item}\n'
                                f'Use kwargs syntax to give it an explicit name: select(my_name=...)',
                            )
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
                    'has_default_idxs': has_default_idxs,
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

    @classmethod
    def _validate_indexes(
        mcs, cls_name: str, namespace: _ModelNamespace, known_idxs: Sequence[IndexDeclaration]
    ) -> None:
        for idx in known_idxs:
            if not isinstance(idx.column, exprs.ColumnRefByName):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'model `{cls_name}`: Invalid {type(idx).__name__} column reference: {idx.column!r}',
                )
            if (
                isinstance(idx, EmbeddingIndex)
                and idx.name is not None
                and not (isinstance(idx.name, str) and is_valid_identifier(idx.name))
            ):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'model `{cls_name}`: Invalid {type(idx).__name__} name: {idx.name!r}',
                )
        # A table with default indexes enabled is not allowed to have explicit B-tree indexes.
        if namespace.table_spec['has_default_idxs'] and any(isinstance(idx, BtreeIndex) for idx in known_idxs):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'model `{cls_name}`: cannot combine `has_default_idxs=True` with explicitly declared B-tree '
                f'index(es); eligible columns are indexed automatically.',
            )
        all_indexed_cols = {idx.column.name for idx in known_idxs}
        for col_name in all_indexed_cols:
            btree_idxs = [idx for idx in known_idxs if isinstance(idx, BtreeIndex) and idx.column.name == col_name]
            if len(btree_idxs) > 1:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'model `{cls_name}`: multiple B-tree indexes for column {col_name!r}.',
                )
            embedding_idxs = [
                idx for idx in known_idxs if isinstance(idx, EmbeddingIndex) and idx.column.name == col_name
            ]
            if len(embedding_idxs) > 1 and any(idx.name is None for idx in embedding_idxs):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'model `{cls_name}`: column {col_name!r} has multiple embedding indexes; they must be '
                    'given explicit names',
                )
        all_index_names = [idx.name for idx in known_idxs if isinstance(idx, EmbeddingIndex) and idx.name is not None]
        if len(all_index_names) != len(set(all_index_names)):
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, f'model `{cls_name}`: index names must be unique')

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
        namespace_dict['__bound_table__'] = None
        namespace_dict['_catalog_dir'] = None

        known_idxs = namespace_dict.get('__indexes__', [])
        if not isinstance(known_idxs, Sequence) or not all(isinstance(idx, IndexDeclaration) for idx in known_idxs):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'model `{cls_name}`: `__indexes__` must be a sequence of EmbeddingIndex or BtreeIndex instances.',
            )
        mcs._validate_indexes(cls_name, namespace, known_idxs)
        namespace_dict['__indexes__'] = list(known_idxs)  # normalize

        cls = super().__new__(mcs, cls_name, bases, namespace_dict)
        assert hasattr(bases[0], '__registered_models__')  # This was checked in __prepare__()
        bases[0].__registered_models__[namespace.table_spec['name']] = cls
        return cls

    def _resolve_tbl(cls, catalog_dir: str, if_not_exists: Literal['error', 'ignore']) -> Table | None:
        import pixeltable as pxt

        if cls._catalog_dir is not None and catalog_dir != cls._catalog_dir:
            raise excs.RequestError(
                excs.ErrorCode.ALREADY_BOUND,
                f'Cannot bind `{cls.__name__}` at {catalog_dir!r}: it is already bound at {cls._catalog_dir!r}.',
            )

        bound_path = f'{catalog_dir}{cls.__table_spec__["name"]}'
        return pxt.get_table(bound_path, if_not_exists=if_not_exists)

    @property
    def is_bound(cls) -> bool:
        return cls._catalog_dir is not None

    @classmethod
    def _dir_prefix(cls, catalog_dir: str) -> str:
        catalog_dir = catalog_dir.rstrip('/')
        _ = catalog.Path.parse(catalog_dir, allow_empty_path=True)  # validate
        return f'{catalog_dir}/' if catalog_dir != '' else ''

    def _bind(cls, catalog_dir: str = '') -> pxt.Table:
        catalog_dir = cls._dir_prefix(catalog_dir)

        tbl = cls._resolve_tbl(catalog_dir, if_not_exists='error')

        if cls.is_bound:
            return tbl

        else:
            col_refs = {col_name: getattr(tbl, col_name) for col_name in tbl.columns()}

            # Table ops succeeded; now update the class.
            for col_name, col_ref in col_refs.items():
                setattr(cls, col_name, col_ref)
            cls._catalog_dir = catalog_dir
            return tbl

    def _create(cls, catalog_dir: str = '') -> tuple[Table, bool]:
        """Returns the table and whether it was created now (False if it already existed)."""
        catalog_dir = cls._dir_prefix(catalog_dir)

        if cls.is_bound:
            tbl = cls._resolve_tbl(catalog_dir, if_not_exists='error')
            assert tbl is not None
            return tbl, False

        table_spec: TableSpec = cls.__table_spec__

        # Bind the base query to an actual Query over the (already-existing) base table. This happens client-side,
        # outside any transaction; the resulting Query references real columns and so is serializable to whichever
        # catalog owns the table being created.
        base: pxt.Query | None = None
        if table_spec['base'] is not None:
            base = table_spec['base']._bind(catalog_dir)

        # The model's own column specs, with `type` annotations resolved to ColumnTypes (so they're serializable
        # for a proxied catalog). Computed value expressions still carry ColumnRefByNames referencing
        # sibling and base columns; those are substituted by the catalog that owns the table (create_from_model).
        columns: dict[str, ColumnSpec] = {}
        for name, col_spec in cls.__columns__.items():
            spec = col_spec.copy()
            if 'type' in spec:
                spec['type'] = ts.ColumnType.normalize_type(  # type: ignore[typeddict-item]
                    spec['type'], allow_builtin_types=False
                )
            columns[name] = spec

        bound_path = f'{catalog_dir}{table_spec["name"]}'
        tbl_path = catalog.Path.parse(bound_path)

        cat = get_runtime().get_catalog(tbl_path)
        tbl, was_created = cat.create_from_model(
            path=tbl_path,
            columns=columns,
            display_name=table_spec['display_name'],
            has_default_idxs=table_spec['has_default_idxs'],
            media_validation=table_spec['media_validation'],
            comment=table_spec['comment'],
            custom_metadata=table_spec['custom_metadata'],
            iterator=table_spec['iterator'],
            base=base,
            idxs=cls.__indexes__,
        )

        if was_created:
            Env.get().console_logger.info(f'Created {tbl._path()!r} from {table_spec["display_name"]}.')

        return cls._bind(catalog_dir), was_created

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
        return cls._resolve_tbl(cls._catalog_dir, if_not_exists='error')


def prepare_model(
    tbl_handle: TableVersionHandle,
    columns: dict[str, ColumnSpec],
    display_name: str,
    iterator: func.GeneratingFunctionCall | None,
    base: 'pxt.Query | None',
    idxs: list[IndexDeclaration],
) -> tuple[func.GeneratingFunctionCall | None, list[catalog.Column], list[catalog.IndexSpec]]:
    """
    Given model declarations in the form of columns, base, iterator, and index specifications, along with
    the relevant metadata, assembles lists of additional columns and additional indices to be created in the table.
    The outputs will be fully resolved (ColumnRefByNames replaced with actual ColumnRefs and the index-spec
    dataclass instances replaced with actual instances of index.IndexBase).

    Returns: a tuple of (rebound iterator, additional columns, additional idxs).
    """

    # View columns always go in a specific order:
    # - iterator columns first
    # - then columns from the base query's select_list
    #     (but not if it's a select(*): then just inherit the base table's columns)
    # - finally, the view's additional_columns.

    # A registry of visible columns of the table (base table/query columns, iterator columns,
    # and additional columns).
    user_cols: dict[str, catalog.Column] = {}

    # A substitution dictionary resolving ColumnRefByNames to actual ColumnRefs. It holds only the base tables'
    # columns; the columns of the table being created have no ids yet, so references to those stay ColumnRefByName.
    subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()

    # Names of the columns of the table being created, accumulated as they are processed.
    preceding_names: set[str] = set()

    # First the iterator columns, if present.
    if iterator is not None:
        # Rebind the iterator, resolving its argument references against the base table.
        assert base is not None
        base_tbl_subst_dict = exprs.ExprDict[exprs.Expr](
            (exprs.ColumnRefByName(col.name), exprs.ColumnRef(col.column_version_md()))
            for col in base._first_tbl.columns()
        )
        subst_args = [arg.substitute(base_tbl_subst_dict) for arg in iterator.args]
        subst_kwargs = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.kwargs.items()}
        subst_bound_args = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.bound_args.items()}
        iterator = func.GeneratingFunctionCall(
            iterator.it, subst_args, subst_kwargs, subst_bound_args, iterator.outputs, iterator.validation_error
        )
        # Build substitutions for the iterator's output columns.
        for name, output in iterator.outputs.items():
            catalog_col = catalog.Column.create(name, {'type': output.col_type, 'stored': output.is_stored})  # type: ignore[arg-type]
            catalog_col.tbl_handle = tbl_handle
            user_cols[name] = catalog_col
            preceding_names.add(name)

    if base is not None:
        # Build substitutions for the base table/query's columns.
        if base.select_list is None:
            # select(*): all visible columns from the base table
            for col in base._first_tbl.columns():
                # Iterator column names take precedence over base table column names in the model namespace, so
                # only update the substitution dicts if the name isn't already present.
                if col.name not in user_cols:
                    user_cols[col.name] = col
                    ref = exprs.ColumnRef(col.column_version_md())
                    subst_dict[exprs.ColumnRefByName(col.name)] = ref
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

                if col_name is not None:
                    # Column names that arrived via an explicit select list take precedence over iterator column
                    # names in the model namespace, so here we always update the dicts.
                    catalog_col = catalog.Column.create(col_name, expr.col_type)
                    catalog_col.tbl_handle = tbl_handle
                    user_cols[col_name] = catalog_col
                    preceding_names.add(col_name)

    # Process any additional columns specified in the view model body.
    additional_cols: list[catalog.Column] = []
    for name, spec in columns.items():
        subst_spec = spec.copy()
        if 'value' in subst_spec:
            subst_spec['value'] = subst_spec['value'].substitute(subst_dict)
            # whatever is still a ColumnRefByName has to name a column of this table that precedes this one, so that
            # it has a value by the time this column is computed
            unresolved = [
                ref for ref in subst_spec['value'].subexprs(exprs.ColumnRefByName) if ref.name not in preceding_names
            ]
            if len(unresolved) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} in {display_name} references columns that are not in '
                    f"the model's scope: {[ref.name for ref in unresolved]}",
                )

        # subst_dict holds exactly the inherited base columns, we don't allow an explicitly named column to shadow one
        if exprs.ColumnRefByName(name) in subst_dict:
            assert base is not None
            raise excs.AlreadyExistsError(
                excs.ErrorCode.COLUMN_ALREADY_EXISTS,
                f'Column {name!r} already exists in the base table {base._first_tbl.tbl_name()!r}.',
            )
        catalog_col = catalog.Column.create(name, subst_spec)
        catalog_col.tbl_handle = tbl_handle
        additional_cols.append(catalog_col)
        user_cols[name] = catalog_col
        preceding_names.add(name)

    return iterator, additional_cols, _resolve_model_idxs(idxs, user_cols, display_name)


def _resolve_model_idxs(
    idxs: list[IndexDeclaration], user_cols: dict[str, catalog.Column], display_name: str
) -> list[catalog.IndexSpec]:
    """Resolve each declared index against the model's visible columns.

    The returned specs record the indexed column by name. These columns names need to be substituted with
    the corresponding catalog Columns.
    """
    resolved_idxs: list[catalog.IndexSpec] = []
    for idx_spec in idxs:
        idx_name: str | None = idx_spec.name if isinstance(idx_spec, EmbeddingIndex) else None
        if not isinstance(idx_spec.column, exprs.ColumnRefByName):
            idx_display_name = f'Index {idx_name!r}' if idx_name is not None else 'Index'
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{idx_display_name} in {display_name} has an invalid column reference.'
            )
        col_name = idx_spec.column.name
        idx_display_name = f'Index {idx_name!r}' if idx_name is not None else f'Index on column {col_name!r}'
        if col_name not in user_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{idx_display_name} in {display_name} references unknown column {col_name!r}.',
            )
        idx: index.IndexBase
        if isinstance(idx_spec, EmbeddingIndex):
            idx = index.EmbeddingIndex(
                metric=idx_spec.metric,
                precision=idx_spec.precision,
                embed=idx_spec.embedding,
                string_embed=idx_spec.string_embed,
                image_embed=idx_spec.image_embed,
                audio_embed=idx_spec.audio_embed,
                video_embed=idx_spec.video_embed,
                document_embed=idx_spec.document_embed,
                column=user_cols[col_name],
            )
        else:
            assert isinstance(idx_spec, BtreeIndex)
            # TODO(PXT-1294): a model always describes a data-versioned table, so the index gets a value column
            idx = index.BtreeIndex(uses_value_col=True)
        resolved_idxs.append(catalog.IndexSpec(col_name, idx_name, idx))

    return resolved_idxs


class TableSchemaChangeSet(TypedDict):
    """
    Schema change operations applied to a single table.

    Used for proxy communication of changes that need to be applied to a catalog table during update_all().
    """

    path: catalog.Path

    # name -> (spec, origin). A 'base_query' column comes from the view's base query `select()` list and resolves
    # against the base table's columns; a 'model_body' column resolves against the view's own visible columns.
    new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]]
    dropped_columns: list[str]
    new_idxs: list[IndexDeclaration]
    dropped_idxs: list[str]

    # tbl_id of the table to update, and {tbl_id: schema_version} for its version path, captured when the diff was
    # computed.
    tbl_id: UUID
    schema_versions: dict[UUID, int]


def prepare_model_updates(
    tvp: catalog.TableVersionPath,
    display_name: str,
    new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]],
    new_idxs: list[IndexDeclaration],
) -> tuple[list[catalog.Column], list[catalog.IndexSpec]]:
    """
    Given `new_columns` and `new_idxs` as declared by a model, resolves them into proper catalog abstractions
    in preparation for catalog changes. This is the analog of `prepare_model()` for `update_all()`.

    Each column in `new_columns` is a (spec, origin) pair. A 'base_query' column comes from the view's base query
    `select()` list and is resolved against the base table's columns; a 'model_body' column is resolved against the
    view's own visible columns.
    """

    user_cols: dict[str, catalog.Column] = {}
    subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()  # ColumnRefByName -> ColumnRef

    # Pre-populate the user columns and substitution dict with the existing table's user columns.
    # This includes iterator columns and base table columns.
    for col in tvp.columns():
        user_cols[col.name] = col
        subst_dict[exprs.ColumnRefByName(col.name)] = exprs.ColumnRef(
            col.column_version_md(), perform_validation=(col.media_validation == MediaValidation.ON_READ)
        )

    # Base-query columns are projections of the base query and resolve against the base table's columns (which,
    # for a `select()` view, are not among the view's own visible columns above).
    has_base_query_cols = any(origin == 'base_query' for _, origin in new_columns.values())
    base_subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
    if has_base_query_cols:
        assert tvp.base is not None
        for col in tvp.base.columns():
            base_subst_dict[exprs.ColumnRefByName(col.name)] = exprs.ColumnRef(
                col.column_version_md(), perform_validation=(col.media_validation == MediaValidation.ON_READ)
            )

    tbl_handle = tvp.tbl_version

    # Process base-query columns first, so a model-body column may reference a newly-projected base-query column.
    ordered_names = [n for n, (_, origin) in new_columns.items() if origin == 'base_query'] + [
        n for n, (_, origin) in new_columns.items() if origin != 'base_query'
    ]

    # substitute ColumnRefByName with ColumnRef for existing columns; references to added columns are left as
    # ColumnRefByName, to be resolved once ids have been assigned
    resolved_cols: list[catalog.Column] = []
    preceding_names: set[str] = set()
    for name in ordered_names:
        spec, origin = new_columns[name]
        resolved_spec = spec.copy()
        if 'value' in resolved_spec:
            resolve_against = base_subst_dict if origin == 'base_query' else subst_dict
            resolved_spec['value'] = resolved_spec['value'].substitute(resolve_against)
            # whatever is still a ColumnRefByName has to name a column that precedes this one in resolved_cols:
            # anything else is either outside the model's scope or would be evaluated before it has a value
            unresolved = [
                ref for ref in resolved_spec['value'].subexprs(exprs.ColumnRefByName) if ref.name not in preceding_names
            ]
            if len(unresolved) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} in {display_name} references columns that are not in '
                    f"the model's scope: {[ref.name for ref in unresolved]}",
                )

        # a new column can only collide with an inherited one: a name already among this table's own columns
        # wouldn't have been diffed as new
        if exprs.ColumnRefByName(name) in subst_dict:
            assert tvp.base is not None
            raise excs.AlreadyExistsError(
                excs.ErrorCode.COLUMN_ALREADY_EXISTS,
                f'Column {name!r} already exists in the base table {tvp.base.tbl_name()!r}.',
            )

        catalog_col = catalog.Column.create(name, resolved_spec)
        catalog_col.tbl_handle = tbl_handle
        resolved_cols.append(catalog_col)
        preceding_names.add(name)
        user_cols[name] = catalog_col

    # Resolve each declared index against the model's visible columns.
    resolved_idxs: list[catalog.IndexSpec] = []
    for idx_spec in _resolve_model_idxs(new_idxs, user_cols, display_name):
        assert isinstance(idx_spec.indexed_column, str)
        resolved_idxs.append(idx_spec._replace(indexed_column=user_cols[idx_spec.indexed_column]))

    return resolved_cols, resolved_idxs


class SchemaChangeIndexRef(TypedDict):
    index_type: Literal['btree', 'embedding']
    columns: list[str]
    name: str | None


class SchemaChangeOpDetails(TypedDict, total=False):
    """Operands of a SchemaChangeOp, rendered as strings to survive serialization"""

    type: str  # the new type for a column add or alter
    value: str  # the new computed value expression for a column add or alter
    index_ref: SchemaChangeIndexRef  # the new index for an index add or alter


class SchemaChangeOp(TypedDict):
    """
    A single schema change operation (eg, add column, drop column, etc).

    Mirrored by pixeltable_cli.schema_types.SchemaChangeOp; adding, removing or retyping a field here means
    doing the same there.
    """

    target: Literal['column', 'index', 'table']

    # column name, index name, or for 'table', the differing attribute:
    # 'kind' | 'iterator' | 'view_filter' | 'view_sample' | 'media_validation' | 'comment' | 'custom_metadata'
    # can be None if target == 'index'.
    name: str | None

    op: Literal['add', 'drop', 'alter']
    severity: Literal['additive', 'destructive', 'unsupported']
    model: Any | None  # model-side value; None for drops
    existing: Any | None  # catalog-side value; None for adds
    description: str

    # the change's operands
    details: SchemaChangeOpDetails


# Mirrored by pixeltable_cli.schema_types.DiffResolution; a value added here has to be added there too
DiffResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported']


class TableDiff(TypedDict):
    """How one model differs from its catalog table.

    Mirrored by pixeltable_cli.schema_types.TableDiff; adding, removing or retyping a field here means doing
    the same there.
    """

    path: str  # catalog path of the table
    model_cls: str  # model class name, so an agent can map back to code
    kind: Literal['table', 'view']
    exists: bool
    resolution: DiffResolution
    ops: list[SchemaChangeOp]

    # identity of the existing table, as of the read this diff was computed from; None if it doesn't exist yet
    tbl_id: UUID | None

    # schema versions of the TableVersionPath
    schema_versions: dict[UUID, int] | None


# Table-level attribute names that are reported as a single grouped diff (as opposed to `kind`/`iterator`/`filter`/
# `sample`, which each get their own diff line).
_TABLE_PROP_NAMES: tuple[str, ...] = ('media_validation', 'comment', 'custom_metadata', 'has_default_idxs')


def _resolution(exists: bool, ops: list[SchemaChangeOp]) -> DiffResolution:
    """Reduce a table's list of operations to the single action `update_all()` would take."""
    if not exists:
        return 'create'
    if len(ops) == 0:
        return 'up_to_date'
    severities = {op['severity'] for op in ops}
    if 'unsupported' in severities:
        return 'unsupported'
    if 'destructive' in severities:
        return 'update_destructive'
    return 'update_additive'


@dataclasses.dataclass
class _ColumnProperties:
    """The comparable properties of a column, either from a model or from an existing table."""

    type: str
    value: str | None
    primary_key: bool
    stored: bool
    media_validation: str | None  # None for non-media columns
    comment: str | None
    custom_metadata: Any
    destination: str | None

    @classmethod
    def from_spec(cls, spec: ColumnSpec, default_media_validation: str) -> _ColumnProperties:
        """The comparable properties of a column declared by spec, resolved to match a stored column's metadata.

        A computed column's value expression carries ColumnRefByName placeholders, but those render identically to
        the ColumnRefs in the stored expression, so the display strings are directly comparable. Defaults mirror
        Column.create (stored=True, primary_key=False) and a media column's media_validation falls back to
        the table default, as it does on the stored column.
        """
        col_type = _col_type_from_spec(spec)
        value = spec.get('value')
        comment = spec.get('comment')
        return cls(
            type=repr(col_type),
            value=exprs.Expr.from_object(value).display_str(inline=False) if value is not None else None,
            primary_key=spec.get('primary_key', False),
            stored=spec.get('stored', True),
            media_validation=(spec.get('media_validation') or default_media_validation)
            if col_type.is_media_type()
            else None,
            comment=comment if comment else None,
            custom_metadata=spec.get('custom_metadata'),
            destination=str(spec['destination']) if spec.get('destination') is not None else None,
        )

    @classmethod
    def from_metadata(cls, col_md: ColumnMetadata) -> _ColumnProperties:
        """The comparable properties of an existing column, drawn from its `ColumnMetadata`."""
        return cls(
            type=col_md['type_'],
            value=col_md['computed_with'],
            primary_key=col_md['is_primary_key'],
            stored=col_md['is_stored'],
            media_validation=col_md['media_validation'],
            comment=col_md['comment'],
            custom_metadata=col_md['custom_metadata'],
            destination=col_md['destination'],
        )


@dataclasses.dataclass
class _TableProperties:
    """The comparable properties of a table, either from a model or from an existing table."""

    media_validation: str
    comment: str | None
    custom_metadata: Any

    @classmethod
    def from_model(cls, model: TableModelMeta) -> _TableProperties:
        """The comparable table-level properties declared by a model."""
        spec = model.__table_spec__
        return cls(
            media_validation=spec['media_validation'].name.lower(),
            comment=spec['comment'],
            custom_metadata=spec['custom_metadata'],
        )

    @classmethod
    def from_metadata(cls, md: TableMetadata) -> _TableProperties:
        """The comparable table-level properties of an existing table, drawn from its `TableMetadata`."""
        return cls(
            media_validation=md['media_validation'], comment=md['comment'], custom_metadata=md['custom_metadata']
        )


def _user_columns(model: TableModelMeta) -> dict[str, ColumnSpec]:
    """The model's declared columns, plus any its base query projects via a `select()` clause."""
    specs: dict[str, ColumnSpec] = dict(model.__columns__)
    base = model.__table_spec__['base']
    if base is not None and base.select_clause is not None:
        items, named_items = base.select_clause
        for item in items:
            assert isinstance(item, exprs.ColumnRefByName)  # "anonymous" compound expressions are not allowed here
            specs[item.name] = {'value': item, 'stored': False}
        for col_name, expr in named_items.items():
            specs[col_name] = {'value': expr, 'stored': not isinstance(expr, exprs.ColumnRefByName)}
    return specs


def _base_query_columns(model: TableModelMeta) -> set[str]:
    """Names of the columns a model's base query projects via its `select()` clause (empty if there is none)."""
    base = model.__table_spec__['base']
    if base is None or base.select_clause is None:
        return set()
    items, named_items = base.select_clause
    return {item.name for item in items} | set(named_items.keys())


def _format_column_spec(spec: ColumnSpec) -> str:
    """A display string for a column spec. The value expression is rendered via str() (not repr()), so a bare
    ColumnRefByName placeholder shows as its column name (e.g. extra1) rather than ColumnRefByName('extra1'),
    matching how it renders inside a compound expression and how the stored value expression renders."""
    parts = []
    for key, val in spec.items():
        rendered = str(val) if key == 'value' else repr(val)
        parts.append(f'{key!r}: {rendered}')
    return '{' + ', '.join(parts) + '}'


def _add_column_change(col_name: str, spec: ColumnSpec) -> SchemaChangeOp:
    details: SchemaChangeOpDetails = {'type': repr(_col_type_from_spec(spec))}
    value = spec.get('value')
    if value is not None:
        details['value'] = exprs.Expr.from_object(value).display_str(inline=False)
    return SchemaChangeOp(
        target='column',
        name=col_name,
        op='add',
        severity='additive',
        model=_format_column_spec(spec),
        existing=None,
        description=f'column {col_name!r} will be added',
        details=details,
    )


def _as_idx_ref(idx: IndexDeclaration) -> SchemaChangeIndexRef:
    if isinstance(idx, BtreeIndex):
        return SchemaChangeIndexRef(index_type='btree', columns=[idx.column.name], name=None)
    else:
        return SchemaChangeIndexRef(index_type='embedding', columns=[idx.column.name], name=idx.name)


def _add_index_change(idx: IndexDeclaration) -> SchemaChangeOp:
    # str(), not .name: a ModelColumnRef renders as its bare column name, and a spec holding anything else
    # is reported as it stands rather than dropped from the plan
    idx_ref = _as_idx_ref(idx)
    idx_name = idx_ref['name']
    return SchemaChangeOp(
        target='index',
        name=idx_name,
        op='add',
        severity='additive',
        model=str(idx),
        existing=None,
        description=(
            f'{type(idx).__name__} {idx_name!r} will be added'
            if idx_name is not None
            else f'{type(idx).__name__} on column(s) {idx_ref["columns"]!r} will be added'
        ),
        details={'index_ref': idx_ref},
    )


def validate_models(registered_models: dict[str, TableModelMeta], catalog_dir: str) -> dict[str, TableDiff]:
    """
    Analyze each registered model against the current catalog state, summarizing the schema changes that creating
    the models would entail, along with any incompatibilities with an already-existing table of the same name.
    This is purely informational: it neither modifies the catalog nor raises on incompatibilities.
    All metadata reads happen in a single transaction, and each diff's tbl_id and schema_versions record the catalog
    state it was computed against.
    """
    from .catalog import retry_loop

    catalog_dir = TableModelMeta._dir_prefix(catalog_dir)

    @retry_loop(for_write=False)
    def op() -> dict[str, TableDiff]:
        results: dict[str, TableDiff] = {}

        for name, model in registered_models.items():
            user_cols = _user_columns(model)
            model_cols = set(user_cols.keys())
            base = model.__table_spec__['base']
            model_kind: Literal['table', 'view'] = 'table' if base is None else 'view'
            iterator = model.__table_spec__['iterator']
            model_iterator = None if iterator is None else iterator.display_str()
            model_filter = None if base is None or base.where_clause is None else str(base.where_clause)
            model_sample = None if base is None or base.sample_clause is None else str(base.sample_clause)

            bound_path = f'{catalog_dir}{name}'
            existing = model._resolve_tbl(catalog_dir, if_not_exists='ignore')

            ops: list[SchemaChangeOp]

            if existing is None:
                # The table does not yet exist; every column and index is an addition.
                ops = [_add_column_change(col_name, user_cols[col_name]) for col_name in sorted(model_cols)]
                ops += [_add_index_change(idx) for idx in model.__indexes__]
                results[name] = TableDiff(
                    path=bound_path,
                    model_cls=model.__name__,
                    kind=model_kind,
                    exists=False,
                    resolution=_resolution(False, ops),
                    ops=ops,
                    tbl_id=None,
                    schema_versions=None,
                )
                continue

            existing_md = existing.get_metadata()
            tbl_path = existing._tbl_path

            # Restrict the existing columns to those defined in this table (i.e. not inherited from a base) and not
            # produced by an iterator, so that they line up with the model's own declared columns.
            existing_cols = {
                col_name
                for col_name, col_md in existing_md['columns'].items()
                if col_md['defined_in'] == existing_md['name'] and not col_md['is_iterator_col']
            }

            ops = []

            # has_default_idxs mismatch is unsupported.
            model_default_idxs = model.__table_spec__['has_default_idxs']
            existing_default_idxs = existing_md['has_default_idxs']
            if model_default_idxs != existing_default_idxs:
                ops.append(
                    SchemaChangeOp(
                        target='table',
                        name='has_default_idxs',
                        op='alter',
                        severity='unsupported',
                        model=model_default_idxs,
                        existing=existing_default_idxs,
                        description=f'`{model.__name__}` specifies has_default_idxs={model_default_idxs}, '
                        f'but {name!r} was created with has_default_idxs={existing_default_idxs}',
                        details={},
                    )
                )

            # Structural mismatches (kind/iterator/filter/sample); each is unsupported (requires a manual migration).
            if model_kind != existing_md['kind']:
                ops.append(
                    SchemaChangeOp(
                        target='table',
                        name='kind',
                        op='alter',
                        severity='unsupported',
                        model=model_kind,
                        existing=existing_md['kind'],
                        description=(
                            f'`{model.__name__}` specifies a {model_kind}, but {name!r} is a {existing_md["kind"]}'
                        ),
                        details={},
                    )
                )
            for attr, model_val, existing_val in (
                ('iterator', model_iterator, existing_md['iterator_call']),
                ('view_filter', model_filter, existing_md['view_filter']),
                ('view_sample', model_sample, existing_md['view_sample']),
            ):
                if model_val != existing_val:
                    ops.append(
                        SchemaChangeOp(
                            target='table',
                            name=attr,
                            op='alter',
                            severity='unsupported',
                            model=model_val,
                            existing=existing_val,
                            description=f'{attr} mismatch: model={model_val!r}, existing={existing_val!r}',
                            details={},
                        )
                    )

            # Table-level properties that differ (media_validation/comment/custom_metadata); unsupported for now.
            model_table_props = _TableProperties.from_model(model)
            existing_table_props = _TableProperties.from_metadata(existing_md)
            for prop in model_table_props.__dataclass_fields__:
                model_val = getattr(model_table_props, prop)
                existing_val = getattr(existing_table_props, prop)
                if model_val != existing_val:
                    ops.append(
                        SchemaChangeOp(
                            target='table',
                            name=prop,
                            op='alter',
                            severity='unsupported',
                            model=model_val,
                            existing=existing_val,
                            description=f'table property {prop!r}: model={model_val!r}, existing={existing_val!r}',
                            details={},
                        )
                    )

            # Columns present in both, whose properties differ; unsupported for now (some alterations will later be
            # applicable via allow_destructive=True).
            default_media_validation = model.__table_spec__['media_validation'].name.lower()
            for col_name in sorted(model_cols & existing_cols):
                model_props = _ColumnProperties.from_spec(user_cols[col_name], default_media_validation)
                existing_props = _ColumnProperties.from_metadata(existing_md['columns'][col_name])
                altered = [
                    prop
                    for prop in model_props.__dataclass_fields__
                    if getattr(model_props, prop) != getattr(existing_props, prop)
                ]
                if len(altered) > 0:
                    ops.append(
                        SchemaChangeOp(
                            target='column',
                            name=col_name,
                            op='alter',
                            severity='unsupported',
                            model={prop: getattr(model_props, prop) for prop in altered},
                            existing={prop: getattr(existing_props, prop) for prop in altered},
                            description=f'column {col_name!r} has altered properties: {", ".join(altered)}',
                            details={},
                        )
                    )

            # Additive/destructive column and index changes.
            for col_name in sorted(model_cols - existing_cols):
                ops.append(_add_column_change(col_name, user_cols[col_name]))
            for col_name in sorted(existing_cols - model_cols):
                ops.append(
                    SchemaChangeOp(
                        target='column',
                        name=col_name,
                        op='drop',
                        severity='destructive',
                        model=None,
                        existing=None,
                        description=f'column {col_name!r} will be dropped',
                        details={},
                    )
                )

            model_idxs = model.__indexes__
            existing_idxs = list(existing_md['indexes'].values())

            if model_default_idxs or existing_default_idxs:
                # If has_default_idxs is declared, then we don't need to compare B-tree indexes, since B-tree index
                # comparison is implicit in column comparison.
                model_idxs = [idx for idx in model_idxs if not isinstance(idx, BtreeIndex)]
                existing_idxs = [idx_md for idx_md in existing_idxs if idx_md['index_type'] != 'btree']

            # Diff the indexes. We first scan through `model_idxs` looking for matches in `existing_idxs`, removing
            # those matches as we find them. Anything left over in `existing_idxs` is flagged for removal.
            # TODO: The IndexMetadata structure technically allows for multicol indexes, but they're not supported yet;
            #     here we assume a single column
            for idx in model_idxs:
                if isinstance(idx, BtreeIndex):
                    # Btree index: they're parameterless, so we simply check if a btree index exists in the catalog
                    # for this column.
                    existing_btree_idxs = [
                        i
                        for i, idx_md in enumerate(existing_idxs)
                        if idx_md['columns'][0] == idx.column.name and idx_md['index_type'] == 'btree'
                    ]
                    assert len(existing_btree_idxs) <= 1
                    if len(existing_btree_idxs) == 0:
                        ops.append(_add_index_change(idx))
                    else:
                        existing_idxs.pop(existing_btree_idxs[0])
                elif idx.name is not None:
                    # Named embedding index: check if an index of the same name exists in the catalog.
                    # TODO: Allow for renaming embedding indexes?
                    existing_named_idxs = [
                        (i, idx_md)
                        for i, idx_md in enumerate(existing_idxs)
                        if idx_md['name'] == idx.name and idx_md['index_type'] == 'embedding'
                    ]
                    assert len(existing_named_idxs) <= 1
                    if len(existing_named_idxs) == 0:
                        ops.append(_add_index_change(idx))
                    else:
                        i, idx_md = existing_named_idxs[0]
                        if (
                            idx_md['columns'] != [idx.column.name]
                            or idx_md['parameters']['metric'] != idx.metric
                            or idx_md['parameters']['precision'] != idx.precision
                            or idx_md['parameters']['embedding'] != str(idx.as_fn_call())
                        ):
                            idx_ref = _as_idx_ref(idx)
                            ops.append(
                                SchemaChangeOp(
                                    target='index',
                                    name=idx_ref['name'],
                                    op='alter',
                                    severity='unsupported',
                                    model=str(idx),
                                    existing=idx_md,
                                    description=f'named index {idx.name!r} has altered properties',
                                    details={'index_ref': idx_ref},
                                )
                            )
                        existing_idxs.pop(i)
                else:
                    # Unnamed embedding index: check if an index of identical structure exists in the catalog.
                    matching_idxs = [
                        i
                        for i, idx_md in enumerate(existing_idxs)
                        if idx_md['index_type'] == 'embedding'
                        and idx_md['columns'] == [idx.column.name]
                        and idx_md['parameters']['metric'] == idx.metric
                        and idx_md['parameters']['precision'] == idx.precision
                        and idx_md['parameters']['embedding'] == str(idx.as_fn_call())
                    ]
                    assert len(matching_idxs) <= 1
                    if len(matching_idxs) == 0:
                        ops.append(_add_index_change(idx))
                    else:
                        existing_idxs.pop(matching_idxs[0])

            # Any remaining items in existing_idxs are indexes that exist in the catalog but not in the model.
            for idx_md in existing_idxs:
                idx_name = idx_md['name']
                idx_ref = SchemaChangeIndexRef(
                    index_type=idx_md['index_type'], columns=idx_md['columns'], name=idx_name
                )
                ops.append(
                    SchemaChangeOp(
                        target='index',
                        name=idx_name,
                        op='drop',
                        severity='destructive',
                        model=None,
                        existing=None,
                        description=f'index {idx_name!r} will be dropped',
                        details={'index_ref': idx_ref},
                    )
                )

            results[name] = TableDiff(
                path=bound_path,
                model_cls=model.__name__,
                kind=model_kind,
                exists=True,
                resolution=_resolution(True, ops),
                ops=ops,
                tbl_id=tbl_path.tbl_id,
                schema_versions=tbl_path.schema_versions(),
            )

        return results

    return op()


def _format_diff(name: str, diff: TableDiff) -> list[str]:
    """Human-readable lines describing how the model named `name` differs from the current catalog state."""
    kind = diff['kind']
    if not diff['exists']:
        return [
            f'{kind.capitalize()} {name!r} (from model `{diff["model_cls"]}`) does not yet exist, and will be CREATED.'
        ]

    ops = diff['ops']
    if len(ops) == 0:
        return []

    def by(target: str, op: str | None = None, names: tuple[str, ...] | None = None) -> list[SchemaChangeOp]:
        return [
            c
            for c in ops
            if c['target'] == target and (op is None or c['op'] == op) and (names is None or c['name'] in names)
        ]

    detail: list[str] = []

    for c in by('table', names=('kind',)):
        detail.append(f'  kind mismatch (FATAL): {c["description"]}')
    for attr, label in (('iterator', 'iterator'), ('view_filter', 'filter'), ('view_sample', 'sample')):
        for c in by('table', names=(attr,)):
            detail.append(f'  {label} mismatch (FATAL):')
            detail.append(f'    model {label}   : {c["model"]}')
            detail.append(f'    existing {label}: {c["existing"]}')

    table_props = by('table', names=_TABLE_PROP_NAMES)
    if len(table_props) > 0:
        detail.append('  the following table properties have changed (FATAL):')
        for c in table_props:
            detail.append(f'    {c["name"]}: model={c["model"]!r}, existing={c["existing"]!r}')

    altered_cols = by('column', op='alter')
    if len(altered_cols) > 0:
        detail.append('  the following columns have altered properties (FATAL):')
        for c in altered_cols:
            for prop, model_val in c['model'].items():
                detail.append(f'    {c["name"]!r} {prop}: model={model_val!r}, existing={c["existing"][prop]!r}')

    new_cols = by('column', op='add')
    if len(new_cols) > 0:
        detail.append('  the following columns are new to the model, and will be ADDED:')
        for c in new_cols:
            detail.append(f'    {c["name"]!r} = {c["model"]}')

    dropped_cols = by('column', op='drop')
    if len(dropped_cols) > 0:
        detail.append('  the following columns are no longer in the model, and will be DROPPED:')
        for c in dropped_cols:
            detail.append(f'    {c["name"]!r}')

    new_idxs = by('index', op='add')
    if len(new_idxs) > 0:
        detail.append('  the following indexes are new to the model, and will be ADDED:')
        for c in new_idxs:
            detail.append(f'    {c["model"]}')

    dropped_idxs = by('index', op='drop')
    if len(dropped_idxs) > 0:
        detail.append('  the following indexes are no longer in the model, and will be DROPPED:')
        for c in dropped_idxs:
            detail.append(f'    {c["name"]!r}')

    changed_idxs = by('index', op='alter')
    if len(changed_idxs) > 0:
        detail.append('  the following named indexes have altered properties (FATAL):')
        for c in changed_idxs:
            detail.append(f'    {c["name"]!r}')

    return [f'{kind.capitalize()} {name!r} (from model `{diff["model_cls"]}`) has differences:', *detail]


# closing lines of the refusals raised by create_all()/update_all(), phrased for the Python API
_PY_MISMATCH_HINT = 'Call `update_all()` instead if you intended to also modify existing tables.'
PY_DESTRUCTIVE_HINT = (
    'If you wish to apply these changes, re-run `update_all()` with `allow_destructive=True`.\n'
    'If you intended to rename columns or indexes instead of dropping them, apply those changes '
    'directly with `pxt.move()`.'
)


def model_base(cls_name: str = 'TableModel') -> type[TableModelMeta]:
    # mypy fundamentally does not understand metaclasses.
    cls = TableModelMeta(cls_name, (), {}, name='')
    registered_models: dict[str, TableModelMeta] = {}
    cls.__registered_models__ = registered_models  # type: ignore[attr-defined]

    def _bind_all(catalog_dir: str = '') -> None:
        for model in registered_models.values():
            model._bind(catalog_dir)

    def _create_models(catalog_dir: str, expect_created: set[str]) -> None:
        """Create every model that doesn't exist yet and bind all of them.

        Raises ConcurrencyError if a model named in expect_created already exists.
        """
        for name, model in registered_models.items():
            tbl, was_created = model._create(catalog_dir)
            if name in expect_created and not was_created:
                raise excs.ConcurrencyError(
                    excs.ErrorCode.CONCURRENT_MODIFICATION,
                    f'Table {str(tbl._path())!r} was created concurrently; re-run the operation.',
                )

    def _create_all(catalog_dir: str = '') -> dict[str, TableDiff]:
        """Returns the diff that was applied, per model: 'create' for the tables created now, 'up_to_date'
        for those that already matched. Raises rather than returning a partially applied diff."""
        # `create_all()` only creates tables; it never mutates an existing one. If any existing table differs from
        # its model, refuse.
        diffs = validate_models(registered_models, catalog_dir)
        changed = [(name, d) for name, d in diffs.items() if d['exists'] and d['resolution'] != 'up_to_date']
        if len(changed) > 0:
            detail = '\n'.join(line for name, d in changed for line in _format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                f'One or more existing tables differ from their models.\n{detail}\n{_PY_MISMATCH_HINT}',
            )

        _create_models(catalog_dir, {name for name, d in diffs.items() if not d['exists']})
        return diffs

    def _get_model_diff(catalog_dir: str = '') -> dict[str, TableDiff]:
        return validate_models(registered_models, catalog_dir)

    def _diff_all(catalog_dir: str = '') -> None:
        diffs = _get_model_diff(catalog_dir)
        lines: list[str] = []
        for name, d in diffs.items():
            lines.extend(_format_diff(name, d))
        Env.get().console_logger.info('\n'.join(lines) if len(lines) > 0 else 'Catalog is up to date.')

    def _update_all(catalog_dir: str = '', *, allow_destructive: bool = False) -> dict[str, TableDiff]:
        """Reconcile every registered model with the catalog.

        Returns the diff that was applied, per model. The compare-and-swap in update_from_model() and the
        concurrency check in _create_models() both abort if the catalog moved, so the returned diff is what
        reached the store.

        Not atomic: migrations and creations run in separate transactions, so a failure raises with part of the
        diff applied. Re-running reconciles whatever is left.

        Destructive changes without allow_destructive raise DESTRUCTIVE_SCHEMA_CHANGE.
        """
        diffs = validate_models(registered_models, catalog_dir)

        if len(diffs) == 0:
            # No updates *or* create statements.
            Env.get().console_logger.info('Catalog is up to date.')
            return diffs

        fatal = [(name, d) for name, d in diffs.items() if d['resolution'] == 'unsupported']
        if len(fatal) > 0:
            detail = '\n'.join(line for name, d in fatal for line in _format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                'One or more tables cannot be updated, because their models are inconsistent '
                'with the existing table(s) in the catalog.\n'
                f'{detail}\n'
                'Adjust the existing table(s) manually, or adjust the models to be consistent with the catalog.',
            )

        destructive = [(name, d) for name, d in diffs.items() if d['resolution'] == 'update_destructive']
        if len(destructive) > 0 and not allow_destructive:
            detail = '\n'.join(line for name, d in destructive for line in _format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
                f'The following updates would result in destructive catalog changes.\n{detail}\n{PY_DESTRUCTIVE_HINT}',
            )

        # Apply column/index changes to existing tables. Brand-new tables are handled by `_create_all()` below.
        update_diffs = [
            (name, d) for name, d in diffs.items() if d['resolution'] in ('update_additive', 'update_destructive')
        ]

        if len(update_diffs) > 0:
            catalog_dir = TableModelMeta._dir_prefix(catalog_dir)
            change_sets: list[TableSchemaChangeSet] = []
            for name, d in update_diffs:
                model = registered_models[name]
                new_col_names = {c['name'] for c in d['ops'] if c['target'] == 'column' and c['op'] == 'add'}
                dropped_col_names = [c['name'] for c in d['ops'] if c['target'] == 'column' and c['op'] == 'drop']
                new_idx_refs = [
                    c['details']['index_ref'] for c in d['ops'] if c['target'] == 'index' and c['op'] == 'add'
                ]
                dropped_idx_names = [c['name'] for c in d['ops'] if c['target'] == 'index' and c['op'] == 'drop']

                # Resolve `type` annotations to ColumnTypes, mirroring `_create()`, and tag each column's origin.
                # Iterate in declaration order (not the diff's sorted order), so a new column may depend on an
                # earlier new column, as it can at create time.
                user_cols = _user_columns(model)
                base_query_cols = _base_query_columns(model)
                new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]] = {}
                for col_name, col_spec in user_cols.items():
                    if col_name not in new_col_names:
                        continue
                    spec = col_spec.copy()
                    if 'type' in spec:
                        spec['type'] = ts.ColumnType.normalize_type(  # type: ignore[typeddict-item]
                            spec['type'], allow_builtin_types=False
                        )
                    origin: Literal['base_query', 'model_body'] = (
                        'base_query' if col_name in base_query_cols else 'model_body'
                    )
                    new_columns[col_name] = (spec, origin)

                # resolve idx_refs to IndexDeclarations. (We can't simply go by index name, since there may be unnamed
                # indexes.) Instead we compare the (index_type, name, columns) tuple; if there are two unnamed indexes
                # with the same type, then they *must* have different columns, so the tuple uniquely identifies the
                # index.
                new_idxs: list[IndexDeclaration] = []
                for idx_ref in new_idx_refs:
                    matching_idxs = [
                        idx
                        for idx in model.__indexes__
                        if (idx_ref['index_type'] == 'btree') == isinstance(idx, BtreeIndex)
                        and idx_ref['name'] == (idx.name if isinstance(idx, EmbeddingIndex) else None)
                        and [idx.column.name] == idx_ref['columns']
                    ]
                    assert len(matching_idxs) == 1
                    new_idxs.append(matching_idxs[0])

                # only an existing table is updated, so the diff recorded what it was computed against
                assert d['tbl_id'] is not None and d['schema_versions'] is not None
                change_sets.append(
                    TableSchemaChangeSet(
                        path=catalog.Path.parse(f'{catalog_dir}{name}'),
                        new_columns=new_columns,
                        dropped_columns=dropped_col_names,
                        new_idxs=new_idxs,
                        dropped_idxs=dropped_idx_names,
                        tbl_id=d['tbl_id'],
                        schema_versions=d['schema_versions'],
                    )
                )

            # All models share `catalog_dir`, hence a single catalog; apply every table's changes in one transaction.
            cat = get_runtime().get_catalog(change_sets[0]['path'])
            cat.update_from_model(change_sets)

        # Now create any new tables, and bind every model to its table. The diff computed above is the one being
        # applied, so the models it found up-to-date are not re-examined against the catalog.
        try:
            _create_models(catalog_dir, {name for name, d in diffs.items() if d['resolution'] == 'create'})
        except excs.Error as e:
            # the migrations above are already committed; name them, so that a failure here doesn't read as
            # though the catalog were untouched. Augmenting in place keeps the exception's type and fields.
            # e.message excludes e.detail, which is diagnostic text that must not become part of the message
            if len(update_diffs) > 0:
                migrated = ', '.join(repr(d['path']) for _, d in update_diffs)
                e.args = (
                    f'{e.message}\n\nThe following table(s) were already migrated: {migrated}. '
                    'Re-run update_all() to finish reconciling.',
                )
            raise
        return diffs

    cls.bind_all = _bind_all  # type: ignore[attr-defined]
    cls.create_all = _create_all  # type: ignore[attr-defined]
    cls.get_model_diff = _get_model_diff  # type: ignore[attr-defined]
    cls.diff_all = _diff_all  # type: ignore[attr-defined]
    cls.update_all = _update_all  # type: ignore[attr-defined]

    return cls  # type: ignore[return-value]
