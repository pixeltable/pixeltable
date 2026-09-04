"""The declaration vocabulary a schema file is written in, and the metaclass that captures it.

Case sensitivity
----------------
A class body is Python; the catalog is Pixeltable. Names follow the rules of their own domain, and fold where they
cross -- in resolution.prepare_model() and diff.user_columns()/base_query_columns(), not here.

A column the model declares is a Python *binding* and keeps its spelling until the crossing: __columns__ is keyed as
written, and a re-cased class-body reference is a NameError. Once the class is bound, M.MyCol and M.mycol are the same
attribute, but M.MYCOL raises while M.table.MYCOL resolves. A name that is only a string denoting a Pixeltable
identifier (name=, EmbeddingIndex(name=...)) folds on arrival.

The namespace binds each key to a ColumnRefByName, which folds the name it carries: the binding is Python, the
reference is Pixeltable, so _bind() needs no side table.

Hence declaring one spelling twice fails here, while Foo and foo -- two good Python names -- collide only at
the crossing, in create_all() and the update_all() diff.
"""

# ruff: noqa: N804  # Neither mypy nor ruff seems to understand metaclasses.

from __future__ import annotations
import __future__

import dataclasses
import sys
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal, MutableMapping, Sequence, TypedDict, cast
from uuid import UUID, uuid4

from typing_extensions import TypeForm

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.config import URI, ConfigVar
from pixeltable.env import Env
from pixeltable.exprs import ColumnRefByName
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from ..globals import MediaValidation, col_type_from_spec, fold_identifier, is_valid_identifier
from ..table import Table
from ..table_version_handle import TableVersionHandle
from ..types import TableVersionMd
from ..utils import create_table_version_md
from .resolution import prepare_model

if TYPE_CHECKING:
    from .query import ModelQuery

# the model each declared path was synthesized for, keyed by its synthesized table id; a query over a model
# consults this to name the model it is declared over and to bind itself to that model's table
MODEL_BY_DECLARED_TBL_ID: dict[UUID, 'TableModelMeta'] = {}

# Table methods exposed as class-level operations on the model.
# how an unbound model answers each method it forwards: a query over what it declares carries a clause or
# prints itself, and reading or writing rows needs the table that does not exist yet. Every other forwarded
# method is the table's alone, and reports the model as unbound.
DECLARABLE_QUERY_METHODS: frozenset[str] = frozenset(
    ('describe', 'distinct', 'group_by', 'join', 'limit', 'order_by', 'sample', 'select', 'where')
)
ROW_METHODS: frozenset[str] = frozenset(
    ('collect', 'count', 'cursor', 'delete', 'head', 'recompute_columns', 'show', 'tail', 'update')
)

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
assert DECLARABLE_QUERY_METHODS <= FORWARDED_TABLE_METHODS
assert ROW_METHODS <= FORWARDED_TABLE_METHODS
assert not (DECLARABLE_QUERY_METHODS & ROW_METHODS)


@dataclasses.dataclass(frozen=True)
class Column:
    """A column specification used in a TableModel or ViewModel definition."""

    type: TypeForm | None = None
    value: Any = None
    primary_key: bool | None = None
    stored: bool | None = None
    media_validation: Literal['on_read', 'on_write'] | None = None
    destination: str | Path | ConfigVar[URI] | None = None
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

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            # This is a frozen dataclass, we have to use __setattr__ to update it.
            object.__setattr__(self, 'name', fold_identifier(self.name))

    def resolved_embeddings(self) -> dict[ts.ColumnType.Type, func.Function]:
        """The embedding function serving each column type.

        The functions are rendered the way index renders them: the resolution binds a function's defaults and picks the
        overload matching each type.
        """
        from pixeltable import index

        return index.EmbeddingIndex.resolve_embeddings(
            self.embedding, self.string_embed, self.image_embed, self.audio_embed, self.video_embed, self.document_embed
        )

    def as_fn_call(self) -> exprs.FunctionCall:
        """The embedding call for the indexed column's own type, which is what the index materializes."""
        assert isinstance(self.column, exprs.ColumnRefByName)
        col_type = self.column.col_type
        embeddings = self.resolved_embeddings()
        if col_type._type not in embeddings:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'EmbeddingIndex has no embedding function defined for type: {col_type}'
            )
        return embeddings[col_type._type](self.column)

    def resolved_embedding_fns(self) -> list[str]:
        """The rendered embedding functions, as the index's metadata records them."""
        return [str(fn) for fn in self.resolved_embeddings().values()]

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
    is_data_versioned: bool


def _contains_aggregate(expr: exprs.Expr) -> bool:
    """Whether the expression computes a value over a set of rows rather than from one row."""
    return expr.contains_(cls=exprs.FunctionCall, filter=lambda e: cast(exprs.FunctionCall, e).is_agg_fn_call)


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
    # Keyed by folded name.
    reserved_cols: dict[str, Literal['base table', 'base query', 'iterator']]

    # The scope in which the class body is defined; used to evaluate stringized type annotations (see
    # set_col_type()) and prebind annotations.
    caller: FrameType

    # The next two are populated only by prebind_annotations().

    # Annotation types for names that are also assigned in the body, applied after the assignment so that
    # the type check happens in the order it would under eager annotations.
    pending_ann_types: dict[str, Any]

    # Column names in source-declaration order; None on the eager path, where the body itself orders them.
    decl_order: list[str] | None

    def __init__(self, table_spec: TableSpec, caller: FrameType) -> None:
        super().__init__()

        self.table_spec = table_spec
        self.known_cols = {}
        self.reserved_cols = {}
        self.caller = caller
        self.pending_ann_types = {}
        self.decl_order = None

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
            if key in self.pending_ann_types:
                # An annotated assignment (`col: SomeType = Column(...)`). Under eager annotations the
                # annotation is recorded right after the value, so apply it here to get the same check.
                self.set_col_type(key, self.pending_ann_types.pop(key))

    def add_reserved_column_ref(
        self, name: str, col_type: ts.ColumnType, kind: Literal['base table', 'base query', 'iterator']
    ) -> None:
        """Add `name` as a reserved column (it is resolvable in the class body, and its symbol cannot be reused,
        but it does not have a ColumnSpec and will not be included in the list of columns for the view to create).
        """
        assert name == fold_identifier(name)
        self.reserved_cols[name] = kind
        super().__setitem__(name, ColumnRefByName(name, col_type))

    def _check_reserved(self, name: str) -> None:
        """Reject `name` if it is already produced by the base query or the iterator."""
        kind = self.reserved_cols.get(fold_identifier(name))
        if kind is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{name!r} is already defined by the {kind}; it cannot be redeclared.'
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
            if _contains_aggregate(expr):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} aggregates over the table: {expr}\nA computed column is evaluated one '
                    'row at a time, so it cannot aggregate.',
                )
            spec = {'value': expr}
        self.known_cols[name] = spec
        # Add the column to the namespace so that it can be referenced in subsequent expressions in the class body.
        super().__setitem__(name, exprs.ColumnRefByName(name, col_type_from_spec(spec)))

    def set_col_type(self, name: str, type_: Any) -> None:
        self._check_reserved(name)
        if isinstance(type_, str):
            # Under from __future__ import annotations (PEP 563), annotations arrive as strings. Evaluate
            # the string in the scope where the model class is defined to recover the actual type.
            try:
                type_ = eval(type_, self.caller.f_globals, self.caller.f_locals)
            except Exception as exc:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Could not resolve the type annotation {type_!r} for column {name!r}: {exc}',
                ) from exc
        type_ = ts.ColumnType.normalize_type(type_, allow_builtin_types=False)
        if name in self.known_cols:
            # We previously processed this column via set_col_value(). Sanity check the type.
            if col_type_from_spec(self.known_cols[name]) != type_:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'Conflicting type annotation for column {name!r}.'
                )
            return
        # Bare annotation (col: SomeType): record the spec and make the name referenceable in the body.
        self.known_cols[name] = {'type': type_}  # type: ignore[typeddict-item]
        super().__setitem__(name, exprs.ColumnRefByName(name, type_))

    def prebind_annotations(self, cls_name: str, display_name: str) -> None:
        """
        Register the class body's annotations before the body runs, for the PEP 649 deferred-annotation
        path where they produce no namespace operations of their own.

        Bare annotations are registered immediately, so that the names resolve as typed column references
        in the statements that follow. Names that are also assigned in the body are deferred to
        `__setitem__`, which applies them once the assignment has been processed.
        """
        from . import _code_frame_utils  # inline import, since we only need this on Python 3.14+

        body_code = _code_frame_utils.find_class_body_code(self.caller, cls_name)
        if body_code is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{display_name}: could not resolve column type annotations; try adding '
                '`from __future__ import annotations` to your module.',
            )

        try:
            annotation_types = _code_frame_utils.evaluate_annotations(body_code, self, self.caller.f_globals)
        except Exception as exc:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{display_name}: could not resolve column type annotations; try adding '
                '`from __future__ import annotations` to your module.',
            ) from exc

        annotations = _code_frame_utils.annotation_lines(body_code)
        assign_lines = _code_frame_utils.assignment_lines(body_code)

        # `annotation_types` is the authoritative set of annotations; the recovered lines only order them. Ignore
        # any name the two don't agree on, and fall back to appending annotations the line scan missed.
        ordered_anns = [(col_name, line) for col_name, line in annotations if col_name in annotation_types]
        found = {col_name for col_name, _line in ordered_anns}
        ordered_anns.extend((col_name, sys.maxsize) for col_name in annotation_types if col_name not in found)

        # Declaration order is the source order of annotations and assignments interleaved. A name that is
        # both annotated and assigned occupies a single position, contributed by whichever comes first.
        by_line = sorted(ordered_anns + assign_lines, key=lambda entry: entry[1])
        decl_order: list[str] = []
        for col_name, _line in by_line:
            if col_name not in decl_order:
                decl_order.append(col_name)
        self.decl_order = decl_order

        assigned = {col_name for col_name, _line in assign_lines}
        for col_name, _line in ordered_anns:
            if col_name.startswith('_'):
                continue  # not a column, matching the eager path
            if col_name in assigned:
                self.pending_ann_types[col_name] = annotation_types[col_name]
            else:
                self.set_col_type(col_name, annotation_types[col_name])

    def apply_decl_order(self) -> None:
        """Reorder `known_cols` into source-declaration order, if it is known independently of the body."""
        if self.decl_order is None:
            return
        ordered = {name: self.known_cols[name] for name in self.decl_order if name in self.known_cols}
        ordered.update({name: spec for name, spec in self.known_cols.items() if name not in ordered})
        self.known_cols = ordered


def _bind_query_templates(e: exprs.Expr, catalog_dir: str) -> exprs.Expr:
    """Rebind QueryTemplateFunction calls of ModelQuery instances to the equivalent Query of the bound model."""
    from .query import ModelQuery

    subst: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
    for fn_call in e.subexprs(exprs.FunctionCall):
        fn = fn_call.fn
        if not isinstance(fn, func.QueryTemplateFunction) or not isinstance(fn.template_query, ModelQuery):
            continue
        assert fn_call.group_by_start_idx == fn_call.group_by_stop_idx  # a query udf takes no window clause
        rebound = func.QueryTemplateFunction(
            fn.template_query.bind(catalog_dir),
            list(fn.signature.parameters.values()),
            return_scalar=fn.return_scalar,
            path=fn.self_path,
            name=fn.self_name,
            comment=fn.comment(),
        )
        # we need to substitute the FunctionCall itself, not just the Function instance
        subst[fn_call] = rebound(*fn_call.args, **fn_call.kwargs)
    if len(subst) == 0:
        return e
    result = e.substitute(subst)
    # substitute() returns a replacement without descending into it, so a call nested in another one is missed
    assert not any(
        isinstance(c.fn, func.QueryTemplateFunction) and isinstance(c.fn.template_query, ModelQuery)
        for c in result.subexprs(exprs.FunctionCall)
    )
    return result


class TableModelMeta(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    __table_spec__: TableSpec
    __columns__: dict[str, ColumnSpec]
    __indexes__: list[IndexDeclaration]
    __bound_table__: Table | None

    _catalog_dir: str | None
    _table_path: catalog.TableMdPath | None

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
        _is_data_versioned: bool = True,
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
            if not isinstance(name, str) or not is_valid_identifier(name, allow_hyphens=True):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'{display_name}: `name` must be a valid Pixeltable identifier.'
                )
            tbl_name = fold_identifier(name)

            base_models = bases[0].__registered_models__  # type: ignore[attr-defined]
            if tbl_name in base_models:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'{display_name} has name {tbl_name!r}, but that name was '
                    f'previously used by `{base_models[tbl_name].__name__}`.',
                )

            # Validate base
            if base is not None:
                # imported here because query.py imports this module at module scope, so importing it back
                # at module scope would leave one of the two half-initialized
                from .query import ModelQuery

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
                base.validate(display_name)
                base_model = base.model_cls
                if len(base_model.__bases__) == 0 or base_model.__bases__[0] is not bases[0]:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'{display_name}: `base` must reference a Pixeltable model with the same '
                        f'`model_base()` as `{cls_name}`.',
                    )

            # A view is always of the same kind as its base. Operational views aren't supported yet.
            if base is not None and not _is_data_versioned:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'{display_name}: operational views are not supported yet.'
                )
            if base is not None and not base.model_cls.__table_spec__['is_data_versioned']:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{display_name}: the base table and the view have mismatching is_data_versioned',
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
            # (see _ModelNamespace.set_col_type) can be evaluated, and so that deferred annotations can be
            # bound promptly on Python 3.14+.
            # sys._getframe(1) is the frame executing the class ...: statement
            # (__build_class__ is a C function and creates no frame).
            caller = sys._getframe(1)

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
                    'is_data_versioned': _is_data_versioned,
                },
                caller=caller,
            )

            if base is not None:
                # Make the select list's named columns referenceable in the body.
                for expr, col_name in base._effective_select_list:
                    if col_name is None:
                        continue
                    assert is_valid_identifier(col_name)  # since it must be a Python symbol
                    namespace.add_reserved_column_ref(
                        col_name, expr.col_type, 'base table' if base.select_list is None else 'base query'
                    )

            if iterator is not None:
                # Likewise for the iterator's outputs: referenceable, but created by the iterator.
                for col_name, output in iterator.outputs.items():
                    assert is_valid_identifier(col_name)
                    namespace.add_reserved_column_ref(col_name, output.col_type, 'iterator')

            has_future_annotations = bool(caller.f_code.co_flags & __future__.annotations.compiler_flag)
            if sys.version_info >= (3, 14) and not has_future_annotations:
                # On Python 3.14+ without `from __future__ import annotations`, annotations are not evaluated eagerly
                # (PEP 649), so the model's column annotations produce no namespace operations and body references to
                # them would raise `NameError` before we ever reach `__new__`. In this situation we recover the
                # annotations from the class body's code object instead and make them available promptly.
                namespace.prebind_annotations(cls_name, display_name)

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
        namespace.apply_decl_order()

        if len(namespace.known_cols) == 0 and namespace.table_spec['base'] is None:
            raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, 'Empty table schema not allowed.')

        # "normalize" the namespace to a plain dict; at this point, we're done with the special namespace treatment
        namespace_dict = dict(namespace)
        namespace_dict['__table_spec__'] = namespace.table_spec
        namespace_dict['__columns__'] = namespace.known_cols
        namespace_dict['__bound_table__'] = None
        namespace_dict['_catalog_dir'] = None
        namespace_dict['_table_path'] = None

        known_idxs = namespace_dict.get('__indexes__', [])
        if not isinstance(known_idxs, Sequence) or not all(isinstance(idx, IndexDeclaration) for idx in known_idxs):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'model `{cls_name}`: `__indexes__` must be a sequence of EmbeddingIndex or BtreeIndex instances.',
            )
        mcs._validate_indexes(cls_name, namespace, known_idxs)
        namespace_dict['__indexes__'] = list(known_idxs)  # normalize

        cls = super().__new__(mcs, cls_name, bases, namespace_dict)
        # the placeholders were built while the class body ran, before there was a class to point at
        for value in namespace_dict.values():
            if isinstance(value, ColumnRefByName):
                value.model_cls = cls
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

    def declared_models(cls) -> list[TableModelMeta]:
        """The models declared on this base, in declaration order."""
        return list(cls.__registered_models__.values())

    def referenced_functions(cls) -> list[func.Function]:
        """Every function this model references, without duplicates."""
        declared_exprs: list[exprs.Expr] = [
            col_spec['value'] for col_spec in cls.__columns__.values() if col_spec.get('value') is not None
        ]
        base = cls.__table_spec__['base']
        if base is not None:
            declared_exprs.extend(e for e, _ in base._effective_select_list)
            declared_exprs.extend(base._component_exprs())
        iterator = cls.__table_spec__['iterator']
        if iterator is not None:
            declared_exprs.extend(iterator.args)
            declared_exprs.extend(iterator.kwargs.values())

        fns = [fn_call.fn for e in declared_exprs for fn_call in e.subexprs(exprs.FunctionCall)]
        fns.extend(
            embedding
            for idx in cls.__indexes__
            if isinstance(idx, EmbeddingIndex)
            for embedding in (
                idx.embedding,
                idx.string_embed,
                idx.image_embed,
                idx.audio_embed,
                idx.video_embed,
                idx.document_embed,
            )
            if embedding is not None
        )
        return list(dict.fromkeys(fns))

    def _bind(cls, catalog_dir: str = '') -> pxt.Table:
        catalog_dir = catalog.Path.dir_prefix(catalog_dir)

        tbl = cls._resolve_tbl(catalog_dir, if_not_exists='error')

        if cls.is_bound:
            return tbl

        else:
            col_refs = {col_name: getattr(tbl, col_name) for col_name in tbl.columns()}

            # Table ops succeeded; now update the class.
            for col_name, col_ref in col_refs.items():
                setattr(cls, col_name, col_ref)
            # Replace ColumnRefByName placeholders with actual column references. ColumnRefByName.name is a folded
            # name, and so are the keys of col_refs.
            for attr, value in list(vars(cls).items()):
                if isinstance(value, exprs.ColumnRefByName) and value.name in col_refs:
                    setattr(cls, attr, col_refs[value.name])
            cls._catalog_dir = catalog_dir
            return tbl

    def _create(cls, catalog_dir: str = '') -> tuple[Table, bool]:
        """Returns the table and whether it was created now (False if it already existed)."""
        catalog_dir = catalog.Path.dir_prefix(catalog_dir)

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
            base = table_spec['base'].bind(catalog_dir)

        # The model's own column specs, with type annotations resolved to ColumnTypes (so they're serializable
        # for a proxied catalog). Computed value expressions still carry ColumnRefByNames referencing
        # sibling and base columns; those are substituted by the catalog that owns the table (create_from_model).
        columns: dict[str, ColumnSpec] = {}
        for name, col_spec in cls.__columns__.items():
            spec = col_spec.copy()
            if 'type' in spec:
                spec['type'] = ts.ColumnType.normalize_type(  # type: ignore[typeddict-item]
                    spec['type'], allow_builtin_types=False
                )
            if 'value' in spec:
                spec['value'] = _bind_query_templates(spec['value'].copy(), catalog_dir)
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
            is_data_versioned=table_spec['is_data_versioned'],
        )

        if was_created:
            Env.get().console_logger.info(f'Created {tbl._path()!r} from {table_spec["display_name"]}.')

        return cls._bind(catalog_dir), was_created

    def __getattr__(cls, item: str) -> Any:
        if item in FORWARDED_TABLE_METHODS:
            if not cls.is_bound and item in DECLARABLE_QUERY_METHODS:
                from .query import ModelQuery

                # a query over what this model declares, which carries the clause this call adds
                return getattr(ModelQuery.for_model(cls), item)
            if not cls.is_bound and item in ROW_METHODS:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{item}(): `{cls.__name__}`, which is not bound to a table, holds no rows; '
                    f'create the table with `{cls.__name__}.create()` or `pxt.create_all()` first.',
                )
            try:
                return getattr(cls.table, item)
            except excs.RequestError as exc:
                raise AttributeError(f'{item}(): {exc}') from exc
        return super().__getattribute__(item)

    def table_path(cls) -> catalog.TableMdPath:
        """The TableMdPath constructed from metadata for this model."""
        if cls._table_path is not None:
            return cls._table_path

        from ..view import View

        spec = cls.__table_spec__
        tbl_id = uuid4()  # we need a table id
        handle = TableVersionHandle(catalog.TableVersionKey(tbl_id, None))
        base = None if spec['base'] is None else spec['base'].to_declared_query()
        # prepare_model() substitutes column references in place, so hand it copies: inspecting a model must
        # leave what it declares untouched
        columns: dict[str, ColumnSpec] = {}
        for col_name, col_spec in cls.__columns__.items():
            copied = col_spec.copy()
            if 'value' in copied:
                copied['value'] = copied['value'].copy()
            columns[col_name] = copied
        iterator, cols, idxs = prepare_model(
            handle, columns, spec['display_name'], spec['iterator'], base, cls.__indexes__, spec['is_data_versioned']
        )

        md: TableVersionMd
        base_md: list[TableVersionMd] = []
        if base is None:
            # create_table_version_md() requires IndexSpec.indexed_column to be a Column, not str;
            # View._create_md() takes the name and resolves it against the view's visible columns
            cols_by_name = {col.name: col for col in cols if col.name is not None}
            idxs = [
                idx._replace(indexed_column=cols_by_name[idx.indexed_column])
                if isinstance(idx.indexed_column, str) and idx.indexed_column in cols_by_name
                else idx
                for idx in idxs
            ]
            md = create_table_version_md(
                tbl_id=tbl_id,
                name=spec['name'],
                cols=cols,
                comment=spec['comment'],
                custom_metadata=spec['custom_metadata'],
                media_validation=spec['media_validation'],
                has_default_idxs=spec['has_default_idxs'],
                view_md=None,
                is_data_versioned=spec['is_data_versioned'],
                additional_idxs=idxs,
            )
        else:
            base_path = base._from_clause._first_tbl
            assert isinstance(base_path, catalog.TableMdPath)
            md = View._create_md(
                tbl_id=tbl_id,
                name=spec['name'],
                base=base_path,
                select_list=base.select_list,
                additional_columns=cols,
                predicate=base.where_clause,
                sample_clause=base.sample_clause,
                is_snapshot=False,  # a model has no way to declare one
                has_default_idxs=spec['has_default_idxs'],
                comment=spec['comment'],
                custom_metadata=spec['custom_metadata'],
                media_validation=spec['media_validation'],
                iterator_call=iterator,
                additional_idxs=idxs,
            )
            # from_md() takes the path leaf first, so walk the base's own chain out of it
            while True:
                base_md.append(base_path.md)
                if base_path.base is None:
                    break
                base_path = base_path.base

        MODEL_BY_DECLARED_TBL_ID[tbl_id] = cls
        cls._table_path = catalog.TableMdPath.from_md(
            [md, *base_md], is_anon_snapshot=False, catalog_uri=catalog.path.ROOT_PATH
        )
        return cls._table_path

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
