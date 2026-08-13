"""The declaration vocabulary a schema file is written in, and the metaclass that captures it."""

from __future__ import annotations
import __future__

import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, MutableMapping, TypedDict, cast
from uuid import UUID, uuid4

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.config import URI, ConfigVar
from pixeltable.env import Env
from pixeltable.exprs import ColumnRefByName
from pixeltable.query_clauses import FromClause, JoinType, SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from ..globals import MediaValidation, col_type_from_spec, is_valid_identifier
from ..metadata_types import TableVersionMd
from ..table import Table
from ..table_version_handle import TableVersionHandle
from ..utils import create_table_version_md
from .resolution import prepare_model

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

    type: type | None = None
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


def _contains_aggregate(expr: exprs.Expr) -> bool:
    """Whether the expression computes a value over a set of rows rather than from one row."""
    return expr.contains_(cls=exprs.FunctionCall, filter=lambda e: cast(exprs.FunctionCall, e).is_agg_fn_call)


# the model a declared path was synthesized for, so that a query over it can name and bind its model
_MODEL_BY_DECLARED_TBL_ID: dict[UUID, TableModelMeta] = {}


class ModelQuery(pxt.Query):
    """A query declared over a model, before that model is bound to a table.

    Its from-clause is the shape the model declares, so it is a Query in every respect except that it cannot
    be executed: the columns it references belong to a table that does not exist yet. bind() produces the
    equivalent query over a real table.
    """

    @property
    def model_cls(self) -> TableModelMeta:
        """The model whose declared shape this query is written against."""
        model_cls = _MODEL_BY_DECLARED_TBL_ID.get(self._from_clause.tbls[0].tbl_id)
        assert model_cls is not None, self._from_clause.tbls[0].tbl_id
        return model_cls

    @classmethod
    def for_model(cls, model_cls: TableModelMeta) -> ModelQuery:
        """A query over everything model_cls declares."""
        return cls(from_clause=FromClause(tbls=[model_cls.table_path()]))

    def validate(self, model_name: str) -> None:
        """Validate that this query can be used to define a view."""
        from ..view import View

        View.validate_view_query(self, prefix=f'{model_name}: ')

        # a view model turns each select() item into a class attribute, so every item needs a name
        if self.select_list is None:
            return
        for item, name in self.select_list:
            if name is None and not item.is_column_ref:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'{model_name}: `base` select() list may contain only direct column references '
                    f'or named expressions, but contains an anonymous compound expression: {item}\n'
                    f'Use kwargs syntax to give it an explicit name: select(my_name=...)',
                )

    def to_declared_query(self) -> pxt.Query:
        """The equivalent query whose column references identify the columns of the model's declared shape.

        Metadata assembly distinguishes a column reference from a computed expression, which a query that only
        names its columns cannot support.
        """
        declared_path = self._from_clause.tbls[0]
        subst: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
        for col_md in declared_path.column_md():
            if col_md.name is not None:
                subst[ColumnRefByName(col_md.name)] = exprs.ColumnRef(col_md)
        return self._substituted(declared_path, subst)

    def bind(self, catalog_dir: str) -> pxt.Query:
        """The equivalent query over the table this query's model resolves to under catalog_dir."""
        tbl = self.model_cls._bind(catalog_dir)
        subst: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
        for col_name in tbl.columns():
            subst[ColumnRefByName(col_name)] = getattr(tbl, col_name)
        return self._substituted(tbl._tbl_path, subst)

    def _substituted(self, path: catalog.TablePath, subst: exprs.ExprDict[exprs.Expr]) -> pxt.Query:
        """A plain Query over `path`, with this query's clauses rewritten by `subst`."""
        # a similarity expression names its indexed column and the table version holding the index, neither of
        # which a substitution by column name reaches
        declared_path = self._from_clause.tbls[0]
        for sim in {s.id: s for e in self._component_exprs() for s in e.subexprs(exprs.SimilarityExpr)}.values():
            assert sim.qcol_id is not None
            if sim.qcol_id.tbl_id == path.tbl_id:
                continue  # already indexed against this path
            col_name = declared_path.get_column_md(sim.qcol_id).name
            assert col_name is not None
            new_md = path.get_column_md_by_name(col_name)
            if new_md is None:
                raise excs.RequestError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'Table {path.tbl_name()!r} has no column {col_name!r}, which a similarity() call references.',
                )
            subst[sim] = exprs.SimilarityExpr(
                sim.components[0].copy().substitute(subst),
                idx_name=sim.idx_name,
                qcol_id=new_md.qcolid,
                table_version_key=catalog.TableVersionKey(new_md.qcolid.tbl_id, new_md.col_effective_version),
            )

        def rebound(e: exprs.Expr) -> exprs.Expr:
            return e.copy().substitute(subst)

        return pxt.Query(
            from_clause=FromClause(tbls=[path]),
            select_list=None if self.select_list is None else [(rebound(e), n) for e, n in self.select_list],
            where_clause=None if self.where_clause is None else rebound(self.where_clause),
            group_by_clause=None if self.group_by_clause is None else [rebound(e) for e in self.group_by_clause],
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=None
            if self.order_by_clause is None
            else [(rebound(e), asc) for e, asc in self.order_by_clause],
            limit=None if self.limit_val is None else rebound(self.limit_val),
            offset=None if self.offset_val is None else rebound(self.offset_val),
            sample_clause=None
            if self.sample_clause is None
            else dataclasses.replace(
                self.sample_clause, stratify_exprs=[rebound(e) for e in self.sample_clause.stratify_exprs]
            ),
        )

    def join(self, *args: Any, **kwargs: Any) -> pxt.Query:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'join(): a query over model `{self.model_cls.__name__}` cannot be joined; '
            'join the tables the models are bound to instead.',
        )

    def _unbound(self, op: str) -> excs.RequestError:
        return excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'{op}: this query is declared over model `{self.model_cls.__name__}`, which is not bound to a '
            'table; create the tables first, or call the operation on a bound model.',
        )

    # Query's execution surface, which needs the table this query does not have
    def collect(self) -> Any:
        raise self._unbound('collect()')

    def _collect(self, args: dict[str, Any] | None = None, *, media_as_urls: bool = False) -> Any:
        raise self._unbound('collect()')

    async def _acollect(self, args: dict[str, Any] | None = None) -> Any:
        raise self._unbound('collect()')

    def cursor(self) -> Any:
        raise self._unbound('cursor()')

    def show(self, n: int = 20) -> Any:
        raise self._unbound('show()')

    def head(self, n: int = 10) -> Any:
        raise self._unbound('head()')

    def tail(self, n: int = 10) -> Any:
        raise self._unbound('tail()')

    def count(self) -> int:
        raise self._unbound('count()')

    def describe(self) -> None:
        raise self._unbound('describe()')

    def update(self, value_spec: dict[str, Any], cascade: bool = True) -> Any:
        raise self._unbound('update()')

    def recompute_columns(self, *columns: Any, errors_only: bool = False, cascade: bool = True) -> Any:
        raise self._unbound('recompute_columns()')

    def delete(self) -> Any:
        raise self._unbound('delete()')

    def to_coco_dataset(self) -> Any:
        raise self._unbound('to_coco_dataset()')

    def to_pytorch_dataset(self, image_format: str = 'pt') -> Any:
        raise self._unbound('to_pytorch_dataset()')


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
    known_idxs: dict[str, IndexDeclaration]

    # Names that are produced by the base query or iterator; these cannot be redefined in the model.
    reserved_cols: dict[str, Literal['base query', 'iterator']]

    # The scope in which the class body is defined; used to evaluate stringized type annotations (see
    # set_col_type). Populated from the defining frame in TableModelMeta.__prepare__.
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
        super().__setitem__(name, ColumnRefByName(name, col_type))

    def _check_reserved(self, name: str) -> None:
        if name in self.reserved_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{name!r} is already defined by the {self.reserved_cols[name]}; it cannot be redeclared.',
            )

    def set_col_value(self, name: str, value: Any) -> None:
        self._check_reserved(name)
        if isinstance(value, (EmbeddingIndex, BtreeIndex)):
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
            # Under from __future__ import annotations (PEP 563) -- and mandatory on Python 3.14+, where
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
            # We previously processed this column via set_col_value(). Sanity check the type.
            if col_type_from_spec(self.known_cols[name]) != type_:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'Conflicting type annotation for column {name!r}.'
                )
            return
        # Bare annotation (col: SomeType): record the spec and make the name referenceable in the body.
        self.known_cols[name] = {'type': type_}  # type: ignore[typeddict-item]
        super().__setitem__(name, exprs.ColumnRefByName(name, type_))


def _validate_model_declaration(cls_name: str, namespace: _ModelNamespace) -> None:
    """Validate a model's declarations against each other, once its class body has run."""
    if len(namespace.known_cols) == 0 and namespace.table_spec['base'] is None:
        raise excs.RequestError(excs.ErrorCode.INVALID_SCHEMA, 'Empty table schema not allowed.')

    # A table with default indexes enabled is not allowed to have explicit B-tree indexes.
    if namespace.table_spec['has_default_idxs']:
        btree_idx_names = [name for name, idx in namespace.known_idxs.items() if isinstance(idx, BtreeIndex)]
        if len(btree_idx_names) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'model `{cls_name}`: cannot combine has_default_idxs=True with explicitly declared B-tree '
                f'index(es) {btree_idx_names}; eligible columns are indexed automatically.',
            )


class TableModelMeta(type):
    """
    Metaclass that collects annotated column definitions and other table metadata from a class body.
    """

    __table_spec__: TableSpec
    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, IndexDeclaration]
    __bound_table__: Table | None

    _catalog_dir: str | None
    _table_path: catalog.TableMdPath | None

    @classmethod
    def __prepare__(  # type: ignore[override]
        mcs,  # noqa: N804  # Neither mypy nor ruff seems to understand metaclasses.
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
                base.validate(display_name)
                base_model = base.model_cls
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
            # (see _ModelNamespace.set_col_type) can be evaluated. sys._getframe(1) is the frame executing
            # the class ...: statement (__build_class__ is a C function and creates no frame).
            caller = sys._getframe(1)

            # On Python 3.14+, annotations are not evaluated eagerly (PEP 649), so the model's column annotations
            # would be dropped and body references to them would raise NameError *before* we ever reach
            # __new__. from __future__ import annotations restores the eager (stringized) behavior the model
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

            if base is not None and base.select_list is not None:
                # Make the select list's named columns referenceable in the body.
                for expr, col_name in base.select_list:
                    if col_name is None:
                        continue
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

        _validate_model_declaration(cls_name, namespace)

        # "normalize" the namespace to a plain dict; at this point, we're done with the special namespace treatment
        namespace_dict = dict(namespace)
        namespace_dict['__table_spec__'] = namespace.table_spec
        namespace_dict['__columns__'] = namespace.known_cols
        namespace_dict['__indexes__'] = namespace.known_idxs
        namespace_dict['__bound_table__'] = None
        namespace_dict['_catalog_dir'] = None
        namespace_dict['_table_path'] = None

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
                    spec['type'], nullable_default=True, allow_builtin_types=False
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
                return getattr(ModelQuery.for_model(cls), item)
            else:
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
        # leave what it declares untouched, and it can be inspected any number of times
        columns: dict[str, ColumnSpec] = {}
        for col_name, col_spec in cls.__columns__.items():
            copied = col_spec.copy()
            if 'value' in copied:
                copied['value'] = copied['value'].copy()
            columns[col_name] = copied
        iterator, cols, idxs = prepare_model(
            handle, columns, spec['display_name'], spec['iterator'], base, cls.__indexes__
        )

        md: TableVersionMd
        base_md: list[TableVersionMd] = []
        if base is None:
            cols_by_name = {col.name: col for col in cols if col.name is not None}
            assert all(isinstance(idx_spec.indexed_column, str) for idx_spec in idxs)  # indexed cols identified by name
            resolved_idxs = [
                catalog.IndexSpec(
                    indexed_column=cols_by_name[cast(str, spec_.indexed_column)], idx_name=spec_.idx_name, idx=spec_.idx
                )
                for spec_ in idxs
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
                is_data_versioned=True,
                additional_idxs=resolved_idxs,
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

        _MODEL_BY_DECLARED_TBL_ID[tbl_id] = cls
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
