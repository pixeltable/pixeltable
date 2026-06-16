from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .catalog import retry_loop
from .globals import MediaValidation, is_valid_identifier
from .table import Table
from .table_version import TableVersion, TableVersionKey
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

    column_spec: ColumnSpec

    def __init__(self, name: str, column_spec: ColumnSpec | None = None) -> None:
        col_type: ts.ColumnType
        if 'type' in column_spec:
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
        return f'Column.{self.name}'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('name', self.name)]

    def _equals(self, other: _PlaceholderColumnRef) -> bool:
        return self.name == other.name

    def eval(self, data_row: exprs.data_row.DataRow, row_builder: exprs.row_builder.RowBuilder) -> None:
        raise AssertionError('It should never be possible to observe a placeholder in an execution context.')


@dataclass
class _PlaceholderColumnSpec:
    column_spec: ColumnSpec


class _PlaceholderFactory:
    """
    Class for creating placeholder column objects that can be used in TableModel definitions.

    This allows users to reference column names in computed expressions without needing to define them
    a priori as actual columns.
    """

    def __getattr__(self, key: str) -> _PlaceholderColumnRef:
        if not isinstance(key, str) or not is_valid_identifier(key):
            raise AttributeError(f'Invalid column name: {key}')
        return _PlaceholderColumnRef(key)

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
        """
        Alternate usage of `Column` for defining a column with additional metadata.
        Wraps all provided arguments in a `ColumnSpec`.
        """
        column_spec: ColumnSpec = {
            'type': type,
            'value': value,
            'primary_key': primary_key,
            'stored': stored,
            'media_validation': media_validation,
            'destination': destination,
            'custom_metadata': custom_metadata,
            'comment': comment,
        }
        return _PlaceholderColumnSpec(column_spec)


Column: _PlaceholderFactory = _PlaceholderFactory()


class _ColumnCtx:
    new_tbl_handle: uuid.UUID
    known_cols: dict[str, _PlaceholderColumnRef]
    known_idxs: dict[str, EmbeddingIndex]

    def __init__(self) -> None:
        self.new_tbl_id = uuid.uuid4()
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
                        f'Column specification for {name!r} must define `type` or `value`, but not both.',
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

    def __init__(self, name: str) -> None:
        super().__init__()
        self.column_ctx = _ColumnCtx()
        # Pre-seed __annotations__ so the compiler routes bare annotations through
        # our recorder rather than a plain dict it would otherwise create.
        super().__setitem__('__annotations__', _AnnotationRecorder(self))
        super().__setitem__(name, self.column_ctx)

    def __setitem__(self, key: str, value: Any) -> None:
        print(f'Setting {key} = {value!r}')
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

    __columns__: dict[str, _PlaceholderColumnRef]
    __indexes__: dict[str, EmbeddingIndex]
    __table_name__: str
    __base_table__: str | Table | 'pxt.Query' | None
    __iterator__: func.GeneratingFunctionCall | None

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        import pixeltable as pxt

        # Remove the _ColumnCtx object; it's no longer needed, and we need to "normalize" the namespace
        column_ctx = namespace.pop(name)
        assert isinstance(column_ctx, _ColumnCtx)

        if len(bases) == 0:
            # This is the TableModel or ViewModel base class itself; no additional processing.
            return super().__new__(mcs, name, bases, namespace)

        # 1. Validate __table_name__

        if '__table_name__' not in namespace:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{bases[-1].__name__} `{name}` does not define a __table_name__.'
            )
        if not isinstance(namespace['__table_name__'], str) or not is_valid_identifier(
            namespace['__table_name__'], allow_hyphens=True
        ):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'__table_name__ for {bases[-1].__name__} `{name}` must be a valid Pixeltable identifier.',
            )

        # 2. Process declarations. We need to process bare annotations (such as `col1: int`) as well as
        #    values (such as `col2 = Column.col1 + 1` or `idx0 = EmbeddingIndex(...)`).

        namespace['__columns__'] = column_ctx.known_cols
        namespace['__indexes__'] = column_ctx.known_idxs

        # for col_name in column_ctx.known_cols:
        #     namespace.pop(col_name)
        # for idx_name in column_ctx.known_idxs:
        #     namespace.pop(idx_name)

        # 3. Validate __base_table__ and __iterator__ declarations.

        if bases[0].__name__ == 'TableModel':
            if '__base_table__' in namespace:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'__base_table__ not allowed for a TableModel; `{name}` must subclass ViewModel instead.',
                )
            if '__iterator__' in namespace:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'__iterator__ not allowed for a TableModel; `{name}` must subclass ViewModel instead.',
                )
            namespace['__base_table__'] = None
            namespace['__iterator__'] = None

        else:
            assert bases[0].__name__ == 'ViewModel'  # The only other possibility
            base_table = namespace.get('__base_table__')
            if base_table is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA, f'ViewModel `{name}` does not define a __base_table__.'
                )
            if isinstance(base_table, TableModelMetaclass) and issubclass(base_table, (TableModel, ViewModel)):
                # Base table is specified as a model; replace with the string name of that table
                namespace['__base_table__'] = base_table.__table_name__
            elif not isinstance(base_table, (str, Table, pxt.Query)):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Invalid __base_table__ for ViewModel `{name}`: must be a valid base table name, an existing '
                    'table or query, or another TableModel/ViewModel class.',
                )

            if '__iterator__' in namespace:
                iterator = namespace['__iterator__']
                if not (iterator is None or isinstance(iterator, func.GeneratingFunctionCall)):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'__iterator__ for ViewModel `{name}` must be a valid iterator reference.',
                    )
            else:
                namespace['__iterator__'] = None

        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def __prepare__(mcs, name: str, bases: tuple[type, ...], **kwargs: Any) -> dict[str, Any]:
        return _ModelNamespace(name)

    def _resolve_tbl(cls) -> Table:
        import pixeltable as pxt

        return pxt.get_table(cls.__table_name__)

    def _resolve_column(cls, col_name: str) -> exprs.ColumnRef:
        return getattr(cls._resolve_tbl(), col_name)

    def create(cls, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error') -> Table:
        import pixeltable as pxt

        tbl_media_validation = 'on_write'  # TODO: allow configuring this at the table level

        columns: dict[str, _PlaceholderColumnRef] = cls.__columns__
        indexes: dict[str, EmbeddingIndex] = cls.__indexes__

        catalog_columns: list[catalog.Column] = []
        subst_dict = exprs.ExprDict[exprs.ColumnRef]()

        tbl_id = uuid.uuid4()
        tbl_handle = TableVersionHandle(TableVersionKey(tbl_id, None))
        for name, placeholder in columns.items():
            subst_spec: ColumnSpec = placeholder.column_spec.copy()
            if 'value' in subst_spec:
                subst_spec['value'] = subst_spec['value'].substitute(subst_dict)
            catalog_col = catalog.Column.create(name, subst_spec)
            catalog_col.tbl_handle = tbl_handle
            catalog_col.id = len(subst_dict)
            catalog_columns.append(catalog_col)
            # Explicitly set perform_validation in order to avoid prematurely deferencing table properties.
            # It defaults to the table-level media_validation if not set in the ColumnSpec.
            subst_dict[placeholder] = exprs.ColumnRef(
                catalog_col, perform_validation=subst_spec.get('media_validation', tbl_media_validation) == 'on_read'
            )

        # Create the table with its non-computed columns
        # initial_schema = {col_name: col_spec for col_name, col_spec in columns.items() if col_spec.get('value') is None}

        cat = get_runtime().catalog
        tbl_path = catalog.Path.parse(cls.__table_name__)

        if issubclass(cls, ViewModel):
            base = cls.__base_table__
            iterator = cls.__iterator__
            assert isinstance(base, (str, pxt.Table, pxt.Query)), type(base)
            if isinstance(base, str):
                base = pxt.get_table(base)
                assert base is not None
            assert iterator is None or isinstance(iterator, func.GeneratingFunctionCall)
            tbl = pxt.create_view(
                tbl_name, base, additional_columns=initial_schema, iterator=iterator, if_exists=if_exists
            )
        else:
            create_fn = retry_loop(for_write=True)(
                lambda: cat._create_table(
                    tbl_path, catalog_columns, if_exists, None, None, None, MediaValidation.ON_WRITE, True, True, tbl_id
                )
            )
            cat._roll_forward_ids.clear()
            tbl_id_, _ = create_fn()
            assert tbl_id == tbl_id_
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

        return tbl

    def __getattr__(cls, item: str) -> Any:
        if item in FORWARDED_TABLE_METHODS:
            return getattr(cls._resolve_tbl(), item)
        if is_valid_identifier(item):
            return cls._resolve_column(item)
        return super().__getattribute__(item)

    @property
    def table(cls) -> Table:
        """The underlying [`Table`][pixeltable.Table] this model is bound to.

        Use this to access the full table and query API with static type information; e.g.,
        `MyModel.table.where(...).order_by(...).collect()`.
        """
        return cls._resolve_tbl()


class TableModel(metaclass=TableModelMetaclass):
    """
    Base class for declarative Pixeltable table models.
    """


class ViewModel(metaclass=TableModelMetaclass):
    """
    Base class for declarative Pixeltable view models.
    """
