from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.types import ColumnSpec

from .catalog import retry_loop
from .globals import is_valid_identifier
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
        'pull',
        'push',
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
assert all(hasattr(Table, method) for method in FORWARDED_TABLE_METHODS)


class _PlaceholderColumnRef(exprs.Expr):
    """
    A placeholder column reference used in TableModel definitions,
    which gets substituted with an actual ColumnRef during Table creation.
    """

    def __init__(self, name: str, column_spec: ColumnSpec | None = None) -> None:
        # Placeholders have column type `InvalidType(nullable=False)`, the universal subtype.

        super().__init__(ts.InvalidType(nullable=False))
        self.name = name
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

    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, EmbeddingIndex]
    __table_name__: str
    __base_table__: str | Table | 'pxt.Query' | None
    __iterator__: func.GeneratingFunctionCall | None

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        import pixeltable as pxt

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

        annotations = namespace.get('__annotations__', {})

        # Build a dictionary of declarations: name -> (annotation, namespace_value)
        decls: dict[str, tuple[Any, Any]] = {}

        for attr_name, annotation in annotations.items():
            if attr_name.startswith('_'):
                continue
            value = namespace.get(attr_name)  # It might also have a value
            decls[attr_name] = (annotation, value)

        for attr_name, value in namespace.items():
            if attr_name.startswith('_') or attr_name in decls:
                # Skip if we already processed it in the preceding loop
                continue
            decls[attr_name] = (None, value)  # No annotation (or we'd have processed it before)

        # Now process the declarations to build column and index dictionaries.
        columns: dict[str, ColumnSpec] = {}
        indexes: dict[str, EmbeddingIndex] = {}
        for attr_name, (annotation, value) in decls.items():
            if value is None:
                assert annotation is not None
                # Pure type annotation; no value
                columns[attr_name] = ColumnSpec(type=annotation)
            elif isinstance(value, EmbeddingIndex):
                indexes[attr_name] = value
            elif isinstance(value, _PlaceholderColumnSpec):
                spec = value.column_spec
                if annotation is not None:
                    if spec.get('type') is None:
                        # Fill the annotation into the ColumnSpec if it's missing
                        spec = spec.copy()
                        spec['type'] = annotation
                    elif spec['type'] != annotation:
                        raise excs.RequestError(
                            excs.ErrorCode.INVALID_SCHEMA,
                            f'Type annotation for column {attr_name!r} '
                            'conflicts with the `type=` argument in `Column()`',
                        )
                columns[attr_name] = spec
            else:
                # Computed column expression.
                expr = exprs.Expr.from_object(value)
                columns[attr_name] = ColumnSpec(type=annotation, value=expr)

            # Remove the attribute from the namespace so that the metaclass __getattr__ handler can resolve it into
            # proper ColumnRef instances at runtime.
            namespace.pop(attr_name, None)

        namespace['__columns__'] = columns
        namespace['__indexes__'] = indexes

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

    def _resolve_tbl(cls) -> Table:
        import pixeltable as pxt

        return pxt.get_table(cls.__table_name__)

    def _resolve_column(cls, col_name: str) -> exprs.ColumnRef:
        return getattr(cls._resolve_tbl(), col_name)

    def create(cls, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error') -> Table:
        import pixeltable as pxt

        columns: dict[str, ColumnSpec] = cls.__columns__
        indexes: dict[str, EmbeddingIndex] = cls.__indexes__

        # Create the table with its non-computed columns
        initial_schema = {col_name: col_spec for col_name, col_spec in columns.items() if col_spec.get('value') is None}

        tbl: pxt.Table
        tbl_name = cls.__table_name__
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
            tbl = pxt.create_table(tbl_name, initial_schema, if_exists=if_exists)

        @retry_loop(for_write=True, write_tvps=[tbl._tbl_version_path], lock_mutable_tree=True)
        def finish_schema() -> None:
            # Now add computed columns, in declaration order, substituting placeholder references to existing columns
            # with actual references. Each time we add a new computed column, add it to the substitution dictionary
            # so that subsequent columns can reference it.

            # Initialize the substitution dictionary with all existing columns in the table. This will include the
            # non-computed columns that were just created, as well as any existing (computed or non-) visible columns
            # of this view's ancestors.
            subst_dict = {getattr(Column, col_name): getattr(tbl, col_name) for col_name in tbl.columns()}

            # Now add the new computed columns.
            for col_name, col_spec in columns.items():
                expr = col_spec.get('value')
                if expr is not None:
                    realized_expr: exprs.Expr = expr.copy().substitute(subst_dict)
                    residual_placeholders = list(realized_expr.subexprs(_PlaceholderColumnRef))
                    if len(residual_placeholders) > 0:
                        raise excs.RequestError(
                            excs.ErrorCode.INVALID_SCHEMA,
                            f'Computed column {col_name!r} references undefined column(s): {residual_placeholders}',
                        )
                    tbl.add_computed_column(
                        **{col_name: realized_expr},
                        stored=col_spec.get('stored'),
                        destination=col_spec.get('destination'),
                        custom_metadata=col_spec.get('custom_metadata'),
                        comment=col_spec.get('comment', ''),
                    )
                    subst_dict[getattr(Column, col_name)] = getattr(tbl, col_name)

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

        finish_schema()
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
