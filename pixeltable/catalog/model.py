from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.types import ColumnSpec

from .catalog import retry_loop
from .globals import is_valid_identifier
from .table import Table


class _PlaceholderColumnRef(exprs.Expr):
    """A placeholder column reference used in TableModel definitions,
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
    """Class for creating placeholder column objects that can be used in TableModel definitions.

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
        # Route each provided argument into a ColumnSpec, omitting any that were not supplied
        # (ColumnSpec is total=False, so absent keys are simply left out).
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


Column = _PlaceholderFactory()


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
    """Metaclass that collects annotated column definitions from a class body.

    Processes the class namespace to build an ordered mapping of column names to
    _ColumnDescriptor instances. Annotations become typed columns; bare Expr
    assignments become computed columns.
    """

    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, EmbeddingIndex]
    __table_name__: str

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        annotations = namespace.get('__annotations__', {})

        # name -> (annotation, namespace_value)
        decls: dict[str, tuple[Any, Any]] = {}
        for attr_name, annotation in annotations.items():
            if attr_name.startswith('_'):
                continue
            value = namespace.get(attr_name)
            decls[attr_name] = (annotation, value)

        for attr_name, value in namespace.items():
            if attr_name.startswith('_') or attr_name in decls:
                continue
            decls[attr_name] = (None, value)

        columns: dict[str, ColumnSpec] = {}
        indexes: dict[str, EmbeddingIndex] = {}
        for attr_name, (annotation, value) in decls.items():
            if value is None:
                assert annotation is not None
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
                expr = exprs.Expr.from_object(value)
                columns[attr_name] = ColumnSpec(type=annotation, value=expr)

            # Remove the attribute from the namespace so that the metaclass __getattr__ handler can resolve it into
            # proper ColumnRef instances.
            if attr_name in namespace:
                del namespace[attr_name]

        if '__table_name__' not in namespace and len(bases) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'Table model `{name}` does not define a table name.'
            )

        namespace['__columns__'] = columns
        namespace['__indexes__'] = indexes

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

        # Create the table with non-computed columns
        initial_schema = {col_name: col_spec for col_name, col_spec in columns.items() if col_spec.get('value') is None}
        tbl = pxt.create_table(cls.__table_name__, initial_schema, if_exists=if_exists)

        @retry_loop(for_write=True, write_tvps=[tbl._tbl_version_path], lock_mutable_tree=True)
        def finish_schema() -> None:
            # Now add computed columns, in declaration order, substituting placeholder references to existing columns
            # with actual references. Each time we add a new computed column, add it to the substitution dictionary
            # so that subsequent columns can reference it.
            subst_dict = {
                getattr(Column, col_name): getattr(tbl, col_name)
                for col_name, col_spec in columns.items()
                if col_spec.get('value') is None
            }

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
                    tbl.add_computed_column(**{col_name: realized_expr})
                    subst_dict[getattr(Column, col_name)] = getattr(tbl, col_name)

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
        if item in cls.__columns__:
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
