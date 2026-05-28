from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from pixeltable import exceptions as excs, exprs, type_system as ts
from pixeltable.types import ColumnSpec, EmbeddingIndexSpec

from .catalog import retry_loop
from .globals import is_valid_identifier
from .table import Table


class _PlaceholderColumnRef(exprs.Expr):
    """A placeholder column reference used in TableModel definitions,
    which gets substituted with an actual ColumnRef during Table creation.
    """

    def __init__(self, name: str):
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


Column = _PlaceholderFactory()


@dataclass
class TableSpec:
    name: str
    primary_key: str | list[str] | None = None
    comment: str | None = None
    custom_metadata: dict[str, Any] | None = None
    media_validation: Literal['on_read', 'on_write'] = 'on_write'

    def __getattr__(self, key: str) -> _PlaceholderColumnRef:
        if not isinstance(key, str) or not is_valid_identifier(key):
            raise AttributeError(f'Invalid column name: {key}')
        return _PlaceholderColumnRef(key)


class TableModelMetaclass(type):
    """Metaclass that collects annotated column definitions from a class body.

    Processes the class namespace to build an ordered mapping of column names to
    _ColumnDescriptor instances. Annotations become typed columns; bare Expr
    assignments become computed columns.
    """

    __columns__: dict[str, ColumnSpec]
    __indexes__: dict[str, EmbeddingIndexSpec]
    __table_name__: str
    __table_spec__: TableSpec

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
        indexes: dict[str, EmbeddingIndexSpec] = {}
        for attr_name, (annotation, value) in decls.items():
            if value is None:
                assert annotation is not None
                columns[attr_name] = ColumnSpec(type=annotation)
            elif isinstance(value, TableSpec):
                # TODO: This is not a full implementation of TableSpec; it's just to illustrate the proposal.
                #     I'll defer that until/if we decide to use that pattern. (But it won't be hard to implement.)
                namespace['__table_name__'] = value.name
                namespace['__table_spec__'] = value
            elif isinstance(value, EmbeddingIndexSpec):
                indexes[attr_name] = value
            elif isinstance(value, dict):
                spec = value
                if annotation is not None:
                    if spec['type'] is None:
                        # Fill the annotation into the ColumnSpec if it's missing
                        spec = spec.copy()
                        spec['type'] = annotation
                    elif spec['type'] != annotation:
                        raise excs.RequestError(
                            excs.ErrorCode.INVALID_SCHEMA,
                            f'Type annotation for column {attr_name!r} conflicts with type in ColumnSpec',
                        )
                columns[attr_name] = spec  # type: ignore[assignment]
            else:
                expr = exprs.Expr.from_object(value)
                columns[attr_name] = ColumnSpec(type=annotation, value=expr)

        namespace['__columns__'] = columns
        namespace['__indexes__'] = indexes
        return super().__new__(mcs, name, bases, namespace)

    def create(cls, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error') -> Table:
        import pixeltable as pxt

        columns: dict[str, ColumnSpec] = cls.__columns__
        indexes: dict[str, EmbeddingIndexSpec] = cls.__indexes__

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


class TableModel(metaclass=TableModelMetaclass):
    """
    Base class for declarative Pixeltable table models.
    """
