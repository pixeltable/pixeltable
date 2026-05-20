from __future__ import annotations

from typing import Any, Literal, NamedTuple

from pixeltable import exceptions as excs, exprs, type_system as ts
from pixeltable.types import ColumnSpec

from .globals import is_valid_identifier
from .table import Table


class _PlaceholderColumnMetaclass(type):
    """Metaclass for creating placeholder column objects that can be used in TableModel definitions.

    This allows users to reference column names in computed expressions without needing to define them
    a priori as actual columns.
    """

    def __getattr__(cls, key: str) -> exprs.Variable:
        # The placeholder reference is given by a variable with column type `InvalidType(nullable=False)`,
        # the universal subtype.
        if not isinstance(key, str) or not is_valid_identifier(key):
            raise AttributeError(f'Invalid column name: {key}')
        return exprs.Variable(name=key, col_type=ts.InvalidType(nullable=False))

    def __hasattr__(cls, key: str) -> bool:
        return isinstance(key, str) and is_valid_identifier(key)


class Column(metaclass=_PlaceholderColumnMetaclass): ...


class TableModelMetaclass(type):
    """Metaclass that collects annotated column definitions from a class body.

    Processes the class namespace to build an ordered mapping of column names to
    _ColumnDescriptor instances. Annotations become typed columns; bare Expr
    assignments become computed columns.
    """

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> TableModelMetaclass:
        annotations = namespace.get('__annotations__', {})

        # name -> (annotation, namespace_value)
        column_decls: dict[str, tuple[Any, Any]] = {}
        for attr_name, annotation in annotations.items():
            if attr_name.startswith('_'):
                continue
            value = namespace.get(attr_name)
            column_decls[attr_name] = (annotation, value)

        for attr_name, value in namespace.items():
            if attr_name.startswith('_') or attr_name in column_decls:
                continue
            column_decls[attr_name] = (None, value)

        columns: dict[str, ColumnSpec] = {}
        for attr_name, (annotation, value) in column_decls.items():
            if value is None:
                assert annotation is not None
                columns[attr_name] = ColumnSpec(type=annotation)
            elif isinstance(value, dict):
                spec = value
                if annotation is not None:
                    if spec.type is None:
                        # Fill the annotation into the ColumnSpec if it's missing
                        spec = spec.copy()
                        spec.type = annotation
                    elif spec.type != annotation:
                        raise excs.RequestError(
                            excs.ErrorCode.INVALID_SCHEMA,
                            f'Type annotation for column {attr_name!r} conflicts with type in ColumnSpec',
                        )
                columns[attr_name] = spec
            elif isinstance(value, exprs.Expr):
                columns[attr_name] = ColumnSpec(type=annotation, value=value)
            else:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Value for column {attr_name!r} must be a `ColumnSpec` or a computed expression, but has type `{type(value).__name__}`',
                )

        namespace['__columns__'] = columns
        return super().__new__(mcs, name, bases, namespace)

    def create(cls, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error') -> Table:
        import pixeltable as pxt

        columns: dict[str, ColumnSpec] = cls.__columns__

        # Create the table with non-computed columns
        initial_schema = {col_name: col_spec for col_name, col_spec in columns.items() if col_spec.get('value') is None}
        tbl = pxt.create_table(cls.__table_name__, initial_schema, if_exists=if_exists)

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
                vars = list(realized_expr.subexprs(exprs.Variable))
                if len(vars) > 0:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_SCHEMA,
                        f'Computed column {col_name!r} references undefined column(s): {vars}',
                    )
                tbl.add_computed_column(**{col_name: realized_expr})
                subst_dict[getattr(Column, col_name)] = getattr(tbl, col_name)

        return tbl


class TableModel(metaclass=TableModelMetaclass):
    """Base class for declarative Pixeltable table schemas.

    Usage::

        class MyTable(TableSchema):
            text: pxt.String
            image: pxt.Image
            # computed columns can be added via assignment with an Expr


        t = pxt.create_table('my_table', MyTable.to_schema())
    """
