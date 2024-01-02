from __future__ import annotations

import dataclasses
import logging
from typing import Optional, List, Dict, Any, Union, Tuple, Set, Callable
from uuid import UUID
import json

import sqlalchemy as sql

from .globals import is_valid_identifier, is_system_column_name
from .table import Table
from .column import Column
from .table_version import TableVersion
from ..env import Env
from ..metadata import schema
from pixeltable import exceptions as exc
import pixeltable.type_system as ts
import pixeltable.exprs as exprs


_logger = logging.getLogger('pixeltable')

class MutableTable(Table):
    """Base class for tables that allow mutations, ie, InsertableTable and View"""

    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int = 0
        # TODO: change to num_computed_columns (the number of computed slots isn't really meaningful to the user)
        num_computed_values: int = 0
        num_excs: int = 0
        updated_cols: List[str] = dataclasses.field(default_factory=list)
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    def __init__(self, id: UUID, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(id, dir_id, tbl_version.name, tbl_version)

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

    def add_column(self, col: Column, print_stats: bool = False) -> MutableTable.UpdateStatus:
        """Adds a column to the table.

        Args:
            col: The column to add.

        Returns:
            execution status

        Raises:
            Error: If the column name is invalid or already exists.

        Examples:
            Add an int column with ``None`` values:

            >>> tbl.add_column(Column('new_col', IntType()))

            For a table with int column ``x``, add a column that is the factorial of ``x``. Note that the names of
            the parameters of the ``computed_with`` Callable must correspond to existing column names (the column
            values are then passed as arguments to the Callable):

            >>> tbl.add_column(Column('factorial', IntType(), computed_with=lambda x: math.factorial(x)))

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees (note that in this case, the column type is inferred from the ``computed_with`` expression):

            >>> tbl.add_column(Column('rotated', computed_with=tkbl.frame.rotate(90)))
            'added ...'
        """
        self._check_is_dropped()
        self._verify_column(col, self.column_names())
        return self.tbl_version.add_column(col, print_stats=print_stats)

    @classmethod
    def _validate_column_spec(cls, name: str, spec: Dict[str, Any]) -> None:
        """Check integrity of user-supplied Column spec

        We unfortunately can't use something like jsonschema for validation, because this isn't strictly a JSON schema
        (on account of containing Python Callables or Exprs).
        """
        assert isinstance(spec, dict)
        valid_keys = {'type', 'value', 'stored', 'indexed'}
        has_type = False
        for k in spec.keys():
            if k not in valid_keys:
                raise exc.Error(f'Column {name}: invalid key {k!r}')

        if 'type' in spec:
            has_type = True
            if not isinstance(spec['type'], ts.ColumnType):
                raise exc.Error(f'Column {name}: type must be a ColumnType, got {spec["type"]}')

        if 'value' in spec:
            value_spec = spec['value']
            value_expr = exprs.Expr.from_object(value_spec)
            if value_expr is None:
                # needs to be a Callable
                if not isinstance(value_spec, Callable):
                    raise exc.Error(
                        f'Column {name}: value needs to be either a Pixeltable expression or a Callable, '
                        f'but it is a {type(value_spec)}')
                if 'type' not in spec:
                    raise exc.Error(f'Column {name}: type is required if value is a Callable')
            else:
                has_type = True
                if 'type' in spec:
                    raise exc.Error(f'Column {name}: type is redundant if value is a Pixeltable expression')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise exc.Error(f'Column {name}: stored must be a bool, got {spec["stored"]}')
        if 'indexed' in spec and not isinstance(spec['indexed'], bool):
            raise exc.Error(f'Column {name}: indexed must be a bool, got {spec["indexed"]}')
        if not has_type:
            raise exc.Error(f'Column {name}: type is required')

    @classmethod
    def _create_columns(cls, schema: Dict[str, Any]) -> List[Column]:
        """Construct list of Columns, given schema"""
        columns: List[Column] = []
        for name, spec in schema.items():
            col_type: Optional[ts.ColumnType] = None
            value_expr: Optional[exprs.Expr] = None
            stored: Optional[bool] = None
            indexed: Optional[bool] = None
            primary_key: Optional[bool] = None

            if isinstance(spec, ts.ColumnType):
                col_type = spec
            elif isinstance(spec, exprs.Expr):
                value_expr = spec
            elif isinstance(spec, dict):
                cls._validate_column_spec(name, spec)
                col_type = spec.get('type')
                value_expr = spec.get('value')
                stored = spec.get('stored')
                indexed = spec.get('indexed')
                primary_key = spec.get('primary_key')

            column = Column(
                name, col_type=col_type, computed_with=value_expr, stored=stored, indexed=indexed,
                primary_key=primary_key)
            columns.append(column)
        return columns

    @classmethod
    def _verify_column(cls, col: Column, existing_column_names: Set[str]) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        if is_system_column_name(col.name):
            raise exc.Error(f'Column name {col.name} is reserved')
        if not is_valid_identifier(col.name):
            raise exc.Error(f"Invalid column name: '{col.name}'")
        if col.name in existing_column_names:
            raise exc.Error(f'Duplicate column name: {col.name}')
        if col.stored is False and not (col.is_computed and col.col_type.is_image_type()):
            raise exc.Error(f'Column {col.name}: stored={col.stored} only applies to computed image columns')
        if col.stored is False and not (col.col_type.is_image_type() and not col.has_window_fn_call()):
            raise exc.Error(
                f'Column {col.name}: stored={col.stored} is not valid for image columns computed with a streaming function')
        if col.stored is None:
            col.stored = not (col.is_computed and col.col_type.is_image_type() and not col.has_window_fn_call())

    @classmethod
    def _verify_schema(cls, schema: List[Column]) -> None:
        """Check integrity of user-supplied schema and set defaults"""
        column_names: Set[str] = set()
        for col in schema:
            cls._verify_column(col, column_names)
            column_names.add(col.name)

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.

        Args:
            name: The name of the column to drop.

        Raises:
            Error: If the column does not exist or if it is referenced by a computed column.

        Example:
            >>> tbl.drop_column('factorial')
        """
        self._check_is_dropped()
        self.tbl_version.drop_column(name)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist or if the new name is invalid or already exists.

        Example:
            >>> tbl.rename_column('factorial', 'fac')
        """
        self._check_is_dropped()
        self.tbl_version.rename_column(old_name, new_name)

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> MutableTable.UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
        """
        from pixeltable import exprs
        update_targets: List[Tuple[Column, exprs.Expr]] = []
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise exc.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name not in self.tbl_version.cols_by_name:
                raise exc.Error(f'Column {col_name} unknown')
            col = self.tbl_version.cols_by_name[col_name]
            if col.is_computed:
                raise exc.Error(f'Column {col_name} is computed and cannot be updated')
            if col.primary_key:
                raise exc.Error(f'Column {col_name} is a primary key column and cannot be updated')
            if col.col_type.is_media_type():
                raise exc.Error(f'Column {col_name} has type image/video/audio and cannot be updated')

            # make sure that the value is compatible with the column type
            # check if this is a literal
            try:
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError:
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise exc.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.matches(value_expr.col_type):
                    raise exc.Error((
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ))
            update_targets.append((col, value_expr))

        from pixeltable.plan import Planner
        if where is not None:
            if not isinstance(where, exprs.Predicate):
                raise exc.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.tbl_version, where)
            if analysis_info.similarity_clause is not None:
                raise exc.Error('nearest() cannot be used with update()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise exc.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        return self.tbl_version.update(update_targets, where, cascade)

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        self.tbl_version.revert()

