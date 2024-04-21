from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union, Any, List, Dict, Optional, Callable, Set, Tuple, Iterable
from uuid import UUID

import pandas as pd
import sqlalchemy as sql

import pixeltable
import pixeltable.catalog as catalog
import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.metadata.schema as schema
import pixeltable.type_system as ts
from .column import Column
from .globals import is_valid_identifier, is_system_column_name, UpdateStatus
from .schema_object import SchemaObject
from .table_version import TableVersion
from .table_version_path import TableVersionPath

_logger = logging.getLogger('pixeltable')

class Table(SchemaObject):
    """Base class for all tabular SchemaObjects."""

    ROWID_COLUMN_NAME = '_rowid'

    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version_path: TableVersionPath):
        super().__init__(id, name, dir_id)
        self.is_dropped = False
        self.tbl_version_path = tbl_version_path

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with env.Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self._id})

    def version(self) -> int:
        """Return the version of this table. Used by tests to ascertain version changes."""
        return self.tbl_version_path.tbl_version.version

    def _tbl_version(self) -> TableVersion:
        """Return TableVersion for just this table."""
        return self.tbl_version_path.tbl_version

    def __hash__(self) -> int:
        return hash(self._tbl_version().id)

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise excs.Error(f'{self.display_name()} {self.name} has been dropped')

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        """Return a ColumnRef for the given column name.
        """
        return getattr(self.tbl_version_path, col_name)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        """Return a ColumnRef for the given column name, or a DataFrame for the given slice.
        """
        return self.tbl_version_path.__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path)

    def select(self, *items: Any, **named_items: Any) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path).select(*items, **named_items)

    def where(self, pred: 'exprs.Predicate') -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path).where(pred)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path).order_by(*items, asc=asc)

    def collect(self) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().collect()

    def show(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return rows from this table.
        """
        return self.df().show(*args, **kwargs)

    def head(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return the first n rows inserted into this table."""
        return self.df().head(*args, **kwargs)

    def tail(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        """Return the last n rows inserted into this table."""
        return self.df().tail(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table."""
        return self.df().count()

    def column_names(self) -> List[str]:
        """Return the names of the columns in this table."""
        return [c.name for c in self.tbl_version_path.columns()]

    def column_types(self) -> Dict[str, ts.ColumnType]:
        """Return the names of the columns in this table."""
        return {c.name: c.col_type for c in self.tbl_version_path.columns()}

    @property
    def comment(self) -> str:
        return self.tbl_version.comment

    @comment.setter
    def comment(self, new_comment: Optional[str]):
        self.tbl_version.set_comment(new_comment)

    @property
    def num_retained_versions(self):
        return self.tbl_version.num_retained_versions

    @num_retained_versions.setter
    def num_retained_versions(self, new_num_retained_versions: int):
        self.tbl_version.set_num_retained_versions(new_num_retained_versions)

    def _description(self) -> pd.DataFrame:
        cols = self.tbl_version_path.columns()
        df = pd.DataFrame({
            'Column Name': [c.name for c in cols],
            'Type': [str(c.col_type) for c in cols],
            'Computed With': [c.value_expr.display_str(inline=False) if c.value_expr is not None else '' for c in cols],
        })
        return df

    def _description_html(self) -> pd.DataFrame:
        pd_df = self._description()
        # white-space: pre-wrap: print \n as newline
        # th: center-align headings
        return pd_df.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) \
            .hide(axis='index')

    def describe(self) -> None:
        try:
            __IPYTHON__
            from IPython.display import display
            display(self._description_html())
        except NameError:
            print(self.__repr__())

    # TODO: Display comments in _repr_html()
    def __repr__(self) -> str:
        description_str = self._description().to_string(index=False)
        if self.comment is None:
            comment = ''
        else:
            comment = f'{self.comment}\n'
        return f'{self.display_name()} \'{self._name}\'\n{comment}{description_str}'

    def _repr_html_(self) -> str:
        return self._description_html()._repr_html_()

    def _drop(self) -> None:
        self._check_is_dropped()
        self.tbl_version_path.tbl_version.drop()
        self.is_dropped = True
        # update catalog
        cat = catalog.Catalog.get()
        del cat.tbls[self._id]

    # TODO Factor this out into a separate module.
    # The return type is unresolvable, but torch can't be imported since it's an optional dependency.
    def to_pytorch_dataset(self, image_format : str = 'pt') -> 'torch.utils.data.IterableDataset':
        """Return a PyTorch Dataset for this table.
            See DataFrame.to_pytorch_dataset()
        """
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path).to_pytorch_dataset(image_format=image_format)

    def to_coco_dataset(self) -> Path:
        """Return the path to a COCO json file for this table.
            See DataFrame.to_coco_dataset()
        """
        from pixeltable.dataframe import DataFrame
        return DataFrame(self.tbl_version_path).to_coco_dataset()

    def __setitem__(self, column_name: str, value: Union[ts.ColumnType, exprs.Expr, Callable, dict]) -> None:
        """Adds a column to the table
        Args:
            column_name: the name of the new column
            value: column type or value expression or column specification dictionary:
                column type: a Pixeltable column type (if the table already contains rows, it must be nullable)
                value expression: a Pixeltable expression that computes the column values
                column specification: a dictionary with possible keys 'type', 'value', 'stored'
        Examples:
            Add an int column with ``None`` values:

            >>> tbl['new_col'] = IntType(nullable=True)

            For a table with int column ``int_col``, add a column that is the factorial of ``int_col``. The names of
            the parameters of the Callable must correspond to existing column names (the column values are then passed
            as arguments to the Callable). In this case, the return type cannot be inferred and needs to be specified
            explicitly:

            >>> tbl['factorial'] = {'value': lambda int_col: math.factorial(int_col), 'type': IntType()}

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees. In this case, the column type is inferred from the expression. Also, the column is not stored
            (by default, computed image columns are not stored but recomputed on demand):

            >>> tbl['rotated'] = tbl.frame.rotate(90)

            Do the same, but now the column is stored:

            >>> tbl['rotated'] = {'value': tbl.frame.rotate(90), 'stored': True}
        """
        if not isinstance(column_name, str):
            raise excs.Error(f'Column name must be a string, got {type(column_name)}')
        if not is_valid_identifier(column_name):
            raise excs.Error(f'Invalid column name: {column_name!r}')

        new_col = self._create_columns({column_name: value})[0]
        self._verify_column(new_col, self.column_names())
        return self.tbl_version_path.tbl_version.add_column(new_col)

    def add_column(
            self, *,
            type: Optional[ts.ColumnType] = None, stored: Optional[bool] = None, print_stats: bool = False,
            **kwargs: Any
    ) -> UpdateStatus:
        """Adds a column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form ``column-name=type|value-expression``.
            type: The type of the column. Only valid and required if ``value-expression`` is a Callable.
            stored: Whether the column is materialized and stored or computed on demand. Only valid for image columns.
            print_stats: If ``True``, print execution metrics.

        Returns:
            execution status

        Raises:
            Error: If the column name is invalid or already exists.

        Examples:
            Add an int column with ``None`` values:

            >>> tbl.add_column(new_col=IntType())

            Alternatively, this can also be expressed as:

            >>> tbl['new_col'] = IntType()

            For a table with int column ``int_col``, add a column that is the factorial of ``int_col``. The names of
            the parameters of the Callable must correspond to existing column names (the column values are then passed
            as arguments to the Callable). In this case, the column type needs to be specified explicitly:

            >>> tbl.add_column(factorial=lambda int_col: math.factorial(int_col), type=IntType())

            Alternatively, this can also be expressed as:

            >>> tbl['factorial'] = {'value': lambda int_col: math.factorial(int_col), 'type': IntType()}

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees. In this case, the column type is inferred from the expression. Also, the column is not stored
            (by default, computed image columns are not stored but recomputed on demand):

            >>> tbl.add_column(rotated=tbl.frame.rotate(90))

            Alternatively, this can also be expressed as:

            >>> tbl['rotated'] = tbl.frame.rotate(90)

            Do the same, but now the column is stored:

            >>> tbl.add_column(rotated=tbl.frame.rotate(90), stored=True)

            Alternatively, this can also be expressed as:

            >>> tbl['rotated'] = {'value': tbl.frame.rotate(90), 'stored': True}
        """
        self._check_is_dropped()
        # verify kwargs and construct column schema dict
        if len(kwargs) != 1:
            raise excs.Error((
                f'add_column() requires exactly one keyword argument of the form "column-name=type|value-expression", '
                f'got {len(kwargs)} instead ({", ".join(list(kwargs.keys()))})'
            ))
        col_name, spec = next(iter(kwargs.items()))
        col_schema: Dict[str, Any] = {}
        if isinstance(spec, ts.ColumnType):
            if type is not None:
                raise excs.Error(f'add_column(): keyword argument "type" is redundant')
            col_schema['type'] = spec
        else:
            if isinstance(spec, exprs.Expr) and type is not None:
                raise excs.Error(f'add_column(): keyword argument "type" is redundant')
            col_schema['value'] = spec
        if type is not None:
            col_schema['type'] = type
        if stored is not None:
            col_schema['stored'] = stored

        new_col = self._create_columns({col_name: col_schema})[0]
        self._verify_column(new_col, self.column_names())
        return self.tbl_version_path.tbl_version.add_column(new_col, print_stats=print_stats)

    @classmethod
    def _validate_column_spec(cls, name: str, spec: Dict[str, Any]) -> None:
        """Check integrity of user-supplied Column spec

        We unfortunately can't use something like jsonschema for validation, because this isn't strictly a JSON schema
        (on account of containing Python Callables or Exprs).
        """
        assert isinstance(spec, dict)
        valid_keys = {'type', 'value', 'stored'}
        has_type = False
        for k in spec.keys():
            if k not in valid_keys:
                raise excs.Error(f'Column {name}: invalid key {k!r}')

        if 'type' in spec:
            has_type = True
            if not isinstance(spec['type'], ts.ColumnType):
                raise excs.Error(f'Column {name}: "type" must be a ColumnType, got {spec["type"]}')

        if 'value' in spec:
            value_spec = spec['value']
            value_expr = exprs.Expr.from_object(value_spec)
            if value_expr is None:
                # needs to be a Callable
                if not isinstance(value_spec, Callable):
                    raise excs.Error(
                        f'Column {name}: value needs to be either a Pixeltable expression or a Callable, '
                        f'but it is a {type(value_spec)}')
                if 'type' not in spec:
                    raise excs.Error(f'Column {name}: "type" is required if value is a Callable')
            else:
                has_type = True
                if 'type' in spec:
                    raise excs.Error(f'Column {name}: "type" is redundant if value is a Pixeltable expression')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise excs.Error(f'Column {name}: "stored" must be a bool, got {spec["stored"]}')
        if not has_type:
            raise excs.Error(f'Column {name}: "type" is required')

    @classmethod
    def _create_columns(cls, schema: Dict[str, Any]) -> List[Column]:
        """Construct list of Columns, given schema"""
        columns: List[Column] = []
        for name, spec in schema.items():
            col_type: Optional[ts.ColumnType] = None
            value_expr: Optional[exprs.Expr] = None
            stored: Optional[bool] = None
            primary_key: Optional[bool] = None

            if isinstance(spec, ts.ColumnType):
                # TODO: create copy
                col_type = spec
            elif isinstance(spec, exprs.Expr):
                # create copy so we can modify it
                value_expr = spec.copy()
            elif isinstance(spec, Callable):
                raise excs.Error((
                    f'Column {name} computed with a Callable: specify using a dictionary with '
                    f'the "value" and "type" keys (e.g., "{name}": {{"value": <Callable>, "type": IntType()}})'
                ))
            elif isinstance(spec, dict):
                cls._validate_column_spec(name, spec)
                col_type = spec.get('type')
                value_expr = spec.get('value')
                if value_expr is not None and isinstance(value_expr, exprs.Expr):
                    # create copy so we can modify it
                    value_expr = value_expr.copy()
                stored = spec.get('stored')
                primary_key = spec.get('primary_key')

            column = Column(
                name, col_type=col_type, computed_with=value_expr, stored=stored, is_pk=primary_key)
            columns.append(column)
        return columns

    @classmethod
    def _verify_column(cls, col: Column, existing_column_names: Set[str]) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        if is_system_column_name(col.name):
            raise excs.Error(f'Column name {col.name} is reserved')
        if not is_valid_identifier(col.name):
            raise excs.Error(f"Invalid column name: '{col.name}'")
        if col.name in existing_column_names:
            raise excs.Error(f'Duplicate column name: {col.name}')
        if col.stored is False and not (col.is_computed and col.col_type.is_image_type()):
            raise excs.Error(f'Column {col.name}: stored={col.stored} only applies to computed image columns')
        if col.stored is False and not (col.col_type.is_image_type() and not col.has_window_fn_call()):
            raise excs.Error((
                f'Column {col.name}: stored={col.stored} is not valid for image columns computed with a streaming '
                f'function'))
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

        Examples:
            Drop column ``factorial``:

            >>> tbl.drop_column('factorial')
        """
        self._check_is_dropped()
        self.tbl_version_path.tbl_version.drop_column(name)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist or if the new name is invalid or already exists.

        Examples:
            Rename column ``factorial`` to ``fac``:

            >>> tbl.rename_column('factorial', 'fac')
        """
        self._check_is_dropped()
        self.tbl_version_path.tbl_version.rename_column(old_name, new_name)

    def add_embedding_index(
            self, col_name: str, *, idx_name: Optional[str] = None,
            text_embed: Optional[pixeltable.Function] = None, img_embed: Optional[pixeltable.Function] = None
    ) -> None:
        """Add an index to the table.
        Args:
            col_name: name of column to index
            idx_name: name of index, which needs to be unique for the table; if not provided, a name will be generated
            idx_type: type of index (one of 'embedding')

        Raises:
            Error: If an index with that name already exists for the table or if the column does not exist.

        Examples:
            Add an index to the ``img`` column:

            >>> tbl.add_embedding_index('img', text_embed=...)

            Add another index to the ``img`` column, with a specific name:

            >>> tbl.add_embedding_index('img', idx_name='clip_idx', text_embed=...)
        """
        if self.tbl_version_path.is_snapshot():
            raise excs.Error('Cannot add an index to a snapshot')
        self._check_is_dropped()
        col = self.tbl_version_path.get_column(col_name, include_bases=True)
        if col is None:
            raise excs.Error(f'Column {col_name} unknown')
        if idx_name is not None and idx_name in self.tbl_version_path.tbl_version.idxs_by_name:
            raise excs.Error(f'Duplicate index name: {idx_name}')
        from pixeltable.index import EmbeddingIndex
        # create the EmbeddingIndex instance to verify args
        idx = EmbeddingIndex(col, text_embed=text_embed, img_embed=img_embed)
        status = self.tbl_version_path.tbl_version.add_index(col, idx_name=idx_name, idx=idx)
        # TODO: how to deal with exceptions here? drop the index and raise?

    def drop_index(self, *, column_name: Optional[str] = None, idx_name: Optional[str] = None) -> None:
        """Drop an index from the table.

        Args:
            column_name: The name of the column whose index to drop. Invalid if the column has multiple indices.
            idx_name: The name of the index to drop.

        Raises:
            Error: If the index does not exist.

        Examples:
            Drop index on the ``img`` column:

            >>> tbl.drop_index(column_name='img')
        """
        if self.tbl_version_path.is_snapshot():
            raise excs.Error('Cannot drop an index from a snapshot')
        self._check_is_dropped()
        if (column_name is None) == (idx_name is None):
            raise excs.Error('Exactly one of column_name or idx_name must be provided')
        tbl_version = self.tbl_version_path.tbl_version

        if idx_name is not None:
            if idx_name not in tbl_version.idxs_by_name:
                raise excs.Error(f'Index {idx_name} does not exist')
            idx_id = tbl_version.idxs_by_name[idx_name].id
        else:
            col = self.tbl_version_path.get_column(column_name, include_bases=True)
            if col is None:
                raise excs.Error(f'Column {column_name} unknown')
            if col.tbl.id != tbl_version.id:
                raise excs.Error(
                    f'Column {column_name}: cannot drop index from column that belongs to base ({col.tbl.name})')
            idx_ids = [info.id for info in tbl_version.idxs_by_name.values() if info.col.id == col.id]
            if len(idx_ids) == 0:
                raise excs.Error(f'Column {column_name} does not have an index')
            if len(idx_ids) > 1:
                raise excs.Error(f'Column {column_name} has multiple indices; specify idx_name instead')
            idx_id = idx_ids[0]
        self.tbl_version_path.tbl_version.drop_index(idx_id)

    def update(
            self, value_spec: dict[str, Any], where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            value_spec: a dictionary mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.

        Examples:
            Set column `int_col` to 1 for all rows:

            >>> tbl.update({'int_col': 1})

            Set column `int_col` to 1 for all rows where `int_col` is 0:

            >>> tbl.update({'int_col': 1}, where=tbl.int_col == 0)

            Set `int_col` to the value of `other_int_col` + 1:

            >>> tbl.update({'int_col': tbl.other_int_col + 1})

            Increment `int_col` by 1 for all rows where `int_col` is 0:

            >>> tbl.update({'int_col': tbl.int_col + 1}, where=tbl.int_col == 0)
        """
        if self.tbl_version_path.is_snapshot():
            raise excs.Error('Cannot update a snapshot')
        self._check_is_dropped()

        update_spec = self._validate_update_spec(value_spec, allow_pk=False, allow_exprs=True)
        from pixeltable.plan import Planner
        if where is not None:
            if not isinstance(where, exprs.Predicate):
                raise excs.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.tbl_version_path, where)
            if analysis_info.similarity_clause is not None:
                raise excs.Error('nearest() cannot be used with update()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        return self.tbl_version_path.tbl_version.update(update_spec, where, cascade)

    def batch_update(self, rows: Iterable[dict[str, Any]], cascade: bool = True) -> UpdateStatus:
        """Update rows in this table.

        Args:
            rows: an Iterable of dictionaries containing values for the updated columns plus values for the primary key
                  columns.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.

        Examples:
            Update the 'name' and 'age' columns for the rows with ids 1 and 2 (assuming 'id' is the primary key):

            >>> tbl.update([{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 40}])
        """
        if self.tbl_version_path.is_snapshot():
            raise excs.Error('Cannot update a snapshot')
        self._check_is_dropped()

        row_updates: List[Dict[Column, exprs.Expr]] = []
        pk_col_names = set(c.name for c in self.tbl_version_path.tbl_version.primary_key_columns())

        # pseudo-column _rowid: contains the rowid of the row to update and can be used instead of the primary key
        has_rowid = self.ROWID_COLUMN_NAME in rows[0]
        rowids: list[Tuple[int, ...]] = []
        if len(pk_col_names) == 0 and not has_rowid:
            raise excs.Error('Table must have primary key for batch update')

        for row_spec in rows:
            col_vals = self._validate_update_spec(row_spec, allow_pk=not has_rowid, allow_exprs=False)
            if has_rowid:
                # we expect the _rowid column to be present for each row
                assert self.ROWID_COLUMN_NAME in row_spec
                rowids.append(row_spec[self.ROWID_COLUMN_NAME])
            else:
                col_names = set(col.name for col in col_vals.keys())
                if any(pk_col_name not in col_names for pk_col_name in pk_col_names):
                    missing_cols = pk_col_names - set(col.name for col in col_vals.keys())
                    raise excs.Error(f'Primary key columns ({", ".join(missing_cols)}) missing in {row_spec}')
            row_updates.append(col_vals)
        return self.tbl_version_path.tbl_version.batch_update(row_updates, rowids, cascade)

    def _validate_update_spec(
            self, value_spec: dict[str, Any], allow_pk: bool, allow_exprs: bool
    ) -> dict[Column, 'pixeltable.exprs.Expr']:
        from pixeltable import exprs
        update_targets: dict[Column, exprs.Expr] = {}
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name == self.ROWID_COLUMN_NAME:
                # ignore pseudo-column _rowid
                continue
            col = self.tbl_version_path.get_column(col_name, include_bases=False)
            if col is None:
                # TODO: return more informative error if this is trying to update a base column
                raise excs.Error(f'Column {col_name} unknown')
            if col.is_computed:
                raise excs.Error(f'Column {col_name} is computed and cannot be updated')
            if col.is_pk and not allow_pk:
                raise excs.Error(f'Column {col_name} is a primary key column and cannot be updated')
            if col.col_type.is_media_type():
                raise excs.Error(f'Column {col_name} has type image/video/audio/document and cannot be updated')

            # make sure that the value is compatible with the column type
            try:
                # check if this is a literal
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError:
                if not allow_exprs:
                    raise excs.Error(
                        f'Column {col_name}: value {val!r} is not a valid literal for this column '
                        f'(expected {col.col_type})')
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.matches(value_expr.col_type):
                    raise excs.Error((
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ))
            update_targets[col] = value_expr

        return update_targets


    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        if self.tbl_version_path.is_snapshot():
            raise excs.Error('Cannot revert a snapshot')
        self._check_is_dropped()
        self.tbl_version_path.tbl_version.revert()
