from __future__ import annotations

import abc
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Set, Tuple, Type, Union, overload
from uuid import UUID

import pandas as pd
import sqlalchemy as sql

import pixeltable
import pixeltable.catalog as catalog
import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.metadata.schema as schema
import pixeltable.type_system as ts

from .column import Column
from .globals import _ROWID_COLUMN_NAME, UpdateStatus, is_system_column_name, is_valid_identifier
from .schema_object import SchemaObject
from .table_version import TableVersion
from .table_version_path import TableVersionPath

_logger = logging.getLogger('pixeltable')

class Table(SchemaObject):
    """Base class for table objects (base tables, views, snapshots)."""

    __PREDEF_SYMBOLS: Optional[set[str]] = None

    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version_path: TableVersionPath):
        super().__init__(id, name, dir_id)
        self._is_dropped = False
        self._tbl_version_path = tbl_version_path
        from pixeltable.func import QueryTemplateFunction
        self._queries: dict[str, QueryTemplateFunction] = {}

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        super()._move(new_name, new_dir_id)
        with env.Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self._id})

    def version(self) -> int:
        """Return the version of this table. Used by tests to ascertain version changes."""
        return self._tbl_version.version

    @property
    def _tbl_version(self) -> TableVersion:
        """Return TableVersion for just this table."""
        return self._tbl_version_path.tbl_version

    def __hash__(self) -> int:
        return hash(self._tbl_version.id)

    def _check_is_dropped(self) -> None:
        if self._is_dropped:
            raise excs.Error(f'{self._display_name()} {self._name} has been dropped')

    # Returns `True` if the given name is a predefined attribute of the `Table` class, such as `where` or `group_by`
    # (as opposed to an instance attribute such as a column name or query name).
    @classmethod
    def _is_system_attr(cls, name: str) -> bool:
        if cls.__PREDEF_SYMBOLS is None:
            cls.__PREDEF_SYMBOLS = set(dir(catalog.InsertableTable))
        return name in cls.__PREDEF_SYMBOLS

    def __getattr__(
            self, name: str
    ) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.func.QueryTemplateFunction']:
        """Return a ColumnRef or QueryTemplateFunction for the given name.
        """
        if name in self._queries:
            return self._queries[name]
        return getattr(self._tbl_version_path, name)

    def __getitem__(
            self, index: object
    ) -> Union[
        'pixeltable.func.QueryTemplateFunction', 'pixeltable.exprs.ColumnRef', 'pixeltable.DataFrame'
    ]:
        """Return a ColumnRef or QueryTemplateFunction for the given name, or a DataFrame for the given slice.
        """
        if isinstance(index, str) and index in self._queries:
            return self._queries[index]
        return self._tbl_version_path.__getitem__(index)

    def list_views(self, *, recursive: bool = True) -> list[str]:
        """
        Returns a list of all views and snapshots of this `Table`.

        Args:
            recursive: If `False`, returns only the immediate successor views of this `Table`. If `True`, returns
                all sub-views (including views of views, etc.)
        """
        return [t._path for t in self._get_views(recursive=recursive)]

    def _get_views(self, *, recursive: bool = True) -> list['Table']:
        dependents = catalog.Catalog.get().tbl_dependents[self._id]
        if recursive:
            return dependents + [t for view in dependents for t in view._get_views(recursive=True)]
        else:
            return dependents

    def _df(self) -> 'pixeltable.dataframe.DataFrame':
        """Return a DataFrame for this table.
        """
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path)

    def select(self, *items: Any, **named_items: Any) -> 'pixeltable.DataFrame':
        """Return a [`DataFrame`][pixeltable.DataFrame] for this table."""
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).select(*items, **named_items)

    def where(self, pred: 'exprs.Expr') -> 'pixeltable.DataFrame':
        """Return a [`DataFrame`][pixeltable.DataFrame] for this table."""
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).where(pred)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pixeltable.DataFrame':
        """Return a [`DataFrame`][pixeltable.DataFrame] for this table."""
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).order_by(*items, asc=asc)

    def group_by(self, *items: 'exprs.Expr') -> 'pixeltable.DataFrame':
        """Return a [`DataFrame`][pixeltable.DataFrame] for this table."""
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).group_by(*items)

    def limit(self, n: int) -> 'pixeltable.DataFrame':
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).limit(n)

    def collect(self) -> 'pixeltable.dataframe.DataFrameResultSet':
        """Return rows from this table."""
        return self._df().collect()

    def show(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':
        """Return rows from this table.
        """
        return self._df().show(*args, **kwargs)

    def head(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':
        """Return the first n rows inserted into this table."""
        return self._df().head(*args, **kwargs)

    def tail(
            self, *args, **kwargs
    ) -> 'pixeltable.dataframe.DataFrameResultSet':
        """Return the last n rows inserted into this table."""
        return self._df().tail(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table."""
        return self._df().count()

    def column_names(self) -> list[str]:
        """Return the names of the columns in this table."""
        return [c.name for c in self._tbl_version_path.columns()]

    def column_types(self) -> dict[str, ts.ColumnType]:
        """Return the names of the columns in this table."""
        return {c.name: c.col_type for c in self._tbl_version_path.columns()}

    def query_names(self) -> list[str]:
        """Return the names of the registered queries for this table."""
        return list(self._queries.keys())

    @property
    def base(self) -> Optional['Table']:
        """
        The base table of this `Table`. If this table is a view, returns the `Table`
        from which it was derived. Otherwise, returns `None`.
        """
        if self._tbl_version_path.base is None:
            return None
        base_id = self._tbl_version_path.base.tbl_version.id
        return catalog.Catalog.get().tbls[base_id]

    @property
    def comment(self) -> str:
        return self._tbl_version.comment

    @comment.setter
    def comment(self, new_comment: Optional[str]):
        self._tbl_version.set_comment(new_comment)

    @property
    def num_retained_versions(self):
        return self._tbl_version.num_retained_versions

    @num_retained_versions.setter
    def num_retained_versions(self, new_num_retained_versions: int):
        self._tbl_version.set_num_retained_versions(new_num_retained_versions)

    def _description(self) -> pd.DataFrame:
        cols = self._tbl_version_path.columns()
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
        """
        Print the table schema.
        """
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
        return f'{self._display_name()} \'{self._name}\'\n{comment}{description_str}'

    def _repr_html_(self) -> str:
        return self._description_html()._repr_html_()

    def _drop(self) -> None:
        self._check_is_dropped()
        self._tbl_version.drop()
        self._is_dropped = True
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
        return DataFrame(self._tbl_version_path).to_pytorch_dataset(image_format=image_format)

    def to_coco_dataset(self) -> Path:
        """Return the path to a COCO json file for this table.
            See DataFrame.to_coco_dataset()
        """
        from pixeltable.dataframe import DataFrame
        return DataFrame(self._tbl_version_path).to_coco_dataset()

    def __setitem__(self, col_name: str, spec: Union[ts.ColumnType, exprs.Expr]) -> None:
        """
        Adds a column to the table. This is an alternate syntax for `add_column()`; the meaning of

        >>> tbl['new_col'] = IntType()

        is exactly equivalent to

        >>> tbl.add_column(new_col=IntType())

        For details, see the documentation for [`add_column()`][pixeltable.catalog.Table.add_column].
        """
        if not isinstance(col_name, str):
            raise excs.Error(f'Column name must be a string, got {type(col_name)}')
        if not isinstance(spec, (ts.ColumnType, exprs.Expr)):
            raise excs.Error(f'Column spec must be a ColumnType or an Expr, got {type(spec)}')
        self.add_column(**{col_name: spec})

    def add_column(
            self,
            *,
            type: Optional[ts.ColumnType] = None,
            stored: Optional[bool] = None,
            print_stats: bool = False,
            **kwargs: Union[ts.ColumnType, exprs.Expr, Callable]
    ) -> UpdateStatus:
        """
        Adds a column to the table.

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
            raise excs.Error(
                f'add_column() requires exactly one keyword argument of the form "column-name=type|value-expression"; '
                f'got {len(kwargs)} instead ({", ".join(list(kwargs.keys()))})'
            )
        col_name, spec = next(iter(kwargs.items()))
        if not is_valid_identifier(col_name):
            raise excs.Error(f'Invalid column name: {col_name!r}')
        if self._is_system_attr(col_name):
            raise excs.Error(f'{col_name!r} is a keyword in Pixeltable; please use a different column name')
        if isinstance(spec, (ts.ColumnType, exprs.Expr)) and type is not None:
            raise excs.Error(f'add_column(): keyword argument "type" is redundant')

        col_schema: dict[str, Any] = {}
        if isinstance(spec, ts.ColumnType):
            col_schema['type'] = spec
        else:
            col_schema['value'] = spec
        if type is not None:
            col_schema['type'] = type
        if stored is not None:
            col_schema['stored'] = stored

        new_col = self._create_columns({col_name: col_schema})[0]
        self._verify_column(new_col, self.column_names(), self.query_names())
        return self._tbl_version.add_column(new_col, print_stats=print_stats)

    @classmethod
    def _validate_column_spec(cls, name: str, spec: dict[str, Any]) -> None:
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
    def _create_columns(cls, schema: dict[str, Any]) -> list[Column]:
        """Construct list of Columns, given schema"""
        columns: list[Column] = []
        for name, spec in schema.items():
            col_type: Optional[ts.ColumnType] = None
            value_expr: Optional[exprs.Expr] = None
            primary_key: Optional[bool] = None
            stored = True

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
                stored = spec.get('stored', True)
                primary_key = spec.get('primary_key')

            column = Column(
                name, col_type=col_type, computed_with=value_expr, stored=stored, is_pk=primary_key)
            columns.append(column)
        return columns

    @classmethod
    def _verify_column(
            cls, col: Column, existing_column_names: Set[str], existing_query_names: Optional[Set[str]] = None
    ) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        if is_system_column_name(col.name):
            raise excs.Error(f'Column name {col.name!r} is reserved')
        if not is_valid_identifier(col.name):
            raise excs.Error(f"Invalid column name: {col.name!r}")
        if col.name in existing_column_names:
            raise excs.Error(f'Duplicate column name: {col.name!r}')
        if existing_query_names is not None and col.name in existing_query_names:
            raise excs.Error(f'Column name conflicts with a registered query: {col.name!r}')
        if col.stored is False and not (col.is_computed and col.col_type.is_image_type()):
            raise excs.Error(f'Column {col.name!r}: stored={col.stored} only applies to computed image columns')
        if col.stored is False and col.has_window_fn_call():
            raise excs.Error((
                f'Column {col.name!r}: stored={col.stored} is not valid for image columns computed with a streaming '
                f'function'))

    @classmethod
    def _verify_schema(cls, schema: list[Column]) -> None:
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

        if name not in self._tbl_version.cols_by_name:
            raise excs.Error(f'Unknown column: {name}')
        col = self._tbl_version.cols_by_name[name]

        dependent_user_cols = [c for c in col.dependent_cols if c.name is not None]
        if len(dependent_user_cols) > 0:
            raise excs.Error(
                f'Cannot drop column `{name}` because the following columns depend on it:\n'
                f'{", ".join(c.name for c in dependent_user_cols)}'
            )

        # See if this column has a dependent store. We need to look through all stores in all
        # (transitive) views of this table.
        dependent_stores = [
            (view, store)
            for view in [self] + self._get_views(recursive=True)
            for store in view._tbl_version.external_stores.values()
            if col in store.get_local_columns()
        ]
        if len(dependent_stores) > 0:
            dependent_store_names = [
                store.name if view._id == self._id else f'{store.name} (in view `{view._name}`)'
                for view, store in dependent_stores
            ]
            raise excs.Error(
                f'Cannot drop column `{name}` because the following external stores depend on it:\n'
                f'{", ".join(dependent_store_names)}'
            )

        self._tbl_version.drop_column(col)

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
        self._tbl_version.rename_column(old_name, new_name)

    def add_embedding_index(
            self, col_name: str, *, idx_name: Optional[str] = None,
            string_embed: Optional[pixeltable.Function] = None, image_embed: Optional[pixeltable.Function] = None,
            metric: str = 'cosine'
    ) -> None:
        """Add an index to the table.
        Args:
            col_name: name of column to index
            idx_name: name of index, which needs to be unique for the table; if not provided, a name will be generated
            string_embed: function to embed text; required if the column is a text column
            image_embed: function to embed images; required if the column is an image column
            metric: distance metric to use for the index; one of 'cosine', 'ip', 'l2'; default is 'cosine'

        Raises:
            Error: If an index with that name already exists for the table or if the column does not exist.

        Examples:
            Add an index to the ``img`` column:

            >>> tbl.add_embedding_index('img', image_embed=...)

            Add another index to the ``img`` column, using the inner product as the distance metric,
            and with a specific name; ``string_embed`` is also specified in order to search with text:

            >>> tbl.add_embedding_index(
                'img', idx_name='clip_idx', image_embed=..., string_embed=..., metric='ip')
        """
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot add an index to a snapshot')
        self._check_is_dropped()
        col = self._tbl_version_path.get_column(col_name, include_bases=True)
        if col is None:
            raise excs.Error(f'Column {col_name} unknown')
        if idx_name is not None and idx_name in self._tbl_version.idxs_by_name:
            raise excs.Error(f'Duplicate index name: {idx_name}')
        from pixeltable.index import EmbeddingIndex

        # create the EmbeddingIndex instance to verify args
        idx = EmbeddingIndex(col, metric=metric, string_embed=string_embed, image_embed=image_embed)
        status = self._tbl_version.add_index(col, idx_name=idx_name, idx=idx)
        # TODO: how to deal with exceptions here? drop the index and raise?

    def drop_embedding_index(self, *, column_name: Optional[str] = None, idx_name: Optional[str] = None) -> None:
        """Drop an embedding index from the table.

        Args:
            column_name: The name of the column whose embedding index to drop. Invalid if the column has multiple
                embedding indices.
            idx_name: The name of the index to drop.

        Raises:
            Error: If the index does not exist.

        Examples:
            Drop embedding index on the ``img`` column:

            >>> tbl.drop_embedding_index(column_name='img')
        """
        self._drop_index(column_name=column_name, idx_name=idx_name, _idx_class=index.EmbeddingIndex)

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
        self._drop_index(column_name=column_name, idx_name=idx_name)

    def _drop_index(
            self, *, column_name: Optional[str] = None, idx_name: Optional[str] = None,
            _idx_class: Optional[Type[index.IndexBase]] = None
    ) -> None:
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot drop an index from a snapshot')
        self._check_is_dropped()
        if (column_name is None) == (idx_name is None):
            raise excs.Error("Exactly one of 'column_name' or 'idx_name' must be provided")

        if idx_name is not None:
            if idx_name not in self._tbl_version.idxs_by_name:
                raise excs.Error(f'Index {idx_name!r} does not exist')
            idx_id = self._tbl_version.idxs_by_name[idx_name].id
        else:
            col = self._tbl_version_path.get_column(column_name, include_bases=True)
            if col is None:
                raise excs.Error(f'Column {column_name!r} unknown')
            if col.tbl.id != self._tbl_version.id:
                raise excs.Error(
                    f'Column {column_name!r}: cannot drop index from column that belongs to base ({col.tbl.name}!r)')
            idx_info = [info for info in self._tbl_version.idxs_by_name.values() if info.col.id == col.id]
            if _idx_class is not None:
                idx_info = [info for info in idx_info if isinstance(info.idx, _idx_class)]
            if len(idx_info) == 0:
                raise excs.Error(f'Column {column_name!r} does not have an index')
            if len(idx_info) > 1:
                raise excs.Error(f"Column {column_name!r} has multiple indices; specify 'idx_name' instead")
            idx_id = idx_info[0].id
        self._tbl_version.drop_index(idx_id)

    @overload
    def insert(
            self, rows: Iterable[dict[str, Any]], /, *, print_stats: bool = False, fail_on_exception: bool = True
    ) -> UpdateStatus: ...

    @overload
    def insert(self, *, print_stats: bool = False, fail_on_exception: bool = True, **kwargs: Any) -> UpdateStatus: ...

    @abc.abstractmethod
    def insert(
            self, rows: Optional[Iterable[dict[str, Any]]] = None, /, *, print_stats: bool = False,
            fail_on_exception: bool = True, **kwargs: Any
    ) -> UpdateStatus:
        """Inserts rows into this table. There are two mutually exclusive call patterns:

        To insert multiple rows at a time:
        ``insert(rows: Iterable[dict[str, Any]], /, *, print_stats: bool = False, fail_on_exception: bool = True)``

        To insert just a single row, you can use the more convenient syntax:
        ``insert(*, print_stats: bool = False, fail_on_exception: bool = True, **kwargs: Any)``

        Args:
            rows: (if inserting multiple rows) A list of rows to insert, each of which is a dictionary mapping column
                names to values.
            kwargs: (if inserting a single row) Keyword-argument pairs representing column names and values.
            print_stats: If ``True``, print statistics about the cost of computed columns.
            fail_on_exception:
                Determines how exceptions in computed columns and invalid media files (e.g., corrupt images)
                are handled.
                If ``False``, store error information (accessible as column properties 'errortype' and 'errormsg')
                for those cases, but continue inserting rows.
                If ``True``, raise an exception that aborts the insert.

        Returns:
            execution status

        Raises:
            Error: if a row does not match the table schema or contains values for computed columns

        Examples:
            Insert two rows into a table with three int columns ``a``, ``b``, and ``c``. Column ``c`` is nullable.

            >>> tbl.insert([{'a': 1, 'b': 1, 'c': 1}, {'a': 2, 'b': 2}])

            Insert a single row into a table with three int columns ``a``, ``b``, and ``c``.

            >>> tbl.insert(a=1, b=1, c=1)
        """
        raise NotImplementedError

    def update(
            self, value_spec: dict[str, Any], where: Optional['pixeltable.exprs.Expr'] = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            value_spec: a dictionary mapping column names to literal values or Pixeltable expressions.
            where: a predicate to filter rows to update.
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
        self._check_is_dropped()
        return self._tbl_version.update(value_spec, where, cascade)

    def batch_update(
            self, rows: Iterable[dict[str, Any]], cascade: bool = True,
            if_not_exists: Literal['error', 'ignore', 'insert'] = 'error'
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            rows: an Iterable of dictionaries containing values for the updated columns plus values for the primary key
                  columns.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
            if_not_exists: Specifies the behavior if a row to update does not exist:

                - `'error'`: Raise an error.
                - `'ignore'`: Skip the row silently.
                - `'insert'`: Insert the row.

        Examples:
            Update the `name` and `age` columns for the rows with ids 1 and 2 (assuming `id` is the primary key).
            If either row does not exist, this raises an error:

            >>> tbl.update([{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 40}])

            Update the `name` and `age` columns for the row with `id` 1 (assuming `id` is the primary key) and insert
            the row with new `id` 3 (assuming this key does not exist):

            >>> tbl.update(
                [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 3, 'name': 'Bob', 'age': 40}],
                if_not_exists='insert')
        """
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot update a snapshot')
        self._check_is_dropped()
        rows = list(rows)

        row_updates: list[dict[Column, exprs.Expr]] = []
        pk_col_names = set(c.name for c in self._tbl_version.primary_key_columns())

        # pseudo-column _rowid: contains the rowid of the row to update and can be used instead of the primary key
        has_rowid = _ROWID_COLUMN_NAME in rows[0]
        rowids: list[Tuple[int, ...]] = []
        if len(pk_col_names) == 0 and not has_rowid:
            raise excs.Error('Table must have primary key for batch update')

        for row_spec in rows:
            col_vals = self._tbl_version._validate_update_spec(row_spec, allow_pk=not has_rowid, allow_exprs=False)
            if has_rowid:
                # we expect the _rowid column to be present for each row
                assert _ROWID_COLUMN_NAME in row_spec
                rowids.append(row_spec[_ROWID_COLUMN_NAME])
            else:
                col_names = set(col.name for col in col_vals.keys())
                if any(pk_col_name not in col_names for pk_col_name in pk_col_names):
                    missing_cols = pk_col_names - set(col.name for col in col_vals.keys())
                    raise excs.Error(f'Primary key columns ({", ".join(missing_cols)}) missing in {row_spec}')
            row_updates.append(col_vals)
        return self._tbl_version.batch_update(
            row_updates, rowids, error_if_not_exists=if_not_exists == 'error',
            insert_if_not_exists=if_not_exists == 'insert', cascade=cascade)

    def delete(self, where: Optional['pixeltable.exprs.Expr'] = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """
        raise NotImplementedError

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot revert a snapshot')
        self._check_is_dropped()
        self._tbl_version.revert()

    @overload
    def query(self, py_fn: Callable) -> 'pixeltable.func.QueryTemplateFunction': ...

    @overload
    def query(
            self, *, param_types: Optional[list[ts.ColumnType]] = None
    ) -> Callable[[Callable], 'pixeltable.func.QueryTemplateFunction']: ...

    def query(self, *args: Any, **kwargs: Any) -> Any:
        def make_query_template(
                py_fn: Callable, param_types: Optional[list[ts.ColumnType]]
        ) -> 'pixeltable.func.QueryTemplateFunction':
            if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
                # this is a named function in a module
                function_path = f'{py_fn.__module__}.{py_fn.__qualname__}'
            else:
                function_path = None
            query_name = py_fn.__name__
            if query_name in self.column_names():
                raise excs.Error(f'Query name {query_name!r} conflicts with existing column')
            if query_name in self._queries:
                raise excs.Error(f'Duplicate query name: {query_name!r}')
            import pixeltable.func as func
            query_fn = func.QueryTemplateFunction.create(
                py_fn, param_types=param_types, path=function_path, name=query_name)
            self._queries[query_name] = query_fn
            return query_fn

            # TODO: verify that the inferred return type matches that of the template
            # TODO: verify that the signature doesn't contain batched parameters

        if len(args) == 1:
            assert len(kwargs) == 0 and callable(args[0])
            return make_query_template(args[0], None)
        else:
            assert len(args) == 0 and len(kwargs) == 1 and 'param_types' in kwargs
            return lambda py_fn: make_query_template(py_fn, kwargs['param_types'])

    @property
    def external_stores(self) -> list[str]:
        return list(self._tbl_version.external_stores.keys())

    def _link_external_store(self, store: 'pixeltable.io.ExternalStore') -> None:
        """
        Links the specified `ExternalStore` to this table.
        """
        if self._tbl_version.is_snapshot:
            raise excs.Error(f'Table `{self._name}` is a snapshot, so it cannot be linked to an external store.')
        self._check_is_dropped()
        if store.name in self.external_stores:
            raise excs.Error(f'Table `{self._name}` already has an external store with that name: {store.name}')
        _logger.info(f'Linking external store `{store.name}` to table `{self._name}`')
        self._tbl_version.link_external_store(store)
        print(f'Linked external store `{store.name}` to table `{self._name}`.')

    def unlink_external_stores(
            self,
            stores: Optional[str | list[str]] = None,
            *,
            delete_external_data: bool = False,
            ignore_errors: bool = False
    ) -> None:
        """
        Unlinks this table's external stores.

        Args:
            stores: If specified, will unlink only the specified named store or list of stores. If not specified,
                will unlink all of this table's external stores.
            ignore_errors (bool): If `True`, no exception will be thrown if a specified store is not linked
                to this table.
            delete_external_data (bool): If `True`, then the external data store will also be deleted. WARNING: This
                is a destructive operation that will delete data outside Pixeltable, and cannot be undone.
        """
        self._check_is_dropped()
        all_stores = self.external_stores

        if stores is None:
            stores = all_stores
        elif isinstance(stores, str):
            stores = [stores]

        # Validation
        if not ignore_errors:
            for store in stores:
                if store not in all_stores:
                    raise excs.Error(f'Table `{self._name}` has no external store with that name: {store}')

        for store in stores:
            self._tbl_version.unlink_external_store(store, delete_external_data=delete_external_data)
            print(f'Unlinked external store from table `{self._name}`: {store}')

    def sync(
            self,
            stores: Optional[str | list[str]] = None,
            *,
            export_data: bool = True,
            import_data: bool = True
    ) -> 'pixeltable.io.SyncStatus':
        """
        Synchronizes this table with its linked external stores.

        Args:
            stores: If specified, will synchronize only the specified named store or list of stores. If not specified,
                will synchronize all of this table's external stores.
            export_data: If `True`, data from this table will be exported to the external stores during synchronization.
            import_data: If `True`, data from the external stores will be imported to this table during synchronization.
        """
        self._check_is_dropped()
        all_stores = self.external_stores

        if stores is None:
            stores = all_stores
        elif isinstance(stores, str):
            stores = [stores]

        for store in stores:
            if store not in all_stores:
                raise excs.Error(f'Table `{self._name}` has no external store with that name: {store}')

        from pixeltable.io import SyncStatus

        sync_status = SyncStatus.empty()
        for store in stores:
            store_obj = self._tbl_version.external_stores[store]
            store_sync_status = store_obj.sync(self, export_data=export_data, import_data=import_data)
            sync_status = sync_status.combine(store_sync_status)

        return sync_status

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + self.column_names() + self.query_names()

    def _ipython_key_completions_(self) -> list[str]:
        return self.column_names() + self.query_names()
