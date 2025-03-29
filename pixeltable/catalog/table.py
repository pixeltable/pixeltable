from __future__ import annotations

import abc
import builtins
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Union, overload

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip
from uuid import UUID

import pandas as pd
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.catalog as catalog
import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.metadata.schema as schema
import pixeltable.type_system as ts
from pixeltable.env import Env

from ..exprs import ColumnRef
from ..utils.description_helper import DescriptionHelper
from ..utils.filecache import FileCache
from .column import Column
from .globals import (
    _ROWID_COLUMN_NAME,
    IfExistsParam,
    IfNotExistsParam,
    MediaValidation,
    UpdateStatus,
    is_system_column_name,
    is_valid_identifier,
)
from .schema_object import SchemaObject
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath

if TYPE_CHECKING:
    import torch.utils.data

    import pixeltable.plan

_logger = logging.getLogger('pixeltable')


class Table(SchemaObject):
    """
    A handle to a table, view, or snapshot. This class is the primary interface through which table operations
    (queries, insertions, updates, etc.) are performed in Pixeltable.
    """

    # Every user-invoked operation that runs an ExecNode tree (directly or indirectly) needs to call
    # FileCache.emit_eviction_warnings() at the end of the operation.

    _is_dropped: bool
    __tbl_version_path: TableVersionPath

    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version_path: TableVersionPath):
        super().__init__(id, name, dir_id)
        self._is_dropped = False
        self.__tbl_version_path = tbl_version_path

    # @property
    # def _has_dependents(self) -> bool:
    #     """Returns True if this table has any dependent views, or snapshots."""
    #     return len(self._get_views(recursive=False)) > 0

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        self._check_is_dropped()
        super()._move(new_name, new_dir_id)
        conn = env.Env.get().conn
        stmt = sql.text(
            (
                f'UPDATE {schema.Table.__table__} '
                f'SET {schema.Table.dir_id.name} = :new_dir_id, '
                f"    {schema.Table.md.name} = jsonb_set({schema.Table.md.name}, '{{name}}', (:new_name)::jsonb) "
                f'WHERE {schema.Table.id.name} = :id'
            )
        )
        conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self._id})

    def get_metadata(self) -> dict[str, Any]:
        """
        Retrieves metadata associated with this table.

        Returns:
            A dictionary containing the metadata, in the following format:

                ```python
                {
                    'base': None,  # If this is a view or snapshot, will contain the name of its base table
                    'schema': {
                        'col1': StringType(),
                        'col2': IntType(),
                    },
                    'version': 22,
                    'schema_version': 1,
                    'comment': '',
                    'num_retained_versions': 10,
                    'is_view': False,
                    'is_snapshot': False,
                    'media_validation': 'on_write',
                }
                ```
        """
        self._check_is_dropped()
        with env.Env.get().begin_xact():
            md = super().get_metadata()
            md['base'] = self._base._path() if self._base is not None else None
            md['schema'] = self._schema
            md['version'] = self._version
            md['schema_version'] = self._tbl_version.get().schema_version
            md['comment'] = self._comment
            md['num_retained_versions'] = self._num_retained_versions
            md['media_validation'] = self._media_validation.name.lower()
            return md

    @property
    def _version(self) -> int:
        """Return the version of this table. Used by tests to ascertain version changes."""
        return self._tbl_version.get().version

    @property
    def _tbl_version(self) -> TableVersionHandle:
        """Return TableVersion for just this table."""
        return self._tbl_version_path.tbl_version

    @property
    def _tbl_version_path(self) -> TableVersionPath:
        self._check_is_dropped()
        return self.__tbl_version_path

    def __hash__(self) -> int:
        return hash(self._tbl_version.id)

    def _check_is_dropped(self) -> None:
        if self._is_dropped:
            raise excs.Error(f'{self._display_name()} {self._name} has been dropped')

    def __getattr__(self, name: str) -> 'pxt.exprs.ColumnRef':
        """Return a ColumnRef for the given name."""
        col = self._tbl_version_path.get_column(name)
        if col is None:
            raise AttributeError(f'Column {name!r} unknown')
        return ColumnRef(col)

    def __getitem__(self, name: str) -> 'pxt.exprs.ColumnRef':
        """Return a ColumnRef for the given name."""
        return getattr(self, name)

    def list_views(self, *, recursive: bool = True) -> list[str]:
        """
        Returns a list of all views and snapshots of this `Table`.

        Args:
            recursive: If `False`, returns only the immediate successor views of this `Table`. If `True`, returns
                all sub-views (including views of views, etc.)

        Returns:
            A list of view paths.
        """
        self._check_is_dropped()
        with env.Env.get().begin_xact():
            return [t._path() for t in self._get_views(recursive=recursive)]

    def _get_views(self, *, recursive: bool = True) -> list['Table']:
        cat = catalog.Catalog.get()
        view_ids = cat.get_view_ids(self._id)
        views = [cat.get_table_by_id(id) for id in view_ids]
        if recursive:
            views.extend([t for view in views for t in view._get_views(recursive=True)])
        return views

    def _df(self) -> 'pxt.dataframe.DataFrame':
        """Return a DataFrame for this table."""
        # local import: avoid circular imports
        from pixeltable.plan import FromClause

        return pxt.DataFrame(FromClause(tbls=[self._tbl_version_path]))

    def select(self, *items: Any, **named_items: Any) -> 'pxt.DataFrame':
        """Select columns or expressions from this table.

        See [`DataFrame.select`][pixeltable.DataFrame.select] for more details.
        """
        return self._df().select(*items, **named_items)

    def where(self, pred: 'exprs.Expr') -> 'pxt.DataFrame':
        """Filter rows from this table based on the expression.

        See [`DataFrame.where`][pixeltable.DataFrame.where] for more details.
        """
        return self._df().where(pred)

    def join(
        self,
        other: 'Table',
        *,
        on: Optional['exprs.Expr'] = None,
        how: 'pixeltable.plan.JoinType.LiteralType' = 'inner',
    ) -> 'pxt.DataFrame':
        """Join this table with another table."""
        return self._df().join(other, on=on, how=how)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pxt.DataFrame':
        """Order the rows of this table based on the expression.

        See [`DataFrame.order_by`][pixeltable.DataFrame.order_by] for more details.
        """
        return self._df().order_by(*items, asc=asc)

    def group_by(self, *items: 'exprs.Expr') -> 'pxt.DataFrame':
        """Group the rows of this table based on the expression.

        See [`DataFrame.group_by`][pixeltable.DataFrame.group_by] for more details.
        """
        return self._df().group_by(*items)

    def limit(self, n: int) -> 'pxt.DataFrame':
        return self._df().limit(n)

    def collect(self) -> 'pxt.dataframe.DataFrameResultSet':
        """Return rows from this table."""
        return self._df().collect()

    def show(self, *args, **kwargs) -> 'pxt.dataframe.DataFrameResultSet':
        """Return rows from this table."""
        return self._df().show(*args, **kwargs)

    def head(self, *args, **kwargs) -> 'pxt.dataframe.DataFrameResultSet':
        """Return the first n rows inserted into this table."""
        return self._df().head(*args, **kwargs)

    def tail(self, *args, **kwargs) -> 'pxt.dataframe.DataFrameResultSet':
        """Return the last n rows inserted into this table."""
        return self._df().tail(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table."""
        return self._df().count()

    @property
    def columns(self) -> list[str]:
        """Return the names of the columns in this table."""
        cols = self._tbl_version_path.columns()
        return [c.name for c in cols]

    @property
    def _schema(self) -> dict[str, ts.ColumnType]:
        """Return the schema (column names and column types) of this table."""
        return {c.name: c.col_type for c in self._tbl_version_path.columns()}

    @property
    def _base(self) -> Optional['Table']:
        """
        The base table of this `Table`. If this table is a view, returns the `Table`
        from which it was derived. Otherwise, returns `None`.
        """
        if self._tbl_version_path.base is None:
            return None
        base_id = self._tbl_version_path.base.tbl_version.id
        return catalog.Catalog.get().get_table_by_id(base_id)

    @property
    def _bases(self) -> list['Table']:
        """
        The ancestor list of bases of this table, starting with its immediate base.
        """
        bases = []
        base = self._base
        while base is not None:
            bases.append(base)
            base = base._base
        return bases

    @property
    def _comment(self) -> str:
        return self._tbl_version.get().comment

    @property
    def _num_retained_versions(self):
        return self._tbl_version.get().num_retained_versions

    @property
    def _media_validation(self) -> MediaValidation:
        return self._tbl_version.get().media_validation

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def _descriptors(self) -> DescriptionHelper:
        """
        Constructs a list of descriptors for this table that can be pretty-printed.
        """
        helper = DescriptionHelper()
        helper.append(self._title_descriptor())
        helper.append(self._col_descriptor())
        idxs = self._index_descriptor()
        if not idxs.empty:
            helper.append(idxs)
        stores = self._external_store_descriptor()
        if not stores.empty:
            helper.append(stores)
        if self._comment:
            helper.append(f'COMMENT: {self._comment}')
        return helper

    def _title_descriptor(self) -> str:
        title: str
        if self._base is None:
            title = f'Table\n{self._path()!r}'
        else:
            title = f'View\n{self._path()!r}'
            title += f'\n(of {self.__bases_to_desc()})'
        return title

    def _col_descriptor(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'Column Name': col.name,
                'Type': col.col_type._to_str(as_schema=True),
                'Computed With': col.value_expr.display_str(inline=False) if col.value_expr is not None else '',
            }
            for col in self._tbl_version_path.columns()
            if columns is None or col.name in columns
        )

    def __bases_to_desc(self) -> str:
        bases = self._bases
        assert len(bases) >= 1
        if len(bases) <= 2:
            return ', '.join(repr(b._path()) for b in bases)
        else:
            return f'{bases[0]._path()!r}, ..., {bases[-1]._path()!r}'

    def _index_descriptor(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        from pixeltable import index

        pd_rows = []
        for name, info in self._tbl_version.get().idxs_by_name.items():
            if isinstance(info.idx, index.EmbeddingIndex) and (columns is None or info.col.name in columns):
                display_embed = info.idx.string_embed if info.col.col_type.is_string_type() else info.idx.image_embed
                if info.idx.string_embed is not None and info.idx.image_embed is not None:
                    embed_str = f'{display_embed} (+1)'
                else:
                    embed_str = str(display_embed)
                row = {
                    'Index Name': name,
                    'Column': info.col.name,
                    'Metric': str(info.idx.metric.name.lower()),
                    'Embedding': embed_str,
                }
                pd_rows.append(row)
        return pd.DataFrame(pd_rows)

    def _external_store_descriptor(self) -> pd.DataFrame:
        pd_rows = []
        for name, store in self._tbl_version.get().external_stores.items():
            row = {'External Store': name, 'Type': type(store).__name__}
            pd_rows.append(row)
        return pd.DataFrame(pd_rows)

    def ensure_md_loaded(self) -> None:
        """Ensure that table metadata is loaded."""
        for col in self._tbl_version.get().cols_by_id.values():
            _ = col.value_expr

    def describe(self) -> None:
        """
        Print the table schema.
        """
        self._check_is_dropped()
        if getattr(builtins, '__IPYTHON__', False):
            from IPython.display import display

            display(self._repr_html_())
        else:
            print(repr(self))

    def _drop(self) -> None:
        self._check_is_dropped()
        self._tbl_version.get().drop()
        self._is_dropped = True

    # TODO Factor this out into a separate module.
    # The return type is unresolvable, but torch can't be imported since it's an optional dependency.
    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        """Return a PyTorch Dataset for this table.
        See DataFrame.to_pytorch_dataset()
        """
        return self._df().to_pytorch_dataset(image_format=image_format)

    def to_coco_dataset(self) -> Path:
        """Return the path to a COCO json file for this table.
        See DataFrame.to_coco_dataset()
        """
        return self._df().to_coco_dataset()

    def _column_has_dependents(self, col: Column) -> bool:
        """Returns True if the column has dependents, False otherwise."""
        assert col is not None
        assert col.name in self._schema.keys()
        if any(c.name is not None for c in col.dependent_cols):
            return True
        return any(
            col in store.get_local_columns()
            for view in [self] + self._get_views(recursive=True)
            for store in view._tbl_version.get().external_stores.values()
        )

    def _ignore_or_drop_existing_columns(self, new_col_names: list[str], if_exists: IfExistsParam) -> list[str]:
        """Check and handle existing columns in the new column specification based on the if_exists parameter.

        If `if_exists='ignore'`, returns a list of existing columns, if any, in `new_col_names`.
        """
        assert not self.get_metadata()['is_snapshot']
        existing_col_names = set(self._schema.keys())
        cols_to_ignore = []
        for new_col_name in new_col_names:
            if new_col_name in existing_col_names:
                if if_exists == IfExistsParam.ERROR:
                    raise excs.Error(f'Duplicate column name: {new_col_name!r}')
                elif if_exists == IfExistsParam.IGNORE:
                    cols_to_ignore.append(new_col_name)
                elif if_exists == IfExistsParam.REPLACE or if_exists == IfExistsParam.REPLACE_FORCE:
                    if new_col_name not in self._tbl_version.get().cols_by_name:
                        # for views, it is possible that the existing column
                        # is a base table column; in that case, we should not
                        # drop/replace that column. Continue to raise error.
                        raise excs.Error(f'Column {new_col_name!r} is a base table column. Cannot replace it.')
                    col = self._tbl_version.get().cols_by_name[new_col_name]
                    # cannot drop a column with dependents; so reject
                    # replace directive if column has dependents.
                    if self._column_has_dependents(col):
                        raise excs.Error(
                            f'Column {new_col_name!r} already exists and has dependents. Cannot {if_exists.name.lower()} it.'
                        )
                    self.drop_column(new_col_name)
                    assert new_col_name not in self._tbl_version.get().cols_by_name
        return cols_to_ignore

    def add_columns(
        self,
        schema: dict[str, Union[ts.ColumnType, builtins.type, _GenericAlias]],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        """
        Adds multiple columns to the table. The columns must be concrete (non-computed) columns; to add computed columns,
        use [`add_computed_column()`][pixeltable.catalog.Table.add_computed_column] instead.

        The format of the `schema` argument is identical to the format of the schema in a call to
        [`create_table()`][pixeltable.globals.create_table].

        Args:
            schema: A dictionary mapping column names to types.
            if_exists: Determines the behavior if a column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has no dependents.

                Note that the `if_exists` parameter is applied to all columns in the schema.
                To apply different behaviors to different columns, please use [`add_column()`][pixeltable.Table.add_column] for each column.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If any column name is invalid, or already exists and `if_exists='error'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            Add multiple columns to the table `my_table`:

            >>> tbl = pxt.get_table('my_table')
            ... schema = {
            ...     'new_col_1': pxt.Int,
            ...     'new_col_2': pxt.String,
            ... }
            ... tbl.add_columns(schema)
        """
        self._check_is_dropped()
        if self.get_metadata()['is_snapshot']:
            raise excs.Error('Cannot add column to a snapshot.')
        col_schema = {
            col_name: {'type': ts.ColumnType.normalize_type(spec, nullable_default=True, allow_builtin_types=False)}
            for col_name, spec in schema.items()
        }

        with Env.get().begin_xact():
            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                list(col_schema.keys()), IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if all columns to be added already exist and user asked to ignore
            # existing columns, there's nothing to do.
            for cname in cols_to_ignore:
                assert cname in col_schema
                del col_schema[cname]
            if len(col_schema) == 0:
                return UpdateStatus()
            new_cols = self._create_columns(col_schema)
            for new_col in new_cols:
                self._verify_column(new_col)
            status = self._tbl_version.get().add_columns(new_cols, print_stats=False, on_error='abort')
            FileCache.get().emit_eviction_warnings()
            return status

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: Union[ts.ColumnType, builtins.type, _GenericAlias, exprs.Expr],
    ) -> UpdateStatus:
        """
        Adds an ordinary (non-computed) column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form `col_name=col_type`.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has no dependents.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If the column name is invalid, or already exists and `if_exists='erorr'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            Add an int column:

            >>> tbl.add_column(new_col=pxt.Int)

            Alternatively, this can also be expressed as:

            >>> tbl.add_columns({'new_col': pxt.Int})
        """
        self._check_is_dropped()
        # verify kwargs
        if self._tbl_version.get().is_snapshot:
            raise excs.Error('Cannot add column to a snapshot.')
        # verify kwargs and construct column schema dict
        if len(kwargs) != 1:
            raise excs.Error(
                f'add_column() requires exactly one keyword argument of the form "col_name=col_type"; '
                f'got {len(kwargs)} instead ({", ".join(kwargs.keys())})'
            )
        col_type = next(iter(kwargs.values()))
        if not isinstance(col_type, (ts.ColumnType, type, _GenericAlias)):
            raise excs.Error(
                f'The argument to add_column() must be a type; did you intend to use add_computed_column() instead?'
            )
        return self.add_columns(kwargs, if_exists=if_exists)

    def add_computed_column(
        self,
        *,
        stored: Optional[bool] = None,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        if_exists: Literal['error', 'ignore', 'replace'] = 'error',
        **kwargs: exprs.Expr,
    ) -> UpdateStatus:
        """
        Adds a computed column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form `col_name=expression`.
            stored: Whether the column is materialized and stored or computed on demand.
            print_stats: If `True`, print execution metrics during evaluation.
            on_error: Determines the behavior if an error occurs while evaluating the column expression for at least one
                row.

                - `'abort'`: an exception will be raised and the column will not be added.
                - `'ignore'`: execution will continue and the column will be added. Any rows
                    with errors will have a `None` value for the column, with information about the error stored in the
                    corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, iff it has no dependents.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If the column name is invalid or already exists and `if_exists='error'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            For a table with an image column `frame`, add an image column `rotated` that rotates the image by
            90 degrees:

            >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90))

            Do the same, but now the column is unstored:

            >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90), stored=False)
        """
        self._check_is_dropped()
        if self.get_metadata()['is_snapshot']:
            raise excs.Error('Cannot add column to a snapshot.')
        if len(kwargs) != 1:
            raise excs.Error(
                f'add_computed_column() requires exactly one keyword argument of the form "column-name=type|value-expression"; '
                f'got {len(kwargs)} arguments instead ({", ".join(list(kwargs.keys()))})'
            )
        col_name, spec = next(iter(kwargs.items()))
        if not is_valid_identifier(col_name):
            raise excs.Error(f'Invalid column name: {col_name!r}')

        col_schema: dict[str, Any] = {'value': spec}
        if stored is not None:
            col_schema['stored'] = stored

        with Env.get().begin_xact():
            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                [col_name], IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if the column to add already exists and user asked to ignore
            # exiting column, there's nothing to do.
            if len(cols_to_ignore) != 0:
                assert cols_to_ignore[0] == col_name
                return UpdateStatus()

            new_col = self._create_columns({col_name: col_schema})[0]
            self._verify_column(new_col)
            status = self._tbl_version.get().add_columns([new_col], print_stats=print_stats, on_error=on_error)
            FileCache.get().emit_eviction_warnings()
            return status

    @classmethod
    def _validate_column_spec(cls, name: str, spec: dict[str, Any]) -> None:
        """Check integrity of user-supplied Column spec

        We unfortunately can't use something like jsonschema for validation, because this isn't strictly a JSON schema
        (on account of containing Python Callables or Exprs).
        """
        assert isinstance(spec, dict)
        valid_keys = {'type', 'value', 'stored', 'media_validation'}
        for k in spec.keys():
            if k not in valid_keys:
                raise excs.Error(f'Column {name}: invalid key {k!r}')

        if 'type' not in spec and 'value' not in spec:
            raise excs.Error(f"Column {name}: 'type' or 'value' must be specified")

        if 'type' in spec:
            if not isinstance(spec['type'], (ts.ColumnType, type, _GenericAlias)):
                raise excs.Error(f'Column {name}: "type" must be a type or ColumnType, got {spec["type"]}')

        if 'value' in spec:
            value_expr = exprs.Expr.from_object(spec['value'])
            if value_expr is None:
                raise excs.Error(f'Column {name}: value must be a Pixeltable expression.')
            if 'type' in spec:
                raise excs.Error(f"Column {name}: 'type' is redundant if 'value' is specified")

        if 'media_validation' in spec:
            _ = catalog.MediaValidation.validated(spec['media_validation'], f'Column {name}: media_validation')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise excs.Error(f'Column {name}: "stored" must be a bool, got {spec["stored"]}')

    @classmethod
    def _create_columns(cls, schema: dict[str, Any]) -> list[Column]:
        """Construct list of Columns, given schema"""
        columns: list[Column] = []
        for name, spec in schema.items():
            col_type: Optional[ts.ColumnType] = None
            value_expr: Optional[exprs.Expr] = None
            primary_key: Optional[bool] = None
            media_validation: Optional[catalog.MediaValidation] = None
            stored = True

            if isinstance(spec, (ts.ColumnType, type, _GenericAlias)):
                col_type = ts.ColumnType.normalize_type(spec, nullable_default=True, allow_builtin_types=False)
            elif isinstance(spec, exprs.Expr):
                # create copy so we can modify it
                value_expr = spec.copy()
                value_expr.bind_rel_paths()
            elif isinstance(spec, dict):
                cls._validate_column_spec(name, spec)
                if 'type' in spec:
                    col_type = ts.ColumnType.normalize_type(
                        spec['type'], nullable_default=True, allow_builtin_types=False
                    )
                value_expr = spec.get('value')
                if value_expr is not None and isinstance(value_expr, exprs.Expr):
                    # create copy so we can modify it
                    value_expr = value_expr.copy()
                    value_expr.bind_rel_paths()
                stored = spec.get('stored', True)
                primary_key = spec.get('primary_key')
                media_validation_str = spec.get('media_validation')
                media_validation = (
                    catalog.MediaValidation[media_validation_str.upper()] if media_validation_str is not None else None
                )
            else:
                raise excs.Error(f'Invalid value for column {name!r}')

            column = Column(
                name,
                col_type=col_type,
                computed_with=value_expr,
                stored=stored,
                is_pk=primary_key,
                media_validation=media_validation,
            )
            columns.append(column)
        return columns

    @classmethod
    def _verify_column(cls, col: Column) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        if is_system_column_name(col.name):
            raise excs.Error(f'{col.name!r} is a reserved name in Pixeltable; please choose a different column name.')
        if not is_valid_identifier(col.name):
            raise excs.Error(f'Invalid column name: {col.name!r}')
        if col.stored is False and not col.is_computed:
            raise excs.Error(f'Column {col.name!r}: stored={col.stored} only applies to computed columns')
        if col.stored is False and col.has_window_fn_call():
            raise excs.Error(
                (
                    f'Column {col.name!r}: stored={col.stored} is not valid for image columns computed with a streaming '
                    f'function'
                )
            )

    @classmethod
    def _verify_schema(cls, schema: list[Column]) -> None:
        """Check integrity of user-supplied schema and set defaults"""
        column_names: set[str] = set()
        for col in schema:
            cls._verify_column(col)
            column_names.add(col.name)

    def __check_column_name_exists(self, column_name: str, include_bases: bool = False) -> None:
        col = self._tbl_version_path.get_column(column_name, include_bases)
        if col is None:
            raise excs.Error(f'Column {column_name!r} unknown')

    def __check_column_ref_exists(self, col_ref: ColumnRef, include_bases: bool = False) -> None:
        exists = self._tbl_version_path.has_column(col_ref.col, include_bases)
        if not exists:
            raise excs.Error(f'Unknown column: {col_ref.col.qualified_name}')

    def drop_column(self, column: Union[str, ColumnRef], if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        """Drop a column from the table.

        Args:
            column: The name or reference of the column to drop.
            if_not_exists: Directive for handling a non-existent column. Must be one of the following:

                - `'error'`: raise an error if the column does not exist.
                - `'ignore'`: do nothing if the column does not exist.

        Raises:
            Error: If the column does not exist and `if_exists='error'`,
                or if it is referenced by a dependent computed column.

        Examples:
            Drop the column `col` from the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_column('col')

            Drop the column `col` from the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_column(tbl.col)

            Drop the column `col` from the table `my_table` if it exists, otherwise do nothing:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_col(tbl.col, if_not_exists='ignore')
        """
        self._check_is_dropped()
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot drop column from a snapshot.')
        col: Column = None
        _if_not_exists = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
        if isinstance(column, str):
            col = self._tbl_version_path.get_column(column, include_bases=False)
            if col is None:
                if _if_not_exists == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Column {column!r} unknown')
                assert _if_not_exists == IfNotExistsParam.IGNORE
                return
            col = self._tbl_version.get().cols_by_name[column]
        else:
            exists = self._tbl_version_path.has_column(column.col, include_bases=False)
            if not exists:
                if _if_not_exists == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Unknown column: {column.col.qualified_name}')
                assert _if_not_exists == IfNotExistsParam.IGNORE
                return
            col = column.col

        dependent_user_cols = [c for c in col.dependent_cols if c.name is not None]
        if len(dependent_user_cols) > 0:
            raise excs.Error(
                f'Cannot drop column `{col.name}` because the following columns depend on it:\n'
                f'{", ".join(c.name for c in dependent_user_cols)}'
            )

        with Env.get().begin_xact():
            # See if this column has a dependent store. We need to look through all stores in all
            # (transitive) views of this table.
            dependent_stores = [
                (view, store)
                for view in [self] + self._get_views(recursive=True)
                for store in view._tbl_version.get().external_stores.values()
                if col in store.get_local_columns()
            ]
            if len(dependent_stores) > 0:
                dependent_store_names = [
                    store.name if view._id == self._id else f'{store.name} (in view `{view._name}`)'
                    for view, store in dependent_stores
                ]
                raise excs.Error(
                    f'Cannot drop column `{col.name}` because the following external stores depend on it:\n'
                    f'{", ".join(dependent_store_names)}'
                )

            self._tbl_version.get().drop_column(col)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist, or if the new name is invalid or already exists.

        Examples:
            Rename the column `col1` to `col2` of the table `my_table`:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.rename_column('col1', 'col2')
        """
        with Env.get().begin_xact():
            self._tbl_version.get().rename_column(old_name, new_name)

    def _list_index_info_for_test(self) -> list[dict[str, Any]]:
        """
        Returns list of all the indexes on this table. Used for testing.

        Returns:
            A list of index information, each containing the index's
            id, name, and the name of the column it indexes.
        """
        assert not self._is_dropped
        index_info = []
        for idx_name, idx in self._tbl_version.get().idxs_by_name.items():
            index_info.append({'_id': idx.id, '_name': idx_name, '_column': idx.col.name})
        return index_info

    def add_embedding_index(
        self,
        column: Union[str, ColumnRef],
        *,
        idx_name: Optional[str] = None,
        embedding: Optional[pxt.Function] = None,
        string_embed: Optional[pxt.Function] = None,
        image_embed: Optional[pxt.Function] = None,
        metric: str = 'cosine',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        """
        Add an embedding index to the table. Once the index is created, it will be automatically kept up-to-date as new
        rows are inserted into the table.

        To add an embedding index, one must specify, at minimum, the column to be indexed and an embedding UDF.
        Only `String` and `Image` columns are currently supported. Here's an example that uses a
        [CLIP embedding][pixeltable.functions.huggingface.clip] to index an image column:

        >>> from pixeltable.functions.huggingface import clip
        ... embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
        ... tbl.add_embedding_index(tbl.img, embedding=embedding_fn)

        Once the index is created, similiarity lookups can be performed using the `similarity` pseudo-function.

        >>> reference_img = PIL.Image.open('my_image.jpg')
        ... sim = tbl.img.similarity(reference_img)
        ... tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

        If the embedding UDF is a multimodal embedding (supporting more than one data type), then lookups may be
        performed using any of its supported types. In our example, CLIP supports both text and images, so we can
        also search for images using a text description:

        >>> sim = tbl.img.similarity('a picture of a train')
        ... tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

        Args:
            column: The name of, or reference to, the column to be indexed; must be a `String` or `Image` column.
            idx_name: An optional name for the index. If not specified, a name such as `'idx0'` will be generated
                automatically. If specified, the name must be unique for this table.
            embedding: The UDF to use for the embedding. Must be a UDF that accepts a single argument of type `String`
                or `Image` (as appropriate for the column being indexed) and returns a fixed-size 1-dimensional
                array of floats.
            string_embed: An optional UDF to use for the string embedding component of this index.
                Can be used in conjunction with `image_embed` to construct multimodal embeddings manually, by
                specifying different embedding functions for different data types.
            image_embed: An optional UDF to use for the image embedding component of this index.
                Can be used in conjunction with `string_embed` to construct multimodal embeddings manually, by
                specifying different embedding functions for different data types.
            metric: Distance metric to use for the index; one of `'cosine'`, `'ip'`, or `'l2'`.
                The default is `'cosine'`.
            if_exists: Directive for handling an existing index with the same name. Must be one of the following:

                - `'error'`: raise an error if an index with the same name already exists.
                - `'ignore'`: do nothing if an index with the same name already exists.
                - `'replace'` or `'replace_force'`: replace the existing index with the new one.

        Raises:
            Error: If an index with the specified name already exists for the table and `if_exists='error'`, or if the specified column does not exist.

        Examples:
            Add an index to the `img` column of the table `my_table`:

            >>> from pixeltable.functions.huggingface import clip
            ... tbl = pxt.get_table('my_table')
            ... embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
            ... tbl.add_embedding_index(tbl.img, embedding=embedding_fn)

            Alternatively, the `img` column may be specified by name:

            >>> tbl.add_embedding_index('img', embedding=embedding_fn)

            Add a second index to the `img` column, using the inner product as the distance metric,
            and with a specific name:

            >>> tbl.add_embedding_index(
            ...     tbl.img,
            ...     idx_name='ip_idx',
            ...     embedding=embedding_fn,
            ...     metric='ip'
            ... )

            Add an index using separately specified string and image embeddings:

            >>> tbl.add_embedding_index(
            ...     tbl.img,
            ...     string_embed=string_embedding_fn,
            ...     image_embed=image_embedding_fn
            ... )
        """
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot add an index to a snapshot')
        col: Column
        if isinstance(column, str):
            self.__check_column_name_exists(column, include_bases=True)
            col = self._tbl_version_path.get_column(column, include_bases=True)
        else:
            self.__check_column_ref_exists(column, include_bases=True)
            col = column.col

        with Env.get().begin_xact():
            if idx_name is not None and idx_name in self._tbl_version.get().idxs_by_name:
                _if_exists = IfExistsParam.validated(if_exists, 'if_exists')
                # An index with the same name already exists.
                # Handle it according to if_exists.
                if _if_exists == IfExistsParam.ERROR:
                    raise excs.Error(f'Duplicate index name: {idx_name}')
                if not isinstance(self._tbl_version.get().idxs_by_name[idx_name].idx, index.EmbeddingIndex):
                    raise excs.Error(
                        f'Index `{idx_name}` is not an embedding index. Cannot {_if_exists.name.lower()} it.'
                    )
                if _if_exists == IfExistsParam.IGNORE:
                    return
                assert _if_exists == IfExistsParam.REPLACE or _if_exists == IfExistsParam.REPLACE_FORCE
                self.drop_index(idx_name=idx_name)
                assert idx_name not in self._tbl_version.get().idxs_by_name
            from pixeltable.index import EmbeddingIndex

            # create the EmbeddingIndex instance to verify args
            idx = EmbeddingIndex(
                col, metric=metric, embed=embedding, string_embed=string_embed, image_embed=image_embed
            )
            status = self._tbl_version.get().add_index(col, idx_name=idx_name, idx=idx)
            # TODO: how to deal with exceptions here? drop the index and raise?
            FileCache.get().emit_eviction_warnings()

    def drop_embedding_index(
        self,
        *,
        column: Union[str, ColumnRef, None] = None,
        idx_name: Optional[str] = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """
        Drop an embedding index from the table. Either a column name or an index name (but not both) must be
        specified. If a column name or reference is specified, it must be a column containing exactly one
        embedding index; otherwise the specific index name must be provided instead.

        Args:
            column: The name of, or reference to, the column from which to drop the index.
                    The column must have only one embedding index.
            idx_name: The name of the index to drop.
            if_not_exists: Directive for handling a non-existent index. Must be one of the following:

                - `'error'`: raise an error if the index does not exist.
                - `'ignore'`: do nothing if the index does not exist.

                Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
                and it does not exist, or when `column` is specified and it has no index.
                `if_not_exists` does not apply to non-exisitng column.

        Raises:
            Error: If `column` is specified, but the column does not exist, or it contains no embedding
                indices and `if_not_exists='error'`, or the column has multiple embedding indices.
            Error: If `idx_name` is specified, but the index is not an embedding index, or
                the index does not exist and `if_not_exists='error'`.

        Examples:
            Drop the embedding index on the `img` column of the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(column='img')

            Drop the embedding index on the `img` column of the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(column=tbl.img)

            Drop the embedding index `idx1` of the table `my_table` by index name:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(idx_name='idx1')

            Drop the embedding index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(idx_name='idx1', if_not_exists='ignore')
        """
        if (column is None) == (idx_name is None):
            raise excs.Error("Exactly one of 'column' or 'idx_name' must be provided")

        col: Column = None
        if idx_name is None:
            if isinstance(column, str):
                self.__check_column_name_exists(column, include_bases=True)
                col = self._tbl_version_path.get_column(column, include_bases=True)
            else:
                self.__check_column_ref_exists(column, include_bases=True)
                col = column.col
            assert col is not None

        with Env.get().begin_xact():
            self._drop_index(col=col, idx_name=idx_name, _idx_class=index.EmbeddingIndex, if_not_exists=if_not_exists)

    def drop_index(
        self,
        *,
        column: Union[str, ColumnRef, None] = None,
        idx_name: Optional[str] = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """
        Drop an index from the table. Either a column name or an index name (but not both) must be
        specified. If a column name or reference is specified, it must be a column containing exactly one index;
        otherwise the specific index name must be provided instead.

        Args:
            column: The name of, or reference to, the column from which to drop the index.
                    The column must have only one embedding index.
            idx_name: The name of the index to drop.
            if_not_exists: Directive for handling a non-existent index. Must be one of the following:

                - `'error'`: raise an error if the index does not exist.
                - `'ignore'`: do nothing if the index does not exist.

                Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
                and it does not exist, or when `column` is specified and it has no index.
                `if_not_exists` does not apply to non-exisitng column.

        Raises:
            Error: If `column` is specified, but the column does not exist, or it contains no
                indices or multiple indices.
            Error: If `idx_name` is specified, but the index does not exist.

        Examples:
            Drop the index on the `img` column of the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(column_name='img')

            Drop the index on the `img` column of the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(tbl.img)

            Drop the index `idx1` of the table `my_table` by index name:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(idx_name='idx1')

            Drop the index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(idx_name='idx1', if_not_exists='ignore')

        """
        if (column is None) == (idx_name is None):
            raise excs.Error("Exactly one of 'column' or 'idx_name' must be provided")

        col: Column = None
        if idx_name is None:
            if isinstance(column, str):
                self.__check_column_name_exists(column, include_bases=True)
                col = self._tbl_version_path.get_column(column, include_bases=True)
            else:
                self.__check_column_ref_exists(column, include_bases=True)
                col = column.col
            assert col is not None

        with Env.get().begin_xact():
            self._drop_index(col=col, idx_name=idx_name, if_not_exists=if_not_exists)

    def _drop_index(
        self,
        *,
        col: Optional[Column] = None,
        idx_name: Optional[str] = None,
        _idx_class: Optional[type[index.IndexBase]] = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        if self._tbl_version_path.is_snapshot():
            raise excs.Error('Cannot drop an index from a snapshot')
        assert (col is None) != (idx_name is None)

        if idx_name is not None:
            _if_not_exists = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
            if idx_name not in self._tbl_version.get().idxs_by_name:
                if _if_not_exists == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Index {idx_name!r} does not exist')
                assert _if_not_exists == IfNotExistsParam.IGNORE
                return
            idx_id = self._tbl_version.get().idxs_by_name[idx_name].id
        else:
            if col.tbl.id != self._tbl_version.id:
                raise excs.Error(
                    f'Column {col.name!r}: cannot drop index from column that belongs to base ({col.tbl.get().name}!r)'
                )
            idx_info = [info for info in self._tbl_version.get().idxs_by_name.values() if info.col.id == col.id]
            if _idx_class is not None:
                idx_info = [info for info in idx_info if isinstance(info.idx, _idx_class)]
            if len(idx_info) == 0:
                _if_not_exists = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
                if _if_not_exists == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Column {col.name!r} does not have an index')
                assert _if_not_exists == IfNotExistsParam.IGNORE
                return
            if len(idx_info) > 1:
                raise excs.Error(f"Column {col.name!r} has multiple indices; specify 'idx_name' instead")
            idx_id = idx_info[0].id
        self._tbl_version.get().drop_index(idx_id)

    @overload
    def insert(
        self,
        rows: Iterable[dict[str, Any]],
        /,
        *,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self, *, print_stats: bool = False, on_error: Literal['abort', 'ignore'] = 'abort', **kwargs: Any
    ) -> UpdateStatus: ...

    @abc.abstractmethod  # type: ignore[misc]
    def insert(
        self,
        rows: Optional[Iterable[dict[str, Any]]] = None,
        /,
        *,
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        **kwargs: Any,
    ) -> UpdateStatus:
        """Inserts rows into this table. There are two mutually exclusive call patterns:

        To insert multiple rows at a time:

        ```python
        insert(
            rows: Iterable[dict[str, Any]],
            /,
            *,
            print_stats: bool = False,
            on_error: Literal['abort', 'ignore'] = 'abort'
        )```

        To insert just a single row, you can use the more concise syntax:

        ```python
        insert(
            *,
            print_stats: bool = False,
            on_error: Literal['abort', 'ignore'] = 'abort',
            **kwargs: Any
        )```

        Args:
            rows: (if inserting multiple rows) A list of rows to insert, each of which is a dictionary mapping column
                names to values.
            kwargs: (if inserting a single row) Keyword-argument pairs representing column names and values.
            print_stats: If `True`, print statistics about the cost of computed columns.
            on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
                invalid media file (such as a corrupt image) for one of the inserted rows.

                - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
                - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
                  with errors will have a `None` value for that cell, with information about the error stored in the
                  corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.

        Returns:
            An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.

        Raises:
            Error: If one of the following conditions occurs:

                - The table is a view or snapshot.
                - The table has been dropped.
                - One of the rows being inserted does not conform to the table schema.
                - An error occurs during processing of computed columns, and `on_error='ignore'`.

        Examples:
            Insert two rows into the table `my_table` with three int columns ``a``, ``b``, and ``c``.
            Column ``c`` is nullable:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.insert([{'a': 1, 'b': 1, 'c': 1}, {'a': 2, 'b': 2}])

            Insert a single row using the alternative syntax:

            >>> tbl.insert(a=3, b=3, c=3)
        """
        raise NotImplementedError

    def update(
        self, value_spec: dict[str, Any], where: Optional['pxt.exprs.Expr'] = None, cascade: bool = True
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
        with Env.get().begin_xact():
            status = self._tbl_version.get().update(value_spec, where, cascade)
            FileCache.get().emit_eviction_warnings()
            return status

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
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
        rows = list(rows)

        row_updates: list[dict[Column, exprs.Expr]] = []
        pk_col_names = set(c.name for c in self._tbl_version.get().primary_key_columns())

        # pseudo-column _rowid: contains the rowid of the row to update and can be used instead of the primary key
        has_rowid = _ROWID_COLUMN_NAME in rows[0]
        rowids: list[tuple[int, ...]] = []
        if len(pk_col_names) == 0 and not has_rowid:
            raise excs.Error('Table must have primary key for batch update')

        for row_spec in rows:
            col_vals = self._tbl_version.get()._validate_update_spec(
                row_spec, allow_pk=not has_rowid, allow_exprs=False
            )
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

        with Env.get().begin_xact():
            status = self._tbl_version.get().batch_update(
                row_updates,
                rowids,
                error_if_not_exists=if_not_exists == 'error',
                insert_if_not_exists=if_not_exists == 'insert',
                cascade=cascade,
            )
            FileCache.get().emit_eviction_warnings()
            return status

    def delete(self, where: Optional['pxt.exprs.Expr'] = None) -> UpdateStatus:
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
        with Env.get().begin_xact():
            self._tbl_version.get().revert()

    @property
    def external_stores(self) -> list[str]:
        return list(self._tbl_version.get().external_stores.keys())

    def _link_external_store(self, store: 'pxt.io.ExternalStore') -> None:
        """
        Links the specified `ExternalStore` to this table.
        """
        if self._tbl_version.get().is_snapshot:
            raise excs.Error(f'Table `{self._name}` is a snapshot, so it cannot be linked to an external store.')
        if store.name in self.external_stores:
            raise excs.Error(f'Table `{self._name}` already has an external store with that name: {store.name}')
        _logger.info(f'Linking external store `{store.name}` to table `{self._name}`')
        with Env.get().begin_xact():
            self._tbl_version.get().link_external_store(store)
            env.Env.get().console_logger.info(f'Linked external store `{store.name}` to table `{self._name}`.')

    def unlink_external_stores(
        self,
        stores: Optional[str | list[str]] = None,
        *,
        delete_external_data: bool = False,
        ignore_errors: bool = False,
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

        with Env.get().begin_xact():
            for store in stores:
                self._tbl_version.get().unlink_external_store(store, delete_external_data=delete_external_data)
                env.Env.get().console_logger.info(f'Unlinked external store from table `{self._name}`: {store}')

    def sync(
        self, stores: Optional[str | list[str]] = None, *, export_data: bool = True, import_data: bool = True
    ) -> 'pxt.io.SyncStatus':
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

        sync_status = pxt.io.SyncStatus.empty()
        with Env.get().begin_xact():
            for store in stores:
                store_obj = self._tbl_version.get().external_stores[store]
                store_sync_status = store_obj.sync(self, export_data=export_data, import_data=import_data)
                sync_status = sync_status.combine(store_sync_status)

        return sync_status

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + list(self._schema.keys())

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._schema.keys())
