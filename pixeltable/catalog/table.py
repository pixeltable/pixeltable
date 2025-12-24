from __future__ import annotations

import abc
import builtins
import datetime
import json
import logging
from keyword import iskeyword as is_python_keyword
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal
from uuid import UUID

import pandas as pd
import sqlalchemy as sql
from typing_extensions import overload

import pixeltable as pxt
from pixeltable import catalog, env, exceptions as excs, exprs, index, type_system as ts
from pixeltable.catalog.table_metadata import (
    ColumnMetadata,
    EmbeddingIndexParams,
    IndexMetadata,
    TableMetadata,
    VersionMetadata,
)
from pixeltable.metadata import schema
from pixeltable.metadata.utils import MetadataUtils
from pixeltable.utils.object_stores import ObjectOps

from ..exprs import ColumnRef
from ..utils.description_helper import DescriptionHelper
from ..utils.filecache import FileCache
from .column import Column
from .globals import (
    _ROWID_COLUMN_NAME,
    IfExistsParam,
    IfNotExistsParam,
    MediaValidation,
    is_system_column_name,
    is_valid_identifier,
)
from .schema_object import SchemaObject
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .update_status import UpdateStatus

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip


if TYPE_CHECKING:
    import torch.utils.data

    import pixeltable.plan
    from pixeltable.globals import TableDataSource


_logger = logging.getLogger('pixeltable')


class Table(SchemaObject):
    """
    A handle to a table, view, or snapshot. This class is the primary interface through which table operations
    (queries, insertions, updates, etc.) are performed in Pixeltable.

    Every user-invoked operation that runs an ExecNode tree (directly or indirectly) needs to call
    FileCache.emit_eviction_warnings() at the end of the operation.
    """

    # the chain of TableVersions needed to run queries and supply metadata (eg, schema)
    _tbl_version_path: TableVersionPath

    # the physical TableVersion backing this Table; None for pure snapshots
    _tbl_version: TableVersionHandle | None

    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version_path: TableVersionPath):
        super().__init__(id, name, dir_id)
        self._tbl_version_path = tbl_version_path
        self._tbl_version = None

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        old_name = self._name
        old_dir_id = self._dir_id

        cat = catalog.Catalog.get()

        @cat.register_undo_action
        def _() -> None:
            # TODO: We should really be invalidating the Table instance and forcing a reload.
            self._name = old_name
            self._dir_id = old_dir_id

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

    # this is duplicated from SchemaObject so that our API docs show the docstring for Table
    def get_metadata(self) -> 'TableMetadata':
        """
        Retrieves metadata associated with this table.

        Returns:
            A [TableMetadata][pixeltable.TableMetadata] instance containing this table's metadata.
        """
        from pixeltable.catalog import retry_loop

        @retry_loop(for_write=False)
        def op() -> 'TableMetadata':
            return self._get_metadata()

        return op()

    def _get_metadata(self) -> TableMetadata:
        tvp = self._tbl_version_path
        tv = tvp.tbl_version.get()
        columns = tvp.columns()
        column_info: dict[str, ColumnMetadata] = {}
        for col in columns:
            column_info[col.name] = ColumnMetadata(
                name=col.name,
                type_=col.col_type._to_str(as_schema=True),
                version_added=col.schema_version_add,
                is_stored=col.is_stored,
                is_primary_key=col.is_pk,
                media_validation=col.media_validation.name.lower() if col.media_validation is not None else None,  # type: ignore[typeddict-item]
                computed_with=col.value_expr.display_str(inline=False) if col.value_expr is not None else None,
                defined_in=col.get_tbl().name,
            )

        indices = tv.idxs_by_name.values()
        index_info: dict[str, IndexMetadata] = {}
        for info in indices:
            if isinstance(info.idx, index.EmbeddingIndex):
                col_ref = ColumnRef(info.col)
                embedding = info.idx.embeddings[info.col.col_type._type](col_ref)
                index_info[info.name] = IndexMetadata(
                    name=info.name,
                    columns=[info.col.name],
                    index_type='embedding',
                    parameters=EmbeddingIndexParams(
                        metric=info.idx.metric.name.lower(),  # type: ignore[typeddict-item]
                        embedding=str(embedding),
                        embedding_functions=[str(fn) for fn in info.idx.embeddings.values()],
                    ),
                )

        return TableMetadata(
            name=self._name,
            path=self._path(),
            columns=column_info,
            indices=index_info,
            is_replica=tv.is_replica,
            is_view=False,
            is_snapshot=False,
            version=self._get_version(),
            version_created=datetime.datetime.fromtimestamp(tv.created_at, tz=datetime.timezone.utc),
            schema_version=tvp.schema_version(),
            comment=self._get_comment(),
            media_validation=self._get_media_validation().name.lower(),  # type: ignore[typeddict-item]
            base=None,
        )

    def _get_version(self) -> int:
        """Return the version of this table. Used by tests to ascertain version changes."""
        return self._tbl_version_path.version()

    def _get_pxt_uri(self) -> str | None:
        with catalog.Catalog.get().begin_xact(tbl_id=self._id):
            return catalog.Catalog.get().get_additional_md(self._id).get('pxt_uri')

    def __hash__(self) -> int:
        return hash(self._tbl_version_path.tbl_id)

    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        """Return a ColumnRef for the given name."""
        col = self._tbl_version_path.get_column(name)
        if col is None:
            raise AttributeError(f'Unknown column: {name}')
        return ColumnRef(col, reference_tbl=self._tbl_version_path)

    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
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
        from pixeltable.catalog import retry_loop

        # we need retry_loop() here, because we end up loading Tables for the views
        @retry_loop(tbl=self._tbl_version_path, for_write=False)
        def op() -> list[str]:
            return [t._path() for t in self._get_views(recursive=recursive)]

        return op()

    def _get_views(self, *, recursive: bool = True, mutable_only: bool = False) -> list['Table']:
        cat = catalog.Catalog.get()
        view_ids = cat.get_view_ids(self._id)
        views = [cat.get_table_by_id(id) for id in view_ids]
        if mutable_only:
            views = [t for t in views if t._tbl_version_path.is_mutable()]
        if recursive:
            views.extend(t for view in views for t in view._get_views(recursive=True, mutable_only=mutable_only))
        return views

    def select(self, *items: Any, **named_items: Any) -> 'pxt.Query':
        """Select columns or expressions from this table.

        See [`Query.select`][pixeltable.Query.select] for more details.
        """
        from pixeltable.catalog import Catalog
        from pixeltable.plan import FromClause

        query = pxt.Query(FromClause(tbls=[self._tbl_version_path]))
        if len(items) == 0 and len(named_items) == 0:
            return query  # Select(*); no further processing is necessary

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            return query.select(*items, **named_items)

    def where(self, pred: 'exprs.Expr') -> 'pxt.Query':
        """Filter rows from this table based on the expression.

        See [`Query.where`][pixeltable.Query.where] for more details.
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            return self.select().where(pred)

    def join(
        self, other: 'Table', *, on: 'exprs.Expr' | None = None, how: 'pixeltable.plan.JoinType.LiteralType' = 'inner'
    ) -> 'pxt.Query':
        """Join this table with another table."""
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            return self.select().join(other, on=on, how=how)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'pxt.Query':
        """Order the rows of this table based on the expression.

        See [`Query.order_by`][pixeltable.Query.order_by] for more details.
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            return self.select().order_by(*items, asc=asc)

    def group_by(self, *items: 'exprs.Expr') -> 'pxt.Query':
        """Group the rows of this table based on the expression.

        See [`Query.group_by`][pixeltable.Query.group_by] for more details.
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            return self.select().group_by(*items)

    def distinct(self) -> 'pxt.Query':
        """Remove duplicate rows from table."""
        return self.select().distinct()

    def limit(self, n: int) -> 'pxt.Query':
        return self.select().limit(n)

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> pxt.Query:
        """Choose a shuffled sample of rows

        See [`Query.sample`][pixeltable.Query.sample] for more details.
        """
        return self.select().sample(
            n=n, n_per_stratum=n_per_stratum, fraction=fraction, seed=seed, stratify_by=stratify_by
        )

    def collect(self) -> 'pxt._query.ResultSet':
        """Return rows from this table."""
        return self.select().collect()

    def show(self, *args: Any, **kwargs: Any) -> 'pxt._query.ResultSet':
        """Return rows from this table."""
        return self.select().show(*args, **kwargs)

    def head(self, *args: Any, **kwargs: Any) -> 'pxt._query.ResultSet':
        """Return the first n rows inserted into this table."""
        return self.select().head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> 'pxt._query.ResultSet':
        """Return the last n rows inserted into this table."""
        return self.select().tail(*args, **kwargs)

    def count(self) -> int:
        """Return the number of rows in this table."""
        return self.select().count()

    def columns(self) -> list[str]:
        """Return the names of the columns in this table."""
        cols = self._tbl_version_path.columns()
        return [c.name for c in cols]

    def _get_schema(self) -> dict[str, ts.ColumnType]:
        """Return the schema (column names and column types) of this table."""
        return {c.name: c.col_type for c in self._tbl_version_path.columns()}

    def get_base_table(self) -> 'Table' | None:
        return self._get_base_table()

    @abc.abstractmethod
    def _get_base_table(self) -> 'Table' | None:
        """The base's Table instance. Requires a transaction context"""

    def _get_base_tables(self) -> list['Table']:
        """The ancestor list of bases of this table, starting with its immediate base. Requires a transaction context"""
        bases: list[Table] = []
        base = self._get_base_table()
        while base is not None:
            bases.append(base)
            base = base._get_base_table()
        return bases

    @property
    @abc.abstractmethod
    def _effective_base_versions(self) -> list[int | None]:
        """The effective versions of the ancestor bases, starting with its immediate base."""

    def _get_comment(self) -> str:
        return self._tbl_version_path.comment()

    def _get_num_retained_versions(self) -> int:
        return self._tbl_version_path.num_retained_versions()

    def _get_media_validation(self) -> MediaValidation:
        return self._tbl_version_path.media_validation()

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def _descriptors(self) -> DescriptionHelper:
        """
        Constructs a list of descriptors for this table that can be pretty-printed.
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=False):
            helper = DescriptionHelper()
            helper.append(self._table_descriptor())
            helper.append(self._col_descriptor())
            idxs = self._index_descriptor()
            if not idxs.empty:
                helper.append(idxs)
            stores = self._external_store_descriptor()
            if not stores.empty:
                helper.append(stores)
            if self._get_comment():
                helper.append(f'COMMENT: {self._get_comment()}')
            return helper

    def _col_descriptor(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'Column Name': col.name,
                'Type': col.col_type._to_str(as_schema=True),
                'Computed With': col.value_expr.display_str(inline=False) if col.value_expr is not None else '',
            }
            for col in self._tbl_version_path.columns()
            if columns is None or col.name in columns
        )

    def _index_descriptor(self, columns: list[str] | None = None) -> pd.DataFrame:
        from pixeltable import index

        if self._tbl_version is None:
            return pd.DataFrame([])
        pd_rows = []
        for name, info in self._tbl_version.get().idxs_by_name.items():
            if isinstance(info.idx, index.EmbeddingIndex) and (columns is None or info.col.name in columns):
                col_ref = ColumnRef(info.col)
                embedding = info.idx.embeddings[info.col.col_type._type](col_ref)
                row = {
                    'Index Name': name,
                    'Column': info.col.name,
                    'Metric': str(info.idx.metric.name.lower()),
                    'Embedding': str(embedding),
                }
                pd_rows.append(row)
        return pd.DataFrame(pd_rows)

    def _external_store_descriptor(self) -> pd.DataFrame:
        pd_rows = []
        for name, store in self._tbl_version_path.tbl_version.get().external_stores.items():
            row = {'External Store': name, 'Type': type(store).__name__}
            pd_rows.append(row)
        return pd.DataFrame(pd_rows)

    def describe(self) -> None:
        """
        Print the table schema.
        """
        if getattr(builtins, '__IPYTHON__', False):
            from IPython.display import Markdown, display

            display(Markdown(self._repr_html_()))
        else:
            print(repr(self))

    # TODO Factor this out into a separate module.
    # The return type is unresolvable, but torch can't be imported since it's an optional dependency.
    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        """Return a PyTorch Dataset for this table.
        See Query.to_pytorch_dataset()
        """
        return self.select().to_pytorch_dataset(image_format=image_format)

    def to_coco_dataset(self) -> Path:
        """Return the path to a COCO json file for this table.
        See Query.to_coco_dataset()
        """
        return self.select().to_coco_dataset()

    def _column_has_dependents(self, col: Column) -> bool:
        """Returns True if the column has dependents, False otherwise."""
        assert col is not None
        assert col.name in self._get_schema()
        cat = catalog.Catalog.get()
        if any(c.name is not None for c in cat.get_column_dependents(col.get_tbl().id, col.id)):
            return True
        assert self._tbl_version is not None
        return any(
            col in store.get_local_columns()
            for view in (self, *self._get_views(recursive=True))
            for store in view._tbl_version.get().external_stores.values()
        )

    def _ignore_or_drop_existing_columns(self, new_col_names: list[str], if_exists: IfExistsParam) -> list[str]:
        """Check and handle existing columns in the new column specification based on the if_exists parameter.

        If `if_exists='ignore'`, returns a list of existing columns, if any, in `new_col_names`.
        """
        assert self._tbl_version is not None
        existing_col_names = set(self._get_schema().keys())
        cols_to_ignore = []
        for new_col_name in new_col_names:
            if new_col_name in existing_col_names:
                if if_exists == IfExistsParam.ERROR:
                    raise excs.Error(f'Duplicate column name: {new_col_name}')
                elif if_exists == IfExistsParam.IGNORE:
                    cols_to_ignore.append(new_col_name)
                elif if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE):
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
                            f'Column {new_col_name!r} already exists and has dependents. '
                            f'Cannot {if_exists.name.lower()} it.'
                        )
                    self.drop_column(new_col_name)
                    assert new_col_name not in self._tbl_version.get().cols_by_name
        return cols_to_ignore

    def add_columns(
        self,
        schema: dict[str, ts.ColumnType | builtins.type | _GenericAlias],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        """
        Adds multiple columns to the table. The columns must be concrete (non-computed) columns; to add computed
        columns, use [`add_computed_column()`][pixeltable.catalog.Table.add_computed_column] instead.

        The format of the `schema` argument is a dict mapping column names to their types.

        Args:
            schema: A dictionary mapping column names to types.
            if_exists: Determines the behavior if a column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has no
                    dependents.

                Note that the `if_exists` parameter is applied to all columns in the schema.
                To apply different behaviors to different columns, please use
                [`add_column()`][pixeltable.Table.add_column] for each column.

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
        from pixeltable.catalog import Catalog

        # lock_mutable_tree=True: we might end up having to drop existing columns, which requires locking the tree
        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('add columns to')
            col_schema = {
                col_name: {'type': ts.ColumnType.normalize_type(spec, nullable_default=True, allow_builtin_types=False)}
                for col_name, spec in schema.items()
            }

            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                list(col_schema.keys()), IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if all columns to be added already exist and user asked to ignore
            # existing columns, there's nothing to do.
            for cname in cols_to_ignore:
                assert cname in col_schema
                del col_schema[cname]
            result = UpdateStatus()
            if len(col_schema) == 0:
                return result
            new_cols = self._create_columns(col_schema)
            for new_col in new_cols:
                self._verify_column(new_col)
            assert self._tbl_version is not None
            result += self._tbl_version.get().add_columns(new_cols, print_stats=False, on_error='abort')
            FileCache.get().emit_eviction_warnings()
            return result

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: ts.ColumnType | builtins.type | _GenericAlias | exprs.Expr,
    ) -> UpdateStatus:
        """
        Adds an ordinary (non-computed) column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form `col_name=col_type`.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace'` or `'replace_force'`: drop the existing column and add the new column, if it has
                    no dependents.

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
        # verify kwargs and construct column schema dict
        if len(kwargs) != 1:
            raise excs.Error(
                f'add_column() requires exactly one keyword argument of the form `col_name=col_type`; '
                f'got {len(kwargs)} arguments instead ({", ".join(kwargs.keys())})'
            )
        col_type = next(iter(kwargs.values()))
        if not isinstance(col_type, (ts.ColumnType, type, _GenericAlias)):
            raise excs.Error(
                'The argument to add_column() must be a type; did you intend to use add_computed_column() instead?'
            )
        return self.add_columns(kwargs, if_exists=if_exists)

    def add_computed_column(
        self,
        *,
        stored: bool | None = None,
        destination: str | Path | None = None,
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
            destination: An object store reference for persisting computed files.
            print_stats: If `True`, print execution metrics during evaluation.
            on_error: Determines the behavior if an error occurs while evaluating the column expression for at least one
                row.

                - `'abort'`: an exception will be raised and the column will not be added.
                - `'ignore'`: execution will continue and the column will be added. Any rows
                    with errors will have a `None` value for the column, with information about the error stored in the
                    corresponding `tbl.col_name.errormsg` and `tbl.col_name.errortype` fields.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, iff it has
                    no dependents.

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
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('add columns to')
            if len(kwargs) != 1:
                raise excs.Error(
                    f'add_computed_column() requires exactly one keyword argument of the form '
                    '`col_name=col_type` or `col_name=expression`; '
                    f'got {len(kwargs)} arguments instead ({", ".join(kwargs.keys())})'
                )
            col_name, spec = next(iter(kwargs.items()))
            if not is_valid_identifier(col_name):
                raise excs.Error(f'Invalid column name: {col_name}')

            col_schema: dict[str, Any] = {'value': spec}
            if stored is not None:
                col_schema['stored'] = stored

            if destination is not None:
                col_schema['destination'] = destination

            # Raise an error if the column expression refers to a column error property
            if isinstance(spec, exprs.Expr):
                for e in spec.subexprs(expr_class=exprs.ColumnPropertyRef, traverse_matches=False):
                    if e.is_cellmd_prop():
                        raise excs.Error(
                            f'Use of a reference to the {e.prop.name.lower()!r} property of another column '
                            f'is not allowed in a computed column.'
                        )

            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                [col_name], IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if the column to add already exists and user asked to ignore
            # existing column, there's nothing to do.
            result = UpdateStatus()
            if len(cols_to_ignore) != 0:
                assert cols_to_ignore[0] == col_name
                return result

            new_col = self._create_columns({col_name: col_schema})[0]
            self._verify_column(new_col)
            assert self._tbl_version is not None
            result += self._tbl_version.get().add_columns([new_col], print_stats=print_stats, on_error=on_error)
            FileCache.get().emit_eviction_warnings()
            return result

    @classmethod
    def _validate_column_spec(cls, name: str, spec: dict[str, Any]) -> None:
        """Check integrity of user-supplied Column spec

        We unfortunately can't use something like jsonschema for validation, because this isn't strictly a JSON schema
        (on account of containing Python Callables or Exprs).
        """
        assert isinstance(spec, dict)
        valid_keys = {'type', 'value', 'stored', 'media_validation', 'destination'}
        for k in spec:
            if k not in valid_keys:
                raise excs.Error(f'Column {name!r}: invalid key {k!r}')

        if 'type' not in spec and 'value' not in spec:
            raise excs.Error(f"Column {name!r}: 'type' or 'value' must be specified")

        if 'type' in spec and not isinstance(spec['type'], (ts.ColumnType, type, _GenericAlias)):
            raise excs.Error(f"Column {name!r}: 'type' must be a type or ColumnType; got {spec['type']}")

        if 'value' in spec:
            value_expr = exprs.Expr.from_object(spec['value'])
            if value_expr is None:
                raise excs.Error(f"Column {name!r}: 'value' must be a Pixeltable expression.")
            if 'type' in spec:
                raise excs.Error(f"Column {name!r}: 'type' is redundant if 'value' is specified")

        if 'media_validation' in spec:
            _ = catalog.MediaValidation.validated(spec['media_validation'], f'Column {name!r}: media_validation')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise excs.Error(f"Column {name!r}: 'stored' must be a bool; got {spec['stored']}")

        d = spec.get('destination')
        if d is not None and not isinstance(d, (str, Path)):
            raise excs.Error(f'Column {name!r}: `destination` must be a string or path; got {d}')

    @classmethod
    def _create_columns(cls, schema: dict[str, Any]) -> list[Column]:
        """Construct list of Columns, given schema"""
        columns: list[Column] = []
        for name, spec in schema.items():
            col_type: ts.ColumnType | None = None
            value_expr: exprs.Expr | None = None
            primary_key: bool = False
            media_validation: catalog.MediaValidation | None = None
            stored = True
            destination: str | None = None

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
                primary_key = spec.get('primary_key', False)
                media_validation_str = spec.get('media_validation')
                media_validation = (
                    catalog.MediaValidation[media_validation_str.upper()] if media_validation_str is not None else None
                )
                destination = spec.get('destination')
            else:
                raise excs.Error(f'Invalid value for column {name!r}')

            column = Column(
                name,
                col_type=col_type,
                computed_with=value_expr,
                stored=stored,
                is_pk=primary_key,
                media_validation=media_validation,
                destination=destination,
            )
            # Validate the column's resolved_destination. This will ensure that if the column uses a default (global)
            # media destination, it gets validated at this time.
            ObjectOps.validate_destination(column.destination, column.name)
            columns.append(column)

        return columns

    @classmethod
    def validate_column_name(cls, name: str) -> None:
        """Check that a name is usable as a pixeltable column name"""
        if is_system_column_name(name) or is_python_keyword(name):
            raise excs.Error(f'{name!r} is a reserved name in Pixeltable; please choose a different column name.')
        if not is_valid_identifier(name):
            raise excs.Error(f'Invalid column name: {name}')

    @classmethod
    def _verify_column(cls, col: Column) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        cls.validate_column_name(col.name)
        if col.stored is False and not col.is_computed:
            raise excs.Error(f'Column {col.name!r}: `stored={col.stored}` only applies to computed columns')
        if col.stored is False and col.has_window_fn_call():
            raise excs.Error(
                (
                    f'Column {col.name!r}: `stored={col.stored}` is not valid for image columns computed with a '
                    f'streaming function'
                )
            )
        if col._explicit_destination is not None and not (col.stored and col.is_computed):
            raise excs.Error(f'Column {col.name!r}: `destination` property only applies to stored computed columns')

    @classmethod
    def _verify_schema(cls, schema: list[Column]) -> None:
        """Check integrity of user-supplied schema and set defaults"""
        for col in schema:
            cls._verify_column(col)

    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
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
        from pixeltable.catalog import Catalog

        cat = Catalog.get()

        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        with cat.begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('drop columns from')
            col: Column = None
            if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')

            if isinstance(column, str):
                col = self._tbl_version_path.get_column(column)
                if col is None:
                    if if_not_exists_ == IfNotExistsParam.ERROR:
                        raise excs.Error(f'Unknown column: {column}')
                    assert if_not_exists_ == IfNotExistsParam.IGNORE
                    return
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.Error(f'Cannot drop base table column {col.name!r}')
                col = self._tbl_version.get().cols_by_name[column]
            else:
                exists = self._tbl_version_path.has_column(column.col)
                if not exists:
                    if if_not_exists_ == IfNotExistsParam.ERROR:
                        raise excs.Error(f'Unknown column: {column.col.qualified_name}')
                    assert if_not_exists_ == IfNotExistsParam.IGNORE
                    return
                col = column.col
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.Error(f'Cannot drop base table column {col.name!r}')

            dependent_user_cols = [c for c in cat.get_column_dependents(col.get_tbl().id, col.id) if c.name is not None]
            if len(dependent_user_cols) > 0:
                raise excs.Error(
                    f'Cannot drop column {col.name!r} because the following columns depend on it:\n'
                    f'{", ".join(c.name for c in dependent_user_cols)}'
                )

            views = self._get_views(recursive=True, mutable_only=True)

            # See if any view predicates depend on this column
            dependent_views: list[tuple[Table, exprs.Expr]] = []
            for view in views:
                if view._tbl_version is not None:
                    predicate = view._tbl_version.get().predicate
                    if predicate is not None:
                        for predicate_col in exprs.Expr.get_refd_column_ids(predicate.as_dict()):
                            if predicate_col.tbl_id == col.get_tbl().id and predicate_col.col_id == col.id:
                                dependent_views.append((view, predicate))

            if len(dependent_views) > 0:
                dependent_views_str = '\n'.join(
                    f'view: {view._path()}, predicate: {predicate}' for view, predicate in dependent_views
                )
                raise excs.Error(
                    f'Cannot drop column {col.name!r} because the following views depend on it:\n{dependent_views_str}'
                )

            # See if this column has a dependent store. We need to look through all stores in all
            # (transitive) views of this table.
            col_handle = col.handle
            dependent_stores = [
                (view, store)
                for view in (self, *views)
                for store in view._tbl_version.get().external_stores.values()
                if col_handle in store.get_local_columns()
            ]
            if len(dependent_stores) > 0:
                dependent_store_names = [
                    store.name if view._id == self._id else f'{store.name} (in view {view._name!r})'
                    for view, store in dependent_stores
                ]
                raise excs.Error(
                    f'Cannot drop column {col.name!r} because the following external stores depend on it:\n'
                    f'{", ".join(dependent_store_names)}'
                )
            all_columns = self.columns()
            if len(all_columns) == 1 and col.name == all_columns[0]:
                raise excs.Error(
                    f'Cannot drop column {col.name!r} because it is the last remaining column in this table.'
                    f' Tables must have at least one column.'
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
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=False):
            self._tbl_version.get().rename_column(old_name, new_name)

    def _list_index_info_for_test(self) -> list[dict[str, Any]]:
        """
        Returns list of all the indexes on this table. Used for testing.

        Returns:
            A list of index information, each containing the index's
            id, name, and the name of the column it indexes.
        """
        index_info = []
        for idx_name, idx in self._tbl_version.get().idxs_by_name.items():
            index_info.append({'_id': idx.id, '_name': idx_name, '_column': idx.col.name})
        return index_info

    def add_embedding_index(
        self,
        column: str | ColumnRef,
        *,
        idx_name: str | None = None,
        embedding: pxt.Function | None = None,
        string_embed: pxt.Function | None = None,
        image_embed: pxt.Function | None = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        """
        Add an embedding index to the table. Once the index is created, it will be automatically kept up-to-date as new
        rows are inserted into the table.

        To add an embedding index, one must specify, at minimum, the column to be indexed and an embedding UDF.
        Only `String` and `Image` columns are currently supported.

        Args:
            column: The name of, or reference to, the column to be indexed; must be a `String` or `Image` column.
            idx_name: An optional name for the index. If not specified, a name such as `'idx0'` will be generated
                automatically. If specified, the name must be unique for this table and a valid pixeltable column name.
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
            Error: If an index with the specified name already exists for the table and `if_exists='error'`, or if
                the specified column does not exist.

        Examples:
            Add an index to the `img` column of the table `my_table`:

            >>> from pixeltable.functions.huggingface import clip
            >>> tbl = pxt.get_table('my_table')
            >>> embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
            >>> tbl.add_embedding_index(tbl.img, embedding=embedding_fn)

            Alternatively, the `img` column may be specified by name:

            >>> tbl.add_embedding_index('img', embedding=embedding_fn)

            Once the index is created, similarity lookups can be performed using the `similarity` pseudo-function:

            >>> sim = tbl.img.similarity(image='/path/to/my-image.jpg')  # can also be a URL or a PIL image
            >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

            If the embedding UDF is a multimodal embedding (supporting more than one data type), then lookups may be
            performed using any of its supported modalities. In our example, CLIP supports both text and images, so we
            can also search for images using a text description:

            >>> sim = tbl.img.similarity(string='a picture of a train')
            >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

            Audio and video lookups would look like this:

            >>> sim = tbl.img.similarity(audio='/path/to/audio.flac')
            >>> sim = tbl.img.similarity(video='/path/to/video.mp4')

            Multiple indexes can be defined on each column. Add a second index to the `img` column, using the inner
            product as the distance metric, and with a specific name:

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
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('add an index to')
            col = self._resolve_column_parameter(column)

            if idx_name is not None and idx_name in self._tbl_version.get().idxs_by_name:
                if_exists_ = IfExistsParam.validated(if_exists, 'if_exists')
                # An index with the same name already exists.
                # Handle it according to if_exists.
                if if_exists_ == IfExistsParam.ERROR:
                    raise excs.Error(f'Duplicate index name: {idx_name}')
                if not isinstance(self._tbl_version.get().idxs_by_name[idx_name].idx, index.EmbeddingIndex):
                    raise excs.Error(
                        f'Index {idx_name!r} is not an embedding index. Cannot {if_exists_.name.lower()} it.'
                    )
                if if_exists_ == IfExistsParam.IGNORE:
                    return
                assert if_exists_ in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)
                self.drop_index(idx_name=idx_name)
                assert idx_name not in self._tbl_version.get().idxs_by_name
            from pixeltable.index import EmbeddingIndex

            # idx_name must be a valid pixeltable column name
            if idx_name is not None:
                Table.validate_column_name(idx_name)

            # validate EmbeddingIndex args
            idx = EmbeddingIndex(metric=metric, embed=embedding, string_embed=string_embed, image_embed=image_embed)
            _ = idx.create_value_expr(col)
            _ = self._tbl_version.get().add_index(col, idx_name=idx_name, idx=idx)
            # TODO: how to deal with exceptions here? drop the index and raise?
            FileCache.get().emit_eviction_warnings()

    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
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
        from pixeltable.catalog import Catalog

        if (column is None) == (idx_name is None):
            raise excs.Error("Exactly one of 'column' or 'idx_name' must be provided")

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            col: Column = None
            if idx_name is None:
                col = self._resolve_column_parameter(column)
                assert col is not None

            self._drop_index(col=col, idx_name=idx_name, _idx_class=index.EmbeddingIndex, if_not_exists=if_not_exists)

    def _resolve_column_parameter(self, column: str | ColumnRef) -> Column:
        """Resolve a column parameter to a Column object"""
        col: Column = None
        if isinstance(column, str):
            col = self._tbl_version_path.get_column(column)
            if col is None:
                raise excs.Error(f'Unknown column: {column}')
        elif isinstance(column, ColumnRef):
            exists = self._tbl_version_path.has_column(column.col)
            if not exists:
                raise excs.Error(f'Unknown column: {column.col.qualified_name}')
            col = column.col
        else:
            raise excs.Error(f'Invalid column parameter type: {type(column)}')
        return col

    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
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
        from pixeltable.catalog import Catalog

        if (column is None) == (idx_name is None):
            raise excs.Error("Exactly one of 'column' or 'idx_name' must be provided")

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=False):
            col: Column = None
            if idx_name is None:
                col = self._resolve_column_parameter(column)
                assert col is not None

            self._drop_index(col=col, idx_name=idx_name, if_not_exists=if_not_exists)

    def _drop_index(
        self,
        *,
        col: Column | None = None,
        idx_name: str | None = None,
        _idx_class: type[index.IndexBase] | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        from pixeltable.catalog import Catalog

        self.__check_mutable('drop an index from')
        assert (col is None) != (idx_name is None)

        if idx_name is not None:
            if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
            if idx_name not in self._tbl_version.get().idxs_by_name:
                if if_not_exists_ == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Index {idx_name!r} does not exist')
                assert if_not_exists_ == IfNotExistsParam.IGNORE
                return
            idx_info = self._tbl_version.get().idxs_by_name[idx_name]
        else:
            if col.get_tbl().id != self._tbl_version.id:
                raise excs.Error(
                    f'Column {col.name!r}: '
                    f'cannot drop index from column that belongs to base table {col.get_tbl().name!r}'
                )
            idx_info_list = [info for info in self._tbl_version.get().idxs_by_name.values() if info.col.id == col.id]
            if _idx_class is not None:
                idx_info_list = [info for info in idx_info_list if isinstance(info.idx, _idx_class)]
            if len(idx_info_list) == 0:
                if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
                if if_not_exists_ == IfNotExistsParam.ERROR:
                    raise excs.Error(f'Column {col.name!r} does not have an index')
                assert if_not_exists_ == IfNotExistsParam.IGNORE
                return
            if len(idx_info_list) > 1:
                raise excs.Error(f'Column {col.name!r} has multiple indices; specify `idx_name` explicitly to drop one')
            idx_info = idx_info_list[0]

        # Find out if anything depends on this index
        val_col = idx_info.val_col
        dependent_user_cols = [
            c for c in Catalog.get().get_column_dependents(val_col.get_tbl().id, val_col.id) if c.name is not None
        ]
        if len(dependent_user_cols) > 0:
            raise excs.Error(
                f'Cannot drop index {idx_info.name!r} because the following columns depend on it:\n'
                f'{", ".join(c.name for c in dependent_user_cols)}'
            )
        self._tbl_version.get().drop_index(idx_info.id)

    @overload
    def insert(
        self,
        source: TableDataSource,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self, /, *, on_error: Literal['abort', 'ignore'] = 'abort', print_stats: bool = False, **kwargs: Any
    ) -> UpdateStatus: ...

    @abc.abstractmethod
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        """Inserts rows into this table. There are two mutually exclusive call patterns:

        To insert multiple rows at a time:

        ```python
        insert(
            source: TableSourceDataType,
            /,
            *,
            on_error: Literal['abort', 'ignore'] = 'abort',
            print_stats: bool = False,
            **kwargs: Any,
        )
        ```

        To insert just a single row, you can use the more concise syntax:

        ```python
        insert(
            *,
            on_error: Literal['abort', 'ignore'] = 'abort',
            print_stats: bool = False,
            **kwargs: Any
        )
        ```

        Args:
            source: A data source from which data can be imported.
            kwargs: (if inserting a single row) Keyword-argument pairs representing column names and values.
                (if inserting multiple rows) Additional keyword arguments are passed to the data source.
            source_format: A hint about the format of the source data
            schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
                invalid media file (such as a corrupt image) for one of the inserted rows.

                - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
                - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
                    with errors will have a `None` value for that cell, with information about the error stored in the
                    corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
            print_stats: If `True`, print statistics about the cost of computed columns.

        Returns:
            An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.

        Raises:
            Error: If one of the following conditions occurs:

                - The table is a view or snapshot.
                - The table has been dropped.
                - One of the rows being inserted does not conform to the table schema.
                - An error occurs during processing of computed columns, and `on_error='ignore'`.
                - An error occurs while importing data from a source, and `on_error='abort'`.

        Examples:
            Insert two rows into the table `my_table` with three int columns ``a``, ``b``, and ``c``.
            Column ``c`` is nullable:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.insert([{'a': 1, 'b': 1, 'c': 1}, {'a': 2, 'b': 2}])

            Insert a single row using the alternative syntax:

            >>> tbl.insert(a=3, b=3, c=3)

            Insert rows from a CSV file:

            >>> tbl.insert(source='path/to/file.csv')

            Insert Pydantic model instances into a table with two `pxt.Int` columns `a` and `b`:

            >>> class MyModel(pydantic.BaseModel):
            ...     a: int
            ...     b: int
            ...
            ... models = [MyModel(a=1, b=2), MyModel(a=3, b=4)]
            ... tbl.insert(models)
        """
        raise NotImplementedError

    def update(
        self, value_spec: dict[str, Any], where: 'exprs.Expr' | None = None, cascade: bool = True
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            value_spec: a dictionary mapping column names to literal values or Pixeltable expressions.
            where: a predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.

        Returns:
            An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.

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
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('update')
            result = self._tbl_version.get().update(value_spec, where, cascade)
            FileCache.get().emit_eviction_warnings()
            return result

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

            >>> tbl.batch_update(
            ...     [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 40}]
            ... )

            Update the `name` and `age` columns for the row with `id` 1 (assuming `id` is the primary key) and insert
            the row with new `id` 3 (assuming this key does not exist):

            >>> tbl.batch_update(
            ...     [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 3, 'name': 'Bob', 'age': 40}],
            ...     if_not_exists='insert'
            ... )
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('update')
            rows = list(rows)

            row_updates: list[dict[Column, exprs.Expr]] = []
            pk_col_names = {c.name for c in self._tbl_version.get().primary_key_columns()}

            # pseudo-column _rowid: contains the rowid of the row to update and can be used instead of the primary key
            has_rowid = _ROWID_COLUMN_NAME in rows[0]
            rowids: list[tuple[int, ...]] = []
            if len(pk_col_names) == 0 and not has_rowid:
                raise excs.Error('Table must have primary key for batch update')

            for row_spec in rows:
                col_vals = self._tbl_version.get()._validate_update_spec(
                    row_spec, allow_pk=not has_rowid, allow_exprs=False, allow_media=False
                )
                if has_rowid:
                    # we expect the _rowid column to be present for each row
                    assert _ROWID_COLUMN_NAME in row_spec
                    rowids.append(row_spec[_ROWID_COLUMN_NAME])
                else:
                    col_names = {col.name for col in col_vals}
                    if any(pk_col_name not in col_names for pk_col_name in pk_col_names):
                        missing_cols = pk_col_names - {col.name for col in col_vals}
                        raise excs.Error(
                            f'Primary key column(s) {", ".join(repr(c) for c in missing_cols)} missing in {row_spec}'
                        )
                row_updates.append(col_vals)

            result = self._tbl_version.get().batch_update(
                row_updates,
                rowids,
                error_if_not_exists=if_not_exists == 'error',
                insert_if_not_exists=if_not_exists == 'insert',
                cascade=cascade,
            )
            FileCache.get().emit_eviction_warnings()
            return result

    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        """Recompute the values in one or more computed columns of this table.

        Args:
            columns: The names or references of the computed columns to recompute.
            where: A predicate to filter rows to recompute.
            errors_only: If True, only run the recomputation for rows that have errors in the column (ie, the column's
                `errortype` property indicates that an error occurred). Only allowed for recomputing a single column.
            cascade: if True, also update all computed columns that transitively depend on the recomputed columns.

        Examples:
            Recompute computed columns `c1` and `c2` for all rows in this table, and everything that transitively
            depends on them:

            >>> tbl.recompute_columns('c1', 'c2')

            Recompute computed column `c1` for all rows in this table, but don't recompute other columns that depend on
            it:

            >>> tbl.recompute_columns(tbl.c1, tbl.c2, cascade=False)

            Recompute column `c1` and its dependents, but only for rows with `c2` == 0:

            >>> tbl.recompute_columns('c1', where=tbl.c2 == 0)

            Recompute column `c1` and its dependents, but only for rows that have errors in it:

            >>> tbl.recompute_columns('c1', errors_only=True)
        """
        from pixeltable.catalog import Catalog

        cat = Catalog.get()
        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        with cat.begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('recompute columns of')
            if len(columns) == 0:
                raise excs.Error('At least one column must be specified to recompute')
            if errors_only and len(columns) > 1:
                raise excs.Error('Cannot use errors_only=True with multiple columns')

            col_names: list[str] = []
            for column in columns:
                col_name: str
                col: Column
                if isinstance(column, str):
                    col = self._tbl_version_path.get_column(column)
                    if col is None:
                        raise excs.Error(f'Unknown column: {column}')
                    col_name = column
                else:
                    assert isinstance(column, ColumnRef)
                    col = column.col
                    if not self._tbl_version_path.has_column(col):
                        raise excs.Error(f'Unknown column: {col.name}')
                    col_name = col.name
                if not col.is_computed:
                    raise excs.Error(f'Column {col_name!r} is not a computed column')
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.Error(f'Cannot recompute column of a base: {col_name}')
                col_names.append(col_name)

            if where is not None and not where.is_bound_by([self._tbl_version_path]):
                raise excs.Error(f'`where` predicate ({where}) is not bound by {self._display_str()}')

            result = self._tbl_version.get().recompute_columns(
                col_names, where=where, errors_only=errors_only, cascade=cascade
            )
            FileCache.get().emit_eviction_warnings()
            return result

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
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
        with catalog.Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            self.__check_mutable('revert')
            self._tbl_version.get().revert()
            # remove cached md in order to force a reload on the next operation
            self._tbl_version_path.clear_cached_md()

    def push(self) -> None:
        from pixeltable.share import push_replica
        from pixeltable.share.protocol import PxtUri

        pxt_uri = self._get_pxt_uri()
        tbl_version = self._tbl_version_path.tbl_version.get()

        if tbl_version.is_replica:
            raise excs.Error(f'push(): Cannot push replica table {self._name!r}. (Did you mean `pull()`?)')

        if pxt_uri is None:
            raise excs.Error(
                f'push(): Table {self._name!r} has not yet been published to Pixeltable Cloud. '
                'To publish it, use `pxt.publish()` instead.'
            )

        if isinstance(self, catalog.View) and self._is_anonymous_snapshot():
            raise excs.Error(
                f'push(): Cannot push specific-version table handle {tbl_version.versioned_name!r}. '
                'To push the latest version instead:\n'
                f'  t = pxt.get_table({self._name!r})\n'
                f'  t.push()'
            )

        if self._tbl_version is None:
            # Named snapshots never have new versions to push.
            env.Env.get().console_logger.info('push(): Everything up to date.')
            return

        # Parse the pxt URI to extract org/db and create a UUID-based URI for pushing
        parsed_uri = PxtUri(uri=pxt_uri)
        uuid_uri_obj = PxtUri.from_components(org=parsed_uri.org, id=self._id, db=parsed_uri.db)
        uuid_uri = str(uuid_uri_obj)

        push_replica(uuid_uri, self)

    def pull(self) -> None:
        from pixeltable.share import pull_replica
        from pixeltable.share.protocol import PxtUri

        pxt_uri = self._get_pxt_uri()
        tbl_version = self._tbl_version_path.tbl_version.get()

        if not tbl_version.is_replica or pxt_uri is None:
            raise excs.Error(
                f'pull(): Table {self._name!r} is not a replica of a Pixeltable Cloud table (nothing to `pull()`).'
            )

        if isinstance(self, catalog.View) and self._is_anonymous_snapshot():
            raise excs.Error(
                f'pull(): Cannot pull specific-version table handle {tbl_version.versioned_name!r}. '
                'To pull the latest version instead:\n'
                f'  t = pxt.get_table({self._name!r})\n'
                f'  t.pull()'
            )

        # Parse the pxt URI to extract org/db and create a UUID-based URI for pulling
        parsed_uri = PxtUri(uri=pxt_uri)
        uuid_uri_obj = PxtUri.from_components(org=parsed_uri.org, id=self._id, db=parsed_uri.db)
        uuid_uri = str(uuid_uri_obj)

        pull_replica(self._path(), uuid_uri)

    def external_stores(self) -> list[str]:
        return list(self._tbl_version.get().external_stores.keys())

    def _link_external_store(self, store: 'pxt.io.ExternalStore') -> None:
        """
        Links the specified `ExternalStore` to this table.
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=False):
            self.__check_mutable('link an external store to')
            if store.name in self.external_stores():
                raise excs.Error(f'Table {self._name!r} already has an external store with that name: {store.name}')
            _logger.info(f'Linking external store {store.name!r} to table {self._name!r}.')

            store.link(self._tbl_version.get())  # might call tbl_version.add_columns()
            self._tbl_version.get().link_external_store(store)
            env.Env.get().console_logger.info(f'Linked external store {store.name!r} to table {self._name!r}.')

    def unlink_external_stores(
        self, stores: str | list[str] | None = None, *, delete_external_data: bool = False, ignore_errors: bool = False
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
        from pixeltable.catalog import Catalog

        if not self._tbl_version_path.is_mutable():
            return
        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=False):
            all_stores = self.external_stores()

            if stores is None:
                stores = all_stores
            elif isinstance(stores, str):
                stores = [stores]

            # Validation
            if not ignore_errors:
                for store_name in stores:
                    if store_name not in all_stores:
                        raise excs.Error(f'Table {self._name!r} has no external store with that name: {store_name}')

            for store_name in stores:
                store = self._tbl_version.get().external_stores[store_name]
                # get hold of the store's debug string before deleting it
                store_str = str(store)
                store.unlink(self._tbl_version.get())  # might call tbl_version.drop_columns()
                self._tbl_version.get().unlink_external_store(store)
                if delete_external_data and isinstance(store, pxt.io.external_store.Project):
                    store.delete()
                env.Env.get().console_logger.info(f'Unlinked external store from table {self._name!r}: {store_str}')

    def sync(
        self, stores: str | list[str] | None = None, *, export_data: bool = True, import_data: bool = True
    ) -> UpdateStatus:
        """
        Synchronizes this table with its linked external stores.

        Args:
            stores: If specified, will synchronize only the specified named store or list of stores. If not specified,
                will synchronize all of this table's external stores.
            export_data: If `True`, data from this table will be exported to the external stores during synchronization.
            import_data: If `True`, data from the external stores will be imported to this table during synchronization.
        """
        from pixeltable.catalog import Catalog

        if not self._tbl_version_path.is_mutable():
            return UpdateStatus()
        # we lock the entire tree starting at the root base table in order to ensure that all synced columns can
        # have their updates propagated down the tree
        base_tv = self._tbl_version_path.get_tbl_versions()[-1]
        with Catalog.get().begin_xact(tbl=TableVersionPath(base_tv), for_write=True, lock_mutable_tree=True):
            all_stores = self.external_stores()

            if stores is None:
                stores = all_stores
            elif isinstance(stores, str):
                stores = [stores]

            for store in stores:
                if store not in all_stores:
                    raise excs.Error(f'Table {self._name!r} has no external store with that name: {store}')

            sync_status = UpdateStatus()
            for store in stores:
                store_obj = self._tbl_version.get().external_stores[store]
                store_sync_status = store_obj.sync(self, export_data=export_data, import_data=import_data)
                sync_status += store_sync_status

        return sync_status

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + list(self._get_schema().keys())

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._get_schema().keys())

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        """
        Returns information about versions of this table, most recent first.

        `get_versions()` is intended for programmatic access to version metadata; for human-readable
        output, use [`history()`][pixeltable.Table.history] instead.

        Args:
            n: if specified, will return at most `n` versions

        Returns:
            A list of [VersionMetadata][pixeltable.VersionMetadata] dictionaries, one per version retrieved, most
            recent first.

        Examples:
            Retrieve metadata about all versions of the table `tbl`:

            >>> tbl.get_versions()

            Retrieve metadata about the most recent 5 versions of the table `tbl`:

            >>> tbl.get_versions(n=5)
        """
        from pixeltable.catalog import Catalog

        if n is None:
            n = 1_000_000_000
        if not isinstance(n, int) or n < 1:
            raise excs.Error(f'Invalid value for `n`: {n}')

        # Retrieve the table history components from the catalog
        tbl_id = self._id
        # Collect an extra version, if available, to allow for computation of the first version's schema change
        vers_list = Catalog.get().collect_tbl_history(tbl_id, n + 1)

        # Construct the metadata change description dictionary
        md_list = [(vers_md.version_md.version, vers_md.schema_version_md.columns) for vers_md in vers_list]
        md_dict = MetadataUtils._create_md_change_dict(md_list)

        # Construct report lines
        if len(vers_list) > n:
            assert len(vers_list) == n + 1
            over_count = 1
        else:
            over_count = 0

        metadata_dicts: list[VersionMetadata] = []
        for vers_md in vers_list[0 : len(vers_list) - over_count]:
            version = vers_md.version_md.version
            schema_change = md_dict.get(version, None)
            update_status = vers_md.version_md.update_status
            if update_status is None:
                update_status = UpdateStatus()
            change_type: Literal['schema', 'data'] = 'schema' if schema_change is not None else 'data'
            rcs = update_status.row_count_stats + update_status.cascade_row_count_stats
            metadata_dicts.append(
                VersionMetadata(
                    version=version,
                    created_at=datetime.datetime.fromtimestamp(vers_md.version_md.created_at, tz=datetime.timezone.utc),
                    user=vers_md.version_md.user,
                    change_type=change_type,
                    inserts=rcs.ins_rows,
                    updates=rcs.upd_rows,
                    deletes=rcs.del_rows,
                    errors=rcs.num_excs,
                    schema_change=schema_change,
                )
            )

        return metadata_dicts

    def history(self, n: int | None = None) -> pd.DataFrame:
        """
        Returns a human-readable report about versions of this table.

        `history()` is intended for human-readable output of version metadata; for programmatic access,
        use [`get_versions()`][pixeltable.Table.get_versions] instead.

        Args:
            n: if specified, will return at most `n` versions

        Returns:
            A report with information about each version, one per row, most recent first.

        Examples:
            Report all versions of the table:

            >>> tbl.history()

            Report only the most recent 5 changes to the table:

            >>> tbl.history(n=5)
        """
        versions = self.get_versions(n)
        assert len(versions) > 0
        return pd.DataFrame([list(v.values()) for v in versions], columns=list(versions[0].keys()))

    def __check_mutable(self, op_descr: str) -> None:
        if self._tbl_version_path.is_replica():
            raise excs.Error(f'{self._display_str()}: Cannot {op_descr} a replica.')
        if self._tbl_version_path.is_snapshot():
            raise excs.Error(f'{self._display_str()}: Cannot {op_descr} a snapshot.')
