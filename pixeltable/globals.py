from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, TypedDict, Union

import pandas as pd
import pydantic
from pandas.io.formats.style import Styler

from pixeltable import DataFrame, catalog, exceptions as excs, exprs, func, share, type_system as ts
from pixeltable.catalog import Catalog, TableVersionPath
from pixeltable.catalog.insertable_table import OnErrorParameter
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.io.table_data_conduit import DFTableDataConduit, TableDataConduit
from pixeltable.iterators import ComponentIterator

if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]

    RowData = list[dict[str, Any]]
    TableDataSource = Union[
        str,
        os.PathLike,
        Path,  # OS paths, filenames, URLs
        Iterable[dict[str, Any]],  # dictionaries of values
        Iterable[pydantic.BaseModel],  # Pydantic model instances
        catalog.Table,  # Pixeltable Table
        DataFrame,  # Pixeltable DataFrame
        pd.DataFrame,  # pandas DataFrame
        datasets.Dataset,
        datasets.DatasetDict,  # Huggingface datasets
    ]


_logger = logging.getLogger('pixeltable')


def init(config_overrides: dict[str, Any] | None = None) -> None:
    """Initializes the Pixeltable environment."""
    if config_overrides is None:
        config_overrides = {}
    Config.init(config_overrides)
    _ = Catalog.get()


def create_table(
    path: str,
    schema: dict[str, Any] | None = None,
    *,
    source: TableDataSource | None = None,
    source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
    schema_overrides: dict[str, Any] | None = None,
    create_default_idxs: bool = True,
    on_error: Literal['abort', 'ignore'] = 'abort',
    primary_key: str | list[str] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    extra_args: dict[str, Any] | None = None,  # Additional arguments to data source provider
) -> catalog.Table:
    """Create a new base table. Exactly one of `schema` or `source` must be provided.

    If a `schema` is provided, then an empty table will be created with the specified schema.

    If a `source` is provided, then Pixeltable will attempt to infer a data source format and table schema from the
    contents of the specified data, and the data will be imported from the specified source into the new table. The
    source format and/or schema can be specified directly via the `source_format` and `schema_overrides` parameters.

    Args:
        path: Pixeltable path (qualified name) of the table, such as `'my_table'` or `'my_dir.my_subdir.my_table'`.
        schema: Schema for the new table, mapping column names to Pixeltable types.
        source: A data source (file, URL, Table, DataFrame, or list of rows) to import from.
        source_format: Must be used in conjunction with a `source`.
            If specified, then the given format will be used to read the source data. (Otherwise,
            Pixeltable will attempt to infer the format from the source data.)
        schema_overrides: Must be used in conjunction with a `source`.
            If specified, then columns in `schema_overrides` will be given the specified types.
            (Pixeltable will attempt to infer the types of any columns not specified.)
        create_default_idxs: If True, creates a B-tree index on every scalar and media column that is not computed,
            except for boolean columns.
        on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
            invalid media file (such as a corrupt image) for one of the inserted rows.

            - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
            - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
                with errors will have a `None` value for that cell, with information about the error stored in the
                corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
        primary_key: An optional column name or list of column names to use as the primary key(s) of the
            table.
        num_retained_versions: Number of versions of the table to retain.
        comment: An optional comment; its meaning is user-defined.
        media_validation: Media validation policy for the table.

            - `'on_read'`: validate media files at query time
            - `'on_write'`: validate media files during insert/update operations
        if_exists: Determines the behavior if a table already exists at the specified path location.

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing table handle
            - `'replace'`: if the existing table has no views or snapshots, drop and replace it with a new one;
                raise an error if the existing table has views or snapshots
            - `'replace_force'`: drop the existing table and all its views and snapshots, and create a new one
        extra_args: Must be used in conjunction with a `source`. If specified, then additional arguments will be
            passed along to the source data provider.

    Returns:
        A handle to the newly created table, or to an already existing table at the path when `if_exists='ignore'`.
            Please note the schema of the existing table may not match the schema provided in the call.

    Raises:
        Error: if

            - the path is invalid, or
            - the path already exists and `if_exists='error'`, or
            - the path already exists and is not a table, or
            - an error occurs while attempting to create the table, or
            - an error occurs while attempting to import data from the source.

    Examples:
        Create a table with an int and a string column:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String})

        Create a table from a select statement over an existing table `orig_table` (this will create a new table
        containing the exact contents of the query):

        >>> tbl1 = pxt.get_table('orig_table')
        ... tbl2 = pxt.create_table('new_table', tbl1.where(tbl1.col1 < 10).select(tbl1.col2))

        Create a table if it does not already exist, otherwise get the existing table:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String}, if_exists='ignore')

        Create a table with an int and a float column, and replace any existing table:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.Float}, if_exists='replace')

        Create a table from a CSV file:

        >>> tbl = pxt.create_table('my_table', source='data.csv')
    """
    from pixeltable.io.table_data_conduit import UnkTableDataConduit
    from pixeltable.io.utils import normalize_primary_key_parameter

    if (schema is None) == (source is None):
        raise excs.Error('Either a `schema` or a `source` must be provided (but not both)')

    if schema is not None and (len(schema) == 0 or not isinstance(schema, dict)):
        raise excs.Error('`schema` must be a non-empty dictionary')

    path_obj = catalog.Path.parse(path)
    if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
    media_validation_ = catalog.MediaValidation.validated(media_validation, 'media_validation')
    primary_key: list[str] | None = normalize_primary_key_parameter(primary_key)
    data_source: TableDataConduit | None = None
    if source is not None:
        if isinstance(source, str) and source.strip().startswith('pxt://'):
            raise excs.Error(
                'create_table(): Creating a table directly from a cloud URI is not supported.'
                ' Please replicate the table locally first using `pxt.replicate()`:\n'
                "replica_tbl = pxt.replicate('pxt://path/to/remote_table', 'local_replica_name')\n"
                "pxt.create_table('new_table_name', source=replica_tbl)"
            )
        tds = UnkTableDataConduit(source, source_format=source_format, extra_fields=extra_args)
        tds.check_source_format()
        data_source = tds.specialize()
        src_schema_overrides: dict[str, ts.ColumnType] = {}
        if schema_overrides is not None:
            for col_name, py_type in schema_overrides.items():
                col_type = ts.ColumnType.normalize_type(py_type, nullable_default=True, allow_builtin_types=False)
                if col_type is None:
                    raise excs.Error(f'Invalid type for column {col_name!r} in `schema_overrides`: {py_type}')
                src_schema_overrides[col_name] = col_type
        data_source.src_schema_overrides = src_schema_overrides
        data_source.src_pk = primary_key
        data_source.infer_schema()
        schema = data_source.pxt_schema
        primary_key = data_source.pxt_pk
        is_direct_df = data_source.is_direct_df()
    else:
        is_direct_df = False

    if len(schema) == 0 or not isinstance(schema, dict):
        raise excs.Error(
            'Unable to create a proper schema from supplied `source`. Please use appropriate `schema_overrides`.'
        )

    tbl, was_created = Catalog.get().create_table(
        path_obj,
        schema,
        if_exists=if_exists_,
        primary_key=primary_key,
        comment=comment,
        media_validation=media_validation_,
        num_retained_versions=num_retained_versions,
        create_default_idxs=create_default_idxs,
    )

    # TODO: combine data loading with table creation into a single transaction
    if was_created:
        fail_on_exception = OnErrorParameter.fail_on_exception(on_error)
        if isinstance(data_source, DFTableDataConduit):
            df = data_source.pxt_df
            with Catalog.get().begin_xact(tbl=tbl._tbl_version_path, for_write=True, lock_mutable_tree=True):
                tbl._tbl_version.get().insert(None, df, fail_on_exception=fail_on_exception)
        elif data_source is not None and not is_direct_df:
            tbl.insert_table_data_source(data_source=data_source, fail_on_exception=fail_on_exception)

    return tbl


def create_view(
    path: str,
    base: catalog.Table | DataFrame,
    *,
    additional_columns: dict[str, Any] | None = None,
    is_snapshot: bool = False,
    create_default_idxs: bool = False,
    iterator: tuple[type[ComponentIterator], dict[str, Any]] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> catalog.Table | None:
    """Create a view of an existing table object (which itself can be a view or a snapshot or a base table).

    Args:
        path: A name for the view; can be either a simple name such as `my_view`, or a pathname such as
            `dir1.my_view`.
        base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
            base the view on.
        additional_columns: If specified, will add these columns to the view once it is created. The format
            of the `additional_columns` parameter is identical to the format of the `schema_or_df` parameter in
            [`create_table`][pixeltable.create_table].
        is_snapshot: Whether the view is a snapshot. Setting this to `True` is equivalent to calling
            [`create_snapshot`][pixeltable.create_snapshot].
        create_default_idxs: Whether to create default indexes on the view's columns (the base's columns are excluded).
            Cannot be `True` for snapshots.
        iterator: The iterator to use for this view. If specified, then this view will be a one-to-many view of
            the base table.
        num_retained_versions: Number of versions of the view to retain.
        comment: Optional comment for the view.
        media_validation: Media validation policy for the view.

            - `'on_read'`: validate media files at query time
            - `'on_write'`: validate media files during insert/update operations
        if_exists: Directive regarding how to handle if the path already exists.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing view handle
            - `'replace'`: if the existing view has no dependents, drop and replace it with a new one
            - `'replace_force'`: drop the existing view and all its dependents, and create a new one

    Returns:
        A handle to the [`Table`][pixeltable.Table] representing the newly created view. If the path already
            exists and `if_exists='ignore'`, returns a handle to the existing view. Please note the schema
            or the base of the existing view may not match those provided in the call.

    Raises:
        Error: if

            - the path is invalid, or
            - the path already exists and `if_exists='error'`, or
            - the path already exists and is not a view, or
            - an error occurs while attempting to create the view.

    Examples:
        Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 10:

        >>> tbl = pxt.get_table('my_table')
        ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 10))

        Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 10,
        and if it not already exist. Otherwise, get the existing view named `my_view`:

        >>> tbl = pxt.get_table('my_table')
        ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 10), if_exists='ignore')

        Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 100,
        and replace any existing view named `my_view`:

        >>> tbl = pxt.get_table('my_table')
        ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 100), if_exists='replace_force')
    """
    if is_snapshot and create_default_idxs is True:
        raise excs.Error('Cannot create default indexes on a snapshot')
    tbl_version_path: TableVersionPath
    select_list: list[tuple[exprs.Expr, str | None]] | None = None
    where: exprs.Expr | None = None
    if isinstance(base, catalog.Table):
        tbl_version_path = base._tbl_version_path
        sample_clause = None
    elif isinstance(base, DataFrame):
        base._validate_mutable_op_sequence('create_view', allow_select=True)
        tbl_version_path = base._from_clause.tbls[0]
        where = base.where_clause
        sample_clause = base.sample_clause
        select_list = base.select_list
        if sample_clause is not None and not is_snapshot and not sample_clause.is_repeatable:
            raise excs.Error('Non-snapshot views cannot be created with non-fractional or stratified sampling')
    else:
        raise excs.Error('`base` must be an instance of `Table` or `DataFrame`')
    assert isinstance(base, (catalog.Table, DataFrame))

    if tbl_version_path.is_replica():
        raise excs.Error('Cannot create a view or snapshot on top of a replica')

    path_obj = catalog.Path.parse(path)
    if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
    media_validation_ = catalog.MediaValidation.validated(media_validation, 'media_validation')

    if additional_columns is None:
        additional_columns = {}
    else:
        # additional columns should not be in the base table
        for col_name in additional_columns:
            if col_name in [c.name for c in tbl_version_path.columns()]:
                raise excs.Error(
                    f'Column {col_name!r} already exists in the base table '
                    f'{tbl_version_path.get_column(col_name).get_tbl().name}.'
                )

    return Catalog.get().create_view(
        path_obj,
        tbl_version_path,
        select_list=select_list,
        where=where,
        sample_clause=sample_clause,
        additional_columns=additional_columns,
        is_snapshot=is_snapshot,
        create_default_idxs=create_default_idxs,
        iterator=iterator,
        num_retained_versions=num_retained_versions,
        comment=comment,
        media_validation=media_validation_,
        if_exists=if_exists_,
    )


def create_snapshot(
    path_str: str,
    base: catalog.Table | DataFrame,
    *,
    additional_columns: dict[str, Any] | None = None,
    iterator: tuple[type[ComponentIterator], dict[str, Any]] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> catalog.Table | None:
    """Create a snapshot of an existing table object (which itself can be a view or a snapshot or a base table).

    Args:
        path_str: A name for the snapshot; can be either a simple name such as `my_snapshot`, or a pathname such as
            `dir1.my_snapshot`.
        base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
            base the snapshot on.
        additional_columns: If specified, will add these columns to the snapshot once it is created. The format
            of the `additional_columns` parameter is identical to the format of the `schema_or_df` parameter in
            [`create_table`][pixeltable.create_table].
        iterator: The iterator to use for this snapshot. If specified, then this snapshot will be a one-to-many view of
            the base table.
        num_retained_versions: Number of versions of the view to retain.
        comment: Optional comment for the snapshot.
        media_validation: Media validation policy for the snapshot.

            - `'on_read'`: validate media files at query time
            - `'on_write'`: validate media files during insert/update operations
        if_exists: Directive regarding how to handle if the path already exists.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing snapshot handle
            - `'replace'`: if the existing snapshot has no dependents, drop and replace it with a new one
            - `'replace_force'`: drop the existing snapshot and all its dependents, and create a new one

    Returns:
        A handle to the [`Table`][pixeltable.Table] representing the newly created snapshot.
            Please note the schema or base of the existing snapshot may not match those provided in the call.

    Raises:
        Error: if

            - the path is invalid, or
            - the path already exists and `if_exists='error'`, or
            - the path already exists and is not a snapshot, or
            - an error occurs while attempting to create the snapshot.

    Examples:
        Create a snapshot `my_snapshot` of a table `my_table`:

        >>> tbl = pxt.get_table('my_table')
        ... snapshot = pxt.create_snapshot('my_snapshot', tbl)

        Create a snapshot `my_snapshot` of a view `my_view` with additional int column `col3`,
        if `my_snapshot` does not already exist:

        >>> view = pxt.get_table('my_view')
        ... snapshot = pxt.create_snapshot(
        ...     'my_snapshot', view, additional_columns={'col3': pxt.Int}, if_exists='ignore'
        ... )

        Create a snapshot `my_snapshot` on a table `my_table`, and replace any existing snapshot named `my_snapshot`:

        >>> tbl = pxt.get_table('my_table')
        ... snapshot = pxt.create_snapshot('my_snapshot', tbl, if_exists='replace_force')
    """
    return create_view(
        path_str,
        base,
        additional_columns=additional_columns,
        iterator=iterator,
        is_snapshot=True,
        num_retained_versions=num_retained_versions,
        comment=comment,
        media_validation=media_validation,
        if_exists=if_exists,
    )


def publish(
    source: str | catalog.Table,
    destination_uri: str,
    bucket_name: str | None = None,
    access: Literal['public', 'private'] = 'private',
) -> None:
    """
    Publishes a replica of a local Pixeltable table to Pixeltable cloud. A given table can be published to at most one
    URI per Pixeltable cloud database.

    Args:
        source: Path or table handle of the local table to be published.
        destination_uri: Remote URI where the replica will be published, such as `'pxt://org_name/my_dir/my_table'`.
        bucket_name: The name of the bucket to use to store replica's data. The bucket must be registered with
            Pixeltable cloud. If no `bucket_name` is provided, the default storage bucket for the destination
            database will be used.
        access: Access control for the replica.

            - `'public'`: Anyone can access this replica.
            - `'private'`: Only the host organization can access.
    """
    if not destination_uri.startswith('pxt://'):
        raise excs.Error("`destination_uri` must be a remote Pixeltable URI with the prefix 'pxt://'")

    if isinstance(source, str):
        source = get_table(source)

    share.push_replica(destination_uri, source, bucket_name, access)


def replicate(remote_uri: str, local_path: str) -> catalog.Table:
    """
    Retrieve a replica from Pixeltable cloud as a local table. This will create a full local copy of the replica in a
    way that preserves the table structure of the original source data. Once replicated, the local table can be
    queried offline just as any other Pixeltable table.

    Args:
        remote_uri: Remote URI of the table to be replicated, such as `'pxt://org_name/my_dir/my_table'` or
            `'pxt://org_name/my_dir/my_table:5'` (with version 5).
        local_path: Local table path where the replica will be created, such as `'my_new_dir.my_new_tbl'`. It can be
            the same or different from the cloud table name.

    Returns:
        A handle to the newly created local replica table.
    """
    if not remote_uri.startswith('pxt://'):
        raise excs.Error("`remote_uri` must be a remote Pixeltable URI with the prefix 'pxt://'")

    return share.pull_replica(local_path, remote_uri)


def get_table(path: str, if_not_exists: Literal['error', 'ignore'] = 'error') -> catalog.Table | None:
    """Get a handle to an existing table, view, or snapshot.

    Args:
        path: Path to the table.
        if_not_exists: Directive regarding how to handle if the path does not exist.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return `None`

    Returns:
        A handle to the [`Table`][pixeltable.Table].

    Raises:
        Error: If the path does not exist or does not designate a table object.

    Examples:
        Get handle for a table in the top-level directory:

        >>> tbl = pxt.get_table('my_table')

        For a table in a subdirectory:

        >>> tbl = pxt.get_table('subdir.my_table')

        Handles to views and snapshots are retrieved in the same way:

        >>> tbl = pxt.get_table('my_snapshot')

        Get a handle to a specific version of a table:

        >>> tbl = pxt.get_table('my_table:722')
    """
    if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
    path_obj = catalog.Path.parse(path, allow_versioned_path=True)
    tbl = Catalog.get().get_table(path_obj, if_not_exists_)
    return tbl


def move(
    path: str,
    new_path: str,
    *,
    if_exists: Literal['error', 'ignore'] = 'error',
    if_not_exists: Literal['error', 'ignore'] = 'error',
) -> None:
    """Move a schema object to a new directory and/or rename a schema object.

    Args:
        path: absolute path to the existing schema object.
        new_path: absolute new path for the schema object.
        if_exists: Directive regarding how to handle if a schema object already exists at the new path.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return
        if_not_exists: Directive regarding how to handle if the source path does not exist.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return

    Raises:
        Error: If path does not exist or new_path already exists.

    Examples:
        Move a table to a different directory:

        >>>> pxt.move('dir1.my_table', 'dir2.my_table')

        Rename a table:

        >>>> pxt.move('dir1.my_table', 'dir1.new_name')
    """
    if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
    if if_exists_ not in (catalog.IfExistsParam.ERROR, catalog.IfExistsParam.IGNORE):
        raise excs.Error("`if_exists` must be one of 'error' or 'ignore'")
    if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
    if path == new_path:
        raise excs.Error('move(): source and destination cannot be identical')
    path_obj, new_path_obj = catalog.Path.parse(path), catalog.Path.parse(new_path)
    if path_obj.is_ancestor(new_path_obj):
        raise excs.Error(f'move(): cannot move {path!r} into its own subdirectory')
    Catalog.get().move(path_obj, new_path_obj, if_exists_, if_not_exists_)


def drop_table(
    table: str | catalog.Table, force: bool = False, if_not_exists: Literal['error', 'ignore'] = 'error'
) -> None:
    """Drop a table, view, snapshot, or replica.

    Args:
        table: Fully qualified name or table handle of the table to be dropped; or a remote URI of a cloud replica to
            be deleted.
        force: If `True`, will also drop all views and sub-views of this table.
        if_not_exists: Directive regarding how to handle if the path does not exist.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return

    Raises:
        Error: if the qualified name

            - is invalid, or
            - does not exist and `if_not_exists='error'`, or
            - does not designate a table object, or
            - designates a table object but has dependents and `force=False`.

    Examples:
        Drop a table by its fully qualified name:
        >>> pxt.drop_table('subdir.my_table')

        Drop a table by its handle:
        >>> t = pxt.get_table('subdir.my_table')
        ... pxt.drop_table(t)

        Drop a table if it exists, otherwise do nothing:
        >>> pxt.drop_table('subdir.my_table', if_not_exists='ignore')

        Drop a table and all its dependents:
        >>> pxt.drop_table('subdir.my_table', force=True)
    """
    tbl_path: str
    if isinstance(table, catalog.Table):
        # if we're dropping a table by handle, we first need to get the current path, then drop the S lock on
        # the Table record, and then get X locks in the correct order (first containing directory, then table)
        with Catalog.get().begin_xact(for_write=False):
            tbl_path = table._path()
    else:
        assert isinstance(table, str)
        tbl_path = table

    if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')

    if tbl_path.startswith('pxt://'):
        # Remote table
        if force:
            raise excs.Error('Cannot use `force=True` with a cloud replica URI.')
        # TODO: Handle if_not_exists properly
        share.delete_replica(tbl_path)
    else:
        # Local table
        path_obj = catalog.Path.parse(tbl_path)
        Catalog.get().drop_table(path_obj, force=force, if_not_exists=if_not_exists_)


def get_dir_contents(dir_path: str = '', recursive: bool = True) -> 'DirContents':
    """Get the contents of a Pixeltable directory.

    Args:
        dir_path: Path to the directory. Defaults to the root directory.
        recursive: If `False`, returns only those tables and directories that are directly contained in specified
            directory; if `True`, returns all tables and directories that are descendants of the specified directory,
            recursively.

    Returns:
        A [`DirContents`][pixeltable.DirContents] object representing the contents of the specified directory.

    Raises:
        Error: If the path does not exist or does not designate a directory.

    Examples:
        Get contents of top-level directory:

        >>> pxt.get_dir_contents()

        Get contents of 'dir1':

        >>> pxt.get_dir_contents('dir1')
    """
    path_obj = catalog.Path.parse(dir_path, allow_empty_path=True)
    catalog_entries = Catalog.get().get_dir_contents(path_obj, recursive=recursive)
    dirs: list[str] = []
    tables: list[str] = []
    _assemble_dir_contents(dir_path, catalog_entries, dirs, tables)
    dirs.sort()
    tables.sort()
    return DirContents(dirs=dirs, tables=tables)


def _assemble_dir_contents(
    dir_path: str, catalog_entries: dict[str, Catalog.DirEntry], dirs: list[str], tables: list[str]
) -> None:
    for name, entry in catalog_entries.items():
        if name.startswith('_'):
            continue  # Skip system paths
        path = f'{dir_path}.{name}' if len(dir_path) > 0 else name
        if entry.dir is not None:
            dirs.append(path)
            if entry.dir_entries is not None:
                _assemble_dir_contents(path, entry.dir_entries, dirs, tables)
        else:
            assert entry.table is not None
            assert not entry.dir_entries
            tables.append(path)


def list_tables(dir_path: str = '', recursive: bool = True) -> list[str]:
    """List the [`Table`][pixeltable.Table]s in a directory.

    Args:
        dir_path: Path to the directory. Defaults to the root directory.
        recursive: If `False`, returns only those tables that are directly contained in specified directory; if
            `True`, returns all tables that are descendants of the specified directory, recursively.

    Returns:
        A list of [`Table`][pixeltable.Table] paths.

    Raises:
        Error: If the path does not exist or does not designate a directory.

    Examples:
        List tables in top-level directory:

        >>> pxt.list_tables()

        List tables in 'dir1':

        >>> pxt.list_tables('dir1')
    """
    return _list_tables(dir_path, recursive=recursive, allow_system_paths=False)


def _list_tables(dir_path: str = '', recursive: bool = True, allow_system_paths: bool = False) -> list[str]:
    path_obj = catalog.Path.parse(dir_path, allow_empty_path=True, allow_system_path=allow_system_paths)
    contents = Catalog.get().get_dir_contents(path_obj, recursive=recursive)
    return [str(p) for p in _extract_paths(contents, parent=path_obj, entry_type=catalog.Table)]


def create_dir(
    path: str, *, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error', parents: bool = False
) -> catalog.Dir | None:
    """Create a directory.

    Args:
        path: Path to the directory.
        if_exists: Directive regarding how to handle if the path already exists.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing directory handle
            - `'replace'`: if the existing directory is empty, drop it and create a new one
            - `'replace_force'`: drop the existing directory and all its children, and create a new one
        parents: Create missing parent directories.

    Returns:
        A handle to the newly created directory, or to an already existing directory at the path when
            `if_exists='ignore'`. Please note the existing directory may not be empty.

    Raises:
        Error: If

            - the path is invalid, or
            - the path already exists and `if_exists='error'`, or
            - the path already exists and is not a directory, or
            - an error occurs while attempting to create the directory.

    Examples:
        >>> pxt.create_dir('my_dir')

        Create a subdirectory:

        >>> pxt.create_dir('my_dir.sub_dir')

        Create a subdirectory only if it does not already exist, otherwise do nothing:

        >>> pxt.create_dir('my_dir.sub_dir', if_exists='ignore')

        Create a directory and replace if it already exists:

        >>> pxt.create_dir('my_dir', if_exists='replace_force')

        Create a subdirectory along with its ancestors:

        >>> pxt.create_dir('parent1.parent2.sub_dir', parents=True)
    """
    path_obj = catalog.Path.parse(path)
    if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
    return Catalog.get().create_dir(path_obj, if_exists=if_exists_, parents=parents)


def drop_dir(path: str, force: bool = False, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
    """Remove a directory.

    Args:
        path: Name or path of the directory.
        force: If `True`, will also drop all tables and subdirectories of this directory, recursively, along
            with any views or snapshots that depend on any of the dropped tables.
        if_not_exists: Directive regarding how to handle if the path does not exist.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return

    Raises:
        Error: If the path

            - is invalid, or
            - does not exist and `if_not_exists='error'`, or
            - is not designate a directory, or
            - is a direcotory but is not empty and `force=False`.

    Examples:
        Remove a directory, if it exists and is empty:
        >>> pxt.drop_dir('my_dir')

        Remove a subdirectory:

        >>> pxt.drop_dir('my_dir.sub_dir')

        Remove an existing directory if it is empty, but do nothing if it does not exist:

        >>> pxt.drop_dir('my_dir.sub_dir', if_not_exists='ignore')

        Remove an existing directory and all its contents:

        >>> pxt.drop_dir('my_dir', force=True)
    """
    path_obj = catalog.Path.parse(path)  # validate format
    if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
    Catalog.get().drop_dir(path_obj, if_not_exists=if_not_exists_, force=force)


def ls(path: str = '') -> pd.DataFrame:
    """
    List the contents of a Pixeltable directory.

    This function returns a Pandas DataFrame representing a human-readable listing of the specified directory,
    including various attributes such as version and base table, as appropriate.

    To get a programmatic list of the directory's contents, use [get_dir_contents()][pixeltable.get_dir_contents]
    instead.
    """
    from pixeltable.catalog import retry_loop
    from pixeltable.metadata import schema

    cat = Catalog.get()
    path_obj = catalog.Path.parse(path, allow_empty_path=True)
    dir_entries = cat.get_dir_contents(path_obj)

    @retry_loop(for_write=False)
    def op() -> list[list[str]]:
        rows: list[list[str]] = []
        for name, entry in dir_entries.items():
            if name.startswith('_'):
                continue
            if entry.dir is not None:
                kind = 'dir'
                version = ''
                base = ''
            else:
                assert entry.table is not None
                assert isinstance(entry.table, schema.Table)
                tbl = cat.get_table_by_id(entry.table.id)
                md = tbl.get_metadata()
                base = md['base'] or ''
                if base.startswith('_'):
                    base = '<anonymous base table>'
                if md['is_replica']:
                    kind = 'replica'
                elif md['is_snapshot']:
                    kind = 'snapshot'
                elif md['is_view']:
                    kind = 'view'
                else:
                    kind = 'table'
                version = '' if kind == 'snapshot' else str(md['version'])
            rows.append([name, kind, version, base])
        return rows

    rows = op()

    rows = sorted(rows, key=lambda x: x[0])
    df = pd.DataFrame(
        {
            'Name': [row[0] for row in rows],
            'Kind': [row[1] for row in rows],
            'Version': [row[2] for row in rows],
            'Base': [row[3] for row in rows],
        },
        index=([''] * len(rows)),
    )
    return df


def _extract_paths(
    dir_entries: dict[str, Catalog.DirEntry], parent: catalog.Path, entry_type: type[catalog.SchemaObject] | None = None
) -> list[catalog.Path]:
    """Convert nested dir_entries structure to a flattened list of paths."""
    matches: list[str]
    if entry_type is None:
        matches = list(dir_entries.keys())
    elif entry_type is catalog.Dir:
        matches = [name for name, entry in dir_entries.items() if entry.dir is not None]
    else:
        matches = [name for name, entry in dir_entries.items() if entry.table is not None]

    # Filter out system paths
    matches = [name for name in matches if catalog.is_valid_identifier(name)]
    result = [parent.append(name) for name in matches]

    for name, entry in dir_entries.items():
        if len(entry.dir_entries) > 0 and catalog.is_valid_identifier(name):
            result.extend(_extract_paths(entry.dir_entries, parent=parent.append(name), entry_type=entry_type))
    return result


def list_dirs(path: str = '', recursive: bool = True) -> list[str]:
    """List the directories in a directory.

    Args:
        path: Name or path of the directory.
        recursive: If `True`, lists all descendants of this directory recursively.

    Returns:
        List of directory paths.

    Raises:
        Error: If `path_str` does not exist or does not designate a directory.

    Examples:
        >>> cl.list_dirs('my_dir', recursive=True)
        ['my_dir', 'my_dir.sub_dir1']
    """
    path_obj = catalog.Path.parse(path, allow_empty_path=True)  # validate format
    cat = Catalog.get()
    contents = cat.get_dir_contents(path_obj, recursive=recursive)
    return [str(p) for p in _extract_paths(contents, parent=path_obj, entry_type=catalog.Dir)]


def list_functions() -> Styler:
    """Returns information about all registered functions.

    Returns:
        Pandas DataFrame with columns 'Path', 'Name', 'Parameters', 'Return Type', 'Is Agg', 'Library'
    """
    functions = func.FunctionRegistry.get().list_functions()
    paths = ['.'.join(f.self_path.split('.')[:-1]) for f in functions]
    names = [f.name for f in functions]
    params = [
        ', '.join(
            [param_name + ': ' + str(param_type) for param_name, param_type in f.signatures[0].parameters.items()]
        )
        for f in functions
    ]
    pd_df = pd.DataFrame(
        {
            'Path': paths,
            'Function Name': names,
            'Parameters': params,
            'Return Type': [str(f.signatures[0].get_return_type()) for f in functions],
        }
    )
    pd_df = pd_df.style.set_properties(None, **{'text-align': 'left'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    )  # center-align headings
    return pd_df.hide(axis='index')


def tools(*args: func.Function | func.tools.Tool) -> func.tools.Tools:
    """
    Specifies a collection of UDFs to be used as LLM tools. Pixeltable allows any UDF to be used as an input into an
    LLM tool-calling API. To use one or more UDFs as tools, wrap them in a `pxt.tools` call and pass the return value
    to an LLM API.

    The UDFs can be specified directly or wrapped inside a [pxt.tool()][pixeltable.tool] invocation. If a UDF is
    specified directly, the tool name will be the (unqualified) UDF name, and the tool description will consist of the
    entire contents of the UDF docstring. If a UDF is wrapped in a `pxt.tool()` invocation, then the name and/or
    description may be customized.

    Args:
        args: The UDFs to use as tools.

    Returns:
        A `Tools` instance that can be passed to an LLM tool-calling API or invoked to generate tool results.

    Examples:
        Create a tools instance with a single UDF:

        >>> tools = pxt.tools(stock_price)

        Create a tools instance with several UDFs:

        >>> tools = pxt.tools(stock_price, weather_quote)

        Create a tools instance, some of whose UDFs have customized metadata:

        >>> tools = pxt.tools(
        ...     stock_price,
        ...     pxt.tool(weather_quote, description='Returns information about the weather in a particular location.'),
        ...     pxt.tool(traffic_quote, name='traffic_conditions'),
        ... )
    """
    return func.tools.Tools(tools=[arg if isinstance(arg, func.tools.Tool) else tool(arg) for arg in args])


def tool(fn: func.Function, name: str | None = None, description: str | None = None) -> func.tools.Tool:
    """
    Specifies a Pixeltable UDF to be used as an LLM tool with customizable metadata. See the documentation for
    [pxt.tools()][pixeltable.tools] for more details.

    Args:
        fn: The UDF to use as a tool.
        name: The name of the tool. If not specified, then the unqualified name of the UDF will be used by default.
        description: The description of the tool. If not specified, then the entire contents of the UDF docstring
            will be used by default.

    Returns:
        A `Tool` instance that can be passed to an LLM tool-calling API.
    """
    if isinstance(fn, func.AggregateFunction):
        raise excs.Error('Aggregator UDFs cannot be used as tools')

    return func.tools.Tool(fn=fn, name=name, description=description)


def configure_logging(
    *, to_stdout: bool | None = None, level: int | None = None, add: str | None = None, remove: str | None = None
) -> None:
    """Configure logging.

    Args:
        to_stdout: if True, also log to stdout
        level: default log level
        add: comma-separated list of 'module name:log level' pairs; ex.: add='video:10'
        remove: comma-separated list of module names
    """
    return Env.get().configure_logging(to_stdout=to_stdout, level=level, add=add, remove=remove)


def array(elements: Iterable) -> exprs.Expr:
    return exprs.Expr.from_array(elements)


class DirContents(TypedDict):
    """
    Represents the contents of a Pixeltable directory.
    """

    dirs: list[str]
    """List of directory paths contained in this directory."""
    tables: list[str]
    """List of table paths contained in this directory."""
