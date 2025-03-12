import logging
import urllib.parse
from typing import Any, Iterable, Literal, Optional, Union, cast
from uuid import UUID

import pandas as pd
from pandas.io.formats.style import Styler

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
from pixeltable import DataFrame, catalog, func, share
from pixeltable.catalog import Catalog, IfExistsParam, IfNotExistsParam
from pixeltable.dataframe import DataFrameResultSet
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.utils.filecache import FileCache

_logger = logging.getLogger('pixeltable')


def init() -> None:
    """Initializes the Pixeltable environment."""
    _ = Catalog.get()


def _handle_path_collision(
    path: str, expected_obj_type: type[catalog.SchemaObject], expected_snapshot: bool, if_exists: catalog.IfExistsParam
) -> Optional[catalog.SchemaObject]:
    cat = Catalog.get()
    path_obj = catalog.Path(path)
    obj, _, _ = cat.prepare_dir_op(add_dir_path=str(path_obj.parent), add_name=path_obj.name)

    if if_exists == catalog.IfExistsParam.ERROR and obj is not None:
        raise excs.Error(f'Path {path!r} is an existing {type(obj)._display_name()}')
    else:
        is_snapshot = isinstance(obj, catalog.View) and obj._tbl_version_path.is_snapshot()
        if obj is not None and (not isinstance(obj, expected_obj_type) or (expected_snapshot and not is_snapshot)):
            obj_type_str = 'snapshot' if expected_snapshot else expected_obj_type._display_name()
            raise excs.Error(
                f'Path {path!r} already exists but is not a {obj_type_str}. Cannot {if_exists.name.lower()} it.'
            )

    if obj is None:
        return None
    if if_exists == IfExistsParam.IGNORE:
        return obj

    # drop the existing schema object
    if isinstance(obj, catalog.Dir):
        dir_contents = cat.get_dir_contents(obj._id)
        if len(dir_contents) > 0 and if_exists == IfExistsParam.REPLACE:
            raise excs.Error(
                f'Directory {path!r} already exists and is not empty. Use `if_exists="replace_force"` to replace it.'
            )
        _drop_dir(obj._id, path, force=True)
    else:
        assert isinstance(obj, catalog.Table)
        _drop_table(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
    return None

    # obj: Optional[catalog.SchemaObject]
    # if if_exists == catalog.IfExistsParam.ERROR:
    #     _ = cat.get_schema_object(path, raise_if_exists=True)
    #     obj = None
    # else:
    #     obj = cat.get_schema_object(path)
    #     is_snapshot = isinstance(obj, catalog.View) and obj._tbl_version_path.is_snapshot()
    #     if obj is not None and (not isinstance(obj, expected_obj_type) or (expected_snapshot and not is_snapshot)):
    #         obj_type_str = 'snapshot' if expected_snapshot else expected_obj_type._display_name()
    #         raise excs.Error(
    #             f'Path {path!r} already exists but is not a {obj_type_str}. Cannot {if_exists.name.lower()} it.'
    #         )
    # if obj is None:
    #     return None
    #
    # if if_exists == IfExistsParam.IGNORE:
    #     return obj
    #
    # # drop the existing schema object
    # if isinstance(obj, catalog.Dir):
    #     dir_contents = cat.get_dir_contents(obj._id)
    #     if len(dir_contents) > 0 and if_exists == IfExistsParam.REPLACE:
    #         raise excs.Error(
    #             f'Directory {path!r} already exists and is not empty. Use `if_exists="replace_force"` to replace it.'
    #         )
    #     _drop_dir(obj._id, path, force=True)
    # else:
    #     assert isinstance(obj, catalog.Table)
    #     _drop_table(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
    # return None


def create_table(
    path_str: str,
    schema_or_df: Union[dict[str, Any], DataFrame],
    *,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> catalog.Table:
    """Create a new base table.

    Args:
        path_str: Path to the table.
        schema_or_df: Either a dictionary that maps column names to column types, or a
            [`DataFrame`][pixeltable.DataFrame] whose contents and schema will be used to pre-populate the table.
        primary_key: An optional column name or list of column names to use as the primary key(s) of the
            table.
        num_retained_versions: Number of versions of the table to retain.
        comment: An optional comment; its meaning is user-defined.
        media_validation: Media validation policy for the table.

            - `'on_read'`: validate media files at query time
            - `'on_write'`: validate media files during insert/update operations
        if_exists: Directive regarding how to handle if the path already exists.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing table handle
            - `'replace'`: if the existing table has no views, drop and replace it with a new one
            - `'replace_force'`: drop the existing table and all its views, and create a new one

    Returns:
        A handle to the newly created table, or to an already existing table at the path when `if_exists='ignore'`.
            Please note the schema of the existing table may not match the schema provided in the call.

    Raises:
        Error: if

            - the path is invalid, or
            - the path already exists and `if_exists='error'`, or
            - the path already exists and is not a table, or
            - an error occurs while attempting to create the table.

    Examples:
        Create a table with an int and a string column:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String})

        Create a table from a select statement over an existing table `orig_table` (this will create a new table
        containing the exact contents of the query):

        >>> tbl1 = pxt.get_table('orig_table')
        ... tbl2 = pxt.create_table('new_table', tbl1.where(tbl1.col1 < 10).select(tbl1.col2))

        Create a table if does not already exist, otherwise get the existing table:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String}, if_exists='ignore')

        Create a table with an int and a float column, and replace any existing table:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.Float}, if_exists='replace')
    """
    path = catalog.Path(path_str)
    cat = Catalog.get()

    with env.Env.get().begin():
        if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
        existing = _handle_path_collision(path_str, catalog.InsertableTable, False, if_exists_)
        if existing is not None:
            assert isinstance(existing, catalog.Table)
            return existing

        dir = cat.get_schema_object(str(path.parent), expected=catalog.Dir, raise_if_not_exists=True)
        assert dir is not None

        df: Optional[DataFrame] = None
        if isinstance(schema_or_df, dict):
            schema = schema_or_df
        elif isinstance(schema_or_df, DataFrame):
            df = schema_or_df
            schema = df.schema
        elif isinstance(schema_or_df, DataFrameResultSet):
            raise excs.Error(
                '`schema_or_df` must be either a schema dictionary or a Pixeltable DataFrame. (Is there an extraneous call to `collect()`?)'
            )
        else:
            raise excs.Error('`schema_or_df` must be either a schema dictionary or a Pixeltable DataFrame.')

        if len(schema) == 0:
            raise excs.Error(f'Table schema is empty: `{path_str}`')

        if primary_key is None:
            primary_key = []
        elif isinstance(primary_key, str):
            primary_key = [primary_key]
        else:
            if not isinstance(primary_key, list) or not all(isinstance(pk, str) for pk in primary_key):
                raise excs.Error('primary_key must be a single column name or a list of column names')

        tbl = catalog.InsertableTable._create(
            dir._id,
            path.name,
            schema,
            df,
            primary_key=primary_key,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=catalog.MediaValidation.validated(media_validation, 'media_validation'),
        )
        cat.add_tbl(tbl)
        return tbl


def create_view(
    path_str: str,
    base: Union[catalog.Table, DataFrame],
    *,
    additional_columns: Optional[dict[str, Any]] = None,
    is_snapshot: bool = False,
    iterator: Optional[tuple[type[ComponentIterator], dict[str, Any]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> Optional[catalog.Table]:
    """Create a view of an existing table object (which itself can be a view or a snapshot or a base table).

    Args:
        path_str: A name for the view; can be either a simple name such as `my_view`, or a pathname such as
            `dir1.my_view`.
        base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
            base the view on.
        additional_columns: If specified, will add these columns to the view once it is created. The format
            of the `additional_columns` parameter is identical to the format of the `schema_or_df` parameter in
            [`create_table`][pixeltable.create_table].
        is_snapshot: Whether the view is a snapshot. Setting this to `True` is equivalent to calling
            [`create_snapshot`][pixeltable.create_snapshot].
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
    where: Optional[exprs.Expr] = None
    select_list: Optional[list[tuple[exprs.Expr, Optional[str]]]] = None
    if isinstance(base, catalog.Table):
        tbl_version_path = base._tbl_version_path
    elif isinstance(base, DataFrame):
        base._validate_mutable('create_view', allow_select=True)
        if len(base._from_clause.tbls) > 1:
            raise excs.Error('Cannot create a view of a join')
        tbl_version_path = base._from_clause.tbls[0]
        where = base.where_clause
        select_list = base.select_list
    else:
        raise excs.Error('`base` must be an instance of `Table` or `DataFrame`')
    assert isinstance(base, catalog.Table) or isinstance(base, DataFrame)

    path = catalog.Path(path_str)
    cat = Catalog.get()

    with Env.get().begin():
        if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
        existing = _handle_path_collision(path_str, catalog.View, is_snapshot, if_exists_)
        if existing is not None:
            assert isinstance(existing, catalog.View)
            return existing

        dir = cat.get_schema_object(str(path.parent), expected=catalog.Dir, raise_if_not_exists=True)
        assert dir is not None

        if additional_columns is None:
            additional_columns = {}
        else:
            # additional columns should not be in the base table
            for col_name in additional_columns.keys():
                if col_name in [c.name for c in tbl_version_path.columns()]:
                    raise excs.Error(
                        f'Column {col_name!r} already exists in the base table '
                        f'{tbl_version_path.get_column(col_name).tbl.get().name}.'
                    )
        if iterator is None:
            iterator_class, iterator_args = None, None
        else:
            iterator_class, iterator_args = iterator

        view = catalog.View._create(
            dir._id,
            path.name,
            base=tbl_version_path,
            select_list=select_list,
            additional_columns=additional_columns,
            predicate=where,
            is_snapshot=is_snapshot,
            iterator_cls=iterator_class,
            iterator_args=iterator_args,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=catalog.MediaValidation.validated(media_validation, 'media_validation'),
        )
        FileCache.get().emit_eviction_warnings()
        cat.add_tbl(view)
        return view


def create_snapshot(
    path_str: str,
    base: Union[catalog.Table, DataFrame],
    *,
    additional_columns: Optional[dict[str, Any]] = None,
    iterator: Optional[tuple[type[ComponentIterator], dict[str, Any]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> Optional[catalog.Table]:
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
        ... snapshot = pxt.create_snapshot('my_snapshot', view, additional_columns={'col3': pxt.Int}, if_exists='ignore')

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


def get_table(path: str) -> catalog.Table:
    """Get a handle to an existing table, view, or snapshot.

    Args:
        path: Path to the table.

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
    """
    with Env.get().begin():
        obj = Catalog.get().get_schema_object(path, expected=catalog.Table, raise_if_not_exists=True)
        assert isinstance(obj, catalog.Table)
        return obj


def move(path: str, new_path: str) -> None:
    """Move a schema object to a new directory and/or rename a schema object.

    Args:
        path: absolute path to the existing schema object.
        new_path: absolute new path for the schema object.

    Raises:
        Error: If path does not exist or new_path already exists.

    Examples:
        Move a table to a different directory:

        >>>> pxt.move('dir1.my_table', 'dir2.my_table')

        Rename a table:

        >>>> pxt.move('dir1.my_table', 'dir1.new_name')
    """
    if path == new_path:
        raise excs.Error('move(): source and destination cannot be identical')
    cat = Catalog.get()
    with Env.get().begin():
        path_obj, new_path_obj = catalog.Path(path), catalog.Path(new_path)
        if path_obj.is_ancestor(new_path_obj):
            raise excs.Error(f'move(): cannot move {path!r} into its own subdirectory')
        _, dest_dir, src_obj = cat.prepare_dir_op(
            add_dir_path=str(new_path_obj.parent),
            add_name=new_path_obj.name,
            drop_dir_path=str(path_obj.parent),
            drop_name=path_obj.name,
            raise_if_exists=True,
            raise_if_not_exists=True,
        )
        src_obj._move(new_path_obj.name, dest_dir._id)

    #     obj = cat.get_schema_object(path, raise_if_not_exists=True)
    #     new_p = catalog.Path(new_path)
    #     dest_dir_path = str(new_p.parent)
    #     dest_dir = cat.get_schema_object(dest_dir_path, expected=catalog.Dir, raise_if_not_exists=True)
    #     _ = cat.get_schema_object(new_path, raise_if_exists=True)
    #     obj._move(new_p.name, dest_dir._id)
    #
    # with Env.get().begin():
    #     obj = cat.get_schema_object(path, raise_if_not_exists=True)
    #     new_p = catalog.Path(new_path)
    #     dest_dir_path = str(new_p.parent)
    #     dest_dir = cat.get_schema_object(dest_dir_path, expected=catalog.Dir, raise_if_not_exists=True)
    #     _ = cat.get_schema_object(new_path, raise_if_exists=True)
    #     obj._move(new_p.name, dest_dir._id)


def drop_table(
    table: Union[str, catalog.Table], force: bool = False, if_not_exists: Literal['error', 'ignore'] = 'error'
) -> None:
    """Drop a table, view, or snapshot.

    Args:
        table: Fully qualified name, or handle, of the table to be dropped.
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
    cat = Catalog.get()
    tbl: Optional[catalog.Table]
    if isinstance(table, catalog.Table):
        with Env.get().begin():
            table = cat.get_tbl_path(table._id)
    with Env.get().begin():
        if isinstance(table, str):
            path_obj = catalog.Path(table)  # validate path
            if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
            _, _, src_obj = cat.prepare_dir_op(
                drop_dir_path=str(path_obj.parent),
                drop_name=path_obj.name,
                drop_expected=catalog.Table,
                raise_if_not_exists=if_not_exists_ == catalog.IfNotExistsParam.ERROR and not force,
            )
            if src_obj is None:
                _logger.info(f'Skipped table `{table}` (does not exist).')
                return
            assert isinstance(src_obj, catalog.Table)
            tbl = src_obj
        else:
            # TODO: correct locks
            tbl = table
        _drop_table(tbl, force=force, is_replace=False)


def _drop_table(tbl: catalog.Table, force: bool, is_replace: bool) -> None:
    cat = Catalog.get()
    view_ids = cat.get_views(tbl._id)
    if len(view_ids) > 0:
        view_paths = [cat.get_tbl_path(id) for id in view_ids]
        if force:
            for view_path in view_paths:
                drop_table(view_path, force=True)
        else:
            is_snapshot = tbl._tbl_version_path.is_snapshot()
            obj_type_str = 'Snapshot' if is_snapshot else tbl._display_name().capitalize()
            msg: str
            if is_replace:
                msg = (
                    f'{obj_type_str} {tbl._path()} already exists and has dependents: {", ".join(view_paths)}. '
                    "Use `if_exists='replace_force'` to replace it."
                )
            else:
                msg = f'{obj_type_str} {tbl._path()} has dependents: {", ".join(view_paths)}'
            raise excs.Error(msg)
    tbl._drop()
    _logger.info(f'Dropped table `{tbl._path()}`.')


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
    _ = catalog.Path(dir_path, empty_is_valid=True)  # validate format
    cat = Catalog.get()
    with Env.get().begin():
        dir = cat.get_schema_object(dir_path, expected=catalog.Dir, raise_if_not_exists=True)
        contents = cat.get_dir_contents(dir._id, recursive=recursive)
        return _extract_paths(contents, prefix=dir_path, entry_type=catalog.Table)


def create_dir(
    path: str, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error'
) -> Optional[catalog.Dir]:
    """Create a directory.

    Args:
        path: Path to the directory.
        if_exists: Directive regarding how to handle if the path already exists.
            Must be one of the following:

            - `'error'`: raise an error
            - `'ignore'`: do nothing and return the existing directory handle
            - `'replace'`: if the existing directory is empty, drop it and create a new one
            - `'replace_force'`: drop the existing directory and all its children, and create a new one

    Returns:
        A handle to the newly created directory, or to an already existing directory at the path when `if_exists='ignore'`.
            Please note the existing directory may not be empty.

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
    """
    path_obj = catalog.Path(path)
    cat = Catalog.get()

    with env.Env.get().begin():
        if_exists_ = catalog.IfExistsParam.validated(if_exists, 'if_exists')
        existing = _handle_path_collision(path, catalog.Dir, False, if_exists_)
        if existing is not None:
            assert isinstance(existing, catalog.Dir)
            return existing

        parent = cat.get_schema_object(str(path_obj.parent))
        assert parent is not None
        dir = catalog.Dir._create(parent._id, path_obj.name)
        Env.get().console_logger.info(f'Created directory {path!r}.')
        return dir


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
    _ = catalog.Path(path)  # validate format
    cat = Catalog.get()
    if_not_exists_ = catalog.IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
    path_obj = catalog.Path(path)
    with Env.get().begin():
        _, _, schema_obj = cat.prepare_dir_op(
            drop_dir_path=str(path_obj.parent),
            drop_name=path_obj.name,
            drop_expected=catalog.Dir,
            raise_if_not_exists=if_not_exists_ == catalog.IfNotExistsParam.ERROR and not force,
        )
        if schema_obj is None:
            _logger.info(f'Directory {path!r} does not exist, skipped drop_dir().')
            return
        _drop_dir(schema_obj._id, path, force=force)


def _drop_dir(dir_id: UUID, path: str, force: bool = False) -> None:
    cat = Catalog.get()
    dir_entries = cat.get_dir_contents(dir_id, recursive=False)
    if len(dir_entries) > 0 and not force:
        raise excs.Error(f'Directory {path!r} is not empty.')
    tbl_paths = [_join_path(path, entry.table.md['name']) for entry in dir_entries.values() if entry.table is not None]
    dir_paths = [_join_path(path, entry.dir.md['name']) for entry in dir_entries.values() if entry.dir is not None]

    for tbl_path in tbl_paths:
        # check if the table still exists, it might be a view that already got force-deleted
        if cat.get_schema_object(tbl_path, expected=catalog.Table, for_update=True) is not None:
            drop_table(tbl_path, force=True)
    for dir_path in dir_paths:
        drop_dir(dir_path, force=True)
    cat.drop_dir(dir_id)
    _logger.info(f'Removed directory {path!r}.')


def _join_path(path: str, name: str) -> str:
    """Append name to path, if path is not empty."""
    return name if path == '' else f'{path}.{name}'


def _extract_paths(
    dir_entries: dict[str, Catalog.DirEntry], prefix: str, entry_type: Optional[type[catalog.SchemaObject]] = None
) -> list[str]:
    """Convert nested dir_entries structure to a flattened list of paths."""
    matches: list[str]
    if entry_type is None:
        matches = list(dir_entries.keys())
    elif entry_type is catalog.Dir:
        matches = [name for name, entry in dir_entries.items() if entry.dir is not None]
    else:
        matches = [name for name, entry in dir_entries.items() if entry.table is not None]
    result = [_join_path(prefix, name) for name in matches]
    for name, entry in [(name, entry) for name, entry in dir_entries.items() if len(entry.dir_entries) > 0]:
        result.extend(_extract_paths(entry.dir_entries, prefix=_join_path(prefix, name), entry_type=entry_type))
    return result


def publish_snapshot(dest_uri: str, table: catalog.Table) -> None:
    parsed_uri = urllib.parse.urlparse(dest_uri)
    if parsed_uri.scheme != 'pxt':
        raise excs.Error(f'Invalid Pixeltable URI (does not start with pxt://): {dest_uri}')
    share.publish_snapshot(dest_uri, table)


def list_dirs(path_str: str = '', recursive: bool = True) -> list[str]:
    """List the directories in a directory.

    Args:
        path_str: Name or path of the directory.
        recursive: If `True`, lists all descendants of this directory recursively.

    Returns:
        List of directory paths.

    Raises:
        Error: If `path_str` does not exist or does not designate a directory.

    Examples:
        >>> cl.list_dirs('my_dir', recursive=True)
        ['my_dir', 'my_dir.sub_dir1']
    """
    _ = catalog.Path(path_str, empty_is_valid=True)  # validate format
    cat = Catalog.get()
    with Env.get().begin():
        dir = cat.get_schema_object(path_str, expected=catalog.Dir, raise_if_not_exists=True)
        contents = cat.get_dir_contents(dir._id, recursive=recursive)
        return _extract_paths(contents, prefix=path_str, entry_type=catalog.Dir)


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
        [dict(selector='th', props=[('text-align', 'center')])]
    )  # center-align headings
    return pd_df.hide(axis='index')


def tools(*args: Union[func.Function, func.tools.Tool]) -> func.tools.Tools:
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


def tool(fn: func.Function, name: Optional[str] = None, description: Optional[str] = None) -> func.tools.Tool:
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
    *,
    to_stdout: Optional[bool] = None,
    level: Optional[int] = None,
    add: Optional[str] = None,
    remove: Optional[str] = None,
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
