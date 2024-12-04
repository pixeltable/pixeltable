import dataclasses
import logging
from typing import Any, Iterable, Optional, Union, Literal
from uuid import UUID

import pandas as pd
import sqlalchemy as sql
from pandas.io.formats.style import Styler
from sqlalchemy.util.preloaded import orm

import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
from pixeltable import DataFrame, catalog, func
from pixeltable.catalog import Catalog
from pixeltable.dataframe import DataFrameResultSet
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache

_logger = logging.getLogger('pixeltable')


def init() -> None:
    """Initializes the Pixeltable environment."""
    _ = Catalog.get()


def create_table(
    path_str: str,
    schema_or_df: Union[dict[str, Any], DataFrame],
    *,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: Literal['on_read', 'on_write'] = 'on_write'
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

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].

    Raises:
        Error: if the path already exists or is invalid.

    Examples:
        Create a table with an int and a string column:

        >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String})

        Create a table from a select statement over an existing table `orig_table` (this will create a new table
        containing the exact contents of the query):

        >>> tbl1 = pxt.get_table('orig_table')
        ... tbl2 = pxt.create_table('new_table', tbl1.where(tbl1.col1 < 10).select(tbl1.col2))
    """
    path = catalog.Path(path_str)
    Catalog.get().paths.check_is_valid(path, expected=None)
    dir = Catalog.get().paths[path.parent]

    df: Optional[DataFrame] = None
    if isinstance(schema_or_df, dict):
        schema = schema_or_df
    elif isinstance(schema_or_df, DataFrame):
        df = schema_or_df
        schema = df.schema
    elif isinstance(schema_or_df, DataFrameResultSet):
        raise excs.Error('`schema_or_df` must be either a schema dictionary or a Pixeltable DataFrame. (Is there an extraneous call to `collect()`?)')
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
        dir._id, path.name, schema, df, primary_key=primary_key, num_retained_versions=num_retained_versions,
        comment=comment, media_validation=catalog.MediaValidation.validated(media_validation, 'media_validation'))
    Catalog.get().paths[path] = tbl

    _logger.info(f'Created table `{path_str}`.')
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
    ignore_errors: bool = False,
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
        ignore_errors: if True, fail silently if the path already exists or is invalid.

    Returns:
        A handle to the [`Table`][pixeltable.Table] representing the newly created view. If the path already
            exists or is invalid and `ignore_errors=True`, returns `None`.

    Raises:
        Error: if the path already exists or is invalid and `ignore_errors=False`.

    Examples:
        Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 10:

        >>> tbl = pxt.get_table('my_table')
        ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 10))
    """
    where: Optional[exprs.Expr] = None
    if isinstance(base, catalog.Table):
        tbl_version_path = base._tbl_version_path
    elif isinstance(base, DataFrame):
        base._validate_mutable('create_view')
        if len(base._from_clause.tbls) > 1:
            raise excs.Error('Cannot create a view of a join')
        tbl_version_path = base._from_clause.tbls[0]
        where = base.where_clause
    else:
        raise excs.Error('`base` must be an instance of `Table` or `DataFrame`')
    assert isinstance(base, catalog.Table) or isinstance(base, DataFrame)
    path = catalog.Path(path_str)
    try:
        Catalog.get().paths.check_is_valid(path, expected=None)
    except Exception as e:
        if ignore_errors:
            return None
        else:
            raise e
    dir = Catalog.get().paths[path.parent]

    if additional_columns is None:
        additional_columns = {}
    if iterator is None:
        iterator_class, iterator_args = None, None
    else:
        iterator_class, iterator_args = iterator

    view = catalog.View._create(
        dir._id, path.name, base=tbl_version_path, additional_columns=additional_columns, predicate=where,
        is_snapshot=is_snapshot, iterator_cls=iterator_class, iterator_args=iterator_args,
        num_retained_versions=num_retained_versions, comment=comment,
        media_validation=catalog.MediaValidation.validated(media_validation, 'media_validation'))
    Catalog.get().paths[path] = view
    _logger.info(f'Created view `{path_str}`.')
    FileCache.get().emit_eviction_warnings()
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
    ignore_errors: bool = False,
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
        comment: Optional comment for the view.
        ignore_errors: if True, fail silently if the path already exists or is invalid.

    Returns:
        A handle to the [`Table`][pixeltable.Table] representing the newly created snapshot. If the path already
            exists or is invalid and `ignore_errors=True`, returns `None`.

    Raises:
        Error: if the path already exists or is invalid and `ignore_errors=False`.

    Examples:
        Create a snapshot of `my_table`:

        >>> tbl = pxt.get_table('my_table')
        ... snapshot = pxt.create_snapshot('my_snapshot', tbl)
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
        ignore_errors=ignore_errors,
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
    p = catalog.Path(path)
    Catalog.get().paths.check_is_valid(p, expected=catalog.Table)
    obj = Catalog.get().paths[p]
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
    p = catalog.Path(path)
    Catalog.get().paths.check_is_valid(p, expected=catalog.SchemaObject)
    new_p = catalog.Path(new_path)
    Catalog.get().paths.check_is_valid(new_p, expected=None)
    obj = Catalog.get().paths[p]
    Catalog.get().paths.move(p, new_p)
    new_dir = Catalog.get().paths[new_p.parent]
    obj._move(new_p.name, new_dir._id)


def drop_table(table: Union[str, catalog.Table], force: bool = False, ignore_errors: bool = False) -> None:
    """Drop a table, view, or snapshot.

    Args:
        table: Fully qualified name, or handle, of the table to be dropped.
        force: If `True`, will also drop all views and sub-views of this table.
        ignore_errors: If `True`, return silently if the table does not exist (without throwing an exception).

    Raises:
        Error: If the name does not exist or does not designate a table object, and `ignore_errors=False`.

    Examples:
        Drop a table by its fully qualified name:
        >>> pxt.drop_table('subdir.my_table')

        Drop a table by its handle:
        >>> t = pxt.get_table('subdir.my_table')
        ... pxt.drop_table(t)

    """
    cat = Catalog.get()
    if isinstance(table, str):
        tbl_path_obj = catalog.Path(table)
        try:
            cat.paths.check_is_valid(tbl_path_obj, expected=catalog.Table)
        except Exception as e:
            if ignore_errors or force:
                _logger.info(f'Skipped table `{table}` (does not exist).')
                return
            else:
                raise e
        tbl = cat.paths[tbl_path_obj]
    else:
        tbl = table
        tbl_path_obj = catalog.Path(tbl._path)

    assert isinstance(tbl, catalog.Table)
    if len(cat.tbl_dependents[tbl._id]) > 0:
        dependent_paths = [dep._path for dep in cat.tbl_dependents[tbl._id]]
        if force:
            for dependent_path in dependent_paths:
                drop_table(dependent_path, force=True)
        else:
            raise excs.Error(f'Table {tbl._path} has dependents: {", ".join(dependent_paths)}')
    tbl._drop()
    del cat.paths[tbl_path_obj]
    _logger.info(f'Dropped table `{tbl._path}`.')


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
    assert dir_path is not None
    path = catalog.Path(dir_path, empty_is_valid=True)
    Catalog.get().paths.check_is_valid(path, expected=catalog.Dir)
    return [str(p) for p in Catalog.get().paths.get_children(path, child_type=catalog.Table, recursive=recursive)]

def _verify_param_if_exists(if_exists: str) -> None:
    """Verify the value of the `if_exists` parameter.

    The value must be one of the following: 'error', 'ignore', 'replace', 'replace_force'.

    Args:
        if_exists: Value of the `if_exists` parameter.

    Raises:
        Error: If the value is invalid.
    """
    if if_exists not in ['error', 'ignore', 'replace', 'replace_force']:
        raise excs.Error(f'Invalid value for `if_exists`: {if_exists}. It must be one of: error, ignore, replace, replace_force.')

def create_dir(path_str: str, if_exists: str = 'error') -> Optional[catalog.Dir]:
    """Create a directory.

    Args:
        path_str: Path to the directory.
        if_exists: Directive on how to handle the scenario where the directory already exists.
            Must be one of the following:
            'error' => raise an error
            'ignore' => return and do nothing
            'replace', or 'replace_force' =>  drop the existing directory and create a new one
            Default is 'error'.

    Raises:
        Error: If the path already exists and if_exists is 'error'
            or other error conditions like invalid path, store errors etc.

    Examples:
        >>> pxt.create_dir('my_dir')

        Create a subdirectory:

        >>> pxt.create_dir('my_dir.sub_dir')

        Create a subdirectory only if does not already exist:

        >>> pxt.create_dir('my_dir.sub_dir', if_exists='ignore')

        Create a directory and replace if it already exists:

        >>> pxt.create_dir('my_dir', if_exists='replace')
    """
    _verify_param_if_exists(if_exists)

    path = catalog.Path(path_str)

    try:
        Catalog.get().paths.check_is_valid(path, expected=None)
    except excs.Error as e:
        if 'already exists' not in str(e) or if_exists == 'error':
            raise e

        # The directory already exists. Handle per user directive.
        if if_exists == 'replace' or if_exists == 'replace_force':
            # Drop the existing directory so it can be replaced.
            # Any error from drop here will not be ignored.
            _logger.info(f'Dropping and recreating directory `{path_str}`.')
            drop_dir(path_str, force=True, ignore_errors=False)
        elif if_exists == 'ignore':
            return None

    parent = Catalog.get().paths[path.parent]
    assert parent is not None
    with orm.Session(Env.get().engine, future=True) as session:
        dir_md = schema.DirMd(name=path.name)
        dir_record = schema.Dir(parent_id=parent._id, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        assert dir_record.id is not None
        assert isinstance(dir_record.id, UUID)
        dir = catalog.Dir(dir_record.id, parent._id, path.name)
        Catalog.get().paths[path] = dir
        session.commit()
        _logger.info(f'Created directory `{path_str}`.')
        print(f'Created directory `{path_str}`.')
        return dir

def drop_dir(path_str: str, force: bool = False, ignore_errors: bool = False) -> None:
    """Remove a directory.

    Args:
        path_str: Name or path of the directory.
        force: If `True`, will also drop all tables and subdirectories of this directory, recursively, along
            with any views or snapshots that depend on any of the dropped tables.
        ignore_errors: if `True`, will return silently instead of throwing an exception if the directory
            does not exist.

    Raises:
        Error: If the path does not exist or does not designate a directory, or if the directory is not empty.

    Examples:
        >>> pxt.drop_dir('my_dir')

        Remove a subdirectory:

        >>> pxt.drop_dir('my_dir.sub_dir')
    """
    cat = Catalog.get()
    path = catalog.Path(path_str)

    try:
        cat.paths.check_is_valid(path, expected=catalog.Dir)
    except Exception as e:
        if ignore_errors or force:
            _logger.info(f'Skipped directory `{path}` (does not exist).')
            return
        else:
            raise e

    children = cat.paths.get_children(path, child_type=None, recursive=True)

    if len(children) > 0 and not force:
        raise excs.Error(f'Directory `{path_str}` is not empty.')

    for child in children:
        assert isinstance(child, catalog.Path)
        # We need to check that the child is still in `cat.paths`, since it is possible it was
        # already deleted as a dependent of a preceding child in the iteration.
        try:
            obj = cat.paths[child]
        except excs.Error:
            continue
        if isinstance(obj, catalog.Dir):
            drop_dir(str(child), force=True)
        else:
            assert isinstance(obj, catalog.Table)
            assert not obj._is_dropped  # else it should have been removed from `cat.paths` already
            drop_table(str(child), force=True)

    with Env.get().engine.begin() as conn:
        dir = Catalog.get().paths[path]
        conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.id == dir._id))
    del Catalog.get().paths[path]
    _logger.info(f'Removed directory `{path_str}`.')


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
    path = catalog.Path(path_str, empty_is_valid=True)
    Catalog.get().paths.check_is_valid(path, expected=catalog.Dir)
    return [str(p) for p in Catalog.get().paths.get_children(path, child_type=catalog.Dir, recursive=recursive)]


def list_functions() -> Styler:
    """Returns information about all registered functions.

    Returns:
        Pandas DataFrame with columns 'Path', 'Name', 'Parameters', 'Return Type', 'Is Agg', 'Library'
    """
    functions = func.FunctionRegistry.get().list_functions()
    paths = ['.'.join(f.self_path.split('.')[:-1]) for f in functions]
    names = [f.name for f in functions]
    params = [
        ', '.join([param_name + ': ' + str(param_type) for param_name, param_type in f.signature.parameters.items()])
        for f in functions
    ]
    pd_df = pd.DataFrame(
        {
            'Path': paths,
            'Function Name': names,
            'Parameters': params,
            'Return Type': [str(f.signature.get_return_type()) for f in functions],
        }
    )
    pd_df = pd_df.style.set_properties(None, **{'text-align': 'left'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]
    )  # center-align headings
    return pd_df.hide(axis='index')


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
    return exprs.InlineArray(elements)
