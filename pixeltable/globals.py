import dataclasses
import logging
from typing import Any, Optional, Union
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

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].

    Raises:
        Error: if the path already exists or is invalid.

    Examples:
        Create a table with an int and a string column:

        >>> table = pxt.create_table('my_table', schema={'col1': IntType(), 'col2': StringType()})

        Create a table from a select statement over an existing table `tbl`:

        >>> table = pxt.create_table('my_table', tbl.where(tbl.col1 < 10).select(tbl.col2))
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
        dir._id,
        path.name,
        schema,
        df,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )
    Catalog.get().paths[path] = tbl

    _logger.info(f'Created table `{path_str}`.')
    return tbl


def create_view(
    path_str: str,
    base: Union[catalog.Table, DataFrame],
    *,
    schema: Optional[dict[str, Any]] = None,
    filter: Optional[exprs.Expr] = None,
    is_snapshot: bool = False,
    iterator: Optional[tuple[type[ComponentIterator], dict[str, Any]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    ignore_errors: bool = False,
) -> Optional[catalog.Table]:
    """Create a view of an existing table object (which itself can be a view or a snapshot or a base table).

    Args:
        path_str: Path to the view.
        base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
            base the view on.
        schema: dictionary mapping column names to column types, value expressions, or to column specifications.
        filter: predicate to filter rows of the base table.
        is_snapshot: Whether the view is a snapshot.
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
        Create a view with an additional int and a string column and a filter:

        >>> view = cl.create_view(
            'my_view', base, schema={'col3': IntType(), 'col4': StringType()}, filter=base.col1 > 10)

        Create a table snapshot:

        >>> snapshot_view = cl.create_view('my_snapshot_view', base, is_snapshot=True)

        Create an immutable view with additional computed columns and a filter:

        >>> snapshot_view = cl.create_view(
            'my_snapshot', base, schema={'col3': base.col2 + 1}, filter=base.col1 > 10, is_snapshot=True)
    """
    if isinstance(base, catalog.Table):
        tbl_version_path = base._tbl_version_path
    elif isinstance(base, DataFrame):
        base._validate_mutable('create_view')
        tbl_version_path = base.tbl
        if base.where_clause is not None and filter is not None:
            raise excs.Error(
                'Cannot specify a `filter` directly if one is already declared in a `DataFrame.where` clause'
            )
        filter = base.where_clause
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

    if schema is None:
        schema = {}
    if iterator is None:
        iterator_class, iterator_args = None, None
    else:
        iterator_class, iterator_args = iterator

    view = catalog.View._create(
        dir._id,
        path.name,
        base=tbl_version_path,
        schema=schema,
        predicate=filter,
        is_snapshot=is_snapshot,
        iterator_cls=iterator_class,
        iterator_args=iterator_args,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )
    Catalog.get().paths[path] = view
    _logger.info(f'Created view `{path_str}`.')
    FileCache.get().emit_eviction_warnings()
    return view


def get_table(path: str) -> catalog.Table:
    """Get a handle to an existing table or view or snapshot.

    Args:
        path: Path to the table.

    Returns:
        A handle to the [`Table`][pixeltable.Table].

    Raises:
        Error: If the path does not exist or does not designate a table object.

    Examples:
        Get handle for a table in the top-level directory:

        >>> table = cl.get_table('my_table')

        For a table in a subdirectory:

        >>> table = cl.get_table('subdir.my_table')

        For a snapshot in the top-level directory:

        >>> table = cl.get_table('my_snapshot')
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

        >>>> cl.move('dir1.my_table', 'dir2.my_table')

        Rename a table:

        >>>> cl.move('dir1.my_table', 'dir1.new_name')
    """
    p = catalog.Path(path)
    Catalog.get().paths.check_is_valid(p, expected=catalog.SchemaObject)
    new_p = catalog.Path(new_path)
    Catalog.get().paths.check_is_valid(new_p, expected=None)
    obj = Catalog.get().paths[p]
    Catalog.get().paths.move(p, new_p)
    new_dir = Catalog.get().paths[new_p.parent]
    obj._move(new_p.name, new_dir._id)


def drop_table(path: str, force: bool = False, ignore_errors: bool = False) -> None:
    """Drop a table or view or snapshot.

    Args:
        path: Path to the [`Table`][pixeltable.Table].
        force: If `True`, will also drop all views or sub-views of this table.
        ignore_errors: Whether to ignore errors if the table does not exist.

    Raises:
        Error: If the path does not exist or does not designate a table object and ignore_errors is False.

    Examples:
        >>> cl.drop_table('my_table')
    """
    cat = Catalog.get()
    path_obj = catalog.Path(path)
    try:
        cat.paths.check_is_valid(path_obj, expected=catalog.Table)
    except Exception as e:
        if ignore_errors or force:
            _logger.info(f'Skipped table `{path}` (does not exist).')
            return
        else:
            raise e
    tbl = cat.paths[path_obj]
    assert isinstance(tbl, catalog.Table)
    if len(cat.tbl_dependents[tbl._id]) > 0:
        dependent_paths = [dep._path for dep in cat.tbl_dependents[tbl._id]]
        if force:
            for dependent_path in dependent_paths:
                drop_table(dependent_path, force=True)
        else:
            raise excs.Error(f'Table {path} has dependents: {", ".join(dependent_paths)}')
    tbl._drop()
    del cat.paths[path_obj]
    _logger.info(f'Dropped table `{path}`.')


def list_tables(dir_path: str = '', recursive: bool = True) -> list[str]:
    """List the [`Table`][pixeltable.Table]s in a directory.

    Args:
        dir_path: Path to the directory. Defaults to the root directory.
        recursive: Whether to list tables in subdirectories as well.

    Returns:
        A list of [`Table`][pixeltable.Table] paths.

    Raises:
        Error: If the path does not exist or does not designate a directory.

    Examples:
        List tables in top-level directory:

        >>> cl.list_tables()
        ['my_table', ...]

        List tables in 'dir1':

        >>> cl.list_tables('dir1')
        [...]
    """
    assert dir_path is not None
    path = catalog.Path(dir_path, empty_is_valid=True)
    Catalog.get().paths.check_is_valid(path, expected=catalog.Dir)
    return [str(p) for p in Catalog.get().paths.get_children(path, child_type=catalog.Table, recursive=recursive)]


def create_dir(path_str: str, ignore_errors: bool = False) -> Optional[catalog.Dir]:
    """Create a directory.

    Args:
        path_str: Path to the directory.
        ignore_errors: if True, silently returns on error

    Raises:
        Error: If the path already exists or the parent is not a directory.

    Examples:
        >>> cl.create_dir('my_dir')

        Create a subdirectory:

        >>> cl.create_dir('my_dir.sub_dir')
    """
    try:
        path = catalog.Path(path_str)
        Catalog.get().paths.check_is_valid(path, expected=None)
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
    except excs.Error as e:
        if ignore_errors:
            return None
        else:
            raise e


def drop_dir(path_str: str, force: bool = False, ignore_errors: bool = False) -> None:
    """Remove a directory.

    Args:
        path_str: Path to the directory.

    Raises:
        Error: If the path does not exist or does not designate a directory or if the directory is not empty.

    Examples:
        >>> cl.drop_dir('my_dir')

        Remove a subdirectory:

        >>> cl.drop_dir('my_dir.sub_dir')
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
        path_str: Path to the directory.
        recursive: Whether to list subdirectories recursively.

    Returns:
        List of directory paths.

    Raises:
        Error: If the path does not exist or does not designate a directory.

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
