import dataclasses
import logging
from typing import Any, Optional, Union, Type

import pandas as pd
import sqlalchemy as sql
from sqlalchemy.util.preloaded import orm

import pixeltable.exceptions as excs
from pixeltable import catalog, func
from pixeltable.catalog import Catalog
from pixeltable.env import Env
from pixeltable.exprs import Predicate
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema

_logger = logging.getLogger('pixeltable')


def init() -> None:
    """Initializes the Pixeltable environment."""
    _ = Catalog.get()


def create_table(
        path_str: str, schema: dict[str, Any], *, primary_key: Optional[Union[str, list[str]]] = None,
        num_retained_versions: int = 10, comment: str = ''
) -> catalog.InsertableTable:
    """Create a new `InsertableTable`.

    Args:
        path_str: Path to the table.
        schema: dictionary mapping column names to column types, value expressions, or to column specifications.
        num_retained_versions: Number of versions of the table to retain.

    Returns:
        The newly created table.

    Raises:
        Error: if the path already exists or is invalid.

    Examples:
        Create a table with an int and a string column:

        >>> table = cl.create_table('my_table', schema={'col1': IntType(), 'col2': StringType()})
    """
    path = catalog.Path(path_str)
    Catalog.get().paths.check_is_valid(path, expected=None)
    dir = Catalog.get().paths[path.parent]

    if len(schema) == 0:
        raise excs.Error(f'Table schema is empty: `{path_str}`')

    if primary_key is None:
        primary_key = []
    elif isinstance(primary_key, str):
        primary_key = [primary_key]
    else:
        if not isinstance(primary_key, list) or not all(isinstance(pk, str) for pk in primary_key):
            raise excs.Error('primary_key must be a single column name or a list of column names')

    tbl = catalog.InsertableTable.create(
        dir._id, path.name, schema, primary_key=primary_key, num_retained_versions=num_retained_versions,
        comment=comment)
    Catalog.get().paths[path] = tbl
    _logger.info(f'Created table `{path_str}`.')
    return tbl

def create_view(
        path_str: str, base: catalog.Table, *, schema: Optional[dict[str, Any]] = None,
        filter: Optional[Predicate] = None,
        is_snapshot: bool = False, iterator_class: Optional[Type[ComponentIterator]] = None,
        iterator_args: Optional[dict[str, Any]] = None, num_retained_versions: int = 10, comment: str = '',
        ignore_errors: bool = False) -> catalog.View:
    """Create a new `View`.

    Args:
        path_str: Path to the view.
        base: Table (ie, table or view or snapshot) to base the view on.
        schema: dictionary mapping column names to column types, value expressions, or to column specifications.
        filter: Predicate to filter rows of the base table.
        is_snapshot: Whether the view is a snapshot.
        iterator_class: Class of the iterator to use for the view.
        iterator_args: Arguments to pass to the iterator class.
        num_retained_versions: Number of versions of the view to retain.
        ignore_errors: if True, fail silently if the path already exists or is invalid.

    Returns:
        The newly created view.

    Raises:
        Error: if the path already exists or is invalid.

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
    assert (iterator_class is None) == (iterator_args is None)
    assert isinstance(base, catalog.Table)
    path = catalog.Path(path_str)
    try:
        Catalog.get().paths.check_is_valid(path, expected=None)
    except Exception as e:
        if ignore_errors:
            return
        else:
            raise e
    dir = Catalog.get().paths[path.parent]

    if schema is None:
        schema = {}
    view = catalog.View.create(
        dir._id, path.name, base=base, schema=schema, predicate=filter, is_snapshot=is_snapshot,
        iterator_cls=iterator_class, iterator_args=iterator_args, num_retained_versions=num_retained_versions,
        comment=comment)
    Catalog.get().paths[path] = view
    _logger.info(f'Created view `{path_str}`.')
    return view


def get_table(path: str) -> catalog.Table:
    """Get a handle to a table (including views and snapshots).

    Args:
        path: Path to the table.

    Returns:
        A `InsertableTable` or `View` object.

    Raises:
        Error: If the path does not exist or does not designate a table.

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
    obj.move(new_p.name, new_dir._id)


def drop_table(path: str, force: bool = False, ignore_errors: bool = False) -> None:
    """Drop a table.

    Args:
        path: Path to the table.
        force: Whether to drop the table even if it has unsaved changes.
        ignore_errors: Whether to ignore errors if the table does not exist.

    Raises:
        Error: If the path does not exist or does not designate a table and ignore_errors is False.

    Examples:
        >>> cl.drop_table('my_table')
    """
    path_obj = catalog.Path(path)
    try:
        Catalog.get().paths.check_is_valid(path_obj, expected=catalog.Table)
    except Exception as e:
        if ignore_errors:
            _logger.info(f'Skipped table `{path}` (does not exist).')
            return
        else:
            raise e
    tbl = Catalog.get().paths[path_obj]
    if len(Catalog.get().tbl_dependents[tbl._id]) > 0:
        dependent_paths = [get_path(dep) for dep in Catalog.get().tbl_dependents[tbl._id]]
        raise excs.Error(f'Table {path} has dependents: {", ".join(dependent_paths)}')
    tbl._drop()
    del Catalog.get().paths[path_obj]
    _logger.info(f'Dropped table `{path}`.')


def list_tables(dir_path: str = '', recursive: bool = True) -> list[str]:
    """List the tables in a directory.

    Args:
        dir_path: Path to the directory. Defaults to the root directory.
        recursive: Whether to list tables in subdirectories as well.

    Returns:
        A list of table paths.

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


def create_dir(path_str: str, ignore_errors: bool = False) -> None:
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
            Catalog.get().paths[path] = catalog.Dir(dir_record.id, parent._id, path.name)
            session.commit()
            _logger.info(f'Created directory `{path_str}`.')
            print(f'Created directory `{path_str}`.')
    except excs.Error as e:
        if ignore_errors:
            return
        else:
            raise e


def rm_dir(path_str: str) -> None:
    """Remove a directory.

    Args:
        path_str: Path to the directory.

    Raises:
        Error: If the path does not exist or does not designate a directory or if the directory is not empty.

    Examples:
        >>> cl.rm_dir('my_dir')

        Remove a subdirectory:

        >>> cl.rm_dir('my_dir.sub_dir')
    """
    path = catalog.Path(path_str)
    Catalog.get().paths.check_is_valid(path, expected=catalog.Dir)

    # make sure it's empty
    if len(Catalog.get().paths.get_children(path, child_type=None, recursive=True)) > 0:
        raise excs.Error(f'Directory {path_str} is not empty')
    # TODO: figure out how to make force=True work in the presence of snapshots
    #        # delete tables
    #        for tbl_path in self.paths.get_children(path, child_type=MutableTable, recursive=True):
    #            self.drop_table(str(tbl_path), force=True)
    #        # rm subdirs
    #        for dir_path in self.paths.get_children(path, child_type=Dir, recursive=False):
    #            self.rm_dir(str(dir_path), force=True)

    with Env.get().engine.begin() as conn:
        dir = Catalog.get().paths[path]
        conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.id == dir._id))
    del Catalog.get().paths[path]
    _logger.info(f'Removed directory {path_str}')


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


def list_functions() -> pd.DataFrame:
    """Returns information about all registered functions.

    Returns:
        Pandas DataFrame with columns 'Path', 'Name', 'Parameters', 'Return Type', 'Is Agg', 'Library'
    """
    functions = func.FunctionRegistry.get().list_functions()
    paths = ['.'.join(f.self_path.split('.')[:-1]) for f in functions]
    names = [f.name for f in functions]
    params = [
        ', '.join(
            [param_name + ': ' + str(param_type) for param_name, param_type in f.signature.parameters.items()])
        for f in functions
    ]
    pd_df = pd.DataFrame({
        'Path': paths,
        'Function Name': names,
        'Parameters': params,
        'Return Type': [str(f.signature.get_return_type()) for f in functions],
    })
    pd_df = pd_df.style.set_properties(**{'text-align': 'left'}) \
        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])  # center-align headings
    return pd_df.hide(axis='index')


def get_path(schema_obj: catalog.SchemaObject) -> str:
    """Returns the path to a SchemaObject.

    Args:
        schema_obj: SchemaObject to get the path for.

    Returns:
        Path to the SchemaObject.
    """
    path_elements: list[str] = []
    dir_id = schema_obj._dir_id
    while dir_id is not None:
        dir = Catalog.get().paths.get_schema_obj(dir_id)
        if dir._dir_id is None:
            # this is the root dir with name '', which we don't want to include in the path
            break
        path_elements.insert(0, dir._name)
        dir_id = dir._dir_id
    path_elements.append(schema_obj._name)
    return '.'.join(path_elements)
