from typing import List, Optional, Dict, Type, Any, Union
import pandas as pd
import logging
import dataclasses

import sqlalchemy as sql
import sqlalchemy.orm as orm

import pixeltable
from pixeltable.metadata import schema
from pixeltable.env import Env
import pixeltable.func as func
import pixeltable.catalog as catalog
from pixeltable import exceptions as excs
from pixeltable.exprs import Predicate
from pixeltable.iterators import ComponentIterator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datasets

__all__ = [
    'Client',
]


_logger = logging.getLogger('pixeltable')

class Client:
    """
    Client for interacting with a Pixeltable environment.
    """

    def __init__(self, reload: bool = False) -> None:
        """Constructs a client.
        """
        env = Env.get()
        env.set_up()
        env.upgrade_metadata()
        if reload:
            catalog.Catalog.clear()
        self.catalog = catalog.Catalog.get()

    def logging(
            self, *, to_stdout: Optional[bool] = None, level: Optional[int] = None,
            add: Optional[str] = None, remove: Optional[str] = None
    ) -> None:
        """Configure logging.

        Args:
            to_stdout: if True, also log to stdout
            level: default log level
            add: comma-separated list of 'module name:log level' pairs; ex.: add='video:10'
            remove: comma-separated list of module names
        """
        if to_stdout is not None:
            Env.get().log_to_stdout(to_stdout)
        if level is not None:
            Env.get().set_log_level(level)
        if add is not None:
            for module, level in [t.split(':') for t in add.split(',')]:
                Env.get().set_module_log_level(module, int(level))
        if remove is not None:
            for module in remove.split(','):
                Env.get().set_module_log_level(module, None)
        if to_stdout is None and level is None and add is None and remove is None:
            Env.get().print_log_config()

    def list_functions(self) -> pd.DataFrame:
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

    def get_path(self, schema_obj: catalog.SchemaObject) -> str:
        """Returns the path to a SchemaObject.

        Args:
            schema_obj: SchemaObject to get the path for.

        Returns:
            Path to the SchemaObject.
        """
        path_elements: List[str] = []
        dir_id = schema_obj._dir_id
        while dir_id is not None:
            dir = self.catalog.paths.get_schema_obj(dir_id)
            if dir._dir_id is None:
                # this is the root dir with name '', which we don't want to include in the path
                break
            path_elements.insert(0, dir._name)
            dir_id = dir._dir_id
        path_elements.append(schema_obj._name)
        return '.'.join(path_elements)

    def create_table(
            self, path_str: str, schema: Dict[str, Any], primary_key: Optional[Union[str, List[str]]] = None,
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
        self.catalog.paths.check_is_valid(path, expected=None)
        dir = self.catalog.paths[path.parent]

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
            dir._id, path.name, schema, primary_key=primary_key, num_retained_versions=num_retained_versions, comment=comment)
        self.catalog.paths[path] = tbl
        _logger.info(f'Created table `{path_str}`.')
        return tbl

    def import_parquet(
        self,
        table_path: str,
        *,
        parquet_path: str,
        schema_override: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> catalog.InsertableTable:
        """Create a new `InsertableTable` from a Parquet file or set of files. Requires pyarrow to be installed.
        Args:
            path_str: Path to the table within pixeltable.
            parquet_path: Path to an individual Parquet file or directory of Parquet files.
            schema_override: Optional dictionary mapping column names to column type to override the default
                            schema inferred from the Parquet file. The column type should be a pixeltable ColumnType.
                            For example, {'col_vid': VideoType()}, rather than {'col_vid': StringType()}.
                            Any fields not provided explicitly will map to types with `pixeltable.utils.parquet.parquet_schema_to_pixeltable_schema`
            kwargs: Additional arguments to pass to `Client.create_table`.

        Returns:
            The newly created table. The table will have loaded the data from the Parquet file(s).
        """
        from pixeltable.utils import parquet

        return parquet.import_parquet(
            self,
            table_path=table_path,
            parquet_path=parquet_path,
            schema_override=schema_override,
            **kwargs,
        )

    def import_huggingface_dataset(
        self,
        table_path: str,
        dataset: Union['datasets.Dataset', 'datasets.DatasetDict'],
        *,
        column_name_for_split: Optional[str] = 'split',
        schema_override: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> catalog.InsertableTable:
        """Create a new `InsertableTable` from a Huggingface dataset, or dataset dict with multiple splits.
            Requires datasets library to be installed.

        Args:
            path_str: Path to the table.
            dataset: Huggingface datasts.Dataset or datasts.DatasetDict to insert into the table.
            column_name_for_split: column name to use for split information. If None, no split information will be stored.
            schema_override: Optional dictionary mapping column names to column type to override the corresponding defaults from
            `pixeltable.utils.hf_datasets.huggingface_schema_to_pixeltable_schema`. The column type should be a pixeltable ColumnType.
            For example, {'col_vid': VideoType()}, rather than {'col_vid': StringType()}.

            kwargs: Additional arguments to pass to `create_table`.

        Returns:
            The newly created table. The table will have loaded the data from the dataset.
        """
        from pixeltable.utils import hf_datasets

        return hf_datasets.import_huggingface_dataset(
            self,
            table_path,
            dataset,
            column_name_for_split=column_name_for_split,
            schema_override=schema_override,
            **kwargs,
        )

    def create_view(
            self, path_str: str, base: catalog.Table, *, schema: Optional[Dict[str, Any]] = None,
            filter: Optional[Predicate] = None,
            is_snapshot: bool = False, iterator_class: Optional[Type[ComponentIterator]] = None,
            iterator_args: Optional[Dict[str, Any]] = None, num_retained_versions: int = 10, comment: str = '',
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
            self.catalog.paths.check_is_valid(path, expected=None)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        dir = self.catalog.paths[path.parent]

        if schema is None:
            schema = {}
        view = catalog.View.create(
            dir._id, path.name, base=base, schema=schema, predicate=filter, is_snapshot=is_snapshot,
            iterator_cls=iterator_class, iterator_args=iterator_args, num_retained_versions=num_retained_versions, comment=comment)
        self.catalog.paths[path] = view
        _logger.info(f'Created view `{path_str}`.')
        return view

    def get_table(self, path: str) -> catalog.Table:
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
        self.catalog.paths.check_is_valid(p, expected=catalog.Table)
        obj = self.catalog.paths[p]
        return obj

    def move(self, path: str, new_path: str) -> None:
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
        self.catalog.paths.check_is_valid(p, expected=catalog.SchemaObject)
        new_p = catalog.Path(new_path)
        self.catalog.paths.check_is_valid(new_p, expected=None)
        obj = self.catalog.paths[p]
        self.catalog.paths.move(p, new_p)
        new_dir = self.catalog.paths[new_p.parent]
        obj.move(new_p.name, new_dir._id)

    def list_tables(self, dir_path: str = '', recursive: bool = True) -> List[str]:
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
        self.catalog.paths.check_is_valid(path, expected=catalog.Dir)
        return [str(p) for p in self.catalog.paths.get_children(path, child_type=catalog.Table, recursive=recursive)]

    def drop_table(self, path: str, force: bool = False, ignore_errors: bool = False) -> None:
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
            self.catalog.paths.check_is_valid(path_obj, expected=catalog.Table)
        except Exception as e:
            if ignore_errors:
                _logger.info(f'Skipped table `{path}` (does not exist).')
                return
            else:
                raise e
        tbl = self.catalog.paths[path_obj]
        if len(self.catalog.tbl_dependents[tbl._id]) > 0:
            dependent_paths = [self.get_path(dep) for dep in self.catalog.tbl_dependents[tbl._id]]
            raise excs.Error(f'Table {path} has dependents: {", ".join(dependent_paths)}')
        tbl._drop()
        del self.catalog.paths[path_obj]
        _logger.info(f'Dropped table `{path}`.')

    def create_dir(self, path_str: str, ignore_errors: bool = False) -> None:
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
            self.catalog.paths.check_is_valid(path, expected=None)
            parent = self.catalog.paths[path.parent]
            assert parent is not None
            with orm.Session(Env.get().engine, future=True) as session:
                dir_md = schema.DirMd(name=path.name)
                dir_record = schema.Dir(parent_id=parent._id, md=dataclasses.asdict(dir_md))
                session.add(dir_record)
                session.flush()
                assert dir_record.id is not None
                self.catalog.paths[path] = catalog.Dir(dir_record.id, parent._id, path.name)
                session.commit()
                _logger.info(f'Created directory `{path_str}`.')
                print(f'Created directory `{path_str}`.')
        except excs.Error as e:
            if ignore_errors:
                return
            else:
                raise e

    def rm_dir(self, path_str: str) -> None:
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
        self.catalog.paths.check_is_valid(path, expected=catalog.Dir)

        # make sure it's empty
        if len(self.catalog.paths.get_children(path, child_type=None, recursive=True)) > 0:
            raise excs.Error(f'Directory {path_str} is not empty')
        # TODO: figure out how to make force=True work in the presence of snapshots
        #        # delete tables
        #        for tbl_path in self.paths.get_children(path, child_type=MutableTable, recursive=True):
        #            self.drop_table(str(tbl_path), force=True)
        #        # rm subdirs
        #        for dir_path in self.paths.get_children(path, child_type=Dir, recursive=False):
        #            self.rm_dir(str(dir_path), force=True)

        with Env.get().engine.begin() as conn:
            dir = self.catalog.paths[path]
            conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.id == dir._id))
        del self.catalog.paths[path]
        _logger.info(f'Removed directory {path_str}')

    def list_dirs(self, path_str: str = '', recursive: bool = True) -> List[str]:
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
        self.catalog.paths.check_is_valid(path, expected=catalog.Dir)
        return [str(p) for p in self.catalog.paths.get_children(path, child_type=catalog.Dir, recursive=recursive)]

    # TODO: for now, named functions are deprecated, until we understand the use case and requirements better
    # def create_function(self, path_str: str, fn: func.Function) -> None:
    #     """Create a stored function.
    #
    #     Args:
    #         path_str: path where the function gets stored
    #         func: previously created Function object
    #
    #     Raises:
    #         Error: if the path already exists or the parent is not a directory
    #
    #     Examples:
    #         Create a function ``detect()`` that takes an image and returns a JSON object, and store it in ``my_dir``:
    #
    #         >>> @pxt.udf(param_types=[ImageType()], return_type=JsonType())
    #         ... def detect(img):
    #         ... ...
    #         >>> cl.create_function('my_dir.detect', detect)
    #     """
    #     if fn.is_module_function:
    #         raise excs.Error(f'Cannot create a named function for a library function')
    #     path = catalog.Path(path_str)
    #     self.catalog.paths.check_is_valid(path, expected=None)
    #     dir = self.catalog.paths[path.parent]
    #
    #     func.FunctionRegistry.get().create_function(fn, dir._id, path.name)
    #     self.catalog.paths[path] = catalog.NamedFunction(fn.id, dir._id, path.name)
    #     fn.md.fqn = str(path)
    #     _logger.info(f'Created function {path_str}')
    #
    # def update_function(self, path_str: str, fn: func.Function) -> None:
    #     """Update the implementation of a stored function.
    #
    #     Args:
    #         path_str: path to the function to be updated
    #         func: new function implementation
    #
    #     Raises:
    #         Error: if the path does not exist or ``func`` has a different signature than the stored function.
    #     """
    #     if fn.is_module_function:
    #         raise excs.Error(f'Cannot update a named function to a library function')
    #     path = catalog.Path(path_str)
    #     self.catalog.paths.check_is_valid(path, expected=catalog.NamedFunction)
    #     named_fn = self.catalog.paths[path]
    #     f = func.FunctionRegistry.get().get_function(id=named_fn._id)
    #     if f.md.signature != fn.md.signature:
    #         raise excs.Error(
    #             f'The function signature cannot be changed. The existing signature is {f.md.signature}')
    #     if f.is_aggregate != fn.is_aggregate:
    #         raise excs.Error(f'Cannot change an aggregate function into a non-aggregate function and vice versa')
    #     func.FunctionRegistry.get().update_function(named_fn._id, fn)
    #     _logger.info(f'Updated function {path_str}')
    #
    # def get_function(self, path_str: str) -> func.Function:
    #     """Get a handle to a stored function.
    #
    #     Args:
    #         path_str: path to the function
    #
    #     Returns:
    #         Function object
    #
    #     Raises:
    #         Error: if the path does not exist or is not a function
    #
    #     Examples:
    #         >>> detect = cl.get_function('my_dir.detect')
    #     """
    #     path = catalog.Path(path_str)
    #     self.catalog.paths.check_is_valid(path, expected=catalog.NamedFunction)
    #     named_fn = self.catalog.paths[path]
    #     assert isinstance(named_fn, catalog.NamedFunction)
    #     fn = func.FunctionRegistry.get().get_function(id=named_fn._id)
    #     fn.md.fqn = str(path)
    #     return fn
    #
    # def drop_function(self, path_str: str, ignore_errors: bool = False) -> None:
    #     """Deletes stored function.
    #
    #     Args:
    #         path_str: path to the function
    #         ignore_errors: if True, does not raise if the function does not exist
    #
    #     Raises:
    #         Error: if the path does not exist or is not a function
    #
    #     Examples:
    #         >>> cl.drop_function('my_dir.detect')
    #     """
    #     path = catalog.Path(path_str)
    #     try:
    #         self.catalog.paths.check_is_valid(path, expected=catalog.NamedFunction)
    #     except excs.Error as e:
    #         if ignore_errors:
    #             return
    #         else:
    #             raise e
    #     named_fn = self.catalog.paths[path]
    #     func.FunctionRegistry.get().delete_function(named_fn._id)
    #     del self.catalog.paths[path]
    #     _logger.info(f'Dropped function {path_str}')

