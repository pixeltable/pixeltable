from typing import List, Optional, Dict, Tuple
import pandas as pd
import logging
import dataclasses
from uuid import UUID

import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable.catalog import \
    TableVersion, SchemaObject, MutableTable, TableSnapshot, View, Table, TableBase, Dir, Column, NamedFunction, Path,\
    PathDict, init_catalog
from pixeltable.metadata import schema
from pixeltable.env import Env
from pixeltable.function import FunctionRegistry, Function
from pixeltable import exceptions as exc
from pixeltable.exprs import Predicate

__all__ = [
    'Client',
]


_logger = logging.getLogger('pixeltable')

class Client:
    """
    Client for interacting with a Pixeltable environment.
    """

    def __init__(self) -> None:
        """Constructs a client.
        """
        Env.get().set_up()
        FunctionRegistry.get().register_nos_functions()
        init_catalog()
        self.paths = PathDict()
        self._load_table_versions()
        self._load_functions()

    def _load_table_versions(self) -> None:
        # key: [id, version]
        # - for the current/live version of a table, version is None
        # - however, TableVersion.version will be set correctly
        self.tbl_versions: Dict[Tuple[UUID, int], TableVersion] = {}
        # load TableVersions
        with orm.Session(Env.get().engine, future=True) as session:
            # load current versions of all non-view tables
            q = session.query(schema.Table, schema.TableSchemaVersion) \
                .select_from(schema.Table) \
                .join(schema.TableSchemaVersion) \
                .where(schema.Table.base_id == None) \
                .where(sql.text((
                f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}")))
            for tbl_record, schema_version_record in q.all():
                tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
                schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
                instance = TableVersion(tbl_record.id, None, tbl_md, tbl_md.current_version, schema_version_md)
                self.tbl_versions[(tbl_record.id, None)] = instance
                tbl = MutableTable(tbl_record.dir_id, instance)
                self.paths.add_schema_obj(tbl.dir_id, instance.name, tbl)

            # load bases for views over snapshots;
            # do this ordered by creation ts so that we can resolve base references in one pass
            ViewAlias = orm.aliased(schema.Table)
            q = session.query(schema.Table, ViewAlias.base_version, schema.TableSchemaVersion) \
                .select_from(schema.Table) \
                .join(ViewAlias, schema.Table.id == ViewAlias.base_id) \
                .join(
                schema.TableVersion,
                sql.and_(
                    schema.TableVersion.tbl_id == schema.Table.id,
                    schema.TableVersion.version == ViewAlias.base_version)) \
                .join(schema.TableSchemaVersion, schema.TableSchemaVersion.tbl_id == schema.Table.id) \
                .where(sql.text((
                f"({schema.TableVersion.__table__}.md->>'schema_version')::int = "
                f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}"))) \
                .order_by(sql.text(f"({schema.TableVersion.__table__}.md->>'created_at')::float"))
            for tbl_record, tbl_version, schema_version_record in q.all():
                assert tbl_version is not None
                assert (tbl_record.id, tbl_version) not in self.tbl_versions
                schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
                if tbl_record.base_id is not None:
                    base = self.tbl_versions[(tbl_record.base_id, tbl_record.base_version)]
                else:
                    base = None
                tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
                instance = TableVersion(tbl_record.id, base, tbl_md, tbl_version, schema_version_md)
                self.tbl_versions[(tbl_record.id, tbl_version)] = instance

            # load views
            q = session.query(schema.Table, schema.TableSchemaVersion) \
                .select_from(schema.Table) \
                .join(schema.TableVersion, schema.Table.id == schema.TableVersion.tbl_id) \
                .join(schema.TableSchemaVersion) \
                .where(schema.Table.base_id != None) \
                .where(sql.text((
                    f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                    f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}"))) \
                .where(sql.text((
                    f"({schema.Table.__table__}.md->>'current_version')::int = "
                    f"{schema.TableVersion.__table__}.{schema.TableVersion.version.name}"))) \
                .order_by(sql.text(f"({schema.TableVersion.__table__}.md->>'created_at')::float"))
            for tbl_record, schema_version_record in q.all():
                base_id, base_version = tbl_record.base_id, tbl_record.base_version
                assert (base_id, tbl_record.base_version) in self.tbl_versions
                base = self.tbl_versions[(base_id, base_version)]
                tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
                schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
                instance = TableVersion(tbl_record.id, base, tbl_md, tbl_md.current_version, schema_version_md)
                self.tbl_versions[(tbl_record.id, None)] = instance
                view = View(tbl_record.dir_id, instance)
                self.paths.add_schema_obj(view.dir_id, instance.name, view)

            # load table versions referenced by snapshots; do this ordered by creation ts so that we can resolve base
            # references in one pass
            q = session.query(schema.TableSnapshot, schema.Table, schema.TableSchemaVersion) \
                .select_from(schema.TableSnapshot) \
                .join(schema.Table) \
                .join(schema.TableVersion,
                      sql.and_(
                          schema.TableSnapshot.tbl_id == schema.TableVersion.tbl_id,
                          schema.TableSnapshot.tbl_version == schema.TableVersion.version)) \
                .join(schema.TableSchemaVersion, schema.TableSchemaVersion.tbl_id == schema.TableSnapshot.tbl_id) \
                .where(sql.text((
                    f"({schema.TableVersion.__table__}.md->>'schema_version')::int = "
                    f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}"))) \
                .order_by(sql.text(f"({schema.TableVersion.__table__}.md->>'created_at')::float"))
            for snapshot_record, tbl_record, schema_version_record in q.all():
                tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
                schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
                if tbl_record.base_id is not None:
                    base = self.tbl_versions[(tbl_record.base_id, tbl_record.base_version)]
                else:
                    base = None
                instance = TableVersion(
                    tbl_record.id, base, tbl_md, snapshot_record.tbl_version, schema_version_md)
                self.tbl_versions[(snapshot_record.tbl_id, snapshot_record.tbl_version)] = instance
                snapshot_md = schema.md_from_dict(schema.TableSnapshotMd, snapshot_record.md)
                snapshot = TableSnapshot(snapshot_record.id, snapshot_record.dir_id, snapshot_md.name, instance)
                self.paths.add_schema_obj(snapshot_record.dir_id, snapshot_md.name, snapshot)

    def _load_functions(self) -> None:
        # load Function metadata; doesn't load the actual callable, which can be large and is only done on-demand by the
        # FunctionRegistry
        with orm.Session(Env.get().engine, future=True) as session:
            q = session.query(schema.Function.id, schema.Function.dir_id, schema.Function.md) \
                .where(sql.text(f"({schema.Function.__table__}.md->>'name')::text IS NOT NULL"))
            for id, dir_id, md in q.all():
                assert 'name' in md
                name = md['name']
                assert name is not None
                named_fn = NamedFunction(id, dir_id, name)
                self.paths.add_schema_obj(dir_id, name, named_fn)

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
        func_info = FunctionRegistry.get().list_functions()
        paths = ['.'.join(info.fqn.split('.')[:-1]) for info in func_info]
        names = [info.fqn.split('.')[-1] for info in func_info]
        pd_df = pd.DataFrame({
            'Path': paths,
            'Function Name': names,
            'Parameters': [
                ', '.join([param_name + ': ' + str(param_type) for param_name, param_type in info.signature.parameters.items()]) for info in func_info
            ],
            'Return Type': [str(info.signature.get_return_type()) for info in func_info],
        })
        pd_df = pd_df.style.set_properties(**{'text-align': 'left'}) \
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])  # center-align headings
        return pd_df.hide(axis='index')

    def get_path(self, schema_obj: SchemaObject) -> str:
        """Returns the path to a SchemaObject.

        Args:
            schema_obj: SchemaObject to get the path for.

        Returns:
            Path to the SchemaObject.
        """
        path_elements: List[str] = []
        dir_id = schema_obj.dir_id
        while dir_id is not None:
            dir = self.paths.get_schema_obj(dir_id)
            if dir.dir_id is None:
                # this is the root dir with name '', which we don't want to include in the path
                break
            path_elements.insert(0, dir.name)
            dir_id = dir.dir_id
        path_elements.append(schema_obj.name)
        return '.'.join(path_elements)

    def create_table(
            self, path_str: str, schema: List[Column], num_retained_versions: int = 10,
            extract_frames_from: Optional[str] = None, extracted_frame_col: Optional[str] = None,
            extracted_frame_idx_col: Optional[str] = None, extracted_fps: Optional[int] = None,
    ) -> MutableTable:
        """Create a new table in the database.

        Args:
            path_str: Path to the table.
            schema: List of Columns in the table.
            num_retained_versions: Number of versions of the table to retain.
            extract_frames_from: Name of the video column from which to extract frames.
            extracted_frame_col: Name of the image column in which to store the extracted frames.
            extracted_frame_idx_col: Name of the int column in which to store the frame indices.
            extracted_fps: Frame rate at which to extract frames. 0: extract all frames.

        Returns:
            The newly created table.

        Raises:
            Error: if the path already exists or is invalid.

        Examples:
            Create a table with an int and a string column:

            >>> table = cl.create_table('my_table', schema=[Column('col1', IntType()), Column('col2', StringType())])

            Create a table to store videos with automatic frame extraction. This requires a minimum of 3 columns:
            a video column, an image column to store the extracted frames, and an int column to store the frame
            indices.

            >>> table = cl.create_table('my_table',
            ... schema=[Column('video', VideoType()), Column('frame', ImageType()), Column('frame_idx', IntType())],
            ... extract_frames_from='video', extracted_frame_col='frame', extracted_frame_idx_col='frame_idx',
            ... extracted_fps=1)
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None)
        dir = self.paths[path.parent]

        # make sure frame extraction params are either fully present or absent
        frame_extraction_param_count = int(extract_frames_from is not None) + int(extracted_frame_col is not None) \
                                       + int(extracted_frame_idx_col is not None) + int(extracted_fps is not None)
        if frame_extraction_param_count != 0 and frame_extraction_param_count != 4:
            raise exc.Error(
                'Frame extraction requires that all parameters (extract_frames_from, extracted_frame_col, '
                'extracted_frame_idx_col, extracted_fps) be specified')
        if extracted_fps is not None and extracted_fps < 0:
            raise exc.Error('extracted_fps must be >= 0')
        tbl = MutableTable.create(
            dir.id, path.name, schema, num_retained_versions, extract_frames_from, extracted_frame_col,
            extracted_frame_idx_col, extracted_fps)
        self.tbl_versions[(tbl.id, None)] = tbl.tbl_version
        self.paths[path] = tbl
        _logger.info(f'Created table {path_str}')
        return tbl

    def create_snapshot(self, snapshot_path: str, tbl_path: str, ignore_errors: bool = False) -> TableSnapshot:
        """Create a snapshot of a table.

        Args:
            snapshot_path: Path to the snapshot.
            tbl_path: Path to the table.
            ignore_errors: Whether to ignore errors if the snapshot already exists.

        Raises:
            Error: If snapshot_path already exists or the parent does not exist.
        """
        try:
            snapshot_path_obj = Path(snapshot_path)
            self.paths.check_is_valid(snapshot_path_obj, expected=None)
            tbl_path_obj = Path(tbl_path)
            self.paths.check_is_valid(tbl_path_obj, expected=Table)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        tbl = self.paths[tbl_path_obj]
        assert isinstance(tbl, Table)
        # tbl.tbl_version is the live/mutable version of the table, but we need a snapshot of it
        tbl_version = tbl.tbl_version
        if (tbl_version.id, tbl_version.version) not in self.tbl_versions:
            # create an immutable copy
            tbl_version = tbl_version.create_snapshot_copy()
            self.tbl_versions[(tbl_version.id, tbl_version.version)] = tbl_version
        dir = self.paths[snapshot_path_obj.parent]
        snapshot = TableSnapshot.create(dir.id, snapshot_path_obj.name, tbl_version)
        self.paths[snapshot_path_obj] = snapshot
        _logger.info(f'Created snapshot {snapshot_path}')
        return snapshot

    def create_view(
            self, path_str: str, base: TableBase, schema: List[Column] = [], filter: Optional[Predicate] = None,
            num_retained_versions: int = 10, ignore_errors: bool = False) -> View:
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=None)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        dir = self.paths[path.parent]

        view = View.create(dir.id, path.name, base.tbl_version, schema, filter, num_retained_versions)
        self.tbl_versions[(view.id, None)] = view.tbl_version
        self.paths[path] = view
        _logger.info(f'Created view {path_str}')
        return view

    def get_table(self, path: str) -> TableBase:
        """Get a handle to a table (including views and snapshots) from the database.

        Args:
            path: Path to the table.

        Returns:
            A :py:class:`MutableTable` or :py:class:`View` or :py:class:`TableSnapshot` object.

        Raises:
            Error: If the path does not exist or does not designate a table.

        Example:
            Get handle for a table in the top-level directory:

            >>> table = cl.get_table('my_table')

            For a table in a subdirectory:

            >>> table = cl.get_table('subdir.my_table')

            For a snapshot in the top-level directory:

            >>> table = cl.get_table('my_snapshot')
        """
        p = Path(path)
        self.paths.check_is_valid(p, expected=TableBase)
        obj = self.paths[p]
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
        p = Path(path)
        self.paths.check_is_valid(p, expected=SchemaObject)
        new_p = Path(new_path)
        self.paths.check_is_valid(new_p, expected=None)
        obj = self.paths[p]
        self.paths.move(p, new_p)
        new_dir = self.paths[new_p.parent]
        obj.move(new_p.name, new_dir.id)

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
        path = Path(dir_path, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=Dir)
        return [str(p) for p in self.paths.get_children(path, child_type=TableBase, recursive=recursive)]

    def drop_table(self, path: str, force: bool = False, ignore_errors: bool = False) -> None:
        """Drop a table from the database.

        Args:
            path: Path to the table.
            force: Whether to drop the table even if it has unsaved changes.
            ignore_errors: Whether to ignore errors if the table does not exist.

        Raises:
            Error: If the path does not exist or does not designate a table and ignore_errors is False.

        Example:
            >>> cl.drop_table('my_table')
        """
        path_obj = Path(path)
        try:
            self.paths.check_is_valid(path_obj, expected=TableBase)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        tbl = self.paths[path_obj]
        tbl.drop()
        del self.paths[path_obj]
        _logger.info(f'Dropped table {path}')

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
            path = Path(path_str)
            self.paths.check_is_valid(path, expected=None)
            parent = self.paths[path.parent]
            assert parent is not None
            with orm.Session(Env.get().engine, future=True) as session:
                dir_md = schema.DirMd(name=path.name)
                dir_record = schema.Dir(parent_id=parent.id, md=dataclasses.asdict(dir_md))
                session.add(dir_record)
                session.flush()
                assert dir_record.id is not None
                self.paths[path] = Dir(dir_record.id, parent.id, path.name)
                session.commit()
                _logger.info(f'Created directory {path_str}')
        except exc.Error as e:
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
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Dir)

        # make sure it's empty
        if len(self.paths.get_children(path, child_type=None, recursive=True)) > 0:
            raise exc.Error(f'Directory {path_str} is not empty')
        # TODO: figure out how to make force=True work in the presence of snapshots
        #        # delete tables
        #        for tbl_path in self.paths.get_children(path, child_type=Table, recursive=True):
        #            self.drop_table(str(tbl_path), force=True)
        #        # rm subdirs
        #        for dir_path in self.paths.get_children(path, child_type=Dir, recursive=False):
        #            self.rm_dir(str(dir_path), force=True)

        with Env.get().engine.begin() as conn:
            dir = self.paths[path]
            conn.execute(sql.delete(schema.Dir.__table__).where(schema.Dir.id == dir.id))
        del self.paths[path]
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

        Example:
            >>> cl.list_dirs('my_dir', recursive=True)
            ['my_dir', 'my_dir.sub_dir1']
        """
        path = Path(path_str, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=Dir)
        return [str(p) for p in self.paths.get_children(path, child_type=Dir, recursive=recursive)]

    def create_function(self, path_str: str, func: Function) -> None:
        """Create a stored function.

        Args:
            path_str: path where the function gets stored
            func: previously created Function object

        Raises:
            Error: if the path already exists or the parent is not a directory
        Examples:
            Create a function ``detect()`` that takes an image and returns a JSON object, and store it in ``my_dir``:

            >>> pt.function(param_types=[ImageType()], return_type=JsonType())
            ... def detect(img):
            ... ...
            >>> cl.create_function('my_dir.detect', detect)
        """
        if func.is_library_function:
            raise exc.Error(f'Cannot create a named function for a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None)
        dir = self.paths[path.parent]

        FunctionRegistry.get().create_function(func, dir.id, path.name)
        self.paths[path] = NamedFunction(func.id, dir.id, path.name)
        func.md.fqn = str(path)
        _logger.info(f'Created function {path_str}')

    def update_function(self, path_str: str, func: Function) -> None:
        """Update the implementation of a stored function.

        Args:
            path_str: path to the function to be updated
            func: new function implementation

        Raises:
            Error: if the path does not exist or ``func`` has a different signature than the stored function.
        """
        if func.is_library_function:
            raise exc.Error(f'Cannot update a named function to a library function')
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        f = FunctionRegistry.get().get_function(id=named_fn.id)
        if f.md.signature != func.md.signature:
            raise exc.Error(
                f'The function signature cannot be changed. The existing signature is {f.md.signature}')
        if f.is_aggregate != func.is_aggregate:
            raise exc.Error(f'Cannot change an aggregate function into a non-aggregate function and vice versa')
        FunctionRegistry.get().update_function(named_fn.id, func)
        _logger.info(f'Updated function {path_str}')

    def get_function(self, path_str: str) -> Function:
        """Get a handle to a stored function.

        Args:
            path_str: path to the function

        Returns:
            Function object

        Raises:
            Error: if the path does not exist or is not a function

        Example:
            >>> detect = cl.get_function('my_dir.detect')
        """
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=NamedFunction)
        named_fn = self.paths[path]
        assert isinstance(named_fn, NamedFunction)
        func = FunctionRegistry.get().get_function(id=named_fn.id)
        func.md.fqn = str(path)
        return func

    def drop_function(self, path_str: str, ignore_errors: bool = False) -> None:
        """Deletes stored function.

        Args:
            path_str: path to the function
            ignore_errors: if True, does not raise if the function does not exist

        Raises:
            Error: if the path does not exist or is not a function

        Example:
            >>> cl.drop_function('my_dir.detect')
        """
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=NamedFunction)
        except exc.Error as e:
            if ignore_errors:
                return
            else:
                raise e
        named_fn = self.paths[path]
        FunctionRegistry.get().delete_function(named_fn.id)
        del self.paths[path]
        _logger.info(f'Dropped function {path_str}')

    def reset_catalog(self) -> None:
        """Delete everything. Test-only.

        :meta private:
        """
        with Env.get().engine.begin() as conn:
            conn.execute(sql.delete(schema.TableSnapshot.__table__))
            conn.execute(sql.delete(schema.TableSchemaVersion.__table__))
            conn.execute(sql.delete(schema.TableVersion.__table__))
            conn.execute(sql.delete(schema.Table.__table__))
            conn.execute(sql.delete(schema.Function.__table__))
            conn.execute(sql.delete(schema.Dir.__table__))
            # delete all data tables
            # TODO: also deleted generated images
            tbl_paths = [
                p for p in self.paths.get_children(Path('', empty_is_valid=True), MutableTable, recursive=True)
            ]
            for tbl_path in tbl_paths:
                tbl = self.paths[tbl_path]
                tbl.store_tbl.drop(conn)
