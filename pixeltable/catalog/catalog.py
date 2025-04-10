from __future__ import annotations

import dataclasses
import functools
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar
from uuid import UUID

import psycopg
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.metadata.schema as schema
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator

from .dir import Dir
from .globals import IfExistsParam, IfNotExistsParam, MediaValidation
from .insertable_table import InsertableTable
from .path import Path
from .schema_object import SchemaObject
from .table import Table
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .view import View

if TYPE_CHECKING:
    from .. import DataFrame, exprs


_logger = logging.getLogger('pixeltable')


def _lock_str(for_update: bool) -> str:
    return 'X' if for_update else 'S'


# TODO: remove once the concurrent update behavior has been debugged
# def _debug_print(for_update: bool, msg: str) -> None:
#     return
#     print(f'{datetime.datetime.now()}: {_lock_str(for_update)}: {msg}')


def _unpack_row(
    row: Optional[sql.engine.Row], entities: list[type[sql.orm.decl_api.DeclarativeBase]]
) -> Optional[list[Any]]:
    """Convert a Row result into a list of entity instances.

    Assumes that the query contains a select() of exactly those entities.
    """
    if row is None:
        return None

    result: list[sql.orm.decl_api.DeclarativeBase] = []
    column_offset = 0

    for entity in entities:
        num_cols = len(entity.__table__.columns)
        data = {name: row[column_offset + i] for i, name in enumerate(entity.__table__.columns.keys())}
        inst = entity(**data)
        result.append(inst)
        column_offset += num_cols

    return result


_MAX_RETRIES = 3
T = TypeVar('T')


def _retry_loop(op: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(op)
    def loop(*args: Any, **kwargs: Any) -> T:
        num_remaining_retries = _MAX_RETRIES
        while True:
            try:
                # in order for retry to work, we need to make sure that there aren't any prior db updates
                # that are part of an ongoing transaction
                assert not Env.get().in_xact()
                with Env.get().begin_xact() as conn:
                    return op(*args, **kwargs)
            except sql.exc.DBAPIError as e:
                if isinstance(e.orig, psycopg.errors.SerializationFailure) and num_remaining_retries > 0:
                    num_remaining_retries -= 1
                    print(f'serialization failure:\n{e}')
                    print('retrying ************************************************************')
                    time.sleep(1)
                else:
                    raise

    return loop


class Catalog:
    """The functional interface to getting access to catalog objects

    All interface functions must be called in the context of a transaction, started with Env.begin().
    """

    _instance: Optional[Catalog] = None

    # key: [id, version]
    # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
    # - snapshot versions: records the version of the snapshot
    _tbl_versions: dict[tuple[UUID, Optional[int]], TableVersion]
    _tbls: dict[UUID, Table]

    @classmethod
    def get(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Remove the instance. Used for testing."""
        cls._instance = None

    def __init__(self) -> None:
        self._tbl_versions = {}
        self._tbls = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self._init_store()

    def get_dir_path(self, dir_id: UUID) -> Path:
        """Return path for directory with given id"""
        conn = Env.get().conn
        names: list[str] = []
        while True:
            q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
            # _debug_print(for_update=False, msg=f'dir id={dir_id}')
            row = conn.execute(q).one()
            dir = schema.Dir(**row._mapping)
            if dir.md['name'] == '':
                break
            names.insert(0, dir.md['name'])
            dir_id = dir.parent_id
        return Path('.'.join(names), empty_is_valid=True, allow_system_paths=True)

    @dataclasses.dataclass
    class DirEntry:
        dir: Optional[schema.Dir]
        dir_entries: dict[str, Catalog.DirEntry]
        table: Optional[schema.Table]

    @_retry_loop
    def get_dir_contents(self, dir_path: Path, recursive: bool = False) -> dict[str, DirEntry]:
        dir = self._get_schema_object(dir_path, expected=Dir, raise_if_not_exists=True)
        return self._get_dir_contents(dir._id, recursive=recursive)

    def _get_dir_contents(self, dir_id: UUID, recursive: bool = False) -> dict[str, DirEntry]:
        """Returns a dict mapping the entry names to DirEntry objects"""
        conn = Env.get().conn
        result: dict[str, Catalog.DirEntry] = {}

        q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
        # _debug_print(for_update=False, msg=f'dirs parent_id={dir_id}')
        rows = conn.execute(q).all()
        for row in rows:
            dir = schema.Dir(**row._mapping)
            dir_contents: dict[str, Catalog.DirEntry] = {}
            if recursive:
                dir_contents = self._get_dir_contents(dir.id, recursive=True)
            result[dir.md['name']] = self.DirEntry(dir=dir, dir_entries=dir_contents, table=None)

        q = sql.select(schema.Table).where(schema.Table.dir_id == dir_id)
        # _debug_print(for_update=False, msg=f'tbls parent_id={dir_id}')
        rows = conn.execute(q).all()
        for row in rows:
            tbl = schema.Table(**row._mapping)
            result[tbl.md['name']] = self.DirEntry(dir=None, dir_entries={}, table=tbl)

        return result

    @_retry_loop
    def move(self, path: Path, new_path: Path) -> None:
        _, dest_dir, src_obj = self._prepare_dir_op(
            add_dir_path=new_path.parent,
            add_name=new_path.name,
            drop_dir_path=path.parent,
            drop_name=path.name,
            raise_if_exists=True,
            raise_if_not_exists=True,
        )
        src_obj._move(new_path.name, dest_dir._id)

    def _prepare_dir_op(
        self,
        add_dir_path: Optional[Path] = None,
        add_name: Optional[str] = None,
        drop_dir_path: Optional[Path] = None,
        drop_name: Optional[str] = None,
        drop_expected: Optional[type[SchemaObject]] = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
    ) -> tuple[Optional[SchemaObject], Optional[SchemaObject], Optional[SchemaObject]]:
        """
        Validates paths and acquires locks needed for a directory operation, ie, add/drop/rename (add + drop) of a
        directory entry.

        The target entry is either a table or directory. The directory operation can include
        - adding an entry (<add_dir_path>.<add_name>)
        - dropping an entry (<drop_dir_path>.<drop_name>)

        Returns: (existing SchemaObject of add path, Dir of add path, existing SchemaObject of drop path)

        Locking protocol:
        - X locks on the immediate parent directories of the added/dropped entries; this prevents concurrent
          modifications of the parent
        - lock parent before child
        - if both add and drop (= two directories are involved), lock the directories in a pre-determined order
          (in this case, by name) in order to prevent deadlocks between concurrent directory modifications
        """
        assert (add_dir_path is None) == (add_name is None)
        assert (drop_dir_path is None) == (drop_name is None)
        dir_paths: set[Path] = set()
        if add_dir_path is not None:
            dir_paths.add(add_dir_path)
        if drop_dir_path is not None:
            dir_paths.add(drop_dir_path)

        add_dir: Optional[schema.Dir] = None
        drop_dir: Optional[schema.Dir] = None
        for p in sorted(list(dir_paths)):
            dir = self._get_dir(p, for_update=True)
            if dir is None:
                raise excs.Error(f'Directory {str(p)!r} does not exist.')
            if p == add_dir_path:
                add_dir = dir
            if p == drop_dir_path:
                drop_dir = dir

        add_obj: Optional[SchemaObject] = None
        if add_dir is not None:
            add_obj = self._get_dir_entry(add_dir.id, add_name, for_update=True)
            if add_obj is not None and raise_if_exists:
                add_path = add_dir_path.append(add_name)
                raise excs.Error(f'Path {str(add_path)!r} already exists.')

        drop_obj: Optional[SchemaObject] = None
        if drop_dir is not None:
            drop_path = drop_dir_path.append(drop_name)
            drop_obj = self._get_dir_entry(drop_dir.id, drop_name, for_update=True)
            if drop_obj is None and raise_if_not_exists:
                raise excs.Error(f'Path {str(drop_path)!r} does not exist.')
            if drop_obj is not None and drop_expected is not None and not isinstance(drop_obj, drop_expected):
                raise excs.Error(
                    f'{str(drop_path)!r} needs to be a {drop_expected._display_name()} '
                    f'but is a {type(drop_obj)._display_name()}'
                )

        add_dir_obj = Dir(add_dir.id, add_dir.parent_id, add_dir.md['name']) if add_dir is not None else None
        return add_obj, add_dir_obj, drop_obj

    def _get_dir_entry(self, dir_id: UUID, name: str, for_update: bool = False) -> Optional[SchemaObject]:
        user = Env.get().user
        conn = Env.get().conn

        # check for subdirectory
        q = sql.select(schema.Dir).where(
            schema.Dir.parent_id == dir_id, schema.Dir.md['name'].astext == name, schema.Dir.md['user'].astext == user
        )
        if for_update:
            q = q.with_for_update()
        # _debug_print(for_update, f'dir name={name!r} parent={dir_id}')
        # row = conn.execute(q).one_or_none()
        # if row is not None:
        #     dir_record = schema.Dir(**row._mapping)
        #     return Dir(dir_record.id, dir_record.parent_id, name)
        rows = conn.execute(q).all()
        if len(rows) > 1:
            assert False, rows
        if len(rows) == 1:
            dir_record = schema.Dir(**rows[0]._mapping)
            return Dir(dir_record.id, dir_record.parent_id, name)

        # check for table
        q = sql.select(schema.Table.id).where(
            schema.Table.dir_id == dir_id,
            schema.Table.md['name'].astext == name,
            schema.Table.md['user'].astext == user,
        )
        if for_update:
            q = q.with_for_update()
        # _debug_print(for_update, f'table name={name!r} parent={dir_id}')
        tbl_id = conn.execute(q).scalar_one_or_none()
        if tbl_id is not None:
            if not tbl_id in self._tbls:
                self._tbls[tbl_id] = self._load_tbl(tbl_id)
            return self._tbls[tbl_id]

        return None

    def _get_schema_object(
        self,
        path: Path,
        expected: Optional[type[SchemaObject]] = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
        for_update: bool = False,
    ) -> Optional[SchemaObject]:
        """Return the schema object at the given path, or None if it doesn't exist.

        Raises Error if
        - the parent directory doesn't exist'
        - raise_if_exists is True and the path exists
        - raise_if_not_exists is True and the path does not exist
        - expected is not None and the existing object has a different type
        """
        if path.is_root:
            # the root dir
            if expected is not None and expected is not Dir:
                raise excs.Error(
                    f'{str(path)!r} needs to be a {expected._display_name()} but is a {Dir._display_name()}'
                )
            dir = self._get_dir(path, for_update=for_update)
            if dir is None:
                raise excs.Error(f'Unknown user: {Env.get().user}')
            return Dir(dir.id, dir.parent_id, dir.md['name'])

        parent_path = path.parent
        parent_dir = self._get_dir(parent_path, for_update=False)
        if parent_dir is None:
            raise excs.Error(f'Directory {str(parent_path)!r} does not exist.')
        obj = self._get_dir_entry(parent_dir.id, path.name, for_update=for_update)

        if obj is None and raise_if_not_exists:
            raise excs.Error(f'Path {str(path)!r} does not exist.')
        elif obj is not None and raise_if_exists:
            raise excs.Error(f'Path {str(path)!r} is an existing {type(obj)._display_name()}.')
        elif obj is not None and expected is not None and not isinstance(obj, expected):
            raise excs.Error(
                f'{str(path)!r} needs to be a {expected._display_name()} but is a {type(obj)._display_name()}.'
            )
        return obj

    def get_table_by_id(self, tbl_id: UUID) -> Optional[Table]:
        if not tbl_id in self._tbls:
            tbl = self._load_tbl(tbl_id)
            if tbl is None:
                return None
            self._tbls[tbl_id] = tbl
        return self._tbls[tbl_id]

    @_retry_loop
    def create_table(
        self,
        path: Path,
        schema: dict[str, Any],
        df: 'DataFrame',
        if_exists: IfExistsParam,
        primary_key: Optional[list[str]],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
    ) -> Table:
        existing = self._handle_path_collision(path, InsertableTable, False, if_exists)
        if existing is not None:
            assert isinstance(existing, Table)
            return existing

        dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
        assert dir is not None

        tbl = InsertableTable._create(
            dir._id,
            path.name,
            schema,
            df,
            primary_key=primary_key,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation,
        )
        self._tbls[tbl._id] = tbl
        return tbl

    @_retry_loop
    def create_view(
        self,
        path: Path,
        base: TableVersionPath,
        select_list: Optional[list[tuple[exprs.Expr, Optional[str]]]],
        where: Optional[exprs.Expr],
        additional_columns: Optional[dict[str, Any]],
        is_snapshot: bool,
        iterator: Optional[tuple[type[ComponentIterator], dict[str, Any]]],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> Table:
        from pixeltable.utils.filecache import FileCache

        existing = self._handle_path_collision(path, View, is_snapshot, if_exists)
        if existing is not None:
            assert isinstance(existing, View)
            return existing

        dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
        assert dir is not None
        if iterator is None:
            iterator_class, iterator_args = None, None
        else:
            iterator_class, iterator_args = iterator
        view = View._create(
            dir._id,
            path.name,
            base=base,
            select_list=select_list,
            additional_columns=additional_columns,
            predicate=where,
            is_snapshot=is_snapshot,
            iterator_cls=iterator_class,
            iterator_args=iterator_args,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation,
        )
        FileCache.get().emit_eviction_warnings()
        self._tbls[view._id] = view
        return view

    @_retry_loop
    def create_replica(self, path: Path, md: list[schema.FullTableMd], if_exists: IfExistsParam) -> Table:
        """
        Creates table, table_version, and table_schema_version records for a replica with the given metadata.
        The metadata should be presented in standard "ancestor order", with the table being replicated at
        list position 0 and the (root) base table at list position -1.
        """
        existing = self._handle_path_collision(path, View, False, if_exists)
        if existing is not None:
            if existing._id != tbl_id:
                raise excs.Error(
                    f"An attempt was made to create a replica table at {path!r} with if_exists='ignore', "
                    'but a different table already exists at that location.'
                )
            assert isinstance(existing, View)
            return existing

        # If this table UUID already exists in the catalog, it's an error
        # TODO: Handle the case where it already exists as an anonymous table (because it was an ancestor of
        #     a different replica)
        tbl_id = UUID(md[0].tbl_md.tbl_id)
        if Catalog.get().get_table_by_id(tbl_id) is not None:
            raise excs.Error(
                f'That table has already been replicated as {self._tbls[tbl_id]._path()!r}. \n'
                f'Drop the existing replica if you wish to re-create it.'
            )

        # Ensure that the system directory exists
        self._create_dir(Path('_system', allow_system_paths=True), if_exists=IfExistsParam.IGNORE, parents=False)

        # Save replica metadata to the store for the table and all its ancestors. If one or more proper ancestors
        # do not yet exist in the store, they will be created as anonymous system tables.
        self.__save_replica_md(path, md[0])
        for ancestor_md in md[1:]:
            ancestor_id = UUID(ancestor_md.tbl_md.tbl_id)
            # TODO: Check if it already exists under a different name
            replica_path = Path(f'_system.replica_{ancestor_id.hex}', allow_system_paths=True)
            self.__save_replica_md(replica_path, ancestor_md)

        # Update the catalog (as a final step, after all DB operations completed successfully)
        # Only the table being replicated is actually visible in the catalog. The others might in fact not be
        # valid tables on their own, since their version and/or schema_version numbers might correspond to
        # TableVersion and/or TableSchemaVersion records that have not been replicated.
        self._tbls[tbl_id] = self._load_tbl(tbl_id)
        return self._tbls[tbl_id]

    def __save_replica_md(self, path: Path, md: schema.FullTableMd) -> None:
        _logger.info(f'Creating replica table at {path!r} with ID: {md.tbl_md.tbl_id}')
        dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
        assert dir is not None

        conn = Env.get().conn
        tbl_id = md.tbl_md.tbl_id

        new_tbl_md: Optional[schema.TableMd] = None
        new_version_md: Optional[schema.TableVersionMd] = None
        new_schema_version_md: Optional[schema.TableSchemaVersionMd] = None

        # We need to ensure that the table metadata in the catalog always reflects the latest observed version of
        # this table. (In particular, if this is a base table, then its table metadata need to be consistent
        # with the latest version of this table having a replicated view somewhere in the catalog.)
        q: sql.Executable = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
        existing_md_row = conn.execute(q).one_or_none()

        if existing_md_row is None:
            # No existing table, so create a new record.
            # If this is a proper ancestor of the table being replicated, it might be the case that the current_version
            # and/or current_schema_version records in the given TableMd are strictly greater than the versions found
            # in the given TableVersionMd and/or TableSchemaVersionMd. If that's the case, we need to adjust them so
            # that they point to known versions. This will cause no harm, since those proper ancestors will not be
            # directly instantiable.
            q = sql.insert(schema.Table.__table__).values(
                id=tbl_id,
                dir_id=dir._id,
                md=dataclasses.asdict(
                    dataclasses.replace(
                        md.tbl_md,
                        name=path.name,
                        user=Env.get().user,
                        is_replica=True,
                        current_version=md.version_md.version,
                        current_schema_version=md.schema_version_md.schema_version,
                    )
                ),
            )
            conn.execute(q)
        elif md.tbl_md.current_version > existing_md_row.md['current_version']:
            # New metadata is more recent than the metadata currently stored in the DB; we'll update the record
            # in place in the DB.
            # The same consideration applies to current_version etc. as before, except now we also need to account for
            # other versions already in the DB, which might be more recent.
            new_current_version = max(existing_md_row.md['current_version'], md.version_md.version)
            new_current_schema_version = max(
                existing_md_row.md['current_schema_version'], md.schema_version_md.schema_version
            )
            new_tbl_md = dataclasses.replace(
                md.tbl_md,
                name=path.name,
                user=Env.get().user,
                is_replica=True,
                current_version=new_current_version,
                current_schema_version=new_current_schema_version,
            )

        # Now see if a TableVersion record already exists in the DB for this table version. If not, insert it. If
        # it already exists, check that the existing record is identical to the new one.
        q = (
            sql.select(schema.TableVersion.md)
            .where(schema.TableVersion.tbl_id == tbl_id)
            .where(sql.text(f"({schema.TableVersion.__table__}.md->>'version')::int = {md.version_md.version}"))
        )
        existing_version_md_row = conn.execute(q).one_or_none()
        if existing_version_md_row is None:
            new_version_md = md.version_md
        else:
            existing_version_md = schema.md_from_dict(schema.TableVersionMd, existing_version_md_row.md)
            if existing_version_md != md.version_md:
                raise excs.Error(
                    f'The version metadata for the replica {path!r}:{md.version_md.version} is inconsistent with '
                    'the metadata recorded from a prior replica.\n'
                    'This is likely due to data corruption in the replicated table.'
                )

        # Do the same thing for TableSchemaVersion.
        q = (
            sql.select(schema.TableSchemaVersion.md)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
            .where(
                sql.text(
                    f"({schema.TableSchemaVersion.__table__}.md->>'schema_version')::int = "
                    f'{md.schema_version_md.schema_version}'
                )
            )
        )
        existing_schema_version_md_row = conn.execute(q).one_or_none()
        if existing_schema_version_md_row is None:
            new_schema_version_md = md.schema_version_md
        else:
            existing_schema_version_md = schema.md_from_dict(
                schema.TableSchemaVersionMd, existing_schema_version_md_row.md
            )
            if existing_schema_version_md != md.schema_version_md:
                raise excs.Error(
                    f'The schema version metadata for the replica {path!r}:{md.schema_version_md.schema_version} '
                    'is inconsistent with the metadata recorded from a prior replica.\n'
                    'This is likely due to data corruption in the replicated table.'
                )

        self.save_tbl_md(UUID(tbl_id), new_tbl_md, new_version_md, new_schema_version_md)

    @_retry_loop
    def get_table(self, path: Path) -> Table:
        obj = Catalog.get()._get_schema_object(path, expected=Table, raise_if_not_exists=True)
        assert isinstance(obj, Table)
        obj._tbl_version.get().ensure_md_loaded()
        return obj

    @_retry_loop
    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        _, _, src_obj = self._prepare_dir_op(
            drop_dir_path=path.parent,
            drop_name=path.name,
            drop_expected=Table,
            raise_if_not_exists=if_not_exists == IfNotExistsParam.ERROR and not force,
        )
        if src_obj is None:
            _logger.info(f'Skipped table {str(path)!r} (does not exist).')
            return
        assert isinstance(src_obj, Table)
        self._drop_tbl(src_obj, force=force, is_replace=False)

    def _drop_tbl(self, tbl: Table, force: bool, is_replace: bool) -> None:
        """
        Drop the table (and recursively its views, if force == True).

        Locking protocol:
        - X-lock base before X-locking any view
        - deadlock-free wrt to TableVersion.insert() (insert propagation also proceeds top-down)
        - X-locks parent dir prior to calling TableVersion.drop(): prevent concurrent creation of another SchemaObject
          in the same directory with the same name (which could lead to duplicate names if we get rolled back)
        """
        view_ids = self.get_view_ids(tbl._id, for_update=True)
        if len(view_ids) > 0:
            if not force:
                is_snapshot = tbl._tbl_version_path.is_snapshot()
                obj_type_str = 'Snapshot' if is_snapshot else tbl._display_name().capitalize()
                msg: str
                if is_replace:
                    msg = (
                        f'{obj_type_str} {tbl._path()} already exists and has dependents. '
                        "Use `if_exists='replace_force'` to replace it."
                    )
                else:
                    msg = f'{obj_type_str} {tbl._path()} has dependents.'
                raise excs.Error(msg)

            for view_id in view_ids:
                view = self.get_table_by_id(view_id)
                self._drop_tbl(view, force=force, is_replace=is_replace)

        _ = self.get_dir(tbl._dir_id, for_update=True)  # X-lock the parent directory
        tbl._drop()
        assert tbl._id in self._tbls
        del self._tbls[tbl._id]
        _logger.info(f'Dropped table `{tbl._path()}`.')

    @_retry_loop
    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        return self._create_dir(path, if_exists, parents)

    def _create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        # existing = self._handle_path_collision(path, Dir, False, if_exists)
        # if existing is not None:
        #     assert isinstance(existing, Dir)
        #     return existing
        #
        # parent = self._get_schema_object(path.parent)
        # assert parent is not None
        # dir = Dir._create(parent._id, path.name)
        # Env.get().console_logger.info(f'Created directory {str(path)!r}.')
        # return dir

        if parents:
            # start walking down from the root
            last_parent: Optional[SchemaObject] = None
            for ancestor in path.ancestors():
                ancestor_obj = self._get_schema_object(ancestor, expected=Dir)
                assert ancestor_obj is not None or last_parent is not None
                last_parent = Dir._create(last_parent._id, ancestor.name) if ancestor_obj is None else ancestor_obj
            parent = last_parent
        else:
            parent = self._get_schema_object(path.parent)
        existing = self._handle_path_collision(path, Dir, False, if_exists)
        if existing is not None:
            assert isinstance(existing, Dir)
            return existing
        assert parent is not None
        dir = Dir._create(parent._id, path.name)
        Env.get().console_logger.info(f'Created directory {str(path)!r}.')
        return dir

    @_retry_loop
    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        _, _, schema_obj = self._prepare_dir_op(
            drop_dir_path=path.parent,
            drop_name=path.name,
            drop_expected=Dir,
            raise_if_not_exists=if_not_exists == IfNotExistsParam.ERROR and not force,
        )
        if schema_obj is None:
            _logger.info(f'Directory {str(path)!r} does not exist; skipped drop_dir().')
            return
        self._drop_dir(schema_obj._id, path, force=force)

    def _drop_dir(self, dir_id: UUID, dir_path: Path, force: bool = False) -> None:
        conn = Env.get().conn
        if not force:
            # check for existing entries
            q = sql.select(sql.func.count()).select_from(schema.Dir).where(schema.Dir.parent_id == dir_id)
            num_subdirs = conn.execute(q).scalar()
            q = sql.select(sql.func.count()).select_from(schema.Table).where(schema.Table.dir_id == dir_id)
            num_tbls = conn.execute(q).scalar()
            if num_subdirs + num_tbls > 0:
                raise excs.Error(f'Directory {str(dir_path)!r} is not empty.')

        # drop existing subdirs
        dir_q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id).with_for_update()
        for row in conn.execute(dir_q).all():
            self._drop_dir(row.id, dir_path.append(row.md['name']), force=True)

        # drop existing tables
        tbl_q = sql.select(schema.Table).where(schema.Table.dir_id == dir_id).with_for_update()
        for row in conn.execute(tbl_q).all():
            tbl = self.get_table_by_id(row.id)
            # this table would have been dropped already if it's a view of a base we dropped earlier
            if tbl is not None:
                self._drop_tbl(tbl, force=True, is_replace=False)

        # self.drop_dir(dir_id)
        # _debug_print(for_update=True, msg=f'drop dir id={dir_id}')
        conn.execute(sql.delete(schema.Dir).where(schema.Dir.id == dir_id))
        _logger.info(f'Removed directory {str(dir_path)!r}.')

    def get_view_ids(self, tbl_id: UUID, for_update: bool = False) -> list[UUID]:
        """Return the ids of views that directly reference the given table"""
        conn = Env.get().conn
        q = sql.select(schema.Table.id).where(sql.text(f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r}"))
        if for_update:
            q = q.with_for_update()
        # _debug_print(for_update=False, msg=f'views of tbl id={tbl_id}')
        result = [r[0] for r in conn.execute(q).all()]
        return result

    def get_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        if (tbl_id, effective_version) not in self._tbl_versions:
            self._tbl_versions[(tbl_id, effective_version)] = self._load_tbl_version(tbl_id, effective_version)
        return self._tbl_versions[(tbl_id, effective_version)]

    def add_tbl_version(self, tbl_version: TableVersion) -> None:
        """Explicitly add a TableVersion"""
        self._tbl_versions[(tbl_version.id, tbl_version.effective_version)] = tbl_version
        # if this is a mutable view, also record it in the base
        if tbl_version.is_view and tbl_version.effective_version is None:
            base = tbl_version.base.get()
            base.mutable_views.append(TableVersionHandle(tbl_version.id, tbl_version.effective_version))

    def remove_tbl_version(self, tbl_version: TableVersion) -> None:
        assert (tbl_version.id, tbl_version.effective_version) in self._tbl_versions
        del self._tbl_versions[(tbl_version.id, tbl_version.effective_version)]

    def get_dir(self, dir_id: UUID, for_update: bool = False) -> Optional[Dir]:
        """Return the Dir with the given id, or None if it doesn't exist"""
        conn = Env.get().conn
        q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
        if for_update:
            q = q.with_for_update()
        # _debug_print(for_update=False, msg=f'dir id={dir_id!r}')
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        dir_record = schema.Dir(**row._mapping)
        return Dir(dir_record.id, dir_record.parent_id, dir_record.md['name'])

    def _get_dir(self, path: Path, for_update: bool = False) -> Optional[schema.Dir]:
        """
        Locking protocol:
        - S locks on all ancestors
        - X lock on dir if for_update == True, otherwise also an S lock
        """
        user = Env.get().user
        conn = Env.get().conn
        if path.is_root:
            q = sql.select(schema.Dir).where(schema.Dir.parent_id.is_(None), schema.Dir.md['user'].astext == user)
            if for_update:
                q = q.with_for_update()
            # _debug_print(for_update, 'root dir')
            row = conn.execute(q).one_or_none()
            return schema.Dir(**row._mapping) if row is not None else None
        else:
            parent_dir = self._get_dir(path.parent, for_update=False)
            if parent_dir is None:
                return None
            q = sql.select(schema.Dir).where(
                schema.Dir.parent_id == parent_dir.id,
                schema.Dir.md['name'].astext == path.name,
                schema.Dir.md['user'].astext == user,
            )
            if for_update:
                q = q.with_for_update()
            # _debug_print(for_update, f'dir {str(path)}')
            row = conn.execute(q).one_or_none()
            return schema.Dir(**row._mapping) if row is not None else None

    def _load_tbl(self, tbl_id: UUID) -> Optional[Table]:
        _logger.info(f'Loading table {tbl_id}')
        from .insertable_table import InsertableTable
        from .view import View

        conn = Env.get().conn
        q = (
            sql.select(schema.Table, schema.TableSchemaVersion)
            .join(schema.TableSchemaVersion)
            .where(schema.Table.id == schema.TableSchemaVersion.tbl_id)
            # Table.md['current_schema_version'] == TableSchemaVersion.schema_version
            .where(
                sql.text(
                    f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                    f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                )
            )
            .where(schema.Table.id == tbl_id)
        )
        # _debug_print(for_update=False, msg=f'load table id={tbl_id!r}')
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        tbl_record, schema_version_record = _unpack_row(row, [schema.Table, schema.TableSchemaVersion])

        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        view_md = tbl_md.view_md
        if view_md is None:
            # this is a base table
            if (tbl_id, None) not in self._tbl_versions:
                self._tbl_versions[(tbl_id, None)] = self._load_tbl_version(tbl_id, None)
            tbl = InsertableTable(tbl_record.dir_id, TableVersionHandle(tbl_id, None))
            return tbl

        # this is a view; determine the sequence of TableVersions to load
        tbl_version_path: list[tuple[UUID, Optional[int]]] = []
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        pure_snapshot = view_md.is_snapshot and view_md.predicate is None and len(schema_version_md.columns) == 0
        if pure_snapshot:
            # this is a pure snapshot, without a physical table backing it; we only need the bases
            pass
        else:
            effective_version = 0 if view_md.is_snapshot else None  # snapshots only have version 0
            tbl_version_path.append((tbl_id, effective_version))
        tbl_version_path.extend((UUID(tbl_id), version) for tbl_id, version in view_md.base_versions)

        # load TableVersions, starting at the root
        base_path: Optional[TableVersionPath] = None
        view_path: Optional[TableVersionPath] = None
        for id, effective_version in tbl_version_path[::-1]:
            if (id, effective_version) not in self._tbl_versions:
                self._tbl_versions[(id, effective_version)] = self._load_tbl_version(id, effective_version)
            view_path = TableVersionPath(TableVersionHandle(id, effective_version), base=base_path)
            base_path = view_path
        view = View(tbl_id, tbl_record.dir_id, tbl_md.name, view_path, snapshot_only=pure_snapshot)
        # TODO: also load mutable views
        return view

    def load_tbl_md(self, tbl_id: UUID, effective_version: Optional[int]) -> schema.FullTableMd:
        """
        Loads metadata from the store for a given table UUID and version.
        """
        _logger.info(f'Loading metadata for table version: {tbl_id}:{effective_version}')
        conn = Env.get().conn

        q = (
            sql.select(schema.Table, schema.TableVersion, schema.TableSchemaVersion)
            .select_from(schema.Table)
            .where(schema.Table.id == tbl_id)
            .join(schema.TableVersion)
            .where(schema.TableVersion.tbl_id == tbl_id)
            .join(schema.TableSchemaVersion)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
        )

        if effective_version is not None:
            # we are loading a specific version
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id AND tv.version = effective_version)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND tv.md.schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = q.where(sql.text(f"({schema.TableVersion.__table__}.md->>'version')::int = {effective_version}")).where(
                sql.text(
                    (
                        f"({schema.TableVersion.__table__}.md->>'schema_version')::int = "
                        f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                    )
                )
            )
        else:
            # we are loading the current version
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id AND t.current_version = tv.version)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND t.current_schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = q.where(
                sql.text(
                    f"({schema.Table.__table__}.md->>'current_version')::int = "
                    f'{schema.TableVersion.__table__}.{schema.TableVersion.version.name}'
                )
            ).where(
                sql.text(
                    (
                        f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                        f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                    )
                )
            )

        row = conn.execute(q).one_or_none()
        assert row is not None, f'Table record not found: {tbl_id}:{effective_version}'
        tbl_record, version_record, schema_version_record = _unpack_row(
            row, [schema.Table, schema.TableVersion, schema.TableSchemaVersion]
        )
        assert tbl_record.id == tbl_id
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        version_md = schema.md_from_dict(schema.TableVersionMd, version_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)

        return schema.FullTableMd(tbl_md, version_md, schema_version_md)

    def save_tbl_md(
        self,
        tbl_id: UUID,
        tbl_md: Optional[schema.TableMd],
        version_md: Optional[schema.TableVersionMd],
        schema_version_md: Optional[schema.TableSchemaVersionMd],
    ) -> None:
        """
        Saves metadata to the store. If specified, `tbl_md` will be updated in place (only one such record can exist
        per UUID); `version_md` and `schema_version_md` will be inserted as new records.

        If inserting `version_md` or `schema_version_md` would be a primary key violation, an exception will be raised.
        """
        conn = Env.get().conn

        if tbl_md is not None:
            conn.execute(
                sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(tbl_md)})
                .where(schema.Table.id == tbl_id)
            )

        if version_md is not None:
            conn.execute(
                sql.insert(schema.TableVersion.__table__).values(
                    tbl_id=tbl_id, version=version_md.version, md=dataclasses.asdict(version_md)
                )
            )

        if schema_version_md is not None:
            conn.execute(
                sql.insert(schema.TableSchemaVersion.__table__).values(
                    tbl_id=tbl_id,
                    schema_version=schema_version_md.schema_version,
                    md=dataclasses.asdict(schema_version_md),
                )
            )

    def delete_tbl_md(self, tbl_id: UUID) -> None:
        """
        Deletes all table metadata from the store for the given table UUID.
        """
        conn = Env.get().conn
        conn.execute(sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == tbl_id))

    def load_tbl_hierarchy_md(self, tbl: Table) -> list[schema.FullTableMd]:
        """
        Load metadata for the given table along with all its ancestors.
        """
        # TODO: First acquire X-locks for all relevant metadata entries

        # Load metadata for every table in the TableVersionPath for `tbl`.
        md = [self.load_tbl_md(tv.id, tv.effective_version) for tv in tbl._tbl_version_path.get_tbl_versions()]

        # If `tbl` is a named pure snapshot, we're not quite done, since the snapshot metadata won't appear in the
        # TableVersionPath. We need to prepend it separately.
        if tbl._id != tbl._tbl_version.id:
            snapshot_md = self.load_tbl_md(tbl._id, 0)
            md = [snapshot_md, *md]

        return md

    def _load_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        tbl_md, _, schema_version_md = self.load_tbl_md(tbl_id, effective_version)
        view_md = tbl_md.view_md

        _logger.info(f'Loading table version: {tbl_id}:{effective_version}')
        conn = Env.get().conn

        # load mutable view ids
        q = sql.select(schema.Table.id).where(
            sql.text(
                f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r} "
                "AND md->'view_md'->'base_versions'->0->1 IS NULL"
            )
        )
        mutable_view_ids = [r[0] for r in conn.execute(q).all()]
        mutable_views = [TableVersionHandle(id, None) for id in mutable_view_ids]

        if view_md is None:
            # this is a base table
            tbl_version = TableVersion(
                tbl_id, tbl_md, effective_version, schema_version_md, mutable_views=mutable_views
            )
            return tbl_version

        assert len(view_md.base_versions) > 0  # a view needs to have a base
        pure_snapshot = view_md.is_snapshot and view_md.predicate is None and len(schema_version_md.columns) == 0
        assert not pure_snapshot  # a pure snapshot doesn't have a physical table backing it, no point in loading it

        base: TableVersionHandle
        base_path: Optional[TableVersionPath] = None  # needed for live view
        if view_md.is_snapshot:
            base = TableVersionHandle(UUID(view_md.base_versions[0][0]), view_md.base_versions[0][1])
        else:
            base_path = TableVersionPath.from_md(tbl_md.view_md.base_versions)
            base = base_path.tbl_version

        tbl_version = TableVersion(
            tbl_id,
            tbl_md,
            effective_version,
            schema_version_md,
            base_path=base_path,
            base=base,
            mutable_views=mutable_views,
        )
        return tbl_version

    def _init_store(self) -> None:
        """One-time initialization of the stored catalog. Idempotent."""
        self.create_user(None)
        _logger.info(f'Initialized catalog.')

    def create_user(self, user: Optional[str]) -> None:
        """
        Creates a catalog record (root directory) for the specified user, if one does not already exist.
        """
        with Env.get().begin_xact():
            session = Env.get().session
            # See if there are any directories in the catalog matching the specified user.
            if session.query(schema.Dir).where(schema.Dir.md['user'].astext == user).count() > 0:
                # At least one such directory exists; no need to create a new one.
                return

            dir_md = schema.DirMd(name='', user=user, additional_md={})
            dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
            session.add(dir_record)
            session.flush()
            _logger.info(f'Added root directory record for user: {user!r}')

    def _handle_path_collision(
        self, path: Path, expected_obj_type: type[SchemaObject], expected_snapshot: bool, if_exists: IfExistsParam
    ) -> Optional[SchemaObject]:
        obj, _, _ = self._prepare_dir_op(add_dir_path=path.parent, add_name=path.name)

        if if_exists == IfExistsParam.ERROR and obj is not None:
            raise excs.Error(f'Path {str(path)!r} is an existing {type(obj)._display_name()}')
        else:
            is_snapshot = isinstance(obj, View) and obj._tbl_version_path.is_snapshot()
            if obj is not None and (not isinstance(obj, expected_obj_type) or (expected_snapshot and not is_snapshot)):
                obj_type_str = 'snapshot' if expected_snapshot else expected_obj_type._display_name()
                raise excs.Error(
                    f'Path {str(path)!r} already exists but is not a {obj_type_str}. Cannot {if_exists.name.lower()} it.'
                )

        if obj is None:
            return None
        if if_exists == IfExistsParam.IGNORE:
            return obj

        # drop the existing schema object
        if isinstance(obj, Dir):
            dir_contents = self._get_dir_contents(obj._id)
            if len(dir_contents) > 0 and if_exists == IfExistsParam.REPLACE:
                raise excs.Error(
                    f'Directory {str(path)!r} already exists and is not empty. Use `if_exists="replace_force"` to replace it.'
                )
            self._drop_dir(obj._id, path, force=True)
        else:
            assert isinstance(obj, Table)
            self._drop_tbl(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
        return None
