from __future__ import annotations

import dataclasses
import functools
import logging
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, TypeVar
from uuid import UUID

import psycopg
import sqlalchemy as sql

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema

from .column import Column
from .dir import Dir
from .globals import IfExistsParam, IfNotExistsParam, MediaValidation, QColumnId
from .insertable_table import InsertableTable
from .path import Path
from .schema_object import SchemaObject
from .table import Table
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .view import View

if TYPE_CHECKING:
    from pixeltable.plan import SampleClause

    from .. import DataFrame, exprs


_logger = logging.getLogger('pixeltable')


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


# -1: unlimited
# for now, we don't limit the number of retries, because we haven't seen situations where the actual number of retries
# grows uncontrollably
_MAX_RETRIES = -1

T = TypeVar('T')


def _retry_loop(*, for_write: bool) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(op: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(op)
        def loop(*args: Any, **kwargs: Any) -> T:
            num_retries = 0
            while True:
                try:
                    # in order for retry to work, we need to make sure that there aren't any prior db updates
                    # that are part of an ongoing transaction
                    assert not Env.get().in_xact
                    with Catalog.get().begin_xact(for_write=for_write, convert_db_excs=False):
                        return op(*args, **kwargs)
                except sql.exc.DBAPIError as e:
                    # TODO: what other exceptions should we be looking for?
                    if isinstance(e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)):
                        if num_retries < _MAX_RETRIES or _MAX_RETRIES == -1:
                            num_retries += 1
                            _logger.debug(f'Retrying ({num_retries}) after {type(e.orig)}')
                            time.sleep(random.uniform(0.1, 0.5))
                        else:
                            raise excs.Error(f'Serialization retry limit ({_MAX_RETRIES}) exceeded') from e
                    else:
                        raise
                except Exception as e:
                    # for informational/debugging purposes
                    _logger.debug(f'retry_loop(): passing along {e}')
                    raise

        return loop

    return decorator


class Catalog:
    """The functional interface to getting access to catalog objects

    All interface functions must be called in the context of a transaction, started with Catalog.begin_xact().

    Caching and invalidation of metadata:
    - Catalog caches TableVersion instances in order to avoid excessive metadata loading
    - for any specific table version (ie, combination of id and effective version) there can be only a single
      Tableversion instance in circulation; the reason is that each TV instance has its own store_tbl.sa_tbl, and
      mixing multiple instances of sqlalchemy Table objects in the same query (for the same underlying table) leads to
      duplicate references to that table in the From clause (ie, incorrect Cartesian products)
    - in order to allow multiple concurrent Python processes to perform updates (data and/or schema) against a shared
      Pixeltable instance, Catalog needs to reload metadata from the store when there are changes
    - concurrent changes are detected by comparing TableVersion.version/view_sn with the stored current version
      (TableMd.current_version/view_sn)
    - cached live TableVersion instances (those with effective_version == None) are validated against the stored
      metadata on transaction boundaries; this is recorded in TableVersion.is_validated
    - metadata validation is only needed for live TableVersion instances (snapshot instances are immutable)
    """

    _instance: Optional[Catalog] = None

    # cached TableVersion instances; key: [id, version]
    # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
    # - snapshot versions: records the version of the snapshot
    _tbl_versions: dict[tuple[UUID, Optional[int]], TableVersion]
    _tbls: dict[UUID, Table]
    _in_write_xact: bool  # True if we're in a write transaction
    _x_locked_tbl_ids: set[UUID]  # non-empty for write transactions

    # cached column dependencies
    # - key: table id, value: mapping from column id to its dependencies
    # - only maintained for dependencies between non-snapshot table versions
    # - can contain stale entries (stemming from invalidated TV instances)
    _column_dependencies: dict[UUID, dict[QColumnId, set[QColumnId]]]

    # column dependents are recomputed at the beginning of every write transaction and only reflect the locked tree
    _column_dependents: Optional[dict[QColumnId, set[QColumnId]]]

    @classmethod
    def get(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Remove the instance. Used for testing."""
        # invalidate all existing instances to force reloading of metadata
        for tbl_version in cls._instance._tbl_versions.values():
            # _logger.debug(
            #     f'Invalidating table version {tbl_version.id}:{tbl_version.effective_version} ({id(tbl_version):x})'
            # )
            tbl_version.is_validated = False
        cls._instance = None

    def __init__(self) -> None:
        self._tbl_versions = {}
        self._tbls = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self._in_write_xact = False
        self._x_locked_tbl_ids = set()
        self._column_dependencies = {}
        self._column_dependents = None
        self._init_store()

    def _dropped_tbl_error_msg(self, tbl_id: UUID) -> str:
        return f'Table was dropped (no record found for {tbl_id})'

    def validate(self) -> None:
        """Validate structural consistency of cached metadata"""
        for (tbl_id, effective_version), tbl_version in self._tbl_versions.items():
            assert tbl_id == tbl_version.id, f'{tbl_id} != {tbl_version.id}'
            assert tbl_version.effective_version == tbl_version.version or tbl_version.effective_version is None, (
                f'{tbl_version.effective_version} != {tbl_version.version} for id {tbl_id}'
            )
            assert effective_version == tbl_version.effective_version, (
                f'{effective_version} != {tbl_version.effective_version} for id {tbl_id}'
            )
            assert len(tbl_version.mutable_views) == 0 or tbl_version.is_mutable, (
                f'snapshot_id={tbl_version.id} mutable_views={tbl_version.mutable_views}'
            )

            if tbl_version.is_view and tbl_version.is_mutable and tbl_version.is_validated:
                # make sure this mutable view is recorded in a mutable base
                base = tbl_version.base
                assert base is not None
                if base.effective_version is None:
                    assert (base.id, None) in self._tbl_versions
                    base_tv = self._tbl_versions[base.id, None]
                    if not base_tv.is_validated:
                        continue
                    mutable_view_ids = ', '.join(str(tv.id) for tv in self._tbl_versions[base.id, None].mutable_views)
                    mutable_view_names = ', '.join(
                        tv._tbl_version.name
                        for tv in self._tbl_versions[base.id, None].mutable_views
                        if tv._tbl_version is not None
                    )
                    assert TableVersionHandle.create(tbl_version) in self._tbl_versions[base.id, None].mutable_views, (
                        f'{tbl_version.name} ({tbl_version.id}) missing in {mutable_view_ids} ({mutable_view_names})'
                    )

            if len(tbl_version.mutable_views) > 0:
                # make sure we also loaded mutable view metadata, which is needed to detect column dependencies
                for v in tbl_version.mutable_views:
                    assert v.effective_version is None, f'{v.id}:{v.effective_version}'

    @contextmanager
    def begin_xact(
        self,
        *,
        tbl: Optional[TableVersionPath] = None,
        for_write: bool = False,
        lock_mutable_tree: bool = False,
        convert_db_excs: bool = True,
    ) -> Iterator[sql.Connection]:
        """
        Return a context manager that yields a connection to the database. Idempotent.

        It is mandatory to call this method, not Env.begin_xact(), if the transaction accesses any table data
        or metadata.

        If tbl != None, follows this locking protocol:
        - validates/reloads the TableVersion instances of tbl's ancestors (in the hope that this reduces potential
          SerializationErrors later on)
        - if for_write == True, x-locks Table record (by updating Table.lock_dummy; see _acquire_tbl_xlock())
        - if for_write == False, validates TableVersion instance
        - if lock_mutable_tree == True, also x-locks all mutable views of the table
        - this needs to be done in a retry loop, because Postgres can decide to abort the transaction
          (SerializationFailure, LockNotAvailable)
        - for that reason, we do all lock acquisition prior to doing any real work (eg, compute column values),
          to minimize the probability of loosing that work due to a forced abort

        If convert_db_excs == True, converts DBAPIErrors into excs.Errors.
        """
        if Env.get().in_xact:
            if tbl is not None and for_write:
                # make sure that we requested the required table lock at the beginning of the transaction
                assert tbl.tbl_id in self._x_locked_tbl_ids, f'{tbl.tbl_id} not in {self._x_locked_tbl_ids}'
            yield Env.get().conn
            return

        # tv_msg = '\n'.join(
        #     [
        #         f'{tv.id}:{tv.effective_version} : tv={id(tv):x} sa_tbl={id(tv.store_tbl.sa_tbl):x}'
        #         for tv in self._tbl_versions.values()
        #     ]
        # )
        # _logger.debug(f'begin_xact(): {tv_msg}')
        num_retries = 0
        while True:
            try:
                self._in_write_xact = False
                self._x_locked_tbl_ids = set()
                self._column_dependents = None

                with Env.get().begin_xact() as conn:
                    if tbl is not None:
                        try:
                            if not self._acquire_path_locks(
                                tbl=tbl, for_write=for_write, lock_mutable_tree=lock_mutable_tree
                            ):
                                # this is a snapshot
                                yield conn
                                return

                            if for_write:
                                if lock_mutable_tree:
                                    self._x_locked_tbl_ids = self._get_mutable_tree(tbl.tbl_id)
                                    self._compute_column_dependents(self._x_locked_tbl_ids)
                                else:
                                    self._x_locked_tbl_ids = {tbl.tbl_id}
                                if _logger.isEnabledFor(logging.DEBUG):
                                    # validate only when we don't see errors
                                    self.validate()

                        except sql.exc.DBAPIError as e:
                            if isinstance(
                                e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)
                            ) and (num_retries < _MAX_RETRIES or _MAX_RETRIES == -1):
                                num_retries += 1
                                _logger.debug(f'Retrying ({num_retries}) after {type(e.orig)}')
                                time.sleep(random.uniform(0.1, 0.5))
                                continue
                            else:
                                raise

                    self._in_write_xact = for_write
                    yield conn
                    return

            except sql.exc.DBAPIError as e:
                # we got some db error during the actual operation (not just while trying to get locks on the metadata
                # records): we convert these into Errors, if asked to do so, and abort
                # TODO: what other concurrency-related exceptions should we expect?

                # we always convert UndefinedTable exceptions (they can't be retried)
                if isinstance(e.orig, psycopg.errors.UndefinedTable):
                    # the table got dropped in the middle of the table operation
                    _logger.debug(f'Exception: undefined table ({tbl.tbl_name()}): Caught {type(e.orig)}: {e!r}')
                    assert tbl is not None
                    raise excs.Error(f'Table was dropped: {tbl.tbl_name()}') from None
                elif isinstance(e.orig, psycopg.errors.SerializationFailure) and convert_db_excs:
                    # we still got a serialization error, despite getting x-locks at the beginning
                    msg = f'{tbl.tbl_name()} ({tbl.tbl_id})' if tbl is not None else ''
                    _logger.debug(f'Exception: serialization failure: {msg} ({e})')
                    raise excs.Error(
                        'That Pixeltable operation could not be completed because it conflicted with another '
                        'operation that was run on a different process.\n'
                        'Please re-run the operation.'
                    ) from None
                else:
                    raise

            finally:
                self._in_write_xact = False
                self._x_locked_tbl_ids = set()
                self._column_dependents = None

                # invalidate cached current TableVersion instances
                for tv in self._tbl_versions.values():
                    if tv.effective_version is None:
                        _logger.debug(f'invalidating table version {tv.id}:None (tv={id(tv):x})')
                        tv.is_validated = False

    @property
    def in_write_xact(self) -> bool:
        return self._in_write_xact

    def _acquire_path_locks(
        self, *, tbl: TableVersionPath, for_write: bool = False, lock_mutable_tree: bool = False
    ) -> bool:
        """
        Path locking protocol:
        - refresh cached TableVersions of ancestors (we need those even during inserts, for computed columns that
          reference the base tables)
        - refresh cached TableVersion of tbl or get X-lock, depending on for_write
        - if lock_mutable_tree, also X-lock all mutable views of tbl

        Returns False if trying to lock a pure snapshot with for_write == True
        Raises Error if tbl doesn't exist.
        """
        start_idx = 1 if for_write else 0
        for handle in tbl.get_tbl_versions()[start_idx::-1]:
            _ = self.get_tbl_version(handle.id, handle.effective_version)
        if not for_write:
            return True  # nothing left to lock
        return self._acquire_tbl_xlock(tbl_id=tbl.tbl_id, lock_mutable_tree=lock_mutable_tree, raise_if_not_exists=True)

    def _acquire_tbl_xlock(
        self,
        *,
        tbl_id: Optional[UUID] = None,
        dir_id: Optional[UUID] = None,
        tbl_name: Optional[str] = None,
        lock_mutable_tree: bool = False,
        raise_if_not_exists: bool = False,
    ) -> bool:
        """Force acquisition of an X-lock on a Table record via a blind update

        Either tbl_id or dir_id/tbl_name need to be specified.
        Returns True if the table was locked, False if it was a snapshot or not found.
        If lock_mutable_tree, recursively locks all mutable views of the table.

        Returns False if the table is a snapshot or not found and !raise_if_not_exists.
        """
        where_clause: sql.ColumnElement
        if tbl_id is not None:
            where_clause = schema.Table.id == tbl_id
        else:
            where_clause = sql.and_(schema.Table.dir_id == dir_id, schema.Table.md['name'].astext == tbl_name)
            user = Env.get().user
            if user is not None:
                where_clause = sql.and_(where_clause, schema.Table.md['user'].astext == Env.get().user)

        conn = Env.get().conn
        row = conn.execute(sql.select(schema.Table).where(where_clause).with_for_update(nowait=True)).one_or_none()
        if row is None:
            if raise_if_not_exists:
                raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
            return False  # nothing to lock
        if row.md['view_md'] is not None and row.md['view_md']['is_snapshot']:
            return False  # nothing to lock
        conn.execute(sql.update(schema.Table).values(lock_dummy=1).where(where_clause))

        if not lock_mutable_tree:
            return True
        # also lock mutable views
        tv = self.get_tbl_version(tbl_id, None)
        for view in tv.mutable_views:
            self._acquire_tbl_xlock(tbl_id=view.id, lock_mutable_tree=True, raise_if_not_exists=raise_if_not_exists)
        return True

    def _get_mutable_tree(self, tbl_id: UUID) -> set[UUID]:
        """Returns ids of all tables that form the tree of mutable views starting at tbl_id; includes the root."""
        tv = self.get_tbl_version(tbl_id, None)
        result: set[UUID] = {tv.id}
        for view in tv.mutable_views:
            result.update(self._get_mutable_tree(view.id))
        return result

    def _compute_column_dependents(self, mutable_tree: set[UUID]) -> None:
        """Populate self._column_dependents for all tables in mutable_tree"""
        assert self._column_dependents is None
        self._column_dependents = defaultdict(set)
        for tbl_id in mutable_tree:
            assert tbl_id in self._column_dependencies
            for col, dependencies in self._column_dependencies[tbl_id].items():
                for dependency in dependencies:
                    if dependency.tbl_id not in mutable_tree:
                        continue
                    dependents = self._column_dependents[dependency]
                    dependents.add(col)

    def get_column_dependents(self, tbl_id: UUID, col_id: int) -> set[Column]:
        """Return all Columns that transitively depend on the given column."""
        assert self._column_dependents is not None
        dependents = self._column_dependents[QColumnId(tbl_id, col_id)]
        result: set[Column] = set()
        for dependent in dependents:
            tv = self.get_tbl_version(dependent.tbl_id, None)
            col = tv.cols_by_id[dependent.col_id]
            result.add(col)
        return result

    def _acquire_dir_xlock(
        self, *, parent_id: Optional[UUID] = None, dir_id: Optional[UUID] = None, dir_name: Optional[str] = None
    ) -> None:
        """Force acquisition of an X-lock on a Dir record via a blind update.

        If dir_id is present, then all other conditions are ignored.
        Note that (parent_id==None) is a valid where condition.
        If dir_id is not specified, the user from the environment is added to the directory filters.
        """
        assert (dir_name is None) != (dir_id is None)
        assert not (parent_id is not None and dir_name is None)
        user = Env.get().user
        assert self._in_write_xact
        q = sql.update(schema.Dir).values(lock_dummy=1)
        if dir_id is not None:
            q = q.where(schema.Dir.id == dir_id)
        else:
            q = q.where(schema.Dir.parent_id == parent_id)
            if dir_name is not None:
                q = q.where(schema.Dir.md['name'].astext == dir_name)
            if user is not None:
                q = q.where(schema.Dir.md['user'].astext == user)
        Env.get().conn.execute(q)

    def get_dir_path(self, dir_id: UUID) -> Path:
        """Return path for directory with given id"""
        conn = Env.get().conn
        names: list[str] = []
        while True:
            q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
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

    @_retry_loop(for_write=False)
    def get_dir_contents(self, dir_path: Path, recursive: bool = False) -> dict[str, DirEntry]:
        dir = self._get_schema_object(dir_path, expected=Dir, raise_if_not_exists=True)
        return self._get_dir_contents(dir._id, recursive=recursive)

    def _get_dir_contents(self, dir_id: UUID, recursive: bool = False) -> dict[str, DirEntry]:
        """Returns a dict mapping the entry names to DirEntry objects"""
        conn = Env.get().conn
        result: dict[str, Catalog.DirEntry] = {}

        q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
        rows = conn.execute(q).all()
        for row in rows:
            dir = schema.Dir(**row._mapping)
            dir_contents: dict[str, Catalog.DirEntry] = {}
            if recursive:
                dir_contents = self._get_dir_contents(dir.id, recursive=True)
            result[dir.md['name']] = self.DirEntry(dir=dir, dir_entries=dir_contents, table=None)

        q = sql.select(schema.Table).where(schema.Table.dir_id == dir_id)
        rows = conn.execute(q).all()
        for row in rows:
            tbl = schema.Table(**row._mapping)
            result[tbl.md['name']] = self.DirEntry(dir=None, dir_entries={}, table=tbl)

        return result

    @_retry_loop(for_write=True)
    def move(self, path: Path, new_path: Path) -> None:
        self._move(path, new_path)

    def _move(self, path: Path, new_path: Path) -> None:
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
        for p in sorted(dir_paths):
            dir = self._get_dir(p, lock_dir=True)
            if dir is None:
                raise excs.Error(f'Directory {str(p)!r} does not exist.')
            if p == add_dir_path:
                add_dir = dir
            if p == drop_dir_path:
                drop_dir = dir

        add_obj: Optional[SchemaObject] = None
        if add_dir is not None:
            add_obj = self._get_dir_entry(add_dir.id, add_name, lock_entry=True)
            if add_obj is not None and raise_if_exists:
                add_path = add_dir_path.append(add_name)
                raise excs.Error(f'Path {str(add_path)!r} already exists.')

        drop_obj: Optional[SchemaObject] = None
        if drop_dir is not None:
            drop_path = drop_dir_path.append(drop_name)
            drop_obj = self._get_dir_entry(drop_dir.id, drop_name, lock_entry=True)
            if drop_obj is None and raise_if_not_exists:
                raise excs.Error(f'Path {str(drop_path)!r} does not exist.')
            if drop_obj is not None and drop_expected is not None and not isinstance(drop_obj, drop_expected):
                raise excs.Error(
                    f'{str(drop_path)!r} needs to be a {drop_expected._display_name()} '
                    f'but is a {type(drop_obj)._display_name()}'
                )

        add_dir_obj = Dir(add_dir.id, add_dir.parent_id, add_dir.md['name']) if add_dir is not None else None
        return add_obj, add_dir_obj, drop_obj

    def _get_dir_entry(self, dir_id: UUID, name: str, lock_entry: bool = False) -> Optional[SchemaObject]:
        user = Env.get().user
        conn = Env.get().conn

        # check for subdirectory
        if lock_entry:
            self._acquire_dir_xlock(parent_id=dir_id, dir_id=None, dir_name=name)
        q = sql.select(schema.Dir).where(
            schema.Dir.parent_id == dir_id, schema.Dir.md['name'].astext == name, schema.Dir.md['user'].astext == user
        )
        rows = conn.execute(q).all()
        # The condition below can occur if there is a synchronization failure across multiple processes
        # It indicates database inconsistency.
        if len(rows) > 1:
            raise AssertionError(rows)
        if len(rows) == 1:
            dir_record = schema.Dir(**rows[0]._mapping)
            return Dir(dir_record.id, dir_record.parent_id, name)

        # check for table
        if lock_entry:
            self._acquire_tbl_xlock(dir_id=dir_id, tbl_name=name)
        q = sql.select(schema.Table.id).where(
            schema.Table.dir_id == dir_id,
            schema.Table.md['name'].astext == name,
            schema.Table.md['user'].astext == user,
        )
        tbl_id = conn.execute(q).scalar_one_or_none()
        if tbl_id is not None:
            if tbl_id not in self._tbls:
                _ = self._load_tbl(tbl_id)
            return self._tbls[tbl_id]

        return None

    def _get_schema_object(
        self,
        path: Path,
        expected: Optional[type[SchemaObject]] = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
        lock_parent: bool = False,
        lock_obj: bool = False,
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
            dir = self._get_dir(path, lock_dir=lock_obj)
            if dir is None:
                raise excs.Error(f'Unknown user: {Env.get().user}')
            return Dir(dir.id, dir.parent_id, dir.md['name'])

        parent_path = path.parent
        parent_dir = self._get_dir(parent_path, lock_dir=lock_parent)
        if parent_dir is None:
            raise excs.Error(f'Directory {str(parent_path)!r} does not exist.')
        obj = self._get_dir_entry(parent_dir.id, path.name, lock_entry=lock_obj)

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
        if tbl_id not in self._tbls:
            tbl = self._load_tbl(tbl_id)
            if tbl is None:
                return None
            # # if this is a mutable table, we also need to have its mutable views loaded, in order to track column
            # # dependencies
            # tbl_version = tbl._tbl_version.get()
            # if tbl_version.is_mutable:
            #     for v in tbl_version.mutable_views:
            #         _ = self.get_table_by_id(v.id)
        return self._tbls[tbl_id]

    @_retry_loop(for_write=True)
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

    @_retry_loop(for_write=True)
    def create_view(
        self,
        path: Path,
        base: TableVersionPath,
        select_list: Optional[list[tuple[exprs.Expr, Optional[str]]]],
        where: Optional[exprs.Expr],
        sample_clause: Optional['SampleClause'],
        additional_columns: Optional[dict[str, Any]],
        is_snapshot: bool,
        iterator: Optional[tuple[type[ComponentIterator], dict[str, Any]]],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> Table:
        from pixeltable.utils.filecache import FileCache

        if not is_snapshot and not base.is_snapshot():
            # this is a mutable view of a mutable base; X-lock the base and advance its view_sn before adding the view
            self._acquire_tbl_xlock(tbl_id=base.tbl_id)
            base_tv = self.get_tbl_version(base.tbl_id, None)
            base_tv.tbl_md.view_sn += 1
            result = Env.get().conn.execute(
                sql.update(schema.Table)
                .values({schema.Table.md: dataclasses.asdict(base_tv.tbl_md)})
                .where(schema.Table.id == base.tbl_id)
            )
            assert result.rowcount == 1, result.rowcount

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
            sample_clause=sample_clause,
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

    def create_replica(self, path: Path, md: list[schema.FullTableMd]) -> None:
        """
        Creates table, table_version, and table_schema_version records for a replica with the given metadata.
        The metadata should be presented in standard "ancestor order", with the table being replicated at
        list position 0 and the (root) base table at list position -1.
        """
        assert Env.get().in_xact

        tbl_id = UUID(md[0].tbl_md.tbl_id)

        existing = self._handle_path_collision(path, Table, False, if_exists=IfExistsParam.IGNORE)  # type: ignore[type-abstract]
        if existing is not None and existing._id != tbl_id:
            raise excs.Error(
                f'An attempt was made to create a replica table at {path!r}, '
                'but a different table already exists at that location.'
            )

        # Ensure that the system directory exists.
        self._create_dir(Path('_system', allow_system_paths=True), if_exists=IfExistsParam.IGNORE, parents=False)

        # Now check to see if this table UUID already exists in the catalog.
        existing = Catalog.get().get_table_by_id(tbl_id)
        if existing is not None:
            existing_path = Path(existing._path(), allow_system_paths=True)
            if existing_path != path:
                # It does exist, under a different path from the specified one.
                if not existing_path.is_system_path:
                    raise excs.Error(
                        f'That table has already been replicated as {existing_path!r}.\n'
                        f'Drop the existing replica if you wish to re-create it.'
                    )
                # If it's a system table, then this means it was created at some point as the ancestor of some other
                # table (a snapshot-over-snapshot scenario). In that case, we simply move it to the new (named)
                # location.
                self._move(existing_path, path)

        # Now store the metadata for this replica's proper ancestors. If one or more proper ancestors
        # do not yet exist in the store, they will be created as anonymous system tables.
        # We instantiate the ancestors starting with the base table and ending with the immediate parent of the
        # table being replicated.
        for ancestor_md in md[:0:-1]:
            ancestor_id = UUID(ancestor_md.tbl_md.tbl_id)
            replica = Catalog.get().get_table_by_id(ancestor_id)
            replica_path: Path
            if replica is None:
                # We've never seen this table before. Create a new anonymous system table for it.
                replica_path = Path(f'_system.replica_{ancestor_id.hex}', allow_system_paths=True)
            else:
                # The table already exists in the catalog. The existing path might be a system path (if the table
                # was created as an anonymous base table of some other table), or it might not (if it's a snapshot
                # that was directly replicated by the user at some point). In either case, use the existing path.
                replica_path = Path(replica._path(), allow_system_paths=True)

            # Store the metadata; it could be a new version (in which case a new record will be created), or a known
            # version (in which case the newly received metadata will be validated as identical).
            # If it's a new version, this will result in a new TableVersion record being created.
            self.__store_replica_md(replica_path, ancestor_md)

            # Now we must clear cached metadata for the ancestor table, to force the next table operation to pick up
            # the new TableVersion instance. This is necessary because computed columns of descendant tables might
            # reference columns of the ancestor table that only exist in the new version.
            replica = Catalog.get().get_table_by_id(ancestor_id)
            assert replica is not None  # If it didn't exist before, it must have been created by now.
            replica._tbl_version_path.clear_cached_md()

        # Finally, store the metadata for the table being replicated; as before, it could be a new version or a known
        # version. If it's a new version, then a TableVersion record will be created, unless the table being replicated
        # is a pure snapshot.
        self.__store_replica_md(path, md[0])

    def __store_replica_md(self, path: Path, md: schema.FullTableMd) -> None:
        _logger.info(f'Creating replica table at {path!r} with ID: {md.tbl_md.tbl_id}')
        dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
        assert dir is not None
        assert self._in_write_xact

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
            q = sql.insert(schema.Table.__table__).values(
                id=tbl_id,
                dir_id=dir._id,
                md=dataclasses.asdict(
                    dataclasses.replace(md.tbl_md, name=path.name, user=Env.get().user, is_replica=True)
                ),
            )
            conn.execute(q)
        else:
            assert existing_md_row.md['is_replica']
            if md.tbl_md.current_version > existing_md_row.md['current_version']:
                # New metadata is more recent than the metadata currently stored in the DB; we'll update the record
                # in place in the DB.
                new_tbl_md = dataclasses.replace(md.tbl_md, name=path.name, user=Env.get().user, is_replica=True)

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

        self.store_tbl_md(UUID(tbl_id), None, new_tbl_md, new_version_md, new_schema_version_md)

        if new_version_md is not None and not md.is_pure_snapshot:
            # It's a new version of a table that has a physical store, so we need to create a TableVersion instance.
            TableVersion.create_replica(md)

    @_retry_loop(for_write=False)
    def get_table(self, path: Path) -> Table:
        obj = Catalog.get()._get_schema_object(path, expected=Table, raise_if_not_exists=True)
        assert isinstance(obj, Table)
        # We need to clear cached metadata from tbl_version_path, in case the schema has been changed
        # by another process.
        obj._tbl_version_path.clear_cached_md()
        return obj

    @_retry_loop(for_write=True)
    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        tbl = self._get_schema_object(
            path,
            expected=Table,
            raise_if_not_exists=if_not_exists == IfNotExistsParam.ERROR and not force,
            lock_parent=True,
            lock_obj=False,
        )
        if tbl is None:
            _logger.info(f'Skipped table {str(path)!r} (does not exist).')
            return
        assert isinstance(tbl, Table)

        if isinstance(tbl, View) and tbl._tbl_version_path.is_mutable() and tbl._tbl_version_path.base.is_mutable():
            # this is a mutable view of a mutable base;
            # lock the base before the view, in order to avoid deadlocks with concurrent inserts/updates
            base_id = tbl._tbl_version_path.base.tbl_id
            self._acquire_tbl_xlock(tbl_id=base_id, lock_mutable_tree=False)

        self._drop_tbl(tbl, force=force, is_replace=False)

    def _drop_tbl(self, tbl: Table, force: bool, is_replace: bool) -> None:
        """
        Drop the table (and recursively its views, if force == True).

        Locking protocol:
        - X-lock base before X-locking any view
        - deadlock-free wrt to TableVersion.insert() (insert propagation also proceeds top-down)
        - X-locks parent dir prior to calling TableVersion.drop(): prevent concurrent creation of another SchemaObject
          in the same directory with the same name (which could lead to duplicate names if we get aborted)
        """
        self._acquire_dir_xlock(dir_id=tbl._dir_id)
        self._acquire_tbl_xlock(tbl_id=tbl._id, lock_mutable_tree=False)

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

        # if this is a mutable view of a mutable base, advance the base's view_sn
        if isinstance(tbl, View) and tbl._tbl_version_path.is_mutable() and tbl._tbl_version_path.base.is_mutable():
            base_id = tbl._tbl_version_path.base.tbl_id
            base_tv = self.get_tbl_version(base_id, None)
            base_tv.tbl_md.view_sn += 1
            result = Env.get().conn.execute(
                sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(base_tv.tbl_md)})
                .where(schema.Table.id == base_id)
            )
            assert result.rowcount == 1, result.rowcount

        tv = tbl._tbl_version.get() if tbl._tbl_version is not None else None
        if tv is not None:
            tv = tbl._tbl_version.get()
            # invalidate the TableVersion instance so that existing references to it can find out it has been dropped
            tv.is_validated = False

        self.delete_tbl_md(tbl._id)
        assert tbl._id in self._tbls
        del self._tbls[tbl._id]
        _logger.info(f'Dropped table `{tbl._path()}`.')

        if tv is not None:
            tv.drop()
            assert (tv.id, tv.effective_version) in self._tbl_versions
            del self._tbl_versions[tv.id, tv.effective_version]

    @_retry_loop(for_write=True)
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

    @_retry_loop(for_write=True)
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
        self._acquire_dir_xlock(dir_id=dir_id)
        dir_q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
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
        conn.execute(sql.delete(schema.Dir).where(schema.Dir.id == dir_id))
        _logger.info(f'Removed directory {str(dir_path)!r}.')

    def get_view_ids(self, tbl_id: UUID, for_update: bool = False) -> list[UUID]:
        """Return the ids of views that directly reference the given table"""
        conn = Env.get().conn
        # check whether this table still exists
        q = sql.select(sql.func.count()).select_from(schema.Table).where(schema.Table.id == tbl_id)
        tbl_count = conn.execute(q).scalar()
        if tbl_count == 0:
            raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
        q = sql.select(schema.Table.id).where(sql.text(f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r}"))
        if for_update:
            q = q.with_for_update()
        result = [r[0] for r in conn.execute(q).all()]
        return result

    def get_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        # we need a transaction here, if we're not already in one; if this starts a new transaction,
        # the returned TableVersion instance will not be validated
        with self.begin_xact(for_write=False) as conn:
            tv = self._tbl_versions.get((tbl_id, effective_version))
            if tv is None:
                tv = self._load_tbl_version(tbl_id, effective_version)
            elif not tv.is_validated:
                # only live instances are invalidated
                assert effective_version is None
                # we validate live instances by comparing our cached TableMd.current_version/view_sn to what's stored
                # _logger.debug(f'validating metadata for table {tbl_id}:{tv.version} ({id(tv):x})')
                q = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
                row = conn.execute(q).one_or_none()
                if row is None:
                    raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
                current_version, view_sn = row.md['current_version'], row.md['view_sn']

                # the stored version can be behind TableVersion.version, because we don't roll back the in-memory
                # metadata changes after a failed update operation
                if current_version != tv.version or view_sn != tv.tbl_md.view_sn:
                    # the cached metadata is invalid
                    _logger.debug(
                        f'reloading metadata for table {tbl_id} '
                        f'(cached/current version: {tv.version}/{current_version}, '
                        f'cached/current view_sn: {tv.tbl_md.view_sn}/{view_sn})'
                    )
                    tv = self._load_tbl_version(tbl_id, None)
                else:
                    # the cached metadata is valid
                    tv.is_validated = True

            assert tv.is_validated
            return tv

    def remove_tbl_version(self, tbl_version: TableVersion) -> None:
        assert (tbl_version.id, tbl_version.effective_version) in self._tbl_versions
        del self._tbl_versions[tbl_version.id, tbl_version.effective_version]

    def get_dir(self, dir_id: UUID, for_update: bool = False) -> Optional[Dir]:
        """Return the Dir with the given id, or None if it doesn't exist"""
        conn = Env.get().conn
        if for_update:
            self._acquire_dir_xlock(dir_id=dir_id)
        q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        dir_record = schema.Dir(**row._mapping)
        return Dir(dir_record.id, dir_record.parent_id, dir_record.md['name'])

    def _get_dir(self, path: Path, lock_dir: bool = False) -> Optional[schema.Dir]:
        """
        lock_dir: if True, X-locks target (but not the ancestors)
        """
        user = Env.get().user
        conn = Env.get().conn
        if path.is_root:
            if lock_dir:
                self._acquire_dir_xlock(dir_name='')
            q = sql.select(schema.Dir).where(schema.Dir.parent_id.is_(None), schema.Dir.md['user'].astext == user)
            row = conn.execute(q).one_or_none()
            return schema.Dir(**row._mapping) if row is not None else None
        else:
            parent_dir = self._get_dir(path.parent, lock_dir=False)
            if parent_dir is None:
                return None
            if lock_dir:
                self._acquire_dir_xlock(parent_id=parent_dir.id, dir_name=path.name)
            q = sql.select(schema.Dir).where(
                schema.Dir.parent_id == parent_dir.id,
                schema.Dir.md['name'].astext == path.name,
                schema.Dir.md['user'].astext == user,
            )
            row = conn.execute(q).one_or_none()
            return schema.Dir(**row._mapping) if row is not None else None

    def _load_tbl(self, tbl_id: UUID) -> Optional[Table]:
        """Loads metadata for the table with the given id and caches it."""
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
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        tbl_record, schema_version_record = _unpack_row(row, [schema.Table, schema.TableSchemaVersion])

        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        view_md = tbl_md.view_md
        if view_md is None:
            # this is a base table
            if (tbl_id, None) not in self._tbl_versions:
                _ = self._load_tbl_version(tbl_id, None)
            tbl = InsertableTable(tbl_record.dir_id, TableVersionHandle(tbl_id, None))
            self._tbls[tbl_id] = tbl
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
                _ = self._load_tbl_version(id, effective_version)
            view_path = TableVersionPath(TableVersionHandle(id, effective_version), base=base_path)
            base_path = view_path
        view = View(tbl_id, tbl_record.dir_id, tbl_md.name, view_path, snapshot_only=pure_snapshot)
        self._tbls[tbl_id] = view
        return view

    @_retry_loop(for_write=False)
    def collect_tbl_history(self, tbl_id: UUID, n: Optional[int]) -> list[schema.FullTableMd]:
        """
        Returns the history of up to n versions of the table with the given UUID.

        Args:
            tbl_id: the UUID of the table to collect history for.
            n: Optional limit on the maximum number of versions returned.

        Returns:
            A sequence of rows, ordered by version number
            Each row contains a TableVersion and a TableSchemaVersion object.
        """
        q = (
            sql.select(schema.TableVersion, schema.TableSchemaVersion)
            .select_from(schema.TableVersion)
            .join(
                schema.TableSchemaVersion,
                sql.cast(schema.TableVersion.md['schema_version'], sql.Integer)
                == schema.TableSchemaVersion.schema_version,
            )
            .where(schema.TableVersion.tbl_id == tbl_id)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
            .order_by(schema.TableVersion.version.desc())
        )
        if n is not None:
            q = q.limit(n)
        src_rows = Env.get().session.execute(q).fetchall()
        return [
            schema.FullTableMd(
                None,
                schema.md_from_dict(schema.TableVersionMd, row.TableVersion.md),
                schema.md_from_dict(schema.TableSchemaVersionMd, row.TableSchemaVersion.md),
            )
            for row in src_rows
        ]

    def load_tbl_md(self, tbl_id: UUID, effective_version: Optional[int]) -> schema.FullTableMd:
        """
        Loads metadata from the store for a given table UUID and version.
        """
        # _logger.info(f'Loading metadata for table version: {tbl_id}:{effective_version}')
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
        if row is None:
            raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
        tbl_record, version_record, schema_version_record = _unpack_row(
            row, [schema.Table, schema.TableVersion, schema.TableSchemaVersion]
        )
        assert tbl_record.id == tbl_id
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        version_md = schema.md_from_dict(schema.TableVersionMd, version_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)

        return schema.FullTableMd(tbl_md, version_md, schema_version_md)

    def store_tbl_md(
        self,
        tbl_id: UUID,
        dir_id: Optional[UUID],
        tbl_md: Optional[schema.TableMd],
        version_md: Optional[schema.TableVersionMd],
        schema_version_md: Optional[schema.TableSchemaVersionMd],
    ) -> None:
        """
        Stores metadata to the DB.

        Args:
            tbl_id: UUID of the table to store metadata for.
            dir_id: If specified, the tbl_md will be added to the given directory; if None, the table must already exist
            tbl_md: If specified, `tbl_md` will be inserted, or updated (only one such record can exist per UUID)
            version_md: inserted as a new record if present
            schema_version_md: will be inserted as a new record if present

        If inserting `version_md` or `schema_version_md` would be a primary key violation, an exception will be raised.
        """
        assert self._in_write_xact
        session = Env.get().session

        # Construct and insert or update table record if requested.
        if tbl_md is not None:
            assert tbl_md.tbl_id == str(tbl_id)
            if version_md is not None:
                assert tbl_md.current_version == version_md.version
                assert tbl_md.current_schema_version == version_md.schema_version
            if schema_version_md is not None:
                assert tbl_md.current_schema_version == schema_version_md.schema_version
            if dir_id is not None:
                # We are inserting a record while creating a new table.
                tbl_record = schema.Table(id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(tbl_md))
                session.add(tbl_record)
            else:
                # Update the existing table record.
                result = session.execute(
                    sql.update(schema.Table.__table__)
                    .values({schema.Table.md: dataclasses.asdict(tbl_md)})
                    .where(schema.Table.id == tbl_id)
                )
                assert result.rowcount == 1, result.rowcount

        # Construct and insert new table version record if requested.
        if version_md is not None:
            assert version_md.tbl_id == str(tbl_id)
            if schema_version_md is not None:
                assert version_md.schema_version == schema_version_md.schema_version
            tbl_version_record = schema.TableVersion(
                tbl_id=tbl_id, version=version_md.version, md=dataclasses.asdict(version_md)
            )
            session.add(tbl_version_record)

        # Construct and insert a new schema version record if requested.
        if schema_version_md is not None:
            assert schema_version_md.tbl_id == str(tbl_id)
            schema_version_record = schema.TableSchemaVersion(
                tbl_id=tbl_id, schema_version=schema_version_md.schema_version, md=dataclasses.asdict(schema_version_md)
            )
            session.add(schema_version_record)
        session.flush()  # Inform SQLAlchemy that we want to write these changes to the DB.

    def update_tbl_version_md(self, version_md: Optional[schema.TableVersionMd]) -> None:
        """
        Update the TableVersion.md field in the DB. Typically used to update the cascade row count status.

        Args:
            version_md: TableVersionMd
        """
        assert self._in_write_xact
        session = Env.get().session

        session.execute(
            sql.update(schema.TableVersion.__table__)
            .values({schema.TableVersion.md: dataclasses.asdict(version_md)})
            .where(schema.TableVersion.tbl_id == version_md.tbl_id, schema.TableVersion.version == version_md.version)
        )

        session.flush()  # Inform SQLAlchemy that we want to write these changes to the DB.

    def delete_tbl_md(self, tbl_id: UUID) -> None:
        """
        Deletes all table metadata from the store for the given table UUID.
        """
        conn = Env.get().conn
        conn.execute(sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == tbl_id))

    def load_replica_md(self, tbl: Table) -> list[schema.FullTableMd]:
        """
        Load metadata for the given table along with all its ancestors. The values of TableMd.current_version and
        TableMd.current_schema_version will be adjusted to ensure that the metadata represent a valid (internally
        consistent) table state.
        """
        # TODO: First acquire X-locks for all relevant metadata entries

        # Load metadata for every table in the TableVersionPath for `tbl`.
        md = [self.load_tbl_md(tv.id, tv.effective_version) for tv in tbl._tbl_version_path.get_tbl_versions()]

        # If `tbl` is a named pure snapshot, we're not quite done, since the snapshot metadata won't appear in the
        # TableVersionPath. We need to prepend it separately.
        if isinstance(tbl, View) and tbl._snapshot_only:
            snapshot_md = self.load_tbl_md(tbl._id, 0)
            md = [snapshot_md, *md]

        for ancestor_md in md[1:]:
            # For replica metadata, we guarantee that the current_version and current_schema_version of TableMd
            # match the corresponding values in TableVersionMd and TableSchemaVersionMd. This is to ensure that,
            # when the metadata is later stored in the catalog of a different Pixeltable instance, the values of
            # current_version and current_schema_version will always point to versions that are known to the
            # destination catalog.
            ancestor_md.tbl_md.current_version = ancestor_md.version_md.version
            ancestor_md.tbl_md.current_schema_version = ancestor_md.schema_version_md.schema_version

        return md

    def _load_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        """Creates TableVersion instance from stored metadata and registers it in _tbl_versions."""
        tbl_md, _, schema_version_md = self.load_tbl_md(tbl_id, effective_version)
        view_md = tbl_md.view_md

        conn = Env.get().conn

        # load mutable view ids for mutable TableVersions
        mutable_view_ids: list[UUID] = []
        # If this is a replica, effective_version should not be None. We see this today, because
        # the replica's TV instance's Column instances contain value_expr_dicts that reference the live version.
        # This is presumably a source of bugs, because it ignores schema version changes (eg, column renames).
        # TODO: retarget the value_expr_dict when instantiating Columns for a particular TV instance.
        if effective_version is None and not tbl_md.is_replica:
            q = sql.select(schema.Table.id).where(
                sql.text(
                    f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r} "
                    "AND md->'view_md'->'base_versions'->0->>1 IS NULL"
                )
            )
            mutable_view_ids = [r[0] for r in conn.execute(q).all()]
        mutable_views = [TableVersionHandle(id, None) for id in mutable_view_ids]

        tbl_version: TableVersion
        if view_md is None:
            # this is a base table
            tbl_version = TableVersion(
                tbl_id, tbl_md, effective_version, schema_version_md, mutable_views=mutable_views
            )
        else:
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

        self._tbl_versions[tbl_id, effective_version] = tbl_version
        tbl_version.init()
        return tbl_version

    def record_column_dependencies(self, tbl_version: TableVersion) -> None:
        """Update self._column_dependencies. Only valid for non-snapshot versions."""
        from pixeltable.exprs import Expr

        assert not tbl_version.is_snapshot
        dependencies: dict[QColumnId, set[QColumnId]] = {}
        for col in tbl_version.cols_by_id.values():
            if col.value_expr_dict is None:
                continue
            dependencies[QColumnId(tbl_version.id, col.id)] = Expr.get_refd_column_ids(col.value_expr_dict)
        self._column_dependencies[tbl_version.id] = dependencies

    def _init_store(self) -> None:
        """One-time initialization of the stored catalog. Idempotent."""
        self.create_user(None)
        _logger.info('Initialized catalog.')

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
                    f'Path {str(path)!r} already exists but is not a {obj_type_str}. '
                    f'Cannot {if_exists.name.lower()} it.'
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
                    f'Directory {str(path)!r} already exists and is not empty. '
                    'Use `if_exists="replace_force"` to replace it.'
                )
            self._drop_dir(obj._id, path, force=True)
        else:
            assert isinstance(obj, Table)
            self._drop_tbl(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
        return None
