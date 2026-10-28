"""Functions and primitives related to Catalog locking. See docs/design/catalog-locking.md."""

from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict
from typing import TYPE_CHECKING, Collection, Iterable
from uuid import UUID

import sqlalchemy as sql

from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime

from .path import Path
from .types import TableVersionKey

if TYPE_CHECKING:
    from .table_path import TableVersionPath
    from .table_version_handle import TableVersionHandle


def finalized_view_base_id(tbl_md: schema.TableMd) -> UUID | None:
    """Return the mutable base whose view_sn must advance when this view's metadata is deleted, or None."""
    if tbl_md.view_md is None or not tbl_md.is_mutable:
        return None
    base_id, base_version = tbl_md.view_md.base_versions[0]
    if base_version is not None:
        return None
    if tbl_md.pending_stmt == schema.TableStatement.DROP_TABLE:
        return UUID(base_id)
    if tbl_md.tbl_state == schema.TableState.ROLLBACK and tbl_md.pending_stmt == schema.TableStatement.CREATE_VIEW:
        return UUID(base_id)
    return None


def get_store_tbl_name(tbl_id: UUID, *, is_view: bool) -> str:
    from pixeltable.store import StoreBase

    return StoreBase.storage_name(tbl_id, is_view)


class StaleLockSetError(Exception):
    """The proposed lock set contains a missing table or does not cover all affected tables and directories.

    Catalog clears the relevant cached metadata and retries with a lock set read from the store.
    """


class LockNotAvailableError(Exception):
    """Another transaction holds or is waiting for a conflicting lock."""

    def __init__(self) -> None:
        super().__init__('conflicting table lock held by another transaction')


class TblOpClass(enum.Enum):
    """Classes of operations performed on tables and the catalog."""

    DATA_READ = 'data_read'  # query table rows
    DATA_WRITE = 'data_write'  # insert/update/delete table rows
    MD_READ = 'md_read'  # table or dir metadata read
    MD_WRITE = 'md_update'  # table or dir metadata write
    FINALIZE = 'finalize'  # table pending op finalization


class TblLockMode(enum.Enum):
    """The Postgres table lock modes that Pixeltable uses."""

    # Blocks ACCESS_EXCLUSIVE only
    ACCESS_SHARE = 'ACCESS SHARE', 1
    # Blocks EXCLUSIVE, ACCESS_EXCLUSIVE
    ROW_EXCLUSIVE = 'ROW EXCLUSIVE', 2
    # Blocks ROW_EXCLUSIVE, EXCLUSIVE, ACCESS_EXCLUSIVE
    EXCLUSIVE = 'EXCLUSIVE', 3
    # Blocks all on this list
    ACCESS_EXCLUSIVE = 'ACCESS EXCLUSIVE', 4

    sql_name: str
    # Relative lock strength, used by is_at_least() and strongest_of().
    _strength: int

    def __init__(self, sql_name: str, strength: int) -> None:
        self.sql_name = sql_name
        self._strength = strength

    def is_at_least(self, other: 'TblLockMode') -> bool:
        """Return whether this mode is at least as strong as `other`."""
        return self._strength >= other._strength

    @classmethod
    def strongest_of(cls, modes: Iterable['TblLockMode']) -> 'TblLockMode':
        """Return the strongest mode in the nonempty iterable `modes`."""
        return max(modes, key=lambda mode: mode._strength)


def tbl_lock_mode(op_class: TblOpClass, is_data_versioned: bool) -> TblLockMode:
    """Return the table lock mode required for this operation.

    Operational writes use ROW EXCLUSIVE, which allows concurrent writers. Data-versioned writes use EXCLUSIVE
    so writers run one at a time and preserve a linear version history. Both modes allow readers.

    Schema changes and pending-op finalization use ACCESS EXCLUSIVE, which conflicts with all of these modes.
    """
    if op_class is TblOpClass.DATA_READ:
        return TblLockMode.ACCESS_SHARE
    if op_class is TblOpClass.DATA_WRITE:
        return TblLockMode.EXCLUSIVE if is_data_versioned else TblLockMode.ROW_EXCLUSIVE
    assert op_class in (TblOpClass.MD_WRITE, TblOpClass.FINALIZE)
    return TblLockMode.ACCESS_EXCLUSIVE


def should_wait(op_class: TblOpClass, any_data_versioned: bool) -> bool:
    """Return whether to wait for locks that are not immediately available, instead of failing.

    The policy applies to the entire lock set. Data reads and writes fail fast when every locked table is operational.
    If any table is data-versioned, they wait for all required locks. Schema changes and pending-op finalization
    always wait.
    """
    if op_class in (TblOpClass.MD_WRITE, TblOpClass.FINALIZE):
        return True
    if op_class is TblOpClass.MD_READ:
        # MD_READ takes no locks, so the return value has no effect.
        return False
    assert op_class in (TblOpClass.DATA_READ, TblOpClass.DATA_WRITE)
    return any_data_versioned


@dataclasses.dataclass(frozen=True)
class LockTarget:
    """A store table and the lock mode required for its role in the operation."""

    tbl_id: UUID
    store_tbl_name: str
    mode: TblLockMode

    def lock_stmt(self, *, wait: bool) -> str:
        """Return the LOCK TABLE statement that acquires this target's lock."""
        nowait_clause = '' if wait else ' NOWAIT'
        return f'LOCK TABLE {self.store_tbl_name} IN {self.mode.sql_name} MODE{nowait_clause}'


@dataclasses.dataclass(frozen=True)
class LockSet:
    """Everything a transaction locks, in acquisition order: store tables first, then `dirs` rows."""

    # tables, sorted by store table name, one entry per table
    tbl_targets: tuple[LockTarget, ...] = ()
    # dirs, sorted by directory path
    dir_ids: tuple[UUID, ...] = ()
    # whether to wait for locks that are not immediately available; if False, raise LockNotAvailableError
    wait: bool = True

    def __post_init__(self) -> None:
        names = [t.store_tbl_name for t in self.tbl_targets]
        assert names == sorted(set(names)), names


class LockRequest:
    """The tables and catalog paths that an operation needs, and how it accesses them."""

    op_class: TblOpClass
    # tables whose rows the operation reads, along with their bases
    read_tvps: Collection[TableVersionPath]
    read_tbl_keys: Collection[TableVersionKey]
    # tables the operation writes as op_class; the bases of write_tvps are read
    write_tvps: Collection[TableVersionPath]
    write_tbl_ids: Collection[UUID]
    # also lock each write target's mutable views, recursively
    lock_mutable_tree: bool
    # write catalog paths
    write_paths: Collection[Path]
    # also lock each directory's descendants and tables, or each table's views and snapshots
    lock_path_subtree: bool

    def __init__(
        self,
        op_class: TblOpClass,
        *,
        read_tvps: Collection[TableVersionPath] | None = None,
        read_tbl_keys: Collection[TableVersionKey] | None = None,
        write_tvps: Collection[TableVersionPath] | None = None,
        write_tbl_ids: Collection[UUID] | None = None,
        lock_mutable_tree: bool = False,
        write_paths: Collection[Path] | None = None,
        lock_path_subtree: bool = False,
    ) -> None:
        self.op_class = op_class
        self.read_tvps = read_tvps or ()
        self.read_tbl_keys = read_tbl_keys or ()
        self.write_tvps = write_tvps or ()
        self.write_tbl_ids = write_tbl_ids or ()
        self.lock_mutable_tree = lock_mutable_tree
        self.write_paths = write_paths or ()
        self.lock_path_subtree = lock_path_subtree
        assert self.for_write or not (self.write_tvps or self.write_tbl_ids or self.write_paths), op_class
        assert self.op_class is not TblOpClass.DATA_READ or (self.read_tvps or self.read_tbl_keys), (
            'DATA_READ requires at least one table or table path'
        )
        assert len(self.write_paths) == 0 or self.op_class is TblOpClass.MD_WRITE, op_class

    @property
    def for_write(self) -> bool:
        return self.op_class in (TblOpClass.DATA_WRITE, TblOpClass.MD_WRITE, TblOpClass.FINALIZE)

    def mutable_write_tbl_ids(self) -> list[UUID]:
        """Return the ids of the write targets that are mutable tables."""
        ids = [tvp.tbl_id for tvp in self.write_tvps if tvp.effective_version() is None]
        ids.extend(self.write_tbl_ids)
        return ids

    def tbl_keys(self) -> list[TableVersionKey]:
        """Return the keys of all tables in the request, including the bases on its paths."""
        keys = [key for tvp in (*self.read_tvps, *self.write_tvps) for key in tvp.tbl_keys]
        keys.extend(self.read_tbl_keys)
        keys.extend(TableVersionKey(tbl_id, None) for tbl_id in self.write_tbl_ids)
        return keys

    def single_tbl(self) -> tuple[TableVersionHandle | None, UUID | None]:
        """Return (tbl, None) or (None, tbl_id) iff the request contains exactly one table; else (None, None)."""
        total = len(self.write_tvps) + len(self.read_tvps) + len(self.read_tbl_keys) + len(self.write_tbl_ids)
        if total != 1:
            return None, None
        if self.write_tvps or self.read_tvps:
            return next(iter(self.write_tvps or self.read_tvps)).tbl_version, None
        if self.read_tbl_keys:
            return None, next(iter(self.read_tbl_keys)).tbl_id
        return None, next(iter(self.write_tbl_ids))


def active_tbl_clause(
    *, tbl_id: UUID | None = None, dir_id: UUID | None = None, tbl_name: str | None = None
) -> sql.ColumnElement[bool]:
    """Create a clause that filters out dropped tables in addition to the specified conditions."""
    # avoid tables that are in the process of getting dropped
    clause = sql.func.coalesce(schema.Table.md['pending_stmt'].astext, '-1') != str(
        schema.TableStatement.DROP_TABLE.value
    )
    if tbl_id is not None:
        clause = sql.and_(schema.Table.id == tbl_id, clause)
    if dir_id is not None:
        clause = sql.and_(schema.Table.dir_id == dir_id, clause)
    if tbl_name is not None:
        clause = sql.and_(schema.Table.md['name'].astext == tbl_name, clause)
    return clause


def lookup_dir_md(path: Path) -> schema.Dir | None:
    """Return the current user's directory at `path`, or None if it does not exist. Takes no locks."""
    user = Env.get().user
    conn = get_runtime().conn
    if path.is_root:
        q = sql.select(schema.Dir).where(schema.Dir.parent_id.is_(None), schema.Dir.md['user'].astext == user)
    else:
        parent_dir = lookup_dir_md(path.parent)
        if parent_dir is None:
            return None
        q = sql.select(schema.Dir).where(
            schema.Dir.parent_id == parent_dir.id,
            schema.Dir.md['name'].astext == path.name,
            schema.Dir.md['user'].astext == user,
        )
    row = conn.execute(q).one_or_none()
    return schema.Dir(**row._mapping) if row is not None else None


def lookup_dir_path(dir_id: UUID) -> Path:
    """Return path for directory with given id"""
    assert isinstance(dir_id, UUID)
    conn = get_runtime().conn
    names: list[str] = []
    while True:
        q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
        row = conn.execute(q).one()
        dir = schema.Dir(**row._mapping)
        if dir.md['name'] == '':
            break
        names.insert(0, dir.md['name'])
        dir_id = dir.parent_id
    return Path.parse('/'.join(names), allow_empty_path=True)


class _LockSetBuilder:
    """Collects a LockRequest's lock requirements from stored metadata."""

    request: LockRequest
    conn: sql.Connection
    # Table id -> required operation classes. Start with requested targets, then add related tables.
    # Requested ids may refer to tables that have already been dropped.
    tbl_id_to_op_classes: defaultdict[UUID, set[TblOpClass]]
    # table id -> (store table name, is_data_versioned), for existing tables with physical storage
    store_tbls: dict[UUID, tuple[str, bool]]
    # Directory id -> local catalog path. Paths determine directory lock order.
    dirs_to_lock: dict[UUID, Path]
    # table ids that we already looked up, including those that did not exist in the store
    visited_tbl_ids: set[UUID]
    # table id -> its parent dir id
    tbl_dirs: dict[UUID, UUID]
    # table id -> base id, only for views whose finalization will advance the base's view_sn
    finalized_view_bases: dict[UUID, UUID]

    def __init__(self, request: LockRequest) -> None:
        self.request = request
        self.conn = get_runtime().conn
        self.tbl_id_to_op_classes = defaultdict(set)
        self.store_tbls = {}
        self.dirs_to_lock = {}
        self.visited_tbl_ids = set()
        self.tbl_dirs = {}
        self.finalized_view_bases = {}

    def build(self) -> LockSet:
        self._add_request_targets()
        path_tbl_ids = self._add_write_paths()
        if self.request.lock_mutable_tree:
            self._visit_views(self.request.mutable_write_tbl_ids(), mutable_only=True)
        # Tables at or under the write paths, including their views and snapshots
        path_tree_ids: set[UUID] = set()
        if self.request.lock_path_subtree:
            path_tree_ids = self._visit_views(path_tbl_ids, mutable_only=False)
        self._visit_remaining()
        if self.request.op_class is TblOpClass.FINALIZE:
            self._add_finalized_view_bases()
        self._add_view_dirs(path_tree_ids)
        return self._lock_set()

    def _add_request_targets(self) -> None:
        # A write path locks its leaf for write if it is a live table, and everything else for read.
        for tvp in self.request.read_tvps:
            for key in tvp.tbl_keys:
                self.tbl_id_to_op_classes[key.tbl_id].add(TblOpClass.DATA_READ)
        for tvp in self.request.write_tvps:
            for i, key in enumerate(tvp.tbl_keys):
                is_live_tbl_leaf = i == 0 and key.effective_version is None
                self.tbl_id_to_op_classes[key.tbl_id].add(
                    self.request.op_class if is_live_tbl_leaf else TblOpClass.DATA_READ
                )
        for key in self.request.read_tbl_keys:
            self.tbl_id_to_op_classes[key.tbl_id].add(TblOpClass.DATA_READ)
        for tbl_id in self.request.write_tbl_ids:
            self.tbl_id_to_op_classes[tbl_id].add(self.request.op_class)

    def _add_write_paths(self) -> set[UUID]:
        # Resolve catalog paths; include all directory descendants when lock_path_subtree is requested.
        write_path_tbl_ids: set[UUID] = set()
        for path in self.request.write_paths:
            # path can be a hosted catalog (e.g. pxt://org:db/dir/), but we are only interested in its local form
            path = Path.from_components(path.components)  # noqa: PLW2901
            parent: schema.Dir | None = lookup_dir_md(path.parent)
            if parent is None:
                # the parent dir is gone, nothing to lock
                continue

            # Lock the parent even when the target does not exist, so concurrent creators cannot claim the same name.
            self.dirs_to_lock.setdefault(parent.id, path.parent)

            # If path points at a directory, add it to the lock set
            dir = lookup_dir_md(path)
            if dir is not None:
                self.dirs_to_lock.setdefault(dir.id, path)
                if self.request.lock_path_subtree:
                    write_path_tbl_ids |= self._add_dir_subtree(dir.id, path)
                continue

            # If path does not point at a directory, see if it points at a table, and add that table to the lock set
            tbl_q = sql.select(schema.Table.id).where(active_tbl_clause(dir_id=parent.id, tbl_name=path.name))
            tbl_id = self.conn.execute(tbl_q).scalars().one_or_none()
            if tbl_id is not None:
                write_path_tbl_ids.add(tbl_id)
        for tbl_id in write_path_tbl_ids:
            self.tbl_id_to_op_classes[tbl_id].add(self.request.op_class)
        return write_path_tbl_ids

    def _add_dir_subtree(self, dir_id: UUID, dir_path: Path) -> set[UUID]:
        """Add the directory subtree to dirs_to_lock and return the ids of its visible tables."""
        self.dirs_to_lock[dir_id] = dir_path
        tbl_q = sql.select(schema.Table.id).where(active_tbl_clause(dir_id=dir_id))
        tbl_ids = {r.id for r in self.conn.execute(tbl_q).all()}

        subdir_q = sql.select(schema.Dir.id, schema.Dir.md).where(schema.Dir.parent_id == dir_id)
        for row in self.conn.execute(subdir_q).all():
            tbl_ids |= self._add_dir_subtree(row.id, dir_path.append(row.md['name']))
        return tbl_ids

    def _visit_tables(self, where: sql.ColumnElement, op_class: TblOpClass) -> set[UUID]:
        """Read matching table metadata, record each table's lock requirements, and return its id.

        Add ancestors to the tables to visit, with DATA_READ as their operation class.
        """
        rows = self.conn.execute(sql.select(schema.Table.id, schema.Table.dir_id, schema.Table.md).where(where)).all()
        read_ids: set[UUID] = set()
        for row in rows:
            read_ids.add(row.id)
            self.visited_tbl_ids.add(row.id)
            self.tbl_id_to_op_classes[row.id].add(op_class)
            tbl_md = schema.md_from_dict(schema.TableMd, row.md)
            if row.dir_id is not None:
                self.tbl_dirs[row.id] = row.dir_id
            if not tbl_md.is_pure_snapshot:
                store_tbl_name = get_store_tbl_name(row.id, is_view=tbl_md.view_md is not None)
                self.store_tbls[row.id] = (store_tbl_name, tbl_md.is_data_versioned)
            view_base_id = finalized_view_base_id(tbl_md)
            if view_base_id is not None:
                self.finalized_view_bases[row.id] = view_base_id
            if tbl_md.view_md is not None:
                for hex_id, _ in tbl_md.view_md.base_versions:
                    ancestor_id = UUID(hex_id)
                    self.tbl_id_to_op_classes[ancestor_id].add(TblOpClass.DATA_READ)
        return read_ids

    def _visit_views(self, tbl_ids: Iterable[UUID], *, mutable_only: bool) -> set[UUID]:
        """Visit the given tables and their descendant views, returning all visited ids.

        When `mutable_only` is True, skip views whose base version is pinned.
        """
        snapshot_filter = sql.true()
        if mutable_only:
            # Exclude snapshots by selecting only where the base effective version is None
            snapshot_filter = schema.Table.md['view_md']['base_versions'][0][1].astext.is_(None)

        # Walk one level of views at a time until no unvisited descendants remain.
        visited_ids = set()
        next_ids = set(tbl_ids)
        while len(next_ids) > 0:
            where = sql.and_(
                schema.Table.md['view_md']['base_versions'][0][0].astext.in_([i.hex for i in next_ids]), snapshot_filter
            )
            discovered_view_ids = self._visit_tables(where, self.request.op_class)
            visited_ids |= next_ids
            next_ids = discovered_view_ids - visited_ids
        return visited_ids

    def _visit_remaining(self) -> None:
        # Visiting a table can add its ancestors. Continue until every discovered id has been checked.
        while True:
            todo = self.tbl_id_to_op_classes.keys() - self.visited_tbl_ids
            if len(todo) == 0:
                break
            self._visit_tables(schema.Table.id.in_(todo), TblOpClass.DATA_READ)
            # Mark missing ids as visited too, so dropped tables do not cause an endless loop.
            self.visited_tbl_ids.update(todo)

    def _add_finalized_view_bases(self) -> None:
        # Require an exclusive lock on the base if finalization will advance its view_sn.
        for tbl_id in self.request.write_tbl_ids:
            base_id = self.finalized_view_bases.get(tbl_id)
            if base_id is not None:
                self.tbl_id_to_op_classes[base_id].add(TblOpClass.FINALIZE)

    def _add_view_dirs(self, tbl_ids: set[UUID]) -> None:
        for tbl_id in tbl_ids:
            # Views outside the target directory subtree also need their parent directories locked.
            dir_id = self.tbl_dirs.get(tbl_id)
            if dir_id is not None:
                self.dirs_to_lock.setdefault(dir_id, lookup_dir_path(dir_id))

    def _lock_set(self) -> LockSet:
        sorted_tbl_targets = tuple(
            LockTarget(
                tbl_id=tbl_id,
                store_tbl_name=store_tbl_name,
                mode=TblLockMode.strongest_of(
                    tbl_lock_mode(op_class, is_data_versioned=is_data_versioned)
                    for op_class in self.tbl_id_to_op_classes[tbl_id]
                ),
            )
            for tbl_id, (store_tbl_name, is_data_versioned) in sorted(
                self.store_tbls.items(), key=lambda item: item[1][0]
            )
        )
        sorted_dir_ids = tuple(dir_id for dir_id, _ in sorted(self.dirs_to_lock.items(), key=lambda item: item[1]))
        return LockSet(
            tbl_targets=sorted_tbl_targets,
            dir_ids=sorted_dir_ids,
            wait=should_wait(
                self.request.op_class, any(is_data_versioned for _, is_data_versioned in self.store_tbls.values())
            ),
        )


def resolve_lock_set_from_store(request: LockRequest) -> LockSet:
    """Build a lock set from stored metadata in the current transaction.

    Read raw `tables` rows without constructing TableVersion instances, so pending operations do not prevent
    lock-set discovery. Targets named by id or TableVersionPath include tables pending drop; catalog-path
    lookups include only visible tables.

    Apply the request's `op_class` to write targets and the views or subtrees affected by their writes. Lock other
    tables, including ancestors, for read. If a table has several roles, use the strongest required mode.
    """
    assert get_runtime().in_xact
    return _LockSetBuilder(request).build()
