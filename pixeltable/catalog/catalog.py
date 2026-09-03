from __future__ import annotations

import contextvars
import dataclasses
import enum
import functools
import itertools
import logging
import random
import time
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterator, Literal, Mapping, Sequence, TypeVar
from uuid import UUID, uuid4

import psycopg
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc
from sqlalchemy.dialects.postgresql import array as pg_array

import pixeltable.index as index
from pixeltable import exceptions as excs, exprs, func, telemetry
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec
from pixeltable.utils import fault_injection
from pixeltable.utils.fault_injection import FaultLocation

from .catalog_base import CatalogBase
from .column import Column
from .dir import Dir
from .globals import DirEntry, IfExistsParam, IfNotExistsParam, IndexSpec, MediaValidation, fold_identifier
from .insertable_table import InsertableTable
from .local_table import LocalTable
from .model import IndexDeclaration, TableSchemaChangeSet, prepare_model, prepare_model_updates
from .path import ROOT_PATH, Path
from .schema_object import SchemaObject
from .table_path import TablePath, TableVersionPath
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .tbl_ops import DeleteTableMdOp, OpStatus, TableOp
from .types import QColumnId, TableVersionKey, TableVersionMd
from .update_status import UpdateStatus
from .view import View

if TYPE_CHECKING:
    import pixeltable as pxt
    from pixeltable.plan import SampleClause

    from .. import exprs


_logger = logging.getLogger(__name__)


def _unpack_row(row: sql.engine.Row | None, entities: list[type[sql.orm.decl_api.DeclarativeBase]]) -> list[Any] | None:
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


# if True, write transactions are prohibited
_frozen: contextvars.ContextVar[bool] = contextvars.ContextVar('pxt_catalog_frozen', default=False)

# -1: unlimited
# for now, we don't limit the number of retries, because we haven't seen situations where the actual number of retries
# grows uncontrollably
_MAX_RETRIES = -1

# Max internal cache size
_MAX_TBL_CACHE_SIZE = 1024

T = TypeVar('T')


def _validate_folded_names(tbl_md: schema.TableMd, schema_version_md: schema.SchemaVersionMd) -> None:
    """Verify that the identifiers of a table about to be written are folded."""
    assert tbl_md.name == fold_identifier(tbl_md.name), tbl_md.name
    for idx_md in tbl_md.index_md.values():
        assert idx_md.name == fold_identifier(idx_md.name), idx_md.name
    for schema_col in schema_version_md.columns.values():
        assert schema_col.name is None or schema_col.name == fold_identifier(schema_col.name), schema_col.name


def _is_retryable_exc(e: BaseException) -> bool:
    """True if e is a transient database failure that can be retried."""
    if not isinstance(e, sql_exc.DBAPIError):
        return False
    # connection_invalidated: the connection was terminated by the server (eg, by pg_terminate_backend)
    # TODO: Investigate whether DeadlockDetected points to a bug in our locking protocol, which is
    # supposed to be deadlock-free.
    return e.connection_invalidated or isinstance(
        e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.DeadlockDetected)
    )


def _store_tbl_name(tbl_id: UUID, *, is_view: bool) -> str:
    from pixeltable.store import StoreBase

    return StoreBase.storage_name(tbl_id, is_view)


def retry_read_md_loop(
    *, tbl_keys: Collection[TableVersionKey] | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry loop for an operation that reads catalog metadata but not table row data.

    tbl_keys specifies tables whose metadata the operation reads; their cached state is refreshed, but no locks are
    taken.
    """
    return _retry_loop(op_class=_TblOpClass.MD_READ, read_tbl_keys=tbl_keys)


def retry_read_loop(
    *, tvps: Collection[TableVersionPath] | None = None, tbl_keys: Collection[TableVersionKey] | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry loop for an operation that reads table data. Locks every table in the paths for read."""
    return _retry_loop(op_class=_TblOpClass.DATA_READ, read_tvps=tvps, read_tbl_keys=tbl_keys)


def retry_schema_change_loop(
    *,
    tvps: Collection[TableVersionPath] | None = None,
    tbl_keys: Collection[TableVersionKey] | None = None,
    lock_mutable_tree: bool = False,
    paths: Collection[Path] | None = None,
    lock_path_subtree: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry loop for an operation that writes table md or runs DDL. It obtains exclusive locks on the targets."""
    return _retry_loop(
        op_class=_TblOpClass.MD_WRITE,
        write_tvps=tvps,
        write_tbl_keys=tbl_keys,
        lock_mutable_tree=lock_mutable_tree,
        write_paths=paths,
        lock_path_subtree=lock_path_subtree,
    )


def _retry_loop(
    *,
    op_class: _TblOpClass,
    read_tvps: Collection[TableVersionPath] | None = None,
    read_tbl_keys: Collection[TableVersionKey] | None = None,
    write_tvps: Collection[TableVersionPath] | None = None,
    write_tbl_keys: Collection[TableVersionKey] | None = None,
    lock_mutable_tree: bool = False,
    write_paths: Collection[Path] | None = None,
    lock_path_subtree: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(op: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(op)
        def loop(*args: Any, **kwargs: Any) -> T:
            cat = get_runtime().catalog
            # a retry loop is reentrant
            if cat._in_retry_loop:
                return op(*args, **kwargs)
            num_retries = 0
            while True:
                cat._in_retry_loop = True
                cat._roll_forward_ids.clear()
                try:
                    # in order for retry to work, we need to make sure that there aren't any prior db updates
                    # that are part of an ongoing transaction
                    assert not get_runtime().in_xact
                    with (
                        cat._allow_tbl_md_read(),
                        cat._begin_xact(
                            op_class=op_class,
                            read_tvps=read_tvps,
                            read_tbl_keys=read_tbl_keys,
                            write_tvps=write_tvps,
                            write_tbl_keys=write_tbl_keys,
                            lock_mutable_tree=lock_mutable_tree,
                            write_paths=write_paths,
                            lock_path_subtree=lock_path_subtree,
                            convert_db_excs=False,
                            finalize_pending_ops=True,
                        ),
                    ):
                        return op(*args, **kwargs)
                except PendingTableOpsError as e:
                    Env.get().console_logger.debug(f'retry loop: finalizing pending ops for {e.tbl_id}')
                    cat._finalize_pending_ops(e.tbl_id)
                except sql_exc.DBAPIError as e:
                    if _is_retryable_exc(e):
                        if num_retries < _MAX_RETRIES or _MAX_RETRIES == -1:
                            num_retries += 1
                            _logger.debug(f'Retrying ({num_retries}) after {type(e.orig)}')
                            time.sleep(random.uniform(0.1, 0.5))
                        else:
                            raise excs.ConcurrencyError(
                                excs.ErrorCode.SERIALIZATION_FAILURE,
                                f'Serialization retry limit ({_MAX_RETRIES}) exceeded',
                            ) from e
                    else:
                        raise
                except Exception as e:
                    # for informational/debugging purposes
                    _logger.debug(f'retry loop: passing along {e}')
                    raise
                finally:
                    cat._in_retry_loop = False

        return loop

    return decorator


def retrying_read(op: Callable[[], T], *, read_tvps: Collection[TableVersionPath] | None = None) -> T:
    """Runs a read-only op, retrying transient failures if no transaction is open yet.

    An op that is already inside a transaction joins it and is not retried, since a retry loop cannot be started
    within one.
    """
    if get_runtime().in_xact:
        return op()
    loop = retry_read_md_loop() if read_tvps is None else retry_read_loop(tvps=read_tvps)
    return loop(op)()


class ProhibitedWriteError(Exception):
    """Raised by begin_xact() when _frozen == True."""


class PendingTableOpsError(Exception):
    tbl_id: UUID

    def __init__(self, tbl_id: UUID) -> None:
        self.tbl_id = tbl_id


class _StaleLockSetError(Exception):
    """Raised during the lock set validation at the start of the transaction, this error indicates that the guessed lock
    set for an operation is insufficient to cover all affected paths. This typically means that another process created
    or dropped tables.

    Catalog needs to refresh its caches and try again."""

    def __init__(self) -> None:
        super().__init__('lock set does not match the tables this operation touches')


class _LockNotAvailableError(Exception):
    """A lock is taken by another transaction."""

    def __init__(self) -> None:
        super().__init__('conflicting table lock held by another transaction')


class _TblOpClass(enum.Enum):
    """Classes of operations performed on tables and the catalog."""

    DATA_READ = 'data_read'  # query table rows
    DATA_WRITE = 'data_write'  # insert/update/delete table rows
    MD_READ = 'md_read'  # table or dir metadata read
    MD_WRITE = 'md_update'  # table or dir metadata write
    FINALIZE = 'finalize'  # table pending op finalization


class _TblLockMode(enum.Enum):
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
    # ranks the modes by how exclusive they are, for is_at_least() only
    _strength: int

    def __init__(self, sql_name: str, strength: int) -> None:
        self.sql_name = sql_name
        self._strength = strength

    def is_at_least(self, other: '_TblLockMode') -> bool:
        """True if this mode excludes at least as much as `other`."""
        return self._strength >= other._strength


def _tbl_lock_mode(op_class: _TblOpClass, is_data_versioned: bool) -> _TblLockMode:
    """The lock mode required on a table for an operation of `op_class`.

    A data write is the only class whose mode depends on the table type (operational or data-versioned). ROW EXCLUSIVE
    is self-compatible, so concurrent operational writers share the table; EXCLUSIVE conflicts with itself, which is
    the write-serialization point a data-versioned table needs. Both are compatible with ACCESS SHARE, so neither
    blocks readers.
    """
    if op_class is _TblOpClass.DATA_READ:
        return _TblLockMode.ACCESS_SHARE
    if op_class is _TblOpClass.DATA_WRITE:
        return _TblLockMode.EXCLUSIVE if is_data_versioned else _TblLockMode.ROW_EXCLUSIVE
    assert op_class in (_TblOpClass.MD_WRITE, _TblOpClass.FINALIZE)
    return _TblLockMode.ACCESS_EXCLUSIVE


def _tbl_lock_blocking(op_class: _TblOpClass, is_data_versioned: bool) -> bool:
    """Whether an operation of `op_class` waits for the locks to become available, rather than failing fast.

    The wait policy belongs to the entire operation, not individual tables involved. Today we only support operations
    involving only data-versioned tables or only operational tables, but not both. TBD: the wait policy when mixed
    table types are involved.
    """
    if op_class in (_TblOpClass.MD_WRITE, _TblOpClass.FINALIZE):
        return True
    if op_class is _TblOpClass.MD_READ:
        # MD_READ doesn't take any locks, so the response is immaterial
        return False
    # Reads and data writes wait on a data-versioned table, but fail fast on an operational one.
    return is_data_versioned


@dataclasses.dataclass(frozen=True)
class LockTarget:
    """A store table of a lock set, and the mode it is to be locked in.

    The wait policy is not here: it is one decision per operation (see _tbl_lock_blocking()), where the mode varies
    from table to table within a single lock set. A write locks its target's store table to write it and its bases'
    to read them.
    """

    store_tbl_name: str
    mode: _TblLockMode
    # selects the mode above, and through LockSet the wait policy
    is_data_versioned: bool


@dataclasses.dataclass(frozen=True)
class LockSet:
    """Everything a transaction locks, in acquisition order: store tables first, then `dirs` rows."""

    tbl_targets: tuple[LockTarget, ...] = ()  # sorted by store table name
    dir_ids: tuple[UUID, ...] = ()  # sorted by directory path
    # The kind of the tables this transaction touches, which selects its wait policy; None when it touches none.
    # A set spanning both kinds would have two policies and no single answer. Nothing produces one today, because a
    # query over an operational table is restricted to a single table in its from-clause, so _make_lock_set()
    # asserts instead of choosing.
    is_data_versioned: bool | None = None


class Catalog(CatalogBase):
    """The functional interface to getting access to catalog objects

    Locking: docs/design/catalog-locking.md is the full account; what follows is the brief summary.

    Concurrent access to a table is synchronized with Postgres table locks (LOCK TABLE) on its store tables. The
    store table is the lock, and Postgres's conflict matrix is the protocol. Every lock is transaction-scoped.

    All interface functions must be called in the context of a transaction, started with one of the
    begin_*_xact() methods or via one of the retry_*_loop() decorators. Each of those selects the appropriate level of
    lock exclusivity, and the wait policy.

    Every transaction that locks anything runs the same five steps:
    1. guess the lock set from the metadata cache, or read it from the store in a separate transaction
    2. acquire the locks: store tables first, then `dirs` rows
    3. read current metadata
    4. validate the guess against it, and restart on a mismatch
    5. finally do the actual work

    When calling functions that involve Table or TableVersion instances, the catalog needs to get a chance to finalize
    pending ops against those tables. A table with pending ops is not usable, and step 3 is where that is detected.
    The choice between begin_*_xact() and retry_*_loop() depends on where metadata loads occur relative to the atomic
    operation:
    - If all metadata loads happen at the beginning of an atomic operation (eg, insert/update/delete), use a
      begin_*_xact(). It will finalize pending ops before locking.
    - If metadata loads happen in the middle of an atomic operation, wrap the entire operation in the matching
      retry_*_loop(), which handles pending ops and serialization retries.

    get_tbl_version() manages its own retry loop internally if called outside of a transaction or a retry loop. Callers
    that don't need to perform multiple of these atomically do not need to wrap the call.

    Metadata changes: all Table operations that change metadata need follow this protocol:
    - write the metadata changes to the store in a single transaction, including the op log that implements the updates
    - roll_forward()
    - invalidate any cached TableVersion instances for the affected table and call TVP.clear_cached_md()
    TODO: this is currently only implemented for Table.add_columns()

    Caching and invalidation of metadata:
    - Catalog caches TableVersion instances in order to avoid excessive metadata loading
    - Any updates to the metadata need to include clearing/invalidating the metadata cache
    - Both _tbls and _tbl_versions caches maintain LRU order. At the end of the transaction, Catalog can evict entries
    in excess of _MAX_TBL_CACHE_SIZE from both of them. No eviction during a transaction is possible. To maintain
    the LRU order, every cache hit should move_to_end(key).
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

    # cached TableVersion instances; key: [id, version]
    # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
    # - snapshot versions: records the version of the snapshot
    _tbl_versions: OrderedDict[TableVersionKey, TableVersion]
    _tbls: OrderedDict[TableVersionKey, LocalTable]
    _in_write_xact: bool  # True if we're in a write transaction
    # table locks held in the current transaction: store table name -> mode
    _locks_held: dict[str, _TblLockMode]
    # dir locks held in the current transaction
    _dir_locks_held: set[UUID]
    _modified_tvs: set[TableVersionHandle]  # TableVersion instances modified in the current transaction
    # Ids of Tables that have pending TableOps, in the order they must be finalized.
    _roll_forward_ids: list[UUID]
    _in_retry_loop: bool
    # True within _allow_tbl_md_read(); permits loading table metadata inside begin_xact initialization, or inside
    # a retry_loop, but not in the middle of a regular begin_xact transaction.
    _tbl_md_read_allowed: bool

    # cached column dependencies
    # - key: table id, value: mapping from column id to its dependencies
    # - only maintained for dependencies between non-snapshot table versions
    # - can contain stale entries (stemming from invalidated TV instances)
    _column_dependencies: dict[UUID, dict[QColumnId, set[QColumnId]]]

    # column dependents are recomputed at the beginning of every write transaction and only reflect the tables
    # locked for write
    _column_dependents: dict[QColumnId, set[QColumnId]] | None

    def __init__(self) -> None:
        self._tbl_versions = OrderedDict()
        self._tbls = OrderedDict()
        self._in_write_xact = False
        self._locks_held = {}
        self._dir_locks_held = set()
        self._modified_tvs = set()
        self._roll_forward_ids = []
        self._in_retry_loop = False
        self._tbl_md_read_allowed = False
        self._column_dependencies = {}
        self._column_dependents = None
        self._init_store()

    def __deepcopy__(self, memo: dict[int, object]) -> 'Catalog':
        # Catalog instances are owned by Runtime and never duplicated. Return self here to prevent deepcopies.
        memo[id(self)] = self
        return self

    def _active_tbl_clause(
        self, *, tbl_id: UUID | None = None, dir_id: UUID | None = None, tbl_name: str | None = None
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

    def _validate_tbls_exist(self, tbl_ids: Collection[UUID]) -> None:
        conn = get_runtime().conn
        assert conn is not None
        for tbl_id in tbl_ids:
            q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(tbl_id=tbl_id))
            if conn.execute(q).scalar() == 0:
                raise excs.table_was_dropped(tbl_id)

    def validate_tbls_exist(self, tbl_ids: Collection[UUID]) -> None:
        """Raises TABLE_NOT_FOUND if any id is not a live (not dropped or being dropped) table."""
        # Only retry when this call opens the outermost transaction; when it runs nested inside an existing transaction,
        # the outer transaction handles recovery.
        if get_runtime().in_xact:
            self._validate_tbls_exist(tbl_ids)
        else:
            retry_read_md_loop()(self._validate_tbls_exist)(tbl_ids)

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
                base = tbl_version.base
                assert base is not None, f'View {tbl_version.id} is missing base'
                if base.effective_version is None:
                    key = TableVersionKey(base.id, None)
                    base_tv = self._tbl_versions.get(key, None)
                    if base_tv is not None and base_tv.is_validated and tbl_version.handle not in base_tv.mutable_views:
                        mutable_view_ids = ', '.join(str(tv.id) for tv in base_tv.mutable_views)
                        mutable_view_names = ', '.join(tv.get().name for tv in base_tv.mutable_views)
                        raise AssertionError(
                            f'{tbl_version.name} ({tbl_version.id}) missing in '
                            f'{mutable_view_ids} ({mutable_view_names})'
                        )

            if len(tbl_version.mutable_views) > 0:
                # make sure we also loaded mutable view metadata, which is needed to detect column dependencies
                for v in tbl_version.mutable_views:
                    assert v.effective_version is None, f'{v.id}:{v.effective_version}'

    def mark_modified_tv(self, handle: TableVersionHandle) -> None:
        """Record that the given TableVersion instance was modified in the current transaction"""
        assert get_runtime().in_xact
        self._modified_tvs.add(handle)

    @contextmanager
    def _allow_tbl_md_read(self) -> Iterator[None]:
        """Context manager that sets self._tbl_md_read_allowed and thus allows reading new table metadata."""
        if self._tbl_md_read_allowed:
            # this ctx manager is reentrant
            yield
            return
        self._tbl_md_read_allowed = True
        try:
            yield
        finally:
            self._tbl_md_read_allowed = False

    @contextmanager
    def freeze(self) -> Iterator[None]:
        """Prevent write transactions in this thread.

        Unaffected: reads and writes from other threads.
        """
        token = _frozen.set(True)
        try:
            yield
        finally:
            _frozen.reset(token)

    @contextmanager
    def begin_read_md_xact(self, *, tbl_keys: Collection[TableVersionKey] | None = None) -> Iterator[sql.Connection]:
        """A transaction that reads catalog md and no table data. It takes no locks at all.

        Such a transaction may observe a schema change in progress and report a schema that is about to change,
        which is preferable to list_tables() or describe() blocking or failing. tbl_keys name tables whose md is
        read: their cache entries are refreshed, but nothing is locked.
        """
        with self._begin_xact(op_class=_TblOpClass.MD_READ, read_tbl_keys=tbl_keys) as conn:
            yield conn

    @contextmanager
    def begin_read_xact(
        self, *, tvps: Collection[TableVersionPath] | None = None, tbl_keys: Collection[TableVersionKey] | None = None
    ) -> Iterator[sql.Connection]:
        """A transaction that reads table data. Locks every table it reads in ACCESS SHARE.

        A read's lock set is its paths, base tables included, since a view read reads base data. tbl_keys are the
        tables an expression pulls in (a @pxt.query UDF, say), whose ancestry is resolved from the md cache.
        """
        with self._begin_xact(op_class=_TblOpClass.DATA_READ, read_tvps=tvps, read_tbl_keys=tbl_keys) as conn:
            yield conn

    @contextmanager
    def begin_write_xact(
        self, *, read_tbl_keys: Collection[TableVersionKey] | None = None, tvps: Collection[TableVersionPath]
    ) -> Iterator[sql.Connection]:
        """A transaction that inserts, updates or deletes rows of the given tables.

        The lock set always covers each target's mutable tree, because a data write propagates there.
        read_tbl_keys specifies the tables to be read, e.g. the source of an insert from a query.
        """
        with self._begin_xact(
            op_class=_TblOpClass.DATA_WRITE, read_tbl_keys=read_tbl_keys, write_tvps=tvps, lock_mutable_tree=True
        ) as conn:
            yield conn

    @contextmanager
    def begin_schema_change_xact(
        self,
        *,
        tvps: Collection[TableVersionPath] | None = None,
        tbl_keys: Collection[TableVersionKey] | None = None,
        lock_mutable_tree: bool = False,
        paths: Collection[Path] | None = None,
        lock_path_subtree: bool = False,
    ) -> Iterator[sql.Connection]:
        """A transaction that writes table md or runs DDL. It obtains exclusive locks on the target tables.

        - tvps/tbl_keys: the tables whose md this operation writes
        - lock_mutable_tree: also lock each target's mutable views
        - paths: catalog paths whose object this operation writes. Each is resolved and locked similarly to tables
        - lock_path_subtree: also lock the catalog subtree of each path: for a directory its subdirectories
          and their tables, and for a table its views, snapshots included.
        """
        with self._begin_xact(
            op_class=_TblOpClass.MD_WRITE,
            write_tvps=tvps,
            write_tbl_keys=tbl_keys,
            lock_mutable_tree=lock_mutable_tree,
            write_paths=paths,
            lock_path_subtree=lock_path_subtree,
        ) as conn:
            yield conn

    @contextmanager
    def _begin_finalize_xact(self, tbl_id: UUID) -> Iterator[sql.Connection]:
        """A transaction of a pending-op finalization: ACCESS EXCLUSIVE on the target, and it always waits.

        Its own op class, so that a fault point can tell it from an ordinary schema change. It does not finalize
        pending ops on entry, since finalizing them is what it is for.
        """
        with self._begin_xact(
            op_class=_TblOpClass.FINALIZE,
            write_tbl_keys=[TableVersionKey(tbl_id, None)],
            convert_db_excs=False,
            finalize_pending_ops=False,
        ) as conn:
            yield conn

    @contextmanager
    def _begin_xact(
        self,
        *,
        op_class: _TblOpClass,
        read_tvps: Collection[TableVersionPath] | None = None,
        read_tbl_keys: Collection[TableVersionKey] | None = None,
        write_tvps: Collection[TableVersionPath] | None = None,
        write_tbl_keys: Collection[TableVersionKey] | None = None,
        lock_mutable_tree: bool = False,
        write_paths: Collection[Path] | None = None,
        lock_path_subtree: bool = False,
        convert_db_excs: bool = True,
        finalize_pending_ops: bool = True,
    ) -> Iterator[sql.Connection]:
        """
        Return a context manager that yields a connection to the database. Idempotent.

        Callers use one of the begin_*_xact() methods above, which is where each op class's parameters and what
        they lock are documented. It is mandatory to go through one of those, not Env.begin_xact(), if the
        transaction accesses any table data or metadata.

        Locking protocol (via _acquire_locks()):
        - the write and read targets name the store tables to lock; the metadata cache is refreshed for all of
          them afterwards
        - store table locks are acquired first, before any statement that would pin the transaction's snapshot; the
          mode and the wait policy follow from op_class and the kind of the tables it locks
        - if finalize_pending_ops == True and a PendingTableOpsError is raised, finalizes pending ops and retries
        - this needs to be done in a retry loop, because Postgres can abort the transaction
          (SerializationFailure, LockNotAvailable)
        - for that reason, we do all lock acquisition prior to doing any real work (eg, compute column values),
          to minimize the probability of losing that work due to a forced abort

        If convert_db_excs == True, converts DBAPIErrors into excs.Errors if possible.
        """
        for_write = op_class in (_TblOpClass.DATA_WRITE, _TblOpClass.MD_WRITE, _TblOpClass.FINALIZE)
        assert for_write or not (write_tvps or write_tbl_keys or write_paths), (op_class, write_tvps, write_tbl_keys)
        assert op_class is not _TblOpClass.DATA_READ or (read_tvps or read_tbl_keys), (
            'a read with no target reads md only'
        )
        if for_write and _frozen.get():
            raise ProhibitedWriteError()
        read_tvps = read_tvps or []
        write_tvps = write_tvps or []
        read_tbl_keys = read_tbl_keys or []
        write_tbl_keys = write_tbl_keys or []
        write_paths = write_paths or []
        if get_runtime().in_xact:
            yield get_runtime().conn
            return

        assert not write_paths or op_class is _TblOpClass.MD_WRITE, (write_paths, op_class)
        num_retries = 0
        num_lock_set_retries = 0
        failed_lock_set: LockSet | None = None
        pending_ops_tbl_id: UUID | None = None
        has_exc = False  # True if we exited the 'with ...begin_xact()' block with an exception
        while True:
            if pending_ops_tbl_id is not None:
                Env.get().console_logger.debug(f'_begin_xact(): finalizing pending ops for {pending_ops_tbl_id}')
                self._finalize_pending_ops(pending_ops_tbl_id)
                pending_ops_tbl_id = None

            prev_failed_lock_set = failed_lock_set
            failed_lock_set = None
            try:
                attempt_lock_set = self._resolve_lock_set(
                    op_class=op_class,
                    read_tvps=read_tvps,
                    read_tbl_keys=read_tbl_keys,
                    write_tvps=write_tvps,
                    write_tbl_keys=write_tbl_keys,
                    lock_mutable_tree=lock_mutable_tree,
                    write_paths=write_paths,
                    lock_path_subtree=lock_path_subtree,
                )
            except PendingTableOpsError as e:
                # a table the lock set covers is mid-schema-change: resolve it and rebuild the set, on the same
                # terms as when the check below finds one after the locks are held
                if not finalize_pending_ops:
                    raise
                pending_ops_tbl_id = e.tbl_id
                continue

            wait_for_locks = _tbl_lock_blocking(op_class, is_data_versioned=attempt_lock_set.is_data_versioned or False)

            if prev_failed_lock_set is not None and attempt_lock_set == prev_failed_lock_set:
                # re-reading the metadata produced the set that just failed, so retrying cannot make progress:
                # this is not a stale guess but metadata that does not describe the store
                raise excs.Error(
                    excs.ErrorCode.INTERNAL_ERROR,
                    'Could not determine the tables to lock for this operation; '
                    'the stored metadata does not match the tables that exist.',
                )

            # one span per acquisition attempt; retries show up as sibling spans
            xact_span = telemetry.span_start('pixeltable.catalog.begin_xact', attrs={'pxt.for_write': for_write})
            attempt_exc: BaseException | None = None
            try:
                self._in_write_xact = for_write
                self._locks_held = {}
                self._dir_locks_held = set()
                self._modified_tvs = set()
                self._column_dependents = None
                has_exc = False

                with get_runtime().begin_xact(for_write=for_write) as conn:
                    with self._allow_tbl_md_read():
                        try:
                            self._acquire_locks(
                                op_class=op_class,
                                read_tvps=read_tvps,
                                read_tbl_keys=read_tbl_keys,
                                write_tvps=write_tvps,
                                write_tbl_keys=write_tbl_keys,
                                lock_mutable_tree=lock_mutable_tree,
                                lock_set=attempt_lock_set,
                                blocking=wait_for_locks,
                                finalize_pending_ops=finalize_pending_ops,
                            )
                            self._validate_lock_set(
                                write_op_class=op_class,
                                write_tvps=write_tvps,
                                write_tbl_keys=write_tbl_keys,
                                lock_mutable_tree=lock_mutable_tree,
                                write_paths=write_paths,
                                lock_path_subtree=lock_path_subtree,
                            )
                            if for_write and lock_mutable_tree:
                                self._compute_column_dependents(write_tvps, write_tbl_keys)
                            if _logger.isEnabledFor(logging.DEBUG):
                                # validate only when we don't see errors
                                self.validate()
                        except PendingTableOpsError as e:
                            has_exc = True
                            if finalize_pending_ops:
                                # we remember which table id to finalize
                                pending_ops_tbl_id = e.tbl_id
                            # raise to abort the transaction
                            raise
                        except sql_exc.DBAPIError as e:
                            # Handle retriable errors
                            has_exc = True
                            if _is_retryable_exc(e) and (num_retries < _MAX_RETRIES or _MAX_RETRIES == -1):
                                _logger.debug(f'Retriable error {type(e.orig)} on attempt {num_retries}')
                                num_retries += 1
                                time.sleep(random.uniform(0.1, 0.5))
                                # attempt failed -- don't try to commit the transaction before retrying
                                self._try_rollback(conn)
                                continue
                            raise

                    # success: end the attempt span here so it covers only the acquisition, not the
                    # caller's work under the yield; the finally below then sees None and no-ops
                    telemetry.span_end(xact_span)
                    xact_span = None
                    yield conn
                    return

            except _LockNotAvailableError as e:
                has_exc = True
                # NOWAIT was refused. The holder is not necessarily the schema change: an ACCESS EXCLUSIVE request
                # queued behind an ordinary read refuses everything arriving after it. But a schema change that is
                # queued is still one in progress, and retrying once it completes is still the advice.
                attempt_exc = e
                raise excs.ConcurrencyError(
                    excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS,
                    'A schema change is in progress; this operation cannot run concurrently with it.\n'
                    'Please retry once it completes.',
                ) from e

            except _StaleLockSetError as e:
                has_exc = True
                if _MAX_RETRIES != -1 and num_lock_set_retries >= _MAX_RETRIES:
                    raise excs.ConcurrencyError(
                        excs.ErrorCode.SERIALIZATION_FAILURE, f'Lock set retry limit ({_MAX_RETRIES}) exceeded'
                    ) from e
                num_lock_set_retries += 1
                # remember the set that failed: if reading the metadata afresh reproduces it, the next attempt
                # cannot do any better and the check above the loop body says so
                failed_lock_set = attempt_lock_set
                # drop the metadata the set was derived from, so that the next attempt reads it afresh and rebuilds
                # the set: without the store table that is gone, or with the view that has since been added
                self._invalidate_lock_set(read_tvps, read_tbl_keys, write_tvps, write_tbl_keys)
                continue

            except PendingTableOpsError as e:
                has_exc = True
                if pending_ops_tbl_id is not None:
                    # the next iteration of the loop will deal with pending ops for this table id
                    continue
                else:
                    # we got this exception after getting the initial table locks and therefore need to abort
                    attempt_exc = e
                    raise

            except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
                has_exc = True
                attempt_exc = e
                single_tbl, single_tbl_id = self._get_single_tbl(read_tvps, read_tbl_keys, write_tvps, write_tbl_keys)
                self.convert_sql_exc(e, tbl_id=single_tbl_id, tbl=single_tbl, convert_db_excs=convert_db_excs)
                raise  # re-raise the error if it didn't convert to a pxt.Error

            except (Exception, KeyboardInterrupt) as e:
                has_exc = True
                attempt_exc = e
                _logger.debug(f'Caught {e.__class__}: {e}', exc_info=True)
                raise

            finally:
                # failure: xact_span is still non-None only if this attempt failed before the yield;
                # attempt_exc is None on retry `continue`s (the attempt span ends clean and the retry
                # shows up as a sibling), non-None when the attempt raised
                telemetry.span_end(xact_span, exc=attempt_exc)
                xact_span = None
                self._in_write_xact = False
                self._locks_held.clear()
                self._dir_locks_held.clear()
                self._column_dependents = None

                # invalidate cached current TableVersion instances
                for tv in self._tbl_versions.values():
                    if tv.effective_version is None:
                        tv.is_validated = False

                # invalidate TVPs' cached md
                # TODO: remove this once we stop mutating TV instances in-place
                for tvp in write_tvps:
                    tvp.clear_cached_md()

                if has_exc:
                    # purge all modified TableVersion instances; we can't guarantee they are still consistent with the
                    # stored metadata
                    for handle in self._modified_tvs:
                        self._clear_tv_cache(handle.key)
                    # Clear potentially corrupted cached metadata
                    for tvp in [*write_tvps, *read_tvps]:
                        tvp.clear_cached_md()

                self._evict_caches()
                self._modified_tvs.clear()

    def _try_rollback(self, conn: sql.Connection) -> None:
        """Initiate rollback, ignoring the failure resulting from a possibly already-dead connection."""
        try:
            conn.rollback()
        except sql_exc.DBAPIError:
            pass

    @classmethod
    def _make_lock_set(cls, targets: Collection[LockTarget], dir_ids: Collection[UUID] = ()) -> LockSet:
        """Assemble a LockSet from every target the operation touches, in acquisition order.

        Callers pass targets in whatever order they collected them; sorting is this function's job, since
        deadlock freedom rests on the acquisition order.
        """
        kinds = {t.is_data_versioned for t in targets}
        assert len(kinds) <= 1, f'lock set spans both table kinds, which have different wait policies: {targets}'
        return LockSet(
            tbl_targets=tuple(sorted(targets, key=lambda t: t.store_tbl_name)),
            dir_ids=tuple(dir_ids),
            is_data_versioned=next(iter(kinds), None),
        )

    def _lock_target_from_cache(self, key: TableVersionKey, op_class: _TblOpClass) -> list[LockTarget] | None:
        """Creates a LockTarget for a paritcular table, as a list so that a skipped table can return [].

        Uses metadata cache only. Returns None on a cache miss.
        """
        tv = self._tbl_versions.get(key)
        if tv is None or not tv.is_initialized:
            return None
        return [
            LockTarget(
                store_tbl_name=_store_tbl_name(tv.id, is_view=tv.is_view),
                mode=_tbl_lock_mode(op_class, tv.is_data_versioned),
                is_data_versioned=tv.is_data_versioned,
            )
        ]

    def _path_lock_targets_from_cache(
        self, keys: Sequence[TableVersionKey], leaf_op_class: _TblOpClass
    ) -> list[LockTarget] | None:
        """Lock targets for the given path, represented by its keys in the view-before-base order.

        leaf_op_class applies to the leaf only. For the ancestors, read operation is assumed.

        Uses metadata cache only. Returns None if not all LockTargets can be created from the cache.
        """
        result: list[LockTarget] = []
        for i, key in enumerate(keys):
            lock_targets = self._lock_target_from_cache(key, leaf_op_class if i == 0 else _TblOpClass.DATA_READ)
            if lock_targets is None:
                return None
            # every element along the way except the last one is a view. The last one is a base table.
            is_view = i < len(keys) - 1
            assert all(t.store_tbl_name == _store_tbl_name(key.tbl_id, is_view=is_view) for t in lock_targets), (
                i,
                keys,
            )
            result.extend(lock_targets)
        return result

    def _ancestors_lock_targets_from_cache(
        self, key: TableVersionKey, leaf_op_class: _TblOpClass
    ) -> list[LockTarget] | None:
        """Lock targets for `key` and its ancestors, from cached metadata only.

        Doesn't talk to the store; uses cached metadata only to build the list. If the cached state is insufficient to
        build the chain, returns None."""
        keys: list[TableVersionKey] = []
        current_key = key
        while True:
            tv = self._tbl_versions.get(current_key)
            if tv is None or not tv.is_initialized:
                return None
            keys.append(current_key)
            # Uses TableVersion.base. Cannot rely on TableVersion.path because path is unset on snapshots.
            if tv.base is None:
                return self._path_lock_targets_from_cache(keys, leaf_op_class)
            current_key = tv.base.key

    def _mutable_tree_lock_targets_from_cache(self, tbl_id: UUID, op_class: _TblOpClass) -> list[LockTarget] | None:
        """Returns lock targets for tbl_id's mutable tree: the target and its transitive mutable views.

        Doesn't talk to the store; uses cached metadata only to build the list. If the cached state is unsufficient to
        build the tree, returns None."""
        key = TableVersionKey(tbl_id, None)
        targets = self._lock_target_from_cache(key, op_class)
        if targets is None:
            return None
        result = list(targets)
        for view in self._tbl_versions[key].mutable_views:
            subtree = self._mutable_tree_lock_targets_from_cache(view.id, op_class)
            if subtree is None:
                return None
            result.extend(subtree)
        return result

    def _lock_set_from_store(
        self,
        *,
        write_op_class: _TblOpClass,
        read_tvps: Collection[TableVersionPath],
        read_tbl_keys: Collection[TableVersionKey],
        write_tvps: Collection[TableVersionPath],
        write_tbl_keys: Collection[TableVersionKey],
        lock_mutable_tree: bool,
        write_paths: Collection[Path] = (),
        lock_path_subtree: bool = False,
    ) -> LockSet:
        """The lock set, read from the store rather than guessed from the metadata cache.

        Used when the cache cannot answer. Reads `tables` rows rather than building TableVersions: everything a
        lock set needs is plain md, so this also answers for a table whose schema change is still in flight, whose
        md cannot be interpreted but can be read. Being current, what it returns needs no validation.

        Rows are fetched one level at a time -- each round asks for every id whose row is still unread -- so the
        ancestors cost at most two rounds, a row naming its whole chain, and the views one round per level.

        Deliberately unfiltered by visibility: a table being dropped still has its store table, and an operation
        that touches it has to be excluded from it until the drop finalizes. Path resolution is the exception: a table
        that is already being dropped is not what the caller named, so there it filters by visibility.
        """
        assert get_runtime().in_xact
        conn = get_runtime().conn
        roles: dict[UUID, _TblOpClass] = {}  # id -> the op class it is locked for
        seen: set[UUID] = set()  # ids whose row we tried to read, whether or not it existed
        names: dict[UUID, str] = {}  # id -> store table to lock; absent means there is none to lock
        versioned: set[UUID] = set()
        tbl_dirs: dict[UUID, UUID] = {}  # id -> the directory the table lives in
        mutable_bases: dict[UUID, UUID] = {}  # id -> the id of its base, for a mutable view of a mutable base
        pending: set[UUID] = set()  # ids named by a row we read, still to be read themselves
        # dir id -> path, the sort key of the `dirs` row locks. Always the catalog-local form: Path orders by
        # (org, db, components), and a path a caller named can carry an org where get_dir_path() never does, so
        # mixing the two forms would compare None against a str.
        dirs: dict[UUID, Path] = {}

        def claim(tbl_id: UUID, op_class: _TblOpClass) -> None:
            """Record that tbl_id is locked for op_class, keeping the strongest role it is reached through."""
            held = roles.get(tbl_id)
            # the op classes rank the same way whichever kind they are applied to, so the kind is immaterial here
            if held is None or _tbl_lock_mode(op_class, False).is_at_least(_tbl_lock_mode(held, False)):
                roles[tbl_id] = op_class

        def ingest(where: sql.ColumnElement, role: _TblOpClass) -> set[UUID]:
            """Read the rows matching `where`, record what they say, and return the ids actually read."""
            rows = conn.execute(sql.select(schema.Table.id, schema.Table.dir_id, schema.Table.md).where(where)).all()
            read_ids: set[UUID] = set()
            for row in rows:
                read_ids.add(row.id)
                seen.add(row.id)
                claim(row.id, role)
                tbl_md = schema.md_from_dict(schema.TableMd, row.md)
                if row.dir_id is not None:
                    tbl_dirs[row.id] = row.dir_id
                if not tbl_md.is_pure_snapshot:
                    names[row.id] = _store_tbl_name(row.id, is_view=tbl_md.view_md is not None)
                    if tbl_md.is_data_versioned:
                        versioned.add(row.id)
                if tbl_md.view_md is not None:
                    if tbl_md.is_mutable and tbl_md.view_md.base_versions[0][1] is None:
                        mutable_bases[row.id] = UUID(tbl_md.view_md.base_versions[0][0])
                    # base_versions is the whole ancestor chain, so one row names every ancestor at once
                    for hex_id, _ in tbl_md.view_md.base_versions:
                        ancestor_id = UUID(hex_id)
                        claim(ancestor_id, _TblOpClass.DATA_READ)
                        pending.add(ancestor_id)
            return read_ids

        def read_pending() -> None:
            while True:
                todo = pending - seen
                if len(todo) == 0:
                    return
                # mark them read up front: a row that does not come back does not exist, and asking again would
                # never terminate
                seen.update(todo)
                ingest(schema.Table.id.in_(todo), _TblOpClass.DATA_READ)

        def list_transitive_views(seed: set[UUID], *, mutable_only: bool) -> set[UUID]:
            """The view tree of every id in seed: the seed itself plus its transitive views.

            mutable_only excludes snapshots, which is what a data write propagates to. A drop reaches every view,
            snapshots included, so it asks for the wider tree.
            """
            snapshot_filter = sql.true()
            if mutable_only:
                # Exclude snapshots, i.e. select only where the base effective version is None
                snapshot_filter = schema.Table.md['view_md']['base_versions'][0][1].astext.is_(None)

            visited_ids = set(seed)
            next_ids = set(seed)
            while len(next_ids) > 0:
                where = sql.and_(
                    schema.Table.md['view_md']['base_versions'][0][0].astext.in_([i.hex for i in next_ids]),
                    snapshot_filter,
                )
                found_view_ids = ingest(where, write_op_class)
                next_ids = found_view_ids - visited_ids
                visited_ids |= found_view_ids
            return visited_ids

        def list_dirs(dir_id: UUID, dir_path: Path) -> tuple[dict[UUID, Path], set[UUID]]:
            """The directory and its subdirectories, as dir id -> path, plus the tables they contain."""
            subtree_dirs = {dir_id: dir_path}
            tbl_ids: set[UUID] = set()
            subdir_q = sql.select(schema.Dir.id, schema.Dir.md).where(schema.Dir.parent_id == dir_id)
            for row in conn.execute(subdir_q).all():
                child_dirs, child_tbl_ids = list_dirs(row.id, dir_path.append(row.md['name']))
                subtree_dirs.update(child_dirs)
                tbl_ids |= child_tbl_ids
            tbl_q = sql.select(schema.Table.id).where(self._active_tbl_clause(dir_id=dir_id))
            tbl_ids.update(r.id for r in conn.execute(tbl_q).all())
            return subtree_dirs, tbl_ids

        # what the caller asked for: only a path's leaf is written, its bases are read
        for tvp in read_tvps:
            for key in tvp.tbl_keys:
                claim(key.tbl_id, _TblOpClass.DATA_READ)
        for tvp in write_tvps:
            for i, key in enumerate(tvp.tbl_keys):
                claim(key.tbl_id, write_op_class if i == 0 else _TblOpClass.DATA_READ)
        for key in read_tbl_keys:
            claim(key.tbl_id, _TblOpClass.DATA_READ)
        for key in write_tbl_keys:
            claim(key.tbl_id, write_op_class)

        # resolve each write path to the object it names; with lock_path_subtree, a directory additionally stands
        # for every table of its subtree
        write_path_tbl_ids: set[UUID] = set()
        for path in write_paths:
            parent = self._get_dir(path.parent)
            if parent is None:
                continue  # the parent directory is gone, so nothing under it is ours to lock
            # a caller's path can name a hosted catalog; `dirs` holds only the local form
            local_path = Path.from_components(path.components)
            # the parent is locked even when the path names nothing: a free name is a slot a concurrent creator
            # could take, and the parent's `dirs` row is what holds it
            dirs.setdefault(parent.id, local_path.parent)

            # the path names a directory, a table, or nothing at all; probe dir before table, as _get_dir_entry() does
            if path.is_root:
                dir_id = parent.id  # the root is its own parent, so `parent` is already the directory in question
            else:
                subdir_q = sql.select(schema.Dir.id).where(
                    schema.Dir.parent_id == parent.id, schema.Dir.md['name'].astext == path.name
                )
                dir_id = conn.execute(subdir_q).scalars().one_or_none()
            if dir_id is not None:
                if lock_path_subtree:
                    subtree_dirs, subtree_tbl_ids = list_dirs(dir_id, local_path)
                    dirs.update(subtree_dirs)
                    write_path_tbl_ids |= subtree_tbl_ids
                else:
                    # only this directory's own record changes; its contents are untouched
                    dirs.setdefault(dir_id, local_path)
                continue

            tbl_q = sql.select(schema.Table.id).where(self._active_tbl_clause(dir_id=parent.id, tbl_name=path.name))
            tbl_id = conn.execute(tbl_q).scalars().one_or_none()
            if tbl_id is not None:
                write_path_tbl_ids.add(tbl_id)
        for tbl_id in write_path_tbl_ids:
            claim(tbl_id, write_op_class)

        pending.update(roles)
        read_pending()

        if lock_mutable_tree:
            list_transitive_views(
                set(self._mutable_write_tbl_ids(write_tvps, write_tbl_keys)) & seen, mutable_only=True
            )
            read_pending()

        if lock_path_subtree and len(write_path_tbl_ids) > 0:
            # a table is the parent of its views in the subtree the path names, so they come along whether or not
            # the operation asked for a mutable tree -- and snapshots with them, since a forced drop reaches those
            path_closure = list_transitive_views(write_path_tbl_ids & seen, mutable_only=False)
            read_pending()
            for tbl_id in write_path_tbl_ids:
                # dropping a mutable view bumps its base's view_sn, which is a metadata write to the base
                base_id = mutable_bases.get(tbl_id)
                if base_id is not None:
                    claim(base_id, write_op_class)
            for tbl_id in path_closure:
                # a view can live in a different directory than its base, so the directories are collected from the
                # tables rather than from the subtree walk
                dir_id = tbl_dirs.get(tbl_id)
                if dir_id is not None:
                    dirs.setdefault(dir_id, self.get_dir_path(dir_id))

        targets = [
            LockTarget(
                store_tbl_name=name,
                mode=_tbl_lock_mode(roles[tbl_id], tbl_id in versioned),
                is_data_versioned=tbl_id in versioned,
            )
            for tbl_id, name in names.items()
        ]
        return self._make_lock_set(targets, [dir_id for dir_id, _ in sorted(dirs.items(), key=lambda item: item[1])])

    def _lock_set_from_cache(
        self,
        *,
        write_op_class: _TblOpClass,
        read_tvps: Collection[TableVersionPath] = (),
        read_tbl_keys: Collection[TableVersionKey] = (),
        write_tvps: Collection[TableVersionPath] = (),
        write_tbl_keys: Collection[TableVersionKey] = (),
        lock_mutable_tree: bool = False,
    ) -> LockSet | None:
        """The lock targets that a transaction with the given read and write targets must lock, sorted in acquisition
        order, each with the mode it is to be locked in.

        A read locks its path; a write additionally locks its mutable tree when the write propagates there.
        write_op_class distinguishes a data write from a schema change, which take different modes; FINALIZE builds
        its own, narrower lock set.

        Each store table is locked once per transaction, in the strongest mode that any of its roles needs.

        Lock set is built using cached metadata only, so it's not guaranteed to be up to date with the store. If
        the cached state is not sufficient to build the lock set, None is returned.

        Takes no write paths: resolving one is a store read, so _resolve_lock_set() sends a transaction that has
        one straight to _lock_set_from_store()."""
        targets: dict[str, LockTarget] = {}

        def add(new: Sequence[LockTarget]) -> None:
            for target in new:
                held = targets.get(target.store_tbl_name)
                if held is None or target.mode.is_at_least(held.mode):
                    targets[target.store_tbl_name] = target

        for tvp in read_tvps:
            path_targets = self._path_lock_targets_from_cache(tvp.tbl_keys, _TblOpClass.DATA_READ)
            if path_targets is None:
                return None
            add(path_targets)
        for tvp in write_tvps:
            path_targets = self._path_lock_targets_from_cache(tvp.tbl_keys, write_op_class)
            if path_targets is None:
                return None
            add(path_targets)
        for key in read_tbl_keys:
            ancestors_targets = self._ancestors_lock_targets_from_cache(key, _TblOpClass.DATA_READ)
            if ancestors_targets is None:
                return None
            add(ancestors_targets)
        for key in write_tbl_keys:
            ancestors_targets = self._ancestors_lock_targets_from_cache(key, write_op_class)
            if ancestors_targets is None:
                return None
            add(ancestors_targets)
        if lock_mutable_tree:
            for write_tbl_id in self._mutable_write_tbl_ids(write_tvps, write_tbl_keys):
                tree_targets = self._mutable_tree_lock_targets_from_cache(write_tbl_id, write_op_class)
                if tree_targets is None:
                    return None
                add(tree_targets)
        return self._make_lock_set(targets.values())

    def _resolve_lock_set(
        self,
        *,
        op_class: _TblOpClass,
        read_tvps: Collection[TableVersionPath],
        read_tbl_keys: Collection[TableVersionKey],
        write_tvps: Collection[TableVersionPath],
        write_tbl_keys: Collection[TableVersionKey],
        lock_mutable_tree: bool,
        write_paths: Collection[Path],
        lock_path_subtree: bool,
    ) -> LockSet:
        """The lock set for a transaction with these targets, warming up the metadata cache if it isn't sufficient.

        Runs before the transaction opens, which is what keeps the store read out of it. _lock_set() issues no
        statements, so nothing here pins a snapshot that the locks would then be acquired behind.

        A metadata-only transaction reads catalog metadata and no table data, so it locks nothing: it may observe a
        schema change in progress, which is preferable to `list_tables()` or `describe()` blocking or failing.
        """
        assert not get_runtime().in_xact
        if op_class is _TblOpClass.MD_READ:
            return LockSet()
        lock_set: LockSet | None = None
        if len(write_paths) == 0:
            # a write path is resolved by reading the store: nothing caches what a path names, so a transaction
            # that has one skips the cache entirely
            lock_set = self._lock_set_from_cache(
                write_op_class=op_class,
                read_tvps=read_tvps,
                read_tbl_keys=read_tbl_keys,
                write_tvps=write_tvps,
                write_tbl_keys=write_tbl_keys,
                lock_mutable_tree=lock_mutable_tree,
            )
        if lock_set is None:
            # the cache cannot answer; read the shape of the lock set from the store instead
            with self.begin_read_md_xact():
                lock_set = self._lock_set_from_store(
                    write_op_class=op_class,
                    read_tvps=read_tvps,
                    read_tbl_keys=read_tbl_keys,
                    write_tvps=write_tvps,
                    write_tbl_keys=write_tbl_keys,
                    lock_mutable_tree=lock_mutable_tree,
                    write_paths=write_paths,
                    lock_path_subtree=lock_path_subtree,
                )
        return lock_set

    def _invalidate_lock_set(
        self,
        read_tvps: Collection[TableVersionPath],
        read_tbl_keys: Collection[TableVersionKey],
        write_tvps: Collection[TableVersionPath],
        write_tbl_keys: Collection[TableVersionKey],
    ) -> None:
        """Clear the metadata a lock set was derived from, so that the next attempt rebuilds it from the store.

        Counted rather than silent: a workload that reshapes a tree between attempts can restart repeatedly, and
        that should read as a counter rather than as a hang.
        """
        telemetry.emit('pixeltable.catalog.lock_set_restart')
        _logger.debug('rebuilding the lock set')
        for key in (*(k for tvp in (*read_tvps, *write_tvps) for k in tvp.tbl_keys), *read_tbl_keys, *write_tbl_keys):
            self._clear_tv_cache(key)

    def _validate_lock_set(
        self,
        *,
        write_op_class: _TblOpClass = _TblOpClass.DATA_WRITE,
        write_tvps: Collection[TableVersionPath] = (),
        write_tbl_keys: Collection[TableVersionKey] = (),
        lock_mutable_tree: bool = True,
        write_paths: Collection[Path] = (),
        lock_path_subtree: bool = False,
    ) -> None:
        """Checks that the locks this transaction holds cover what the current metadata says the operation touches.

        Must run after the metadata cache has been validated against the store, so that the trees it compares
        are the ones the store describes.

        Two things can have been guessed wrong: the mutable tree of a write target, and what a write path names.
        A table's ancestry is immutable, so a lock set built from ancestry alone needs no check.

        Raises:
            StaleLockSetError: the locks do not cover it.
        """

        def validate_targets_locked(targets: Collection[LockTarget]) -> None:
            for target in targets:
                held = self._locks_held.get(target.store_tbl_name)
                if held is None or not held.is_at_least(target.mode):
                    _logger.debug(f'lock set mismatch: {target.store_tbl_name} is locked in {held}, not {target.mode}')
                    raise _StaleLockSetError

        if lock_mutable_tree:
            for write_tbl_id in self._mutable_write_tbl_ids(write_tvps, write_tbl_keys):
                mutable_tree = self._mutable_tree_lock_targets_from_cache(write_tbl_id, write_op_class)
                if mutable_tree is None:
                    _logger.debug(f'lock set mismatch: mutable tree of {write_tbl_id} is not fully cached')
                    raise _StaleLockSetError
                validate_targets_locked(mutable_tree)

        if len(write_paths) > 0:
            # re-resolve the paths against the metadata the locks made current, and check the result is covered.
            # The cache cannot answer this, so it is the same store read as before, now under the locks.
            lock_set_from_store = self._lock_set_from_store(
                write_op_class=write_op_class,
                read_tvps=(),
                read_tbl_keys=(),
                write_tvps=write_tvps,
                write_tbl_keys=write_tbl_keys,
                lock_mutable_tree=lock_mutable_tree,
                write_paths=write_paths,
                lock_path_subtree=lock_path_subtree,
            )
            validate_targets_locked(lock_set_from_store.tbl_targets)
            for dir_id in lock_set_from_store.dir_ids:
                if dir_id not in self._dir_locks_held:
                    _logger.debug(f'lock set mismatch: directory {dir_id} is not locked')
                    raise _StaleLockSetError

    @classmethod
    def _mutable_write_tbl_ids(
        cls, write_tvps: Collection[TableVersionPath], write_tbl_keys: Collection[TableVersionKey]
    ) -> list[UUID]:
        """The write targets that have a mutable tree to lock.

        A snapshot has none, and asking for one would look up a live version it does not have. A pinned effective
        version is what distinguishes the two, and unlike is_mutable() it is answerable without reading any md --
        which is the whole constraint a lock set is computed under. The operation itself still fails later, with
        the error that says a snapshot cannot be written.
        """
        ids = [tvp.tbl_id for tvp in write_tvps if tvp.effective_version() is None]
        ids.extend(k.tbl_id for k in write_tbl_keys if k.effective_version is None)
        return ids

    def _is_locked(self, store_tbl_name: str, mode: _TblLockMode) -> bool:
        """True if this transaction holds a lock on the given table with the given or stronger lock mode."""
        assert get_runtime().in_xact
        held = self._locks_held.get(store_tbl_name)
        return held is not None and held.is_at_least(mode)

    def _is_dir_locked(self, dir_id: UUID) -> bool:
        """True if this transaction X-locked the given Dir record."""
        assert get_runtime().in_xact
        return dir_id in self._dir_locks_held

    def _assert_md_write_locked(
        self, tbl_id: UUID, *, is_insert: bool, is_pure_snapshot: bool, dir_id: UUID | None
    ) -> None:
        """Assert that this transaction holds the lock that protects a metadata write to tbl_id.

        A table that has a store table is protected by a self-conflicting mode on it: ACCESS EXCLUSIVE for a schema
        change, and EXCLUSIVE for the version bump a data write on a data-versioned table writes. Either one
        excludes every other md write for the length of the transaction, which is what the record needs. An
        operational data write takes ROW EXCLUSIVE and writes no md.

        A pure snapshot has no store table, so no store table lock can cover it. The parent Dir record's X-lock
        protects it instead: a pure snapshot is a name in a directory and nothing more, and its md record is only
        ever inserted or deleted, always by a path that locks that directory first.
        """
        if is_insert or is_pure_snapshot:
            # Either this record is being inserted, which nobody can contend for -- the id is unpublished and the
            # name is held by the parent Dir's X-lock -- or it is a pure snapshot, which has no store table for a
            # lock to cover and is a name in a directory and nothing else. The dir lock protects both.
            assert dir_id is not None, tbl_id
            assert self._is_dir_locked(dir_id), (tbl_id, dir_id, self._dir_locks_held)
            return
        assert self._has_store_tbl_lock(tbl_id, _TblLockMode.EXCLUSIVE), (tbl_id, self._locks_held)

    def assert_rows_write_locked(self, tv: TableVersion) -> None:
        """Verifies that the transaction holds a lock appropriate for writing rows to this table."""
        mode = _tbl_lock_mode(_TblOpClass.DATA_WRITE, tv.is_data_versioned)
        store_tbl_name = tv.store_tbl._storage_name()
        assert self._is_locked(store_tbl_name, mode), (store_tbl_name, mode, self._locks_held)

    def check_rows_read_locked(self, tv: TableVersion) -> None:
        """Verifies that the transaction holds a lock appropriate for reading rows from the table.

        TODO(PXT-1343): once fixed, this warning should become an assertion.
        """
        read_lock_mode = _tbl_lock_mode(_TblOpClass.DATA_READ, tv.is_data_versioned)
        store_tbl_name = tv.store_tbl._storage_name()
        if not self._is_locked(store_tbl_name, read_lock_mode):
            warnings.warn(
                f'Table {tv.versioned_name} ({store_tbl_name}) was not locked for read at the transaction start',
                excs.PixeltableWarning,
                stacklevel=2,
            )

    def _lock_tables(self, targets: Sequence[LockTarget], *, blocking: bool) -> None:
        """Acquire the given targets' store tables, in the given order.

        A LOCK TABLE statement carries a single mode, so a lock set that spans several takes more than one. They are
        the maximal *runs* of adjacent targets sharing a mode -- never groups gathered from across the set, which
        would reorder acquisition and reintroduce the deadlocks the global sort rules out. Postgres acquires left to
        right within a statement, so run by run is still one global order.

        Raises:
            LockNotAvailableError: blocking=False and a lock is not readily available
            StaleLockSetError: a store table named in the statement does not exist.
        """
        assert get_runtime().in_xact
        names = [t.store_tbl_name for t in targets]
        # Every transaction must use the same relative order of lock acquisition in order to avoid deadlocks
        assert names == sorted(names), names
        # a store table is locked once per transaction, no upgrades
        assert all(name not in self._locks_held for name in names), (names, self._locks_held)
        for mode, run in itertools.groupby(targets, key=lambda t: t.mode):
            self._lock_run([t.store_tbl_name for t in run], mode, blocking=blocking)

    def _lock_run(self, store_tbl_names: list[str], mode: _TblLockMode, *, blocking: bool) -> None:
        """Acquire `mode` on all named store tables in one statement, in the given order.

        Uses a LOCK TABLE statement which, if executed before any reads under REPEATABLE READ, does not establish a read
        snapshot for the transaction.
        """
        assert len(store_tbl_names) > 0
        nowait_clause = '' if blocking else 'NOWAIT'
        stmt = f'LOCK TABLE {", ".join(store_tbl_names)} IN {mode.sql_name} MODE {nowait_clause}'
        try:
            get_runtime().conn.execute(sql.text(stmt))
        except sql_exc.DBAPIError as e:
            _logger.debug(f'{stmt} failed: {e}')
            if isinstance(e.orig, psycopg.errors.LockNotAvailable):
                assert not blocking
                raise _LockNotAvailableError from e
            if isinstance(e.orig, psycopg.errors.UndefinedTable):
                raise _StaleLockSetError from e
            raise
        for store_tbl_name in store_tbl_names:
            self._locks_held[store_tbl_name] = mode

    def _acquire_locks(
        self,
        op_class: _TblOpClass,
        read_tvps: Collection[TableVersionPath],
        read_tbl_keys: Collection[TableVersionKey],
        write_tvps: Collection[TableVersionPath],
        write_tbl_keys: Collection[TableVersionKey],
        lock_set: LockSet,
        blocking: bool,
        lock_mutable_tree: bool = False,
        finalize_pending_ops: bool = True,
    ) -> None:
        """
        Acquires the locks of lock_set, then refreshes the metadata cache for every read and write target -- and,
        when lock_mutable_tree is True, for each write target's mutable tree.

        The store table locks come first, and must stay first: LOCK TABLE is snapshot-exempt, where everything below
        it here is a statement that pins the transaction's snapshot. A transaction that waited for such a lock
        therefore still reads current metadata afterwards. The `dirs` row locks are the statements this argument
        rules out going first: a transaction that waited for one wakes with a snapshot from before the wait.

        The order of the rest matters too: TVPs are processed before keys in both groups so that ancestor-first
        validation (write_tvps -> write_tbl_keys -> read_tvps -> read_tbl_keys) is established before any unordered
        pass runs.
        """
        # a transaction that locks no store table has nothing to observe at either point, and firing there would let
        # the lock-set resolution consume a fault armed for the operation that follows it in the same thread
        if len(lock_set.tbl_targets) > 0:
            fault_injection.process_fault(FaultLocation.CATALOG_BEFORE_TBL_LOCK, op_class=op_class)
            self._lock_tables(lock_set.tbl_targets, blocking=blocking)
            fault_injection.process_fault(FaultLocation.CATALOG_AFTER_TBL_LOCK, op_class=op_class)
        for dir_id in lock_set.dir_ids:
            self._acquire_dir_xlock(dir_id=dir_id)

        # write targets already refreshed, including the tree members reached through one, so that a target reached
        # twice is read once
        refreshed: set[UUID] = set()
        for tvp in write_tvps:
            self._refresh_path_cache(
                tbl=tvp,
                for_write=True,
                mutable_tree=lock_mutable_tree,
                check_pending_ops=finalize_pending_ops,
                refreshed=refreshed,
            )
        for write_key in write_tbl_keys:
            # a write target is always a mutable table, so its key carries no effective version
            assert write_key.effective_version is None, write_key
            self._refresh_tbl_cache(
                key=write_key,
                mutable_tree=lock_mutable_tree,
                check_pending_ops=finalize_pending_ops,
                refreshed=refreshed,
            )
        for tvp in read_tvps:
            self._refresh_path_cache(tbl=tvp, for_write=False, check_pending_ops=finalize_pending_ops)
        for read_key in read_tbl_keys:
            self._refresh_tbl_cache(key=read_key, check_pending_ops=finalize_pending_ops)

    def _evict_caches(self) -> None:
        # Evict LRU _tbls entries
        evicted_tbls: list[TableVersionKey] = []
        while len(self._tbls) > _MAX_TBL_CACHE_SIZE:
            key, _ = self._tbls.popitem(last=False)
            evicted_tbls.append(key)

        # Evict LRU _tbl_versions entries. Reset is_validated to False preemptively in case an instance escapes.
        evicted_tvs: list[TableVersionKey] = []
        while len(self._tbl_versions) > _MAX_TBL_CACHE_SIZE:
            key, tv = self._tbl_versions.popitem(last=False)
            tv.is_validated = False
            evicted_tvs.append(key)

        if evicted_tbls:
            _logger.info(f'Evicted {len(evicted_tbls)} LRU table(s) from cache: {evicted_tbls}')
        if evicted_tvs:
            _logger.info(f'Evicted {len(evicted_tvs)} LRU table version(s) from cache: {evicted_tvs}')

    def _get_single_tbl(
        self,
        read_tvps: Collection[TableVersionPath],
        read_tbl_keys: Collection[TableVersionKey],
        write_tvps: Collection[TableVersionPath],
        write_tbl_keys: Collection[TableVersionKey],
    ) -> tuple[TableVersionHandle | None, UUID | None]:
        """Return (tbl, None) or (None, tbl_id) iff the transaction touches exactly one table; else (None, None)."""
        total = len(write_tvps) + len(read_tvps) + len(read_tbl_keys) + len(write_tbl_keys)
        if total != 1:
            return None, None
        if write_tvps or read_tvps:
            return next(iter(write_tvps or read_tvps)).tbl_version, None
        return None, next(iter(read_tbl_keys or write_tbl_keys)).tbl_id

    def convert_sql_exc(
        self,
        e: sql_exc.StatementError,
        tbl_id: UUID | None = None,
        tbl: TableVersionHandle | None = None,
        convert_db_excs: bool = True,
    ) -> None:
        # we got some db error during the actual operation (not just while trying to get locks on the metadata
        # records); we convert these into pxt.Error exceptions if appropriate

        # we always convert UndefinedTable exceptions (they can't be retried)
        if isinstance(e.orig, psycopg.errors.UndefinedTable):
            # the table got dropped in the middle of the operation
            tbl_name = None
            if tbl is not None:
                tbl_name = tbl.get().name
            _logger.debug(f'Exception: undefined table {(tbl_name or "<unknown>")!r}: Caught {type(e.orig)}: {e!r}')
            raise excs.table_was_dropped(tbl_name) from None
        elif (
            isinstance(
                e.orig,
                (
                    psycopg.errors.SerializationFailure,  # serialization error despite getting x-locks
                    psycopg.errors.InFailedSqlTransaction,  # can happen after tx fails for another reason
                    psycopg.errors.DuplicateColumn,  # if a different process added a column concurrently
                    psycopg.errors.DeadlockDetected,  # locking protocol contention
                ),
            )
            and convert_db_excs
        ):
            msg: str
            if tbl is not None:
                msg = f'{tbl.get().name} ({tbl.id})'
            elif tbl_id is not None:
                msg = f'{tbl_id}'
            else:
                msg = ''
            _logger.debug(f'Exception: {e.orig.__class__}: {msg} ({e})')
            # Suppress the underlying SQL exception unless DEBUG is enabled
            raise_from = e if _logger.isEnabledFor(logging.DEBUG) else None
            if isinstance(e.orig, psycopg.errors.DuplicateColumn):
                # TODO: extend message with the name of the schema column (not the store column)
                raise excs.AlreadyExistsError(excs.ErrorCode.COLUMN_ALREADY_EXISTS, 'Duplicate column') from raise_from
            raise excs.ConcurrencyError(
                excs.ErrorCode.SERIALIZATION_FAILURE,
                'That Pixeltable operation could not be completed because it conflicted with another '
                'operation that was run on a different process.\n'
                'Please re-run the operation.',
            ) from raise_from

    def _refresh_path_cache(
        self,
        *,
        tbl: TableVersionPath,
        for_write: bool = False,
        mutable_tree: bool = False,
        check_pending_ops: bool = True,
        refreshed: set[UUID] | None = None,
    ) -> None:
        """
        Refresh the cached TableVersions along a path, and check what this operation writes for pending ops:
        - the ancestors, which an insert needs too, for computed columns that reference the base tables
        - the leaf, through _refresh_tbl_cache() when for_write, and its mutable tree when mutable_tree

        Raises Error if tbl doesn't exist.
        """
        path_handles = tbl.get_tbl_versions()
        read_handles = path_handles[:0:-1] if for_write else path_handles[::-1]
        for handle in read_handles:
            # update cache
            _ = self._get_tbl_version(handle.key)
        if for_write:
            self._refresh_tbl_cache(
                key=TableVersionKey(tbl.tbl_id, None),
                mutable_tree=mutable_tree,
                check_pending_ops=check_pending_ops,
                refreshed=refreshed,
            )

    def _has_store_tbl_lock(self, tbl_id: UUID, mode: _TblLockMode) -> bool:
        """True if this transaction holds `mode` or stronger on the table's store table, under either name."""
        return any(self._is_locked(_store_tbl_name(tbl_id, is_view=is_view), mode) for is_view in (False, True))

    def _refresh_tbl_cache(
        self,
        *,
        key: TableVersionKey,
        mutable_tree: bool = False,
        check_pending_ops: bool = True,
        refreshed: set[UUID] | None = None,
    ) -> None:
        """
        Refresh a target's cached metadata, and check it for pending ops.

        If mutable_tree, does the same for the target's transitive mutable views, which a write propagates to.
        refreshed, if given, accumulates the ids visited and skips those already there.

        Takes no lock. The target's store table was locked before any statement pinned this transaction's snapshot,
        so what this reads is current as of that acquisition, and no concurrent operation can change it before we
        commit.
        """
        if refreshed is not None:
            if key.tbl_id in refreshed:
                return
            refreshed.add(key.tbl_id)
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Table).where(schema.Table.id == key.tbl_id)).one_or_none()
        if row is None:
            raise excs.table_was_dropped(key.tbl_id)
        tbl_md = schema.md_from_dict(schema.TableMd, row.md)

        if check_pending_ops:
            pending_ops_q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == row.id)
            has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
            if has_pending_ops:
                raise PendingTableOpsError(row.id)

        # check_pending_ops == False means this table's pending ops are in the process of being finalized, so its
        # metadata is still in flux; loading it would also pull in the tables its value exprs reference, which may
        # have pending ops of their own.
        tv: TableVersion | None = None
        if check_pending_ops and not tbl_md.is_pure_snapshot:
            # a caller that names a table without a version -- a write target, or a tree member reached below --
            # leaves it to the md: a snapshot is loaded at the version it pins, a mutable table at None
            load_key = (
                key
                if key.effective_version is not None or not tbl_md.is_snapshot
                else TableVersionKey(key.tbl_id, tbl_md.current_version)
            )
            tv = self._get_tbl_version(load_key)

        if mutable_tree and tbl_md.is_mutable:
            assert tv is not None, key
            for view in tv.mutable_views:
                self._refresh_tbl_cache(
                    key=TableVersionKey(view.id, None),
                    mutable_tree=True,
                    check_pending_ops=check_pending_ops,
                    refreshed=refreshed,
                )

    def _roll_forward(self) -> None:
        """Finalize pending ops for all tables in self._roll_forward_ids, in insertion order.

        The order is important for multi-table drop: views must be dropped before bases.
        """
        for tbl_id in self._roll_forward_ids:
            exc = self._finalize_pending_ops(tbl_id)
            if exc is not None:
                raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'Table operation was aborted with\n{exc!s}') from exc

    def _finalize_pending_ops(self, tbl_id: UUID) -> Exception | None:
        """
        Finalizes all pending ops for the given table, and clears the table version cache for that table.

        Each of its transactions waits for its ACCESS EXCLUSIVE lock, whether this is the owner rolling its own
        schema change forward or a helper doing it on behalf of an operation that ran into the pending ops.
        An owner that has actually died leaves the lock free, so the next helper takes over.

        During tbl_state == ROLLFORWARD (error-free path):
        - executes all remaining pending ops in order op_sn and updates their status to COMPLETED
        - when done, deletes all table ops and resets tbl_state to LIVE
        - if it encounters an exception:
          - if the statement can be aborted, switches tbl_state to ROLLBACK and continues with the rollback protocol
          - otherwise continues with rollforward

        During tbl_state == ROLLBACK (error path):
        - undoes ops in reverse order of op_sn and updates their status to ABORTED
        - this process starts with the first pending op, because it could have been partially executed
        - when done, deletes all table ops and resets tbl_state to LIVE

        If an exception occurred during finalization, that exception is returned. PendingOpsErrors encountered during
        finalization are dealt with recursively.
        """
        num_retries = 0
        is_rollback = False
        tbl_md: schema.TableMd | None = None
        tbl_version: int | None = None
        op: TableOp | None = None
        exc: Exception | None = None
        assert not get_runtime().in_xact, 'Cannot finalize pending ops inside a transaction'
        # If set, a pending table op update rolled over from the previous loop iteration. It saves us 1 transaction per
        # non-transactional table op.
        # Contains: (op, new_op_status, is_final_op)
        rollover_op_update: tuple[TableOp, OpStatus, bool] | None = None

        tbl_q = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)

        while True:
            try:
                with self._begin_finalize_xact(tbl_id) as conn, self._allow_tbl_md_read():
                    # determine table status
                    row = conn.execute(tbl_q).one_or_none()
                    if row is None:
                        _logger.debug(f'Finalize pending ops({tbl_id}): table not found, exiting')
                        return None
                    tbl_md = schema.md_from_dict(schema.TableMd, row.md)
                    _logger.debug(f'Finalize pending ops({tbl_id}): table state: {tbl_md.tbl_state}')
                    if tbl_md.tbl_state == schema.TableState.LIVE:
                        # nothing left to do
                        return None
                    assert tbl_md.tbl_state in (schema.TableState.ROLLFORWARD, schema.TableState.ROLLBACK)
                    is_rollback = tbl_md.tbl_state == schema.TableState.ROLLBACK

                    if rollover_op_update is not None:
                        if self._set_pending_op_status(
                            tbl_id,
                            op=rollover_op_update[0],
                            new_status=rollover_op_update[1],
                            is_final_op=rollover_op_update[2],
                        ):
                            return exc

                        rollover_op_update = None

                    ops = self._read_pending_table_ops(tbl_id)
                    assert len(ops) > 0

                    # determine next op to execute/undo
                    if is_rollback:
                        # last aborted: in chronological order (ie, the one with the lowest op_sn)
                        last_aborted_op = next((op for op in ops if op.status == OpStatus.ABORTED), None)
                        if last_aborted_op is None:
                            # we haven't aborted anything yet and need to start with the first pending op
                            op = next(op for op in ops if op.status == OpStatus.PENDING)
                        else:
                            # we continue aborting completed ops in reverse order;
                            # we haven't aborted the final op yet, otherwise we wouldn't still be in ROLLBACK state
                            assert last_aborted_op.op_sn > 0
                            # undo the op preceding the last aborted one
                            op = ops[last_aborted_op.op_sn - 1]
                        is_final_op = op.op_sn == 0
                    else:
                        # rollforward: we execute the first pending op
                        op = next(op for op in ops if op.status == OpStatus.PENDING)
                        is_final_op = op.op_sn == op.num_ops - 1

                    _logger.debug(
                        f'Finalize pending ops({tbl_id}): finalizing op {op!s}; is_rollback={is_rollback}, '
                        f'is_final_op={is_final_op}, transactional={op.needs_xact}'
                    )

                    tbl_version = tbl_md.current_version if tbl_md.is_snapshot else None
                    tv = (
                        self._get_tbl_version(TableVersionKey(tbl_id, tbl_version), check_pending_ops=False)
                        if op.needs_tv
                        else None
                    )
                    new_op_status = OpStatus.ABORTED if is_rollback else OpStatus.COMPLETED
                    if op.needs_xact:
                        # Mark TableVersion as modified before it is actually modified to make sure that cache is
                        # cleared properly if an error occurs during op execution.
                        if tv is not None:
                            self.mark_modified_tv(tv.handle)
                        if is_rollback:
                            op.undo(tv)
                        else:
                            op.exec(tv)

                        _logger.debug(f'Finalize pending ops({tbl_id}): op {op!s} done, updating status')
                        if self._set_pending_op_status(tbl_id, op, new_op_status, is_final_op=is_final_op):
                            return exc
                        continue

                # this op runs outside of a transaction
                fault_injection.process_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT)
                if is_rollback:
                    op.undo(tv)
                else:
                    op.exec(tv)
                # no need to invalidate tv here: all operations that modify metadata (cached in tv) are executed
                # inside a transaction and therefore wouldn't end up here
                rollover_op_update = (op, new_op_status, is_final_op)

            except AssertionError as e:
                _logger.error(f'Finalize pending ops({tbl_id}): assertion error: {e}', exc_info=True)
                # we need to make sure not to swallow asserts
                raise

            except excs.PixeltableWarning:
                # Tests promote PixeltableWarnings to an error. Re-raise them to avoid getting stuck in a finalize loop.
                raise

            except excs.ConcurrencyError as e:
                if e.error_code is not excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS:
                    raise
                # Not the finalization's own lock, which it always waits for; this is an op whose work ran into a
                # fail-fast operation elsewhere. Pass it to the caller rather than letting the handler below treat
                # it as an op failure, which would abort a statement that has nothing wrong with it.
                _logger.debug(f'Finalize pending ops({tbl_id}): op reported a schema change in progress')
                raise

            except PendingTableOpsError as e:
                # Loading metadata for tbl_id transitively required another table that has its own pending ops:
                # - the xact opened above is already rolled back by exiting the with-block via exception
                # - finalize the dependency first, then continue with this table
                # - recursion is bounded by the dependency DAG of stored expressions
                # - PendingTableOpsError does not propagate outside
                other_exc = self._finalize_pending_ops(e.tbl_id)
                if other_exc is not None:
                    return other_exc
                continue

            except sql_exc.DBAPIError as e:
                # TODO: why are we still seeing these here, instead of them getting taken care of by the retry
                # logic of begin_xact()?
                # InFailedSqlTransaction is specific to finalizing a multi-op statement: an earlier op in the
                # same transaction failed and poisoned it, so the current op needs a fresh transaction.
                if _is_retryable_exc(e) or isinstance(e.orig, psycopg.errors.InFailedSqlTransaction):
                    num_retries += 1
                    _logger.debug(f'Finalize pending ops({tbl_id}): retriable error: {e.orig} of type {type(e.orig)}')
                    log_msg: str
                    if op is not None:
                        log_msg = f'finalize_pending_ops(): retrying ({num_retries}) op {op!s} after {type(e.orig)}'
                    else:
                        log_msg = f'finalize_pending_ops(): retrying ({num_retries}) after {type(e.orig)}'
                    Env.get().console_logger.debug(log_msg)
                    time.sleep(random.uniform(0.1, 0.5))
                    continue
                else:
                    _logger.error(
                        f'Finalize pending ops({tbl_id}): non-retriable error {e} of type {type(e)}', exc_info=True
                    )
                    # TODO: what to do with this?
                    raise

            except Exception as e:
                if excs.is_table_not_found_error(e):
                    _logger.debug(f'Finalize pending ops({tbl_id}): table not found, exiting')
                    # nothing to do
                    return None

                if not is_rollback and tbl_md is not None and tbl_md.pending_stmt.can_abort():
                    _logger.error(
                        f'Finalize pending ops({tbl_id}): aborting statement due to error: {e} of type {type(e)}',
                        exc_info=True,
                    )
                    # we got an error for the last op and can abort this statement: switch to rollback mode
                    exc = e
                    with self._begin_finalize_xact(tbl_id) as conn:
                        stmt = (
                            sql.update(schema.Table)
                            .where(schema.Table.id == tbl_id)
                            .values(md=schema.Table.md.op('||')({'tbl_state': schema.TableState.ROLLBACK.value}))
                        )
                        status = conn.execute(stmt)
                        assert status.rowcount == 1
                else:
                    # log this error but keep going
                    _logger.error(
                        f'Finalize pending ops({tbl_id}): caught error: {e} of type {type(e)} but continuing',
                        exc_info=True,
                    )
            finally:
                self._clear_tv_cache(TableVersionKey(tbl_id, None))

            num_retries = 0

    def _pending_table_ops_update_stmt(
        self, tbl_id: UUID, op: TableOp, new_status: OpStatus, *, is_final_op: bool
    ) -> sql.UpdateBase:
        """
        Generates a PendingTableOp (pendingtableops) update statement for the given op.

        If this op is final, deletes the ops. Otherwise simply updates the op's status.
        """
        pending_ops_stmt: sql.UpdateBase
        if is_final_op:
            _logger.info(f'Finalize pending ops({tbl_id}): deleting pendingtableops with tbl_version={op.tbl_version}')
            pending_ops_stmt = sql.delete(schema.PendingTableOp)
        else:
            _logger.info(
                f'Finalize pending ops({tbl_id}): updating pendingtableops with op_sn={op.op_sn}, '
                f'tbl_version={op.tbl_version}; new status: {new_status}'
            )
            pending_ops_stmt = (
                sql.update(schema.PendingTableOp)
                .where(schema.PendingTableOp.op_sn == op.op_sn)
                .values(op=schema.PendingTableOp.op.op('||')({'status': new_status.value}))
            )
        pending_ops_stmt = pending_ops_stmt.where(schema.PendingTableOp.tbl_id == tbl_id)

        # Add a table version condition. This is necessary to avoid a scenario in which a delayed pending ops finalizer
        # corrupts the table by updating pending ops associated with a future schema change, not the one that it
        # finalized. This issue is described in more detail in PXT-1130.
        # Note: all schema changes except create table increment table version. Create table is not a problem because
        # no other schema change can precede it.
        # Note: the only known gap that this safeguard does not cover is table revert. The way table revert is
        # implemented, it decrements table and schema versions. Which means that if we do a schema change, then revert,
        # then a schema change again, those two schema changes will share a table version value, therefore, with unlucky
        # timing, the pendingtableops table can still get corrupted. The right fix is for that is not here, it is to
        # reimplement revert by advancing schema/data versions, not decrementing them.
        if op.tbl_version is None:
            # Legacy pendingtableop
            pending_ops_stmt = pending_ops_stmt.where(sql.not_(schema.PendingTableOp.op.has_key('tbl_version')))
        else:
            pending_ops_stmt = pending_ops_stmt.where(
                schema.PendingTableOp.op['tbl_version'].cast(sql.Integer) == op.tbl_version
            )
        return pending_ops_stmt

    def _set_pending_op_status(self, tbl_id: UUID, op: TableOp, new_status: OpStatus, *, is_final_op: bool) -> bool:
        """
        Updates the pending op status in the store. If is_final_op, sets table status to LIVE after additional checks.

        Note: is_final_op is a hint that this may have been the last pending table op. Due to possible concurrent schema
        changes, only the store can be the final authority on the state of the table.

        This function must be called inside a transaction holding ACCESS EXCLUSIVE on the table's store table.

        Returns True if no unresolved pending ops remain on the table, and the table's status was set to LIVE. False
        otherwise.
        """
        assert self._has_store_tbl_lock(tbl_id, _TblLockMode.ACCESS_EXCLUSIVE), tbl_id
        pending_ops_stmt = self._pending_table_ops_update_stmt(tbl_id, op, new_status, is_final_op=is_final_op)
        conn = get_runtime().conn
        rowcount = conn.execute(pending_ops_stmt).rowcount
        # Log a message if no pendingtableops rows were matched. DeleteTableMdOp is a special case because it
        # deletes all pendingtableops.
        if rowcount == 0 and not isinstance(op, DeleteTableMdOp):
            _logger.info(
                f'Finalize pending ops({tbl_id}): no PendingTableOp rows matched. Another process may have already '
                'resolved the same pending op concurrently.'
            )

        if not is_final_op:
            _logger.info(f'Finalize pending ops({tbl_id}): not final op, more pending ops to finalize')
            return False

        tbl_ops = self._read_pending_table_ops(tbl_id)
        if len(tbl_ops) > 0:
            _logger.info(f'Finalize pending ops({tbl_id}): more pending ops found')
            return False

        # No remaining pending table ops. Reset the table state.
        reset_tbl_state_stmt = (
            sql.update(schema.Table)
            .where(schema.Table.id == tbl_id)
            .values(md=schema.Table.md.op('||')({'tbl_state': schema.TableState.LIVE.value, 'pending_stmt': None}))
        )

        _logger.info(f'Finalize pending ops({tbl_id}): no more pending ops, resetting table state')
        rowcount = conn.execute(reset_tbl_state_stmt).rowcount
        if rowcount == 0 and not isinstance(op, DeleteTableMdOp):
            _logger.info(
                f'Finalize pending ops({tbl_id}): no Table rows matched. Another process may have deleted the table '
                'concurrently.'
            )

        return True

    def _read_pending_table_ops(self, tbl_id: UUID) -> list[TableOp]:
        """
        Returns the table's pending ops as TableOps, in order.

        Must be called inside a transaction holding ACCESS EXCLUSIVE on the table's store table, which is what
        excludes a concurrent finalization.
        """
        assert self._has_store_tbl_lock(tbl_id, _TblLockMode.ACCESS_EXCLUSIVE), tbl_id
        conn = get_runtime().conn
        q = (
            sql.select(schema.PendingTableOp)
            .where(schema.PendingTableOp.tbl_id == tbl_id)
            .order_by(schema.PendingTableOp.op_sn)
        )
        rows = conn.execute(q).fetchall()
        return [TableOp.from_dict(dict(row.op)) for row in rows]

    def _debug_str(self) -> str:
        tv_str = '\n'.join(str(k) for k in self._tbl_versions)
        tbl_str = '\n'.join(str(k) for k in self._tbls)
        return f'tbl_versions:\n{tv_str}\ntbls:\n{tbl_str}'

    def _get_mutable_tree(self, tbl_id: UUID) -> set[UUID]:
        """Returns ids of all tables that form the tree of mutable views starting at tbl_id; includes the root."""
        key = TableVersionKey(tbl_id, None)
        assert key in self._tbl_versions, f'{key} not in {self._tbl_versions.keys()}\n{self._debug_str()}'
        tv = self._get_tbl_version(key)
        result: set[UUID] = {tv.id}
        for view in tv.mutable_views:
            result.update(self._get_mutable_tree(view.id))
        return result

    def _compute_column_dependents(
        self, write_tvps: Collection[TableVersionPath], write_tbl_keys: Collection[TableVersionKey]
    ) -> None:
        """Populate self._column_dependents over the mutable trees of this transaction's write targets.

        Only called when the write targets were locked with their mutable trees, which is the set this walks. Those
        locks block schema updates, so the dependency graph is current until the end of the transaction.
        """
        assert self._column_dependents is None
        self._column_dependents = defaultdict(set)
        mutable_tbls: set[UUID] = set()
        for tbl_id in self._mutable_write_tbl_ids(write_tvps, write_tbl_keys):
            mutable_tbls |= self._get_mutable_tree(tbl_id)
        for tbl_id in mutable_tbls:
            assert tbl_id in self._column_dependencies, (
                f'{tbl_id} not in {self._column_dependencies.keys()}\n{self._debug_str()}'
            )
            for col, dependencies in self._column_dependencies[tbl_id].items():
                for dependency in dependencies:
                    if dependency.tbl_id not in mutable_tbls:
                        continue
                    dependents = self._column_dependents[dependency]
                    dependents.add(col)

    def record_column_dependencies(self, tbl_version: TableVersion) -> None:
        """Update self._column_dependencies. Only valid for mutable versions."""
        from pixeltable.exprs import ColumnRef, Expr

        assert tbl_version.is_mutable
        dependencies: dict[QColumnId, set[QColumnId]] = {}
        for col in tbl_version.cols_by_id.values():
            if col.value_expr_dict is None:
                continue
            dependencies[QColumnId(tbl_version.id, col.id)] = Expr.get_refd_column_ids(col.value_expr_dict)

        if tbl_version.is_component_view:
            iterator_arg_deps: set[QColumnId] = set()
            iterator_args = tbl_version.iterator_args_expr()
            if iterator_args is not None:
                for col_ref in iterator_args.subexprs(ColumnRef):
                    iterator_arg_deps.add(QColumnId(col_ref.col.tbl_handle.id, col_ref.col.id))
            if len(iterator_arg_deps) > 0:
                for col in tbl_version.iterator_columns():
                    dependencies[QColumnId(tbl_version.id, col.id)] = iterator_arg_deps

        self._column_dependencies[tbl_version.id] = dependencies

    def get_column_dependents(self, tbl_id: UUID, col_id: int) -> set[Column]:
        """Return all Columns that transitively depend on the given column."""
        assert self._column_dependents is not None
        dependents = self._column_dependents[QColumnId(tbl_id, col_id)]
        result: set[Column] = set()
        for dependent in dependents:
            tv = self._get_tbl_version(TableVersionKey(dependent.tbl_id, None))
            col = tv.cols_by_id[dependent.col_id]
            result.add(col)
        return result

    def _mutable_view_tvs(self, tbl_version: TableVersion) -> Iterator[TableVersion]:
        """Return the TableVersions of all transitive mutable views of the given table."""
        for view in tbl_version.mutable_views:
            view_tv = view.get()
            yield view_tv
            yield from self._mutable_view_tvs(view_tv)

    def _acquire_dir_xlock(
        self, *, parent_id: UUID | None = None, dir_id: UUID | None = None, dir_name: str | None = None
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
        result = get_runtime().conn.execute(q.returning(schema.Dir.id))
        self._dir_locks_held.update(row.id for row in result)

    def get_dir_path(self, dir_id: UUID) -> Path:
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

    def _table_error_counts(self) -> dict[UUID, int]:
        """Returns map from table id to the sum of num_excs across that table's versions."""
        md = schema.TableVersion.md
        update_status = md['update_status']
        row_count_excs = sql.func.coalesce(update_status['row_count_stats']['num_excs'].astext.cast(sql.Integer), 0)
        cascade_row_count_excs = sql.func.coalesce(
            update_status['cascade_row_count_stats']['num_excs'].astext.cast(sql.Integer), 0
        )
        errors = sql.func.coalesce(
            sql.cast(sql.func.sum(row_count_excs + cascade_row_count_excs), sql.BigInteger), 0
        ).label('errors')
        stmt = sql.select(schema.TableVersion.tbl_id, errors).group_by(schema.TableVersion.tbl_id)
        rows = get_runtime().conn.execute(stmt).all()
        return {r.tbl_id: r.errors for r in rows}

    @retry_read_md_loop()
    def get_dir_contents(
        self, dir_path: Path, recursive: bool = False, with_error_counts: bool = False
    ) -> dict[str, DirEntry]:
        dir = self._get_schema_object(dir_path, expected=Dir, raise_if_not_exists=True)
        error_counts = self._table_error_counts() if with_error_counts else None
        return self._get_dir_contents(dir._id, recursive=recursive, error_counts=error_counts)

    def _get_dir_contents(
        self, dir_id: UUID, recursive: bool = False, *, error_counts: dict[UUID, int] | None = None
    ) -> dict[str, DirEntry]:
        """Returns a dict mapping the entry names to DirEntry objects"""
        conn = get_runtime().conn
        result: dict[str, DirEntry] = {}

        q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
        rows = conn.execute(q).all()
        for row in rows:
            dir = schema.Dir(**row._mapping)
            dir_contents: dict[str, DirEntry] = {}
            if recursive:
                dir_contents = self._get_dir_contents(dir.id, recursive=True, error_counts=error_counts)
            result[dir.md['name']] = DirEntry(dir=dir, dir_entries=dir_contents, table=None)

        q = sql.select(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
        rows = conn.execute(q).all()
        for row in rows:
            tbl = schema.Table(**row._mapping)
            err_count = error_counts.get(tbl.id, 0) if error_counts is not None else None
            result[tbl.md['name']] = DirEntry(dir=None, dir_entries={}, table=tbl, table_error_count=err_count)

        return result

    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None:
        @retry_schema_change_loop(paths=[path, new_path])
        def move_fn() -> None:
            self._move(path, new_path, if_exists, if_not_exists)

        move_fn()

    def _move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None:
        dest_obj, dest_dir, src_obj = self._prepare_dir_op(
            add_dir_path=new_path.parent,
            add_name=new_path.name,
            drop_dir_path=path.parent,
            drop_name=path.name,
            raise_if_exists=(if_exists == IfExistsParam.ERROR),
            raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR),
        )
        assert dest_obj is None or if_exists == IfExistsParam.IGNORE
        assert src_obj is not None or if_not_exists == IfNotExistsParam.IGNORE
        if dest_obj is None and src_obj is not None:
            # If dest_obj is not None, it means `if_exists='ignore'` and the destination already exists.
            # If src_obj is None, it means `if_not_exists='ignore'` and the source doesn't exist.
            # If dest_obj is None and src_obj is not None, then we can proceed with the move.
            if isinstance(src_obj, LocalTable):
                self._move_table(
                    src_obj._id, new_path.name, dest_dir._id, is_pure_snapshot=src_obj._tbl_version is None
                )
            elif isinstance(src_obj, Dir):
                self._move_dir(src_obj._id, new_path.name, dest_dir._id)
            else:
                raise AssertionError(f'unexpected SchemaObject type: {type(src_obj).__name__}')

    def _prepare_dir_op(
        self,
        add_dir_path: Path | None = None,
        add_name: str | None = None,
        drop_dir_path: Path | None = None,
        drop_name: str | None = None,
        drop_expected: type[SchemaObject] | None = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
    ) -> tuple[SchemaObject | None, Dir | None, SchemaObject | None]:
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
        assert drop_expected in (None, LocalTable, Dir), drop_expected
        assert (add_dir_path is None) == (add_name is None)
        assert (drop_dir_path is None) == (drop_name is None)
        dir_paths: set[Path] = set()
        if add_dir_path is not None:
            dir_paths.add(add_dir_path)
        if drop_dir_path is not None:
            dir_paths.add(drop_dir_path)

        add_dir: schema.Dir | None = None
        drop_dir: schema.Dir | None = None
        for p in sorted(dir_paths):
            dir = self._get_dir(p, lock_dir=True)
            if dir is None:
                # Dir does not exist; raise an appropriate error.
                if add_dir_path is not None or add_name is not None:
                    raise excs.NotFoundError(
                        excs.ErrorCode.DIRECTORY_NOT_FOUND,
                        f'Directory {p!r} does not exist. Create it first with:\npxt.create_dir({p!r})',
                    )
                elif raise_if_not_exists:
                    raise excs.NotFoundError(excs.ErrorCode.DIRECTORY_NOT_FOUND, f'Directory {p!r} does not exist.')
                else:
                    return None, None, None  # parent dir does not exist; nothing to do
            if p == add_dir_path:
                add_dir = dir
            if p == drop_dir_path:
                drop_dir = dir

        add_obj: SchemaObject | None = None
        if add_dir is not None:
            add_obj = self._get_dir_entry(add_dir.id, add_name, lock_entry=True)
            if add_obj is not None and raise_if_exists:
                add_path = add_dir_path.append(add_name)
                raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {add_path!r} already exists.')

        drop_obj: SchemaObject | None = None
        if drop_dir is not None:
            drop_path = drop_dir_path.append(drop_name)
            drop_obj = self._get_dir_entry(drop_dir.id, drop_name, lock_entry=True)
            if drop_obj is None and raise_if_not_exists:
                raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f'Path {drop_path!r} does not exist.')
            if drop_obj is not None and drop_expected is not None and not isinstance(drop_obj, drop_expected):
                expected_name = 'table' if drop_expected is LocalTable else 'directory'
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'{drop_path!r} needs to be a {expected_name} but is a {drop_obj._display_name()}',
                )

        add_dir_obj = Dir(add_dir.id) if add_dir is not None else None
        return add_obj, add_dir_obj, drop_obj

    def _get_dir_entry(
        self, dir_id: UUID, name: str, version: int | None = None, lock_entry: bool = False
    ) -> SchemaObject | None:
        user = Env.get().user
        conn = get_runtime().conn

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
            return Dir(dir_record.id)

        # check for table. The name slot needs no lock of its own: the caller X-locked this directory, and that lock
        # is what guards a name in it.
        q = sql.select(schema.Table.id).where(
            self._active_tbl_clause(dir_id=dir_id, tbl_name=name), schema.Table.md['user'].astext == user
        )
        tbl_id = conn.execute(q).scalars().all()
        assert len(tbl_id) <= 1, name
        if len(tbl_id) == 1:
            return self.get_table_by_id(tbl_id[0], version)

        return None

    def _get_schema_object(
        self,
        path: Path,
        expected: type[SchemaObject] | None = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
        lock_parent: bool = False,
    ) -> SchemaObject | None:
        """Return the schema object at the given path, or None if it doesn't exist.

        Raises Error if
        - the parent directory doesn't exist
        - raise_if_exists is True and the path exists
        - raise_if_not_exists is True and the path does not exist
        - expected is not None and the existing object has a different type
        """
        assert expected in (None, LocalTable, Dir), expected

        if path.is_root:
            # the root dir
            if expected is not None and expected is not Dir:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{path!r} needs to be a table but is a dir')
            dir = self._get_dir(path)
            if dir is None:
                # TODO: why unknown user?
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Unknown user: {Env.get().user}')
            return Dir(dir.id)

        parent_path = path.parent
        parent_dir = self._get_dir(parent_path, lock_dir=lock_parent)
        if parent_dir is None:
            if raise_if_not_exists:
                raise excs.NotFoundError(
                    excs.ErrorCode.DIRECTORY_NOT_FOUND, f'Directory {parent_path!r} does not exist.'
                )
            else:
                return None
        obj = self._get_dir_entry(parent_dir.id, path.name, path.version)

        if obj is None and raise_if_not_exists:
            raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f'Path {path!r} does not exist.')
        elif obj is not None and raise_if_exists:
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {path!r} is an existing {obj._display_name()}.'
            )
        elif obj is not None and expected is not None and not isinstance(obj, expected):
            expected_name = 'table' if expected is LocalTable else 'directory'
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{path!r} needs to be a {expected_name} but is a {obj._display_name()}.',
            )
        return obj

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> LocalTable | None:
        """Loads the table if it isn't already cached, starting its own (re-entrant) transaction to do so.
        Might raise PendingTableOpsError."""
        key = TableVersionKey(tbl_id, version)
        if key not in self._tbls:
            # begin_xact() is re-entrant: it joins the caller's transaction if there is one, and otherwise
            # starts a fresh read transaction (which also permits the metadata load). Cache hits stay xact-free.
            with self.begin_read_md_xact():
                if version is None:
                    tbl = self._load_tbl(tbl_id, ignore_pending_drop=ignore_if_dropped)
                else:
                    tbl = self._load_tbl_at_version(tbl_id, version)
        else:
            tbl = self._tbls.get(key)
            self._tbls.move_to_end(key)
        if tbl is not None:
            Env.get().record_tbl_catalog_uri(tbl._id, ROOT_PATH)
        return tbl

    def create_table(
        self,
        path: Path,
        schema: dict[str, ColumnSpec],
        if_exists: IfExistsParam,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        has_default_idxs: bool,
        is_data_versioned: bool,
    ) -> tuple[LocalTable, bool]:
        """
        Creates a new InsertableTable at the given path.

        If `if_exists == IfExistsParam.IGNORE` and a table `t` already exists at the given path, returns `t, False`.

        Otherwise, creates a new table `t` and returns `t, True` (or raises an exception if the operation fails).
        """

        columns = [Column.create(name, spec) for name, spec in schema.items()]

        return self._create_table(
            path, columns, if_exists, comment, custom_metadata, media_validation, has_default_idxs, is_data_versioned
        )

    def _create_table(
        self,
        path: Path,
        columns: list[Column],
        if_exists: IfExistsParam,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        has_default_idxs: bool,
        is_data_versioned: bool,
        additional_idxs: list[IndexSpec] | None = None,
        explicit_tbl_id: UUID | None = None,
    ) -> tuple[LocalTable, bool]:
        import pixeltable.metadata.schema

        # If a table id is passed in advance, we guarantee that the returned table will be created with that id.
        # Therefore IfExistsParam.IGNORE is incompatible with explicit_tbl_id.
        assert explicit_tbl_id is None or if_exists != IfExistsParam.IGNORE

        if additional_idxs is None:
            additional_idxs = []

        @retry_schema_change_loop(
            paths=[path] if if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE) else [],
            lock_path_subtree=True,
        )
        def create_fn() -> tuple[UUID, bool]:
            existing = self._handle_path_collision(path, InsertableTable, False, if_exists)
            if existing is not None:
                assert isinstance(existing, LocalTable)
                return existing._id, False

            dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
            assert dir is not None

            # This is an actual table creation event; if no explicit_tbl_id was provided, assign a random one now.
            tbl_id = explicit_tbl_id or uuid4()

            md, ops = InsertableTable._create(
                tbl_id,
                path.name,
                columns,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                has_default_idxs=has_default_idxs,
                is_data_versioned=is_data_versioned,
                additional_idxs=additional_idxs,
            )
            assert tbl_id == UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = pixeltable.metadata.schema.TableStatement.CREATE_TABLE
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            self._create_store_tbl(md.tbl_md)
            return tbl_id, True

        tbl_id, is_created = create_fn()
        self._roll_forward()

        @retry_read_md_loop(tbl_keys=[TableVersionKey(tbl_id, None)])
        def get_tbl_fn() -> LocalTable:
            return self.get_table_by_id(tbl_id)

        return get_tbl_fn(), is_created

    def create_view(
        self,
        path: Path,
        base: TablePath,
        select_list: list[tuple[exprs.Expr, str | None]] | None,
        where: exprs.Expr | None,
        sample_clause: 'SampleClause' | None,
        additional_columns: Mapping[str, ColumnSpec] | None,
        is_snapshot: bool,
        has_default_idxs: bool,
        iterator: func.GeneratingFunctionCall | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> tuple[LocalTable, bool]:
        assert isinstance(base, TableVersionPath)

        additional_columns_ = [Column.create(name, spec) for name, spec in additional_columns.items()]

        return self._create_view(
            path,
            base,
            select_list,
            where,
            sample_clause,
            additional_columns_,
            is_snapshot,
            has_default_idxs,
            iterator,
            comment,
            custom_metadata,
            media_validation,
            if_exists,
        )

    def _create_view(
        self,
        path: Path,
        base: TableVersionPath,
        select_list: list[tuple[exprs.Expr, str | None]] | None,
        where: exprs.Expr | None,
        sample_clause: 'SampleClause' | None,
        additional_columns: list[Column],
        is_snapshot: bool,
        has_default_idxs: bool,
        iterator: func.GeneratingFunctionCall | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
        additional_idxs: list[IndexSpec] | None = None,
        explicit_tbl_id: UUID | None = None,
    ) -> tuple[LocalTable, bool]:
        assert explicit_tbl_id is None or if_exists == IfExistsParam.ERROR

        if additional_idxs is None:
            additional_idxs = []

        # a mutable view of a mutable base bumps the base's view_sn, which is a metadata write to the base
        bumps_base_view_sn = not is_snapshot and base.is_mutable()

        @retry_schema_change_loop(
            tvps=[base] if bumps_base_view_sn else None,
            paths=[path] if if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE) else [],
            lock_path_subtree=True,
        )
        def create_fn() -> tuple[UUID, bool]:
            existing = self._handle_path_collision(path, View, is_snapshot, if_exists, base=base)
            if existing is not None:
                assert isinstance(existing, View)
                return existing._id, False

            if bumps_base_view_sn:
                # this is a mutable view of a mutable base; advance the base's view_sn before adding the view. The
                # base is a write target of this transaction, so its store table is already locked.
                assert self._has_store_tbl_lock(base.tbl_id, _TblLockMode.ACCESS_EXCLUSIVE), base.tbl_id
                base_tv = self._get_tbl_version(TableVersionKey(base.tbl_id, None))
                self.mark_modified_tv(base_tv.handle)
                base_tv.tbl_md.view_sn += 1
                result = get_runtime().conn.execute(
                    sql.update(schema.Table)
                    .values({schema.Table.md: dataclasses.asdict(base_tv.tbl_md, dict_factory=schema.md_dict_factory)})
                    .where(schema.Table.id == base.tbl_id)
                )
                assert result.rowcount == 1, result.rowcount

            dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
            assert dir is not None

            tbl_id = explicit_tbl_id or uuid4()

            md, ops = View._create(
                tbl_id,
                path.name,
                base=base,
                select_list=select_list,
                additional_columns=additional_columns,
                predicate=where,
                sample_clause=sample_clause,
                is_snapshot=is_snapshot,
                has_default_idxs=has_default_idxs,
                iterator_call=iterator,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                additional_idxs=additional_idxs,
            )
            assert tbl_id == UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = schema.TableStatement.CREATE_VIEW
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            if not md.tbl_md.is_pure_snapshot:
                self._create_store_tbl(md.tbl_md)
            fault_injection.process_fault(FaultLocation.CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED)
            return tbl_id, True

        view_id, is_created = create_fn()
        if not is_snapshot and base.is_mutable():
            # invalidate base's TableVersion instance, so that it gets reloaded with the new mutable view
            self._clear_tv_cache(base.tbl_version.key)
        self._roll_forward()

        # a snapshot view's cache entry is keyed by its version, which is 0 for a freshly created one
        @retry_read_md_loop(tbl_keys=[TableVersionKey(view_id, 0 if is_snapshot else None)])
        def get_tbl_fn() -> LocalTable:
            return self.get_table_by_id(view_id)

        return get_tbl_fn(), is_created

    def _create_store_tbl(self, tbl_md: schema.TableMd) -> None:
        """Create the store table within the current transaction.

        This maintains the invariant that a table metadata record exists iff the store table is present (except pure
        snapshots that don't maintain a store table).
        """
        assert self._in_write_xact
        assert not tbl_md.is_pure_snapshot
        tbl_id = UUID(tbl_md.tbl_id)
        key = TableVersionKey(tbl_id, tbl_md.current_version if tbl_md.is_snapshot else None)
        tv = self._get_tbl_version(key, check_pending_ops=False)
        assert tv is not None
        tv.store_tbl.create()

    def create_from_model(
        self,
        path: Path,
        columns: dict[str, ColumnSpec],
        display_name: str,
        has_default_idxs: bool,
        media_validation: MediaValidation,
        comment: str | None,
        custom_metadata: Any,
        iterator: func.GeneratingFunctionCall | None,
        base: 'pxt.Query | None',
        idxs: list[IndexDeclaration],
        is_data_versioned: bool,
    ) -> tuple[LocalTable, bool]:
        """Create a table or view from a declarative model.

        A model's column value expressions can contain placeholder references to other columns in the same table.
        Those references arrive as
        ColumnRefByNames and are substituted here, in the catalog that owns path, so they never have to
        be resolved across a proxy boundary. base, when present (i.e. this is a view), is an already-bound Query
        over the existing base table.

        Returns (table, was_created).
        """
        # We allocate the table id up front so that self-referential ColumnRefs (built below) point at it; since
        # this runs in the catalog that owns the table, no such reference ever needs to be serialized.
        tbl_id = uuid4()
        tbl_handle = TableVersionHandle(TableVersionKey(tbl_id, None))

        iterator, additional_cols, resolved_idxs = prepare_model(
            tbl_handle, columns, display_name, iterator, base, idxs, is_data_versioned
        )

        # If the table already exists, rebind to it and report that nothing was created.
        existing = self.get_table(path, IfNotExistsParam.IGNORE)
        if existing is not None:
            return existing, False

        if base is None:
            return self._create_table(
                path=path,
                columns=additional_cols,
                if_exists=IfExistsParam.ERROR,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                has_default_idxs=has_default_idxs,
                is_data_versioned=is_data_versioned,
                additional_idxs=resolved_idxs,
                explicit_tbl_id=tbl_id,
            )

        else:
            assert is_data_versioned, 'TODO: implement for operational tables [PXT-1101]'
            return self._create_view(
                path=path,
                base=base._first_tbl,
                select_list=base.select_list,
                where=base.where_clause,
                sample_clause=base.sample_clause,
                additional_columns=additional_cols,
                is_snapshot=False,
                has_default_idxs=has_default_idxs,
                iterator=iterator,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                if_exists=IfExistsParam.ERROR,
                additional_idxs=resolved_idxs,
                explicit_tbl_id=tbl_id,
            )

    def update_from_model(self, change_sets: list[TableSchemaChangeSet]) -> None:
        """Update tables/views from declarative models.

        If the table does not exist, raises NotFoundError. If the model is incompatible with the existing table,
        raises RequestError.

        Requires that change_sets is ordered topologically, ie, base tables precede their views.
        """
        # fault point:
        # - the diff that produced updates was computed in an earlier read transaction
        # - this call applies it in a later write transaction
        fault_injection.process_fault(FaultLocation.CATALOG_UPDATE_FROM_MODEL_BEFORE_APPLY)
        tbl_ids = [change_set['tbl_id'] for change_set in change_sets]

        @retry_schema_change_loop(tbl_keys=[TableVersionKey(i, None) for i in tbl_ids], lock_mutable_tree=True)
        def update_fn() -> None:
            tbls = [self.get_table_by_id(tbl_id, ignore_if_dropped=True) for tbl_id in tbl_ids]
            # check for tables that were dropped since the diff was computed
            for tbl, change_set in zip(tbls, change_sets):
                if tbl is None:
                    raise excs.ConcurrencyError(
                        excs.ErrorCode.CONCURRENT_MODIFICATION,
                        f'Table {str(change_set["path"])!r} was dropped since update_all() computed its changes; '
                        'please re-run update_all().',
                    )

            # make sure that the tables to which we're applying the schema changes still have the same schema as
            # of the time we computed the diff
            for tbl, change_set in zip(tbls, change_sets):
                assert tbl is not None  # checked above
                if tbl._tbl_version_path.schema_versions() != change_set['schema_versions']:
                    raise excs.ConcurrencyError(
                        excs.ErrorCode.CONCURRENT_MODIFICATION,
                        f'Table {str(change_set["path"])!r} saw schema changes since update_all() computed '
                        'its changes; re-run update_all().',
                    )

            # (tbl_version_path, tbl_version, TableSchemaChangeSet) tuple for each table in the model update
            tbl_info = list(
                zip((tbl._tbl_version_path for tbl in tbls), (tbl._tbl_version.get() for tbl in tbls), change_sets)
            )

            # validate all columns that get dropped, either explicitly or implicitly:
            # - explicitly dropped columns
            # - value columns of explicitly dropped indices
            # - value columns of implicitly dropped indices (= the indexed column was dropped)
            dropped_col_set: set[Column] = set()
            for _, tv, change_set in tbl_info:
                dropped_idxs = [tv.idxs_by_name[name] for name in change_set['dropped_idxs']]
                for name in change_set['dropped_columns']:
                    col = tv.cols_by_name[name]
                    dropped_col_set.add(col)
                    dropped_idxs.extend(tv.idxs_by_col.get(col.qid, []))
                for idx_info in dropped_idxs:
                    dropped_col_set.update(idx_info.columns)

            def dependent_str(c: Column) -> str:
                """How a column that blocks a drop is named in the error, which is by index if it belongs to one."""
                # all user-visible columns have a name
                if c.name is not None:
                    return c.name
                tv = c.get_tbl()
                idx_info = next((i for i in tv.idxs.values() if i.val_col is not None and c.id == i.val_col.id), None)
                assert idx_info is not None
                return f'index {idx_info.name!r} on {tv.name!r}'

            def check_column_dependents(
                dropped: Column | TableVersion.IndexInfo, drop_target: Literal['index', 'column']
            ) -> None:
                if isinstance(dropped, TableVersion.IndexInfo) and dropped.val_col is None:
                    assert dropped.undo_col is None, dropped
                    # Index without value or undo columns -- nothing to do
                    return
                col = dropped.val_col if isinstance(dropped, TableVersion.IndexInfo) else dropped
                # we exclude dependents that themselves are being dropped
                remaining_dependents = [
                    c for c in self.get_column_dependents(col.get_tbl().id, col.id) if c not in dropped_col_set
                ]
                if len(remaining_dependents) > 0:
                    # sorted() for a deterministic error message
                    detail = ', '.join(sorted(dependent_str(c) for c in remaining_dependents))
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'{drop_target.capitalize()} {dropped.name!r} was removed from the model for '
                        f'{col.get_tbl().name!r}, '
                        f'but cannot be dropped because the following depend on it:\n{detail}\n'
                        'Drop those first, or remove them from their models.',
                    )

            for _, tv, change_set in tbl_info:
                for idx_name in change_set['dropped_idxs']:
                    check_column_dependents(tv.idxs_by_name[idx_name], 'index')
                for name in change_set['dropped_columns']:
                    check_column_dependents(tv.cols_by_name[name], 'column')

            # check for dependent view predicates
            mutable_views = {view_tv.id: view_tv for _, tv, _ in tbl_info for view_tv in self._mutable_view_tvs(tv)}
            # a column can appear in more than one view's predicate, so every referencing view is recorded
            views_by_qid: dict[QColumnId, list[TableVersion]] = defaultdict(list)
            for view_tv in mutable_views.values():
                if view_tv.predicate is None:
                    continue
                for col_ref in view_tv.predicate.subexprs(expr_class=exprs.ColumnRef, traverse_matches=False):
                    views_by_qid[col_ref.col_md.qcolid].append(view_tv)
            dropped_cols_by_qid = {col.qid: col for col in dropped_col_set}
            view_dependencies = [
                (dropped_cols_by_qid[qid], view_tv)
                for qid, view_tvs in views_by_qid.items()
                if qid in dropped_cols_by_qid
                for view_tv in view_tvs
            ]
            if len(view_dependencies) > 0:
                # sort() for deterministic error message
                view_dependencies.sort(key=lambda d: (d[0].qid.tbl_id, d[0].qid.col_id, d[1].name))
                detail = '\n'.join(
                    f'column: {col.name}, view: {view_tv.name}, predicate: {view_tv.predicate}'
                    for col, view_tv in view_dependencies
                )
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot drop the following columns, because view predicates depend on them:\n{detail}',
                )

            # Apply per table in forward order (base tables first), so a view's new column can reference a base's
            # new column: the view's resolution sees the base's already-mutated columns through tvp.columns().
            updated_tbl_ids = {tvp.tbl_id for tvp, _, _ in tbl_info}
            applied_tbl_ids: set[UUID] = set()
            for tvp, tv, change_set in tbl_info:
                # make sure we're doing this in base -> view order
                pending_ancestor_ids = (set(tvp.tbl_ids[1:]) & updated_tbl_ids) - applied_tbl_ids
                assert len(pending_ancestor_ids) == 0, f'{tv.name}: bases not yet applied: {pending_ancestor_ids}'

                added_cols, added_idxs = prepare_model_updates(
                    tvp, tv.display_str(), change_set['new_columns'], change_set['new_idxs']
                )
                dropped_cols = [tv.cols_by_name[name] for name in change_set['dropped_columns']]
                dropped_idx_ids = [tv.idxs_by_name[name].id for name in change_set['dropped_idxs']]
                expected_schema_version = change_set['schema_versions'][change_set['tbl_id']]
                _logger.info(
                    f'Applying model updates to {tv.name!r} (id={tv.id}, schema_versions={expected_schema_version}): '
                    f'add columns {[col.name for col in added_cols]}, drop columns {change_set["dropped_columns"]}, '
                    f'add indexes {[spec.idx_name for spec in added_idxs]}, '
                    f'drop indexes {change_set["dropped_idxs"]}'
                )
                tv.apply_schema_change(expected_schema_version, added_cols, dropped_cols, added_idxs, dropped_idx_ids)
                applied_tbl_ids.add(tvp.tbl_id)

        try:
            update_fn()
        except excs.NotFoundError as e:
            # a table identified by the diff may no longer exist: it was dropped, or dropped and recreated at the
            # same path, which gives it a new id. Report the ones that are gone by path, not by internal id.
            # The store is queried directly: a Table cached from before the drop still answers get_table_by_id().
            with self.begin_read_md_xact():
                conn = get_runtime().conn
                q = sql.select(schema.Table.id).where(schema.Table.id.in_(tbl_ids))
                live_tbl_ids = {row.id for row in conn.execute(q)}
            missing = [
                repr(str(change_set['path'])) for change_set in change_sets if change_set['tbl_id'] not in live_tbl_ids
            ]
            if len(missing) == 0:
                raise  # not about a table of this update
            subject = f'Table {missing[0]}' if len(missing) == 1 else f'Tables {", ".join(missing)}'
            verb = 'was' if len(missing) == 1 else 'were'
            raise excs.ConcurrencyError(
                excs.ErrorCode.CONCURRENT_MODIFICATION,
                f'{subject} {verb} dropped or replaced since update_all() computed its changes; re-run update_all().',
            ) from e

    def add_columns(self, tbl: TableVersionPath, cols: list[Column]) -> None:
        @retry_schema_change_loop(tvps=[tbl], lock_mutable_tree=False)
        def add_fn() -> None:
            tv = self._get_tbl_version(TableVersionKey(tbl.tbl_id, None))
            md, ops = tv.add_columns_ops(cols)
            md.tbl_md.pending_stmt = schema.TableStatement.ADD_COLUMNS
            self.write_tbl_md(
                tbl.tbl_id,
                dir_id=None,
                tbl_md=md.tbl_md,
                version_md=md.version_md,
                schema_version_md=md.schema_version_md,
                pending_ops=ops,
            )

        add_fn()
        # force a reload in order to see the new columns/idxs
        self._clear_tv_cache(TableVersionKey(tbl.tbl_id, None))
        self._roll_forward()
        tbl.clear_cached_md()  # force reload of metadata

    def _clear_tv_cache(self, key: TableVersionKey) -> None:
        if key in self._tbl_versions:
            tv = self._tbl_versions[key]
            tv.is_validated = False
            del self._tbl_versions[key]

    @retry_read_md_loop()
    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> LocalTable | None:
        obj = self._get_schema_object(
            path, expected=LocalTable, raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR)
        )
        if obj is None:
            _logger.info(f'Skipped table {path!r} (does not exist).')
            return None

        assert isinstance(obj, LocalTable)
        # We need to clear cached metadata from tbl_version_path, in case the schema has been changed
        # by another process.
        obj._tbl_version_path.clear_cached_md()
        return obj

    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        @retry_schema_change_loop(paths=[path], lock_path_subtree=True)
        def drop_fn() -> None:
            tbl = self._get_schema_object(
                path,
                expected=LocalTable,
                raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR and not force),
                lock_parent=True,
            )
            if tbl is None:
                _logger.info(f'Skipped table {path!r} (does not exist).')
                return
            assert isinstance(tbl, LocalTable)
            self._drop_tbl(tbl, force=force, is_replace=False)

        drop_fn()
        self._roll_forward()

    def _drop_tbl(self, tbl: LocalTable | TableVersionPath, force: bool, is_replace: bool) -> None:
        """
        Drop the table (and recursively its views, if force == True).

        `tbl` can be an instance of `Table` for a user table, or `TableVersionPath` for a hidden (system) table.

        Every lock this needs is already held: the caller named the table by a write path, which locks the store
        tables of the table, its views and its base, plus the `dirs` row of every directory the drop reaches.
        Taking a directory lock here instead would take it out of the sorted order its phase requires.
        """
        is_pure_snapshot: bool
        if isinstance(tbl, TableVersionPath):
            tvp = tbl
            tbl_id = tvp.tbl_id
            tbl = None
            is_pure_snapshot = False
        else:
            tvp = tbl._tbl_version_path
            tbl_id = tbl._id
            is_pure_snapshot = tbl._tbl_version is None

        # capture the path for logging before the drop runs (after drop, tbl is no longer safe to use)
        tbl_path_repr: str = str(tbl_id) if tbl is None else repr(tbl._path())
        if tbl is not None:
            assert self._is_dir_locked(tbl._dir_id()), (tbl_id, tbl._dir_id(), self._dir_locks_held)

        # a drop of a mutable view writes its base's md too, so the base's metadata has to be cached as well
        if isinstance(tbl, View) and tvp.is_mutable() and tvp.base.is_mutable():
            self._refresh_tbl_cache(key=TableVersionKey(tvp.base.tbl_id, None))
        self._refresh_tbl_cache(key=TableVersionKey(tbl_id, None))

        view_ids = self.get_view_ids(tbl_id)

        _logger.debug(f'Preparing to drop table {tbl_id} (force={force!r}).')

        if len(view_ids) > 0:
            if force:
                # recursively drop views first
                for view_id in view_ids:
                    view = self.get_table_by_id(view_id, ignore_if_dropped=True)
                    if view is not None:
                        self._drop_tbl(view, force=force, is_replace=is_replace)

            else:
                # It has dependents and no 'force', so it's an error to drop it.
                assert tbl is not None  # can only occur for a user table
                dependents = f'the following depend on it: {self._tbl_paths_str(view_ids)}'
                msg: str
                if is_replace:
                    msg = (
                        f'{tbl._display_str()} already exists and {dependents}. '
                        "Use `if_exists='replace_force'` to replace it."
                    )
                else:
                    msg = f'{tbl._display_str()} cannot be dropped, {dependents}.'
                raise excs.RequestError(excs.ErrorCode.CONSTRAINT_VIOLATION, msg)

        if is_pure_snapshot:
            # there is no physical table, but we still need to delete the Table record; we can do that right now
            # as part of the current transaction
            self.delete_tbl_md(tbl_id)
        else:
            # invalidate the TableVersion instance when we're done so that existing references to it can find out it
            # has been dropped
            self.mark_modified_tv(tvp.tbl_version)

            # write TableOps to execute the drop, plus the updated Table record
            tv = tvp.tbl_version.get()
            tv.tbl_md.pending_stmt = schema.TableStatement.DROP_TABLE
            drop_ops, new_version, mutable_base_tbl_id = tv.drop_ops()
            if mutable_base_tbl_id is not None:
                # dropping a mutable view changes how writes to the base propagate. The bump happens here, in the
                # transaction that already holds the base's write lock, rather than in the op that deletes the md:
                # a pending op runs with only its own table locked for write, and this writes the base's record.
                self._incr_view_sn(mutable_base_tbl_id)
            self.write_tbl_md(
                tv.id,
                dir_id=None,
                tbl_md=tv.tbl_md,
                version_md=tv.version_md if new_version else None,
                schema_version_md=tv.schema_version_md if new_version else None,
                pending_ops=drop_ops,
                remove_from_dir=True,
            )

        tvp.clear_cached_md()

        # Remove visible Table references.
        keys = [k for k in self._tbls if k.tbl_id == tbl_id]
        assert any(k.effective_version is None for k in keys)  # tables must have an entry with effective_version=None
        for k in keys:
            del self._tbls[k]

        _logger.info(f'Dropped table {tbl_path_repr}.')

    def _incr_view_sn(self, tbl_id: UUID) -> None:
        """Increments the table's view_sn in the store within the current transaction"""
        self._clear_tv_cache(TableVersionKey(tbl_id, None))
        assert self._has_store_tbl_lock(tbl_id, _TblLockMode.ACCESS_EXCLUSIVE), tbl_id
        result = get_runtime().conn.execute(
            sql.update(schema.Table)
            .values(
                md=sql.func.jsonb_set(
                    schema.Table.md,
                    pg_array(['view_sn']),
                    sql.func.to_jsonb(sql.cast(schema.Table.md['view_sn'].astext, sql.Integer) + 1),
                )
            )
            .where(schema.Table.id == tbl_id)
        )
        assert result.rowcount == 1, (tbl_id, result.rowcount)

    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        @retry_schema_change_loop(
            paths=[path] if if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE) else [],
            lock_path_subtree=True,
        )
        def create_fn() -> Dir:
            return self._create_dir(path, if_exists, parents)

        return create_fn()

    def _create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        if parents:
            # start walking down from the root
            last_parent: SchemaObject | None = None
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
        Env.get().console_logger.info(f'Created directory {path!r}.')
        return dir

    def drop_dir(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        @retry_schema_change_loop(paths=[path], lock_path_subtree=True)
        def drop_fn() -> None:
            _, _, schema_obj = self._prepare_dir_op(
                drop_dir_path=path.parent,
                drop_name=path.name,
                drop_expected=Dir,
                raise_if_not_exists=if_not_exists == IfNotExistsParam.ERROR and not force,
            )
            if schema_obj is None:
                _logger.info(f'Directory {path!r} does not exist; skipped drop_dir().')
                return
            self._drop_dir(schema_obj._id, path, force=force)

        drop_fn()
        self._roll_forward()

    def _drop_dir(self, dir_id: UUID, dir_path: Path, force: bool = False) -> None:
        conn = get_runtime().conn
        if not force:
            # check for existing entries
            q = sql.select(sql.func.count()).select_from(schema.Dir).where(schema.Dir.parent_id == dir_id)
            num_subdirs = conn.execute(q).scalar()
            q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
            num_tbls = conn.execute(q).scalar()
            if num_subdirs + num_tbls > 0:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Directory {dir_path!r} is not empty.')

        # drop existing subdirs; every directory of the subtree was locked before the walk started
        assert self._is_dir_locked(dir_id), (dir_id, self._dir_locks_held)
        dir_q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
        for row in conn.execute(dir_q).all():
            self._drop_dir(row.id, dir_path.append(row.md['name']), force=True)

        # drop existing tables
        tbl_q = sql.select(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
        for row in conn.execute(tbl_q).all():
            tbl = self.get_table_by_id(row.id, ignore_if_dropped=True)
            # this table would have been dropped already if it's a view of a base we dropped earlier
            if tbl is not None:
                self._drop_tbl(tbl, force=True, is_replace=False)

        # self.drop_dir(dir_id)
        conn.execute(sql.delete(schema.Dir).where(schema.Dir.id == dir_id))
        _logger.info(f'Removed directory {dir_path!r}.')

    def get_view_ids(self, tbl_id: UUID) -> list[UUID]:
        """Return the ids of views that directly reference the given table"""
        conn = get_runtime().conn
        # check whether this table still exists
        q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(tbl_id=tbl_id))
        tbl_count = conn.execute(q).scalar()
        if tbl_count == 0:
            raise excs.table_was_dropped(tbl_id)
        q = (
            sql.select(schema.Table.id)
            .where(schema.Table.md['view_md']['base_versions'][0][0].astext == tbl_id.hex)
            .where(self._active_tbl_clause())
        )
        return [r[0] for r in conn.execute(q).all()]

    def get_tbl_version(self, key: TableVersionKey, *, validate_initialized: bool = True) -> TableVersion | None:
        """
        Returns the TableVersion instance for the given table version key, and updates the cache if necessary.

        This function can, but doesn't have to be called inside a transaction or a retry loop. It can manage its own
        retry loop internally if necessary.
        """
        if get_runtime().in_xact:
            return self._get_tbl_version(key, validate_initialized=validate_initialized)

        @retry_read_md_loop()
        def do_get_tbl_version() -> TableVersion | None:
            return self._get_tbl_version(key, validate_initialized=validate_initialized)

        return do_get_tbl_version()

    def _get_tbl_version(
        self, key: TableVersionKey, *, check_pending_ops: bool = True, validate_initialized: bool = True
    ) -> TableVersion | None:
        """
        Returns the TableVersion instance for the given table key, and updates the cache if necessary.

        If present in the cache and the instance isn't validated, validates version and view_sn against the stored
        metadata.
        """
        conn = get_runtime().conn
        assert conn is not None
        tv = self._tbl_versions.get(key)
        if tv is not None:
            self._tbl_versions.move_to_end(key)
        if tv is None and not self._tbl_md_read_allowed:
            raise AssertionError(
                'Loading new table metadata is not allowed in the middle of a transaction. '
                'To fix this, either: (1) declare all tables to be accessed upfront when starting the '
                'transaction, or (2) run the operation inside a retry loop.'
            )
        if tv is None:
            tv = self._load_tbl_version(key, check_pending_ops=check_pending_ops)
        elif not tv.is_validated:
            # only live instances are invalidated
            assert key.effective_version is None
            where_clause: sql.ColumnElement[bool]
            if check_pending_ops:
                # if we don't want to see pending ops, we also don't want to see dropped tables
                where_clause = self._active_tbl_clause(tbl_id=key.tbl_id)
            else:
                where_clause = schema.Table.id == key.tbl_id
            q = sql.select(schema.Table.md).where(where_clause)
            row = conn.execute(q).one_or_none()
            if row is None:
                raise excs.table_was_dropped(key.tbl_id)

            reload = False

            # live table; compare our cached TableMd.current_version/view_sn to what's stored
            current_version = row.md['current_version']
            view_sn = row.md['view_sn']
            if current_version != tv.version or view_sn != tv.tbl_md.view_sn:
                _logger.debug(
                    f'reloading metadata for live table {key.tbl_id} '
                    f'(cached/current version: {tv.version}/{current_version}, '
                    f'cached/current view_sn: {tv.tbl_md.view_sn}/{view_sn})'
                )
                reload = True

            # the stored version can be behind TableVersion.version, because we don't roll back the in-memory
            # metadata changes after a failed update operation
            if reload:
                # the cached metadata is invalid
                tv = self._load_tbl_version(key, check_pending_ops=check_pending_ops)
            else:
                # the cached metadata is valid
                tv.is_validated = True

        assert tv.is_validated, f'{key} not validated\n{tv.__dict__}\n{self._debug_str()}'
        if validate_initialized:
            assert tv.is_initialized, f'{key} not initialized\n{tv.__dict__}\n{self._debug_str()}'
        return tv

    def remove_tbl_version(self, key: TableVersionKey) -> None:
        assert isinstance(key, TableVersionKey)
        assert key in self._tbl_versions
        del self._tbl_versions[key]

    def get_dir(self, dir_id: UUID) -> Dir | None:
        """Return the Dir with the given id, or None if it doesn't exist"""
        conn = get_runtime().conn
        q = sql.select(schema.Dir).where(schema.Dir.id == dir_id)
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        dir_record = schema.Dir(**row._mapping)
        return Dir(dir_record.id)

    def read_tbl_record(self, tbl_id: UUID) -> schema.Table:
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Table).where(schema.Table.id == tbl_id)).one_or_none()
        if row is None:
            raise excs.table_was_dropped(tbl_id)
        return schema.Table(**row._mapping)

    def _tbl_paths_str(self, tbl_ids: Sequence[UUID], max_paths: int = 5) -> str:
        """Returns the paths of the given tables, comma-separated and sorted, for use in an error message.

        Reads the stored records directly rather than loading each table, so that it is usable at any point in a
        transaction. Paths are listed in sorted order, and beyond a fixed limit are replaced by a count. A table
        that no longer has a record is left out: this builds the text of an error, and must not raise one itself.
        """
        # every path is read before sorting, so that the ones listed are the first in sorted order rather than
        # an arbitrary subset of the ids
        paths: list[str] = []
        for tbl_id in tbl_ids:
            try:
                record = self.read_tbl_record(tbl_id)
                paths.append(str(self.get_dir_path(record.dir_id).append(record.md['name'])))
            except excs.NotFoundError:
                continue
        paths.sort()
        if len(paths) <= max_paths:
            return ', '.join(repr(p) for p in paths)
        listed = ', '.join(repr(p) for p in paths[:max_paths])
        return f'{listed} and {len(paths) - max_paths} more'

    def read_dir_record(self, dir_id: UUID) -> schema.Dir:
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Dir).where(schema.Dir.id == dir_id)).one_or_none()
        if row is None:
            raise excs.NotFoundError(excs.ErrorCode.DIRECTORY_NOT_FOUND, f'Directory not found: {dir_id}')
        return schema.Dir(**row._mapping)

    def _move_table(self, tbl_id: UUID, new_name: str, new_dir_id: UUID, *, is_pure_snapshot: bool) -> None:
        """Update dir_id/name for tbl_id."""
        self._assert_md_write_locked(tbl_id, is_insert=False, is_pure_snapshot=is_pure_snapshot, dir_id=new_dir_id)
        stmt = (
            sql.update(schema.Table)
            .where(schema.Table.id == tbl_id)
            .values(
                {
                    schema.Table.dir_id: new_dir_id,
                    schema.Table.md: sql.func.jsonb_set(
                        schema.Table.md, pg_array(['name']), sql.func.to_jsonb(new_name)
                    ),
                }
            )
        )
        result = get_runtime().conn.execute(stmt)
        assert result.rowcount == 1, result.rowcount
        # TV.table_md.name is now stale
        self._clear_tv_cache(TableVersionKey(tbl_id, None))

    def _move_dir(self, dir_id: UUID, new_name: str, new_parent_id: UUID) -> None:
        """Update parent_id/name for dir_id."""
        stmt = (
            sql.update(schema.Dir)
            .where(schema.Dir.id == dir_id)
            .values(
                {
                    schema.Dir.parent_id: new_parent_id,
                    schema.Dir.md: sql.func.jsonb_set(schema.Dir.md, pg_array(['name']), sql.func.to_jsonb(new_name)),
                }
            )
        )
        result = get_runtime().conn.execute(stmt)
        assert result.rowcount == 1, result.rowcount

    def _get_dir(self, path: Path, lock_dir: bool = False) -> schema.Dir | None:
        """
        lock_dir: if True, X-locks target (but not the ancestors)
        """
        user = Env.get().user
        conn = get_runtime().conn
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

    def _load_tbl(self, tbl_id: UUID, ignore_pending_drop: bool = False) -> LocalTable | None:
        """Loads metadata for the table with the given id and caches it."""
        from .insertable_table import InsertableTable
        from .view import View

        assert tbl_id is not None
        _logger.info(f'Loading table {tbl_id}')

        conn = get_runtime().conn

        if ignore_pending_drop:
            # check whether this table is in the process of being dropped or has already been dropped
            q: sql.Executable = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
            row = conn.execute(q).one_or_none()
            if row is None or row.md.get('pending_stmt') == schema.TableStatement.DROP_TABLE.value:
                return None

        # check for pending ops
        q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == tbl_id)
        has_pending_ops = conn.execute(q).scalar() > 0
        if has_pending_ops:
            raise PendingTableOpsError(tbl_id)

        q = (
            sql.select(schema.Table, schema.TableSchemaVersion)
            .join(schema.TableSchemaVersion)
            .where(schema.Table.id == schema.TableSchemaVersion.tbl_id)
            .where(
                schema.Table.md['current_schema_version'].cast(sql.Integer) == schema.TableSchemaVersion.schema_version
            )
            .where(schema.Table.id == tbl_id)
        )
        row = conn.execute(q).one_or_none()
        if row is None:
            # the table got dropped
            return None
        tbl_record, _ = _unpack_row(row, [schema.Table, schema.TableSchemaVersion])

        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        view_md = tbl_md.view_md

        if view_md is None:
            # this is a base table
            key = TableVersionKey(tbl_id, None)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)
            tbl = InsertableTable(TableVersionHandle(key))
            self._tbls[key] = tbl
            return tbl

        # this is a view; determine the sequence of TableVersions to load
        tbl_version_path: list[tuple[UUID, int | None]] = []
        if tbl_md.is_pure_snapshot:
            # this is a pure snapshot, without a physical table backing it; we only need the bases
            pass
        else:
            effective_version = (
                0 if view_md is not None and view_md.is_snapshot else None
            )  # snapshots only have version 0
            tbl_version_path.append((tbl_id, effective_version))

        if view_md is not None:
            tbl_version_path.extend((UUID(ancestor_id), version) for ancestor_id, version in view_md.base_versions)

        # load TableVersions, starting at the root
        base_path: TableVersionPath | None = None
        view_path: TableVersionPath | None = None
        for id, effective_version in tbl_version_path[::-1]:
            key = TableVersionKey(id, effective_version)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)
            view_path = TableVersionPath(TableVersionHandle(key), base=base_path)
            base_path = view_path
        view = View(tbl_id, view_path, snapshot_only=tbl_md.is_pure_snapshot)
        self._tbls[TableVersionKey(tbl_id, None)] = view
        return view

    def _load_tbl_at_version(self, tbl_id: UUID, version: int) -> LocalTable | None:
        from .view import View

        # Load the specified TableMd and TableVersionMd records from the db.
        conn = get_runtime().conn
        q: sql.Executable = (
            sql.select(schema.Table, schema.TableVersion)
            .join(schema.TableVersion)
            .where(schema.Table.id == tbl_id)
            .where(schema.Table.id == schema.TableVersion.tbl_id)
            .where(schema.TableVersion.version == version)
        )
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        tbl_record, version_record = _unpack_row(row, [schema.Table, schema.TableVersion])
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        if tbl_md.is_pure_snapshot:
            # a pure snapshot has no physical table to load at a version; resolve it via its base_versions
            return self._load_tbl(tbl_id)
        version_md = schema.md_from_dict(schema.VersionMd, version_record.md)
        tvp = self.construct_tvp(tbl_id, version, tbl_md.ancestors, version_md.created_at)

        # snapshot_only=True: an anonymous snapshot doesn't have a physical table
        view = View(tbl_id, tvp, snapshot_only=True)
        self._tbls[TableVersionKey(tbl_id, version)] = view
        return view

    def construct_tvp(
        self, tbl_id: UUID, version: int, ancestors_of_live_tbl: schema.TableVersionPath, created_at: float
    ) -> TableVersionPath:
        """
        Construct the TableVersionPath for the specified version of the given table. Here `live_ancestors` is the
        list of ancestor table IDs and fixed versions (if any) from the table's metadata. The constructed
        TableVersionPath will preserve any fixed versions from `live_ancestors` (corresponding to a view-over-snapshot
        scenario), while "filling in" the implied versions for any `None` versions.
        """
        # TODO: Currently, we reconstruct the ancestors by inspecting the created_at timestamps of the table and its
        #     ancestors' versions. In the future, we should store the relevant TableVersionPaths in the database, so
        #     that we don't need to rely on timestamps (which might be nondeterministic in distributed execution
        #     scenarios).

        assert get_runtime().in_xact

        # Build the list of ancestor versions, starting with the given table and traversing back to the base table.
        # For each proper ancestor,
        # - If it's an ancestor with a fixed version (view-over-snapshot scenario), we keep the given fixed version.
        # - If it's an ancestor with a live (floating) version, we use the version whose created_at timestamp equals
        #     or most nearly precedes the given TableVersion's created_at timestamp.
        ancestors: list[tuple[UUID, int]] = [(tbl_id, version)]
        for ancestor_id, ancestor_version in ancestors_of_live_tbl:
            if ancestor_version is not None:
                # fixed version; just use it
                ancestors.append((UUID(ancestor_id), ancestor_version))
                continue

            q = (
                sql.select(schema.TableVersion)
                .where(schema.TableVersion.tbl_id == ancestor_id)
                .where(schema.TableVersion.md['created_at'].cast(sql.Float) <= created_at)
                .order_by(schema.TableVersion.md['created_at'].cast(sql.Float).desc())
                .limit(1)
            )
            row = get_runtime().conn.execute(q).one_or_none()
            if row is None:
                # This can happen if an ancestor version is garbage collected; it can also happen in
                # rare circumstances involving table versions created specifically with Pixeltable 0.4.3.
                _logger.info(f'Ancestor {ancestor_id} not found for table {tbl_id}:{version}')
                raise excs.NotFoundError(
                    excs.ErrorCode.TABLE_NOT_FOUND,
                    'The specified table version is no longer valid and cannot be retrieved.',
                )
            ancestor_version_record = _unpack_row(row, [schema.TableVersion])[0]
            ancestor_version_md = schema.md_from_dict(schema.VersionMd, ancestor_version_record.md)
            assert ancestor_version_md.created_at <= created_at
            ancestors.append((UUID(ancestor_id), ancestor_version_md.version))

        # Force any ancestors to be loaded (base table first).
        for anc_id, anc_version in ancestors[::-1]:
            key = TableVersionKey(anc_id, anc_version)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)

        # Now reconstruct the relevant TableVersionPath instance from the ancestor versions.
        tvp: TableVersionPath | None = None
        for anc_id, anc_version in ancestors[::-1]:
            tvp = TableVersionPath(TableVersionHandle(TableVersionKey(anc_id, anc_version)), base=tvp)

        return tvp

    @retry_read_md_loop()
    def collect_tbl_history(self, tbl_id: UUID, n: int | None) -> list[TableVersionMd]:
        return self._collect_tbl_history(tbl_id, n)

    def _collect_tbl_history(self, tbl_id: UUID, n: int | None) -> list[TableVersionMd]:
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
            sql.select(schema.Table, schema.TableVersion, schema.TableSchemaVersion)
            .where(self._active_tbl_clause(tbl_id=tbl_id))
            .join(schema.TableVersion)
            .where(schema.TableVersion.tbl_id == tbl_id)
            .join(schema.TableSchemaVersion)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
            .where(
                schema.TableVersion.md['schema_version'].cast(sql.Integer) == schema.TableSchemaVersion.schema_version
            )
            .order_by(schema.TableVersion.version.desc())
        )
        if n is not None:
            q = q.limit(n)
        src_rows = get_runtime().session.execute(q).fetchall()
        return [
            TableVersionMd(
                tbl_md=schema.md_from_dict(schema.TableMd, row.Table.md),
                version_md=schema.md_from_dict(schema.VersionMd, row.TableVersion.md),
                schema_version_md=schema.md_from_dict(schema.SchemaVersionMd, row.TableSchemaVersion.md),
            )
            for row in src_rows
        ]

    def load_tbl_md(self, key: TableVersionKey) -> TableVersionMd:
        """
        Loads metadata from the store for a given table UUID and version.
        """
        # _logger.info(f'Loading metadata for table version: {tbl_id}:{effective_version}')
        conn = get_runtime().conn

        q = (
            sql.select(schema.Table, schema.TableVersion, schema.TableSchemaVersion)
            .select_from(schema.Table)
            .where(schema.Table.id == key.tbl_id)
            .join(schema.TableVersion)
            .where(schema.TableVersion.tbl_id == key.tbl_id)
            .join(schema.TableSchemaVersion)
            .where(schema.TableSchemaVersion.tbl_id == key.tbl_id)
        )

        if key.effective_version is not None:
            # we are loading a specific version
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id AND tv.version = effective_version)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND tv.md.schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = q.where(
                schema.TableVersion.md['version'].cast(sql.Integer) == key.effective_version,
                schema.TableVersion.md['schema_version'].cast(sql.Integer) == schema.TableSchemaVersion.schema_version,
            )
        else:
            # we are loading the current version
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id AND t.current_version = tv.version)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND t.current_schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = q.where(
                schema.Table.md['current_version'].cast(sql.Integer) == schema.TableVersion.version,
                schema.Table.md['current_schema_version'].cast(sql.Integer) == schema.TableSchemaVersion.schema_version,
            )

        row = conn.execute(q).one_or_none()
        if row is None:
            raise excs.table_was_dropped(key.tbl_id)
        tbl_record, version_record, schema_version_record = _unpack_row(
            row, [schema.Table, schema.TableVersion, schema.TableSchemaVersion]
        )
        assert tbl_record.id == key.tbl_id
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        version_md = schema.md_from_dict(schema.VersionMd, version_record.md)
        schema_version_md = schema.md_from_dict(schema.SchemaVersionMd, schema_version_record.md)

        return TableVersionMd(tbl_md, version_md, schema_version_md)

    def write_tbl_md(
        self,
        tbl_id: UUID,
        dir_id: UUID | None,
        tbl_md: schema.TableMd | None,
        version_md: schema.VersionMd | None,
        schema_version_md: schema.SchemaVersionMd | None,
        pending_ops: list[TableOp] | None = None,
        remove_from_dir: bool = False,
    ) -> None:
        """
        Stores metadata to the DB and adds tbl_id to self._roll_forward_ids if pending_ops is specified.

        Args:
            tbl_id: UUID of the table to store metadata for.
            dir_id: If specified, the tbl_md will be added to the given directory; if None, the table must already exist
            tbl_md: If specified, `tbl_md` will be inserted, or updated (only one such record can exist per UUID)
            version_md: inserted as a new record if present
            schema_version_md: will be inserted as a new record if present

        If inserting `version_md` or `schema_version_md` would be a primary key violation, an exception will be raised.
        """
        assert self._in_write_xact
        assert version_md is None or version_md.created_at > 0.0
        assert pending_ops is None or len(pending_ops) > 0
        assert pending_ops is None or tbl_md is not None  # if we write pending ops, we must also write new tbl_md
        session = get_runtime().session

        # Construct and insert or update table record if requested.
        if tbl_md is not None:
            assert tbl_md.tbl_id == str(tbl_id)
            self._assert_md_write_locked(
                tbl_id, is_insert=dir_id is not None, is_pure_snapshot=tbl_md.is_pure_snapshot, dir_id=dir_id
            )
            if version_md is not None:
                assert tbl_md.current_version == version_md.version
                assert tbl_md.current_schema_version == version_md.schema_version
            if schema_version_md is not None:
                assert tbl_md.current_schema_version == schema_version_md.schema_version
                # Validate that the columns in schema_version_md are consistent with tbl_md.
                sch_col_ids = set(schema_version_md.columns.keys())
                for tbl_col_id, tbl_col_md in tbl_md.column_md.items():
                    if tbl_col_md.is_visible_in_version(tbl_md.current_schema_version):
                        assert tbl_col_id in sch_col_ids, (tbl_md.tbl_id, tbl_col_id)
                        sch_col_ids.remove(tbl_col_id)
                assert len(sch_col_ids) == 0, (tbl_md.tbl_id, sch_col_ids)
                _validate_folded_names(tbl_md, schema_version_md)
            if pending_ops is not None:
                assert tbl_md.pending_stmt is not None
                assert all(op.tbl_id == str(tbl_id) for op in pending_ops)
                assert all(op.op_sn == i for i, op in enumerate(pending_ops))
                assert all(op.num_ops == len(pending_ops) for op in pending_ops)
                tbl_md.tbl_state = schema.TableState.ROLLFORWARD
                assert tbl_id not in self._roll_forward_ids, (tbl_id, self._roll_forward_ids)
                self._roll_forward_ids.append(tbl_id)

            if dir_id is not None:
                # We are inserting a record while creating a new table.
                tbl_record = schema.Table(
                    id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(tbl_md, dict_factory=schema.md_dict_factory)
                )
                session.add(tbl_record)
            else:
                # Update the existing table record.
                values: dict[Any, Any] = {
                    schema.Table.md: dataclasses.asdict(tbl_md, dict_factory=schema.md_dict_factory)
                }
                if remove_from_dir:
                    values.update({schema.Table.dir_id: None})
                result = session.execute(
                    sql.update(schema.Table.__table__).values(values).where(schema.Table.id == tbl_id)
                )
                assert isinstance(result, sql.CursorResult)
                assert result.rowcount == 1, result.rowcount

        # Construct and insert new table version record if requested.
        if version_md is not None:
            assert version_md.tbl_id == str(tbl_id)
            if schema_version_md is not None:
                assert version_md.schema_version == schema_version_md.schema_version
            version_rows = (
                session.query(schema.TableVersion)
                .filter(schema.TableVersion.tbl_id == tbl_id, schema.TableVersion.version == version_md.version)
                .all()
            )
            if len(version_rows) == 0:
                # It's a new table version; insert a new record in the DB for it.
                tbl_version_record = schema.TableVersion(
                    tbl_id=tbl_id, version=version_md.version, md=dataclasses.asdict(version_md)
                )
                session.add(tbl_version_record)
            else:
                # This table version already exists; update it.
                assert len(version_rows) == 1  # must be unique
                version_record = version_rows[0]
                # Validate that the only field that can change is 'additional_md'.
                assert version_record.md == dataclasses.asdict(
                    dataclasses.replace(version_md, additional_md=version_record.md['additional_md'])
                ), (
                    'Table version already exists in store. Expected no change outside of additional_md, '
                    f'but stored version md is {version_record.md} and new one is {version_md}'
                )
                result = session.execute(
                    sql.update(schema.TableVersion.__table__)
                    .values({schema.TableVersion.md: dataclasses.asdict(version_md)})
                    .where(schema.TableVersion.tbl_id == tbl_id, schema.TableVersion.version == version_md.version)
                )
                assert isinstance(result, sql.CursorResult)
                assert result.rowcount == 1, result.rowcount

        # Construct and insert a new schema version record if requested.
        if schema_version_md is not None:
            assert schema_version_md.tbl_id == str(tbl_id)
            schema_version_record = schema.TableSchemaVersion(
                tbl_id=tbl_id, schema_version=schema_version_md.schema_version, md=dataclasses.asdict(schema_version_md)
            )
            session.add(schema_version_record)

        # make sure we don't have any pending ops
        assert session.query(schema.PendingTableOp).filter(schema.PendingTableOp.tbl_id == tbl_id).count() == 0

        if pending_ops is not None:
            for op in pending_ops:
                op_record = schema.PendingTableOp(tbl_id=tbl_id, op_sn=op.op_sn, op=op.to_dict())
                session.add(op_record)

        session.flush()  # Inform SQLAlchemy that we want to write these changes to the DB.

    def delete_current_tbl_version_md(self, tbl_id: UUID) -> None:
        """Removes 'current_version' from stored metadata for table and resets the table to current_version - 1"""
        conn = get_runtime().conn
        q = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
        tbl_md = conn.execute(q).one()[0]
        current_version, current_schema_version = tbl_md['current_version'], tbl_md['current_schema_version']

        # determine preceding schema version
        q = sql.select(schema.TableSchemaVersion.md).where(
            schema.TableSchemaVersion.tbl_id == tbl_id,
            schema.TableSchemaVersion.schema_version == current_schema_version,
        )
        row = conn.execute(q).one_or_none()
        preceding_schema_version: int
        if row is not None:
            schema_version_md = row[0]
            preceding_schema_version = schema_version_md['preceding_schema_version']
        else:
            preceding_schema_version = current_schema_version

        # delete the TableSchemaVersion record, if one was created for this version
        if preceding_schema_version != current_schema_version:
            assert current_version == current_schema_version
            delete_stmt = sql.delete(schema.TableSchemaVersion).where(
                schema.TableSchemaVersion.tbl_id == tbl_id,
                schema.TableSchemaVersion.schema_version == current_schema_version,
            )
            status = conn.execute(delete_stmt)
            assert status.rowcount == 1, status.rowcount

        delete_stmt = sql.delete(schema.TableVersion).where(
            schema.TableVersion.tbl_id == tbl_id, schema.TableVersion.version == current_version
        )
        status = conn.execute(delete_stmt)
        assert status.rowcount == 1, status.rowcount

        # we also need to reset TableMd.current_version/current_schema_version
        version_updates = {'current_version': current_version - 1}
        if preceding_schema_version != current_schema_version:
            version_updates['current_schema_version'] = preceding_schema_version
        update_stmt = (
            sql.update(schema.Table)
            .where(schema.Table.id == tbl_id)
            .values(md=schema.Table.md.op('||')(version_updates))
        )
        self._clear_tv_cache(TableVersionKey(tbl_id, None))
        status = conn.execute(update_stmt)
        assert status.rowcount == 1, status.rowcount

    def store_update_status(self, tbl_id: UUID, version: int, status: UpdateStatus) -> None:
        """Update the TableVersion.md.update_status field"""
        assert self._in_write_xact
        conn = get_runtime().conn

        stmt = (
            sql.update(schema.TableVersion)
            .where(schema.TableVersion.tbl_id == tbl_id, schema.TableVersion.version == version)
            .values(md=schema.TableVersion.md.op('||')({'update_status': dataclasses.asdict(status)}))
        )

        res = conn.execute(stmt)
        assert res.rowcount == 1, res.rowcount

    def delete_tbl_md(self, tbl_id: UUID) -> None:
        """
        Deletes all table metadata from the store for the given table UUID.
        """
        conn = get_runtime().conn
        _logger.info(f'delete_tbl_md({tbl_id})')
        status = conn.execute(sql.delete(schema.TableSchemaVersion).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        assert status.rowcount > 0
        status = conn.execute(sql.delete(schema.TableVersion).where(schema.TableVersion.tbl_id == tbl_id))
        assert status.rowcount > 0
        _ = conn.execute(sql.delete(schema.PendingTableOp).where(schema.PendingTableOp.tbl_id == tbl_id))
        self._clear_tv_cache(TableVersionKey(tbl_id, None))
        deleted = conn.execute(
            sql.delete(schema.Table).where(schema.Table.id == tbl_id).returning(schema.Table.dir_id, schema.Table.md)
        ).all()
        assert len(deleted) == 1, len(deleted)
        deleted_dir_id, deleted_table_md = deleted[0]
        tbl_md = schema.md_from_dict(schema.TableMd, deleted_table_md)
        self._assert_md_write_locked(
            tbl_id, is_insert=False, is_pure_snapshot=tbl_md.is_pure_snapshot, dir_id=deleted_dir_id
        )
        if tbl_md.view_md is not None:
            # the base's cached mutable_views still names the table this just deleted, and its view_sn was already
            # advanced by the transaction that initiated the drop, so revalidation would not notice
            base_id, base_version = tbl_md.view_md.base_versions[0]
            if base_version is None:
                self._clear_tv_cache(TableVersionKey(UUID(base_id), None))

    def read_md_for_export(self, tbl: LocalTable) -> list[TableVersionMd]:
        """
        Load metadata for the given table along with all its ancestors. The values of TableMd.current_version and
        TableMd.current_schema_version will be adjusted to ensure that the metadata represent a valid (internally
        consistent) table state.
        """
        # TODO: First acquire X-locks for all relevant metadata entries
        # TODO: handle concurrent drop()

        # Load metadata for every table in the TableVersionPath for `tbl`.
        md = [self.load_tbl_md(tv.key) for tv in tbl._tbl_version_path.get_tbl_versions()]

        # If `tbl` is a named pure snapshot, we're not quite done, since the snapshot metadata won't appear in the
        # TableVersionPath. We need to prepend it separately.
        if isinstance(tbl, View) and tbl._is_named_pure_snapshot():
            snapshot_md = self.load_tbl_md(TableVersionKey(tbl._id, 0))
            md = [snapshot_md, *md]

        for ancestor_md in md:
            # For exported metadata, we guarantee that the current_version and current_schema_version of TableMd
            # match the corresponding values in TableVersionMd and TableSchemaVersionMd. This is to ensure that,
            # when the metadata is later stored in the catalog of a different Pixeltable instance, the values of
            # current_version and current_schema_version will always point to versions that are known to the
            # destination catalog.
            ancestor_md.tbl_md.current_version = ancestor_md.version_md.version
            ancestor_md.tbl_md.current_schema_version = ancestor_md.schema_version_md.schema_version

        return md

    def _load_tbl_version(self, key: TableVersionKey, *, check_pending_ops: bool = True) -> TableVersion:
        """Creates TableVersion instance from stored metadata and registers it in _tbl_versions."""
        tv_md = self.load_tbl_md(key)
        tbl_md = tv_md.tbl_md
        version_md = tv_md.version_md
        schema_version_md = tv_md.schema_version_md
        view_md = tbl_md.view_md

        conn = get_runtime().conn

        if check_pending_ops:
            # if we care about pending ops, we also care whether the table is in the process of getting dropped
            if tbl_md.pending_stmt == schema.TableStatement.DROP_TABLE:
                raise excs.table_was_dropped(key.tbl_id)

            pending_ops_q = (
                sql.select(sql.func.count())
                .select_from(schema.Table)
                .join(schema.PendingTableOp)
                .where(schema.PendingTableOp.tbl_id == key.tbl_id)
                .where(schema.Table.id == key.tbl_id)
            )
            if key.effective_version is not None:
                # we only care about pending ops if the requested version is the current version
                pending_ops_q = pending_ops_q.where(
                    sql.text(f"({schema.Table.__table__}.md->>'current_version')::int = {key.effective_version}")
                )
            has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
            if has_pending_ops:
                raise PendingTableOpsError(key.tbl_id)

        # load mutable view ids for mutable TableVersions
        mutable_view_ids: list[UUID] = []
        if key.effective_version is None:
            q = (
                sql.select(schema.Table.id)
                .where(schema.Table.md['view_md']['base_versions'][0][0].astext == key.tbl_id.hex)
                .where(schema.Table.md['view_md']['base_versions'][0][1].astext == None)
            )
            mutable_view_ids = [r[0] for r in conn.execute(q).all()]

        mutable_views = [TableVersionHandle(TableVersionKey(id, None)) for id in mutable_view_ids]

        tbl_version: TableVersion
        if view_md is None:
            # this is a base table
            tbl_version = TableVersion(key, tbl_md, version_md, schema_version_md, mutable_views)
        else:
            assert len(view_md.base_versions) > 0  # a view needs to have a base
            assert (
                not tv_md.is_pure_snapshot
            )  # a pure snapshot doesn't have a physical table backing it, no point in loading it

            base: TableVersionHandle
            base_path: TableVersionPath | None = None  # needed for live view
            if view_md.is_snapshot:
                base = TableVersionHandle(
                    TableVersionKey(UUID(view_md.base_versions[0][0]), view_md.base_versions[0][1])
                )
            else:
                base_path = TableVersionPath.from_schema_path(tbl_md.view_md.base_versions)
                base = base_path.tbl_version

            tbl_version = TableVersion(
                key, tbl_md, version_md, schema_version_md, mutable_views, base_path=base_path, base=base
            )

        # register the instance before init()
        self._tbl_versions[key] = tbl_version
        # register this instance as modified, so that it gets purged if the transaction fails, it may not be
        # fully initialized
        self.mark_modified_tv(tbl_version.handle)
        fault_injection.process_fault(FaultLocation.CATALOG_LOAD_TBL_VERSION_BEFORE_INIT)
        tbl_version.init()
        return tbl_version

    def _init_store(self) -> None:
        """One-time initialization of the stored catalog. Idempotent."""
        self.create_user(None)
        _logger.info('Initialized catalog.')

    def create_user(self, user: str | None) -> None:
        """
        Creates a catalog record (root directory) for the specified user, if one does not already exist.
        """
        # - we need to run this as SERIALIZABLE in order to avoid a race when two processes are started against
        #   an empty store (they both see a count of 0 for the root dir and both create a new root dir)
        # - this can only be run inside a new transaction, to ensure the isolation level
        # - we don't use retry_loop() here because this is called from Catalog.__init__() (via _init_store()),
        #   before the Catalog instance is registered with Runtime; retry_loop() would re-enter Catalog().
        assert not get_runtime().in_xact, 'create_user() must run as the outermost transaction'

        num_retries = 0
        while True:
            try:
                with get_runtime().begin_xact(for_write=True, isolation_level='SERIALIZABLE'):
                    session = get_runtime().session
                    assert session is not None
                    # See if there are any directories in the catalog matching the specified user.
                    if session.query(schema.Dir).where(schema.Dir.md['user'].astext == user).count() > 0:
                        # At least one such directory exists; no need to create a new one.
                        return
                    fault_injection.process_fault(FaultLocation.CATALOG_CREATE_USER_AFTER_EXISTS_CHECK)

                    dir_md = schema.DirMd(name='', user=user, additional_md={})
                    dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
                    session.add(dir_record)
                    session.flush()
                    _logger.info(f'Added root directory record for user: {user!r}')
                    return
            except sql_exc.DBAPIError as e:
                if not _is_retryable_exc(e):
                    raise
                if _MAX_RETRIES != -1 and num_retries >= _MAX_RETRIES:
                    raise excs.ConcurrencyError(
                        excs.ErrorCode.SERIALIZATION_FAILURE, f'Serialization retry limit ({_MAX_RETRIES}) exceeded'
                    ) from e
                num_retries += 1
                _logger.debug(f'Retrying ({num_retries}) after {type(e.orig)}')
                time.sleep(random.uniform(0.1, 0.5))

    def _handle_path_collision(
        self,
        path: Path,
        expected_obj_type: type[SchemaObject],
        expected_snapshot: bool,
        if_exists: IfExistsParam,
        *,
        base: TableVersionPath | None = None,
    ) -> SchemaObject | None:
        obj, _, _ = self._prepare_dir_op(add_dir_path=path.parent, add_name=path.name)

        if obj is None:
            return None

        # IfExistsParam.ERROR: Error given there is existing schema object
        if if_exists == IfExistsParam.ERROR:
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {path!r} is an existing {obj._display_name()}'
            )

        # IfExistsParam.IGNORE: Return existing object if it matches expected type (and base table for views/snapshots)
        if if_exists == IfExistsParam.IGNORE:
            # for ignore, we can only return the existing object if it matches the expected type
            is_existing_snapshot = isinstance(obj, View) and obj._tbl_version_path.is_snapshot()
            if not isinstance(obj, expected_obj_type) or (expected_snapshot and not is_existing_snapshot):
                if expected_obj_type is Dir:
                    obj_type_str = 'directory'
                elif expected_obj_type is InsertableTable:
                    obj_type_str = 'table'
                elif expected_obj_type is View:
                    obj_type_str = 'snapshot' if expected_snapshot else 'view'
                else:
                    obj_type_str = expected_obj_type.__name__
                raise excs.AlreadyExistsError(
                    excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {path!r} already exists and is not a {obj_type_str}'
                )
            # for views/snapshots, verify the base table matches
            if isinstance(obj, View):
                obj_base = obj._base_tbl_id
                new_base = base.tbl_id if base is not None else None
                if obj_base != new_base:
                    obj_type_str = 'snapshot' if expected_snapshot else 'view'
                    raise excs.AlreadyExistsError(
                        excs.ErrorCode.PATH_ALREADY_EXISTS,
                        f'Path {path!r} already exists as a {obj_type_str} with a different base table',
                    )
            return obj

        # IfExistsParam.REPLACE or IfExistsParam.REPLACE_FORCE
        assert if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)

        # check to ensure that dirs can only be replaced with dirs, and all table subtypes can replace each other
        if expected_obj_type == Dir and not isinstance(obj, Dir):
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS,
                f'Path {path!r} already exists as a {obj._display_name()}; expected a directory',
            )
        if expected_obj_type != Dir and isinstance(obj, Dir):
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS,
                f'Path {path!r} already exists as a directory; expected a table, view or snapshot',
            )

        # Check for circularity
        if base is not None:
            assert isinstance(obj, LocalTable)  # or else it would have been caught above
            if obj._id in tuple(version.id for version in base.get_tbl_versions()):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    "Cannot use if_exists='replace' with the same name as one of the view's own ancestors.",
                )

        # drop the existing schema object
        if isinstance(obj, Dir):
            dir_contents = self._get_dir_contents(obj._id)
            if len(dir_contents) > 0 and if_exists == IfExistsParam.REPLACE:
                raise excs.AlreadyExistsError(
                    excs.ErrorCode.PATH_ALREADY_EXISTS,
                    f'Directory {path!r} already exists and is not empty. '
                    'Use `if_exists="replace_force"` to replace it.',
                )
            self._drop_dir(obj._id, path, force=True)
        else:
            assert isinstance(obj, LocalTable)
            self._drop_tbl(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
        return None

    def validate_store(self) -> None:
        """Validate the underlying store for testing purposes.
        This function can and should be extended to perform more checks.
        """
        # Some tests intentionally cause warnings (e.g. UDF is gone). Ignore those warnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', excs.PixeltableWarning)
            all_contents = self.get_dir_contents(ROOT_PATH, recursive=True)
            with self.begin_read_md_xact(), self._allow_tbl_md_read():
                for entry in all_contents.values():
                    if entry.table is None:
                        continue
                    id = entry.table.id
                    tbl = self.get_table_by_id(id)
                    assert tbl is not None, id
                    self._validate_table(tbl)

    def _validate_table(self, tbl: pxt.Table) -> None:
        if tbl._tbl_version is None:
            return
        tv = tbl._tbl_version.get()
        sa_tbl = tv.store_tbl.sa_tbl

        # Validate that the Btree index value columns are in sync with the actual columns for latest version rows
        # Example query:
        # SELECT *,
        #        tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_0 !=
        #        tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_3 AS idx_mismatch_idx0,
        #        LEFT(tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_1, 256) !=
        #        tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_5 AS idx_mismatch_idx1,
        #        LEFT(tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_2, 256) !=
        #        tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_7 AS idx_mismatch_idx2
        # FROM   tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2
        # WHERE  tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.v_max > 22
        #        AND ( tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_0 !=
        #                    tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_3
        #               OR LEFT(tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_1, 256) !=
        #                  tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_5
        #               OR LEFT(tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_2, 256) !=
        #                  tbl_b58cafd251c84eb4ab5a13ff6c0f9bd2.col_7 )
        # LIMIT 1;
        select_list: list[sql.ColumnElement | Literal['*']] = ['*']
        conditions: list[sql.ColumnElement] = []
        for idx_info in tv.idxs.values():
            if isinstance(idx_info.idx, index.BtreeIndex) and idx_info.val_col is not None:
                # condition is the invariant violation that we are checking for
                # add it to where clause, and also to select clause for easier debugging
                if idx_info.col.col_type.is_string_type():
                    condition = (
                        sql.func.left(idx_info.col.sa_col, index.BtreeIndex.MAX_STRING_LEN) != idx_info.val_col.sa_col
                    )
                else:
                    condition = idx_info.col.sa_col != idx_info.val_col.sa_col
                conditions.append(condition)
                select_label = f'idx_mismatch_{idx_info.name}'
                select_list.append(condition.label(select_label))

        if len(conditions) > 0:
            # The v_max check (sa_tbl.c.v_max > tv.version) selects "active" rows: their v_max is
            # schema.Table.MAX_VERSION.
            stmt = (
                sql.select(*select_list)
                .select_from(sa_tbl)
                .where((sa_tbl.c.v_max > tv.version) if tv.is_data_versioned else sql.true())
                .where(sql.or_(*conditions))
                .limit(1)
            )
            _logger.info(f'Running index value column validation query on {tbl._display_str()}: {stmt}')
            for row in get_runtime().conn.execute(stmt).all():
                raise AssertionError(
                    f'The table validation query should have returned nothing, but it returned row: {row._asdict()}.\n'
                    f'This means that one of the indexes in {tbl._display_str()} is corrupted, i.e. the index value '
                    'is out of sync with the actual value for a current row. Look for idx_mismatch_*. The query was:\n'
                    f'{stmt}'
                )

        if tv.is_data_versioned:
            # Validate that the index values are NULL for non-latest version rows
            # Example query:
            # SELECT *,
            #        tbl_1d7bb633b5be4c57bd9070707ca4c552.col_3 IS NOT NULL  AS
            #        idx_not_null_idx0,
            #        tbl_1d7bb633b5be4c57bd9070707ca4c552.col_5 IS NOT NULL  AS
            #        idx_not_null_idx1,
            #        tbl_1d7bb633b5be4c57bd9070707ca4c552.col_7 IS NOT NULL  AS
            #        idx_not_null_idx2,
            #        tbl_1d7bb633b5be4c57bd9070707ca4c552.col_11 IS NOT NULL AS
            #        idx_not_null_img_idx2,
            #        tbl_1d7bb633b5be4c57bd9070707ca4c552.col_13 IS NOT NULL AS
            #        idx_not_null_img_idx1
            # FROM   tbl_1d7bb633b5be4c57bd9070707ca4c552
            # WHERE  tbl_1d7bb633b5be4c57bd9070707ca4c552.v_max <= 22
            #        AND ( tbl_1d7bb633b5be4c57bd9070707ca4c552.col_3 IS NOT NULL
            #               OR tbl_1d7bb633b5be4c57bd9070707ca4c552.col_5 IS NOT NULL
            #               OR tbl_1d7bb633b5be4c57bd9070707ca4c552.col_7 IS NOT NULL
            #               OR tbl_1d7bb633b5be4c57bd9070707ca4c552.col_11 IS NOT NULL
            #               OR tbl_1d7bb633b5be4c57bd9070707ca4c552.col_13 IS NOT NULL )
            # LIMIT 1;
            select_list.clear()
            select_list.append('*')
            conditions.clear()
            for idx_info in tv.idxs.values():
                if idx_info.val_col is None:
                    continue
                # condition is the invariant violation that we are checking for
                # add it to where clause, and also to select clause for easier debugging
                condition = idx_info.val_col.sa_col != None
                conditions.append(condition)
                select_label = f'idx_not_null_{idx_info.name}'
                select_list.append(condition.label(select_label))
            if len(conditions) > 0:
                stmt = (
                    sql.select(*select_list)
                    .select_from(sa_tbl)
                    .where(sa_tbl.c.v_max <= tv.version)
                    .where(sql.or_(*conditions))
                    .limit(1)
                )
                _logger.info(f'Running index value column validation query on {tbl._display_str()}: {stmt}')
                for row in get_runtime().conn.execute(stmt).all():
                    raise AssertionError(
                        f'The table validation query should have returned nothing, but it returned row: '
                        f'{row._asdict()}.\nThis means that one of the indexes in {tbl._display_str()} is corrupted, '
                        f'i.e. the index value is not NULL for a non-latest version row. Look for idx_not_null_*. '
                        f'The query was:\n{stmt}'
                    )
