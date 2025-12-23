from __future__ import annotations

import dataclasses
import functools
import logging
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar
from uuid import UUID

import psycopg
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.exception_handler import run_cleanup

from .column import Column
from .dir import Dir
from .globals import IfExistsParam, IfNotExistsParam, MediaValidation, QColumnId
from .insertable_table import InsertableTable
from .path import Path
from .schema_object import SchemaObject
from .table import Table
from .table_version import TableVersion, TableVersionKey, TableVersionMd
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .tbl_ops import TableOp
from .update_status import UpdateStatus
from .view import View

if TYPE_CHECKING:
    from pixeltable.plan import SampleClause

    from .. import exprs


_logger = logging.getLogger('pixeltable')


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


def md_dict_factory(data: list[tuple[str, Any]]) -> dict:
    """Use this to serialize TableMd instances with asdict()"""
    # serialize enums to their values
    return {k: v.value if isinstance(v, Enum) else v for k, v in data}


# -1: unlimited
# for now, we don't limit the number of retries, because we haven't seen situations where the actual number of retries
# grows uncontrollably
_MAX_RETRIES = -1

T = TypeVar('T')


def retry_loop(
    *, tbl: TableVersionPath | None = None, for_write: bool, lock_mutable_tree: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(op: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(op)
        def loop(*args: Any, **kwargs: Any) -> T:
            cat = Catalog.get()
            # retry_loop() is reentrant
            if cat._in_retry_loop:
                return op(*args, **kwargs)

            num_retries = 0
            while True:
                cat._in_retry_loop = True
                try:
                    # in order for retry to work, we need to make sure that there aren't any prior db updates
                    # that are part of an ongoing transaction
                    assert not Env.get().in_xact
                    with Catalog.get().begin_xact(
                        tbl=tbl,
                        for_write=for_write,
                        convert_db_excs=False,
                        lock_mutable_tree=lock_mutable_tree,
                        finalize_pending_ops=True,
                    ):
                        return op(*args, **kwargs)
                except PendingTableOpsError as e:
                    Env.get().console_logger.debug(f'retry_loop(): finalizing pending ops for {e.tbl_id}')
                    Catalog.get()._finalize_pending_ops(e.tbl_id)
                except (sql_exc.DBAPIError, sql_exc.OperationalError) as e:
                    # TODO: what other exceptions should we be looking for?
                    if isinstance(
                        # TODO: Investigate whether DeadlockDetected points to a bug in our locking protocol,
                        #     which is supposed to be deadlock-free.
                        e.orig,
                        (
                            psycopg.errors.SerializationFailure,
                            psycopg.errors.LockNotAvailable,
                            psycopg.errors.DeadlockDetected,
                        ),
                    ):
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
                finally:
                    cat._in_retry_loop = False

        return loop

    return decorator


class PendingTableOpsError(Exception):
    tbl_id: UUID

    def __init__(self, tbl_id: UUID) -> None:
        self.tbl_id = tbl_id


class Catalog:
    """The functional interface to getting access to catalog objects

    All interface functions must be called in the context of a transaction, started with Catalog.begin_xact() or
    via retry_loop().

    When calling functions that involve Table or TableVersion instances, the catalog needs to get a chance to finalize
    pending ops against those tables. To that end,
    - use begin_xact(tbl) or begin_xact(tbl_id) if only accessing a single table
    - use retry_loop() when accessing multiple tables (eg, pxt.ls())

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

    _instance: Catalog | None = None

    # cached TableVersion instances; key: [id, version, anchor_tbl_id]
    # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
    # - snapshot versions: records the version of the snapshot
    # - anchored versions: records the tbl_id of the anchor table (used when the table is a replica)
    _tbl_versions: dict[TableVersionKey, TableVersion]
    _tbls: dict[tuple[UUID, int | None], Table]
    _in_write_xact: bool  # True if we're in a write transaction
    _x_locked_tbl_ids: set[UUID]  # non-empty for write transactions
    _modified_tvs: set[TableVersionHandle]  # TableVersion instances modified in the current transaction
    _roll_forward_ids: set[UUID]  # ids of Tables that have pending TableOps
    _undo_actions: list[Callable[[], None]]
    _in_retry_loop: bool

    # cached column dependencies
    # - key: table id, value: mapping from column id to its dependencies
    # - only maintained for dependencies between non-snapshot table versions
    # - can contain stale entries (stemming from invalidated TV instances)
    _column_dependencies: dict[UUID, dict[QColumnId, set[QColumnId]]]

    # column dependents are recomputed at the beginning of every write transaction and only reflect the locked tree
    _column_dependents: dict[QColumnId, set[QColumnId]] | None

    @classmethod
    def get(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Remove the instance. Used for testing."""
        if cls._instance is not None:
            # invalidate all existing instances to force reloading of metadata
            for tbl_version in cls._instance._tbl_versions.values():
                tbl_version.is_validated = False
        cls._instance = None

    def __init__(self) -> None:
        self._tbl_versions = {}
        self._tbls = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self._in_write_xact = False
        self._x_locked_tbl_ids = set()
        self._modified_tvs = set()
        self._roll_forward_ids = set()
        self._undo_actions = []
        self._in_retry_loop = False
        self._column_dependencies = {}
        self._column_dependents = None
        self._init_store()

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

    def _dropped_tbl_error_msg(self, tbl_id: UUID) -> str:
        return f'Table was dropped (no record found for {tbl_id})'

    def validate(self) -> None:
        """Validate structural consistency of cached metadata"""
        for (tbl_id, effective_version, anchor_tbl_id), tbl_version in self._tbl_versions.items():
            assert tbl_id == tbl_version.id, f'{tbl_id} != {tbl_version.id}'
            assert effective_version is None or anchor_tbl_id is None
            assert tbl_version.effective_version == tbl_version.version or tbl_version.effective_version is None, (
                f'{tbl_version.effective_version} != {tbl_version.version} for id {tbl_id}'
            )
            assert effective_version == tbl_version.effective_version, (
                f'{effective_version} != {tbl_version.effective_version} for id {tbl_id}'
            )
            assert len(tbl_version.mutable_views) == 0 or tbl_version.is_mutable, (
                f'snapshot_id={tbl_version.id} mutable_views={tbl_version.mutable_views}'
            )

            assert anchor_tbl_id is None or tbl_version.is_replica

            if tbl_version.is_view and tbl_version.is_mutable and tbl_version.is_validated:
                # make sure this mutable view is recorded in a mutable base
                base = tbl_version.base
                assert base is not None
                if base.effective_version is None:
                    key = TableVersionKey(base.id, None, None)
                    assert key in self._tbl_versions
                    base_tv = self._tbl_versions[key]
                    if not base_tv.is_validated:
                        continue
                    mutable_view_ids = ', '.join(str(tv.id) for tv in self._tbl_versions[key].mutable_views)
                    mutable_view_names = ', '.join(
                        tv._tbl_version.name
                        for tv in self._tbl_versions[key].mutable_views
                        if tv._tbl_version is not None
                    )
                    assert tbl_version.handle in self._tbl_versions[key].mutable_views, (
                        f'{tbl_version.name} ({tbl_version.id}) missing in {mutable_view_ids} ({mutable_view_names})'
                    )

            if len(tbl_version.mutable_views) > 0:
                # make sure we also loaded mutable view metadata, which is needed to detect column dependencies
                for v in tbl_version.mutable_views:
                    assert v.effective_version is None, f'{v.id}:{v.effective_version}'

    def mark_modified_tvs(self, *handle: TableVersionHandle) -> None:
        """Record that the given TableVersion instances were modified in the current transaction"""
        assert Env.get().in_xact
        self._modified_tvs.update(handle)

    @contextmanager
    def begin_xact(
        self,
        *,
        tbl: TableVersionPath | None = None,
        tbl_id: UUID | None = None,
        for_write: bool = False,
        lock_mutable_tree: bool = False,
        convert_db_excs: bool = True,
        finalize_pending_ops: bool = True,
    ) -> Iterator[sql.Connection]:
        """
        Return a context manager that yields a connection to the database. Idempotent.

        It is mandatory to call this method, not Env.begin_xact(), if the transaction accesses any table data
        or metadata.

        If tbl != None, follows this locking protocol:
        - validates/reloads the TableVersion instances of tbl's ancestors (in the hope that this reduces potential
          SerializationErrors later on)
        - if for_write == True, x-locks Table record (by updating Table.lock_dummy; see _acquire_tbl_lock())
        - if for_write == False, validates TableVersion instance
        - if lock_mutable_tree == True, also x-locks all mutable views of the table
        - this needs to be done in a retry loop, because Postgres can decide to abort the transaction
          (SerializationFailure, LockNotAvailable)
        - for that reason, we do all lock acquisition prior to doing any real work (eg, compute column values),
          to minimize the probability of losing that work due to a forced abort

        If convert_db_excs == True, converts DBAPIErrors into excs.Errors.
        """
        assert tbl is None or tbl_id is None  # at most one can be specified
        if Env.get().in_xact:
            # make sure that we requested the required table lock at the beginning of the transaction
            if for_write:
                if tbl is not None:
                    assert tbl.tbl_id in self._x_locked_tbl_ids, f'{tbl.tbl_id} not in {self._x_locked_tbl_ids}'
                elif tbl_id is not None:
                    assert tbl_id in self._x_locked_tbl_ids, f'{tbl_id} not in {self._x_locked_tbl_ids}'
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
        pending_ops_tbl_id: UUID | None = None
        has_exc = False  # True if we exited the 'with ...begin_xact()' block with an exception
        while True:
            if pending_ops_tbl_id is not None:
                Env.get().console_logger.debug(f'begin_xact(): finalizing pending ops for {pending_ops_tbl_id}')
                self._finalize_pending_ops(pending_ops_tbl_id)
                pending_ops_tbl_id = None

            try:
                self._in_write_xact = for_write
                self._x_locked_tbl_ids = set()
                self._modified_tvs = set()
                self._column_dependents = None
                has_exc = False

                assert not self._undo_actions
                with Env.get().begin_xact(for_write=for_write) as conn:
                    if tbl is not None or tbl_id is not None:
                        try:
                            target: TableVersionHandle | None = None
                            if tbl is not None:
                                if self._acquire_path_locks(
                                    tbl=tbl,
                                    for_write=for_write,
                                    lock_mutable_tree=lock_mutable_tree,
                                    check_pending_ops=finalize_pending_ops,
                                ):
                                    target = tbl.tbl_version
                            else:
                                target = self._acquire_tbl_lock(
                                    tbl_id=tbl_id,
                                    for_write=for_write,
                                    lock_mutable_tree=lock_mutable_tree,
                                    raise_if_not_exists=True,
                                    check_pending_ops=finalize_pending_ops,
                                )

                            if target is None:
                                # didn't get the write lock
                                for_write = False
                            elif for_write:
                                # we know at this point that target is mutable because we got the X-lock
                                if lock_mutable_tree and not target.is_snapshot:
                                    self._x_locked_tbl_ids = self._get_mutable_tree(target.id)
                                    self._compute_column_dependents(self._x_locked_tbl_ids)
                                else:
                                    self._x_locked_tbl_ids = {target.id}
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

                        except (sql_exc.DBAPIError, sql_exc.OperationalError) as e:
                            has_exc = True
                            if isinstance(
                                e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)
                            ) and (num_retries < _MAX_RETRIES or _MAX_RETRIES == -1):
                                num_retries += 1
                                _logger.debug(f'Retrying ({num_retries}) after {type(e.orig)}')
                                time.sleep(random.uniform(0.1, 0.5))
                                assert not self._undo_actions  # We should not have any undo actions at this point
                                continue
                            else:
                                raise

                    assert not self._undo_actions
                    yield conn
                    return

            except PendingTableOpsError:
                has_exc = True
                if pending_ops_tbl_id is not None:
                    # the next iteration of the loop will deal with pending ops for this table id
                    continue
                else:
                    # we got this exception after getting the initial table locks and therefore need to abort
                    raise

            except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
                has_exc = True
                self.convert_sql_exc(e, tbl_id, tbl.tbl_version if tbl is not None else None, convert_db_excs)
                raise  # re-raise the error if it didn't convert to a pxt.Error

            except (Exception, KeyboardInterrupt) as e:
                has_exc = True
                _logger.debug(f'Caught {e.__class__}')
                raise

            finally:
                self._in_write_xact = False
                self._x_locked_tbl_ids.clear()
                self._column_dependents = None

                # invalidate cached current TableVersion instances
                for tv in self._tbl_versions.values():
                    if tv.effective_version is None:
                        _logger.debug(f'invalidating table version {tv} (0x{id(tv):x})')
                        tv.is_validated = False

                if has_exc:
                    # Execute undo actions in reverse order (LIFO)
                    for hook in reversed(self._undo_actions):
                        run_cleanup(hook, raise_error=False)
                    # purge all modified TableVersion instances; we can't guarantee they are still consistent with the
                    # stored metadata
                    for handle in self._modified_tvs:
                        self._clear_tv_cache(handle.key)
                    # Clear potentially corrupted cached metadata
                    if tbl is not None:
                        tbl.clear_cached_md()

                self._undo_actions.clear()
                self._modified_tvs.clear()

    def register_undo_action(self, func: Callable[[], None]) -> Callable[[], None]:
        """Registers a function to be called if the current transaction fails.

        The function is called only if the current transaction fails due to an exception.

        Rollback functions are called in reverse order of registration (LIFO).

        The function should not raise exceptions; if it does, they are logged and ignored.
        """
        assert self.in_write_xact
        self._undo_actions.append(func)
        return func

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
        if isinstance(e.orig, psycopg.errors.UndefinedTable) and tbl is not None:
            # the table got dropped in the middle of the operation
            tbl_name = tbl.get().name
            _logger.debug(f'Exception: undefined table {tbl_name!r}: Caught {type(e.orig)}: {e!r}')
            raise excs.Error(f'Table was dropped: {tbl_name}') from None
        elif (
            # TODO: Investigate whether DeadlockDetected points to a bug in our locking protocol,
            #     which is supposed to be deadlock-free.
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
            raise excs.Error(
                'That Pixeltable operation could not be completed because it conflicted with another '
                'operation that was run on a different process.\n'
                'Please re-run the operation.'
            ) from raise_from

    @property
    def in_write_xact(self) -> bool:
        return self._in_write_xact

    def _acquire_path_locks(
        self,
        *,
        tbl: TableVersionPath,
        for_write: bool = False,
        lock_mutable_tree: bool = False,
        check_pending_ops: bool = True,
    ) -> bool:
        """
        Path locking protocol:
        - refresh cached TableVersions of ancestors (we need those even during inserts, for computed columns that
          reference the base tables)
        - refresh cached TableVersion of tbl or get X-lock, depending on for_write
        - if lock_mutable_tree, also X-lock all mutable views of tbl

        Raises Error if tbl doesn't exist.
        Return False if the lock couldn't be acquired (X-lock on a non-mutable table), True otherwise.
        """
        path_handles = tbl.get_tbl_versions()
        read_handles = path_handles[:0:-1] if for_write else path_handles[::-1]
        for handle in read_handles:
            # update cache
            _ = self.get_tbl_version(handle.key, validate_initialized=True)
        if not for_write:
            return True  # nothing left to lock
        handle = self._acquire_tbl_lock(
            tbl_id=tbl.tbl_id,
            for_write=True,
            lock_mutable_tree=lock_mutable_tree,
            raise_if_not_exists=True,
            check_pending_ops=check_pending_ops,
        )
        # update cache
        _ = self.get_tbl_version(path_handles[0].key, validate_initialized=True)
        return handle is not None

    def _acquire_tbl_lock(
        self,
        *,
        for_write: bool,
        tbl_id: UUID | None = None,
        dir_id: UUID | None = None,
        tbl_name: str | None = None,
        lock_mutable_tree: bool = False,
        raise_if_not_exists: bool = True,
        check_pending_ops: bool = True,
    ) -> TableVersionHandle | None:
        """
        For writes: force acquisition of an X-lock on a Table record via a blind update.

        Either tbl_id or dir_id/tbl_name need to be specified.
        Returns True if the table was locked, False if it was a snapshot or not found.
        If lock_mutable_tree, recursively locks all mutable views of the table.

        Returns a handle to what was locked, None if the lock couldn't be acquired (eg, X-lock on a non-mutable table).
        """
        assert (tbl_id is not None) != (dir_id is not None and tbl_name is not None)
        assert (dir_id is None) == (tbl_name is None)
        where_clause: sql.ColumnElement
        if tbl_id is not None:
            where_clause = schema.Table.id == tbl_id
        else:
            where_clause = sql.and_(schema.Table.dir_id == dir_id, schema.Table.md['name'].astext == tbl_name)
            user = Env.get().user
            if user is not None:
                where_clause = sql.and_(where_clause, schema.Table.md['user'].astext == Env.get().user)

        conn = Env.get().conn
        q = sql.select(schema.Table).where(where_clause)
        if for_write:
            q = q.with_for_update(nowait=True)
        row = conn.execute(q).one_or_none()
        if row is None:
            if raise_if_not_exists:
                raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
            return None  # nothing to lock
        tbl_md = schema.md_from_dict(schema.TableMd, row.md)
        if for_write and tbl_md.is_mutable:
            conn.execute(sql.update(schema.Table).values(lock_dummy=1).where(where_clause))

        if check_pending_ops:
            # check for pending ops after getting table lock
            pending_ops_q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == row.id)
            has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
            if has_pending_ops:
                raise PendingTableOpsError(row.id)

        # TODO: properly handle concurrency for replicas with live views (once they are supported)
        if for_write and not tbl_md.is_mutable:
            return None  # nothing to lock

        key = TableVersionKey(tbl_id, tbl_md.current_version if tbl_md.is_snapshot else None, None)
        if tbl_md.is_mutable and lock_mutable_tree:
            # also lock mutable views
            tv = self.get_tbl_version(key, validate_initialized=True)
            for view in tv.mutable_views:
                self._acquire_tbl_lock(
                    for_write=for_write,
                    tbl_id=view.id,
                    lock_mutable_tree=lock_mutable_tree,
                    raise_if_not_exists=raise_if_not_exists,
                    check_pending_ops=check_pending_ops,
                )
        return TableVersionHandle(key)

    def _roll_forward(self) -> None:
        """Finalize pending ops for all tables in self._roll_forward_ids."""
        for tbl_id in self._roll_forward_ids:
            self._finalize_pending_ops(tbl_id)
            # TODO: handle replicas
            self._clear_tv_cache(TableVersionKey(tbl_id, None, None))

    def _finalize_pending_ops(self, tbl_id: UUID) -> None:
        """Finalizes all pending ops for the given table."""
        num_retries = 0
        while True:
            try:
                tbl_version: int
                op: TableOp | None = None
                delete_next_op_stmt: sql.Delete
                reset_state_stmt: sql.Update
                with self.begin_xact(
                    tbl_id=tbl_id, for_write=True, convert_db_excs=False, finalize_pending_ops=False
                ) as conn:
                    q = (
                        sql.select(schema.Table.md, schema.PendingTableOp)
                        .select_from(schema.Table)
                        .join(schema.PendingTableOp)
                        .where(schema.Table.id == tbl_id)
                        .where(schema.PendingTableOp.tbl_id == tbl_id)
                        .order_by(schema.PendingTableOp.op_sn)
                        .limit(1)
                        .with_for_update()
                    )
                    row = conn.execute(q).one_or_none()
                    if row is None:
                        return
                    view_md = row.md.get('view_md')
                    is_snapshot = False if view_md is None else view_md.get('is_snapshot')
                    assert is_snapshot is not None
                    tbl_version = row.md.get('current_version') if is_snapshot else None
                    op = schema.md_from_dict(TableOp, row.op)
                    delete_next_op_stmt = sql.delete(schema.PendingTableOp).where(
                        schema.PendingTableOp.tbl_id == tbl_id, schema.PendingTableOp.op_sn == row.op_sn
                    )
                    reset_state_stmt = (
                        sql.update(schema.Table)
                        .where(schema.Table.id == tbl_id)
                        .values(
                            md=schema.Table.md.op('||')(
                                {'tbl_state': schema.TableState.LIVE.value, 'pending_stmt': None}
                            )
                        )
                    )
                    _logger.debug(f'finalize_pending_ops({tbl_id}): finalizing op {op!s}')

                    if op.needs_xact:
                        if op.delete_table_md_op is not None:
                            self.delete_tbl_md(tbl_id)
                        else:
                            tv = self.get_tbl_version(
                                TableVersionKey(tbl_id, tbl_version, None),
                                check_pending_ops=False,
                                validate_initialized=True,
                            )
                            # TODO: The above TableVersionKey instance will need to be updated if we see a replica here.
                            # For now, just assert that we don't.
                            assert not tv.is_replica
                            tv.exec_op(op)

                        conn.execute(delete_next_op_stmt)
                        if op.op_sn == op.num_ops - 1:
                            conn.execute(reset_state_stmt)
                            return
                        continue

                # this op runs outside of a transaction
                tv = self.get_tbl_version(
                    TableVersionKey(tbl_id, tbl_version, None), check_pending_ops=False, validate_initialized=True
                )
                tv.exec_op(op)
                with self.begin_xact(
                    tbl_id=tbl_id, for_write=True, convert_db_excs=False, finalize_pending_ops=False
                ) as conn:
                    conn.execute(delete_next_op_stmt)
                    if op.op_sn == op.num_ops - 1:
                        conn.execute(reset_state_stmt)
                        return

            except (sql_exc.DBAPIError, sql_exc.OperationalError) as e:
                # TODO: why are we still seeing these here, instead of them getting taken care of by the retry
                # logic of begin_xact()?
                if isinstance(e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)):
                    num_retries += 1
                    log_msg: str
                    if op is not None:
                        log_msg = f'finalize_pending_ops(): retrying ({num_retries}) op {op!s} after {type(e.orig)}'
                    else:
                        log_msg = f'finalize_pending_ops(): retrying ({num_retries}) after {type(e.orig)}'
                    Env.get().console_logger.debug(log_msg)
                    time.sleep(random.uniform(0.1, 0.5))
                    continue
                else:
                    raise
            except Exception as e:
                Env.get().console_logger.debug(f'finalize_pending_ops(): caught {e}')
                raise

            num_retries = 0

    def _debug_str(self) -> str:
        tv_str = '\n'.join(str(k) for k in self._tbl_versions)
        tbl_str = '\n'.join(str(k) for k in self._tbls)
        return f'tbl_versions:\n{tv_str}\ntbls:\n{tbl_str}'

    def _get_mutable_tree(self, tbl_id: UUID) -> set[UUID]:
        """Returns ids of all tables that form the tree of mutable views starting at tbl_id; includes the root."""
        key = TableVersionKey(tbl_id, None, None)
        assert key in self._tbl_versions, f'{key} not in {self._tbl_versions.keys()}\n{self._debug_str()}'
        tv = self.get_tbl_version(key, validate_initialized=True)
        assert not tv.is_replica
        result: set[UUID] = {tv.id}
        for view in tv.mutable_views:
            result.update(self._get_mutable_tree(view.id))
        return result

    def _compute_column_dependents(self, mutable_tree: set[UUID]) -> None:
        """Populate self._column_dependents for all tables in mutable_tree"""
        assert self._column_dependents is None
        self._column_dependents = defaultdict(set)
        for tbl_id in mutable_tree:
            assert tbl_id in self._column_dependencies, (
                f'{tbl_id} not in {self._column_dependencies.keys()}\n{self._debug_str()}'
            )
            for col, dependencies in self._column_dependencies[tbl_id].items():
                for dependency in dependencies:
                    if dependency.tbl_id not in mutable_tree:
                        continue
                    dependents = self._column_dependents[dependency]
                    dependents.add(col)

    def record_column_dependencies(self, tbl_version: TableVersion) -> None:
        """Update self._column_dependencies. Only valid for mutable versions."""
        from pixeltable.exprs import Expr

        assert tbl_version.is_mutable
        dependencies: dict[QColumnId, set[QColumnId]] = {}
        for col in tbl_version.cols_by_id.values():
            if col.value_expr_dict is None:
                continue
            dependencies[QColumnId(tbl_version.id, col.id)] = Expr.get_refd_column_ids(col.value_expr_dict)
        self._column_dependencies[tbl_version.id] = dependencies

    def get_column_dependents(self, tbl_id: UUID, col_id: int) -> set[Column]:
        """Return all Columns that transitively depend on the given column."""
        assert self._column_dependents is not None
        dependents = self._column_dependents[QColumnId(tbl_id, col_id)]
        result: set[Column] = set()
        for dependent in dependents:
            tv = self.get_tbl_version(TableVersionKey(dependent.tbl_id, None, None), validate_initialized=True)
            col = tv.cols_by_id[dependent.col_id]
            result.add(col)
        return result

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
        Env.get().conn.execute(q)

    def get_dir_path(self, dir_id: UUID) -> Path:
        """Return path for directory with given id"""
        assert isinstance(dir_id, UUID)
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
        return Path.parse('.'.join(names), allow_empty_path=True, allow_system_path=True)

    @dataclasses.dataclass
    class DirEntry:
        dir: schema.Dir | None
        dir_entries: dict[str, Catalog.DirEntry]
        table: schema.Table | None

    @retry_loop(for_write=False)
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

        q = sql.select(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
        rows = conn.execute(q).all()
        for row in rows:
            tbl = schema.Table(**row._mapping)
            result[tbl.md['name']] = self.DirEntry(dir=None, dir_entries={}, table=tbl)

        return result

    @retry_loop(for_write=True)
    def move(self, path: Path, new_path: Path, if_exists: IfExistsParam, if_not_exists: IfNotExistsParam) -> None:
        self._move(path, new_path, if_exists, if_not_exists)

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
            src_obj._move(new_path.name, dest_dir._id)

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
        assert drop_expected in (None, Table, Dir), drop_expected
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
                    raise excs.Error(f'Directory {p!r} does not exist. Create it first with:\npxt.create_dir({p!r})')
                elif raise_if_not_exists:
                    raise excs.Error(f'Directory {p!r} does not exist.')
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
                raise excs.Error(f'Path {add_path!r} already exists.')

        drop_obj: SchemaObject | None = None
        if drop_dir is not None:
            drop_path = drop_dir_path.append(drop_name)
            drop_obj = self._get_dir_entry(drop_dir.id, drop_name, lock_entry=True)
            if drop_obj is None and raise_if_not_exists:
                raise excs.Error(f'Path {drop_path!r} does not exist.')
            if drop_obj is not None and drop_expected is not None and not isinstance(drop_obj, drop_expected):
                expected_name = 'table' if drop_expected is Table else 'directory'
                raise excs.Error(f'{drop_path!r} needs to be a {expected_name} but is a {drop_obj._display_name()}')

        add_dir_obj = Dir(add_dir.id, add_dir.parent_id, add_dir.md['name']) if add_dir is not None else None
        return add_obj, add_dir_obj, drop_obj

    def _get_dir_entry(
        self, dir_id: UUID, name: str, version: int | None = None, lock_entry: bool = False
    ) -> SchemaObject | None:
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
            self._acquire_tbl_lock(for_write=True, dir_id=dir_id, raise_if_not_exists=False, tbl_name=name)
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
        lock_obj: bool = False,
    ) -> SchemaObject | None:
        """Return the schema object at the given path, or None if it doesn't exist.

        Raises Error if
        - the parent directory doesn't exist
        - raise_if_exists is True and the path exists
        - raise_if_not_exists is True and the path does not exist
        - expected is not None and the existing object has a different type
        """
        assert expected in (None, Table, Dir), expected

        if path.is_root:
            # the root dir
            if expected is not None and expected is not Dir:
                raise excs.Error(f'{path!r} needs to be a table but is a dir')
            dir = self._get_dir(path, lock_dir=lock_obj)
            if dir is None:
                raise excs.Error(f'Unknown user: {Env.get().user}')
            return Dir(dir.id, dir.parent_id, dir.md['name'])

        parent_path = path.parent
        parent_dir = self._get_dir(parent_path, lock_dir=lock_parent)
        if parent_dir is None:
            if raise_if_not_exists:
                raise excs.Error(f'Directory {parent_path!r} does not exist.')
            else:
                return None
        obj = self._get_dir_entry(parent_dir.id, path.name, path.version, lock_entry=lock_obj)

        if obj is None and raise_if_not_exists:
            raise excs.Error(f'Path {path!r} does not exist.')
        elif obj is not None and raise_if_exists:
            raise excs.Error(f'Path {path!r} is an existing {obj._display_name()}.')
        elif obj is not None and expected is not None and not isinstance(obj, expected):
            expected_name = 'table' if expected is Table else 'directory'
            raise excs.Error(f'{path!r} needs to be a {expected_name} but is a {obj._display_name()}.')
        return obj

    def get_table_by_id(
        self, tbl_id: UUID, version: int | None = None, ignore_if_dropped: bool = False
    ) -> Table | None:
        """Must be executed inside a transaction. Might raise PendingTableOpsError."""
        if (tbl_id, version) not in self._tbls:
            if version is None:
                return self._load_tbl(tbl_id, ignore_pending_drop=ignore_if_dropped)
            else:
                return self._load_tbl_at_version(tbl_id, version)
        return self._tbls.get((tbl_id, version))

    def create_table(
        self,
        path: Path,
        schema: dict[str, Any],
        if_exists: IfExistsParam,
        primary_key: list[str] | None,
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        create_default_idxs: bool,
    ) -> tuple[Table, bool]:
        """
        Creates a new InsertableTable at the given path.

        If `if_exists == IfExistsParam.IGNORE` and a table `t` already exists at the given path, returns `t, False`.

        Otherwise, creates a new table `t` and returns `t, True` (or raises an exception if the operation fails).
        """

        @retry_loop(for_write=True)
        def create_fn() -> tuple[UUID, bool]:
            import pixeltable.metadata.schema

            existing = self._handle_path_collision(path, InsertableTable, False, if_exists)
            if existing is not None:
                assert isinstance(existing, Table)
                return existing._id, False

            dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
            assert dir is not None

            md, ops = InsertableTable._create(
                path.name,
                schema,
                primary_key=primary_key,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
                create_default_idxs=create_default_idxs,
            )
            tbl_id = UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = pixeltable.metadata.schema.TableStatement.CREATE_TABLE
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            return tbl_id, True

        self._roll_forward_ids.clear()
        tbl_id, is_created = create_fn()
        self._roll_forward()
        with self.begin_xact(tbl_id=tbl_id, for_write=True):
            tbl = self.get_table_by_id(tbl_id)
            _logger.info(f'Created table {tbl._name!r}, id={tbl._id}')
            Env.get().console_logger.info(f'Created table {tbl._name!r}.')
            return tbl, is_created

    def create_view(
        self,
        path: Path,
        base: TableVersionPath,
        select_list: list[tuple[exprs.Expr, str | None]] | None,
        where: exprs.Expr | None,
        sample_clause: 'SampleClause' | None,
        additional_columns: dict[str, Any] | None,
        is_snapshot: bool,
        create_default_idxs: bool,
        iterator: tuple[type[ComponentIterator], dict[str, Any]] | None,
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
    ) -> Table:
        @retry_loop(for_write=True)
        def create_fn() -> UUID:
            if not is_snapshot and base.is_mutable():
                # this is a mutable view of a mutable base; X-lock the base and advance its view_sn before adding
                # the view
                self._acquire_tbl_lock(tbl_id=base.tbl_id, for_write=True)
                base_tv = self.get_tbl_version(TableVersionKey(base.tbl_id, None, None), validate_initialized=True)
                base_tv.tbl_md.view_sn += 1
                result = Env.get().conn.execute(
                    sql.update(schema.Table)
                    .values({schema.Table.md: dataclasses.asdict(base_tv.tbl_md, dict_factory=md_dict_factory)})
                    .where(schema.Table.id == base.tbl_id)
                )
                assert result.rowcount == 1, result.rowcount

            existing = self._handle_path_collision(path, View, is_snapshot, if_exists, base=base)
            if existing is not None:
                assert isinstance(existing, View)
                return existing._id

            dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
            assert dir is not None
            if iterator is None:
                iterator_class, iterator_args = None, None
            else:
                iterator_class, iterator_args = iterator
            md, ops = View._create(
                dir._id,
                path.name,
                base=base,
                select_list=select_list,
                additional_columns=additional_columns,
                predicate=where,
                sample_clause=sample_clause,
                is_snapshot=is_snapshot,
                create_default_idxs=create_default_idxs,
                iterator_cls=iterator_class,
                iterator_args=iterator_args,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
            )
            tbl_id = UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = schema.TableStatement.CREATE_VIEW
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            return tbl_id

        self._roll_forward_ids.clear()
        view_id = create_fn()
        if not is_snapshot and base.is_mutable():
            # invalidate base's TableVersion instance, so that it gets reloaded with the new mutable view
            self._clear_tv_cache(base.tbl_version.key)
            # base_tv = self.get_tbl_version(base.tbl_id, base.tbl_version.effective_version, validate_initialized=True)
            # view_handle = TableVersionHandle(view_id, effective_version=None)
            # base_tv.mutable_views.add(view_handle)

        self._roll_forward()
        with self.begin_xact(tbl_id=view_id, for_write=True):
            return self.get_table_by_id(view_id)

    def _clear_tv_cache(self, key: TableVersionKey) -> None:
        if key in self._tbl_versions:
            tv = self._tbl_versions[key]
            tv.is_validated = False
            del self._tbl_versions[key]

    def create_replica(self, path: Path, md: list[TableVersionMd], create_store_tbls: bool = True) -> None:
        """
        Creates table, table_version, and table_schema_version records for a replica with the given metadata.
        The metadata should be presented in standard "ancestor order", with the table being replicated at
        list position 0 and the (root) base table at list position -1.
        """
        assert self.in_write_xact

        # Acquire locks for any tables in the ancestor hierarchy that might already exist (base table first).
        for ancestor_md in md[::-1]:  # base table first
            self._acquire_tbl_lock(for_write=True, tbl_id=UUID(ancestor_md.tbl_md.tbl_id), raise_if_not_exists=False)

        tbl_id = UUID(md[0].tbl_md.tbl_id)

        existing = self._handle_path_collision(path, Table, False, if_exists=IfExistsParam.IGNORE)  # type: ignore[type-abstract]
        if existing is not None and existing._id != tbl_id:
            raise excs.Error(
                f'An attempt was made to create a replica table at {path!r}, '
                'but a different table already exists at that location.'
            )

        # Ensure that the system directory exists.
        self.__ensure_system_dir_exists()

        # Now check to see if this table already exists in the catalog.
        existing = self.get_table_by_id(tbl_id)
        if existing is not None:
            existing_path = Path.parse(existing._path(), allow_system_path=True)
            if existing_path != path and not existing_path.is_system_path:
                # It does exist, under a different path from the specified one.
                raise excs.Error(
                    f'That table has already been replicated as {existing_path!r}.\n'
                    f'Drop the existing replica if you wish to re-create it.'
                )

        # Now store the metadata for this replica's proper ancestors. If one or more proper ancestors
        # do not yet exist in the store, they will be created as anonymous system tables.
        # We instantiate the ancestors starting with the base table and ending with the immediate parent of the
        # table being replicated.
        for ancestor_md in md[:0:-1]:
            ancestor_id = UUID(ancestor_md.tbl_md.tbl_id)
            replica = self.get_table_by_id(ancestor_id)
            replica_path: Path
            if replica is None:
                # We've never seen this table before. Create a new anonymous system table for it.
                replica_path = Path.parse(f'_system.replica_{ancestor_id.hex}', allow_system_path=True)
            else:
                # The table already exists in the catalog. The existing path might be a system path (if the table
                # was created as an anonymous base table of some other table), or it might not (if it's a snapshot
                # that was directly replicated by the user at some point). In either case, use the existing path.
                replica_path = Path.parse(replica._path(), allow_system_path=True)

            # Store the metadata; it could be a new version (in which case a new record will be created), or a known
            # version (in which case the newly received metadata will be validated as identical).
            # If it's a new version, this will result in a new TableVersion record being created.
            self.__store_replica_md(replica_path, ancestor_md)

            # Now we must clear cached metadata for the ancestor table, to force the next table operation to pick up
            # the new TableVersion instance. This is necessary because computed columns of descendant tables might
            # reference columns of the ancestor table that only exist in the new version.
            replica = self.get_table_by_id(ancestor_id)
            # assert replica is not None  # If it didn't exist before, it must have been created by now.
            if replica is not None:
                replica._tbl_version_path.clear_cached_md()

        # Store the metadata for the table being replicated; as before, it could be a new version or a known version.
        # If it's a new version, then a TableVersion record will be created, unless the table being replicated
        # is a pure snapshot.
        self.__store_replica_md(path, md[0], create_store_tbls)

        # Finally, it's possible that the table already exists in the catalog, but as an anonymous system table that
        # was hidden the last time we checked (and that just became visible when the replica was imported). In this
        # case, we need to make the existing table visible by moving it to the specified path.
        # We need to do this at the end, since `existing_path` needs to first have a non-fragment table version in
        # order to be instantiated as a schema object.
        existing = self.get_table_by_id(tbl_id)
        assert existing is not None
        existing_path = Path.parse(existing._path(), allow_system_path=True)
        if existing_path != path:
            assert existing_path.is_system_path
            self._move(existing_path, path, IfExistsParam.ERROR, IfNotExistsParam.ERROR)

    def __ensure_system_dir_exists(self) -> Dir:
        system_path = Path.parse('_system', allow_system_path=True)
        return self._create_dir(system_path, if_exists=IfExistsParam.IGNORE, parents=False)

    def __store_replica_md(self, path: Path, md: TableVersionMd, create_store_tbl: bool = True) -> None:
        _logger.info(f'Creating replica table at {path!r} with ID: {md.tbl_md.tbl_id}')
        dir = self._get_schema_object(path.parent, expected=Dir, raise_if_not_exists=True)
        assert dir is not None
        assert self._in_write_xact

        conn = Env.get().conn
        tbl_id = md.tbl_md.tbl_id

        new_tbl_md: schema.TableMd | None = None
        new_version_md: schema.VersionMd | None = None
        new_schema_version_md: schema.SchemaVersionMd | None = None
        is_new_tbl_version: bool = False

        # We need to ensure that the table metadata in the catalog always reflects the latest observed version of
        # this table. (In particular, if this is a base table, then its table metadata need to be consistent
        # with the latest version of this table having a replicated view somewhere in the catalog.)
        # TODO: handle concurrent drop() of an existing replica; if we just ignore that Table record here, we can end
        # up with a duplicate key violation; in principle, we should wait for the concurrent drop() to finish
        q: sql.Executable = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
        existing_md_row = conn.execute(q).one_or_none()

        # Update md with the given name, current user, and is_replica flag.
        md = dataclasses.replace(
            md, tbl_md=dataclasses.replace(md.tbl_md, name=path.name, user=Env.get().user, is_replica=True)
        )
        if existing_md_row is None:
            # No existing table, so create a new record.
            q = sql.insert(schema.Table.__table__).values(
                id=tbl_id, dir_id=dir._id, md=dataclasses.asdict(md.tbl_md, dict_factory=md_dict_factory)
            )
            conn.execute(q)
        elif not existing_md_row.md['is_replica']:
            raise excs.Error(
                'An attempt was made to replicate a view whose base table already exists in the local catalog '
                'in its original form.\n'
                'If this is intentional, you must first drop the existing base table:\n'
                f'  pxt.drop_table({str(path)!r})'
            )
        elif md.tbl_md.current_version > existing_md_row.md['current_version']:
            # New metadata is more recent than the metadata currently stored in the DB; we'll update the record
            # in place in the DB.
            new_tbl_md = md.tbl_md

        # Now see if a TableVersion record already exists in the DB for this table version. If not, insert it. If
        # it already exists, check that the existing record is identical to the new one.
        q = (
            sql.select(schema.TableVersion.md)
            .where(schema.TableVersion.tbl_id == tbl_id)
            .where(schema.TableVersion.md['version'].cast(sql.Integer) == md.version_md.version)
        )
        existing_version_md_row = conn.execute(q).one_or_none()
        if existing_version_md_row is None:
            new_version_md = md.version_md
            is_new_tbl_version = True
        else:
            existing_version_md = schema.md_from_dict(schema.VersionMd, existing_version_md_row.md)
            # Validate that the existing metadata are identical to the new metadata, except is_fragment
            # and additional_md which may differ.
            if (
                dataclasses.replace(
                    existing_version_md,
                    is_fragment=md.version_md.is_fragment,
                    additional_md=md.version_md.additional_md,
                )
                != md.version_md
            ):
                raise excs.Error(
                    f'The version metadata for the replica {path!r}:{md.version_md.version} is inconsistent with '
                    'the metadata recorded from a prior replica.\n'
                    'This is likely due to data corruption in the replicated table.'
                )
            if existing_version_md.is_fragment and not md.version_md.is_fragment:
                # This version exists in the DB as a fragment, but we're importing a complete copy of the same version;
                # set the is_fragment flag to False in the DB.
                new_version_md = md.version_md

        # Do the same thing for TableSchemaVersion.
        q = (
            sql.select(schema.TableSchemaVersion.md)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
            .where(
                schema.TableSchemaVersion.md['schema_version'].cast(sql.Integer) == md.schema_version_md.schema_version
            )
        )
        existing_schema_version_md_row = conn.execute(q).one_or_none()
        if existing_schema_version_md_row is None:
            new_schema_version_md = md.schema_version_md
        else:
            existing_schema_version_md = schema.md_from_dict(schema.SchemaVersionMd, existing_schema_version_md_row.md)
            # Validate that the existing metadata are identical to the new metadata.
            if existing_schema_version_md != md.schema_version_md:
                raise excs.Error(
                    f'The schema version metadata for the replica {path!r}:{md.schema_version_md.schema_version} '
                    'is inconsistent with the metadata recorded from a prior replica.\n'
                    'This is likely due to data corruption in the replicated table.'
                )

        self.write_tbl_md(UUID(tbl_id), None, new_tbl_md, new_version_md, new_schema_version_md)

        if is_new_tbl_version and not md.is_pure_snapshot:
            # It's a new version of a table that has a physical store, so we need to create a TableVersion instance.
            TableVersion.create_replica(md, create_store_tbl)

    def get_additional_md(self, tbl_id: UUID) -> dict[str, Any]:
        """Return the additional_md field of the given table."""
        assert Env.get().in_xact
        conn = Env.get().conn
        q = sql.select(schema.Table.additional_md).where(self._active_tbl_clause(tbl_id=tbl_id))
        # TODO: handle concurrent drop()
        row = conn.execute(q).one()
        assert isinstance(row[0], dict)
        return row[0]

    def update_additional_md(self, tbl_id: UUID, additional_md: dict[str, Any]) -> None:
        """
        Update the additional_md field of the given table. The new additional_md is merged with the
        existing one via a JSON dictionary merge, giving preference to the new values.
        """
        assert self._in_write_xact
        conn = Env.get().conn
        q = (
            sql.update(schema.Table)
            .where(schema.Table.id == str(tbl_id))
            .values({schema.Table.additional_md: schema.Table.additional_md.op('||')(additional_md)})
        )
        result = conn.execute(q)
        assert result.rowcount == 1, result.rowcount

    @retry_loop(for_write=False)
    def get_table(self, path: Path, if_not_exists: IfNotExistsParam) -> Table | None:
        obj = Catalog.get()._get_schema_object(
            path, expected=Table, raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR)
        )
        if obj is None:
            _logger.info(f'Skipped table {path!r} (does not exist).')
            return None

        assert isinstance(obj, Table)
        # We need to clear cached metadata from tbl_version_path, in case the schema has been changed
        # by another process.
        obj._tbl_version_path.clear_cached_md()
        return obj

    def drop_table(self, path: Path, if_not_exists: IfNotExistsParam, force: bool) -> None:
        @retry_loop(for_write=True)
        def drop_fn() -> None:
            tbl = self._get_schema_object(
                path,
                expected=Table,
                raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR and not force),
                lock_parent=True,
                lock_obj=False,
            )
            if tbl is None:
                _logger.info(f'Skipped table {path!r} (does not exist).')
                return
            assert isinstance(tbl, Table)

            if isinstance(tbl, View) and tbl._tbl_version_path.is_mutable() and tbl._tbl_version_path.base.is_mutable():
                # this is a mutable view of a mutable base;
                # lock the base before the view, in order to avoid deadlocks with concurrent inserts/updates
                base_id = tbl._tbl_version_path.base.tbl_id
                self._acquire_tbl_lock(tbl_id=base_id, for_write=True, lock_mutable_tree=False)

            self._drop_tbl(tbl, force=force, is_replace=False)

        self._roll_forward_ids.clear()
        drop_fn()
        self._roll_forward()

    def _drop_tbl(self, tbl: Table | TableVersionPath, force: bool, is_replace: bool) -> None:
        """
        Drop the table (and recursively its views, if force == True).

        `tbl` can be an instance of `Table` for a user table, or `TableVersionPath` for a hidden (system) table.

        Returns:
            List of table ids that were dropped.

        Locking protocol:
        - X-lock base before X-locking any view
        - deadlock-free wrt to TableVersion.insert() (insert propagation also proceeds top-down)
        - X-locks parent dir prior to calling TableVersion.drop(): prevent concurrent creation of another SchemaObject
          in the same directory with the same name (which could lead to duplicate names if we get aborted)
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

        if tbl is not None:
            self._acquire_dir_xlock(dir_id=tbl._dir_id)
        self._acquire_tbl_lock(tbl_id=tbl_id, for_write=True, lock_mutable_tree=False)

        view_ids = self.get_view_ids(tbl_id, for_update=True)
        is_replica = tvp.is_replica()
        do_drop = True

        _logger.debug(f'Preparing to drop table {tbl_id} (force={force!r}, is_replica={is_replica}).')

        if len(view_ids) > 0:
            if force:
                # recursively drop views first
                for view_id in view_ids:
                    view = self.get_table_by_id(view_id, ignore_if_dropped=True)
                    if view is not None:
                        self._drop_tbl(view, force=force, is_replace=is_replace)

            elif is_replica:
                # Dropping a replica with dependents and no 'force': just rename it to be a hidden table;
                # the actual table will not be dropped.
                assert tbl is not None  # can only occur for a user table
                system_dir = self.__ensure_system_dir_exists()
                new_name = f'replica_{tbl_id.hex}'
                _logger.debug(f'{tbl._path()!r} is a replica with dependents; renaming to {new_name!r}.')
                tbl._move(new_name, system_dir._id)
                do_drop = False  # don't actually clear the catalog for this table

            else:
                # It has dependents but is not a replica and no 'force', so it's an error to drop it.
                assert tbl is not None  # can only occur for a user table
                msg: str
                if is_replace:
                    msg = (
                        f'{tbl._display_str()} already exists and has dependents. '
                        "Use `if_exists='replace_force'` to replace it."
                    )
                else:
                    msg = f'{tbl._display_str()} has dependents.'
                raise excs.Error(msg)

        # if this is a mutable view of a mutable base, advance the base's view_sn
        if isinstance(tbl, View) and tvp.is_mutable() and tvp.base.is_mutable():
            base_id = tvp.base.tbl_id
            base_tv = self.get_tbl_version(TableVersionKey(base_id, None, None), validate_initialized=True)
            base_tv.tbl_md.view_sn += 1
            result = Env.get().conn.execute(
                sql.update(schema.Table.__table__)
                .values({schema.Table.md: dataclasses.asdict(base_tv.tbl_md, dict_factory=md_dict_factory)})
                .where(schema.Table.id == base_id)
            )
            assert result.rowcount == 1, result.rowcount
            # force reload of base TV instance in order to make its state consistent with the stored metadata
            self._clear_tv_cache(base_tv.key)

        if do_drop:
            if is_pure_snapshot:
                # there is no physical table, but we still need to delete the Table record; we can do that right now
                # as part of the current transaction
                self.delete_tbl_md(tbl_id)
            else:
                # invalidate the TableVersion instance when we're done so that existing references to it can find out it
                # has been dropped
                self.mark_modified_tvs(tvp.tbl_version)

                # write TableOps to execute the drop, plus the updated Table record
                tv = tvp.tbl_version.get()
                tv.tbl_md.pending_stmt = schema.TableStatement.DROP_TABLE
                drop_ops = tv.drop()
                self.write_tbl_md(
                    tv.id,
                    dir_id=None,
                    tbl_md=tv.tbl_md,
                    version_md=None,
                    schema_version_md=None,
                    pending_ops=drop_ops,
                    remove_from_dir=True,
                )

            tvp.clear_cached_md()

        assert (
            is_replica
            or (tbl_id, None) in self._tbls  # non-replica tables must have an entry with effective_version=None
        )

        # Remove visible Table references (we do this even for a replica that was just renamed).
        versions = [version for id, version in self._tbls if id == tbl_id]
        for version in versions:
            del self._tbls[tbl_id, version]

        _logger.info(f'Dropped table {tbl_id if tbl is None else repr(tbl._path())}.')

        if (
            is_replica  # if this is a replica,
            and do_drop  # and it was actually dropped (not just renamed),
            and tvp.base is not None  # and it has a base table,
        ):
            base_tbl = self.get_table_by_id(tvp.base.tbl_id)
            base_tbl_path = None if base_tbl is None else Path.parse(base_tbl._path(), allow_system_path=True)
            if (
                (base_tbl_path is None or base_tbl_path.is_system_path)  # and the base table is hidden,
                and len(self.get_view_ids(tvp.base.tbl_id, for_update=True)) == 0  # and has no other dependents,
            ):
                # then drop the base table as well (possibly recursively).
                _logger.debug(f'Dropping hidden base table {tvp.base.tbl_id} of dropped replica {tbl_id}.')
                # we just dropped the anchor on `tvp.base`; we need to clear the anchor so that we can actually
                # load the TableVersion instance in order to drop it
                self._drop_tbl(tvp.base.anchor_to(None), force=False, is_replace=False)

    @retry_loop(for_write=True)
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
        # Env.get().console_logger.info(f'Created directory {path!r}.')
        # return dir

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
        @retry_loop(for_write=True)
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

        self._roll_forward_ids.clear()
        drop_fn()
        self._roll_forward()

    def _drop_dir(self, dir_id: UUID, dir_path: Path, force: bool = False) -> None:
        conn = Env.get().conn
        if not force:
            # check for existing entries
            q = sql.select(sql.func.count()).select_from(schema.Dir).where(schema.Dir.parent_id == dir_id)
            num_subdirs = conn.execute(q).scalar()
            q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
            num_tbls = conn.execute(q).scalar()
            if num_subdirs + num_tbls > 0:
                raise excs.Error(f'Directory {dir_path!r} is not empty.')

        # drop existing subdirs
        self._acquire_dir_xlock(dir_id=dir_id)
        dir_q = sql.select(schema.Dir).where(schema.Dir.parent_id == dir_id)
        for row in conn.execute(dir_q).all():
            self._drop_dir(row.id, dir_path.append(row.md['name']), force=True)

        # drop existing tables
        tbl_q = sql.select(schema.Table).where(self._active_tbl_clause(dir_id=dir_id)).with_for_update()
        for row in conn.execute(tbl_q).all():
            tbl = self.get_table_by_id(row.id, ignore_if_dropped=True)
            # this table would have been dropped already if it's a view of a base we dropped earlier
            if tbl is not None:
                self._drop_tbl(tbl, force=True, is_replace=False)

        # self.drop_dir(dir_id)
        conn.execute(sql.delete(schema.Dir).where(schema.Dir.id == dir_id))
        _logger.info(f'Removed directory {dir_path!r}.')

    def get_view_ids(self, tbl_id: UUID, for_update: bool = False) -> list[UUID]:
        """Return the ids of views that directly reference the given table"""
        conn = Env.get().conn
        # check whether this table still exists
        q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(tbl_id=tbl_id))
        tbl_count = conn.execute(q).scalar()
        if tbl_count == 0:
            raise excs.Error(self._dropped_tbl_error_msg(tbl_id))
        q = (
            sql.select(schema.Table.id)
            .where(schema.Table.md['view_md']['base_versions'][0][0].astext == tbl_id.hex)
            .where(self._active_tbl_clause())
        )
        if for_update:
            q = q.with_for_update()
        result = [r[0] for r in conn.execute(q).all()]
        return result

    def get_tbl_version(
        self, key: TableVersionKey, *, check_pending_ops: bool = True, validate_initialized: bool = False
    ) -> TableVersion | None:
        """
        Returns the TableVersion instance for the given table and version and updates the cache.

        If present in the cache and the instance isn't validated, validates version and view_sn against the stored
        metadata.
        """
        # we need a transaction here, if we're not already in one; if this starts a new transaction,
        # the returned TableVersion instance will not be validated
        with self.begin_xact(for_write=False) as conn:
            tv = self._tbl_versions.get(key)
            if tv is None:
                tv = self._load_tbl_version(key, check_pending_ops=check_pending_ops)
            elif not tv.is_validated:
                # only live instances are invalidated
                assert key.effective_version is None
                # _logger.debug(f'validating metadata for table {tbl_id}:{tv.version} ({id(tv):x})')
                where_clause: sql.ColumnElement[bool]
                if check_pending_ops:
                    # if we don't want to see pending ops, we also don't want to see dropped tables
                    where_clause = self._active_tbl_clause(tbl_id=key.tbl_id)
                else:
                    where_clause = schema.Table.id == key.tbl_id
                q = sql.select(schema.Table.md).where(where_clause)
                row = conn.execute(q).one_or_none()
                if row is None:
                    raise excs.Error(self._dropped_tbl_error_msg(key.tbl_id))

                reload = False

                if tv.anchor_tbl_id is None:
                    # live non-replica table; compare our cached TableMd.current_version/view_sn to what's stored
                    q = sql.select(schema.Table.md).where(where_clause)
                    row = conn.execute(q).one_or_none()
                    if row is None:
                        raise excs.Error(self._dropped_tbl_error_msg(key.tbl_id))
                    current_version, view_sn = row.md['current_version'], row.md['view_sn']
                    if current_version != tv.version or view_sn != tv.tbl_md.view_sn:
                        _logger.debug(
                            f'reloading metadata for live table {key.tbl_id} '
                            f'(cached/current version: {tv.version}/{current_version}, '
                            f'cached/current view_sn: {tv.tbl_md.view_sn}/{view_sn})'
                        )
                        reload = True

                else:
                    # live replica table; use the anchored version
                    anchor_tbl_version_md = self.head_version_md(tv.anchor_tbl_id)
                    assert anchor_tbl_version_md is not None
                    q = sql.select(schema.TableVersion.md)
                    if check_pending_ops:
                        q = q.join(schema.Table, schema.Table.id == schema.TableVersion.tbl_id).where(
                            self._active_tbl_clause(tbl_id=key.tbl_id)
                        )
                    q = (
                        q.where(schema.TableVersion.tbl_id == key.tbl_id)
                        .where(schema.TableVersion.md['created_at'].cast(sql.Float) <= anchor_tbl_version_md.created_at)
                        .order_by(schema.TableVersion.md['created_at'].cast(sql.Float).desc())
                        .limit(1)
                    )
                    row = conn.execute(q).one_or_none()
                    if row is None:
                        raise excs.Error(self._dropped_tbl_error_msg(key.tbl_id))
                    version = row.md['version']
                    if version != tv.version:  # TODO: How will view_sn work for replicas?
                        _logger.debug(
                            f'reloading metadata for replica table {key.tbl_id} (anchor {key.anchor_tbl_id}) '
                            f'(cached/anchored version: {tv.version}/{version})'
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

            assert tv.anchor_tbl_id == key.anchor_tbl_id
            assert tv.is_validated, f'{key} not validated\n{tv.__dict__}\n{self._debug_str()}'
            if validate_initialized:
                assert tv.is_initialized, f'{key} not initialized\n{tv.__dict__}\n{self._debug_str()}'
            return tv

    def remove_tbl_version(self, key: TableVersionKey) -> None:
        assert isinstance(key, TableVersionKey)
        assert key in self._tbl_versions
        del self._tbl_versions[key]

    def get_dir(self, dir_id: UUID, for_update: bool = False) -> Dir | None:
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

    def _get_dir(self, path: Path, lock_dir: bool = False) -> schema.Dir | None:
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

    def _load_tbl(self, tbl_id: UUID, ignore_pending_drop: bool = False) -> Table | None:
        """Loads metadata for the table with the given id and caches it."""
        from .insertable_table import InsertableTable
        from .view import View

        assert tbl_id is not None
        _logger.info(f'Loading table {tbl_id}')

        conn = Env.get().conn

        if ignore_pending_drop:
            # check whether this table is in the process of being dropped
            q: sql.Executable = sql.select(schema.Table.md).where(schema.Table.id == tbl_id)
            row = conn.execute(q).one()
            if row.md['pending_stmt'] == schema.TableStatement.DROP_TABLE.value:
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
            return None
        tbl_record, _ = _unpack_row(row, [schema.Table, schema.TableSchemaVersion])

        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        view_md = tbl_md.view_md

        if view_md is None and not tbl_md.is_replica:
            # this is a base, non-replica table
            key = TableVersionKey(tbl_id, None, None)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)
            tbl = InsertableTable(tbl_record.dir_id, TableVersionHandle(key))
            self._tbls[tbl_id, None] = tbl
            return tbl

        # this is a view; determine the sequence of TableVersions to load
        tbl_version_path: list[tuple[UUID, int | None]] = []
        anchor_tbl_id = UUID(tbl_md.tbl_id) if tbl_md.is_replica else None
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

        if anchor_tbl_id is not None and self.head_version_md(anchor_tbl_id) is None:
            return None

        # load TableVersions, starting at the root
        base_path: TableVersionPath | None = None
        view_path: TableVersionPath | None = None
        for id, effective_version in tbl_version_path[::-1]:
            # anchor the path elements that have effective_version == None
            key = TableVersionKey(id, effective_version, None if effective_version is not None else anchor_tbl_id)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)
            view_path = TableVersionPath(TableVersionHandle(key), base=base_path)
            base_path = view_path
        view = View(tbl_id, tbl_record.dir_id, tbl_md.name, view_path, snapshot_only=tbl_md.is_pure_snapshot)
        self._tbls[tbl_id, None] = view
        return view

    def _load_tbl_at_version(self, tbl_id: UUID, version: int) -> Table | None:
        from .view import View

        # Load the specified TableMd and TableVersionMd records from the db.
        conn = Env.get().conn
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
        version_md = schema.md_from_dict(schema.VersionMd, version_record.md)
        tvp = self.construct_tvp(tbl_id, version, tbl_md.ancestors, version_md.created_at)

        view = View(tbl_id, tbl_record.dir_id, tbl_md.name, tvp, snapshot_only=True)
        self._tbls[tbl_id, version] = view
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

        assert Env.get().conn is not None

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
            row = Env.get().conn.execute(q).one_or_none()
            if row is None:
                # This can happen if an ancestor version is garbage collected; it can also happen in
                # rare circumstances involving table versions created specifically with Pixeltable 0.4.3.
                _logger.info(f'Ancestor {ancestor_id} not found for table {tbl_id}:{version}')
                raise excs.Error('The specified table version is no longer valid and cannot be retrieved.')
            ancestor_version_record = _unpack_row(row, [schema.TableVersion])[0]
            ancestor_version_md = schema.md_from_dict(schema.VersionMd, ancestor_version_record.md)
            assert ancestor_version_md.created_at <= created_at
            ancestors.append((UUID(ancestor_id), ancestor_version_md.version))

        # Force any ancestors to be loaded (base table first).
        for anc_id, anc_version in ancestors[::-1]:
            key = TableVersionKey(anc_id, anc_version, None)
            if key not in self._tbl_versions:
                _ = self._load_tbl_version(key)

        # Now reconstruct the relevant TableVersionPath instance from the ancestor versions.
        tvp: TableVersionPath | None = None
        for anc_id, anc_version in ancestors[::-1]:
            tvp = TableVersionPath(TableVersionHandle(TableVersionKey(anc_id, anc_version, None)), base=tvp)

        return tvp

    @retry_loop(for_write=False)
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
        src_rows = Env.get().session.execute(q).fetchall()
        return [
            TableVersionMd(
                tbl_md=schema.md_from_dict(schema.TableMd, row.Table.md),
                version_md=schema.md_from_dict(schema.VersionMd, row.TableVersion.md),
                schema_version_md=schema.md_from_dict(schema.SchemaVersionMd, row.TableSchemaVersion.md),
            )
            for row in src_rows
        ]

    def head_version_md(self, tbl_id: UUID) -> schema.VersionMd | None:
        """
        Returns the TableVersionMd for the most recent non-fragment version of the given table.
        """
        conn = Env.get().conn

        q = (
            sql.select(schema.TableVersion.md)
            .where(schema.TableVersion.tbl_id == tbl_id)
            .where(schema.TableVersion.md['is_fragment'].astext == 'false')
            .order_by(schema.TableVersion.md['version'].cast(sql.Integer).desc())
            .limit(1)
        )
        row = conn.execute(q).one_or_none()
        if row is None:
            return None
        assert isinstance(row[0], dict)
        return schema.md_from_dict(schema.VersionMd, row[0])

    def load_tbl_md(self, key: TableVersionKey) -> TableVersionMd:
        """
        Loads metadata from the store for a given table UUID and version.
        """
        anchor_timestamp: float | None = None
        if key.anchor_tbl_id is not None:
            anchored_version_md = self.head_version_md(key.anchor_tbl_id)
            # `anchor_tbl_id` must exist and have at least one non-fragment version, or else this isn't
            # a valid TableVersion specification.
            assert anchored_version_md is not None
            anchor_timestamp = anchored_version_md.created_at

        # _logger.info(f'Loading metadata for table version: {tbl_id}:{effective_version}')
        conn = Env.get().conn

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
        elif anchor_timestamp is not None:
            # we are loading the version that is anchored to the head version of another table (see TableVersion
            # docstring for details)
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND tv.md.schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id AND tv.md.created_at <= anchor_timestamp
            # ORDER BY tv.md.created_at DESC
            # LIMIT 1
            q = (
                q.where(
                    schema.TableVersion.md['created_at'].cast(sql.Float) <= anchor_timestamp,
                    schema.TableVersion.md['schema_version'].cast(sql.Integer)
                    == schema.TableSchemaVersion.schema_version,
                )
                .order_by(schema.TableVersion.md['created_at'].cast(sql.Float).desc())
                .limit(1)
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
            raise excs.Error(self._dropped_tbl_error_msg(key.tbl_id))
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
        session = Env.get().session

        # Construct and insert or update table record if requested.
        if tbl_md is not None:
            assert tbl_md.tbl_id == str(tbl_id)
            if version_md is not None:
                assert tbl_md.current_version == version_md.version
                assert tbl_md.current_schema_version == version_md.schema_version
            if schema_version_md is not None:
                assert tbl_md.current_schema_version == schema_version_md.schema_version
            if pending_ops is not None:
                assert tbl_md.pending_stmt is not None
                assert all(op.tbl_id == str(tbl_id) for op in pending_ops)
                assert all(op.op_sn == i for i, op in enumerate(pending_ops))
                assert all(op.num_ops == len(pending_ops) for op in pending_ops)
                tbl_md.tbl_state = schema.TableState.ROLLFORWARD
                self._roll_forward_ids.add(tbl_id)

            if dir_id is not None:
                # We are inserting a record while creating a new table.
                tbl_record = schema.Table(
                    id=tbl_id, dir_id=dir_id, md=dataclasses.asdict(tbl_md, dict_factory=md_dict_factory)
                )
                session.add(tbl_record)
            else:
                # Update the existing table record.
                values: dict[Any, Any] = {schema.Table.md: dataclasses.asdict(tbl_md, dict_factory=md_dict_factory)}
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
                # Validate that the only fields that can change are 'is_fragment' and 'additional_md'.
                assert version_record.md == dataclasses.asdict(
                    dataclasses.replace(
                        version_md,
                        is_fragment=version_record.md['is_fragment'],
                        additional_md=version_record.md['additional_md'],
                    )
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
                op_record = schema.PendingTableOp(tbl_id=tbl_id, op_sn=op.op_sn, op=dataclasses.asdict(op))
                session.add(op_record)

        session.flush()  # Inform SQLAlchemy that we want to write these changes to the DB.

    def store_update_status(self, tbl_id: UUID, version: int, status: UpdateStatus) -> None:
        """Update the TableVersion.md.update_status field"""
        assert self._in_write_xact
        conn = Env.get().conn

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
        conn = Env.get().conn
        _logger.info(f'delete_tbl_md({tbl_id})')
        conn.execute(sql.delete(schema.TableSchemaVersion.__table__).where(schema.TableSchemaVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.TableVersion.__table__).where(schema.TableVersion.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.PendingTableOp.__table__).where(schema.PendingTableOp.tbl_id == tbl_id))
        conn.execute(sql.delete(schema.Table.__table__).where(schema.Table.id == tbl_id))

    def load_replica_md(self, tbl: Table) -> list[TableVersionMd]:
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
            snapshot_md = self.load_tbl_md(TableVersionKey(tbl._id, 0, None))
            md = [snapshot_md, *md]

        for ancestor_md in md:
            # Set the `is_replica` flag on every ancestor's TableMd.
            ancestor_md.tbl_md.is_replica = True
            # For replica metadata, we guarantee that the current_version and current_schema_version of TableMd
            # match the corresponding values in TableVersionMd and TableSchemaVersionMd. This is to ensure that,
            # when the metadata is later stored in the catalog of a different Pixeltable instance, the values of
            # current_version and current_schema_version will always point to versions that are known to the
            # destination catalog.
            ancestor_md.tbl_md.current_version = ancestor_md.version_md.version
            ancestor_md.tbl_md.current_schema_version = ancestor_md.schema_version_md.schema_version

        for ancestor_md in md[1:]:
            # Also, the table version of every proper ancestor is emphemeral; it does not represent a queryable
            # table version (the data might be incomplete, since we have only retrieved one of its views, not
            # the table itself).
            ancestor_md.version_md.is_fragment = True

        return md

    def _load_tbl_version(self, key: TableVersionKey, *, check_pending_ops: bool = True) -> TableVersion | None:
        """Creates TableVersion instance from stored metadata and registers it in _tbl_versions."""
        tv_md = self.load_tbl_md(key)
        tbl_md = tv_md.tbl_md
        version_md = tv_md.version_md
        schema_version_md = tv_md.schema_version_md
        view_md = tbl_md.view_md

        conn = Env.get().conn

        if check_pending_ops:
            # if we care about pending ops, we also care whether the table is in the process of getting dropped
            if tbl_md.pending_stmt == schema.TableStatement.DROP_TABLE:
                raise excs.Error(self._dropped_tbl_error_msg(key.tbl_id))

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
        if key.effective_version is None and key.anchor_tbl_id is None and not tbl_md.is_replica:
            q = (
                sql.select(schema.Table.id)
                .where(schema.Table.md['view_md']['base_versions'][0][0].astext == key.tbl_id.hex)
                .where(schema.Table.md['view_md']['base_versions'][0][1].astext == None)
            )
            mutable_view_ids = [r[0] for r in conn.execute(q).all()]

        mutable_views = [TableVersionHandle(TableVersionKey(id, None, None)) for id in mutable_view_ids]

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
                    TableVersionKey(UUID(view_md.base_versions[0][0]), view_md.base_versions[0][1], key.anchor_tbl_id)
                )
            else:
                base_path = TableVersionPath.from_md(tbl_md.view_md.base_versions)
                base = base_path.tbl_version

            tbl_version = TableVersion(
                key, tbl_md, version_md, schema_version_md, mutable_views, base_path=base_path, base=base
            )

        # register the instance before init()
        self._tbl_versions[key] = tbl_version
        # register this instance as modified, so that it gets purged if the transaction fails, it may not be
        # fully initialized
        self.mark_modified_tvs(tbl_version.handle)
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
        self,
        path: Path,
        expected_obj_type: type[SchemaObject],
        expected_snapshot: bool,
        if_exists: IfExistsParam,
        *,
        base: TableVersionPath | None = None,
    ) -> SchemaObject | None:
        obj, _, _ = self._prepare_dir_op(add_dir_path=path.parent, add_name=path.name)

        if if_exists == IfExistsParam.ERROR and obj is not None:
            raise excs.Error(f'Path {path!r} is an existing {obj._display_name()}')
        else:
            is_snapshot = isinstance(obj, View) and obj._tbl_version_path.is_snapshot()
            if obj is not None and (not isinstance(obj, expected_obj_type) or (expected_snapshot and not is_snapshot)):
                if expected_obj_type is Dir:
                    obj_type_str = 'directory'
                elif expected_obj_type is InsertableTable:
                    obj_type_str = 'table'
                elif expected_obj_type is View:
                    obj_type_str = 'snapshot' if expected_snapshot else 'view'
                else:
                    raise AssertionError()
                raise excs.Error(
                    f'Path {path!r} already exists but is not a {obj_type_str}. Cannot {if_exists.name.lower()} it.'
                )

        if obj is None:
            return None
        if if_exists == IfExistsParam.IGNORE:
            return obj

        assert if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)

        # Check for circularity
        if obj is not None and base is not None:
            assert isinstance(obj, Table)  # or else it would have been caught above
            if obj._id in tuple(version.id for version in base.get_tbl_versions()):
                raise excs.Error(
                    "Cannot use if_exists='replace' with the same name as one of the view's own ancestors."
                )

        # drop the existing schema object
        if isinstance(obj, Dir):
            dir_contents = self._get_dir_contents(obj._id)
            if len(dir_contents) > 0 and if_exists == IfExistsParam.REPLACE:
                raise excs.Error(
                    f'Directory {path!r} already exists and is not empty. '
                    'Use `if_exists="replace_force"` to replace it.'
                )
            self._drop_dir(obj._id, path, force=True)
        else:
            assert isinstance(obj, Table)
            self._drop_tbl(obj, force=if_exists == IfExistsParam.REPLACE_FORCE, is_replace=True)
        return None
