from __future__ import annotations

import dataclasses
import functools
import logging
import random
import time
from collections import defaultdict
from collections.abc import Collection
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping, TypeVar
from uuid import UUID, uuid4

import psycopg
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc
from sqlalchemy.dialects.postgresql import array as pg_array

import pixeltable.index as index
from pixeltable import exceptions as excs, exprs, func, telemetry
from pixeltable.catalog import model
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec
from pixeltable.utils import fault_injection
from pixeltable.utils.exception_handler import run_cleanup
from pixeltable.utils.fault_injection import FaultLocation

from .catalog_base import CatalogBase
from .column import Column
from .dir import Dir
from .globals import DirEntry, IfExistsParam, IfNotExistsParam, MediaValidation, QColumnId
from .insertable_table import InsertableTable
from .local_table import LocalTable
from .path import ROOT_PATH, Path
from .schema_object import SchemaObject
from .table_path import TablePath, TableVersionPath
from .table_version import TableVersion, TableVersionKey, TableVersionMd
from .table_version_handle import TableVersionHandle
from .tbl_ops import DeleteTableMdOp, OpStatus, TableOp
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


# -1: unlimited
# for now, we don't limit the number of retries, because we haven't seen situations where the actual number of retries
# grows uncontrollably
_MAX_RETRIES = -1

T = TypeVar('T')


def retry_loop(
    *,
    for_write: bool = False,
    read_tvps: Collection[TableVersionPath] | None = None,
    read_tbl_ids: Collection[UUID] | None = None,
    write_tvps: Collection[TableVersionPath] | None = None,
    write_tbl_ids: Collection[UUID] | None = None,
    lock_mutable_tree: bool = False,
    isolation_level: Literal['READ COMMITTED', 'REPEATABLE READ', 'SERIALIZABLE'] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(op: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(op)
        def loop(*args: Any, **kwargs: Any) -> T:
            cat = get_runtime().catalog
            # retry_loop() is reentrant
            if cat._in_retry_loop:
                cat._check_write_locks(write_tvps or [], write_tbl_ids or [], lock_mutable_tree)
                return op(*args, **kwargs)

            num_retries = 0
            while True:
                cat._in_retry_loop = True
                try:
                    # in order for retry to work, we need to make sure that there aren't any prior db updates
                    # that are part of an ongoing transaction
                    assert not get_runtime().in_xact
                    with (
                        cat._allow_tbl_md_read(),
                        cat.begin_xact(
                            for_write=for_write,
                            read_tvps=read_tvps,
                            read_tbl_ids=read_tbl_ids,
                            write_tvps=write_tvps,
                            write_tbl_ids=write_tbl_ids,
                            convert_db_excs=False,
                            lock_mutable_tree=lock_mutable_tree,
                            isolation_level=isolation_level,
                            finalize_pending_ops=True,
                        ),
                    ):
                        return op(*args, **kwargs)
                except PendingTableOpsError as e:
                    Env.get().console_logger.debug(f'retry_loop(): finalizing pending ops for {e.tbl_id}')
                    cat._finalize_pending_ops(e.tbl_id)
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
                            raise excs.ConcurrencyError(
                                excs.ErrorCode.SERIALIZATION_FAILURE,
                                f'Serialization retry limit ({_MAX_RETRIES}) exceeded',
                            ) from e
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


class Catalog(CatalogBase):
    """The functional interface to getting access to catalog objects

    All interface functions must be called in the context of a transaction, started with Catalog.begin_xact() or
    via retry_loop().

    When calling functions that involve Table or TableVersion instances, the catalog needs to get a chance to finalize
    pending ops against those tables. The protocol depends on where metadata loads occur relative to the atomic
    operation:
    - If all metadata loads happen at the beginning of an atomic operation (eg, insert/update/delete), use
      begin_xact(). It will finalize pending ops before locking.
    - If metadata loads happen in the middle of an atomic operation, wrap the entire operation in retry_loop(), which
      handles pending ops and serialization retries.

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
    _tbl_versions: dict[TableVersionKey, TableVersion]
    _tbls: dict[tuple[UUID, int | None], LocalTable]
    _in_write_xact: bool  # True if we're in a write transaction
    _x_locked_tbl_ids: set[UUID]  # Ids of tables exclusively locked for write in the current transaction
    _modified_tvs: set[TableVersionHandle]  # TableVersion instances modified in the current transaction
    _roll_forward_ids: set[UUID]  # ids of Tables that have pending TableOps
    _undo_actions: list[Callable[[], None]]
    _in_retry_loop: bool
    # True within _allow_tbl_md_read(); permits loading table metadata inside begin_xact initialization, or inside
    # a retry_loop, but not in the middle of a regular begin_xact transaction.
    _tbl_md_read_allowed: bool

    # cached column dependencies
    # - key: table id, value: mapping from column id to its dependencies
    # - only maintained for dependencies between non-snapshot table versions
    # - can contain stale entries (stemming from invalidated TV instances)
    _column_dependencies: dict[UUID, dict[QColumnId, set[QColumnId]]]

    # column dependents are recomputed at the beginning of every write transaction and only reflect the locked tree
    _column_dependents: dict[QColumnId, set[QColumnId]] | None

    def __init__(self) -> None:
        self._tbl_versions = {}
        self._tbls = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self._in_write_xact = False
        self._x_locked_tbl_ids = set()
        self._modified_tvs = set()
        self._roll_forward_ids = set()
        self._undo_actions = []
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

    def validate_tbls_exist(self, tbl_ids: Collection[UUID]) -> None:
        """Raises TABLE_NOT_FOUND if any id is not a live (not dropped or being dropped) table."""
        with self.begin_xact():
            conn = get_runtime().conn
            assert conn is not None
            for tbl_id in tbl_ids:
                q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(tbl_id=tbl_id))
                if conn.execute(q).scalar() == 0:
                    raise excs.table_was_dropped(tbl_id)

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
    def begin_xact(
        self,
        *,
        for_write: bool = False,
        read_tvps: Collection[TableVersionPath] | None = None,
        read_tbl_ids: Collection[UUID] | None = None,
        write_tvps: Collection[TableVersionPath] | None = None,
        write_tbl_ids: Collection[UUID] | None = None,
        lock_mutable_tree: bool = False,
        convert_db_excs: bool = True,
        finalize_pending_ops: bool = True,
        isolation_level: Literal['READ COMMITTED', 'REPEATABLE READ', 'SERIALIZABLE'] | None = None,
    ) -> Iterator[sql.Connection]:
        """
        Return a context manager that yields a connection to the database. Idempotent.

        It is mandatory to call this method, not Env.begin_xact(), if the transaction accesses any table data
        or metadata.

        Locking protocol (via _acquire_locks()):
        - write targets (write_tvps, write_tbl_ids): x-locks each Table record (see
          _acquire_write_lock() / _acquire_path_locks())
        - read targets (read_tvps, read_tbl_ids): refreshes the metadata cache, no x-lock
        - if lock_mutable_tree == True, also x-locks all mutable views of each write target
        - if finalize_pending_ops == True and a PendingTableOpsError is raised, finalizes pending ops and retries
        - this needs to be done in a retry loop, because Postgres can abort the transaction
          (SerializationFailure, LockNotAvailable)
        - for that reason, we do all lock acquisition prior to doing any real work (eg, compute column values),
          to minimize the probability of losing that work due to a forced abort

        If convert_db_excs == True, converts DBAPIErrors into excs.Errors if possible.
        """
        assert for_write or not (write_tvps or write_tbl_ids), 'for_write must be True when write targets are specified'
        read_tvps = read_tvps or []
        write_tvps = write_tvps or []
        read_tbl_ids = read_tbl_ids or []
        write_tbl_ids = write_tbl_ids or []
        if get_runtime().in_xact:
            self._check_write_locks(write_tvps, write_tbl_ids, lock_mutable_tree)
            yield get_runtime().conn
            return

        num_retries = 0
        pending_ops_tbl_id: UUID | None = None
        has_exc = False  # True if we exited the 'with ...begin_xact()' block with an exception
        while True:
            if pending_ops_tbl_id is not None:
                Env.get().console_logger.debug(f'begin_xact(): finalizing pending ops for {pending_ops_tbl_id}')
                self._finalize_pending_ops(pending_ops_tbl_id)
                pending_ops_tbl_id = None

            # one span per acquisition attempt; retries show up as sibling spans
            xact_span = telemetry.span_start('pixeltable.catalog.begin_xact', attrs={'pxt.for_write': for_write})
            attempt_exc: BaseException | None = None
            try:
                self._in_write_xact = for_write
                self._x_locked_tbl_ids = set()
                self._modified_tvs = set()
                self._column_dependents = None
                has_exc = False

                assert not self._undo_actions
                with get_runtime().begin_xact(for_write=for_write, isolation_level=isolation_level) as conn:
                    with self._allow_tbl_md_read():
                        try:
                            self._acquire_locks(
                                read_tvps=read_tvps,
                                read_tbl_ids=read_tbl_ids,
                                write_tvps=write_tvps,
                                write_tbl_ids=write_tbl_ids,
                                lock_mutable_tree=lock_mutable_tree,
                                finalize_pending_ops=finalize_pending_ops,
                            )
                            if for_write and lock_mutable_tree:
                                self._compute_column_dependents(self._x_locked_tbl_ids)
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
                            if isinstance(
                                e.orig, (psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)
                            ) and (num_retries < _MAX_RETRIES or _MAX_RETRIES == -1):
                                _logger.debug(f'Retriable error {type(e.orig)} on attempt {num_retries}')
                                num_retries += 1
                                time.sleep(random.uniform(0.1, 0.5))
                                # attempt failed -- don't try to commit the transaction before retrying
                                conn.rollback()
                                assert not self._undo_actions  # We should not have any undo actions at this point
                                continue
                            raise

                    assert not self._undo_actions
                    # success: end the attempt span here so it covers only the acquisition, not the
                    # caller's work under the yield; the finally below then sees None and no-ops
                    telemetry.span_end(xact_span)
                    xact_span = None
                    yield conn
                    return

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
                single_tbl, single_tbl_id = self._get_single_tbl(write_tvps, read_tvps, write_tbl_ids, read_tbl_ids)
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
                self._x_locked_tbl_ids.clear()
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
                    # Execute undo actions in reverse order (LIFO)
                    for hook in reversed(self._undo_actions):
                        run_cleanup(hook, raise_error=False)
                    # purge all modified TableVersion instances; we can't guarantee they are still consistent with the
                    # stored metadata
                    for handle in self._modified_tvs:
                        self._clear_tv_cache(handle.key)
                    # Clear potentially corrupted cached metadata
                    for tvp in [*write_tvps, *read_tvps]:
                        tvp.clear_cached_md()

                self._undo_actions.clear()
                self._modified_tvs.clear()

    def _acquire_locks(
        self,
        read_tvps: Collection[TableVersionPath],
        read_tbl_ids: Collection[UUID],
        write_tvps: Collection[TableVersionPath],
        write_tbl_ids: Collection[UUID],
        lock_mutable_tree: bool = False,
        finalize_pending_ops: bool = True,
    ) -> None:
        """
        Acquires locks on the specified write targets (including their mutable tree, if lock_mutable_tree is True), and
        updates self._x_locked_tbl_ids accordingly.

        Refreshes the metadata cache for the read targets.

        The order matters: TVPs are processed before tbl_ids in both groups so that ancestor-first validation
        (write_tvps -> write_tbl_ids -> read_tvps -> read_tbl_ids) is established before any unordered ID pass runs.
        """
        x_locked_ids: set[UUID] = set()
        for tvp in write_tvps:
            if tvp.tbl_id in x_locked_ids:
                continue
            x_locked_ids.update(
                self._acquire_path_locks(
                    tbl=tvp, for_write=True, lock_mutable_tree=lock_mutable_tree, check_pending_ops=finalize_pending_ops
                )
            )
        for tbl_id in write_tbl_ids:
            if tbl_id in x_locked_ids:
                continue
            x_locked_ids.update(
                self._acquire_write_lock(
                    tbl_id=tbl_id, lock_mutable_tree=lock_mutable_tree, check_pending_ops=finalize_pending_ops
                )
            )
        for tvp in read_tvps:
            self._acquire_path_locks(tbl=tvp, for_write=False, check_pending_ops=finalize_pending_ops)
        for tbl_id in read_tbl_ids:
            self._refresh_tbl_cache(tbl_id=tbl_id, check_pending_ops=finalize_pending_ops)

        self._x_locked_tbl_ids = x_locked_ids

    def register_undo_action(self, func: Callable[[], None]) -> Callable[[], None]:
        """Registers a function to be called if the current transaction fails.

        The function is called only if the current transaction fails due to an exception.

        Rollback functions are called in reverse order of registration (LIFO).

        The function should not raise exceptions; if it does, they are logged and ignored.
        """
        assert self.in_write_xact
        self._undo_actions.append(func)
        return func

    def _get_single_tbl(
        self,
        write_tvps: Collection[TableVersionPath],
        read_tvps: Collection[TableVersionPath],
        write_tbl_ids: Collection[UUID],
        read_tbl_ids: Collection[UUID],
    ) -> tuple[TableVersionHandle | None, UUID | None]:
        """Return (tbl, None) or (None, tbl_id) iff the transaction touches exactly one table; else (None, None)."""
        total = len(write_tvps) + len(read_tvps) + len(read_tbl_ids) + len(write_tbl_ids)
        if total != 1:
            return None, None
        if write_tvps or read_tvps:
            return next(iter(write_tvps or read_tvps)).tbl_version, None
        return None, next(iter(read_tbl_ids or write_tbl_ids))

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
            if isinstance(e.orig, psycopg.errors.DuplicateColumn):
                # TODO: extend message with the name of the schema column (not the store column)
                raise excs.AlreadyExistsError(excs.ErrorCode.COLUMN_ALREADY_EXISTS, 'Duplicate column') from raise_from
            raise excs.ConcurrencyError(
                excs.ErrorCode.SERIALIZATION_FAILURE,
                'That Pixeltable operation could not be completed because it conflicted with another '
                'operation that was run on a different process.\n'
                'Please re-run the operation.',
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
    ) -> set[UUID]:
        """
        Path locking protocol:
        - refresh cached TableVersions of ancestors (we need those even during inserts, for computed columns that
          reference the base tables)
        - refresh cached TableVersion of tbl or get X-lock, depending on for_write
        - if lock_mutable_tree, also X-lock all mutable views of tbl

        Raises Error if tbl doesn't exist.
        Returns the set of table IDs that were X-locked (empty if for_write=False or if the lock couldn't be
        acquired, e.g., tbl is a non-mutable table).
        """
        path_handles = tbl.get_tbl_versions()
        read_handles = path_handles[:0:-1] if for_write else path_handles[::-1]
        for handle in read_handles:
            # update cache
            _ = self._get_tbl_version(handle.key, validate_initialized=True)
        if not for_write:
            return set()  # nothing to lock
        return self._acquire_write_lock(
            tbl_id=tbl.tbl_id, lock_mutable_tree=lock_mutable_tree, check_pending_ops=check_pending_ops
        )

    def _lock_tbl_if_exists(
        self, *, tbl_id: UUID | None = None, dir_id: UUID | None = None, tbl_name: str | None = None
    ) -> None:
        """
        Attempts to acquire an X-lock on a Table record, but does nothing if the table does not exist.

        Either tbl_id or dir_id+tbl_name need to be specified. Used when locking a slot that may not yet
        contain a table (e.g., guard against concurrent creation).
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

        conn = get_runtime().conn
        q = sql.select(schema.Table).where(where_clause).with_for_update(nowait=True)
        row = conn.execute(q).one_or_none()
        if row is None:
            return
        tbl_md = schema.md_from_dict(schema.TableMd, row.md)
        if tbl_md.is_mutable:
            conn.execute(sql.update(schema.Table).values(lock_dummy=1).where(where_clause))

        pending_ops_q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == row.id)
        has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
        if has_pending_ops:
            raise PendingTableOpsError(row.id)

    def _acquire_write_lock(
        self, *, tbl_id: UUID, lock_mutable_tree: bool = False, check_pending_ops: bool = True
    ) -> set[UUID]:
        """
        Acquires an exclusive lock on a Table.

        If lock_mutable_tree, also recursively locks all mutable views of the table.

        Returns the set of locked table IDs, which could be empty if the lock couldn't be acquired,
        e.g., tbl is a non-mutable table.
        """
        where_clause = schema.Table.id == tbl_id
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Table).where(where_clause).with_for_update(nowait=True)).one_or_none()
        if row is None:
            raise excs.table_was_dropped(tbl_id)
        tbl_md = schema.md_from_dict(schema.TableMd, row.md)
        locked: set[UUID] = set()
        if tbl_md.is_mutable:
            locked.add(row.id)
            conn.execute(sql.update(schema.Table).values(lock_dummy=1).where(where_clause))
            # Invalidate the cached TableVersion to make sure we are acting on the latest version after locking.
            cached_tv = self._tbl_versions.get(TableVersionKey(tbl_id, None))
            if cached_tv is not None:
                cached_tv.is_validated = False

        if check_pending_ops:
            # check for pending ops after getting table lock
            pending_ops_q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == row.id)
            has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
            if has_pending_ops:
                raise PendingTableOpsError(row.id)

        if not tbl_md.is_mutable:
            return set()  # nothing to lock

        if lock_mutable_tree:
            # also lock mutable views
            key = TableVersionKey(row.id, tbl_md.current_version if tbl_md.is_snapshot else None)
            tv = self._get_tbl_version(key, validate_initialized=True)
            for view in tv.mutable_views:
                locked.update(
                    self._acquire_write_lock(
                        tbl_id=view.id, lock_mutable_tree=True, check_pending_ops=check_pending_ops
                    )
                )
        return locked

    def _check_write_locks(
        self, write_tvps: Collection[TableVersionPath], write_tbl_ids: Collection[UUID], lock_mutable_tree: bool
    ) -> None:
        """Asserts that all specified write targets (and their mutable trees, if lock_mutable_tree) are locked."""

        def assert_tree_locked(tbl_id: UUID) -> None:
            tree = self._get_mutable_tree(tbl_id)
            assert tree.issubset(self._x_locked_tbl_ids), (
                f'mutable tree of {tbl_id} not fully locked: {tree - self._x_locked_tbl_ids}'
            )

        for tvp in write_tvps:
            assert tvp.tbl_id in self._x_locked_tbl_ids, f'{tvp.tbl_id} not locked: {self._x_locked_tbl_ids}'
            if lock_mutable_tree:
                assert_tree_locked(tvp.tbl_id)
        for tbl_id in write_tbl_ids:
            assert tbl_id in self._x_locked_tbl_ids, f'{tbl_id} not locked: {self._x_locked_tbl_ids}'
            if lock_mutable_tree:
                assert_tree_locked(tbl_id)

    def _refresh_tbl_cache(self, *, tbl_id: UUID, check_pending_ops: bool = True) -> None:
        """
        Refreshes the cached metadata for a table without acquiring any lock.
        """
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Table).where(schema.Table.id == tbl_id)).one_or_none()
        if row is None:
            raise excs.table_was_dropped(tbl_id)
        tbl_md = schema.md_from_dict(schema.TableMd, row.md)

        if check_pending_ops:
            pending_ops_q = sql.select(sql.func.count()).where(schema.PendingTableOp.tbl_id == row.id)
            has_pending_ops = conn.execute(pending_ops_q).scalar() > 0
            if has_pending_ops:
                raise PendingTableOpsError(row.id)

        if not tbl_md.is_pure_snapshot:
            key = TableVersionKey(row.id, tbl_md.current_version if tbl_md.is_snapshot else None)
            self._get_tbl_version(key, validate_initialized=True)

    def _roll_forward(self) -> None:
        """Finalize pending ops for all tables in self._roll_forward_ids."""
        for tbl_id in self._roll_forward_ids:
            exc = self._finalize_pending_ops(tbl_id)
            if exc is not None:
                raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'Table operation was aborted with\n{exc!s}') from exc

    def _finalize_pending_ops(self, tbl_id: UUID) -> Exception | None:
        """
        Finalizes all pending ops for the given table, and clears the table version cache for that table.

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

        tbl_q = sql.select(schema.Table.md).where(schema.Table.id == tbl_id).with_for_update()

        while True:
            try:
                with (
                    self.begin_xact(
                        for_write=True, write_tbl_ids=[tbl_id], convert_db_excs=False, finalize_pending_ops=False
                    ) as conn,
                    self._allow_tbl_md_read(),
                ):
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
                        self._get_tbl_version(
                            TableVersionKey(tbl_id, tbl_version), check_pending_ops=False, validate_initialized=True
                        )
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

            except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
                # TODO: why are we still seeing these here, instead of them getting taken care of by the retry
                # logic of begin_xact()?
                if isinstance(
                    e.orig,
                    (
                        psycopg.errors.SerializationFailure,
                        psycopg.errors.LockNotAvailable,
                        psycopg.errors.InFailedSqlTransaction,
                    ),
                ):
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
                    with self.begin_xact(
                        for_write=True, write_tbl_ids=[tbl_id], convert_db_excs=False, finalize_pending_ops=False
                    ) as conn:
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

        This function must be called inside a transaction with an exclusive table lock held.

        Returns True if no unresolved pending ops remain on the table, and the table's status was set to LIVE. False
        otherwise.
        """
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
        Selects table's pending ops for update and returns them as TableOps in order.

        Must be called inside a transaction with the table selected for update.
        """
        conn = get_runtime().conn
        q = (
            sql.select(schema.PendingTableOp)
            .where(schema.PendingTableOp.tbl_id == tbl_id)
            .order_by(schema.PendingTableOp.op_sn)
            .with_for_update()
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
        tv = self._get_tbl_version(key, validate_initialized=True)
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
            tv = self._get_tbl_version(TableVersionKey(dependent.tbl_id, None), validate_initialized=True)
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
        get_runtime().conn.execute(q)

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

    @retry_loop(for_write=False)
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
            if isinstance(src_obj, LocalTable):
                self._move_table(src_obj._id, new_path.name, dest_dir._id)
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

        # check for table
        if lock_entry:
            self._lock_tbl_if_exists(dir_id=dir_id, tbl_name=name)
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
        assert expected in (None, LocalTable, Dir), expected

        if path.is_root:
            # the root dir
            if expected is not None and expected is not Dir:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{path!r} needs to be a table but is a dir')
            dir = self._get_dir(path, lock_dir=lock_obj)
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
        obj = self._get_dir_entry(parent_dir.id, path.name, path.version, lock_entry=lock_obj)

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
        if (tbl_id, version) not in self._tbls:
            # begin_xact() is re-entrant: it joins the caller's transaction if there is one, and otherwise
            # starts a fresh read transaction (which also permits the metadata load). Cache hits stay xact-free.
            with self.begin_xact(for_write=False):
                if version is None:
                    tbl = self._load_tbl(tbl_id, ignore_pending_drop=ignore_if_dropped)
                else:
                    tbl = self._load_tbl_at_version(tbl_id, version)
        else:
            tbl = self._tbls.get((tbl_id, version))
        if tbl is not None:
            Env.get().record_tbl_catalog_uri(tbl._id, ROOT_PATH)
        return tbl

    def create_table(
        self,
        path: Path,
        schema: dict[str, ColumnSpec],
        if_exists: IfExistsParam,
        primary_key: list[str] | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        create_default_idxs: bool,
        is_versioned: bool,
    ) -> tuple[LocalTable, bool]:
        """
        Creates a new InsertableTable at the given path.

        If `if_exists == IfExistsParam.IGNORE` and a table `t` already exists at the given path, returns `t, False`.

        Otherwise, creates a new table `t` and returns `t, True` (or raises an exception if the operation fails).
        """

        columns = [Column.create(name, spec) for name, spec in schema.items()]

        return self._create_table(
            path,
            columns,
            if_exists,
            primary_key,
            comment,
            custom_metadata,
            media_validation,
            create_default_idxs,
            is_versioned,
        )

    def _create_table(
        self,
        path: Path,
        columns: list[Column],
        if_exists: IfExistsParam,
        primary_key: list[str] | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        create_default_idxs: bool,
        is_versioned: bool,
        additional_idxs: list[tuple[Column, str | None, 'index.IndexBase']] | None = None,
        explicit_tbl_id: UUID | None = None,
    ) -> tuple[LocalTable, bool]:
        import pixeltable.metadata.schema

        # If a table id is passed in advance, we guarantee that the returned table will be created with that id.
        # Therefore IfExistsParam.IGNORE is incompatible with explicit_tbl_id.
        assert explicit_tbl_id is None or if_exists != IfExistsParam.IGNORE

        if primary_key is None:
            primary_key = []
        if additional_idxs is None:
            additional_idxs = []

        @retry_loop(for_write=True)
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
                primary_key=primary_key,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                create_default_idxs=create_default_idxs,
                is_versioned=is_versioned,
                additional_idxs=additional_idxs,
            )
            assert tbl_id == UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = pixeltable.metadata.schema.TableStatement.CREATE_TABLE
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            return tbl_id, True

        self._roll_forward_ids.clear()
        tbl_id, is_created = create_fn()
        self._roll_forward()

        @retry_loop(read_tbl_ids=[tbl_id])
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
        create_default_idxs: bool,
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
            create_default_idxs,
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
        create_default_idxs: bool,
        iterator: func.GeneratingFunctionCall | None,
        comment: str | None,
        custom_metadata: Any,
        media_validation: MediaValidation,
        if_exists: IfExistsParam,
        additional_idxs: list[tuple[Column, str | None, 'index.IndexBase']] | None = None,
        explicit_tbl_id: UUID | None = None,
    ) -> tuple[LocalTable, bool]:
        assert explicit_tbl_id is None or if_exists == IfExistsParam.ERROR

        if additional_idxs is None:
            additional_idxs = []

        @retry_loop(for_write=True)
        def create_fn() -> tuple[UUID, bool]:
            existing = self._handle_path_collision(path, View, is_snapshot, if_exists, base=base)
            if existing is not None:
                assert isinstance(existing, View)
                return existing._id, False

            if not is_snapshot and base.is_mutable():
                # this is a mutable view of a mutable base; X-lock the base and advance its view_sn before adding
                # the view
                base_id = base.tbl_id
                assert len(self._acquire_write_lock(tbl_id=base_id)) == 1, base_id
                self._x_locked_tbl_ids.add(base_id)
                base_tv = self._get_tbl_version(TableVersionKey(base.tbl_id, None), validate_initialized=True)
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
                create_default_idxs=create_default_idxs,
                iterator_call=iterator,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                additional_idxs=additional_idxs,
            )
            assert tbl_id == UUID(md.tbl_md.tbl_id)
            md.tbl_md.pending_stmt = schema.TableStatement.CREATE_VIEW
            self.write_tbl_md(tbl_id, dir._id, md.tbl_md, md.version_md, md.schema_version_md, ops)
            fault_injection.process_fault(FaultLocation.CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED)
            return tbl_id, True

        self._roll_forward_ids.clear()
        view_id, is_created = create_fn()
        if not is_snapshot and base.is_mutable():
            # invalidate base's TableVersion instance, so that it gets reloaded with the new mutable view
            self._clear_tv_cache(base.tbl_version.key)
        self._roll_forward()

        @retry_loop(read_tbl_ids=[view_id])
        def get_tbl_fn() -> LocalTable:
            return self.get_table_by_id(view_id)

        return get_tbl_fn(), is_created

    def create_from_model(
        self,
        path: Path,
        columns: dict[str, ColumnSpec],
        display_name: str,
        create_default_idxs: bool,
        media_validation: MediaValidation,
        comment: str | None,
        custom_metadata: Any,
        iterator: func.GeneratingFunctionCall | None,
        base: 'pxt.Query | None',
        embedding_idxs: dict[str, model.EmbeddingIndex],
    ) -> tuple[LocalTable, bool]:
        """Create a table or view from a declarative model.

        A model's column value expressions can contain placeholder references to other columns in the same table.
        Those references arrive as
        `ModelColumnRef`s and are substituted here, in the catalog that owns `path`, so they never have to
        be resolved across a proxy boundary. `base`, when present (i.e. this is a view), is an already-bound Query
        over the existing base table.

        If a table already exists at `path`, validates the model against it and returns it (idempotent rebind);
        otherwise creates it. Returns `(table, was_created)`.
        """
        # We allocate the table id up front so that self-referential ColumnRefs (built below) point at it; since
        # this runs in the catalog that owns the table, no such reference ever needs to be serialized.
        tbl_id = uuid4()
        tbl_handle = TableVersionHandle(TableVersionKey(tbl_id, None))

        iterator, additional_cols, resolved_idxs = model.prepare_model(
            tbl_handle, columns, display_name, media_validation, iterator, base, embedding_idxs
        )

        # If the table already exists, validate the model against it and rebind (the server enforces its own
        # consistency; we never trust a client to have validated).
        existing = self.get_table(path, IfNotExistsParam.IGNORE)
        if existing is not None:
            self._validate_model(existing, display_name, base, iterator)
            return existing, False

        if base is None:
            return self._create_table(
                path=path,
                columns=additional_cols,
                if_exists=IfExistsParam.ERROR,
                primary_key=None,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                create_default_idxs=create_default_idxs,
                is_versioned=True,
                additional_idxs=resolved_idxs,
                explicit_tbl_id=tbl_id,
            )

        else:
            return self._create_view(
                path=path,
                base=base._first_tbl,
                select_list=base.select_list,
                where=base.where_clause,
                sample_clause=base.sample_clause,
                additional_columns=additional_cols,
                is_snapshot=False,
                create_default_idxs=create_default_idxs,
                iterator=iterator,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                if_exists=IfExistsParam.ERROR,
                additional_idxs=resolved_idxs,
                explicit_tbl_id=tbl_id,
            )

    def update_from_model(self, updates: list[model.Updates]) -> None:
        """Update a table or view from a declarative model.

        If the table does not exist, raises NotFoundError. If the model is incompatible with the existing table,
        raises RequestError.
        """
        tbls = [self.get_table(update['path'], IfNotExistsParam.ERROR) for update in updates]

        @retry_loop(for_write=True, write_tvps=[tbl._tbl_version_path for tbl in tbls], lock_mutable_tree=True)
        def update_fn() -> None:
            # (tbl_version_path, tbl_version, updates) tuple for each table in the model update
            tbl_info = list(
                zip((tbl._tbl_version_path for tbl in tbls), (tbl._tbl_version.get() for tbl in tbls), updates)
            )

            # First drop any columns or indices that are being removed. Do this in *inverse* declaration order so that
            # dependent columns or indices are dropped first.
            for _, tbl_version, update in tbl_info[::-1]:
                for col_name in update['dropped_columns']:
                    col = tbl_version.cols_by_name[col_name]
                    tbl_version.drop_column(col)
                for idx_name in update['dropped_idxs']:
                    idx = tbl_version.idxs_by_name[idx_name]
                    tbl_version.drop_index(idx.id)

            # Now add any new columns or indices, in forward order (base tables first).
            for tvp, tv, update in tbl_info:
                resolved_cols, resolved_idxs = model.prepare_model_updates(
                    tvp, tv.display_str(), update['new_columns'], update['new_idxs']
                )
                tv.add_columns(resolved_cols, print_stats=False, on_error='abort')
                for col, idx_name, idx_base in resolved_idxs:
                    tv.add_index(col, idx_name, idx_base)

        update_fn()

    def _validate_model(
        self,
        existing: LocalTable,
        display_name: str,
        base: pxt.Query | None,
        iterator: func.GeneratingFunctionCall | None,
    ) -> None:
        """Raise if a model's schema is incompatible with an already-existing table of the same name."""
        existing_md = existing.get_metadata()
        model_kind = 'table' if base is None else 'view'
        if model_kind != existing_md['kind']:
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                f'{display_name} is defined as a {model_kind}, '
                f'but the existing {existing_md["path"]!r} is a {existing_md["kind"]}.',
            )

        # TODO: validate table properties (comment, custom_metadata, media_validation, primary_key, etc.)
        # TODO: validate base table query
        # TODO: inspect columns and indices

        bound_iterator_str = 'None' if iterator is None else iterator.display_str()
        if bound_iterator_str != str(existing_md['iterator_call']):
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                f'Iterator for {display_name} does not match the existing table {existing_md["path"]!r}.\n'
                f'  Model iterator: {bound_iterator_str}\n'
                f'  Existing iterator: {existing_md["iterator_call"]}',
            )

    def add_columns(self, tbl: TableVersionPath, cols: list[Column]) -> None:
        @retry_loop(for_write=True, write_tvps=[tbl], lock_mutable_tree=False)
        def add_fn() -> None:
            tv = self._get_tbl_version(TableVersionKey(tbl.tbl_id, None), validate_initialized=True)
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

        self._roll_forward_ids.clear()
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

    @retry_loop(for_write=False)
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
        @retry_loop(for_write=True)
        def drop_fn() -> None:
            tbl = self._get_schema_object(
                path,
                expected=LocalTable,
                raise_if_not_exists=(if_not_exists == IfNotExistsParam.ERROR and not force),
                lock_parent=True,
                lock_obj=False,
            )
            if tbl is None:
                _logger.info(f'Skipped table {path!r} (does not exist).')
                return
            assert isinstance(tbl, LocalTable)
            self._drop_tbl(tbl, force=force, is_replace=False)

        self._roll_forward_ids.clear()
        drop_fn()
        self._roll_forward()

    def _drop_tbl(self, tbl: LocalTable | TableVersionPath, force: bool, is_replace: bool) -> None:
        """
        Drop the table (and recursively its views, if force == True).

        `tbl` can be an instance of `Table` for a user table, or `TableVersionPath` for a hidden (system) table.

        Locking protocol:
        - X-lock base before X-locking any view
        - deadlock-free wrt to TableVersion.insert() (insert propagation also proceeds top-down)
        - X-locks parent dir: prevent concurrent creation of another SchemaObject
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

        # capture the path for logging before the drop runs (after drop, tbl is no longer safe to use)
        tbl_path_repr: str = str(tbl_id) if tbl is None else repr(tbl._path())
        if tbl is not None:
            self._acquire_dir_xlock(dir_id=tbl._dir_id())

        # If the base table needs an update, lock it before locking the view.
        if isinstance(tbl, View) and tvp.is_mutable() and tvp.base.is_mutable():
            base_id = tvp.base.tbl_id
            # Bug(PXT-1198): when multiple tables are getting dropped within one transaction (like when self._drop_dir
            # calls self._drop_tbl), the expected base-before-view lock ordering is currently not guaranteed.
            if base_id not in self._x_locked_tbl_ids:
                self._x_locked_tbl_ids.update(self._acquire_write_lock(tbl_id=base_id))
        self._x_locked_tbl_ids.update(self._acquire_write_lock(tbl_id=tbl_id))

        view_ids = self.get_view_ids(tbl_id, for_update=True)

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
                msg: str
                if is_replace:
                    msg = (
                        f'{tbl._display_str()} already exists and has dependents. '
                        "Use `if_exists='replace_force'` to replace it."
                    )
                else:
                    msg = f'{tbl._display_str()} has dependents.'
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
            drop_ops, new_version = tv.drop_ops()
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

        assert (tbl_id, None) in self._tbls  # tables must have an entry with effective_version=None

        # Remove visible Table references.
        versions = [version for id, version in self._tbls if id == tbl_id]
        for version in versions:
            del self._tbls[tbl_id, version]

        _logger.info(f'Dropped table {tbl_path_repr}.')

    def _incr_view_sn(self, tbl_id: UUID) -> None:
        """Increments the table's view_sn in the store within the current transaction"""
        self._clear_tv_cache(TableVersionKey(tbl_id, None))
        assert self._acquire_write_lock(tbl_id=tbl_id, check_pending_ops=False), tbl_id
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

    @retry_loop(for_write=True)
    def create_dir(self, path: Path, if_exists: IfExistsParam, parents: bool) -> Dir:
        return self._create_dir(path, if_exists, parents)

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
        conn = get_runtime().conn
        if not force:
            # check for existing entries
            q = sql.select(sql.func.count()).select_from(schema.Dir).where(schema.Dir.parent_id == dir_id)
            num_subdirs = conn.execute(q).scalar()
            q = sql.select(sql.func.count()).select_from(schema.Table).where(self._active_tbl_clause(dir_id=dir_id))
            num_tbls = conn.execute(q).scalar()
            if num_subdirs + num_tbls > 0:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Directory {dir_path!r} is not empty.')

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
        if for_update:
            q = q.with_for_update()
        result = [r[0] for r in conn.execute(q).all()]
        return result

    def get_tbl_version(self, key: TableVersionKey, *, validate_initialized: bool = False) -> TableVersion | None:
        """
        Returns the TableVersion instance for the given table version key, and updates the cache if necessary.

        This function can, but doesn't have to be called inside a transaction or a retry loop. It can manage its own
        retry loop internally if necessary.
        """
        if get_runtime().in_xact:
            return self._get_tbl_version(key, validate_initialized=validate_initialized)

        @retry_loop(for_write=False)
        def do_get_tbl_version() -> TableVersion | None:
            return self._get_tbl_version(key, validate_initialized=validate_initialized)

        return do_get_tbl_version()

    def _get_tbl_version(
        self, key: TableVersionKey, *, check_pending_ops: bool = True, validate_initialized: bool = False
    ) -> TableVersion | None:
        """
        Returns the TableVersion instance for the given table key, and updates the cache if necessary.

        If present in the cache and the instance isn't validated, validates version and view_sn against the stored
        metadata.
        """
        conn = get_runtime().conn
        assert conn is not None
        tv = self._tbl_versions.get(key)
        if tv is None and not self._tbl_md_read_allowed:
            raise AssertionError(
                'Loading new table metadata is not allowed in the middle of a transaction. '
                'To fix this, either: (1) declare all tables to be accessed upfront via begin_xact(), '
                'or (2) run the operation inside a retry_loop().'
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
            is_versioned = row.md.get('is_versioned', True)
            current_version = row.md['current_version']
            view_sn = row.md['view_sn']
            if (is_versioned and current_version != tv.version) or view_sn != tv.tbl_md.view_sn:
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

    def get_dir(self, dir_id: UUID, for_update: bool = False) -> Dir | None:
        """Return the Dir with the given id, or None if it doesn't exist"""
        conn = get_runtime().conn
        if for_update:
            self._acquire_dir_xlock(dir_id=dir_id)
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

    def read_dir_record(self, dir_id: UUID) -> schema.Dir:
        conn = get_runtime().conn
        row = conn.execute(sql.select(schema.Dir).where(schema.Dir.id == dir_id)).one_or_none()
        if row is None:
            raise excs.NotFoundError(excs.ErrorCode.DIRECTORY_NOT_FOUND, f'Directory not found: {dir_id}')
        return schema.Dir(**row._mapping)

    def _move_table(self, tbl_id: UUID, new_name: str, new_dir_id: UUID) -> None:
        """Update dir_id/name for tbl_id."""
        # TODO(PXT-1197): Catalog does not properly lock tables for the move
        # This assertion validates a crucial invariant, but it fails today.
        # assert tbl_id in self._x_locked_tbl_ids, f"Table {tbl_id} should be locked for the move but isn't"
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
            self._tbls[tbl_id, None] = tbl
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
        self._tbls[tbl_id, None] = view
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
        status = conn.execute(sql.delete(schema.Table).where(schema.Table.id == tbl_id))
        assert status.rowcount == 1, status.rowcount

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
            except (sql_exc.DBAPIError, sql_exc.OperationalError) as e:
                if not isinstance(
                    e.orig,
                    (
                        psycopg.errors.SerializationFailure,
                        psycopg.errors.LockNotAvailable,
                        psycopg.errors.DeadlockDetected,
                    ),
                ):
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
        all_contents = self.get_dir_contents(ROOT_PATH, recursive=True)
        with self.begin_xact(for_write=False), self._allow_tbl_md_read():
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
            if isinstance(idx_info.idx, index.BtreeIndex):
                # condition is the invariant violation that we are checking for
                # add it to where clause, and also to select clause for easier debugging
                if idx_info.val_col.col_type.is_string_type():
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
                .where((sa_tbl.c.v_max > tv.version) if tv.is_versioned else sql.true())
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

        if tv.is_versioned:
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
