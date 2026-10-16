# Catalog locking

`Catalog` coordinates concurrent reads, writes, and schema changes using Postgres locks on store tables.
This document explains the rules and why they are needed. The implementation is in
`pixeltable/catalog/catalog.py`.

## 1. Lock store tables before reading metadata

Each table or view with stored data has a Postgres store table: `tbl_<uuid.hex>` for tables and
`view_<uuid.hex>` for views. Pixeltable locks these tables with `LOCK TABLE`. Postgres releases the locks
when the transaction ends.

Under REPEATABLE READ, `LOCK TABLE` does not establish the transaction's snapshot. The first query after
lock acquisition does. This means a metadata read sees changes committed while the transaction was waiting
for its store table locks.

**Store table locks must precede every statement that establishes a snapshot.** A metadata query, a directory
row lock, or even `SELECT 1` before `LOCK TABLE` would leave the transaction reading metadata from before the wait.

Acquiring locks first ensures that the metadata read includes changes committed during the wait.
Directory locks come later and need the retry mechanism described in section 5.

## 2. Lock modes and waiting

The `Catalog.begin_*_xact()` methods and `retry_*_loop()` decorators select an operation class (`_TblOpClass`).
That class and the table's versioning, data-versioned or operational, determine the required lock mode.

| Operation class | Operational table | Data-versioned table | Wait policy |
| --- | --- | --- | --- |
| `MD_READ`: catalog metadata | No lock | No lock | N/A |
| `DATA_READ`: query rows | `ACCESS SHARE` | `ACCESS SHARE` | Wait if any locked table is data-versioned |
| `DATA_WRITE`: insert/update/delete | `ROW EXCLUSIVE` | `EXCLUSIVE` | Wait if any locked table is data-versioned |
| `MD_WRITE`: metadata changes and DDL | `ACCESS EXCLUSIVE` | `ACCESS EXCLUSIVE` | Always wait |
| `FINALIZE`: complete pending operations | `ACCESS EXCLUSIVE` | `ACCESS EXCLUSIVE` | Always wait |

Operational writers can hold `ROW EXCLUSIVE` concurrently; Postgres row locks resolve conflicting row updates.
Data-versioned writers use `EXCLUSIVE`, which allows only one writer at a time, so each write builds on the
previous version.
Both modes allow `ACCESS SHARE`, so these table locks do not block readers.

Lock modes apply per table. For example, a schema change can lock its target with `ACCESS EXCLUSIVE` and
its bases with `ACCESS SHARE`.

The wait policy applies to the whole transaction. If a data read or write locks only operational tables,
it uses `NOWAIT` for every table lock. A conflict reports `SCHEMA_CHANGE_IN_PROGRESS`. If any locked table
is data-versioned, the transaction waits for every required table lock. Schema changes and finalization
always wait, regardless of versioning.

### Metadata-only reads

Catalog listings and descriptions read metadata without locking store tables. They don't wait for a schema change
in progress, so the schema they return may already be out of date by the time the caller uses it. A subsequent data
operation acquires its own locks and refreshes metadata.

A metadata read can still encounter pending operations while loading a table. It then finalizes those operations
before continuing; finalization takes locks and may have to wait (section 7).

If an expression reads another table, for example through a `@pxt.query` UDF, the transaction needs a read lock
on that table too. Section 8 describes a current limitation.

## 3. Which objects to lock

An operation's lock set is the store tables and directory rows that need to be locked before that operation can be
safely executed:

- A read locks every table in each `TableVersionPath`, including bases. It does not lock descendant views.
- A data write locks its target and all mutable descendant views, since writes propagate to those views.
- A schema change locks its targets and affected mutable views. Creating or deleting a mutable view changes
  which views receive writes from its mutable base and advances the base's `view_sn`. Both operations therefore
  need an exclusive lock on the base. Finalization acquires that lock as well when dropping a mutable view.
- Dropping a directory locks its subdirectories and tables. Dropping a table also locks its descendant views,
  including snapshots, and the directories containing them.
- A pure snapshot has no store table. Its path includes the stored base versions to read; its own metadata is
  protected by its parent directory lock when created or deleted.

Acquire all store table locks in a single global order, sorted by store table name. Then acquire directory
locks in catalog-path order. Every transaction follows this order to avoid deadlocks.

An operation can require several modes on the same store table, for example when it writes to a table and also
reads it as the base of a view. Lock each store table once, in the strongest required mode. Do not upgrade locks
partway through the transaction.

## 4. Build, lock, and validate

To build the lock set, Catalog needs metadata about views and catalog paths. Reading that metadata in the
operation's transaction would establish its snapshot before locking. Catalog therefore builds a tentative
lock set before opening the transaction:

1. Build the set before opening the operation's transaction (`_resolve_lock_set()`).
2. Open the transaction and lock store tables, then directory rows (`_acquire_locks()`).
3. Refresh metadata and check for pending operations (`_refresh_md_cache()`). Pending operations abort this attempt.
4. Check that the held locks cover the refreshed targets (`_validate_lock_set()`). Retry if they do not.
5. Run the caller's operation.

Use cached `TableVersion` metadata when available. It supplies store table names, bases, mutable views, and each
table's versioning. On a cache miss, read raw `tables` rows in a separate metadata transaction
(`_lock_set_from_store()`).

Catalog paths always require a store read because Catalog does not cache directory structure or path resolution.
Both cached and stored results can become stale before lock acquisition.

A table's ancestors, store table name, and versioning are fixed at creation, though an ancestor may have been
dropped. Mutable views and catalog paths can change and must be checked after locking.

### Why a tentative lock set is safe

Before validation, the transaction only acquires locks, reads metadata, and checks for pending operations.
It performs no caller data reads, row writes, metadata changes, or media I/O, so no costly work is lost if the tentative
lock set turns out to be stale.

Validation runs once, so a validated set must stay valid until the transaction ends. Creating or dropping a mutable
view requires `ACCESS EXCLUSIVE` on its base, which conflicts with every table lock Pixeltable uses, and changing a
directory's entries requires its directory lock. Once the set is validated, other transactions cannot make changes
that invalidate it until this transaction commits or rolls back.

### Retrying a stale lock set

Two conditions raise `_StaleLockSetError`:

- A store table in the set no longer exists. `LOCK TABLE` raises Postgres `UndefinedTable`.
- Refreshed metadata requires a table or directory lock that was not acquired, or a stronger table lock.

Roll back, clear the cached metadata used to build the tentative lock set, and rebuild it from the store. If
the operation's own target was dropped, report `TABLE_NOT_FOUND`. If only a descendant view disappeared, retry
the operation with that view removed from the lock set.

## 5. Directory locks

Directories have no store tables, so directory operations lock rows in `dirs`. Acquire these locks after all
store table locks. Taking a directory lock first would establish a snapshot before a possible wait for a store table
lock.

Directory locks are acquired in the order of the directories' catalog paths, compared component by component: a
directory comes before its subdirectories, and sibling directories are ordered by name. For example, `a` is locked
before `a/b`, which is locked before `a/c` and `b`.

Directory locks update `lock_dummy`, even if its value is unchanged. A transaction may already have a snapshot
when it waits for this lock. If the preceding lock holder updated the row after that snapshot, Postgres raises
a serialization failure. The waiting transaction then retries with a fresh snapshot.

`SELECT ... FOR UPDATE` would not provide this guarantee: a holder could release the lock without changing the
row, leaving the waiter with stale directory contents and no serialization failure.

A table can be created between store table locking and directory locking. Post-lock validation must detect
that the set is incomplete and retry before doing any work.

The parent directory lock also protects names during creation, deletion, and moves. A new table's id is not
visible until its metadata commits, so other transactions cannot yet lock its store table. Pure snapshots have
no store table at all; their metadata is inserted or deleted under the parent directory lock.

## 6. Store tables and metadata commit together

For tables with physical storage, a committed `tables` row must have a corresponding store table, and a
committed store table must have its metadata row. Pure snapshots are the exception because they have no store table.

Create the store table and its indexes in the transaction that inserts its metadata. Drop the store table in
the transaction that deletes its metadata. Postgres DDL is transactional, so rollback removes both changes.

A missing store table therefore means the tentative lock set is stale. Recovery never needs to create a store
table for metadata that has already committed. Creation also needs no `IF NOT EXISTS` handling: the parent
directory lock ensures that only one creator can claim the name.

## 7. Pending operations between transactions

A schema change can span several transactions. Locks are released between them, so locks alone cannot keep data
operations out of an unfinished change.

**A table with pending operations is not usable.** A transaction that finds them aborts its attempt, finalizes
them, and retries. Finalization waits for its locks, even when the operation itself would use `NOWAIT`.

Pending operations run in one of two ways:

- With `needs_xact = True`, finalization executes the operation and records its completion in the same transaction,
  while holding `ACCESS EXCLUSIVE`.
- With `needs_xact = False`, finalization executes the operation outside that transaction and records completion
  in the next transaction. These operations must be idempotent because multiple processes can try to complete them.
  `CreateStoreColumnsOp` and `CreateStoreIdxsOp` run each DDL statement in its own transaction. Statements use
  `IF NOT EXISTS` and acquire `ACCESS EXCLUSIVE`, explicitly for index creation and implicitly for `ALTER TABLE`.
  `DeleteTableMediaFilesOp` deletes a dropped table's media files.

### Finalization always waits

The process that started the change normally finalizes its operations. Another process that encounters them
can also help finish the change. Both wait for `ACCESS EXCLUSIVE`.

This allows recovery after the original process dies: its locks are released, and the next operation touching
the table finishes the pending work. There is no background process responsible for recovery.

## 8. Runtime checks

The implementation checks key locking requirements:

- `_lock_tables()` asserts that targets are sorted by name and that no store table is already locked.
- `_assert_tbl_record_write_locked()` requires `EXCLUSIVE` or stronger on the store table for a table metadata
  write. A new table and a pure snapshot are exempt: instead, they require a lock on the parent directory.
- `StoreBase` calls `assert_rows_write_locked()` before writing store rows.

`SqlNode` calls `check_rows_read_locked()`, which currently warns rather than asserts. Writes do not yet include
all tables read by computed-column `@pxt.query` UDFs in their lock set (PXT-1343). Once that is fixed, the warning
can become an assertion.
