# Catalog Locking

How `Catalog` synchronizes concurrent access to Pixeltable tables. This document covers the principles and the
invariants; the code in `pixeltable/catalog/catalog.py` is the specification.

## 1. The central idea

**Pixeltable synchronizes table access with Postgres table locks: `LOCK TABLE` on store tables.** The store table
is the lock, and Postgres's own conflict matrix is the protocol; it already encodes "DDL excludes everything, DML
excludes nothing but DDL". One protocol covers both table kinds, data-versioned and operational.

Every lock is transaction-scoped. The lock target is a store table, whose name is derived from the table id:
`tbl_<uuid.hex>` for tables, `view_<uuid.hex>` for views.

### Locks come before the snapshot

`LOCK TABLE` is a utility statement, and Postgres exempts it from taking a snapshot. Under REPEATABLE READ the
snapshot is therefore pinned by the first query after the locks are held, which in a Pixeltable transaction is the
metadata read. A transaction that waited five minutes for a lock still reads metadata as of the moment the lock was
granted.

Two things follow, and the rest of the design rests on them:

- **The lock statements must be the first statements in the transaction.** Anything that pins the snapshot first
  reintroduces the staleness in full: a metadata read, a `dirs` row lock, even a `SELECT 1`.
- **No snapshot-currency validation is needed anywhere.** Nothing has to detect that the `tables` row changed after
  our snapshot was pinned and raise a serialization failure, because a lock holder's snapshot is current by
  construction. The one exception is the `dirs` row lock, which is not first (§5).

## 2. Operation classes, modes and wait policies

Each transaction is opened through one of `Catalog.begin_*_xact()` or the matching `retry_*_loop()` decorator, and
that choice names an operation class (`TblOpClass`). The operation class determines the lock mode, and affects
the choice of wait policy.

| Operation class | Lock Mode: operational | Lock Mode: versioned | Waits?: operational | Waits?: versioned |
| --- | --- | --- | --- | --- |
| `MD_READ`: catalog md, no table data | none | none | n/a | n/a |
| `READ`: query rows | `ACCESS SHARE` | `ACCESS SHARE` | no | yes |
| `DATA_WRITE`: insert/update/delete | `ROW EXCLUSIVE` | `EXCLUSIVE` | no | yes |
| `MD_UPDATE`: md write, DDL | `ACCESS EXCLUSIVE` | `ACCESS EXCLUSIVE` | yes | yes |
| `FINALIZE`: pending-op finalization | `ACCESS EXCLUSIVE` | `ACCESS EXCLUSIVE` | yes | yes |

Only `DATA_WRITE` class differs between data-versioned and operational tables. `ROW EXCLUSIVE` is self-compatible,
so operational writers share the table and resolve genuine conflicts on store row locks. `EXCLUSIVE` conflicts with
itself, which serializes the writes that a linear version history requires for data-versioned tables. Both are
compatible with `ACCESS SHARE`.

**Readers are never blocked by writers of either kind, and writers are never blocked by readers.**

**Lock Mode is per table; wait policy is per operation.** A single lock set mixes modes, because a write locks its
target in `ACCESS EXCLUSIVE` and its base tables in `ACCESS SHARE`: it writes the one and reads the others. Each
`LockTarget` therefore carries its own mode. The wait policy is a single decision for the transaction, since the
caller either waits for the locks it needs or it doesn't.

- **Operational reads and writes fail fast** (`NOWAIT`, reported as `SCHEMA_CHANGE_IN_PROGRESS`). Only a schema
  change conflicts with the modes they ask for.
- **Versioned reads and writes wait**, so a versioned operation always eventually succeeds.
- **Schema changes and finalizations always wait**, on both kinds. §7 explains why a finalization can never fail
  fast.

A lock set spanning both kinds would have two policies and no single answer. Nothing produces one today, because an
operational table's queries are restricted to a single table in the from-clause, so `_make_lock_set()` asserts
instead of choosing.

### Metadata-only reads take no locks

`pxt.get_table()`, `list_tables()`, `get_dir_contents()` and `describe()` read catalog metadata, not rows, and
therefore lock nothing. They may observe a schema change in progress and report a schema that is about to change,
which is better than blocking or failing; any operation the caller then performs on the returned handle locks
properly.

The distinction is per call site, not per entry point. A query's expressions can pull in tables whose data is read,
through a `@pxt.query` UDF for instance, and those go through `begin_read_xact()`, which locks them.

## 3. The lock set

**A transaction's lock set is the store table of every table it touches.**

- A **read** locks every table on each `TableVersionPath` it reads, bases included, because a view read reads base
  data. It locks nothing else: a read never recurses into views.
- A **data write** locks its target and every transitive mutable view, because the write propagates there.
- A **schema change** locks the same set as the equivalent write. `create_view` and `drop_view` additionally lock
  the base's store table: adding or dropping a mutable view changes how base writes propagate and bumps the base's
  `view_sn`, so it has to exclude concurrent base writes.
- A **drop** also reaches the whole catalog subtree named by its catalog path (`'dir.subdir.tbl'`): for a directory
  its subdirectories and their tables, for a table its views, snapshots included.
- A table with no store table contributes nothing. Pure snapshots have none, and their bases are already on the
  `TableVersionPath`.

**Acquisition order: one global sort of the whole set by store table name, and nothing else.** The same relative order
of lock acquisition guarantees freedom from deadlocks.

**No lock upgrades.** Each store table is locked exactly once per transaction, in the strongest mode any of its roles
within the transaction needs.

## 4. Guess, lock, validate

Every transaction that locks anything follows the same five steps:

1. **Guess the lock set**, before the transaction opens (`_resolve_lock_set()`).
2. **Open the transaction and acquire the locks**: store table locks first, then the `dirs` row locks (§5). Nothing
   may precede them (§1).
3. **Read current metadata**. A table with pending ops aborts the attempt here (§7).
   Steps 2 and 3 are both in `_acquire_locks()`.
4. **Validate the guess** against that metadata (`_validate_lock_set()`). On a mismatch, abandon the attempt and
   retry with a corrected set.
5. **Do the work.**

Steps 1 and 4 exist because the lock set of a write depends on metadata: which views exist, and which table a
catalog path such as `'a.b.c'` names. But metadata may only be read after the locks are held (§1). The set is
therefore split by whether its shape can change:

- **A `TableVersionPath`'s ancestry is immutable.** `ViewMd.base_versions` is written once, at creation, and no
  operation re-parents a view. A table's kind and its store table name are likewise fixed at creation, so none of
  the attributes the ancestor part of a lock set draws from the cache can change. It needs no validation. It can
  still name a table that has since been dropped, which the recovery below covers.
- **A mutable tree, and the table a catalog path resolves to, can change.** `pxt.drop_table('a.b.c')` cannot name
  `tbl_<uuid.hex>` without resolving `'a.b.c'` first, and by the time the locks are held that path may name a
  different table, or none at all. Both are guessed before the transaction opens, locked, and then checked against the
  current metadata from the store. On a mismatch the attempt is abandoned and retried.

Where the guess comes from, in order of preference:

1. **The existing metadata cache** (`Catalog._tbl_versions`). Everything a lock set needs is already on a cached
   `TableVersion`: `is_view` gives the store table name, `base` gives the ancestor chain, `mutable_views` gives the
   tree, `is_data_versioned` gives the table kind.
2. **The store**, read in a separate read-only transaction before the operation's transaction opens
   (`_lock_set_from_store()`). It interprets plain `tables` rows instead of building `TableVersion`s in order to avoid
   dealing with pending table ops. What it returns is current and needs no validation. A transaction with a write path
   always takes this route, since nothing caches which table a catalog path names.

### Why guessing is safe

**Nothing happens before validation.** Between opening the transaction and the check, i.e. steps 2 and 3, the only
things that run are lock acquisition, the metadata read, and the pending-ops check. No data read, no DML, no md
write, no media I/O. The one exception is nominal: taking a `dirs` row lock is an `UPDATE` (§5), which assigns a
constant to a dummy column and rolls back with the attempt. A wrong guess therefore costs one aborted attempt that
did no work and left nothing behind, and that is what makes guessing acceptable at all: a guess can be arbitrarily
wrong without any consequence beyond that attempt.

**A lock on a table freezes its set of mutable views.** Creating or dropping a mutable view of table T is a
metadata update on T: it takes `ACCESS EXCLUSIVE` on T's store table and bumps T's `view_sn`. Every lock Pixeltable
ever holds conflicts with `ACCESS EXCLUSIVE`. Once the whole tree is locked its shape cannot change, so one
validation pass is enough and there is no window between validating the set and using it. This argument is about
holding the locks, not about the order they were taken in, which is why the global sort of §3 does not weaken it.

### The two ways a guess fails

Both raise `StaleLockSetError` and share one recovery:

- **A store table in the set is gone**: another process dropped that view, and this one has not read metadata
  since.
  Postgres cannot lock what is not there, so `LOCK TABLE` raises `UndefinedTable` and the transaction is unusable.
- **The set does not cover the full tree the current metadata describes**, which the post-lock validation reports.

The recovery: roll back, drop the cached metadata the set was derived from, rebuild the lock set from the store, and
retry. The operation then either proceeds (to step 2) on the corrected set, or reports `table_was_dropped` if it found
that its own target's md is gone.

## 5. Directories

A directory has no store table, so a `dirs` row is the only thing a directory operation can lock. Two lock types
then
coexist in one transaction, in two phases:

1. **Store table locks** (`LOCK TABLE`), acquired in the name order.
2. **`dirs` row locks**, acquired in order by catalog path.

The consistent order of lock acquisition ensures no deadlocks.

**Store table locks have to come first.** A `dirs` row lock is a statement that pins the snapshot. Put it ahead of
`LOCK TABLE` and §1 no longer holds: a transaction that then waits for a store table lock wakes with a snapshot from
before the wait and reads stale table metadata.

**Sorted by catalog path.** Any total order gives deadlock freedom, exactly as in §3; the path is simply the key
`_lock_set_from_store()` already has for every directory it collects.

**A `dirs` row is locked by a blind `UPDATE`** of its `lock_dummy` column, not by `SELECT … FOR UPDATE`. Writing
the row matters precisely because the directory lock is not first: a transaction can wait in phase 2 with its
snapshot already pinned. Since taking the lock is itself an `UPDATE`, whoever held it before us moved the row
version, so our `UPDATE` raises a serialization failure and we retry against a fresh snapshot. Under
`SELECT … FOR UPDATE`, a holder that only read the directory under lock commits without touching the row, and we
would wake with a stale snapshot, no error, and stale directory contents.

Locking a directory last means a table can be created under it in between the two phases, so the set of store tables can
under-cover. That is §4's situation again: nothing has happened yet, so re-reading and restarting costs one
work-free attempt.

**A pure snapshot is a name in a directory and nothing else.** It has no store table, so no store table lock can
cover it. Its `tables` row is only ever inserted or deleted, and every path that does that locks the parent
`dirs` row first.

Creating a table or a view is covered the same way, for a different reason: it does get a store table, created in
the same transaction (§6), but no other transaction can contend for it, because the id appears in no committed row.
The parent's `dirs` X-lock is held to protect against other transaction taking the same table name in the same dir.

## 6. Store table existence

"The store table is the lock" needs the store table to exist. The invariant that guarantees it: **a committed
`tables` row implies an existing store table, and vice versa.**

The invariant is maintained by keeping the DDL that creates and drops store tables inside the metadata transaction
on both ends. `CREATE TABLE`, with its system and user indexes, runs in the transaction that inserts the `tables`
row, and `DROP TABLE` runs in the one that deletes it. These statements are transactional in Postgres, so a failed
or retried creation rolls the store table back with the md. Everything created at that point is empty, so the DDL
is cheap.

Two things the invariant gives us:

- **Recovering from a missing store table is the single branch of §4.** There is no state to reconstruct, and no
  need to answer "is this a create in progress, a dead creator, a drop in progress, or a corrupt catalog".
- **Store-table creation needs no idempotence machinery.** `IF NOT EXISTS` everywhere and swallowed
  duplicate-object errors would only be needed because two processes can concurrently roll the same pending op
  forward. Under the parent directory lock, exactly one process ever creates the store table.

## 7. Pending ops: the gap no lock spans

Locks are transaction-scoped, but a schema change spans several transactions. Between them no lock is held, so
operations like `add_computed_column` or `add_embedding_index` are not one continuously locked stretch.

What keeps data operations out of those gaps is the rule that **a table with pending ops is not usable**: a
transaction that reads md and finds pending ops aborts, and either finalizes them or reports that a schema change
is in progress.

The work inside a pending op is separately protected. An op with `needs_xact = True` runs inside the
finalization's own transaction, under its `ACCESS EXCLUSIVE` lock, and its completion is recorded there too. The
rest (`CreateStoreColumnsOp`, `CreateStoreIdxsOp`) run outside that transaction, each statement in a transaction
of its own, and their completion is recorded only afterwards. That is why they have to be idempotent and safe to
run concurrently, and it is also what lets them lock: every statement runs with `IF NOT EXISTS` and under
`ACCESS EXCLUSIVE`, taken explicitly for `CREATE INDEX` and implicitly by `ALTER TABLE`. Two processes that reach
the same op therefore serialize, and the second finds the work already done rather than repeating it.

### Who finalizes, and why it always waits

A finalization is run either by the **owner**, the process that performed the schema change, rolling its own ops
forward right after the md transaction; or by a **helper**, any other process whose operation ran into the pending
ops, aborted, and now clears them so that it can proceed. Both wait for `ACCESS EXCLUSIVE`, and neither ever fails
fast.

Waiting is what makes an abandoned schema change recoverable. An owner that has died holds no lock, so the next
operation to touch the table becomes a helper and finishes the roll-forward. Nothing else would: there is no background
process to notice.

## 8. How the protocol is enforced

Asserts catch a wrong operation class, or a statement reached without the lock it needs, instead of leaving
either to code review:

- `_lock_tables()`: the targets are sorted by name, and none is already locked.
- `_assert_md_write_locked()`: a metadata write holds a **self-conflicting** mode on the store table, `EXCLUSIVE` or
  `ACCESS EXCLUSIVE`, depending on the table kind.
- `assert_rows_write_locked()` / `assert_rows_read_locked()`: called from `StoreBase` and `SqlNode`, which own every
  statement that reads or writes store rows.
- `_make_lock_set()`: the set does not span both table kinds (§2).

