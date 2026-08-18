Operational Tables Design

Status: Implementation

# Overview

Standard Pixeltable tables serialize all updates. Every row create, update, and delete creates a new table version. This works well for data versioning (data lineage), but the downside is that it creates a fundamental performance bottleneck that restricts workload scaling.

Operational Tables is the new feature that does away with that limitation by giving up data history. The state of the store table will reflect only the current state of the Pixeltable rows, and not the entire history.

## Goals

- Introduce a new type of table with better read and write latency and throughput, and better scalability
- Feature parity with versioned tables except where it doesn’t make sense
- Reuse the existing query planning, execution, expression evaluation, etc. as much as possible
- Unreachable data (deleted rows, dropped columns, deleted Pixeltable-owned media files) is eventually deleted

## Non-Goals

- Non-blocking schema changes

## Terminology

**Data-versioned tables (aka versioned tables in this doc)**: already existing Pixeltable tables that preserve data lineage with time travel, revert, etc.

**Operational tables**: new type of tables designed for performance and with no data lineage.

# Public API

Operational tables are created using the same function that creates versioned tables today:

```python
tbl = pxt.create_table('test', schema, data_versioned=False)
```

`data_versioned`  defaults to `False`.

# Data Model

## Store Format

Store columns of a data-versioned table:

- Pixeltable System Columns
    - [PK] rowid: a unique ID (an int) of a Pixeltable table row. Used to link base table rows with iterator view rows.
    - [PK] pos_0, pos_1, … pos_N: (iterator views only) position of view rows within the parent table. 1 column per view “depth level”.
    - [PK] v_min: the lowest version at which this row version is live (visible)
    - v_max: the lowest version at which this row version is no longer live (`2^63-1` for live rows)
- User columns, some combination of:
    - col_0: regular column value
    - col_1: computed column value
    - col_1_cellmd: cell metadata of a computed column
    - col_2: index value column
    - col_3: index undo column (for fast rollback)
    - … col_N: etc.

For operational tables, we remove v_min, v_max, and index undo columns:

- Pixeltable System Columns
    - [PK] rowid (UUID)
    - [PK] pos_0, pos_1, … pos_N (iterator views only)
- User columns, some combination of:
    - col_0
    - col_1
    - col_1_cellmd
    - col_2: index value column ← still needed for e.g. embedding indexes, but undo column is no longer needed.
    - … col_N

## Rowid

Rowid in an operational table is a pseudorandom UUID, which could be a UUID4 or UUID7 depending on what the underlying store prefers. (UUID7 as a primary key is preferred for local PostgreSQL)

Who generates rowid — Postgresql or Pixeltable — does not matter too much. It would be slightly efficient if it was PG, however the way the code is structured and the rows flow today with versioned table, it’s likely be much simpler if Pixeltable generated the UUIDs just like it generates integer rowids for versioned tables.

## Views

Since operational table’s PK only includes rowid and pos_i columns, an operational view can have a FK reference to its base table. That enforces structural integrity of views, and simplifies row deletion thanks to ON DELETE CASCADE.

## Versioning

Internally, data-versioned tables track table “versions” (row changes) and “schema versions” (schema changes) (the user sees all of these in a unified timeline).

For operational tables, we will keep only the schema history for observability purposes, and to support database branching in the future. The following features won’t be supported:

- Serializable writes
- Data lineage/time travel
- Snapshot views
- `t.head()` and `t.tail()`

With the exception of Pixeltable-owned media files (more on that in Garbage Collection), the rows deletion, and column drop will take effect fully and immediately in operational tables.

## Indexes

### B-tree

Index value columns are not needed for operational tables, except for String column types. We truncate long strings (to 256 characters) for the purposes of b-tree indexes in versioned tables, which would justify a separate index column value in an operational table.

TODO: investigate where this decision comes from, and if it needs to extend to operational tables as a special case.

TODO: op operational tables, do not truncate long strings. That’s going to be the user’s responsibility.

Index undo columns are not needed.

### Embedding

Embedding indexes still need an index value column, just like in versioned tables.

Undo columns are not needed for operational tables.

## Store Column Types

For versioned tables, user columns, regardless of their nullability, are represented by a nullable store column. That is to support add column and drop column on an existing table — a column is not going to have any values for row versions for which it’s not visible.

Data columns in operational tables do not have that problem, so we can consider pushing down the (Pixeltable) column type nullability to the store column. It would add an extra correctness enforcement on the store side (a NULL value cannot be inserted in NOT NULL column).

But computed and index value columns that were added after table creation still start off with no values in it.

What this means is that the added complexity of being more strict with the store column types may not be worth it in the end as there’s very that it gets us (Pixeltable already validates the column data on write).

## Interaction with Versioned Tables

Base tables and views: views of operational tables are always operational; views of versioned tables are (as before) always versioned.

Type conversion: no type conversion is allowed between operational and versioned tables.

Mixed statements: operations involving both types of tables are allowed. That includes joins and other mixed queries, as well as insert/update/delete statements.

# Operations

REPEATABLE READ which is already used for versioned tables, is used for operational tables as well, as it appears to strike a good balance of isolation and performance.

## Read Path

Operational table reads translate pretty directly into store queries. Unlike versioned table reads, we don’t need to worry about row versioning/visibility in a particular version. All row data in the store is current.

However any query should start with a shared lock acquisition (more on that in Schema Change Isolation), and a metadata read/validation.

All in all, an operational table query looks like this:

1. Start a REPEATABLE READ store transaction. Acquire a shared advisory lock on the table
2. If TableVersion is in cache: validate cached md by comparing table version counter and view_sn with the stored values. If TableVersion is not in cache, read, initialize, and cache it.
3. Execute the actual user query
4. Commit transaction
    1. This automatically releases all advisory locks

If multiple tables are involved, the setup is repeated for each one.

## Write Path

Unlike a versioned table, writing into an operational table does not create a new metadata version. Two writes can run and succeed simultaneously, as long as the data changes are not conflicting according to REPEATABLE READ. More on conflicts in Concurrency Model.

The sequence is very similar to reads:

1. Start a REPEATABLE READ store transaction. Acquire a shared lock on the table.
2. Cache or validate table md
3. Execute the plan
    1. Updates literally update store rows. No soft deletion like in versioned tables.
    2. The same with deletes
    3. The same with inserts
4. Commit transaction

## View-Tree propagation

The way changes are propagated to operational view from their bases diverges from that of versioned tables.

Versioned tables’ propagation plans rely heavily on row versions to tell which rows need to be propagated. For example, the way insert is implemented is, all rows are first inserted in the base table. Then, for each mutable view, the propagation plan simply selects from the base table by v_min == base.current_version to find the rows to propagate.

Updates are implemented as soft-delete-then-insert, so they work similarly.

Operational tables don’t keep track of row versions, so the view propagation has to work differently.

For insert and update, 4 high-level approaches are possible.

### Approach 1

Reuse the existing “waterfall” method that versioned tables use. Add an accessory INT column to store tables that contains xact_id of the latest transaction that updated the row. Create an index on it.

On insert or update, set `xact_id=pg_current_xact_id()`

To propagate base table rows, select from base table where `xact_id=pg_current_xact_id()` 

Pros: this approach fits within the existing view propagation plan. Only the mechanism to discover rows to propagate is different.

Cons: some added storage cost

We also considered making it a BOOLEAN column that contains True on rows that were updated within the current transaction. The values would be reset to False before commit. But the extra round trip to reset the values makes it worse than xact_id.

TODO: find out how the index is stored, i.e. all transactions write to the same pages, or each transaction gets its own private set of pages until it commits.

### Approach 2

Reuse the existing “waterfall” method that versioned tables use.

```python
CREATE UNLOGGED TABLE changed_rows (xact_id, table_uuid, rowid);
-- PRIMARY KEY (xact_id, table_uuid, rowid);
```

Create this table in pg to track rowids that were changed in the current transaction. I.e. on base table insert or update, also insert rows into `changed_rows`. Then, during the view propagation, select from `changed_rows` to find out which rows to propagate. Before committing the transaction, delete from `changed_rows` (nothing should ever be committed to this table).

`UNLOGGED` speeds up writes by skipping WAL. We don’t care about WAL since no changes to `changed_rows` need to be durably committed.

TBD: what should be the primary key order of these 3 columns?

Pros: this approach fits within the existing view propagation plan. Only the mechanism to discover rows to propagate is different.

Cons: some added cost on the PG side

A possible tweak: use a private-to-transaction temporary table that’s automatically dropped on commit. This is more convenient but pollutes pg schema and locks pg catalog every time.

### Approach 3

Reuse the existing view propagation plan, but keep the affected rowids in memory.

The base insert/update plan remembers and returns the rowids that were affected. The propagation plans use that for selecting and propagating rows from base table to views. For nested view propagation, row_i values will be added to rowids.

Pros: fits within the existing view propagation plan.

Cons: memory pressure concerns, especially on large inserts/updates (e.g. with a large number of affected rows across the entire mutable tree)

### Approach 4

Create a new insert/update plan for operational tables that takes care of the entire mutable tree on a per row basis.

Insert:

1. Create an exec plan that computes new stored values not only for the base table, but all mutable views recursively
2. Split up the incoming rows in batches
3. Run each batch through the plan, and insert the produced rows in all mutable views

Update:

1. Similarly, create an exec plan that computes the new values for the entire mutable tree of views
2. Select rows from the base table that satisfy the where clause of update
3. Batch those rows up, and run them through the plan
4. Take the plan’s output and replace the view rows with the new ones
    1. For iterator views this means delete all existing rows and insert new ones

Note: PostgreSQL doesn’t allow issuing a new operation until the result of the last one is consumed, which means that one can’t send inserts and updates while still iterating on a query cursor, like this algorithm wants to do. However, PG has cursors:

```python
BEGIN ISOLATION LEVEL REPEATABLE READ;

DECLARE select_from_base_table CURSOR FOR SELECT ... FROM base_table;

-- collect a batch of rows
FETCH NEXT FROM select_from_base_table;

-- process the batch and send inserts/updates to the base table and the views
UPDATE base_table SET ... WHERE ...;
UPDATE view_table SET ... WHERE ...;

-- and so on
FETCH NEXT FROM select_from_base_table;

COMMIT;
```

Note: in a REPEATABLE READ transaction, we see our own writes (i.e. the uncommitted writes performed the same transaction). That applies to FETCH NEXT just as to any other read. However in this case it shouldn’t be a problem because the base table rows are always read first, then updated.

Pros: this scales much better for large inserts and updates than Approach 3 since it doesn’t require keeping all affected rowids in memory. It can also be more efficient in some cases because it optimizes the execution on the entire plan level rather than table level (think base table computations are GPU-intensive, and view computations are external-service-intensive). It works for versioned tables just as well, so we don’t have to have two parallel implementations.

Cons: it is a significant departure of what we are doing for versioned tables, and it would take a substantial effort to build this. Not clear how to batch inserts and updates well (think iterator fanout).

### Approaches Summary

| Approach | Pros | Cons |
| --- | --- | --- |
| 1 (xact_id system column) | Simple, works within the existing design. Does not require extra round trips. | Minor per-row added storage cost |
| 2 (“affected rowids” system table) | Relatively simple, no per-row storage penalty | Extra round-trips to insert per statement to insert and delete from this table |
| 3 (affected rowids in memory) | No extra round-trips to the store or storage cost | Can’t handle massive updates |
| 4 (redesign view propagation) | New design eliminates some round trips, is potentially more efficient in some scenarios, and works for all table types | Substantial effort to implement |

### Deletion Propagation

Deletion propagation to mutable views is easy: as we mentioned above, the store tables for views are created with ON DELETE CASCADE, so Pixeltable only deletes from the base table, and the store takes care of the rest.

## Index Maintenance

All indexes are maintained by Postgres fully automatically. pgvector vectors sit in the same storage engine, so any vector operations inherit ACID properties of the transaction.

# Concurrency Model

## Isolation and Transaction Scope

REPEATABLE READ (RR) is the isolation level of choice for transactions on operational tables.

### Key RR properties

- First data access within a transaction pinpoints a snapshot at which all future reads within that transaction will be performed, so all reads are consistent with one another, and no phantom reads
- Read-your-writes: writes are visible to subsequent reads within the same transaction
- Writes can block until the conflicting transaction commits or rollbacks
- If it commits, the conflict loser transaction fails with serialization failure and needs to start over

Why not READ COMMITTED: RC does not provide snapshot isolation, which is difficult to reason about, leads to unintuitive behaviors and odd bugs (we read base table metadata → it has a view → other process drops the view → we attempt to read view metadata but it’s not there). For those reasons, we upgraded from RC to RR for versioned tables.

Why not SERIALIZABLE: it’s an overkill for our purposes, is very resource-intensive (predicate locks, etc.) and results in serialization failures in scenarios that we don’t need it to.

Pixeltable has application-level snapshot isolation for versioned tables: when the table metadata is read at the top of the transaction, Pixeltable pins the table version, and performs all reads at that version by carefully selecting rows by v_min/v_max. Thus, at least in theory, versioned tables can run on any of the 3 isolation levels supported by PG. That is not the case for operational tables since they lack app-level versioning. For that reason, the choice of the right isolation level for operational table transactions is foundational to their correctness, and to the end user experience.

### Single Transaction

For reasons described just above, it is crucial that every user operation (except schema change), on operational tables runs in one rather than multiple PG transactions:

1. Begin RR transaction
2. Acquire shared table locks (more on that in the next section)
3. Read or validate table metadata
4. Perform the actual CRUD operation
5. Commit transaction

Schema changes are exempt because Pixeltable Catalog fully locks down tables to execute a schema change. Schema change execution can happen across multiple transactions because no write is possible in the affected table(s) until all pending ops are resolved.

## Schema Change Isolation

Schema changes are to be implemented via pending table ops, i.e. the same way as in versioned tables.

However in the absence of writes and schema change serialization (either in the store aka SERIALIZABLE isolation or on the Pixeltable level), extra care needs to be taken for operational tables so that overlapping operations such as schema changes, row reads and writes, do not bring unexpected (incorrect or failed) outcomes.

Some examples of possible problematic scenarios that we want to avoid:

- An embedding index is being created on the table. A concurrent write inserts or updates a row. The timing is unlucky, and the index is missing the value (or has incorrect value) for the affected row.
- A computed column is being created. A concurrent select(*) that is executed on the table returns the new column with only partially computed values.

To keep things simple in the first version, we allow for the following limitations:

1. Every read and write on an operational table starts with a table metadata read
2. Ongoing schema changes, even the long running ones such as add computed column or add index, completely block reads and writes on an operational table.

*(Design review: continue here)*

### Advisory Locks

> PostgreSQL provides a means for creating locks that have application-defined meanings. These are called *advisory locks*, because the system does not enforce their use — it is up to the application to use them correctly. Advisory locks can be useful for locking strategies that are an awkward fit for the MVCC model. For example, a common use of advisory locks is to emulate pessimistic locking strategies typical of so-called “flat file” data management systems. While a flag stored in a table could be used for the same purpose, advisory locks are faster, avoid table bloat, and are automatically cleaned up by the server at the end of the session.
> 

https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS

Use pg’s advisory locks to control the concurrency during schema changes:

- Schema changes, including finalize pending table ops, call `pg_advisory_xact_lock()` to obtain an exclusive lock on the table
- Reads and writes call `pg_advisory_xact_lock_shared()` to obtain a transaction-level shared lock on the table
- Alternatively, reads and writes call `pg_try_advisory_lock_shared()` which does the same thing, but fails fast if the lock is unavailable. This way we can fail table operations if a schema change is ongoing (particularly a long-running one), instead of blocking them.

### Lock ID Implementation Details

Pg locks are identified by two 32-bit integers (or one 64-bit int). Each operational table is assigned a unique 32-bit lock id. To lock a table: pg_advisory_xact_lock(0, t.lock_id). The top 0 dedicates this 32-bit namespace to operational table locks, which leaves the door open for other future applications of advisory locks.

Pixeltable can rely on PG for lock id generation and unique constraint. A sequence can generate ids for operational tables:

```python
CREATE SEQUENCE table_lock_id;
```

And a unique index to enforce no duplicates:

```python
CREATE UNIQUE INDEX tbl_lock_id_unique ON tables (((md->>'lock_id')::int));
```

When lock_id is not present, which is the case for any versioned table, this index works exactly like we need it to: entries like that are simply ignored, and there can be any number of them.

As an alternative, Pixeltable can generate unique lock ids at the application level, but that is undesirable because there’s significant complexity in doing it right.

### Table Locking Implementation Details

Catalog is responsible for locking and unlocking operational tables (actually unlocking happens automatically upon transaction end). This makes sense because Catalog already owns locking versioned tables for writes and other changes.

`Catalog.begin_xact()` has new optional parameters `op_tbls_to_shared_lock`  and `op_tbls_to_exclusive_lock` . `lock_mutable_tree` is extended to apply to operational tables and their views.

Similarly to versioned table locking, Catalog keeps track of the lock state of the current transaction, and it enforces that the correct types of locks are held for metadata reads and writes, and for finalizing pending table ops. Catalog also has public functions (e.g. `Catalog.is_op_tbl_locked(lock_id)`) that allow Table to validate that locks are in place for CRUD ops on table rows.

### Lock Wait Policy

CRUD ops

When a CRUD operation attempts to acquire the shared lock on the table but the lock is not available, two things are possible: the operation can fail immediately or after a short timeout, or it can block until the lock becomes available and proceed then.

For a highly performant type of table under normal circumstance, failing fast with a clear message and appropriate error code appears to be less surprising behavior to the end user than blocking. It also prevents active connections and transactions from piling up on the server.

With that in mind, such operations should be calling `pg_try_advisory_xact_lock_shared()`.

Schema changes

The opposite reasoning applies to schema changes. There is a lot less expectation from the performance of a schema change, so blocking on the lock seems like a reasonable policy for this kind of operations. Besides, if the table is experiencing a steady stream of reads or writes, the time when the table lock is exclusively available may never come, leading to effectively starvation on a schema change. Waiting on the pg lock via `pg_advisory_xact_lock()` , on the other hand, ensures that the schema change eventually gets to go ahead. Postgres advisory locks, while not perfectly fair, do avoid starvation.

### Deadlocks

In order to avoid deadlocks, operations that require the entire mutable tree locked (such as base table write or a schema change that affects views) lock tables in the base-before-views order, which is the same policy as versioned tables.

## Write-Write Conflicts

Writes under REPEATABLE READ acquire an exclusive row-level lock. If the lock is already held by another transaction, the current one (the conflict loser) gets suspended until the lock is available again. At that point, if the conflict winner committed its change, the loser gets a serialization failure and needs to retry.

This policy avoids the lost updates problem, which is a known issue with READ COMMITTED. However there are downsides, including the wait and the retries that are sometimes unexpected.

The following example demonstrates a logically unnecessary transaction failure (two transactions update different values on the same row):

```python
-- process 1
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- ...
UPDATE tbl SET a = 1 WHERE id = 0;
COMMIT;

-- process 2, concurrently:
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- ...
UPDATE tbl SET b = 1 WHERE id = 0;
COMMIT;
```

Still, if these two transactions overlap, the loser will be spuriously aborted, which can lead to expensive work lost.

If this becomes a real issue that Pixeltable customers experience, we can consider explicitly locking rows for update before doing anything expensive.

### Retry policy

It’s best if Pixeltable retries serialization failures, and other retriable errors, rather than surfacing them to the user. Versioned tables retry automatically, so it seems fitting if operational tables will also retry automatically. Besides, implementing the correct retry policy is not very straightforward (think randomized delay jitter, max attempts), therefore it’s best to lift that burden from the user.

Retrying inside Pixeltable opens the door to more efficient retries in the future. As mentioned in the section above, by adjusting the point at which the affected rows are locked for write, we can reduce or eliminate throwaway computation.

The downside of retrying automatically is that it hides real contention issues with the workload from the user.

## Metadata Caching

The table metadata caching of versioned tables translates very well to operational tables. The way it works is: at the end of every transaction, every mutable table in cache gets `is_validated = False`. The next time the table is accessed, that prompts a metadata validation against the store. The validation is relatively cheap: it compares table version and `view_sn` of the stored state. If they match, the cache entry is valid. Otherwise, the metadata is fully reloaded from the store, which is more expensive than the initial check.

Unlike a versioned table, an operational table’s version is not incremented on every write, so data changes in the table do not invalidate the metadata cache. Which is exactly the behavior that we need.

# Garbage Collection

Pixeltable is a multimodal database that utilizes two storage systems under the hood: PostgreSQL for metadata and trivially-sized user values, and blob store/local filesystem for media. Files created by Pixeltable (like output of a computed column) are owned by Pixeltable, and as such, we are responsible for garbage collecting them when they become inaccessible.

Apart from dropping a table entirely, versioned tables have no way of permanently deleting user data, so before now there was no pressing need to properly GC media (although there’s a known problem https://pixeltable.atlassian.net/browse/PXT-1219 that comes of out of that).

But for operational tables, some kind of GC is necessary. When a media file becomes inaccessible to the user because it was deleted or replaced in the table, the file needs to be physically deleted. Apart from simply being a reasonable expected behavior, it is crucial for satisfying GDPR, CCPA and similar regulations.

Note: forever deletion is a desired feature for versioned tables, too, so garbage collection should be extendable to them.

However, simply deleting a file when a row with a reference to that file is updated or deleted doesn’t work. There are multiple issues with it:

- No shared transactions: file operations are not atomic with PG transactions, unless some kind of two-phase commit protocol is implemented. Without such a protocol, we can’t guarantee that files are created and deleted atomically with table rows that reference them. Orphaned files or dangling references are possible.
- Concurrent reads: process 1 starts a long-running query on the table, process 2 in the meantime deletes rows with media. Thanks to snapshot isolation, process 1 continues to see the deleted rows in PG. But if process 2 already deleted the files, process 1 will see dangling references.
- Reference duplication: there are ways to get the same files to be referenced by more than one row in the table, or even different tables. Thus updating and deleting one of those reference does not automatically make the file safe to delete.

These issues call for a more sophisticated GC algorithm.

TODO: if a media file can only be referenced by a single row, does it make GC any easier? Think a journal of files to be deleted in the store

## Approach 1: Mark and Sweep GC

The basic idea behind Mark and Sweep is as follows:

1. List all Pixeltable-owned files in the file store
2. Query from PG all files that are still referenced, and subtract them from the set from the previous step.
3. What remains are the files that are eligible for deletion.

Mark and Sweep GC is relatively expensive and should not run too frequently. Its cost is a product of the amount of data in the database, regardless of the number of files that are eligible for deletion.

When it comes to scheduling, GC can simply run periodically on each database. Various improvements on this are possible that avoid GCing databases more frequently than necessary.

Pros: relatively simple to understand and implement (but by no means easy)

Cons: not very efficient for scenarios with low churn

*This approach is recommended due to a considerable lower effort to implement than the alternative.*

## Approach 2: Reference Counting GC

Keep track of all Pixeltable-owned files in PG, along with the number of references to each one. When a reference to the file is added, increment the counter in the same transaction. When a reference is deleted, decrement it. Files whose reference count reached 0 are eligible to be physically deleted.

Pros: this method does not require a full file store and database scan. The amount of effort to GC is proportional to the number of eligible files.

Cons:

- Fragile: there is a large number of scenarios and codepaths that result in store data changes, and this GC is completely broken unless each such codepath correctly updates reference counts
- Difficult to implement: there’s a lot of trickiness to doing it right

## Recently created files

Pixeltable-owned media files are created before the PostgreSQL transaction commits, which means that all such files begin their life as orphaned. Therefore, regardless of the GC approach we take, recently created files must be exempt from it.

## Defensive soft deletion

One useful practice that is possible within garbage collection is to soft delete unreachable objects first. It can look like a separate table in the store that lists files that are about to be deleted. A file from that list should never be accessed by Pixeltable. If we detect such access attempt, issue a critical alert and automatically disable garbage collection.

Soft-deleted files are hard-deleted after awhile and removed from the list.

It takes a little extra effort to implement this, but the payoff can be great.

# List of Features and Delivery Schedule

| Feature | Planned for Operational Tables? | Delivery Milestone | Notes |
| --- | --- | --- | --- |
| Indexes (b-tree, embedding) | ✅  | v0 |  |
| Computed columns | ✅ | v0 |  |
| Unstored columns | ✅ | v0 |  |
| Queries and CRUD ops except those listed separately | ✅ | v0 |  |
| Mixed queries/DMLs | ✅ | v2+ | CRUD operations that involve both table types |
| Sample queries | ✅ | v0. Postpone to v1 if it’s any extra effort. | TBD: support seed or not? The underlying data can change, therefore reproducibility cannot be guaranteed in general case. Decision: there’s no way to do this without a seed. |
| Aggregation queries | ✅ | v0 |  |
| Head(), tail() queries | ❌ |  | Technically possible to implement, but it’s up for debate if these functions make sense. Note: UUID7 is ordered by timestamp, so `order by rowid` in these ops can produce sensible and useful result for end user. |
| Blocking schema changes | ✅ | v0 |  |
| Non-blocking schema changes | ✅ | v2+ | Reads and writes can run while a long-running schema operation is being executed |
| Live views, iterators | ✅ | v0 | Except sample views maybe? TBD |
| RDBMS import&export | ✅ | v0 |  |
| Media GC | ✅ | v1 |  |
| Basic random ops coverage similar to versioned tables | ✅ | v0 |  |
| Random ops coverage with correctness validation | ✅ | v2+ |  |
| Automated performance testing | ✅ | v2+ |  |
| Lock-free reads and writes | ✅ | v1 |  |
| History (t.get_versions) | Schema changes only | v0 |  |
| Snapshot views | ❌ |  | Impossible by design |
| Time travel | ❌ |  | Impossible by design |
| Revert | ❌ |  | Impossible by design |
|  |  |  |  |
|  |  |  |  |

# Testing

## Locking Protocol Correctness

The correctness of the locking protocol is verified by:

- “unit” tests that utilize `MultiThreadedScenario`
- random ops — we need to come up with some ways to asses the correctness of query results
