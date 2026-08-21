# Primary Key Support for Operational Tables (PXT-1101)

Status: Plan

Companion to `op.md`, which does not cover primary keys.

## Current State

`create_table(..., primary_key=..., _is_data_versioned=False)` fails with
`AssertionError: TODO: implement for operational tables [PXT-1101]` at `store.py:220`, raised while building
`sa_tbl`. The assert fires *after* the table md is written, so the failed create leaves a table entry that can no
longer be loaded — `drop_table` and `drop_dir` both re-hit the same assert (see "Fail before writing md" below).

How PK works for data-versioned tables today:

- A partial unique btree index `pk_idx_<tbl_hex>` over the PK store columns, `WHERE v_max = MAX_VERSION`
  (live rows only), with string columns wrapped in `left(col, BtreeIndex.MAX_STRING_LEN)` (256).
- Violations are mapped to user-facing errors in `StoreBase.sql_insert`: `UniqueViolation` on a `pk_idx_`
  constraint becomes `CONSTRAINT_VIOLATION` / "Duplicate primary key value"; `ProgramLimitExceeded` becomes
  "Primary key value too large for index".
- PK is creation-time only: `InsertableTable._create` sets `col.is_pk`, rejects unknown and nullable columns.
- PK columns are immutable: `_validate_update_spec` rejects them (`table_version.py:1481`) and
  `create_batch_update_plan` subtracts them from the update set (`plan.py:714`), since they are the lookup key.
- `pk_idx_` is a *system* index built directly in `create_sa_tbl` from `is_pk` columns. It is not an entry in
  `tv.idxs`, so it has no index value column and no undo column, and the index framework never touches it.

## Phase A — Creation-Time PK Enforcement (self-contained, ~1 day)

This is all that is needed for `create_table(..., primary_key=..., _is_data_versioned=False)` plus `insert()`.
Independently shippable.

1. **`store.py:create_sa_tbl`** — replace the assert with a branch:
   - data-versioned: today's partial index with `left()` truncation, unchanged.
   - operational: a plain `unique=True` btree on the **raw** PK store columns — no `postgresql_where` (there are
     no expired rows) and **no string truncation**, matching the `BtreeIndex(uses_value_col=False)` policy adopted
     for operational tables in #1515 and stated in `op.md` ("do not truncate long strings").
   - Keep the `pk_idx_` name prefix so the error mapping in `sql_insert` keeps working untouched.
   - Use an *index*, not a `PrimaryKeyConstraint`/`UniqueConstraint`: `StoreBase.create()` reconciles an existing
     store table by looping `sa_tbl.indexes` with `CREATE INDEX IF NOT EXISTS` and has no
     `ALTER TABLE ADD CONSTRAINT` path. An index needs zero changes there. `rowid` remains the Postgres PK —
     views depend on it for the FK / `ON DELETE CASCADE` and for rowid joins.

2. **Error-path parity** — both handlers in `sql_insert` key off the index name and apply as-is. Note the size
   error becomes reachable in a new way: because strings are not truncated, a single PK value over the btree row
   size limit (~2700 bytes) now fails. This is consistent with `test_oversized_index_key`; accept and test it.

3. **NULL semantics** — a unique index treats NULLs as distinct. PK columns must be non-nullable and Pixeltable
   validates on write, so parity with data-versioned tables is fine. Postgres here is 16.10, so
   `postgresql_nulls_not_distinct=True` is available if we want belt-and-braces. Recommend parity plus a test
   asserting a null PK value is rejected by validation.

4. **Redundant default index** — with `has_default_idxs=True`, `create_initial_md` adds a `BtreeIndex` to every
   eligible column, including PK columns. On an operational table the unique PK index is *complete*, so that
   default btree is fully redundant; skip it for PK columns. (On data-versioned tables it is not redundant, since
   `pk_idx_` is partial and truncated.)

5. **Fail before writing md** — `_create_table` writes md and then the `CreateStoreTableOp` roll-forward asserts,
   wedging the table permanently. Any operational-table restriction we keep must be validated in
   `InsertableTable._create` / `create_initial_md` as a `RequestError`, before `write_tbl_md`. Worth a defensive
   audit of the other `PXT-1101` asserts reachable from the create path for the same failure class.

No metadata migration and no schema-version bump: `is_pk` is already persisted in `ColumnMd`, and no operational
table with `is_pk` can exist today because creation always failed. Unlike `convert_49`, there is nothing to
backfill.

## Phase B — Updates: Nothing PK-Specific

Operational-table updates translate directly into SQL `UPDATE` statements, and a Pixeltable PK index is just a
unique Postgres index, so Postgres maintains it with no cooperation from us. Delete-then-reinsert of the same key,
update-then-reinsert, and cascading view recomputation all work with no PK-specific code.

For data-versioned tables, PK correctness is entangled with update mechanics: uniqueness holds only because the
index is partial on `v_max = MAX_VERSION` and because soft-delete-then-insert expires the old row before the new
one lands. None of that applies to operational tables.

Two non-issues, recorded so they are not re-litigated:

- **Same-statement PK permutation** (e.g. `SET id = id + 1` transiently violating an immediately-checked unique
  index) is unreachable, because PK columns are never written — see "Current State" above. No
  `DEFERRABLE INITIALLY IMMEDIATE` constraint is needed, which is what would have forced the
  `ALTER TABLE ADD CONSTRAINT` reconciliation in `create()`.
- **Index framework interaction** — none: `pk_idx_` is not in `tv.idxs`, so `_build_update_columns` and the undo
  column machinery never see it.

What does remain:

- **Unique violations under concurrency** — two concurrent writers inserting (or upserting) the same key. Newly
  reachable on operational tables, since data-versioned writes are serialized. It already behaves correctly:
  `sql_insert` maps it to `CONSTRAINT_VIOLATION`, and `catalog.py`'s retry loop deliberately does not retry 23505
  (only `SerializationFailure` / `LockNotAvailable` / `DeadlockDetected`). So this is a test plus a sentence in
  `op.md`'s Write-Write Conflicts section, not code. Open question: whether
  `batch_update(if_not_exists='insert')` should retry rather than surface, since its lookup-then-insert is not
  atomic across writers. Recommend surfacing in v0.
- **Scheduling dependency, reversed** — `_rowid`-keyed `batch_update` needs int→UUID widening for operational
  tables (`rowids: list[tuple[int, ...]]` in `local_table`, `plan.py`, `exec/RowUpdateNode`). Until that lands,
  the PK is the *only* usable key for `batch_update` on an operational table, which makes Phase A a prerequisite
  for operational-table `batch_update` rather than a follow-on to it.

## Phase C — Peripheral Surfaces (~half a day of audits)

- **Views** — `create_view` takes no `primary_key`, and `primary_key_columns()` is per-`TableVersion`, so
  operational views need no work. Add a `RequestError` (not an assert) if any path can propose a PK'd view.
- **io / import** — `create_table(source=..., primary_key=...)` routes PK through `TableDataConduit`
  (`table_data_conduit.py:152`); works once Phase A lands. Add a test with `_is_data_versioned=False`.
- **serving / dashboard** — `serving/_fastapi.py` identifies rows by `is_primary_key` in two places; verify those
  endpoints against an operational table (they may need the UUID rowid fallback).
- **Model API** — `metadata/utils.py:76` asserts `is_pk` never changes across a model diff. With PK reachable on
  operational tables, make that a user-facing error instead of an assert.
- Optionally extend `StoreBase.validate()` to assert the `pk_idx_` index exists when the table has PK columns.

## Testing

Parametrize `tests/test_primary_key_index.py` over `data_versioned`: single and composite PK, duplicate
rejection, delete-then-reinsert, batch-with-duplicate atomicity, oversized PK. Plus operational-only cases:

- long string PKs that do **not** collide at 256 characters (the inverse of `test_string_pk_truncation`)
- `reload_catalog` survival
- `pg_indexes` shows a unique, non-partial index
- `MultiThreadedScenario`: two writers inserting the same PK concurrently
