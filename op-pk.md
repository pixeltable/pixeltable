# Primary Key Support for Operational Tables (PXT-1101)

Status: Phase A implemented; Phase B/C blocked (see below)

Companion to `op.md`, which does not cover primary keys.

## Current State

Before Phase A, `create_table(..., primary_key=..., _is_data_versioned=False)` failed with
`AssertionError: TODO: implement for operational tables [PXT-1101]` while building `sa_tbl`. The assert fired
*after* the table md was written, so a failed create left a table entry that could no longer be loaded —
`drop_table` and `drop_dir` both re-hit the same assert (see "Fail before writing md" below).

How PK works for data-versioned tables today:

- A partial unique btree index `pk_idx_<tbl_hex>` over the PK store columns, `WHERE v_max = MAX_VERSION`
  (live rows only), with string columns wrapped in `left(col, BtreeIndex.MAX_STRING_LEN)` (256).
- Violations are mapped to user-facing errors in `StoreBase.sql_insert`: `UniqueViolation` on a `pk_idx_`
  constraint becomes `CONSTRAINT_VIOLATION` / "Duplicate primary key value"; `ProgramLimitExceeded` becomes
  "Primary key value too large for index".
- PK is creation-time only: `InsertableTable._create` sets `col.is_pk`, rejects unknown and nullable columns.
- PK columns are immutable: `_validate_update_spec` rejects them (`table_version.py:1232`) and
  `create_batch_update_plan` subtracts them from the update set (`plan.py:714`), since they are the lookup key.
- `pk_idx_` is a *system* index built directly in `create_sa_tbl` from `is_pk` columns. It is not an entry in
  `tv.idxs`, so it has no index value column and no undo column, and the index framework never touches it.

## Phase A — Creation-Time PK Enforcement (self-contained, ~1 day)

This is all that is needed for `create_table(..., primary_key=..., _is_data_versioned=False)` plus `insert()`.
Independently shippable.

1. **`store.py:create_sa_tbl`** — replace the assert with a branch:
   - data-versioned: today's partial index with `left()` truncation, unchanged.
   - operational: a `UniqueConstraint` over the **raw** PK store columns — no predicate (there are no expired
     rows) and **no string truncation**, matching the `BtreeIndex(uses_value_col=False)` policy adopted for
     operational tables in #1515 and stated in `op.md` ("do not truncate long strings"). It goes in
     `extra_constraints`, next to the `PrimaryKeyConstraint` over the rowid columns that is already there.
   - Name it `pk_{tbl_version.id.hex}` — not `pk_idx_`, which is a misnomer for a constraint and stays on the
     data-versioned index. Both names are produced by one helper, `store.pk_constraint_name()`, which
     `sql_insert`'s two error handlers also use. That single source of truth is load-bearing: the handlers
     originally hardcoded the `pk_idx_` prefix, and renaming the constraint without touching them let raw
     `IntegrityError`s escape instead of `CONSTRAINT_VIOLATION`. Postgres names a unique constraint's backing
     index after the constraint, so the name is what appears in `pg_indexes` and in violation messages either
     way.
   - `rowid` stays the Postgres `PRIMARY KEY`, so the Pixeltable PK cannot be one: a table gets one PK, and
     `ADD PRIMARY KEY` would additionally require `NOT NULL` store columns, which Pixeltable deliberately does
     not declare — see `create_sa_tbl`, "all storage columns are nullable (we deal with null errors in
     Pixeltable directly)".

2. **Error-path parity** — both handlers in `sql_insert` work for both kinds once they resolve the name through
   `pk_constraint_name()` instead of hardcoding a prefix. The `UniqueViolation` handler compares
   `diag.constraint_name`, which is the constraint name; the `ProgramLimitExceeded` handler matches the same
   string against the message text, which names the backing *index*. Note the size error
   becomes reachable in a new way: because strings are not truncated, a single PK value can now exceed the btree
   row size limit (~2700 bytes). It has to be *incompressible* to do so — Postgres pglz-compresses an oversized
   index entry before rejecting it, so a long run of one character still fits. Consistent with
   `test_oversized_index_key`; accept and test it.

3. **NULL semantics** — unaffected by the constraint-vs-index choice: both treat NULLs as distinct, and both
   accept `NULLS NOT DISTINCT`. PK columns must be non-nullable and Pixeltable validates on write, so parity
   with data-versioned tables is fine. Postgres here is 16.10, so `postgresql_nulls_not_distinct=True` is
   available if we want belt-and-braces. Recommend parity; no new test needed, since the rejection happens in
   `InsertableTable._create` before any store-table work and is already covered for both table kinds by
   `test_schema_spec` (`tests/test_table.py:2039`).

4. **Fail before writing md** — `_create_table` writes md and then the `CreateStoreTableOp` roll-forward asserts,
   wedging the table permanently. Any operational-table restriction we keep must be validated in
   `InsertableTable._create` / `create_table_version_md` (`catalog/utils.py`) as a `RequestError`, before
   `write_tbl_md`. Worth a defensive audit of the other `PXT-1101` asserts reachable from the create path for
   the same failure class.

5. **Make the creation-time invariant checkable** — the constraint is created only by `CREATE TABLE`, so
   correctness depends on the PK definition never changing after creation (see below). Guard it from the tests
   through the public API only: a duplicate insert must fail with `CONSTRAINT_VIOLATION`. A passing duplicate is
   the observable symptom of a lost constraint, and it needs no catalog introspection. See Testing.

No metadata migration and no schema-version bump: `is_pk` is already persisted in `ColumnMd`, and no operational
table with `is_pk` can exist today because creation always failed. Unlike `convert_49`, there is nothing to
backfill.

### Why a `UniqueConstraint`, not a unique index

Postgres implements a unique constraint *as* a unique index, so enforcement, error text, and every planner
optimization that exploits uniqueness (`relation_has_unique_index_for`, join removal, `DISTINCT`/`GROUP BY`
elision, single-row selectivity) are identical either way — those all read `pg_index.indisunique`, never
`pg_constraint`. The choice is therefore about what the extra `pg_constraint` row buys, at the cost of the
expressiveness a bare index has (partial predicates, expressions, `CREATE CONCURRENTLY`, independent `DROP`).

For an operational table the index's extra expressiveness is unused — the PK is a plain, untruncated column
list. What the constraint buys:

- **Catalog-declared intent.** `information_schema.table_constraints` is where BI tools, ORMs, dbt, and
  SQLAlchemy reflection look for a unique key; a bare index is invisible to all of them. Operational tables are
  meant to be ordinary Postgres tables that external tooling can read directly, so this is the leading argument.
- **A named arbiter for upsert.** `INSERT ... ON CONFLICT ON CONSTRAINT pk_idx_<hex> DO UPDATE` is available.
  See the `batch_update(if_not_exists='insert')` note in Phase B — this is what makes that path atomic across
  writers. `ON CONFLICT (cols)` inference also works against a bare index, but naming the constraint cannot
  accidentally match a user-created unique index over the same columns.
- **FK targetability.** `REFERENCES tbl(pk_cols)` requires a unique or PK constraint; a bare unique index is
  rejected. Nothing needs this today (there are no FKs between store tables — `ForeignKey` appears only in
  `metadata/schema.py`, for the catalog tables), but the option is kept rather than spent.
- **`DEFERRABLE` stays reachable.** The PK *definition* is immutable, which is what makes the constraint safe;
  the constraint's one exclusive capability concerns PK *values* becoming mutable. If `_validate_update_spec`
  ever relaxes, `DEFERRABLE INITIALLY IMMEDIATE` is the only way to allow a same-statement PK permutation, and
  an index can never be made deferrable.

The cost, stated plainly: `StoreBase.create()` reconciles indexes with `CREATE INDEX IF NOT EXISTS`, so a
missing index self-heals; there is no `ADD CONSTRAINT IF NOT EXISTS`, so a missing constraint has no repair
path. This is safe today because `create()` has exactly one caller (`CreateStoreTableOp.exec`,
`tbl_ops.py:97`) — its "table already exists" branch is reachable only on retry of that same op or from a
concurrent creator, and in both cases the `CREATE TABLE` that made the table already carried the constraint.
Schema changes never call `create()`; they go through `add_column`/`drop_column` plus an in-memory
`create_sa_tbl()`. But that makes "PK is creation-time only" load-bearing rather than merely convenient, which
is what item 5 above guards.

Secondary cost: promotion is free (`ALTER TABLE ... ADD CONSTRAINT ... UNIQUE USING INDEX`, catalog-only) while
demotion requires an index rebuild, since `DROP CONSTRAINT` drops the backing index. So the index would have
been the more reversible choice. Accepted: a one-time rebuild on a decision we do not expect to reverse.

## Phase B — Updates: Nothing PK-Specific

**Not implementable yet, and nothing here needs to be.** Operational-table updates do not exist:
`TableVersion.update` (`table_version.py:1114`), `batch_update` (`:1164`), `recompute_columns` (`:1275`), and
`propagate_update` (`:1328`) all still `assert self.is_data_versioned`. This phase is therefore a note for
whoever implements those, recording the conclusion that they need no PK-specific work — not a work item of its
own.

The reasoning: operational-table updates translate directly into SQL `UPDATE` statements, and a Pixeltable PK is
just a unique Postgres constraint, so Postgres maintains it with no cooperation from us. Delete-then-reinsert of
the same key, update-then-reinsert, and cascading view recomputation should all work with no PK-specific code.

For data-versioned tables, PK correctness is entangled with update mechanics: uniqueness holds only because the
index is partial on `v_max = MAX_VERSION` and because soft-delete-then-insert expires the old row before the new
one lands. None of that applies to operational tables.

Two non-issues, recorded so they are not re-litigated:

- **Same-statement PK permutation** (e.g. `SET id = id + 1` transiently violating an immediately-checked unique
  constraint) is unreachable, because PK columns are never written — see "Current State" above. So the
  constraint does not need to be declared `DEFERRABLE`; it just leaves that door open if PK columns ever become
  writable.
- **Index framework interaction** — none: `pk_idx_` is not in `tv.idxs`, so `_build_update_columns`, the undo
  column machinery, and `_offending_idx` (which iterates `tv.idxs`) never see it. Nor is it in
  `sa_tbl.indexes`, whose only consumer is the reconciliation loop in `create()`.

What does remain:

- **Unique violations under concurrency** — *not* reachable, contrary to an earlier draft of this plan.
  Operational-table writes are serialized by the same catalog write lock as data-versioned ones:
  `_acquire_write_lock` takes `SELECT ... FOR UPDATE NOWAIT` on the table's md row (`catalog.py:752`), and a
  second writer retries on `LockNotAvailable` until the first commits. Verified empirically — two writers
  inserting the same key never have conflicting inserts in flight; the loser observes the committed row and gets
  the ordinary duplicate error, exactly as in the single-threaded case. There is therefore nothing
  concurrency-specific to test, and nothing to add to `op.md`'s Write-Write Conflicts section.
  Still open for whenever `batch_update` reaches operational tables: `if_not_exists='insert'` does a
  lookup-then-insert, which the table lock makes safe today, but if that lock is ever relaxed for operational
  tables the atomic form is `INSERT ... ON CONFLICT ON CONSTRAINT pk_idx_<hex> DO UPDATE`. Naming the constraint
  as the arbiter is one of the reasons for choosing a constraint over an index (see Phase A).
- **Scheduling dependency, reversed** — `_rowid`-keyed `batch_update` needs int→UUID widening for operational
  tables (`rowids: list[tuple[int, ...]]` in `local_table`, `plan.py`, `exec/RowUpdateNode`). Until that lands,
  the PK is the *only* usable key for `batch_update` on an operational table, which makes Phase A a prerequisite
  for operational-table `batch_update` rather than a follow-on to it.

## Phase C — Peripheral Surfaces

Only the first item is actionable. The other three are blocked behind operational-table features that do not
exist yet, so PK is not what is holding them up; they are listed with their blocking assert so nobody re-audits
them.

- **io / import** — done, and needs no new test. `create_table(source=..., primary_key=...)` routes PK through
  `TableDataConduit` (`table_data_conduit.py:152`) into the ordinary insert path, and nothing under
  `pixeltable/io/` references `is_data_versioned` — the conduit resolves a source into rows and is otherwise
  blind to the table kind. So the two halves are already covered independently and cannot interact: PK threading
  through the conduit (including the non-nullable coercion) by `test_import_pandas_csv`
  (`tests/io/test_pandas.py:133`), and PK enforcement on an operational table by `test_primary_key_index`. Note
  that the existing import tests could not be parametrized anyway: `import_csv`/`import_pandas` take no
  `_is_data_versioned` argument, and adding one to a public API for a test's sake is not worth it.
- **Views** — blocked: `create_view` asserts `is_data_versioned` (`globals.py:391`), so operational views do not
  exist and no path can propose a PK'd view. The `RequestError` guard has nothing to guard yet.
- **serving / dashboard** — blocked: the row-identifying endpoints (`_fastapi.py:1619`, `:2246`) route through
  `update`/`batch_update`, which assert. They fail on operational tables for reasons unrelated to PK.
- **Model API** — blocked: `create_from_model` asserts `is_data_versioned` (`catalog.py:1797`), so the `is_pk`
  assert at `metadata/utils.py:76` is unreachable for operational tables. Turn it into a user-facing error when
  the Model API gains operational-table support; that is also where the creation-time-only invariant behind
  Phase A item 5 should be enforced.

## Testing

Everything is verified through the public API — a duplicate insert failing with `CONSTRAINT_VIOLATION` is the
observable proof that the constraint exists, so no test reads `pg_constraint` or `information_schema`.

Parametrize all of `tests/test_primary_key_index.py` over the `is_data_versioned` fixture
(`tests/conftest.py:343`, ids `data_versioned`/`operational`). Where a feature is not yet supported on
operational tables, gate that part of the test body with `if is_data_versioned:` rather than leaving the test
data-versioned-only — the house pattern, cf. `tests/test_index.py:1056` and `tests/test_alter_column.py:30`.
When op-table support lands, the guard is deleted and the coverage is already written.

**Runs unchanged on both** — `test_single_pk`, `test_composite_pk`,
`test_batch_with_duplicate_fails_atomically`, `test_prohibited_pk_col_ops`, and
`test_pk_index_row_too_large`. Note that the last one already exceeds the btree row size *with* truncation (11
string columns × 256 chars), so it raises on both kinds for the same reason; delete-then-reinsert also carries
over, since `delete` has no `is_data_versioned` assert.

**Gated behind `if is_data_versioned:`** because the feature is unimplemented for operational tables —
`TableVersion.update` and `TableVersion.batch_update` still assert (`table_version.py:1114`, `:1164`). That is
`test_batch_update_with_pk_index` plus any update-driven assertion elsewhere in the file. Delete the guards when
op-table update support lands; the coverage is then already written.

**Not parametrized** — `test_string_pk_truncation` stays data-versioned-only and unchanged: truncation is the
one place the two kinds genuinely diverge, so there is no shared body to parametrize. Its operational
counterpart is a separate test, `test_long_string_pk`:

- strings that differ only past 256 characters are distinct keys, because the raw value is indexed. This is the
  assertion that would fail if the operational branch ever reintroduced truncation.
- a single *incompressible* PK value over the btree row size limit raises "Primary key value too large for
  index". Newly reachable — a data-versioned table truncates the same value to 256 and accepts it. It has to be
  incompressible: Postgres pglz-compresses an oversized index entry before rejecting it, so `'b' * 4096` fits
  after all. Uses the same seeded `random.choices` construction as `test_oversized_index_key`
  (`tests/test_index.py:1568`).

**Extend `test_single_pk`** — it already covers duplicate rejection and `reload_catalog` survival. Add a non-PK
`add_column` plus `drop_column` between the insert and the reload. That is the case worth adding: those paths
rebuild `sa_tbl` through `create_sa_tbl()` without calling `create()`, so they are what would expose the
constraint being lost, and the assertion is just that a duplicate insert still fails afterwards.

**No concurrency test.** The scenario it was meant to cover cannot occur — see Phase B: writes to one table are
serialized by the catalog write lock, so two writers never have conflicting PK inserts in flight. A second
writer observes the committed row and gets the same duplicate error `test_single_pk` already asserts. An attempt
at this test hung for exactly this reason (the blocked writer held the table lock while the other retried on
`LockNotAvailable` to the scenario timeout), which is how the claim was disproved.
