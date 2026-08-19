# Table Locking — Design and Implementation Plan

Target stores: local PostgreSQL, PlanetScale PostgreSQL. CockroachDB is out of scope.

## 1. Summary

Locking is done with Postgres's native `LOCK TABLE` statement on the **store tables** of the tables an operation
touches. There are no lock ids, no advisory-lock namespace, and no lock metadata: the relation is the lock, and
Postgres's own conflict matrix — which already encodes "DDL excludes everything, DML excludes nothing but DDL" —
is the protocol.

**This is one protocol for both table kinds, and it replaces the current protocol for data-versioned tables as
well as supplying the new one for operational tables.** Data-versioned tables today exclude concurrent writers
and schema changes with `SELECT … FOR UPDATE NOWAIT` on the `tables` row plus `UPDATE lock_dummy`; that mechanism
goes away entirely and is replaced by relation locks. The two kinds then differ in exactly two parameters — **how
a data write locks** (shared among writers vs. exclusive) and **whether an operation waits or fails fast** —
and in nothing else. §11 lists the resulting behavior changes for existing versioned-table users; §7.5 step 7 is
where they land.

### 1.1 Operation classes and lock modes

Four operation classes, one lock mode each per table kind:

| Operation class | Operational: mode | Operational: wait policy | Versioned: mode | Versioned: wait policy |
| --- | --- | --- | --- | --- |
| **Read** — any query | `ACCESS SHARE` | `NOWAIT` | `ACCESS SHARE` | block |
| **Data write** — insert/update/delete | `ROW EXCLUSIVE` | `NOWAIT` | `EXCLUSIVE` | block |
| **Metadata update** — md write, DDL | `ACCESS EXCLUSIVE` | block | `ACCESS EXCLUSIVE` | block |
| **Pending-op finalize** | `ACCESS EXCLUSIVE` | block (owner) / `NOWAIT` (helper) | `ACCESS EXCLUSIVE` | block (owner) / `NOWAIT` (helper) |

The single mode difference is the data-write row: `ROW EXCLUSIVE` is self-compatible, so operational writers
share the table; `EXCLUSIVE` is self-conflicting, so versioned writers serialize — which is what a linear version
history requires. Both are compatible with `ACCESS SHARE`, so neither blocks readers. Everything else is the same
mode for both kinds, and only the wait policy differs.

Pending-op finalize takes the same mode as a metadata update but is listed separately because its wait policy
depends on who is running it (§4.4) and because the pending-ops rule, not the lock, is what excludes operations
in the gaps between its transactions (§10.1).

### 1.2 Blocking matrix — operational tables

Rows are the operation already in flight and holding its locks; columns are the operation arriving afterwards.
Both matrices are derived from Postgres's mode-conflict table, measured in §9.2, and what each class does is
spelled out in §4.

| in flight ↓ / arriving → | Read | Data write | Metadata update | Pending-op finalize |
| --- | --- | --- | --- | --- |
| **Read** | proceeds | proceeds | waits | waits (owner) / fails fast (helper) |
| **Data write** | proceeds | **proceeds** | waits | waits (owner) / fails fast (helper) |
| **Metadata update** | **fails fast** | **fails fast** | waits | waits (owner) / fails fast (helper) |
| **Pending-op finalize** | **fails fast** | **fails fast** | waits | waits (owner) / fails fast (helper) |

### 1.3 Blocking matrix — data-versioned tables

| in flight ↓ / arriving → | Read | Data write | Metadata update | Pending-op finalize |
| --- | --- | --- | --- | --- |
| **Read** | proceeds | proceeds | waits | waits (owner) / fails fast (helper) |
| **Data write** | proceeds | **waits** | waits | waits (owner) / fails fast (helper) |
| **Metadata update** | **waits** | **waits** | waits | waits (owner) / fails fast (helper) |
| **Pending-op finalize** | **waits** | **waits** | waits | waits (owner) / fails fast (helper) |

The two matrices differ in exactly the five bolded cells, all of which follow from the two parameters above.
One is the mode difference: concurrent data writes share the table on an operational table and serialize on a
versioned one. The other four are the wait-policy difference, in one block: an arriving read or data write fails
fast against an in-flight metadata update or pending-op finalize on an operational table, where on a versioned
one it waits. Every other cell is identical for the two kinds.

What both matrices share is the invariant that matters most: **readers are never blocked by writers of either
kind, and writers are never blocked by readers.** That is the MVCC property we must not lose, and it is why the
modes are drawn from Postgres's matrix rather than from a plain shared/exclusive lock.

### 1.4 Why `LOCK TABLE` and not advisory locks

Two things make this work, and both are properties of `LOCK TABLE` specifically:

1. **`LOCK TABLE` does not pin the REPEATABLE READ snapshot.** It is a utility statement that Postgres exempts
   from taking a snapshot (`PlannedStmtRequiresSnapshot()` excludes `LockStmt`). So the snapshot is pinned by the
   *first query after* the locks are held — the metadata read. A transaction that waited five minutes for a lock
   still reads metadata as of the moment it was granted. Measured, §9.1: an advisory lock in the same scenario
   leaves the waiter with a pre-wait snapshot; `LOCK TABLE` leaves it with a fresh one.
2. **Consequently there is no snapshot-currency validation anywhere.** The second job of today's
   `SELECT … FOR UPDATE NOWAIT` — raising `SerializationFailure` when the `tables` row changed after our snapshot
   was pinned — becomes unnecessary for every table that has a store table, which under §5's invariant is every
   table that is visible at all. Metadata updates keep taking it as a uniform serialization point, but it is
   load-bearing in only two places: pure snapshots, which have no relation of their own, and creation's T1,
   which writes the `tables` row in the same transaction that creates the relation (§4.3).

Performance is not a counterweight: measured in §9.4, `LOCK TABLE` and advisory locks are indistinguishable on
the read and data-write paths, and the one mode that costs more — `ACCESS EXCLUSIVE`, by ~20µs on a single
relation and growing with the size of the lock set (~40µs at 16, ~95µs at 64), plus a WAL flush — is taken only
by schema changes, which already run DDL costing orders of magnitude more.

The hard requirement this buys with: **the lock statements must be the first statements in the transaction.**
Anything that pins the snapshot first — an md read, a `SELECT 1` — reintroduces the staleness in full. Measured,
§9.1, case 3.

### What this removes

- `SELECT … FOR UPDATE NOWAIT` + `UPDATE lock_dummy` as the write/schema-change exclusion mechanism for
  **data-versioned** tables, and `Table.lock_dummy` itself (`dirs.lock_dummy` stays — it is the name-slot lock
  of §5.4).
- The retry storm on contended versioned writes. Today two concurrent inserts collide on `FOR UPDATE NOWAIT` →
  `LockNotAvailable` → sleep → redo the whole operation. Now the second one waits in Postgres's lock queue and
  then proceeds with a fresh `current_version`/`next_row_id`: wait-then-succeed instead of fail-then-redo.
- Lock ids, the id sequence, the unique constraint, key bit-arithmetic, the `TableMd`/column plumbing, the
  metadata version bump, the converter, and the regenerated DB dump. None of it is needed.

### Non-goals

- Non-blocking schema changes (`op.md` non-goal).
- Session-scoped locking of any kind. Everything is transaction-scoped, which is also what a transaction-mode
  connection pooler in front of Postgres requires. §10.1 is the gap this leaves.

## 2. Lock targets

The lock target is a store table, named from the table id — `tbl_<uuid.hex>` for tables, `view_<uuid.hex>` for
views (`StoreBase.storage_name()`). No metadata read is needed to compute the name.

**The lock set** of a transaction is the store table of every table it touches:

- a read locks every table on each `TableVersionPath` it reads, base included — a view read reads base data;
- a data write locks its target and, when `lock_mutable_tree=True`, every transitive mutable view, because the
  write propagates there;
- a schema change locks the same set as the equivalent write, plus, for `create_view`/`drop_view`, the **base's**
  store table: adding or dropping a mutable view changes how writes to the base propagate, so it must exclude
  concurrent base writes (today that exclusion comes from X-locking the base's `tables` row);
- a table with no store table contributes nothing: pure snapshots have none, and their bases are already in the
  path.

**Order**: one global sort of the whole lock set by **table id**, and nothing else. Deadlock freedom needs only
that every transaction acquire overlapping relations in the same relative order, which any total order gives;
base-before-view was inherited from traversing the tree node by node and buys nothing once the set is acquired as
a set.

Sort by table id, not by store table name, even though the name is what goes into the statement — the two orders
differ, because of the `tbl_`/`view_` prefix, and the id is the safer key for three reasons:

- **It is intrinsic and always in hand.** The name is derived: `storage_name()` needs `is_view`, which a bare
  table id does not carry and neither does a `TableVersionHandle` (it holds only the id and effective version;
  `is_view` costs a metadata load). Inside a `TableVersionPath` it is recoverable positionally — every
  non-terminal member has a base and is therefore a view, the root is the table — but a sort key that has to be
  reconstructed differently depending on how the target was supplied is a sort key that can be reconstructed
  inconsistently.
- **It is independent of the naming convention.** A future change to `storage_name()` — a new prefix, or
  unifying `tbl_`/`view_` — would silently reorder a name-based sort. Two Pixeltable versions running
  concurrently against one database across a rolling upgrade would then disagree about the order and could
  deadlock, with nothing in either version looking wrong.
- **The mapping has to happen anyway.** `LOCK TABLE` takes names, so the lock set is a set of (id, name) pairs
  regardless; sorting those pairs by id and emitting the names costs nothing.

Using it everywhere is the whole obligation. One `LOCK TABLE a, b, c IN <mode> MODE` statement covers a whole tree
in one round trip, acquiring left to right; if a mixed set ever needs two modes, issue the statements in the same
sorted target order (in practice a lock set has a single mode, because a table tree is all one kind and reads use
`ACCESS SHARE` for both kinds). The `tables`-row locks that metadata updates take (§4.3) are keyed by table id, so
they follow the same sort literally, and they always come *after* the relation locks — no transaction ever holds a
row lock while waiting for a relation lock.

What base-first bought, and what dropping it costs: it froze a node's child set from the moment the *base* was
locked, so views could not appear under it mid-acquisition. Under a global sort the base may be locked last, so
the tree can change while the statement is still acquiring — but the validation of §2.1 step 4 runs after all the
locks are held and catches exactly that, so the cost is a marginally higher guess-mismatch rate on trees that are
being reshaped concurrently, not a correctness gap.

**No lock upgrades.** Each class takes, up front, a mode at least as strong as what its own statements will take
later (`ACCESS SHARE` for `SELECT`, `ROW EXCLUSIVE` for DML, `ACCESS EXCLUSIVE` for `ALTER`/`DROP`), so no
transaction ever escalates, and the classic upgrade deadlock cannot occur.

### 2.1 Discovering the lock set before reading metadata

The lock set for `lock_mutable_tree=True` depends on metadata — which views exist — but metadata may only be read
*after* locking. Split the set in two:

- **Static**: the target and all of its **ancestors**. A view's ancestry is immutable — `ViewMd.base_versions` is
  written once, at creation (`view.py`), and no operation re-parents a view — so this part of the set can never
  be the *wrong shape*. When the caller passes a `TableVersionPath`, which every `Table` handle carries, it comes
  for free and needs no metadata at all; when the caller passes a bare table id, it comes from the cache instead
  (see below).
- **Dynamic**: the target's transitive mutable **descendants**. Only metadata knows about those, they change as
  views are created and dropped, and only operations with `lock_mutable_tree=True` need them. This is the only
  genuinely dynamic part, and the only reason the guess-and-validate scheme exists.

**A read's lock set is entirely static.** `_acquire_path_locks(for_write=False)` does not recurse into views,
because a read touches only its own path. So reads — the latency-sensitive class, and the only one that runs at
operational-table rates — never guess, never validate a guess, and never restart. The problem below is confined
to data writes and metadata updates.

For those, the dynamic part is guessed, locked, and then validated against the metadata that the locks made
current:

1. **Guess** the descendants from the lock-set cache.
2. **Lock** the resulting set, sorted by §2's global order, in one statement per mode.
3. **Read and validate metadata** — fresh, per §1.4. Loading a table's md already discovers its direct mutable
   views: `_load_tbl_version()` queries `tables` for `view_md.base_versions[0] == <id>` on every load, regardless
   of what was locked, so a view that was missed cannot hide from the check.
4. **Validate the guess**: compare the locked set against the tree the md describes. On a mismatch, abort
   and retry — the retry's guess is correct, because the md load just refreshed the cache. On a match, proceed.

**Targets given as bare table ids.** `begin_xact()` accepts `read_tbl_ids`/`write_tbl_ids` as well as
`read_tvps`/`write_tvps`, and a bare `UUID` carries no ancestry — so for those the static part is not free
either. It is not one problem but four cases, and only two of them need anything:

- **Ids that are already a flattened path.** `TableVersionPath.tbl_ids` returns `[self, *base.tbl_ids]`, so a
  caller that passes it (`local_table.py:282`, and the from-clause contribution to
  `Query.referenced_tbl_ids()`) has already included every ancestor. Nothing to resolve.
- **Ids that come from expressions.** `referenced_tbl_ids()` also unions in `Expr.list_tbl_ids()` — tables
  referenced by a query's expressions, e.g. through a `@pxt.query` UDF. If such a table is a view, its
  ancestors are *not* in the set, and its data *is* read, so its path has to be locked. This is the case that
  needs the cache: the entry supplies the ancestor chain, and because ancestry is immutable a cached chain is
  never the wrong shape.
- **Metadata-only reads.** `get_table_by_id()`, `validate_tbls_exist()`, the post-create `get_tbl_fn()`, and
  `view.py`'s base-md read take no locks at all (§4.1), so they need no lock set.
- **`_finalize_pending_ops(write_tbl_ids=[tbl_id])`.** Needs the full path, not just the target: `LoadViewOp`
  executes a view-load plan that reads base data. Same resolution as the second case.

So the cache entry has to be self-contained for an id-only target — ancestors included, not just descendants
(§2.3) — and the warm-up transaction already reads exactly what that requires, since `view_md.base_versions`
gives the whole chain in one md load. It is also what supplies each member's store table *name*: a bare id does
not say whether it belongs to a table or a view, so the name cannot be computed from the id alone. The lock
*order* is unaffected, since §2 sorts by id.

### 2.2 Why guessing is safe

Two invariants carry this.

**Nothing happens before validation.** No data read beyond metadata, no DML, no md write, no media I/O — steps
1–4 of §3 only acquire locks, read metadata and check for pending ops. A wrong guess therefore costs exactly one
aborted attempt that did no work and left nothing behind. This is what licenses guessing at all: a guess can be
arbitrarily wrong — a stale tree shape, a stale table kind — without any consequence beyond that one attempt.

**A lock on a node freezes that node's child set.** Adding or dropping a mutable view under node N is a metadata
update *on N*: it bumps N's `view_sn` and takes `ACCESS EXCLUSIVE` on N's store table (§2). Every mode we ever
hold conflicts with `ACCESS EXCLUSIVE` — `ACCESS SHARE`, `ROW EXCLUSIVE` and `EXCLUSIVE` do, and so does
`ACCESS EXCLUSIVE` itself. So once we hold a lock on every node of the tree, the tree's shape cannot change while
we hold them, and one validation pass is enough: there is no need to re-check, and no window between validating
and using the set. Note this argument is about *holding* the locks, not about the order they were taken in, which
is why §2's global sort does not weaken it.

### 2.3 The lock-set cache

```python
# Catalog

@dataclasses.dataclass(frozen=True)
class LockSetInfo:
    is_data_versioned: bool
    # The target's ancestors. Immutable per table (§2.1), so this never goes stale in shape.
    # Redundant when the caller passes a TableVersionPath; load-bearing when it passes a bare table id.
    ancestor_store_tbls: frozenset[str]
    # The target's own store table, so an id-only target needs nothing but this entry.
    store_tbl: str
    # Store tables of the target's transitive mutable views.
    view_store_tbls: frozenset[str]

# Keyed by target table id. A stale entry is detected by the validation of §2.1 and costs one restart;
# it can never cause incorrect locking.
_lock_set_cache: dict[UUID, LockSetInfo]
```

The entry stores **membership, not order**: the lock set is the union of the three fields, sorted at lock time by
table id per §2. Each member is an (id, name) pair, since the id is the sort key and the name is what the
statement needs. That is the simplification a global sort buys — the entry no longer has to be a per-target
sequence in which the same view sits at a different position depending on whose subtree it was cached under, and
there is no per-entry ordering invariant to preserve when the cache is refreshed or partially invalidated.
Sorting a handful of strings per transaction is not worth avoiding.

The cache need not be a new structure: the entries are derivable from the `TableVersion` instances the catalog
already caches (`tv.mutable_views`, `tv.is_data_versioned`, and the path in `tv.path`), so this can be a tolerant
variant of `_get_mutable_tree()` that returns what is cached instead of asserting that the tree is present.

**A cache miss is warmed up in its own transaction.** Before the operation's transaction opens, run a read-only
transaction that loads the target's metadata — which also discovers its direct mutable views, since
`_load_tbl_version()` queries `tables` for `view_md.base_versions[0] == <id>` on every load — recursing into those
views, and populate the cache from what it read. Then proceed with the warm sequence above: lock, read md,
validate.

That leaves exactly one locking path. The warm-up holds no locks, so what it reads is precisely what a warm entry
is — a guess, validated after locking by step 4 — and nothing about the cold path needs its own default guess,
its own reasoning about under-locking, or its own wait policy: after the warm-up the table's kind is known, so
the policy is the real one from §1.1.

Where it runs: immediately before the outermost `begin_xact()`, inside `retry_loop()` if one is active. A nested
`begin_xact()` never warms up, because the outermost one already acquired the locks. If the warm-up runs into a
table with pending ops or a table that is gone, it reports that from there — the existing `PendingTableOpsError`
and `table_was_dropped` paths, just raised before any lock is held rather than after.

Cost: one extra read-only transaction on the first touch of a table in a process, and after an invalidation
(§2.4). Every subsequent operation on that table is warm.

The cache is a hint, not authoritative state, so entries **survive an aborted attempt**: the md load that
detected a mismatch read committed data, which is exactly what the retry's guess should be built from.

Guess mismatches should be counted (telemetry, plus a debug log line). A workload that reshapes a
tree between every attempt could in principle restart repeatedly; the retry loop bounds it exactly as it bounds
serialization failures, but it should surface as a counter rather than looking like a hang.

### 2.4 A cached entry that names a dropped relation

A cached sequence can name a store table that no longer exists: another process dropped that view, and this
process has not read metadata since. §5's invariant does not help here — under it the relation and the md row
disappear in the same transaction, so a cache built before the drop names a relation that is simply gone. This is
not an oversight in the guessing scheme; it is the one failure mode the scheme has to handle explicitly, because
Postgres cannot lock what is not there and the attempt kills the transaction.

Measured error shapes (§9.5) — all one error:

| Situation | Error |
| --- | --- |
| a member of a multi-relation `LOCK TABLE` does not exist | `UndefinedTable` (42P01); the message names the relation |
| the relation is dropped while we are waiting for the lock | `UndefinedTable` (42P01), identically |
| anything issued afterwards in the same transaction | `InFailedSqlTransaction` (25P02) |

So there is exactly one error to catch, the transaction is unusable once it arrives, and — because the lock
statements come first (§7.4) — nothing has happened that needs unwinding.

Handling, in the lock helper and `begin_xact()`:

1. **Convert** `UndefinedTable` raised by a lock statement into `StoreTableMissingError`, and roll back. Note this
   must be scoped to the lock statement: `UndefinedTable` from a later statement means something else entirely
   (`convert_sql_exc()` already maps that to `table_was_dropped`).
2. **Invalidate**: drop the lock-set cache entry for the target, and any entry keyed by a table appearing in the
   failed sequence. Do not repair the sequence in place — metadata is the source of truth and re-reading it is
   cheap. The relation named in the error message goes to the debug log, not into the control flow: the
   resolution is the same whichever member was missing, so nothing depends on parsing it.
3. **Re-warm** with §2.3's warm-up transaction. It reads current metadata, in which the dropped view is no longer
   among `mutable_views` — under §5's invariant the relation and the md row disappear in the same transaction, so
   there is no intermediate state in which the warm-up could still find the view — and the new sequence therefore
   cannot name it.
4. **Retry** the operation, bounded by the retry loop like any other restart.

If the missing relation is the operation's own target or one of its ancestors rather than a guessed descendant,
step 3 is also the diagnosis, and it lands on §5.3: the table is gone, so `excs.table_was_dropped`.

Contrast with the benign case: a stale entry naming a view that still *exists* but is no longer part of the tree
locks successfully, over-excludes harmlessly, and is caught by the ordinary validation of §2.1 step 4. Only a
*dropped* relation takes the path above. Both outcomes are one wasted, work-free attempt.

Two things this must not become: a silent retry loop (each recovery increments the same counter as a guess
mismatch, per §2.3), and a special case for reads. A read's lock set is entirely static (§2.1), so the only
relation it can find missing is its own target or an ancestor — i.e. §5.3, not a stale guess.

## 3. The protocol

Every transaction, in this order:

1. **Lock**: one `LOCK TABLE … IN <mode> MODE [NOWAIT]` per mode, over the lock set in its established order
   (§2, §2.3). Nothing else may precede this — see §7.4 for how that is enforced.
2. **Read/validate metadata**: unchanged logic (`_get_tbl_version()` / cache validation against `version` and
   `view_sn`). The snapshot is pinned here, after the locks, so what it reads is current as of lock acquisition.
3. **Validate the lock set** against that metadata (§2.1 step 4): the tree we locked must be the tree the md
   describes. Mismatch → abort and retry with the corrected set. Nothing below this line has run yet, which is
   what makes a wrong guess free (§2.2).
4. **Check pending ops**: unchanged. A table with pending ops is not usable; see §4.4.
5. **Do the work**, and for metadata updates take the `tables`-row X-lock before writing md (§4.3).

### 3.1 Wait policies

- **`NOWAIT`** for operational reads and writes, and — on both kinds — for a *helper* pending-op finalization
  (§4.4). Postgres raises `55P03 lock_not_available` immediately, which the lock helper converts to:

  ```python
  raise excs.ConcurrencyError(
      excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS,
      f'{tbl_name!r}: a schema change is in progress; this operation cannot run concurrently with it.\n'
      'Please retry once it completes.'
  )
  ```

  New error code `SCHEMA_CHANGE_IN_PROGRESS = 7002, 409, True`; `is_retryable` describes the condition for API
  clients, not an instruction to `retry_loop()`. This conversion must happen **at the lock call site**:
  `_is_retryable_exc()` currently treats `LockNotAvailable` as retryable with `_MAX_RETRIES = -1`, so leaving it
  to the classifier would turn fail-fast into retry-forever.

- **Blocking** for versioned reads and writes, for metadata updates, and for an *owner* pending-op finalization.
  No timeout: Postgres's lock queue is close enough to fair that a stream of readers cannot starve a schema
  change, and a waiter now wakes up with a usable snapshot, so waiting is productive rather than a prelude to a
  retry.

- If we later want "fail fast, but tolerate a brief blip", the knob is `SET LOCAL lock_timeout = '<n>ms'` instead
  of `NOWAIT`. `SET` is also snapshot-exempt, so it can precede the locks without breaking §1.4's requirement.

### 3.2 Privileges

`ACCESS SHARE` needs `SELECT` and `ROW EXCLUSIVE` needs a DML privilege. For the stronger modes the rule is
version-dependent: through PostgreSQL 16 an `UPDATE`, `DELETE` or `TRUNCATE` privilege permits *any* mode
(measured on 16.10 — a `SELECT`-only role is denied everything above `ACCESS SHARE`, a role with DML privileges
is allowed all eight modes), while PostgreSQL 17 introduced `MAINTAIN` as the privilege for them. Pixeltable
creates its own store tables and is therefore their owner, so all four modes are available either way. Worth
recording as a deployment constraint: a role that is neither the owner nor a holder of DML privileges could not
take `EXCLUSIVE` for versioned writes. Such a role could not run schema changes either, so it is not a
configuration we support today; verify the PostgreSQL 17 rule before relying on a non-owner read-write role.

## 4. What each operation class does

### 4.1 Reads

`ACCESS SHARE` on every store table in the path. That is the same lock the query's own `SELECT` would take; the
point of taking it explicitly up front is ordering: it makes the metadata read, the planning, and the execution
all happen inside a window during which no schema change can commit.

Reads take no row locks, take no `tables`-row lock, and conflict with nothing except a schema change.

Operational reads use `NOWAIT`; versioned reads block, which preserves today's property that a read eventually
succeeds. (Today a versioned read that runs into a schema change joins the roll-forward and waits that way.)

**Metadata-only reads take no lock at all.** `pxt.get_table()`, `list_tables()`, `get_dir_contents()`,
`describe()` and the post-create `get_tbl_fn()` read catalog metadata, not rows. They may therefore observe a
schema change in progress and show a schema that is about to change — acceptable, since any operation the caller
then performs on the returned handle locks properly. This keeps listing and describing a database from ever
failing or blocking. The distinction is per call site, not per parameter: `read_tbl_ids` on a *query*
transaction (tables referenced by expressions, whose data is read) locks; the catalog paths pass `md_only=True`
and skip it.

### 4.2 Data writes

- Operational: `ROW EXCLUSIVE`, `NOWAIT`. Compatible with itself, so concurrent inserts/deletes into the same
  operational table do not wait on each other at all; genuine conflicts resolve where they belong, on store row
  locks, and surface as `SerializationFailure` for the existing retry loop. No `tables`-row lock, and no md
  write: verified that `TableVersion._insert()` and `propagate_delete()` call `_write_md()` only
  `if self.is_data_versioned` (and rowids are UUID7s, so `next_row_id` is unused). `propagate_update()` and
  `recompute_columns()` still assert `is_data_versioned`; when they gain operational support the same "no md
  write" property must hold.
- Versioned: `EXCLUSIVE`, blocking. Compatible with `ACCESS SHARE`, so readers are unaffected; conflicts with
  itself, which is the write-serialization point that the linear version history needs. This *replaces* the
  `tables`-row X-lock for versioned writes: the version counter is read after the lock, from a fresh snapshot.

### 4.3 Schema changes and metadata updates

`ACCESS EXCLUSIVE` on every store table in the lock set, blocking, plus `SELECT … FROM tables … FOR UPDATE` on
the `tables` row(s) before writing md.

The row lock is no longer the exclusion mechanism — the relation lock is — and it is no longer needed as a
freshness check either, because md is read after the relation locks. It is kept for two narrower reasons: pure
snapshots have no relation of their own to lock, and creation's T1 writes the `tables` row in the same
transaction that creates the relation, so for that one statement the row is the only thing that exists to
serialize on. Keeping it unconditionally means md-modifying paths have one uniform serialization point.
`UPDATE lock_dummy` is dropped; `FOR UPDATE` alone provides the row lock and, where it still matters,
first-updater-wins detection.

This class covers everything that writes `tables`/`tableversions`/`tableschemaversions`: `add_columns`,
`add_computed_column`, `drop_column`, `rename_column`, `alter_column`, `add_btree_index`, `add_embedding_index`,
`drop_index`, `drop_embedding_index`, `revert`, `create_table`, `create_view`, `drop_table`, `drop_dir`,
`update_from_model`, `_incr_view_sn`, and every pending-op finalization transaction.

Selection is explicit — `begin_xact()`/`retry_loop()` gain `for_schema_change: bool = False` — and enforced, so
a path that forgets it fails in tests rather than silently:

- `Catalog.write_tbl_md()` asserts `ACCESS EXCLUSIVE` is held on the table's store table (or that the table has
  none).
- `_set_pending_op_status()` / `_finalize_pending_ops()` assert the same.
- `TableVersion._insert()` / `propagate_delete()` / `propagate_update()` assert the appropriate data-write mode.

### 4.4 Pending-op resolution

Each finalization transaction is a schema change by the rules above: `ACCESS EXCLUSIVE` + the `tables`-row lock.
Between those transactions, and during the non-transactional ops (`CreateStoreIdxsOp`, `CreateStoreColumnsOp`,
`DeleteTableMediaFilesOp`), no lock spans the gap — §10.1. What covers the gap is the existing rule that
**a table with pending ops is not usable**: any transaction that sees pending ops aborts and either finalizes them
or reports that a schema change is in progress.

Who finalizes:

- The **owner** (`_roll_forward()` right after its own md transaction) blocks for the lock.
- A **helper** (`begin_xact()`/`retry_loop()` after a `PendingTableOpsError`) tries `ACCESS EXCLUSIVE` with
  `NOWAIT`. Failure means the owner is alive and working, so the helper's operation fails with
  `SCHEMA_CHANGE_IN_PROGRESS` rather than blocking behind — and then performing — someone else's long-running
  schema change. Success means the owner is dead or between transactions, and the roll-forward proceeds, which is
  what keeps an abandoned schema change from wedging a table forever.

`_finalize_pending_ops()` takes `blocking: bool` to distinguish the two. Note that every pending op now runs
against an existing relation (§5), so there is no third case: the "the store table is missing, so the helper must
finalize regardless" exception that the old ordering required is gone, and with it the only situation in which a
fail-fast operation had to block instead.

## 5. Relation existence: the create/drop invariant

"The relation is the lock" needs the relation to exist. Today it sometimes doesn't: creation publishes the md row
in one transaction and creates the store table in a later one, and drop removes the store table before the md
row, so on both ends there is a committed window in which a visible `tables` row has no relation behind it. A
transaction landing in either window cannot lock, and the recovery has to reconstruct which of several states the
table is in.

**This design makes that window go away, as a prerequisite rather than an optimization.** The invariant:

> A visible `tables` row implies an existing store table, and vice versa.

It is established by moving relation-*existence* DDL into the metadata transaction on both ends. Relation-*shape*
DDL (`ALTER TABLE ADD/DROP COLUMN`, `CREATE INDEX`) stays where it is, in pending ops, and is unaffected: it does
not change whether the relation can be locked.

### 5.1 The two changes

**Create**: `CREATE TABLE`, with its system and user indexes, moves out of `CreateStoreTableOp` and into T1 —
the transaction that inserts the `tables` row. `CreateStoreTableOp` goes away.

- Postgres DDL is transactional, so a failed or retried T1 rolls the relation back together with the md. That
  replaces the compensating `CreateStoreTableOp.undo()`; the table-drop half of it folds into
  `CreateTableMdOp.undo()`, so a failure in a *later* op (a view load, say) still cleans up both.
- Everything created at T1 is empty, so all of this DDL is cheap — including user indexes. `create_table(source=…)`
  inserts in a later transaction, and a view's rows arrive via `LoadViewOp`. The "an HNSW build must not sit in
  the md transaction" objection applies to `add_embedding_index` on a *populated* table, which is a different path
  and keeps its pending op.
- The relation name is uuid-derived and T1 holds the parent `dirs` row X-lock, so nothing can race it. Plain
  `CREATE TABLE` suffices; no `IF NOT EXISTS`.

**Drop**: `DropStoreTableOp` folds into `DeleteTableMdOp`, so the relation and the md row disappear in the same
transaction. This is a small change — `DropStoreTableOp` already has `needs_tv = False` and carries `is_view`, so
it computes the relation name without a `TableVersion` — and it does not fight the
`pendingtableops.tbl_id → tables.id` foreign key, because the finalizer already deletes all pending-op rows in
the transaction that runs the final op.

`DeleteTableMediaFilesOp` still runs before that, so media files can be gone while the table is still visible.
That is today's behavior, it is orthogonal to locking, and `op.md`'s garbage-collection section is where it gets
fixed.

### 5.2 What this buys

The reason to do it as a prerequisite rather than a follow-up is that it removes the most intricate part of the
protocol rather than optimizing it:

- **The recovery path collapses to one branch** on anything this design can produce: `UndefinedTable` on your
  own target means the table is gone; raise `excs.table_was_dropped`. The reconstruction of "is this a create in
  progress, a dead creator, a drop in progress, or a corrupt catalog" survives only as a legacy recovery for
  databases written before the invariant (§5.3 step 3).
- **A read can no longer trigger someone else's table creation.** Under the old ordering, a plain `SELECT` that
  arrived mid-creation had to finalize the creator's pending ops to make progress — including when the creator
  was alive and merely slow.
- **`store_tbl.create()`'s idempotence machinery stops being needed on the create path.** Every DDL statement in
  its own transaction, `IF NOT EXISTS` everywhere, duplicate-object errors swallowed, `LOCK TABLE` +
  `UndefinedTable`-retry inside `_exec_if_not_exists()` — that exists because two processes can concurrently roll
  the same `CreateStoreTableOp` forward; the code says so ("we always need If Not Exists to avoid race conditions
  between concurrent Pixeltable processes"). The race exists *because* creation is a pending op. Under the dir
  lock in T1 exactly one process ever creates the relation. The machinery stays for the ADD COLUMN / CREATE INDEX
  paths, which remain pending ops.
- **§4.4's "when the store table does not exist, a helper must always finalize" exception disappears**, and
  with it the only case where a fail-fast operation had to block instead.

What it does *not* remove: §2.4. A cached lock set can still name a view that has since been dropped, because
under the invariant the relation and the md row vanish *together* — the guess is stale either way and
`LOCK TABLE` still fails. That is a property of guessing the lock set from a cache, not of the create/drop
windows.

### 5.3 The residual `UndefinedTable` cases

`LOCK TABLE` on a missing relation raises `UndefinedTable` and aborts the transaction, so it still has to be
handled. Checking existence first is not an option: `to_regclass()` or a `pg_tables` lookup is a query and would
pin the snapshot ahead of the locks (§1.4).

1. The lock helper converts `UndefinedTable` into `StoreTableMissingError` — the same error whether the relation
   was already gone or was dropped while we waited for it (measured, §9.5) — and rolls back.
2. `begin_xact()` invalidates the lock-set cache and re-warms in a fresh transaction (§2.4 steps 2–3). Two
   outcomes, decided by *which* relation was missing:
   - a **guessed descendant** — a view dropped since the guess was cached: the re-warm fixes the lock set and the
     operation retries. That is §2.4, and it is the common case.
   - the operation's **own target or an ancestor**: the table is gone. Raise `excs.table_was_dropped`.
3. **Legacy and abandoned states.** A database written by an earlier version can contain a table whose creation
   was interrupted: md row, pending ops, no relation. The invariant does not hold retroactively, so the
   roll-forward branch survives for exactly this case — if the re-warm finds pending ops on a target whose
   relation is missing, finalize them and retry. It is no longer reachable by any operation this design performs,
   which is the point: it becomes a recovery path rather than a routine one, and can be exercised deliberately in
   tests instead of accidentally in production.
4. Anything else — a live md row, no pending ops, no relation — is a corrupt catalog. Raise `INTERNAL_ERROR`
   rather than papering over it.

### 5.4 What the create and drop paths become

**Create.** T1: X-lock the parent `dirs` row (the name slot); for `create_view`, also take `ACCESS EXCLUSIVE` on
the **base's** store table and the base's `tables`-row X-lock, since a new mutable view changes how base writes
propagate (§2). Then `INSERT` the `tables` row and `CREATE TABLE` (+ indexes) for the new relation, and commit.
Note the ordering constraint of §1.4 still applies to the *other* tables T1 touches: the base's lock comes before
any query, whereas `CREATE TABLE` — a `CreateStmt`, which does require a snapshot — comes after T1's own metadata
reads, which is fine because T1's freshness comes from the base lock it took first. Pending ops that remain:
`CreateTableMdOp`, the undo-only record whose rollback deletes the md row and drops the relation, plus
`LoadViewOp` for a view. A plain table has nothing else — its indexes were created in T1.

**Drop.** A schema change like any other: `ACCESS EXCLUSIVE` on the store tables of the table and its mutable
tree, plus the `tables`-row X-lock; md is written with `pending_stmt=DROP_TABLE` and ops
`[DeleteTableMediaFilesOp, DeleteTableMdOp]`. The final op deletes the md row and drops the relation together.

A transaction *waiting* for a lock on a relation that is being dropped gets `UndefinedTable` rather than the lock
when the dropper commits (§9.5) and takes §5.3's path, landing on `table_was_dropped` — because by then the md
row is gone too.

### 5.5 Non-transactional store DDL

`CreateStoreColumnsOp` (add column) and `CreateStoreIdxsOp` run outside a transaction, statement by statement,
each statement taking its own `ACCESS EXCLUSIVE`. A reader cannot see a torn *statement*, but it can slip between
two statements of the same op. It is kept out by the pending-ops rule (§4.4), not by locks: the ops are still
pending at that point, so no data operation proceeds. §10.1 records what remains.

These are relation-*shape* changes, so they never make a relation unlockable — which is why they can stay
non-transactional while creation and drop cannot.

## 6. Dropping more than one table

`drop_table` on a single table is covered by §5.4. Two variants need their own note, because they widen the lock
set rather than changing the protocol:

- **A table with views** (`drop_table(force=True)`) and **`drop_dir(force=True)`** drop several tables at once.
  Each table keeps its own md row, its own pending ops, and therefore its own final op that deletes its md row and
  drops its relation together — so the invariant of §5 holds per table, and there is no state in which one of them
  is visible without its relation. The lock set is the union of all of them, sorted by §2's global order, which
  is what keeps two concurrent drops of overlapping trees from deadlocking.
- **Dropping a view** additionally takes `ACCESS EXCLUSIVE` on the **base's** store table and bumps its `view_sn`,
  for the same reason `create_view` does (§2): it changes how writes to the base propagate, so it has to exclude
  concurrent base writes.

## 7. Implementation

### 7.1 The lock helper

This lives in `catalog.py`, next to `_acquire_locks()` — not in a module of its own. With relation locks there is
no key encoding, no id allocation and no bit arithmetic left to hide behind an interface: what remains is one
enum and one statement builder, used by exactly one caller, and the mode selection that drives it (§1.1's table,
applied in `_acquire_locks()`) lives in `Catalog` anyway. A separate module would be indirection without a
boundary.

```python
# pixeltable/catalog/catalog.py

class TblLockMode(enum.Enum):
    """Postgres table lock modes used by Pixeltable, weakest first."""

    READ = 'ACCESS SHARE'            # queries
    OP_WRITE = 'ROW EXCLUSIVE'       # operational-table data writes; compatible with itself
    VERSIONED_WRITE = 'EXCLUSIVE'    # versioned data writes; serializes writers, not readers
    SCHEMA_CHANGE = 'ACCESS EXCLUSIVE'  # md updates, DDL, pending-op resolution


def _lock_tables(self, store_tbl_names: Sequence[str], mode: TblLockMode, *, nowait: bool) -> None:
    """Acquire `mode` on all named store tables in one statement, in the given order.

    Must be called before any statement that requires a snapshot: LOCK TABLE is snapshot-exempt, which is what
    lets a transaction that waits here still read current metadata afterwards.

    Raises:
        excs.ConcurrencyError(SCHEMA_CHANGE_IN_PROGRESS): nowait=True and a conflicting lock is held.
        StoreTableMissingError: a named store table does not exist (see the creation/drop protocol).
    """
```

Nothing else in the codebase issues `LOCK TABLE` except `StoreBase._exec_if_not_exists()`, which keeps its own
(it locks a single table it is about to alter, inside its own transaction).

### 7.2 Catalog

- `_lock_set_cache` (§2.3), the guess validation, and the restart on mismatch — reusing `begin_xact()`'s existing
  retry structure; plus the warm-up transaction that populates the cache on a miss, which runs before the
  outermost `begin_xact()` (§2.3).
- `_acquire_locks()` gains a first phase that computes the lock set, sorts it, and issues the lock statements.
  It has to expand the two id-only inputs (`read_tbl_ids`/`write_tbl_ids`) into paths via the cache, since a
  bare id carries no ancestry (§2.1); TVP inputs need no expansion.
  `_acquire_path_locks()`, `_acquire_write_lock()` and `_refresh_tbl_cache()` no longer lock anything themselves
  for tables that have a store table — they keep the md-cache work, and `_acquire_write_lock()` keeps the
  `tables`-row `FOR UPDATE` for the schema-change class only.
- `_locks_held: dict[str, TblLockMode]` (store table name → mode) for the assertions of §4.3 and for
  `is_locked()`; cleared in the same `finally` block that clears the write-target set.
- `_x_locked_tbl_ids` → `_write_locked_tbl_ids`: it now means "write targets locked in this transaction, in the
  mode appropriate for the class", which is all its consumers (`_check_write_locks()`,
  `_compute_column_dependents()`) need.
- `for_schema_change` on `begin_xact()`/`retry_loop()`, `md_only` on the catalog-read paths, and
  `blocking` on `_finalize_pending_ops()`.
- `StoreTableMissingError` handling in `begin_xact()`, alongside `PendingTableOpsError`.
- Drop `UPDATE lock_dummy` for tables; remove `Table.lock_dummy` (a later metadata version can drop the column —
  no migration is otherwise needed by this design).

### 7.3 Fault-injection hooks

- `CATALOG_AFTER_TBL_LOCK` — after a successful acquisition, inside the transaction with the locks held. The
  mode is available to the fault, so a test can park a specific class.
- `CATALOG_BEFORE_TBL_LOCK` — before the lock statement, for tests that need a thread parked on the way in.

### 7.4 Enforcing "locks first"

The whole design rests on no statement preceding the locks. Enforce it rather than trusting review: a
`before_cursor_execute` event on the engine counts statements per transaction (debug/test builds), and
`_lock_tables()` asserts the count is zero. Checked as part of this work: the engine has no `pool_pre_ping`, and
`Runtime.begin_xact()` issues no statements of its own, so today the invariant holds — the assertion is there to
keep it holding.

### 7.5 Sequencing

0. **Prerequisite — the relation-existence invariant (§5.1).** `CREATE TABLE` (+ indexes) moves into the md
   transaction and `CreateStoreTableOp` is removed; `DropStoreTableOp` folds into `DeleteTableMdOp`;
   `CreateStoreTableOp.undo()`'s drop folds into `CreateTableMdOp.undo()`. The one non-trivial piece is that T1
   now needs an `sa_tbl` for a table whose md it has just written, i.e. instantiating the `TableVersion` in memory
   from that md instead of reloading it after commit. This lands first and independently: it is a simplification
   of the create/drop paths on its own terms, testable with the existing suite and with §8.5's create/drop
   scenarios, and it does not depend on any locking change.
1. `TblLockMode` and `_lock_tables()` in `catalog.py` (§7.1), with the `pg_locks` assertions of §8.6 as their
   first tests — there is no key encoding left to unit-test in isolation, so the mode a class takes is verified by
   observing the lock it actually holds.
2. Lock-set computation: the static part from the `TableVersionPath` or, for id-only targets, the cache (§2.1);
   the cache, warm-up and guess validation of §2.1–2.3; `_locks_held`, accessors, renames. No behavior change.
3. Acquisition for **operational** tables only, with the `for_schema_change` annotations, the new error code, and
   `md_only`. Operational reads/writes stop taking the `tables`-row X-lock here.
4. `StoreTableMissingError` and the §5.3 recovery path — one branch for a dropped target, one for the stale guess
   of §2.4, plus the legacy pending-ops branch.
5. Enforcement assertions (§4.3, §7.4).
6. Fault hooks and the concurrency tests of §8.
7. Extend to **data-versioned** tables: `ACCESS SHARE` reads, `EXCLUSIVE` writes, `ACCESS EXCLUSIVE` schema
   changes; drop `FOR UPDATE NOWAIT` + `lock_dummy` from the versioned write path.

Step 0 is a prerequisite for steps 3–4 rather than a parallel track: without the invariant, step 4 has to
reconstruct which of four states a table with a missing relation is in, and step 3 has to let a fail-fast read
finalize someone else's creation. Steps 0–6 are shippable together and leave versioned-table behavior untouched.
Step 7 is the one that changes behavior for existing users (§11).

## 8. Testing

The tests exist to pin down §1.2 and §1.3 — **both matrices, in full, for both table kinds**. That means all
16 (in-flight, arriving) cells per kind, 32 in total, across the four operation classes: read, data write,
metadata update, and pending-op finalize. The point is not only that conflicting things exclude each other, but
that non-conflicting things *don't*: a design that made everything block would pass a conflict-only test suite
and would destroy the property §1.3 exists to protect.

All three outcomes are deterministically observable with `MultiThreadedScenario`, so the matrix is expressed as
data and driven by one parameterized test rather than as a list of hand-written pairs.

### 8.1 Three outcome helpers

The holder is always parked mid-operation by a `BlockFault` at `CATALOG_AFTER_TBL_LOCK`, i.e. inside its
transaction with its locks held. What differs is how the arriving operation is asserted.

**`assert_proceeds`** needs no new machinery: run the arriving operation to completion as an ordinary scenario
step while the holder is parked, then unblock. If it completes, it did not block.

**`assert_fails_fast`**: the arriving operation must raise `ConcurrencyError(SCHEMA_CHANGE_IN_PROGRESS)`
promptly, and must not be retried by `retry_loop()` (assert no retry sleeps occurred and that the table is
unchanged).

**`assert_waits`** needs a positive observation of waiting, or the scenario would just stall. A `pg_locks` probe
on a separate connection gives a deterministic one:

```python
def wait_until_blocked_on(store_tbl_name: str, timeout: float = 5.0) -> None:
    """Block until some backend is waiting for a lock on store_tbl_name; raise TimeoutError if none does."""
    # SELECT count(*) FROM pg_locks
    #  WHERE locktype = 'relation' AND relation = :name::regclass AND NOT granted
```

used as a scenario step between "start the arriving operation on its own thread" and "unblock the holder". A
final step joins the thread and asserts the operation succeeded *after* the wait. A false pass would require the
arriving operation to complete while the holder is parked, which is precisely what `assert_proceeds` tests and
what the regression would look like.

### 8.2 The matrix test

```python
class OpClass(enum.Enum):
    READ = 'read'
    WRITE = 'write'
    MD_UPDATE = 'md_update'
    FINALIZE = 'finalize'

class Outcome(enum.Enum):
    PROCEEDS = 'proceeds'
    WAITS = 'waits'
    FAILS_FAST = 'fails_fast'

# (in-flight, arriving) -> outcome.  Mirrors §1.2 and §1.3 cell for cell.
# The FINALIZE column is the *helper* variant (NOWAIT); the owner variant is covered separately in §8.3.
MATRIX: dict[bool, dict[tuple[OpClass, OpClass], Outcome]] = {
    # is_data_versioned=False -- §1.2
    False: {
        (READ,      READ): PROCEEDS,   (READ,      WRITE): PROCEEDS,
        (READ,      MD_UPDATE): WAITS, (READ,      FINALIZE): FAILS_FAST,
        (WRITE,     READ): PROCEEDS,   (WRITE,     WRITE): PROCEEDS,
        (WRITE,     MD_UPDATE): WAITS, (WRITE,     FINALIZE): FAILS_FAST,
        (MD_UPDATE, READ): FAILS_FAST, (MD_UPDATE, WRITE): FAILS_FAST,
        (MD_UPDATE, MD_UPDATE): WAITS, (MD_UPDATE, FINALIZE): FAILS_FAST,
        (FINALIZE,  READ): FAILS_FAST, (FINALIZE,  WRITE): FAILS_FAST,
        (FINALIZE,  MD_UPDATE): WAITS, (FINALIZE,  FINALIZE): FAILS_FAST,
    },
    # is_data_versioned=True -- §1.3; differs in exactly the five cells marked below
    True: {
        (READ,      READ): PROCEEDS,   (READ,      WRITE): PROCEEDS,
        (READ,      MD_UPDATE): WAITS, (READ,      FINALIZE): FAILS_FAST,
        (WRITE,     READ): PROCEEDS,   (WRITE,     WRITE): WAITS,        # <- differs
        (WRITE,     MD_UPDATE): WAITS, (WRITE,     FINALIZE): FAILS_FAST,
        (MD_UPDATE, READ): WAITS,      (MD_UPDATE, WRITE): WAITS,        # <- differs (x2)
        (MD_UPDATE, MD_UPDATE): WAITS, (MD_UPDATE, FINALIZE): FAILS_FAST,
        (FINALIZE,  READ): WAITS,      (FINALIZE,  WRITE): WAITS,        # <- differs (x2)
        (FINALIZE,  MD_UPDATE): WAITS, (FINALIZE,  FINALIZE): FAILS_FAST,
    },
}

@pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
@pytest.mark.parametrize('is_data_versioned', [False, True])
@pytest.mark.parametrize('in_flight,arriving', itertools.product(OpClass, OpClass))
def test_blocking_matrix(
    self, is_data_versioned: bool, in_flight: OpClass, arriving: OpClass, uses_db: None, fault_injection: None
) -> None:
    ...
```

`in_flight` and `arriving` are turned into operations by one factory per class, so each class is written once
and exercised in both roles and for both kinds:

| `OpClass` | Operation | Lock it takes (op / versioned) |
| --- | --- | --- |
| `READ` | `t.select(t.a).collect()` | `ACCESS SHARE` / `ACCESS SHARE` |
| `WRITE` | `t.insert(...)` | `ROW EXCLUSIVE` / `EXCLUSIVE` |
| `MD_UPDATE` | `t.add_computed_column(...)` | `ACCESS EXCLUSIVE` / `ACCESS EXCLUSIVE` |
| `FINALIZE` | roll-forward of a table left with pending ops | `ACCESS EXCLUSIVE` / `ACCESS EXCLUSIVE` |

`FINALIZE` as the *in-flight* holder is set up by parking a schema change *inside a finalization transaction*
rather than by parking a normal md update, so the `FINALIZE` row genuinely tests the pending-op path instead of
standing in for it. `CATALOG_AFTER_TBL_LOCK` (§7.3) fires there too, so the fault needs enough context to tell
the finalize case from an ordinary md update; the pre-existing `CATALOG_FINALIZE_PENDING_OPS_NON_XACT` point is
no help here, since it runs outside any transaction and therefore holds no locks. `FINALIZE` as the *arriving*
operation is a helper reaching a table with pending ops (§4.4), which is why its column is `FAILS_FAST` for both
kinds.

Four cells carry an extra assertion beyond the outcome, because the outcome alone would not catch the bug they
guard:

- `(MD_UPDATE, READ)` and `(MD_UPDATE, WRITE)`, versioned: after unblocking, the arriving operation must observe
  the **post-change** schema. This is the §1.4 freshness property, spelled out as its own test in §8.4.
- `(WRITE, WRITE)`, versioned: the second write must succeed **without a serialization-failure retry** — the
  wait-then-succeed behavior that replaces today's fail-then-redo.
- `(WRITE, WRITE)`, operational: both writes must land, and the table must contain both sets of rows.

### 8.3 Cases outside the matrix

- **Owner-variant finalize**: the `FINALIZE` column above is the helper (`NOWAIT`). The owner variant blocks;
  assert that a `_roll_forward()` following its own md transaction waits for a parked holder rather than failing,
  for both kinds.
- **No false sharing**: a metadata update on table A must not block a read or a write on unrelated table B, for
  both kinds. This is the test that catches an over-broad lock set.
- **View trees**: a write to a base with `lock_mutable_tree=True` must exclude a metadata update on its mutable
  view, and must *not* exclude operations on an unrelated view. Both kinds.
- **`create_view` vs base write**: a base write parked with its lock held must block `create_view` on that base;
  after unblocking, the view must contain the written rows. Both kinds.
- **Stale lock-set guess** (§2.1–2.3), the tests that keep the optimistic scheme honest. A guess can only be
  wrong in two shapes, because ancestry is immutable and a view therefore only ever joins the tree or leaves it:
  a view was **added** (the guess under-locks, and validation has to catch it) or a view was **dropped** (its
  relation is gone, and the lock statement fails). One test each:
  - *view added behind the cache*: warm one thread's cache on a base with no views, create a view from another
    thread, then run a base write on the first thread. It must lock the new view too — assert the write
    propagated to the view, and that the guess-mismatch counter incremented exactly once.
  - *view dropped behind the cache*: same setup, but drop the view instead. The write's `LOCK TABLE` hits a
    missing relation for a *derived* member and must recover as a stale guess (§5.3 step 2) — succeeding, never
    surfacing `UndefinedTable`, and never reporting `table_was_dropped` — with one recovery counted. Before the
    §2.4 handling this test fails with a raw `UndefinedTable`, which is what makes it worth having.
  - *cold cache on a versioned table*: first touch in a fresh process (`reload_catalog()`), asserting via the
    `pg_locks` probe of §8.6 that the write holds `EXCLUSIVE` — not `ROW EXCLUSIVE` — on its **first** attempt,
    with zero guess mismatches. That is what distinguishes the warm-up of §2.3 from guessing a default and
    correcting it.
  - *cold cache does not fail fast*: a metadata update parked with the lock held, and a first-touch read of a
    **versioned** table from a cache-cold thread. It must wait and then succeed, not raise
    `SCHEMA_CHANGE_IN_PROGRESS`: after the warm-up the kind is known, so the policy applied is the real one.

### 8.4 The freshness regression test

The whole reason for `LOCK TABLE` over advisory locks, and the one test that would fail on an otherwise
correct-looking implementation. Written for versioned tables because that is where the arriving write waits
rather than failing fast:

```python
@pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
def test_write_blocked_by_schema_change_sees_new_schema(self, uses_db: None, fault_injection: None) -> None:
    """A write that waited for a schema change must see the schema that landed while it waited.

    With a snapshot pinned before the wait, the insert would silently leave the new computed column NULL.
    """
    t = pxt.create_table('test', {'a': pxt.Int | None}, _is_data_versioned=True)
    fault = BlockFault()
    insert_done = Event()

    (
        MultiThreadedScenario()
        # Thread 0: add a computed column, parked inside its transaction with ACCESS EXCLUSIVE held
        .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
        .then_run_until(
            thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=t.a + 1)
        )
        # Thread 1: start an insert that must block on the lock, and confirm it is actually waiting
        .then_run(thread_id=1, name='start insert', fn=lambda: start_thread(t.insert, [{'a': 1}], insert_done))
        .then_run(thread_id=1, name='confirm blocked', fn=lambda: wait_until_blocked_on(store_name(t)))
        .then_unblock(thread_id=1, fault=fault)
        .then_run(thread_id=1, name='join insert', fn=lambda: insert_done.wait(timeout=10))
        .execute()
    )

    # the row was inserted after the schema change, so the computed column must be populated
    assert t.select(t.a, t.b).collect() == [{'a': 1, 'b': 2}]
```

### 8.5 Creation and drop (§5, §6)

Run for both kinds; the create/drop protocol does not depend on the kind, so this is parameterized the same way.

First, the invariant of §5 itself, which is what makes the rest of this section short:

- **Never md without a relation**: park a creator at each of its remaining fault points (inside T1 before commit,
  and between pending ops) and, from another thread, assert that the table is either invisible or fully lockable —
  never visible-but-unlockable. Same for a drop parked between `DeleteTableMediaFilesOp` and the final op. This is
  the test that would have failed under the old ordering, and it is what licenses removing the branches below.
- **T1 rolls back cleanly**: `ExceptionFault` inside T1 after the `CREATE TABLE`. Assert no md row *and* no
  orphaned relation — Postgres rolls the DDL back with the md, which is the property that replaces
  `CreateStoreTableOp.undo()`.
- **A later op fails**: `ExceptionFault` in `LoadViewOp` after T1 committed. The rollback path must delete the md
  row and drop the relation together (`CreateTableMdOp.undo()`), leaving neither behind.

Then the residual cases of §5.3:

- **Dropped under a live handle**: thread 0 drops the table; thread 1, holding a `Table` handle, reads → clean
  `table_was_dropped`, never `UndefinedTable`.
- **Waiter loses the relation**: thread 1 blocks on a lock; thread 0 completes a drop; thread 1 must surface
  `table_was_dropped`.
- **Legacy interrupted create**: construct the pre-invariant state directly — an md row with pending ops and no
  relation — and assert a subsequent read rolls it forward and succeeds. This state is no longer reachable through
  any code path, so the test has to build it deliberately; that is the point of keeping the branch.

Two more, specific to a drop in progress (the stale-lock-set tests proper live in §8.3):

- **Mid-drop, pending ops unresolved**: park thread 1 mid-drop, between `DeleteTableMediaFilesOp` and the final
  op, so the view's md row and relation are both still there but its pending ops are not. Thread 0's write to the
  base must take the `PendingTableOpsError` route, finish the drop, and then succeed. Under §5's invariant this
  is the *only* mid-drop state a locker can observe — the relation can no longer be gone while the md row
  remains — which is why the §2.4 recovery has no branch for it.
- **Ancestor dropped**: thread 1 drops the base of the view thread 0 holds a handle to; thread 0's read must
  surface `table_was_dropped`, not `UndefinedTable` and not a stale-guess retry loop.

### 8.6 Non-concurrent coverage

- `pg_locks` assertions: for each of the four operation classes **on each kind**, exactly the expected
  (relation, mode) set is held. This is the cheapest way to pin §1.1's mode table down as a test rather than a
  comment, and it is what catches a class that silently takes the wrong kind's mode.
- Metadata-only reads (`list_tables`, `describe`, `get_table`) hold **no** relation locks, and succeed while a
  schema change is parked with `ACCESS EXCLUSIVE` held. Both kinds.
- `make slimtest`, then in full: `tests/test_catalog.py`, `tests/test_concurrent.py`,
  `tests/test_concurrent_model.py`, `tests/test_table.py`, `tests/test_index.py`, `tests/test_migration.py`,
  and the random-ops harnesses. `test_concurrent_add_column_insert` asserts today's versioned semantics
  (a concurrent insert succeeds while `add_column` is mid-finalize) and must be rewritten in step 7 — that is
  exactly the semantics being changed, and the `(MD_UPDATE, WRITE)` versioned cell is its replacement.

## 9. Empirical verification

Run against the local Pixeltable Postgres instance while writing this document.

### 9.1 Blocking on a lock and snapshot freshness

A holder in a REPEATABLE READ transaction takes a lock, updates a row, and holds for 1.5s; a waiter (also RR)
takes a conflicting lock as its first statement, then reads the row:

| Waiter's first statement | Waited | Snapshot after grant |
| --- | --- | --- |
| `SELECT pg_advisory_xact_lock_shared(k)` | 1.19s | **stale** — read the pre-wait value |
| `LOCK TABLE … IN ACCESS SHARE MODE` | 1.20s | **fresh** |
| `SELECT 1 FROM …` first, then `LOCK TABLE …` | 0.00s | **stale** |

Row 1 is why advisory locks would still need currency validation. Row 2 is the property this design is built on.
Row 3 is why §7.4 exists — note it waited 0.00s for the lock because the preceding `SELECT` had already waited,
having pinned the snapshot first.

### 9.2 Mode conflicts

`LOCK TABLE … NOWAIT` from a second session while the first holds the given mode:

| Held | Requested | Result |
| --- | --- | --- |
| `ACCESS SHARE` | `ACCESS SHARE` | compatible |
| `ROW EXCLUSIVE` | `ROW EXCLUSIVE` | compatible |
| `ROW EXCLUSIVE` | `ACCESS SHARE` | compatible |
| `EXCLUSIVE` | `ACCESS SHARE` | compatible |
| `EXCLUSIVE` | `EXCLUSIVE` | conflict (`LockNotAvailable`) |
| `EXCLUSIVE` | `ROW EXCLUSIVE` | conflict |
| `ACCESS EXCLUSIVE` | `ACCESS SHARE` | conflict |
| `ACCESS EXCLUSIVE` | `ROW EXCLUSIVE` | conflict |
| `ACCESS EXCLUSIVE` | `EXCLUSIVE` | conflict |
| `ACCESS EXCLUSIVE` | `ACCESS EXCLUSIVE` | conflict |

That is §1.2's and §1.3's blocking matrices, with `NOWAIT` producing `psycopg.errors.LockNotAvailable`.

### 9.3 Row-lock currency check (still relevant only where no relation exists)

In an RR transaction whose snapshot predates a committed `UPDATE … SET md = …` of the row:

| Probe | Result |
| --- | --- |
| plain re-read | stale value, no error |
| `SELECT … FOR SHARE` | `SerializationFailure` |
| `SELECT … FOR KEY SHARE` | succeeds, **no error** — `md` is not a key column, so the writer's `FOR NO KEY UPDATE` does not conflict |

Recorded because it is a trap for anyone who later reaches for `FOR KEY SHARE` as a cheap check: it does not
detect anything here.

### 9.4 `LOCK TABLE` vs advisory locks: performance

Measured against the local Pixeltable Postgres over a Unix socket (`wal_level=replica`, `synchronous_commit=on`),
so the numbers isolate lock cost from network latency. An empty `BEGIN; SELECT 1; COMMIT` round trip is ~35µs on
this setup and is the baseline every row includes.

Uncontended acquisition of K objects in one statement:

| K | `LOCK TABLE` ACCESS SHARE | advisory shared | `LOCK TABLE` ACCESS EXCLUSIVE | advisory exclusive |
| --- | --- | --- | --- | --- |
| 1 | 39.8µs | 39.7µs | 65.0µs | 44.8µs |
| 8 | 43.2µs | 45.8µs | — | — |
| 16 | 44.9µs | 49.3µs | 97.0µs | 55.5µs |
| 64 | 67.8µs | 66.3µs | 158.8µs (p99 1.27ms) | 63.3µs |

For every mode except `ACCESS EXCLUSIVE` the two mechanisms are indistinguishable: ~5µs for the first object and
~0.4µs per additional one, either way. The `ACCESS EXCLUSIVE` gap is WAL, not lock-manager work:

| Statement | WAL bytes per transaction |
| --- | --- |
| `SELECT 1` | 0 |
| `LOCK … ACCESS SHARE` | 0 |
| `LOCK … EXCLUSIVE` | 0 |
| `LOCK … ACCESS EXCLUSIVE` x1 | 88 |
| `LOCK … ACCESS EXCLUSIVE` x16 | 810 |
| `pg_advisory_xact_lock` | 0 |

Postgres logs `AccessExclusiveLock` as an `XLOG_STANDBY_LOCK` record so hot standbys can take the lock on the
recovering transaction's behalf and protect their read queries from structural changes being replayed underneath
them — it is the only mode logged because a standby query only ever holds `ACCESS SHARE`, and
`ACCESS EXCLUSIVE` is the only mode that conflicts with that. The record is emitted at acquisition (48 bytes),
not at commit, so the standby holds the lock
before the structural WAL arrives. See §10.2.

Two consequences worth recording:

- **`ACCESS EXCLUSIVE` forces XID assignment**, alone among the modes (verified: `ACCESS SHARE`,
  `ROW EXCLUSIVE`, `EXCLUSIVE` and advisory locks all leave `pg_current_xact_id_if_assigned()` NULL). The WAL
  record is keyed by xid, since that is how the standby knows when to release. So a metadata update is never a
  read-only transaction from Postgres's point of view even when it only writes md, it writes a commit record
  (the remaining 40 bytes above), and it burns a transaction id. Immaterial for wraparound at our schema-change
  rate; relevant to a proxy that classifies transactions as read-only.
- With `synchronous_commit=on` this is a WAL flush per metadata-update transaction, which is the source of the
  1.27ms p99 tail at 64 relations. Every such transaction already runs `ALTER TABLE`/`CREATE INDEX` costing
  orders of magnitude more, so it does not change the class's cost profile.

Everything else measured neutral or in `LOCK TABLE`'s favor:

| Dimension | `LOCK TABLE` | advisory |
| --- | --- | --- |
| `NOWAIT` fail-fast (incl. raise + xact abort) | 50.3µs | 45.7µs |
| Wakeup latency, holder release → waiter granted (p50) | 464µs | 506µs |
| Cold backend: connect + 16 locks | 1.28ms | 1.29ms |
| Concurrent throughput, 8 threads, compatible modes | 5393 xact/s | 5396 xact/s (5334 with no locking at all) |

The last row matters for §1.2's operational write/write cell: neither mechanism is a scalability bottleneck at
this concurrency, so the throughput of concurrent operational writes is set by the store row locks, not by the
table lock.

**Shared-memory footprint favors `LOCK TABLE`,** because an explicit relation lock merges into the lock entry the
query takes anyway:

| Transaction | Lock entries held |
| --- | --- |
| query only (1 table + 2 indexes) | 4 relation |
| `LOCK TABLE` then the same query | 4 relation — the explicit lock is free |
| advisory then the same query | 4 relation + 1 advisory = 5 |

(Each count includes one entry for the `pg_locks` view the probe itself reads: the query's own footprint is the
table plus its 2 indexes. `LOCK TABLE` on its own takes 1 entry for that table — indexes are locked by the
statements that use them, not by `LOCK TABLE`.)

Advisory keys are strictly additive slots; relation locks on tables we are about to touch cost nothing. The
capacity of that shared pool is a deployment constraint either way — §10.4.

### 9.5 Locking a relation that is not there

The failure modes behind §2.4 and §5.3, measured directly:

| Probe | Result |
| --- | --- |
| `LOCK TABLE a, b IN ACCESS SHARE MODE` where `b` does not exist | `UndefinedTable`, sqlstate `42P01`, message `relation "b" does not exist` |
| waiter blocked on `LOCK TABLE c …`; holder drops `c` and commits | `UndefinedTable`, sqlstate `42P01`, same message form |
| any statement issued after either of the above, same transaction | `InFailedSqlTransaction`, sqlstate `25P02` |

Three things follow. There is a single error to catch, so no OID-vs-name variant needs separate handling. The
message identifies the missing relation, which is enough for a debug log but is not needed for control flow. And
the transaction is dead on arrival, which is why the recovery is "roll back, invalidate, re-warm, retry" rather
than anything in-transaction — and why it is safe: the lock statements run before any work (§7.4), so there is
nothing to undo.

## 10. Known gaps

### 10.1 No lock spans a whole schema change

Locks are transaction-scoped and a schema change spans several transactions plus non-transactional DDL, so there
are windows during a long `add_computed_column` or `add_embedding_index` in which no lock is held. What keeps
data operations out of those windows is the pending-ops rule (§4.4), not locking: they abort and either finalize
or report a schema change in progress. The residual is the one §4.4 describes: a helper may end up performing
another process's schema-change work when it happens to arrive in a gap and finds the lock free. Under §5's
invariant that is now the only residual — a helper is never *obliged* to take over, since there is no longer a
state in which a table is visible but unlockable.

Closing it needs a lock that outlives a transaction. Session-scoped locking is ruled out by transaction-mode
pooling and by leak risk on pooled connections; the tractable alternative is a lease row in `tables` (owner id +
heartbeat), which is a self-contained follow-up. Recommend deferring until it demonstrably bites.

### 10.2 `ACCESS EXCLUSIVE` and read replicas

Explicit `ACCESS EXCLUSIVE` is recorded in WAL as a standby lock and can cancel queries on a physical replica
(`max_standby_streaming_delay`). `ALTER TABLE` already does this, so the change is that md-only schema changes
(e.g. `rename_column`) now take it as well. Worth knowing for a PlanetScale deployment with read replicas; not a
correctness issue.

The same WAL record also gives md-only schema changes a small *local* cost they did not have before: ~88 bytes
of WAL and, under `synchronous_commit=on`, a flush at commit; and it forces XID assignment, so such a
transaction is no longer read-only from Postgres's point of view. Quantified in §9.4. Negligible next to the DDL
these transactions already run, but it is the reason `ACCESS EXCLUSIVE` is the one mode where `LOCK TABLE` costs
measurably more than an advisory lock.

### 10.3 Versioned reads can now wait

A versioned read blocks behind a schema change where today it joins the roll-forward. It still eventually
succeeds, and it now succeeds *against the post-change schema* rather than a half-applied one. If unbounded
waiting turns out to be worse than failing, `lock_timeout` (§3.1) is the knob.

### 10.4 Lock-table capacity is a shared, finite pool

Postgres sizes its lock table at `max_locks_per_transaction × (max_connections + max_prepared_transactions)`
entries — 6400 in the bundled pgserver, where `max_locks_per_transaction` is 64 and `max_connections` is 100.
The name is misleading: it is a *global* pool, not a per-transaction cap, so a single transaction may exceed 64
(verified in §9.4: 64 relations locked in one transaction succeeds). What is bounded is the total across all
backends, and exhaustion surfaces as `out of shared memory / You might need to increase
max_locks_per_transaction` — an error, not a wait.

This design increases pressure on that pool only in the `lock_mutable_tree=True` case, where one transaction
locks a base plus every transitive mutable view. Two things bound the risk:

- An explicit relation lock **merges with** the entry the transaction's own queries take on the same relation
  (§9.4), so locking a table we are about to read or write is free in slot terms. The additional consumption is
  only for relations locked but not otherwise touched — in practice, tree members a given statement skips.
- Indexes are not locked by `LOCK TABLE`, only by the statements that use them, so the lock set does not
  multiply by index count.

The exposure is therefore roughly `concurrent writers × mutable tree size`. A deployment with deep view trees
and high write concurrency should raise `max_locks_per_transaction`; it is a restart-only GUC, which is worth
knowing before it bites in production rather than after. Not a correctness issue, and no action needed for the
bundled server at current tree sizes — recorded so the failure mode is recognizable.

## 11. Behavior changes to review

The first two rows are operational tables; the middle block is **data-versioned** tables, i.e. existing users,
and lands in §7.5 step 7; the last two affect both kinds. The protocol change is not confined to the new table
kind.

| Change | Who is affected | Assessment |
| --- | --- | --- |
| Operational reads/writes fail fast during a schema change | operational tables | intended; `op.md` v0 |
| Concurrent operational writes to one table no longer serialize | operational tables | the feature |
| Versioned write exclusion moves from the `tables` row (`FOR UPDATE NOWAIT` + `lock_dummy`) to `EXCLUSIVE` on the store table | existing users | the versioned-side protocol replacement; enables every row below |
| Versioned reads block during a schema change instead of joining the roll-forward | existing users | correctness win, latency change |
| Contended versioned writes wait instead of failing and redoing | existing users | win: no thrown-away work |
| Versioned reads and writes can no longer observe a half-applied schema change | existing users | correctness win |
| Metadata-only reads never lock | everyone | `list_tables`/`describe` stop being affected by schema changes |
| `rename_column` and other md-only changes now take `ACCESS EXCLUSIVE` on the store table | replicas | see §10.2 |

## 12. Decisions

Confirmed:

1. Native `LOCK TABLE` on store tables, transaction-scoped, modes per §1; no advisory locks, no lock ids, no
   metadata migration. The deciding argument is snapshot freshness (§1.4, measured §9.1); performance is not a
   counterweight to it, measured §9.4 — the two mechanisms are indistinguishable on the read and data-write
   paths, and `LOCK TABLE` is ahead on shared-memory footprint. The one place it costs more is `ACCESS
   EXCLUSIVE` — ~20µs on one relation, ~95µs on 64, plus a WAL flush and a forced XID — paid only by metadata
   updates.
2. Reads and operational data writes fail fast (operational) or block (versioned reads); schema changes and
   versioned writes block indefinitely.
3. The `tables`-row X-lock survives as the uniform md-level serialization point for metadata updates, and only
   as `FOR UPDATE`; it is load-bearing only for pure snapshots and creation's T1 (§4.3). `lock_dummy` goes away
   for tables.
4. CockroachDB is out of scope.
5. The dynamic part of the lock set (mutable descendants) is **guessed, locked, then validated** against the
   metadata the locks made current, with a restart on mismatch (§2.1). Safe because nothing but metadata is read
   before the validation, and because a lock on a node freezes that node's child set (§2.2). A cache miss is
   filled by a **warm-up transaction** before the operation's transaction opens (§2.3), so there is one locking
   path and the wait policy is always the real one.
6. **Relation-existence DDL moves into the metadata transaction, as a prerequisite** (§5): `CREATE TABLE` into T1
   with `CreateStoreTableOp` removed, `DropStoreTableOp` folded into `DeleteTableMdOp`. This establishes "a
   visible `tables` row implies an existing store table", which collapses the missing-relation recovery to a
   single branch, stops a plain read from ever having to finalize someone else's table creation, removes §4.4's
   must-block exception, and makes `store_tbl.create()`'s duplicate-tolerance machinery unnecessary on the create
   path. It lands as step 0 of §7.5, before any locking change, since it is a create/drop simplification on its
   own terms.

Open:

7. **Metadata-only reads take no lock** (§4.1). This is a change from the earlier draft, where they did. It makes
   `list_tables`/`describe` immune to schema changes, at the cost of their being able to show a schema that is
   mid-change.
8. **Helper pending-op finalization is fail-fast**, with no exception now that §5's invariant removes the
   missing-relation case (§4.4). An abandoned schema change is still recovered by the next operation that finds
   the lock free.
