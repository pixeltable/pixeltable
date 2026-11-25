# Critical components and invariants

This register lists the parts of Pixeltable core where a mistake threatens data, security, compatibility or the
correctness of results, and the guarantees that protect them.

This register is a permanent work in progress: add and change components and invariants as the code and our
understanding of it evolve. Keep implementation detail in the code and in design docs such as
`docs/design/catalog-locking.md`; this file says what must hold and where to look.

## How to read an entry

Each component lists the paths it owns and its invariants. Each invariant gives:

- **Why**: what fails for users if it breaks.
- **Breaks when**: likely ways to violate it, including from outside the component's own files.
- **Enforced in**: the modules and mechanisms that uphold it.

Paths are only starting points. A change outside every listed path can still affect a component through its
callers, its configuration, its dependencies or the data it writes.

---

## CATALOG: catalog, table versions and persisted metadata

**Paths:** `pixeltable/catalog/**`, `pixeltable/runtime.py` (transactions and their isolation level),
`pixeltable/metadata/**`, `tests/data/dbdumps/**`, `tool/create_test_db_dump.py`

### CATALOG-0: Every metadata format change is versioned and migrated

Databases created by any earlier release load and upgrade correctly.

- **Why:** a user upgrading Pixeltable must not lose or misread existing tables. A bad migration cannot be undone by
  reverting the code; it has already rewritten the user's metadata.
- **Breaks when:** persisted metadata (the dataclasses in `pixeltable/metadata/schema.py`, JSON keys, enum values,
  serialized expressions or functions) changes shape or meaning without a metadata version bump and a converter; a
  converter is edited after release; a class or function that persisted metadata refers to by name is renamed (for
  example a pending-op class, a UDF's module path or an expression class).
- **Enforced in:** `pixeltable/metadata/`: the metadata version and the converters that upgrade older metadata.

### CATALOG-1: Cached metadata matches committed state

When a transaction fails, every cached table version it modified is discarded. When a transaction starts, a cached
live table version is checked against the stored metadata and reloaded if another process changed it. Each table
version has one cached instance.

- **Why:** the cache serves every later operation in the process. A cached object that holds rolled-back or stale
  changes makes later reads and writes act on a schema that does not exist in the database, and the failure shows up
  far from its cause. A second instance of the same table version produces duplicate FROM entries and unintended
  joins.
- **Breaks when:** a table version is modified in place without being marked as modified; a failure path skips the
  cache purge; metadata cached outside the catalog's caches is not cleared at the end of a transaction; a cached
  live version is used without validation; a second instance of a table version is created.
- **Enforced in:** the transaction entry points and cache validation in `pixeltable/catalog/catalog.py`; the
  in-place updates in `pixeltable/catalog/table_version.py`, each of which marks the version as modified.

### CATALOG-2: Locks are taken in a fixed order before the transaction snapshot is set

- **Why:** concurrent reads, writes and schema changes must not deadlock, read stale metadata, or write under a
  missing lock. Failures are intermittent and hard to reproduce.
- **Breaks when:** metadata or store data is read before locks are acquired; locks are acquired in a different
  order; a write happens without a write lock; a transaction changes the default `REPEATABLE READ` isolation level
  without a stated reason, as the deliberate exceptions in `pixeltable/catalog/catalog.py` give. See
  `docs/design/catalog-locking.md`.
- **Enforced in:** lock acquisition in `pixeltable/catalog/locking.py`; lock-set resolution and validation, the
  metadata cache refresh, and the write-lock assertions in `pixeltable/catalog/catalog.py`.

### CATALOG-3: Each user operation is atomic

A data operation runs in one transaction. A schema operation commits its metadata in one transaction and finishes
or undoes its remaining steps through the pending-op log. Either way the operation takes effect completely or not
at all, and creates at most one new table version.

- **Why:** an operation that fails or is interrupted must not leave part of its effect behind; a half-applied
  operation produces a state the user never asked for and cannot tell from correct data.
- **Breaks when:** an operation commits in more than one transaction; a write runs outside the operation's
  transaction; a schema step runs outside both the transaction and the pending-op log; an op that runs outside a
  transaction is not idempotent or not safe to run concurrently; a table with unfinished pending ops is used.
- **Enforced in:** the read and write transaction entry points in `pixeltable/catalog/catalog.py` that every table
  operation uses; pending-op finalization in `pixeltable/catalog/catalog.py`; the pending ops in
  `pixeltable/catalog/tbl_ops.py`.

Known exceptions, where a failure can leave part of the operation applied:

- Model reconciliation (`create_all()`, `update_all()`) and the `pxt schema update` and `pxt schema prune` commands
  create, migrate or drop each table in a separate operation. On failure, the CLI commands name the tables already
  changed, and re-running reconciles the rest.
- `create_table()` with a `source`, and the `import_*()` functions built on it, create the table in one transaction
  and load the data in another.
- `revert()` deletes the reverted version's media before its transaction commits; if the commit fails, the media is
  gone.

### CATALOG-4: Snapshots and past versions are immutable

- **Why:** users rely on snapshots and versioned reads to reproduce results.
- **Breaks when:** a mutating operation skips the mutability check; revert removes a version that a snapshot still
  references.
- **Enforced in:** the mutability check that every mutating table operation runs (`pixeltable/catalog/`); revert's
  refusal to remove a version that a snapshot references (`pixeltable/catalog/table_version.py`).

### CATALOG-5: Schema reconciliation destroys only what it is authorized and scoped to destroy

- **Why:** atomicity does not stop an operation that succeeds from deleting the wrong data. A dropped column, index
  or table loses its data, and reverting the code does not bring it back. A plan applied to a schema that changed
  after it was computed can drop what someone else just added.
- **Breaks when:** a destructive change is applied without the flag, or the diff classifies a destructive change as
  additive; the schema-version check or the dropped-or-created check before applying a plan is weakened; update
  starts dropping tables; pruning reaches outside the catalog directory, includes a table the app file defines, or
  uses `force`; two paths that name different tables compare as equal, or two that name the same table do not.
- **Enforced in:** the diff and its destructive classification, the `allow_destructive` check and pruning in
  `pixeltable/catalog/model/`; the schema-version compare-and-swap when applying model updates in
  `pixeltable/catalog/catalog.py`.

Model reconciliation and pruning guarantee:

- **Explicit authorization:** a change that the diff classifies as destructive, such as dropping a column or an
  index, is applied only with `allow_destructive=True` or `--allow-destructive`. `create_all()` never changes an
  existing table, and `update_all()` never drops a table.
- **No stale plans:** a diff computed in one transaction is applied in a later one only if nothing it depends on
  moved: a table that was dropped, created or changed schema in between aborts the operation. On a hosted database,
  an update is refused while the deployed UDF sources differ from the local ones.
- **Scoped pruning:** pruning drops only tables under the given catalog directory that the app file does not
  define, and never with `force`, so a table that something outside the pruned set depends on stays.

**Higher-risk changes:** any change to `metadata/schema.py` or a converter; a new pending op; changes to lock
acquisition or transaction boundaries; changes to catalog caching or to in-place updates of table versions;
changes to destructive-change classification, plan concurrency checks or pruning scope; renaming a class that
metadata references by name.

**Lower-risk changes:** a clearer error message.

---

## INCREMENTAL: views and computed columns stay consistent with their bases

**Paths:** `pixeltable/catalog/table_version.py`, `pixeltable/catalog/view.py`, `pixeltable/plan.py`,
`pixeltable/exec/**`, `pixeltable/store.py`

### INCR-0: Every data change propagates to all dependent views and computed columns

Insert, update, batch update, recompute and delete cascade recursively to every mutable view, under one version
bump.

- **Why:** a view or computed column that silently misses a change returns wrong data with no error, and the wrong
  values persist.
- **Breaks when:** a write path bypasses propagation; the view load or update plans miss or duplicate rows; an
  update re-inserts a row before soft-deleting the old one (this breaks primary-key uniqueness); view membership is
  recomputed incorrectly.
- **Enforced in:** the insert, update, recompute and delete paths and their propagation to views in
  `pixeltable/catalog/table_version.py`; the view load and update plans in `pixeltable/plan.py`; soft deletion in
  `pixeltable/store.py`.

### INCR-1: Revert restores the previous version exactly

- **Why:** revert is the user's recovery tool. A partial revert leaves rows, indexes or views in a state that never
  existed.
- **Breaks when:** revert does not restore or remove some per-version state (a column, an index, an undo value,
  media); revert does not reach every dependent view.
- **Enforced in:** revert in `pixeltable/catalog/table_version.py`.

**Higher-risk changes:** changes to propagation, update or revert logic; a new write path; changes to view
membership or iterator views.

---

## QUERY: expression semantics and SQL translation

**Paths:** `pixeltable/exprs/**`, `pixeltable/_query.py`, `pixeltable/_query_base.py`, `pixeltable/plan.py`,
`pixeltable/query_clauses.py`, `pixeltable/exec/**`

### QUERY-0: An expression gives the same result in SQL and in Python

The planner decides where an expression runs. The user must not be able to tell.

- **Why:** a difference shows up as wrong query results or as computed columns that disagree with queries, often
  only for edge cases such as negative numbers, nulls, empty strings or time zones.
- **Breaks when:** an expression's SQL translation and its Python evaluation disagree. New or edited translations
  and built-in functions with both forms are the usual source. No general SQL-versus-Python comparison test exists,
  so a new translation without its own comparison test is unverified.
- **Enforced in:** each expression class's SQL translation and Python evaluation in `pixeltable/exprs/`.

### QUERY-1: Only fully translatable predicates are pushed to SQL

- **Why:** a partially translated filter returns wrong rows.
- **Breaks when:** only part of a predicate is pushed to SQL and the rest is dropped; an expression counts as
  translatable to SQL although its translation is incomplete.
- **Enforced in:** the analyzer in `pixeltable/plan.py`, which splits where clauses into SQL and Python parts; the
  SQL translation cache in `pixeltable/exprs/`.

### QUERY-2: A query plan preserves the meaning of the query as a whole

The plan keeps which rows the result contains and how many times, the grouping and window partitions, and the
requested order. Filters, aggregation, limit and offset run at points in the plan where they keep that meaning.

- **Why:** expressions that are each correct still return wrong results, with no error, when the plan combines them
  wrongly.
- **Breaks when:** a limit or offset runs before a filter that is evaluated in Python; a join loses or duplicates
  matches, or binds a predicate to the wrong table; aggregation state resets at a batch boundary; window partitions
  or the requested order are lost.
- **Enforced in:** plan construction in `pixeltable/plan.py`; joins, ordering, filtering (which applies limit and
  offset after the filter) and aggregation in `pixeltable/exec/`.

### QUERY-3: Values and errors stay with the rows they belong to

Batched and concurrent evaluation, including async UDF calls, keeps each value and each error with its input row,
and keeps the input order when the query needs it. An evaluation error follows the operation's `on_error` policy and
propagates to the expressions that depend on it.

- **Why:** a correct value stored in the wrong row is persistent corruption that no error reveals. A misplaced or
  dropped error makes invalid data look valid.
- **Breaks when:** a batch's outputs are matched to the wrong inputs; completion order replaces a required row
  order; an error lands in the wrong cell or does not reach its dependents.
- **Enforced in:** the expression evaluation node, its schedulers and its row buffer in
  `pixeltable/exec/expr_eval/`; per-row slots and error tracking in `pixeltable/exprs/row_builder.py`.

**Higher-risk changes:** a new or changed SQL translation; changes to the planner's split between SQL and Python;
changes to null handling in comparisons; changes to where limit, offset, filters or aggregation run; changes to
joins or window functions; changes to batching, scheduling or ordering in expression evaluation.

---

## INDEX: B-tree and embedding indexes

**Paths:** `pixeltable/index/**`, `pixeltable/exprs/comparison.py`, `pixeltable/exprs/similarity_expr.py`,
`pixeltable/utils/dbms.py`

### INDEX-0: Index values track the rows they index across every write and revert

An index value column is computed like any other column on insert, update and recompute. On a data-versioned table,
soft deletion moves a row's index value to the index's undo column, so the store index sees only live rows, and
revert moves it back.

- **Why:** a stale, missing or expired index value makes a query return wrong rows or miss rows, with no error.
- **Breaks when:** a write path changes an indexed column without recomputing its index value; soft deletion or
  revert skips the move between the value and undo columns; an index is added to a data-versioned table without an
  undo column; expired rows remain visible to the store index.
- **Enforced in:** soft deletion in `pixeltable/store.py`; index creation and revert in
  `pixeltable/catalog/table_version.py`.

### INDEX-1: A truncated B-tree value narrows a string filter but never decides it

A B-tree index value for a string column holds only a prefix of the string. A comparison may use the index value
alone only when the result cannot depend on the truncated part; otherwise it must also compare the full stored
value.

- **Why:** treating a truncated value as the complete string matches or orders rows by a shared prefix and returns
  wrong rows, with no error.
- **Breaks when:** a comparison with a value of unknown or full length uses the index value without also checking
  the stored column; a new operator, such as a prefix or pattern match, uses the index value; the truncation length
  changes while existing index values keep the old length; the store database's `C` collation changes, which the
  prefix comparisons rely on.
- **Enforced in:** B-tree use in comparison translation in `pixeltable/exprs/`; truncation in
  `pixeltable/index/btree.py`.

### INDEX-2: Similarity search keeps its metric, its query compatibility and its approximation contract

A query is embedded with the index's own embedding function for the query's type, or must be a vector of the
index's dimension. Similarity scores keep the meaning of the declared metric, and ordering by similarity puts the
most similar rows in the direction the user asked for. The vector index is approximate (HNSW): a search may miss
some true nearest neighbors, and Pixeltable does not promise exact results.

- **Why:** a query embedded differently from the indexed values, or a flipped score or order direction, ranks
  irrelevant rows as the best matches, with no error.
- **Breaks when:** a query embedding comes from a different function or has a different dimension or precision
  than the index; the similarity expression and the ORDER BY clause disagree on the metric or its sign; the index's
  operator class does not match its metric; HNSW parameters, precision or the operator class change for existing
  indexes without a rebuild; documentation or code presents approximate results as exact.
- **Enforced in:** the metric, operator class and query embedding in `pixeltable/index/embedding_index.py`; the
  similarity expression in `pixeltable/exprs/`; vector index creation in `pixeltable/utils/dbms.py`.

**Higher-risk changes:** changes to index value or undo columns, B-tree truncation or its use in comparisons,
similarity scoring or ordering, embedding resolution, or vector index parameters; a new index type.

---

## TYPES_UDF: type system and UDF contracts

**Paths:** `pixeltable/type_system.py`, `pixeltable/func/**`, `pixeltable/exprs/function_call.py`,
`pixeltable/exceptions.py`

### TYPES-0: UDF annotations determine column types

- **Why:** the type becomes part of the table schema and is persisted; a wrong type corrupts or rejects data.
- **Breaks when:** a Python type maps to a different column type than before; the supertype rules pick a different
  type; signature creation infers different parameter or return types.
- **Enforced in:** signature creation and the `@udf` decorator in `pixeltable/func/`; the Python-type mapping and
  supertype rules in `pixeltable/type_system.py`.

### TYPES-1: Null arguments to non-nullable parameters short-circuit to null

A nullable parameter passes null to the UDF; `*args` and `**kwargs` are exempt.

- **Why:** changing this changes the result of every UDF call on nullable data.
- **Enforced in:** argument binding for function calls in `pixeltable/exprs/function_call.py`.

### TYPES-2: Built-in functions never modify their arguments; user functions get a copy of constants

Built-ins receive the shared value of every argument. User functions receive a deep copy of a constant argument,
so modifying it does not change the value for later rows.

- **Why:** a function that modifies a shared value corrupts it for every later row and for other expressions that
  read it.
- **Breaks when:** a built-in modifies an argument, including a nested value inside a shallow copy; a user function
  is treated as a built-in; the defensive copy for user functions is removed or made shallow.
- **Enforced in:** argument passing in `pixeltable/exprs/function_call.py`; the defensive copy of constants in
  `pixeltable/exprs/`.

### TYPES-3: User errors use a specific `Error` subclass with the matching `ErrorCode`

- **Why:** applications and the HTTP layer map error codes to behavior and status codes.
- **Enforced in:** the `Error` constructor in `pixeltable/exceptions.py`.

**Higher-risk changes:** changes to type inference, nullability handling or argument passing; changing the return
type of an existing built-in (stored columns already use it).

---

## STORAGE: store tables and media files

**Paths:** `pixeltable/store.py`, `pixeltable/utils/object_stores.py`, `pixeltable/utils/*_store.py`,
`pixeltable/utils/filecache.py`, `pixeltable/exec/object_store_save_node.py`,
`pixeltable/exec/cell_materialization_node.py`

### STORAGE-0: Stored values keep their meaning

- **Why:** stored rows outlive the code that wrote them.
- **Breaks when:** store column naming (`col_<id>`, `col_<id>_cellmd`), row identity (`rowid`) or the
  `v_min`/`v_max` version semantics are redefined; live rows share a primary key; a stored type's encoding changes.
- **Enforced in:** store column naming in `pixeltable/catalog/column.py`; the row identity columns in
  `pixeltable/store.py`; column destinations in `pixeltable/metadata/schema.py`.

### STORAGE-1: Pixeltable deletes only media that a table owns, and only when nothing still needs it

- **Why:** deleted media cannot be recovered by a code revert. User files outside a table's prefix must never be
  deleted.
- **Breaks when:** a deletion reaches objects the table does not own, for example after object naming
  (`{tbl_hex}_{col_id}_{tbl_version}_{uuid}`) changes; revert proceeds while a snapshot references the version; a
  table with dependents is dropped without `force`.
- **Enforced in:** object naming in `pixeltable/utils/object_stores.py` and each object store's delete; media
  deletion in `pixeltable/catalog/table_version.py` and its pending op in `pixeltable/catalog/tbl_ops.py`; table
  drop in `pixeltable/catalog/catalog.py`.

### STORAGE-2: Temporary files reach their destination or are removed; the file cache is advisory

- **Why:** leaked temp files and an unbounded cache fill users' disks.
- **Enforced in:** file placement in `pixeltable/utils/object_stores.py`, which moves or deletes only temp-store
  files; the file cache in `pixeltable/utils/filecache.py`.

**Higher-risk changes:** any change to deletion, object naming, destinations or stored encodings; a new object
store backend.

**Lower-risk changes:** a retry tweak in an object store client.

---

## PUBLIC_API: SDK and CLI contracts

**Paths:** `pixeltable/__init__.py`, `pixeltable/api.py`, the public methods of the table classes in
`pixeltable/catalog/`, `pixeltable/functions/**`, `pixeltable/io/**`, `pixeltable_cli/**`, `docs/public_api.opml`,
`docs/release/docs.json`, `tool/generate_type_stubs.py`

### API-0: Public names and behavior stay compatible

- **Why:** users' code and stored computed columns call these functions. A removed name, a changed signature or a
  changed default breaks them on upgrade.
- **Breaks when:** a public name is removed or renamed; a parameter's meaning, order or default changes; observable
  behavior changes.
- **Enforced in:** module `__all__` lists; `docs/public_api.opml` decides which elements are documented (no
  automated check).

### API-1: CLI commands, flags and output formats stay compatible

- **Why:** scripts and CI pipelines parse CLI output.

### API-2: Built-in UDFs are consistent in shape; provider UDFs are unopinionated

Built-in UDFs, including provider UDFs, follow the same signature and behavior conventions as similar built-ins.
Provider UDFs expose the provider's API as fully and accurately as possible, with no default model, parameter value
or prompt of their own.

- **Why:** a released signature or default is public API. Changing a default later silently changes results for
  every caller who relied on it, and a default model goes stale when the provider retires it. Built-ins that differ
  in shape make users learn each one separately.
- **Breaks when:** a new or changed built-in departs from the shape of similar built-ins; a provider UDF sets its own
  default model, parameter value or prompt, or hides, renames or reinterprets a provider parameter.
- **Enforced in:** nothing automated; review.

**Higher-risk changes:** every public API change, including additions: a new or removed public name or parameter,
a rename, a new or changed default, a changed parameter meaning, and changed CLI output.

---

## CLOUD_API: the client side of the Pixeltable Cloud boundary

The pixeltable client talks to Pixeltable Cloud only through the protocol modules below. That boundary is the
contract between the two repositories. Behind it, each side may change freely.

**Paths (the boundary):** `pixeltable/service/management_protocol.py`, `pixeltable/service/proxy_protocol.py`,
`pixeltable/service/pxtfs_protocol.py`, `pixeltable/utils/cloud_utils.py` (request and response models)
**Paths (client side):** `pixeltable/service/**`, `pixeltable/utils/pxt_store.py`

### CLOUD-0: Released clients keep working with deployed Cloud, and the reverse

- **Why:** users run released client versions for a long time, and Cloud deploys independently. A breaking wire
  change breaks every installed client at once, and a client release cannot be recalled.
- **Breaks when:** a request or response field is removed or renamed; a field's type or meaning changes; an
  optional field becomes required; an operation's semantics change; the proxy protocol version or the metadata
  version (sent as `schema_version`) is bumped, which changes what Cloud accepts. The management protocol has no
  version number, so only additive changes are safe there; its models ignore unknown fields so that new server
  fields do not break them. A boundary change needs a matching pixeltable-cloud PR, or a statement in the PR
  description that Cloud needs none.
- **Enforced in:** the pydantic models in the boundary modules; protocol version checking in the proxy daemon and
  client in `pixeltable/service/`.

### CLOUD-1: Credentials are chosen by a fixed precedence and kept private

When an API key is configured, the client uses it and ignores any cached session or trial; the cache holds at most
one of the two per control plane. Only idempotent operations are retried. The session cache is private to the user
and written atomically.

- **Why:** a wrong precedence acts as the wrong identity or organization. A readable cache leaks sessions.
- **Enforced in:** credential selection and the retry rules in `pixeltable/service/management_client.py`; the
  session cache in `pixeltable/service/session_cache.py`.

**Higher-risk changes:** any field removal, rename or type change; a new required field; a protocol version
change; changes to credential precedence, retries or the session cache.

**Lower-risk changes:** client-side refactoring behind the boundary with unchanged messages; a new optional response
field that the client ignores when absent.

---

## SERVING: HTTP serving and the local daemon

**Paths:** `pixeltable/serving/**`, `pixeltable/utils/http_server.py`, `pixeltable_cli/server/**`; rules in
`.github/instructions/serving.instructions.md`

### SERVE-0: Error responses do not leak diagnostics

- **Why:** stack traces, secrets and private paths in responses reach untrusted callers.
- **Enforced in:** error serialization without detail in `pixeltable/exceptions.py`; HTTP error handling in
  `pixeltable/serving/`.

### SERVE-1: Local servers listen on loopback unless the user explicitly binds the daemon wider

The daemon, the media file server and served services bind to loopback, and the daemon accepts only loopback Host
headers. `pxt daemon --host` is the one explicit way to bind wider: it prints a warning and turns the Host check
off, because the daemon's API is unauthenticated. A daemon bound that way must sit behind a proxy that
authenticates its callers.

- **Why:** the media file server serves any absolute path, and the daemon acts with the user's credentials and
  serves destructive routes without authentication. Reachable beyond the machine without an authenticating proxy,
  either one exposes the user's machine and data.
- **Breaks when:** a server binds beyond loopback by default or through a setting that does not say so; the wider
  bind stops warning; the Host-header or content-type checks are relaxed on a loopback bind; the media file server
  or a service gains a non-loopback bind or caller; documentation suggests exposing the daemon without an
  authenticating proxy.
- **Enforced in:** the default bind address, the `--host` warning and the Host-header check in
  `pixeltable_cli/server/`; loopback binding of services in `pixeltable/serving/`; the loopback-only media file
  server that `pixeltable/env.py` starts (`pixeltable/utils/http_server.py`).

**Higher-risk changes:** changes to authentication, error serialization, bind addresses or request validation.

---

## CONFIG_SECRETS: configuration, credentials and telemetry

**Paths:** `pixeltable/config.py`, `pixeltable/env.py`, `pixeltable/utils/http.py`, `pixeltable/utils/sql.py`,
`pixeltable/telemetry.py`, `pixeltable/telemetry_schemas.py`, `docs/release/docs.json`

### SECRETS-0: Configuration is resolved in a fixed order and unknown keys are rejected

`pxt.init()` overrides, then environment variables, then the config file.

- **Enforced in:** value lookup and the list of known options in `pixeltable/config.py`.

### SECRETS-1: Secrets are redacted before they are logged, displayed or exported

- **Why:** logs, telemetry and CLI output end up in bug reports and third-party systems.
- **Breaks when:** a URL, connection string, header or config value is logged without redaction; a telemetry
  attribute can carry one.
- **Enforced in:** the redaction helpers in `pixeltable/utils/http.py` and `pixeltable/utils/sql.py`, and their uses
  in `pixeltable/env.py` and `pixeltable_cli/`.

### SECRETS-2: `integrations.telemetry.enabled` in `docs/release/docs.json` stays `true`

- **Enforced in:** nothing automated; AGENTS.md states the rule.

---

## RELEASE: packaging and dependencies

**Paths:** `pyproject.toml`, `uv.lock`, `packages/**`, `Makefile`

### RELEASE-0: Dependencies stay installable at their declared minimums, and the lockfile matches

- **Why:** users install Pixeltable next to other packages. A raised floor or a new required dependency can break
  their environment, and a bad release cannot be recalled.
- **Breaks when:** a required dependency is added; a version bound is raised or removed; `uv.lock` is edited by
  hand.
- **Enforced in:** `.github/workflows/pytest.yml` (`uv lock --check`); `tool/ci_tool.py` (the minimum-dependencies
  configuration installs the lowest direct versions).

### RELEASE-1: The package version comes from git tags

- **Enforced in:** `pyproject.toml` (a dynamic version from git tags) and the build hook that writes
  `pixeltable/_version.py`.

**Higher-risk changes:** a new required dependency; a raised minimum version of a core dependency; packaging
changes.

**Lower-risk changes:** a new optional extra; a lockfile refresh with no bound changes.

---

## DEV_PROCESS: checks, review policy and agent instructions

These files do not run in production, but they decide what reaches it.

**Paths:** `.github/workflows/**`, `.github/scripts/**`, `.github/risk-assessment/**`, `.github/CODEOWNERS`,
`.github/instructions/**`, `.claude/**`, `AGENTS.md`, `CLAUDE.md`, `Makefile` (check targets), `pyproject.toml`
(`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`), `tests/conftest.py`, this file

### PROCESS-0: A change that weakens a check needs scrutiny of what the check protected

- **Why:** a disabled test, a loosened lint rule or a skipped CI job lets later changes break guarantees silently.
- **Breaks when:** tests are deleted, skipped or marked as expected failures; CI triggers or matrices narrow; lint
  or type-checking rules loosen; a workflow or agent gets wider permissions.

### PROCESS-1: Review policy, agent instructions and this register change only with human judgment

- **Why:** they decide how much human review every later change gets.
- **Breaks when:** this register, the risk-assessment check, `.claude/` commands and skills, `AGENTS.md` or
  `CLAUDE.md`, CODEOWNERS, or a workflow that runs an agent or holds secrets is edited without human review.

**Higher-risk changes:** removing or skipping tests; changes to workflow triggers, permissions or secrets; changes
to this register or to review instructions.

**Lower-risk changes:** adding tests; a typo fix in instructions.

---

## Lower-risk changes

Examples of lower-risk changes outside the components above, as long as they do not reach a component through its
callers, its data or its configuration:

- Documentation, notebooks, examples and sample apps.
- Internal changes to provider integrations under `pixeltable/functions/` that leave their signatures and results
  unchanged and do not touch shared helpers or type handling.
- New tests and test utilities that add coverage.
- Dashboard presentation changes that do not touch dashboard-facing server APIs.
- Logging, progress reporting and comments.
