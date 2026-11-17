# Critical components and invariants

**Status: draft, pending review by the core team.** The component list, the invariants and the examples below were
drafted from the code and existing docs. They are not yet an agreed classification.

This register lists the parts of Pixeltable core where a mistake threatens data, security, compatibility or the
correctness of results, and the guarantees that protect them. It serves two readers:

- **Reviewers** use it to find the guarantees a change could affect, where they are enforced and which tests cover
  them.
- **The risk-assessment check** (`.github/workflows/risk-assessment.yml`, instructions in
  `.claude/commands/pxt-risk.md`) reads it on every non-draft PR to decide whether 1 or 2 humans must be involved.
  The check reads this file from the base branch, so a PR's changes to it apply only after the PR merges.

The whole team owns this file. When a change adds, moves or removes an enforcement point, update the entry in the
same PR. When the check reports a risk in "Risks the register does not cover yet", decide whether it belongs here.
Keep implementation detail in the code and in design docs such as `docs/design/catalog-locking.md`; this file says
what must hold and where to look.

## How to read an entry

Each component lists the paths it owns, the facets of the policy it affects, and its invariants. Each invariant
gives:

- **Why**: what fails for users if it breaks.
- **Breaks when**: the kinds of change that can violate it, including changes outside the component's own files.
- **Enforced in**: `path:Symbol` references to the code that upholds it.
- **Tests**: the tests that would catch a violation.

Paths are starting points, not limits. A change outside every listed path can still affect a component through its
callers, its configuration, its dependencies or the data it writes.

Facets: architecture, semantic correctness, durable state, security and isolation, resource bounds, compatibility
and recovery.

---

## CATALOG: catalog, table versions and persisted metadata

**Paths:** `pixeltable/catalog/catalog.py`, `pixeltable/catalog/locking.py`, `pixeltable/catalog/table_version.py`,
`pixeltable/catalog/table_version_handle.py`, `pixeltable/catalog/tbl_ops.py`, `pixeltable/catalog/table.py`,
`pixeltable/catalog/view.py`, `pixeltable/catalog/column.py`, `pixeltable/metadata/**`, `tests/data/dbdumps/**`,
`tool/create_test_db_dump.py`
**Facets:** durable state, compatibility and recovery, semantic correctness

### CATALOG-1: Every metadata format change is versioned and migrated

Databases created by any earlier release load and upgrade correctly.

- **Why:** a user upgrading Pixeltable must not lose or misread existing tables. A bad migration cannot be undone by
  reverting the code; it has already rewritten the user's metadata.
- **Breaks when:** a change alters the shape or meaning of persisted metadata (`pixeltable/metadata/schema.py`
  dataclasses, JSON keys, enum values, serialized expressions or functions) without bumping `VERSION`, adding a
  converter, a note and a test dump. Also when a converter is edited after release, or when a class or function
  name that persisted metadata refers to by string is renamed (for example a `TableOp` subclass, a UDF path, an
  `Expr` class name).
- **Enforced in:** `pixeltable/metadata/__init__.py:VERSION`, `pixeltable/metadata/__init__.py:register_converter`,
  `pixeltable/metadata/__init__.py:upgrade_md`, `pixeltable/metadata/notes.py:VERSION_NOTES`,
  `pixeltable/metadata/converters/convert_<N>.py` (keyed by the old version).
- **Tests:** `tests/test_migration.py:TestMigration.test_db_migration` restores every
  `tests/data/dbdumps/pixeltable-vNNN-test.dump.gz` and upgrades it, and asserts that the current `VERSION` has a
  dump and a note.
- **Note:** metadata is stored as untyped JSONB, and converters work on the stored JSON, where integer dict keys are
  strings. Pending ops are deserialized by class name (`pixeltable/catalog/tbl_ops.py`).

### CATALOG-2: Locks are taken in a fixed order before the transaction snapshot is set

- **Why:** concurrent reads, writes and schema changes must not deadlock, read stale metadata, or write under a
  missing lock. Failures are intermittent and hard to reproduce.
- **Breaks when:** a change reads metadata or store data before acquiring locks, acquires locks in a different order,
  writes without a write lock, or changes `REPEATABLE READ` assumptions. See `docs/design/catalog-locking.md`.
- **Enforced in:** `pixeltable/catalog/locking.py:XactLocks.acquire`, `pixeltable/catalog/catalog.py:Catalog._resolve_lock_set`,
  `Catalog._validate_lock_set`, `Catalog._refresh_md_cache`, `Catalog._assert_tbl_record_write_locked`,
  `Catalog.assert_rows_write_locked`.
- **Tests:** `tests/test_catalog.py` (`test_blocking_matrix`, `test_concurrent_schema_changes`,
  `test_drop_view_concurrent_insert`, `test_md_only_reads_do_not_block`), `tests/test_concurrent.py`.

### CATALOG-3: Store DDL and metadata commit together; deferred work survives interruption

- **Why:** an interrupted operation must leave a table that is either unchanged or recoverable, never half-created.
- **Breaks when:** a change moves store DDL outside the metadata transaction, adds a pending op that is not
  idempotent but runs outside a transaction, or lets a table with unfinished pending ops be used.
- **Enforced in:** `pixeltable/catalog/catalog.py:Catalog._finalize_pending_ops`, `pixeltable/catalog/tbl_ops.py`
  (`TableOp.needs_xact`; ops with `needs_xact=False` must be idempotent: `CreateStoreColumnsOp`,
  `CreateStoreIdxsOp`, `DeleteTableMediaFilesOp`).
- **Tests:** `tests/test_catalog.py` (`test_finalize_pending_ops_*`), `tests/test_fault_injection.py`.

### CATALOG-4: Snapshots and past versions are immutable

- **Why:** users rely on snapshots and versioned reads to reproduce results.
- **Breaks when:** a new mutating operation skips the mutability check, or revert removes a version that a snapshot
  still references.
- **Enforced in:** `pixeltable/catalog/table.py:Table._check_mutable`, `TableVersion.is_mutable` and the
  `assert self.is_mutable` checks in `pixeltable/catalog/table_version.py`, the snapshot reference check in
  `TableVersion._revert`.
- **Tests:** `tests/test_snapshot.py` (`test_errors`, `test_basic`), `tests/test_table.py:test_revert`.

**Examples.** 1 human: a clearer error message in `Catalog`; a new read-only metadata accessor with tests.
2 humans: any change to `metadata/schema.py` or a converter; a new pending op; changes to lock acquisition or
transaction boundaries; renaming a class that metadata references by name.

---

## INCREMENTAL: views and computed columns stay consistent with their bases

**Paths:** `pixeltable/catalog/table_version.py`, `pixeltable/catalog/view.py`, `pixeltable/plan.py`,
`pixeltable/exec/**`, `pixeltable/store.py`
**Facets:** semantic correctness, durable state

### INCR-1: Every data change propagates to all dependent views and computed columns

Insert, update, batch update, recompute and delete cascade recursively to every mutable view, under one version
bump.

- **Why:** a view or computed column that silently misses a change returns wrong data with no error, and the wrong
  values persist.
- **Breaks when:** a change adds a new write path that bypasses propagation, changes the view load or update plans,
  reorders the soft delete and re-insert of an update (this breaks primary-key uniqueness), or changes how view
  membership is recomputed.
- **Enforced in:** `pixeltable/catalog/table_version.py:TableVersion._insert`, `TableVersion.update`,
  `TableVersion.batch_update`, `TableVersion.recompute_columns`, `TableVersion._apply_update_plan`,
  `TableVersion._propagate_update`, `TableVersion.propagate_delete`; `pixeltable/plan.py:Planner.create_view_load_plan`,
  `Planner.create_update_plan`, `Planner.create_batch_update_plan`, `Planner.create_view_update_plan`;
  `pixeltable/store.py:StoreBase.soft_delete_rows`.
- **Tests:** `tests/test_view.py` (`test_computed_cols`, `test_update_base_column`,
  `test_update_changes_view_membership*`, `test_upsert_propagation`, `test_base_update_propagation`,
  `test_view_update_propagation`, `test_noop_delete_of_base`), `tests/test_component_view.py` (`test_update`,
  `test_chained_views`, `test_snapshot`).
- **Note:** operational (non-data-versioned) tables do not propagate yet; `_update_rows` and `_propagate_update`
  assert this (PXT-1101).

### INCR-2: Revert restores the previous version exactly

- **Why:** revert is the user's recovery tool. A partial revert leaves rows, indexes or views in a state that never
  existed.
- **Breaks when:** a change adds per-version state (a column, an index, an undo value, media) that revert does not
  restore or remove, or changes revert's recursion into views.
- **Enforced in:** `pixeltable/catalog/table_version.py:TableVersion._revert`.
- **Tests:** `tests/test_table.py:test_revert`, `tests/test_history.py`; revert cases in `tests/test_index.py`,
  `tests/test_alter_column.py`, `tests/test_video.py`, `tests/test_audio.py`.

**Examples.** 1 human: a new exec node used only by a new, isolated feature, with tests. 2 humans: changes to
propagation, update or revert logic; a new write path; changes to view membership or iterator views.

---

## QUERY: expression semantics and SQL translation

**Paths:** `pixeltable/exprs/**`, `pixeltable/_query.py`, `pixeltable/_query_base.py`, `pixeltable/plan.py`,
`pixeltable/query_clauses.py`, `pixeltable/exec/sql_node.py`
**Facets:** semantic correctness

### QUERY-1: An expression gives the same result in SQL and in Python

The planner decides where an expression runs. The user must not be able to tell.

- **Why:** a difference shows up as wrong query results or as computed columns that disagree with queries, often
  only for edge cases such as negative numbers, nulls, empty strings or time zones.
- **Breaks when:** a change adds or edits a `sql_expr()` translation, an `eval()` implementation or a built-in
  function that has both forms. For example, Postgres and Python disagree on the sign of `%` for negative operands
  (fixed in #1950).
- **Enforced in:** each `Expr` subclass's `sql_expr` and `eval`, for example
  `pixeltable/exprs/arithmetic_expr.py:ArithmeticExpr.sql_expr` and `ArithmeticExpr.eval`;
  `pixeltable/exprs/sql_element_cache.py:SqlElementCache`.
- **Tests:** `tests/test_exprs.py::test_int_mod_negative_operands` compares the two paths for `%` and `//`. No
  general SQL-versus-Python comparison exists, so a new translation needs its own comparison test.

### QUERY-2: Only fully translatable predicates are pushed to SQL

- **Why:** a partially translated filter returns wrong rows.
- **Breaks when:** a change alters how where clauses are split, or what `SqlElementCache` considers translatable.
- **Enforced in:** `pixeltable/plan.py:Analyzer.__init__` (`split_conjuncts`), `Analyzer._analyze_agg`.
- **Tests:** `tests/test_query.py`, `tests/test_exprs.py`.

**Examples.** 1 human: a new Python-only expression with tests. 2 humans: a new or changed `sql_expr`; changes to
the planner's split between SQL and Python; changes to null handling in comparisons.

---

## TYPES_UDF: type system and UDF contracts

**Paths:** `pixeltable/type_system.py`, `pixeltable/func/**`, `pixeltable/exprs/function_call.py`,
`pixeltable/exceptions.py`
**Facets:** semantic correctness, compatibility

### TYPES-1: UDF annotations determine column types

- **Why:** the type becomes part of the table schema and is persisted; a wrong type corrupts or rejects data.
- **Breaks when:** a change alters `ColumnType.from_python_type`, supertype rules or signature creation.
- **Enforced in:** `pixeltable/func/signature.py:Signature.create_parameters`, `pixeltable/func/udf.py:udf`,
  `pixeltable/type_system.py:ColumnType.from_python_type`, `ColumnType.is_supertype_of`, `ColumnType.supertype`.
- **Tests:** `tests/test_types.py` (`test_from_python_type`, `test_supertype`, `test_infer`), `tests/test_function.py`.

### TYPES-2: Null arguments to non-nullable parameters short-circuit to null

A nullable parameter passes null to the UDF; `*args` and `**kwargs` are exempt.

- **Why:** changing this changes the result of every UDF call on nullable data.
- **Enforced in:** `pixeltable/exprs/function_call.py:FunctionCall.make_args`.
- **Tests:** `tests/test_exprs.py::test_null_args`, `::test_varargs_none`.

### TYPES-3: Built-in functions do not modify their arguments

Built-ins receive a constant argument's shared value; user functions receive a copy.

- **Why:** a built-in that mutates its argument corrupts the constant for every later row.
- **Breaks when:** a function under `pixeltable.` mutates an argument, or `is_builtin` or the copying changes.
- **Enforced in:** `pixeltable/exprs/function_call.py:FunctionCall._arg_val`, `pixeltable/func/function.py:Function.is_builtin`.
- **Tests:** `tests/test_function.py::test_udf_modifies_args`.

### TYPES-4: User errors use a specific `Error` subclass with the matching `ErrorCode`

- **Why:** applications and the HTTP layer map error codes to behavior and status codes.
- **Enforced in:** `pixeltable/exceptions.py:Error.__init__`.
- **Tests:** `tests/test_exceptions.py`.

**Examples.** 1 human: a new provider wrapper in `pixeltable/functions/` that follows existing patterns. 2 humans:
changes to type inference, nullability handling or argument passing; changing the return type of an existing
built-in (stored columns already use it).

---

## STORAGE: store tables and media files

**Paths:** `pixeltable/store.py`, `pixeltable/utils/object_stores.py`, `pixeltable/utils/local_store.py`,
`pixeltable/utils/s3_store.py`, `pixeltable/utils/gcs_store.py`, `pixeltable/utils/azure_store.py`,
`pixeltable/utils/pxt_store.py`, `pixeltable/utils/filecache.py`, `pixeltable/exec/object_store_save_node.py`,
`pixeltable/exec/cell_materialization_node.py`
**Facets:** durable state, resource bounds

### STORAGE-1: Stored values keep their meaning

- **Why:** stored rows outlive the code that wrote them.
- **Breaks when:** a change alters store column naming (`col_<id>`, `col_<id>_cellmd`), row identity (`rowid`), the
  `v_min`/`v_max` version semantics, primary-key uniqueness on live rows, or the encoding of a stored type.
- **Enforced in:** `pixeltable/catalog/column.py:Column.store_name_from_id`,
  `pixeltable/store.py:StoreBase._create_rowid_columns`, `pixeltable/metadata/schema.py:ColumnMd.destination`.
- **Tests:** `tests/test_migration.py`, `tests/test_table.py`, `tests/test_primary_key_index.py`.

### STORAGE-2: Pixeltable deletes only media that a table owns, and only when nothing still needs it

- **Why:** deleted media cannot be recovered by a code revert. User files outside a table's prefix must never be
  deleted.
- **Breaks when:** a change alters object naming (`{tbl_hex}_{col_id}_{tbl_version}_{uuid}`), the deletion scope,
  or the guards that make deletion safe: revert refuses while a snapshot references the version, and dropping a
  table with dependents requires `force`.
- **Enforced in:** `pixeltable/utils/object_stores.py:ObjectPath.create_prefix_raw`,
  `pixeltable/catalog/table_version.py:TableVersion.delete_media`, `ObjectOps.delete` and each store's `delete`,
  `pixeltable/catalog/tbl_ops.py:DeleteTableMediaFilesOp`, `pixeltable/catalog/catalog.py:Catalog._drop_tbl`.
- **Tests:** `tests/test_destination.py` (`test_destination`, `test_dest_put_failure*`,
  `test_cell_materialization_dest`), `tests/utils.py:check_media_store_count`.

### STORAGE-3: Temporary files reach their destination or are removed; the file cache is advisory

- **Why:** leaked temp files and an unbounded cache fill users' disks.
- **Enforced in:** `ObjectOps.put_file_resolved` (only `TempStore` paths are relocated or deleted),
  `pixeltable/utils/filecache.py:FileCache`.
- **Tests:** `tests/test_file_cache.py`, `tests/test_pxt_store.py`, `tests/test_inlined_objects.py`.

**Examples.** 1 human: a retry tweak in an object store client with tests. 2 humans: any change to deletion, object
naming, destinations or stored encodings; a new object store backend.

---

## PUBLIC_API: SDK and CLI contracts

**Paths:** `pixeltable/__init__.py`, `pixeltable/api.py`, `pixeltable/catalog/table.py` (public methods),
`pixeltable/functions/__init__.py`, `pixeltable_cli/**`, `docs/public_api.opml`, `docs/release/docs.json`,
`tool/generate_type_stubs.py`
**Facets:** compatibility

### API-1: Public names and behavior stay compatible

- **Why:** users' code and stored computed columns call these functions. A removed name, a changed signature or a
  changed default breaks them on upgrade.
- **Breaks when:** a change removes or renames a public name, changes a parameter's meaning, order or default, or
  changes observable behavior. Additions are compatible.
- **Enforced in:** module `__all__` lists; `docs/public_api.opml` decides which elements are documented (no
  automated check).
- **Tests:** `tests/test_api_exports.py::test_public_exports`.

### API-2: CLI commands, flags and output formats stay compatible

- **Why:** scripts and CI pipelines parse CLI output.
- **Tests:** `tests/pixeltable_cli/` (`test_smoke.py`, `test_auth.py`, `test_service.py`).

**Examples.** 1 human: a new optional parameter; a new public function with docs and tests. 2 humans: removing or
renaming a public name; changing a default or the meaning of a parameter; changing CLI output that scripts read.

---

## CLOUD_API: the client side of the Pixeltable Cloud boundary

The pixeltable client talks to Pixeltable Cloud only through the protocol modules below. That boundary is the
contract between the two repositories. Behind it, each side may change freely.

**Paths (the boundary):** `pixeltable/service/management_protocol.py`, `pixeltable/service/proxy_protocol.py`,
`pixeltable/service/pxtfs_protocol.py`, `pixeltable/utils/cloud_utils.py` (request and response models)
**Paths (client side):** `pixeltable/service/**`, `pixeltable/utils/pxt_store.py`
**Facets:** compatibility and recovery, security and isolation

### CLOUD-1: Released clients keep working with deployed Cloud, and the reverse

- **Why:** users run released client versions for a long time, and Cloud deploys independently. A breaking wire
  change breaks every installed client at once, and a client release cannot be recalled.
- **Breaks when:** a change removes or renames a request or response field, changes a field's type or meaning,
  makes an optional field required, or changes an operation's semantics. The management protocol has no version
  constant, so only additive changes are safe there; its models use `ConfigDict(extra='ignore')` to tolerate new
  server fields. A change to `proxy_protocol.py:PROTOCOL_VERSION` or to `metadata.VERSION` (sent as
  `schema_version`) changes what Cloud accepts.
- **Enforced in:** the pydantic models in the boundary modules; `pixeltable/service/proxy_dispatch.py:handle` and
  `pixeltable/service/proxy_protocol.py:protocol_mismatch_message` (version mismatch); the matching handling in
  `pixeltable/service/proxy_client.py`.
- **Tests:** `tests/test_management_protocol.py`, `tests/test_proxy_daemon.py` (`test_protocol_mismatch_*`,
  `test_client_trusts_protocol_version_fields`).
- **Note:** a change to the boundary needs a matching pixeltable-cloud PR or a statement that Cloud needs none. Link
  it in the PR description.

### CLOUD-2: Credentials are chosen by a fixed precedence and kept private

An API key outranks a session, which outranks a trial. Only idempotent operations are retried. The session cache is
private to the user and written atomically.

- **Why:** a wrong precedence acts as the wrong identity or organization. A readable cache leaks sessions.
- **Enforced in:** `pixeltable/service/management_client.py:configured_credential`, `_api_key_source`,
  `_IDEMPOTENT_OPS`, `_RETRIED_STATUS_CODES`; `pixeltable/service/session_cache.py:_read_sessions`,
  `_write_sessions`.
- **Tests:** `tests/pixeltable_cli/test_auth.py::test_whoami_api_key_outranks_session`,
  `tests/test_session_cache.py` (`test_private_on_posix`, `test_windows_mode_semantics`).

**Examples.** 1 human: client-side refactoring behind the boundary with unchanged messages; a new optional response
field that the client ignores when absent. 2 humans: any field removal, rename or type change; a new required
field; a `PROTOCOL_VERSION` change; changes to credential precedence, retries or the session cache.

---

## SERVING: HTTP serving and the local daemon

**Paths:** `pixeltable/serving/**`, `pixeltable/utils/http_server.py`, `pixeltable_cli/server/**`; rules in
`.github/instructions/serving.instructions.md`
**Facets:** security and isolation, resource bounds

### SERVE-1: Error responses do not leak diagnostics

- **Why:** stack traces, secrets and private paths in responses reach untrusted callers.
- **Enforced in:** `pixeltable/exceptions.py:Error.to_dict` (`with_detail=False`),
  `pixeltable/serving/_fastapi.py:_http_exception`, `_run_endpoint_op`.
- **Tests:** `tests/serving/test_fastapi.py`, `tests/serving/test_cloud_errors.py`.

### SERVE-2: Local servers answer only loopback requests

- **Why:** the media file server serves any absolute path, and the daemon acts with the user's credentials. Exposing
  either beyond loopback exposes the user's machine.
- **Breaks when:** a bind address changes, Host-header or content-type checks are relaxed, or the file server gains
  a non-loopback caller.
- **Enforced in:** `pixeltable_cli/server/daemon.py:_LOOPBACK`, `pixeltable_cli/server/http_server.py:bind`,
  `_DaemonHandler._refusal`; `pixeltable/env.py` (`make_server('127.0.0.1', 0)`) for
  `pixeltable/utils/http_server.py:AbsolutePathHandler`.
- **Tests:** `tests/pixeltable_cli/test_daemon.py` (`test_host_header*`, `test_bind_loopback_hostname_spellings`),
  `tests/test_http_server.py`.

**Examples.** 1 human: a new endpoint option that reuses the existing error and auth handling. 2 humans: changes
to authentication, error serialization, bind addresses or request validation.

---

## CONFIG_SECRETS: configuration, credentials and telemetry

**Paths:** `pixeltable/config.py`, `pixeltable/env.py`, `pixeltable/utils/http.py`, `pixeltable/utils/sql.py`,
`pixeltable/telemetry.py`, `pixeltable/telemetry_schemas.py`, `docs/release/docs.json`
**Facets:** security and isolation

### SECRETS-1: Configuration is resolved in a fixed order and unknown keys are rejected

`pxt.init()` overrides, then environment variables, then the config file.

- **Enforced in:** `pixeltable/config.py:Config.get_value`, `Config.lookup_env`, `KNOWN_CONFIG_OPTIONS`.
- **Tests:** `tests/test_config.py`, `tests/test_env.py`.

### SECRETS-2: Secrets are redacted before they are logged, displayed or exported

- **Why:** logs, telemetry and CLI output end up in bug reports and third-party systems.
- **Breaks when:** a change logs a URL, connection string, header or config value without redaction, or adds a
  telemetry attribute that can carry one.
- **Enforced in:** `pixeltable/utils/http.py:redact_url`, `pixeltable/utils/sql.py:redact_db_url`,
  `pixeltable/env.py:Env.redacted_db_url`, `pixeltable_cli/utils.py:_redact_env_value`.
- **Tests:** `tests/test_http_utils.py`, `tests/pixeltable_cli/test_config.py`.

### SECRETS-3: `integrations.telemetry.enabled` in `docs/release/docs.json` stays `true`

- **Enforced in:** nothing automated; AGENTS.md states the rule.

---

## RELEASE: packaging and dependencies

**Paths:** `pyproject.toml`, `uv.lock`, `packages/**`, `Makefile`
**Facets:** compatibility and recovery

### RELEASE-1: Dependencies stay installable at their declared minimums, and the lockfile matches

- **Why:** users install Pixeltable next to other packages. A raised floor or a new required dependency can break
  their environment, and a bad release cannot be recalled.
- **Breaks when:** a change adds a required dependency, raises or removes a version bound, or edits `uv.lock` by
  hand.
- **Enforced in:** `.github/workflows/pytest.yml` (`uv lock --check`), `tool/ci_tool.py` (the `minimum-deps`
  configuration installs with `--resolution lowest-direct`).

### RELEASE-2: The package version comes from git tags

- **Enforced in:** `pyproject.toml` (`dynamic = ["version"]`, `uv-dynamic-versioning`), the hatch version hook that
  writes `pixeltable/_version.py`.

**Examples.** 1 human: a new optional extra; a lockfile refresh with no bound changes. 2 humans: a new required
dependency; a raised minimum version of a core dependency; packaging changes.

---

## CONTROL: checks, policy and agent instructions

These files do not run in production, but they decide what reaches it.

**Paths:** `.github/workflows/**`, `.github/scripts/**`, `.github/risk-assessment/**`, `.github/CODEOWNERS`,
`.github/instructions/**`, `.claude/**`, `AGENTS.md`, `CLAUDE.md`, `Makefile` (check targets), `pyproject.toml`
(`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`), `tests/conftest.py`, this file
**Facets:** architecture, compatibility and recovery

### CONTROL-1: A change that weakens a check needs scrutiny of what the check protected

- **Why:** a disabled test, a loosened lint rule or a skipped CI job lets later changes break guarantees silently.
- **Breaks when:** a change deletes, skips or marks tests as expected failures, narrows CI triggers or matrices,
  loosens lint or type-checking rules, or widens the permissions of a workflow or agent.

### CONTROL-2: Review policy, agent instructions and this register change only with human judgment

- **Why:** they decide how much human review every later change gets.
- **Breaks when:** a change edits this register, the risk-assessment check, `.claude/` commands and skills,
  `AGENTS.md` or `CLAUDE.md`, CODEOWNERS, or any workflow that runs an agent or holds secrets.

**Examples.** 1 human: adding tests; a new CI job that only adds coverage; a typo fix in instructions. 2 humans:
removing or skipping tests; changes to workflow triggers, permissions or secrets; changes to this register or to
review instructions.

---

## Routine by default

Unless a change also touches a component above, through its callers, its data or its configuration, these are
routine (1 human):

- Documentation, notebooks, examples and sample apps.
- Provider integrations under `pixeltable/functions/` that follow the existing patterns, without changing shared
  helpers, type handling or built-in semantics.
- New tests and test utilities that add coverage.
- Dashboard presentation changes that do not touch dashboard-facing server APIs.
- Logging, progress reporting and comments.
