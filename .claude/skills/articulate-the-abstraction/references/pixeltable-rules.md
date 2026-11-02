# Pixeltable rules

Conventions Marcel enforces in the Pixeltable code, grouped by area, with the PR where he stated each one.
They reflect the code as of 2026-10. Before you cite a helper, file or test directory in a review, confirm
it still exists; when the code has moved on, apply the principle behind the rule instead.

## Catalog and transactions

- The user-facing Table classes (Table, LocalTable) validate requests. TableVersion executes and only
  asserts. Catalog owns metadata retrieval, caching and locking. Nothing outside the catalog module relies
  on TableVersion internals. (#640, #1450, #1090, #1324)
- Keep Catalog from growing: logic that does not need Catalog belongs in LocalTable. (#1442)
- Use `begin_xact()` when every metadata load happens at the start of an atomic operation (insert, update,
  delete). Wrap metadata access outside that in `retry_loop()`, in its decorator form for uniformity. A
  lookup helper does not start its own transaction. (#1182, #1434)
- Everything that writes table metadata clears the cached metadata, on every exit path. (#1138)
- `_tbls` and `_tbl_versions` are caches: look up by id, and follow each `get()` with `move_to_end()` for
  the LRU order. (#564, #1481)
- Locks are acquired base table first; the locking protocol is meant to be deadlock-free. A lock function's
  contract guarantees the lock. (#917, #898, #1247)
- A table is created in a single transaction with a single version. (#1364)
- Every mutating Table method calls `_check_mutable()` before anything else, with a test in
  `tests/test_snapshot.py`. (#1472)
- Persisted metadata refers to a column by its qualified id, ordered (tbl_id, col_id). (#193, #395)
- Ids never derive from names. (#1472)
- Replicas exist only for specific table versions and do not support `revert()`. Code that hardcodes
  `anchor_tbl_id=None` asserts `not is_replica`. (#841, #917, #1233)
- A property never raises, runs a transaction or touches concurrency control. (#569, #581)
- Return a copy of metadata, never the cached structure. (#1102)

## Metadata and migrations

- `pixeltable/metadata/schema.py` contains only stored metadata. (#564)
- A stored-format change gets a migration: a converter plus a `VERSION_NOTES` entry that mentions the
  exact field. Compatibility code in `from_dict()` does not replace a migration: "Migration, please.
  Otherwise we'll have to deal with this forever." (#1153, #395)
- A new field with a default that matches current behavior can avoid a migration or a backfill. (#1065,
  #1233, #1472)
- A converter updates every affected structure, index columns included. (#250)
- A fact about the stored table goes in ColumnMd; a fact about one schema version goes in
  SchemaVersionMd. (#1090)
- Boolean metadata follows the `has_<x>` pattern, and the `create_table()` parameter, the TableMd field and
  the `get_metadata()` key share that one name. (#1472)
- `version` changes only with data, `schema_version` only with the schema. (#193)
- The test DB dump covers every expression type and type variant. (#195, #1128)

## Table models (pixeltable/catalog/model)

- A model refers only to its own columns. (#1600)
- The base `select()` list accepts only column references; computed columns go in the class body. (#1479)
- Index names are optional. Tests cover unnamed indices and a named index next to an unnamed one.
  (#1472, #1518)
- Terminology: "user columns"; diff severities are additive, destructive and unsupported, as in other
  migration frameworks. (#1479)
- Diff results are TypedDicts, and every change record fills in `on`. (#1479, #1518)

## Type system and expressions

- Use the type predicates (`is_array_type()` and its siblings), never comparisons against `_type`. (#1096)
- Type-specific conversion lives in ColumnType (`create_literal()`, `to_sa_type()`). (#87, #1065, #1324)
- `from_python_type()` never returns a shared ColumnType instance. (#123)
- An array dtype guarantees the physical format, not only that a conversion is possible. (#940)
- An Expr keeps its children only in `components`. (#280)
- Only things with an `eval()` are Exprs. Table operators such as sampling and joins are not. Placeholders
  use `Variable`. (#591, #1364)
- DataRow slots contain only in-memory values, and arrays only as `np.ndarray`. (#1007, #1065)
- `__repr__` produces valid syntax. (#1096)
- Every numpy or matrix operation gets a comment on its output shape. (#1154)

## Planner and executor

- Syntactic elements belong in `plan.py`. The SQL for an operator belongs in its ExecNode. Per-execution
  state (seeds, clients, thread pools) belongs in ExecContext, so plans can be cached. (#591, #766, #828)
- The planner's term is "output columns". (#1258)
- A write-side node does not belong in a read plan. (#766)

## UDFs and provider integrations

- Every UDF has a docstring in the style of the other providers, and a test. (#632, #1019)
- UDFs are async and use the provider's async client. No blocking IO inside an async UDF. (#632, #962,
  #1206)
- Embedding UDFs use `conditional_return_type`. A multimodal embedding is one overload per modality, and
  its test uses it inside an embedding index. (#962)
- Images in UDF signatures are `PIL.Image.Image`. (#1019)
- A UDF over a plural endpoint returns the full list. (#632, #870)
- JSON output gets a TypedDict, with field docs inline. (#1385, #1336)
- An order-dependent aggregate requires ordering, as `make_video()` does. (#1336)
- `require_package()` goes in `init()`, and a new provider package goes in `pyproject.toml`. (#962, #1336)
- A model cache needs a lock. (#1385)
- Expose the wrapped library's options instead of fixing them. (#331)
- A helper useful to several providers goes in `pixeltable/utils/` (HTTP helpers in
  `pixeltable/utils/http.py`), not in one provider module. (#963, #1023)

## Env, Runtime, CLI and services

- Env owns global state and clients, one client per service. Runtime owns per-thread state. Runtime teardown
  closes every client. (#766, #1427, #1462)
- The CLI stays agent-friendly: an epilogue and `--json` on every command, one file per command, exact
  command lines in hints, and no information printed a second time when `--json` already contains it.
  (#1319, #1440, #1600, #1745)
- CLI commands behave the same for local and hosted databases. Services are addressed by path
  (`target: PxtPath`), because names are not unique. (#1593)
- `Error.detail` is not for clients. (#1533)
- Retired areas: `pixeltable/share` (data sharing was removed) and `tests/cloud`. (#1440, #1559)

## Tests

- `pxt_raises(code, match=...)` for Pixeltable errors; `pytest.raises(..., match=...)` for other
  exceptions.
- Use ReloadTester when queries run through it, and give those queries an `order_by()`. Without queries,
  call `reload_catalog()` directly. Parametrize over reload where both matter. (#395, #406, #1119, #1363)
- CLI features get tests under `tests/pixeltable_cli/`, with test apps in `tests/pixeltable_cli/apps`.
  (#1564, #1580, #1593)
- Serving features that return media check for fetchable URLs inside the existing
  `tests/serving/test_fastapi.py` cases. (#1604)
- Snapshot behavior is tested in `tests/test_snapshot.py`. (#1472)
- Reuse the shared fixtures and helpers in `tests/utils.py` (`local_embedding`, the standard test tables,
  `get_image_files()`), and register new test media there. (#625, #639, #1434)
- Check the status returned by `insert()` and `add_column()`. (#123)
- Split a test file that has become hard to navigate. (#979)

## Docs

- Register new public symbols in `docs/public_api.opml`. (#1417)
- Examples go in the Examples section: a description line, then `>>>` with `...` continuation lines.
  (#426, #991)
- No hand-written defaults. (#422, #653)
- Link related functions, and refer to functions with `()`. (#244, #1336)
- Re-run notebooks to clear stale output; give `select()` expressions column names. (#351, #1021)
- Postgres documentation and Python stdlib docstrings are the models for structure and tone. (#233)

## Code conventions

- Explicit comparisons: `is not None`, `len(x) > 0`, `== ''`. (#1090, #1301, #1627)
- A class-level declaration for every instance member. (#496, #1324, #1627)
- Module-level imports; import modules rather than names to avoid circular imports, and no `import
  pixeltable as pxt` inside the library. Function-local imports only for optional packages. (#88, #1044)
- Protected `_name` rather than private `__name`. Classmethods or instance methods, not `@staticmethod`.
  (#32, #422)
- Refer to functions with `()` in comments and messages. (#1203, #1545)
- No backticks in code comments unless the file already uses them. (#193, #237, #1515)
- Established vocabulary: `xact`, `v_min`/`v_max`, `*_exprs` for expression lists, `ts` for timestamp,
  `min_`/`max_` prefixes, "visible" (not "live"), `if_exists='replace'` (not "overwrite"), `env` (not
  "environment"), "image build" (not "runtime update"), plain "org" and "db" for user-supplied names, with
  no coined term. (#710, #912, #1090, #1301, #1319, #1440, #1609)
