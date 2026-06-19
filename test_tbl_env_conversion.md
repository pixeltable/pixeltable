# `test_tbl` → `test_tbl_env` dual-mode conversion: results & failure diagnosis

Converting every test that uses the local-only `test_tbl` fixture to the dual-mode
`test_tbl_env` fixture (`create_test_tbl(uses_env('test_tbl'))`, parametrized `[local, proxy]`),
then running both modes and diagnosing every proxy-mode failure.

Date: 2026-06-18. Branch: `localproxy`.

## Conversion summary

| file | converted | proxy pass | proxy fail |
|------|-----------|-----------|-----------|
| tests/test_function.py | 8 | 8 | 0 |
| tests/functions/test_globals.py | 1 | 1 | 0 |
| tests/test_query.py | 7 | 0 | 7 |
| tests/test_exprs.py | 25 | 15 | 10 |
| tests/test_table.py | 14 | 6 | 8 |
| tests/test_view.py | 2 | 0 | 2 |
| **total** | **57** | **30** | **27** |

All converted tests pass in **local** mode (the conversion is behavior-preserving locally).
`test_function.py` and `test_globals.py` pass fully in both modes — their UDFs are module-/class-level
(resolvable on the daemon by import path).

### Skipped (left on `test_tbl`/`uses_db`) — local-only co-fixtures or non-fixture-shaped
- `test_exprs::test_props` — `img_tbl` (media)
- `test_table::test_update`, `test_delete` — `small_img_tbl` (media)
- `test_table::test_repr` — `local_embed` (embedding index)
- `test_index::test_update_img` — `img_tbl` (media)
- `test_snapshot::test_errors` — `local_embed`
- `test_query::test_count`, `test_count_errors` — `small_img_tbl`
- `test_sample::test_sample_create_insert_table` — builds its own tables via `create_sample_data(_local_path, …)`; not a fixture swap

## Failures grouped by root cause

Legend — **[P]** product gap (proxy should support this; fixable in the proxy layer);
**[T]** test needs adaptation for dual-mode (behavior legitimately differs by catalog).

### Cat 1 — Table referenced by **bare string name** over proxy  **[T]**
`pxt.get_table('test_tbl')` / drop-by-name resolves in the local catalog root, but a hosted table
lives at a `pxt://local:<db>/…` path. → `NotFoundError: Path 'test_tbl' does not exist` (catalog.py:1408).
- `test_query::test_head_tail` (`reload_catalog()` then `pxt.get_table('test_tbl')`)
- `test_exprs::test_in`
- `test_table::test_drop_table_force`, `test_drop_table_force_via_handle`

Fix: wrap the name in the env path-builder `p(...)`; needs the test to also take `uses_env`. Not a product gap.

### Cat 2 — `create_view` with a **proxy base** not routed  **[P]**
`pxt.create_view(name, <proxy table>, …)` reaches local `Catalog.create_view`, which does
`assert isinstance(base, TableVersionPath)` (catalog.py:1516); a proxy table's `_tbl_path` is a
`TableMdPath`, so it asserts.
- `test_view::test_add_column_to_view`, `test_recompute_column`
- `test_exprs::test_base_table_col_refs`
- `test_table::test_add_computed_column` (creates a view path)

Fix: route `create_view` to the base table's catalog (`CatalogProxy.create_view`) instead of always local.
(Bare view names also need `p(...)`, Cat 1.)

### Cat 3 — UDF/Function not resolvable on the daemon  **[P]/[T]**
Function serialized **by id** (client-local registry) rather than by import path → daemon can't find it.
→ `NotFoundError: Function with id <uuid> not found` (proxy_client.py).
- `test_query::test_limit_0`
- `test_exprs::test_comparison`, `test_apply`, `test_json_dumps`

Causes: `.apply(<callable>)` (id-registered — known limitation, **[T]**), or builtins/UDAs serialized by id
that the daemon should resolve (**[P]** — confirm per case).

### Cat 4 — Query terminal / mutation asserts `is_local` (no proxy routing)  **[P]**
- `cursor()` → `_output_row_iterator` asserts `is_local` (_query.py:829):
  `test_query::test_cursor_lifecycle`, `test_cursor_row`
- `where().update()/delete()` → `_first_tbl` asserts `is_local` (_query.py:494):
  `test_query::test_update_delete_where`, `test_mutation_op_restrictions`;
  `test_exprs::test_arithmetic_exprs`, `test_astype`
- create-from-query / query op asserts at `_query.py:651`:
  `test_table::test_create_from_query`, `test_insert_query`

Fix: give these query paths a remote branch that dispatches to the daemon (as `collect`/`head`/`tail` already do).

### Cat 5 — `ColumnRef` resolves its table via the **local** catalog → "Table was dropped"  **[P]**
Some expr construction/eval path still calls `ColumnRef.col` (`_col.get()` → local catalog) for a hosted
table id. Seen at column_ref.py:568 (`eval`) → :96 (`col`) and during `_from_dict` (server-side eval).
→ `NotFoundError: Table was dropped (no record found for <uuid>)` (catalog.py:2263).
- `test_exprs::test_compound_predicates`, `test_window_fns`
- `test_table::test_drop_column`

Fix: same class as the `ColumnRef.select` fix already landed — route remaining `ColumnRef.col`
resolutions (and the server-side `_from_dict`/`eval` path) to the column's catalog.

### Cat 6 — Metadata load mid-transaction over proxy  **[P]**
Window-function computed column triggers a metadata load inside an active transaction on the daemon.
→ `AssertionError: Loading new table metadata is not allowed in the middle of a transaction`.
- `test_table::test_computed_window_fn`

Fix: declare the table up front via `begin_xact()` (or wrap in `retry_loop`) on the relevant server path.

### Cat 7 — `proxy_protocol` serialization gap  **[P]**
A value in an `add_column` spec can't be serialized. → `RequestError: Unknown type: False`.
- `test_table::test_add_column` (one sub-case)

Fix: extend `proxy_protocol.serialize` for the offending value, or convert it client-side.

### Cat 8 — Validation-boundary error-code mismatch  **[T]**
Invalid input validated client-side over proxy raises a different code than locally.
→ `AssertionError: expected 'UNSUPPORTED_OPERATION', got 'INVALID_TYPE'`.
- `test_table::test_add_column` (a different sub-case)

Fix: guard the invalid-input sub-case local-only (`if p('') == ''`), per the established recipe.

### Cat 9 — `repr`/`describe` embeds the catalog-specific path  **[T]**
The rendered "From <table>" line shows the table's path, which differs local vs proxy.
- `test_query::test_repr`

Fix: assert on a path-agnostic shape, or guard local-only.

### Cat 10 — Test reaches into a `LocalTable` internal absent on `TableProxy`  **[T]/[P]**
`test_udas` accesses `tbl._tbl_version`; `TableProxy.__getattr__` rejects `_`-prefixed names
(table_proxy.py:128) → `AttributeError: _tbl_version`.
- `test_exprs::test_udas`

Fix: have the test use a public accessor, or add the accessor to `TableProxy`.

## Status update (Cats 1, 5, 7, 8 resolved)

- **Cat 7+8** — fixed in core (genuine public-API bug): `TableProxy.add_column` now validates the kwargs
  shape (`len(kwargs)==1`) before `normalize_schema`, so `add_column(c5=Int, stored=False)` raises
  `UNSUPPORTED_OPERATION` like local instead of `INVALID_TYPE`. `test_add_column` now dual-mode green.
- **Cat 1** — `test_head_tail`, `test_in`, `test_add_column`: wrapped bare table names in `p(...)`
  (`pxt.get_table(p('test_tbl'))`, `get_table(p(t._name()))`) — now dual-mode green. `test_drop_table_force`
  / `_via_handle` reverted to local-only (they build views over the base → Cat 2, out of scope).
- **Cat 5** — per the rule "tests must not poke at non-public API": `test_compound_predicates` no longer
  uses `SqlElementCache`/`.col`; it asserts the filtered query results instead → dual-mode green.
  `test_window_fns` (iterator views) and `test_drop_column` (cross-catalog `dummy_t`, reload+bare-get,
  `.col.qualified_name`) reverted to local-only — they depend on genuinely-unsupported-over-proxy features.

Principle applied: do **not** patch core to support a test that reaches into internals; strip the
internal-API use and assert observable behavior. Core was changed only for the real public-API bug (Cat 7+8).

## Embedding-index over proxy — plan

Goal: make `add_embedding_index` + `.similarity()` (and `embedding()`) work over the proxy so the
embedding tests (`test_compute_with_idx`, `test_index.py` embedding cases, `test_table::test_repr`, etc.)
can run dual-mode.

### Current state
- **`add_embedding_index` / `drop_embedding_index`**: already dispatched to the daemon
  (`TableProxy.add_embedding_index` → `_dispatch` → `proxy_dispatch._add_embedding_index`). Should work
  **iff** the embedding function serializes and resolves on the daemon — `clip` etc. are module-level UDFs
  (OK); a function-local test embedding (`local_embed`) won't resolve (separate function-resolution gap).
- **`:idx` result cells** (the embedding vectors, `ndarray`) already serialize (array support landed).
- **`.similarity()`** is the blocker. `SimilarityExpr.__init__` (similarity_expr.py:57-92) resolves the
  index **client-side via the local catalog**: `col_ref.tbl_version.get()` (line 58), `col_ref.col`
  (line 59), then `tv.get_idx(...)` (line 78) for name resolution + the modality/dimensionality check.
  For a hosted column the local catalog has no such table → `NotFoundError: Table was dropped`.

### Fix shape (mirror the `ColumnRef.select`/`col_md` routing already landed)
1. **Defer index resolution to the server for hosted columns.** In `SimilarityExpr.__init__`, when the
   column belongs to a hosted catalog (`Env.get().tbl_catalog_uri(col_md.qcolid.tbl_id) != ''`):
   - derive `qcol_id` and `table_version_key` from `col_ref.col_md` / `col_ref.tbl_version` **without**
     `.get()` / `.col` (no local-catalog load — the handle already carries the key), and
   - **skip** the client-side `tv.get_idx(...)` resolution + modality/dimensionality validation.
   The query then serializes (`as_dict`) and runs on the daemon, whose `_from_dict` deserialization branch
   (lines 62-75) already resolves the column + index against the **server's** catalog, and
   `_resolve_idx()` / `compute_query_embedding()` run server-side. For local, keep the current eager path.
2. **Validation moves server-side.** The modality/dimensionality error (lines 85-92) and idx-name
   ambiguity now surface at server eval time. Tests asserting those errors (`test_index` validation cases)
   may see a different path over proxy → assert on the error code, or guard local-only.
3. **`embedding()`** (`ColumnRef.embedding`, column_ref.py:449) resolves the index client-side the same
   way → apply the same deferral if `embedding()` over proxy is needed.
4. **Verify `add_embedding_index` end-to-end over proxy** with a module-level embedding (`clip`): create
   index → `.similarity(string=...)` → `order_by(sim).collect()` returns ranked rows; confirm the
   embedding function and `EmbeddingIndex` metadata round-trip in the request.

### Open questions / risks
- Deriving `table_version_key` from `col_md` for a snapshot/view column (effective_version pinning) —
  reuse exactly what `ColumnRef` does to build its `TableVersionHandle` so behavior matches.
- `SimilarityExpr._id` depends on `table_version_key`/`qcol_id`/`idx_name`; with `idx_name` left `None`
  client-side (resolved server-side), the client-computed id differs from a fully-resolved one — confirm
  that doesn't break plan-cache/equality (the `_id_attrs` `SimilarityExpr _id_attrs` follow-up is related).
- Function-local embedding UDFs remain unsupported (daemon can't import them) — orthogonal to this work.

## Recommended next steps (by leverage)
1. **Cat 4 (query routing)** — unblocks 8 tests; mirror the existing `collect`/`head`/`tail` remote dispatch for `cursor`, `where().update/delete`, and create-from-query.
2. **Cat 5 (ColumnRef.col routing)** — unblocks 3; finish what the `ColumnRef.select` fix started, incl. the server-side `_from_dict`/`eval` path.
3. **Cat 2 (create_view over a proxy base)** — unblocks 4; route to the base's catalog.
4. **Cat 1 / 8 / 9 (test adaptations)** — un-gate with `p(...)` or local-only guards; not product work.
5. **Cat 6 / 7 / 3 / 10** — smaller, per-case product fixes.
