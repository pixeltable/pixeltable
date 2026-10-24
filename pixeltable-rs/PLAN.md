# pixeltable-rs: plan

A functional clone of Pixeltable in which only the public API is Python and everything behind it
(catalog, storage, versioning, expression typing and evaluation, query planning and execution) is Rust,
built as a PyO3 extension with maturin. Behavior must match Pixeltable as its own test suite observes it;
the internals may be completely different (no Postgres, no SQLAlchemy).

Success: an unmodified subset of Pixeltable's tests (`../tests/test_*.py`, copied byte for byte) passes
against this package, run through a replacement `conftest.py`/`utils.py` that use only the public API plus
two test hooks.

## Layout

```
pixeltable-rs/
  PLAN.md, README.md
  Cargo.toml              Rust crate `pxtcore`, cdylib, built into `pixeltable._core`
  pyproject.toml          maturin backend; python-source = "python"; module-name = "pixeltable._core"
  src/                    Rust: the backend
    lib.rs                #[pymodule] _core: bindings only, no logic
    error.rs              PxtError {code, message} -> raised as the matching pixeltable.exceptions class
    value.rs              Value enum + Python <-> Value conversion
    types.rs              column types: nullability, supertype, matching, validation/coercion of values
    expr/                 expression IR, construction-time typing, (de)serialization, evaluation
    functions/            builtin functions and aggregates (string, math, timestamp, json, array, ...)
    catalog/              paths, directories, tables, views, snapshots, columns, schema + data versions
    store/                row storage with version ranges, persistence to disk, reload
    exec/                 query planning/execution: scan, filter, join, group/aggregate, window, order, limit, sample
  python/pixeltable/      Python: the public API surface only
  compat/                 the test harness (see Validation)
```

## The Python/Rust boundary (rule for every contributor)

Python may contain only:
- the public classes and functions users call (`pxt.create_table`, `Table`, `Query`, `ResultSet`, `UpdateStatus`,
  expression objects with operator overloading, `pxt.udf`/`uda`/`query`/`iterator` decorators, `pxt.Int` ...),
  as thin wrappers that hold Rust handles and forward calls;
- things that are inherently Python: introspecting UDF signatures and type hints, pickling/resolving Python
  callables, converting PIL images, numpy arrays, pandas frames and pydantic models at the boundary;
- the public exception classes and `ErrorCode` (mirror `../pixeltable/exceptions.py`: same names, codes, classes);
- stubs for public submodules (`pixeltable.functions.video`, ...) whose functions are not implemented yet.

Python must not contain: catalog or row state, version bookkeeping, query planning or execution, evaluation
of builtin operators/functions, view propagation, or computed-column materialization. Rust may call back into
Python only to run user code (UDFs, UDAs, iterators), to (un)pickle callables, and for media I/O through
PIL/numpy. A grep for dicts of rows or loops over rows in `python/` is a review failure.

Error messages and `ErrorCode`s are produced in Rust and must match what the tests assert (codes are checked by
`pxt_raises`). Rust raises them by constructing `pixeltable.exceptions._error_class(code)(code, message)`.

## Core design

- Values: `Value` = Null | Bool | Int(i64) | Float(f64) | String | Timestamp (UTC micros) | Date | Uuid | Binary |
  Json (serde_json, plus opaque Python objects nested in JSON) | Array (dtype, shape, bytes) | Media (image/video/
  audio/document as a path or URL; images produced in memory are written to the media dir) | PyObj (opaque, in flight only).
- Types: mirror Pixeltable's ColumnType model (Int, Float, Bool, String, Timestamp, Date, UUID, Binary, Json[schema],
  Array[shape, dtype], Image[w,h,mode], Video, Audio, Document; nullable flag). Python `type_system` classes are the
  public objects; their algebra (supertype, matching, validation) is computed in Rust.
- Expressions: Python expression objects wrap a Rust `Expr` handle. Every operator, function call, JSON path, slice and
  method builds a new Rust node, which types itself at construction (so type errors surface where Pixeltable raises
  them). Node kinds: column ref, literal, arithmetic, comparison, compound predicate, is-null, isin, JSON path,
  JSON mapper, array slice, type cast, inline list/dict/array, function call (builtin or Python UDF), aggregate,
  window call, similarity. Exprs serialize to JSON (computed columns, view predicates, `Query.as_dict()`); Python UDFs
  serialize as a descriptor (module + qualname, or pickled bytes for local functions) that Python resolves on reload.
- Catalog: directories, tables, views (filter and/or iterator over a base at a version or live), snapshots, per-table
  data versions and schema versions, column ids that survive renames, comments and custom metadata, primary keys,
  btree and embedding indexes, `_is_data_versioned=False` (operational) tables.
- Storage: per table, rows `(rowid, v_min, v_max, cells, cell errors)`; a mutation creates a new version; time travel
  (`t:3`), snapshots and `revert()` read version ranges. Everything persists under `$PIXELTABLE_HOME` (default
  `~/.pixeltable`) in this package's own format (for example JSON catalog + one binary file per table, written
  atomically). After `pixeltable._testing.reload_catalog()` every handle must be re-resolvable and every query must
  return the same results from disk.
- Execution: row-at-a-time or batched interpretation in Rust; Python is called only for UDF/UDA/iterator bodies.
  Default scan order is rowid order. Order-by null placement matches Postgres (ASC nulls last, DESC nulls first).
  Inserts and updates materialize stored computed columns and propagate to views, in dependency order.

## Phases

1. Foundation (one agent): crate and package build with `maturin develop`; exceptions; type system; directories;
   create/get/drop/list tables; insert; select/where/collect with column refs, literals, arithmetic, comparisons;
   persistence and reload; the two test hooks. In parallel, a second agent builds the harness.
2. Fan-out (parallel agents, one git worktree each, merging to the branch often), each owning a test area:
   - tables and DDL: `test_dirs`, `test_table` (DDL, metadata, insert, schema), `test_alter_column`,
     `test_primary_key_index`, `test_types`, `test_array_type`, `test_uuid_type`
   - queries and expressions: `test_query`, `test_exprs` (JSON paths/mappers, aggregates, window functions, order by,
     group by, joins, distinct, sample via `test_sample`)
   - functions: `test_function` (udf, uda, expr_udf, query, overloads, batching), `test_iterator`,
     `test_component_view`
   - views and versions: `test_view`, `test_snapshot`, update/delete/revert/time travel parts of `test_table`
3. Judging (separate agents, not the implementers): run the harness, verify test files are byte-identical to
   upstream, audit the Python/Rust boundary, look for test-specific special cases, and report gaps.
   Findings go back to implementers; phases 2 and 3 repeat.

## Validation

- `compat/sync_tests.py` copies the selected upstream test files verbatim into `compat/tests/`, and
  `compat/check_unmodified.py` fails if any copied file differs from `../tests/`.
- `compat/tests/conftest.py` and `compat/tests/utils.py` replace the upstream ones, using only the public API plus
  the hooks `pixeltable._testing.reset_db()` (drop everything, start empty) and `pixeltable._testing.reload_catalog()`
  (drop all in-memory state; the next call reloads from disk). Only the `local` database root exists.
- Upstream test modules import some internal modules at the top (`pixeltable.env`, `pixeltable.exprs`, ...). The
  harness satisfies those imports with stub modules whose attributes raise when used, so a test that touches an
  internal fails instead of the whole file failing to import. The package itself does not ship internal modules.
- `compat/run.py` runs the selected files and writes a summary (passed/failed/error per test) to
  `compat/results/latest.json`; `compat/PASSING.txt` lists the tests that pass, and a regression is any test that
  leaves it.
- Never special-case a test (by name, table name or literal) in the package; fix the general behavior instead.
