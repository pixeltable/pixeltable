# pixeltable-rs architecture

A map of the internals for contributors. PLAN.md has the goals and the rules (the Python/Rust boundary is
binding); this file says where things are and how to extend them. Keep it current when you change a mechanism.

## Build and test

```sh
cd pixeltable-rs
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop            # debug build, installs `pixeltable` (editable)
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release  # optimized
PYO3_PYTHON=$PWD/.venv/bin/python cargo test --lib          # Rust unit tests
.venv/bin/python -m pytest                                  # own end-to-end tests (tests_rs/)
cd compat && ../.venv/bin/python run.py test_dirs test_table test_query   # upstream tests (see compat/README.md)
.venv/bin/python tools/gen_function_stubs.py                # regenerate pixeltable.functions/iterators/io stubs
```

Never run Python with the repository root (`/home/user/pixeltable`) as cwd or on `sys.path`: `import pixeltable`
would pick up upstream.

## Layout

```
src/
  lib.rs            module declarations + #[pymodule] _core (nothing else)
  error.rs          ErrorCode (mirrors the Python enum by name), Error, pxt_err!/bail!, py_repr()
  types.rs          ColumnType = TypeKind + nullable: matches, supertype, repr, as_dict/from_dict, array subscripts
  value.rs          Value (cell values), NdArray, ordering/equality (Postgres semantics), py_float_repr
  value/py.rs       Python <-> Value: coerce() (create_literal), validate() (validate_literal), infer_literal(),
                    to_py() (query results), default time zone, numpy arrays via rust-numpy
  expr/mod.rs       Expr IR: Expr { kind: ExprKind, col_type }, ExprRef = Arc<Expr>, operator enums
  expr/build.rs     typed constructors (type the node, raise upstream's construction-time errors)
  expr/display.rs   str(expr) as upstream renders it
  expr/serde.rs     Expr <-> JSON, tagged Value <-> JSON
  expr/eval.rs      RowContext trait, eval(), SQL-like NULL semantics
  functions/        builtin registry (mod.rs: Builtin, FuncRef, make_call) + one file per module (string.rs, math.rs)
  catalog/mod.rs    Catalog: dirs, tables (Arc<TableMd>), lazily loaded row data; commit(); lookups
  catalog/path.rs   Path parsing/validation/folding (upstream messages)
  catalog/table.rs  TableMd, ColumnMd, SchemaVersionMd, VersionMd, ViewMd, ResolvedColumn
  catalog/dirs.rs   directory ops, path-collision handling (if_exists), listings
  catalog/tables.rs create/get/drop table, add/drop/rename column, version bumps
  catalog/views.rs  views/snapshots (filter over a base, live or pinned), column resolution through bases
  catalog/runtime.rs the process-wide Catalog behind a mutex; reload/reset hooks; reserved column names
  store/mod.rs      files under $PIXELTABLE_HOME, atomic writes
  store/rows.rs     TableData / StoredRow with version ranges, cell errors
  exec/mod.rs       QuerySpec (select/where/order_by/limit, validation, JSON), QueryInput snapshot, execute(), count()
  exec/insert.rs    insert pipeline: InsertPlan -> build_rows -> commit
  exec/compute.rs   computed columns: eval_computed(), AddComputedPlan (plan -> compute -> commit)
  bindings/         PyO3 glue only: types.rs (_core.ColumnType), expr.rs (_core.Expr, MethodRef), catalog.rs
                    (module-level functions), table.rs (tbl_* functions), query.rs (QuerySpec, ResultSet, cursor)
python/pixeltable/
  __init__.py       the public names (same as upstream's __init__)
  exceptions.py     copy of upstream's public module
  type_system.py    public type classes; each instance holds `_core: _core.ColumnType`; type-hint interpretation
  types.py          TypedDicts (ColumnSpec, DirContents, TreeNode, ...), as upstream
  _globals.py       pxt.create_table() etc.: argument forwarding
  _table.py         Table/InsertableTable/View handles (id + pinned version), Dir
  _query.py         Query (wraps _core.QuerySpec), ResultSet (wraps _core.ResultSet), ResultCursor
  _exprs.py         Expr = _core.Expr
  func/__init__.py  placeholder Function/udf/uda/... (decorators work, calls raise NotImplementedError)
  functions/, iterators/, io/   generated stubs; functions/_registry.py resolves names to Rust builtins
  _metadata.py, _update_status.py, _row.py   public data classes copied from upstream
  _testing.py       reset_db(), reload_catalog()
  _unimplemented.py helpers for not-yet-implemented API
tools/gen_function_stubs.py   generates the stub modules from upstream sources (ast only)
tests_rs/                     pixeltable-rs's own tests
```

Do not add Python modules named like upstream internals (`env`, `exprs`, `catalog`, `exec`, `utils`, ...): the
compat harness stubs those names. Private modules start with `_`.

## Errors

Rust returns `error::Result<T>`. `Error::Pxt { code, message }` becomes, at the boundary (`impl From<Error> for
PyErr`), `pixeltable.exceptions._error_class(code)(code, message)`, so the class always matches the code. Other
variants map to TypeError/AttributeError/ValueError/NotImplementedError, and `Error::Python` re-raises an exception
from Python code that Rust called. Write messages exactly as upstream formats them; `py_repr(s)` reproduces `{s!r}`.
Unimplemented features: `Error::not_implemented("what")` -> `NotImplementedError("what is not implemented yet in
pixeltable-rs")`.

## Values and types

`Value` is Null | Bool | Int(i64) | Float(f64) | String | Timestamp(UTC micros) | Date(days since 1970-01-01) |
Uuid | Binary | Json(serde_json) | Array(Arc<NdArray>) | Media(path or URL) | PyObj (in flight only, never stored).
In-memory media (PIL images, bytes) are written to `media/<db>/<tbl id>/` on insert (`bindings/table.rs`,
`save_media`). Timestamps: naive inputs are interpreted in the default time zone (`PIXELTABLE_TIME_ZONE`, `TZ`, the
system zone, else UTC) and results are returned as aware datetimes in it.

`ColumnType` serializes to upstream's `as_dict()` form (`{'_classname': 'IntType', 'nullable': False}`), which is
also what the catalog stores. The Python wrapper classes are built with `ColumnType._from_core(core)`.

## Expressions

Python `_core.Expr` objects hold an `ExprRef`. Every operator calls a constructor in `expr/build.rs`, which computes
the node type and raises upstream's construction-time errors; operands that are not expressions become literals
(`bindings/expr.rs`, `operand()` / `from_object()`). Node kinds: ColumnRef {tbl_id, col_id, name, tbl_version},
Literal, Arithmetic, StringOp, Comparison (a literal on the left of a column is moved right), Compound (& | ~),
IsNull, InList/InExpr (isin), TypeCast, InlineList/Dict/Array (collapse to literals when constant), JsonPath and
ArraySlice (IR only, evaluation not implemented), FunctionCall {func: FuncRef, args, kwargs}.

To add a node kind: variant in `ExprKind` + `children()`, constructor in `build.rs`, arms in `display.rs`,
`serde.rs`, `eval.rs`, and the Python-facing method in `bindings/expr.rs`.

Serialization (`expr/serde.rs`): `{"_classname": ..., "col_type": <type dict>, "components": [...], ...}`; literal
values use a tagged encoding (`{"$ts": micros}`, `{"$json": ...}`, `{"$array": ...}`) so they round-trip exactly.
Computed columns store their value expression in `ColumnMd.value_expr`; views store their predicate in
`ViewMd.predicate`; `Query.as_dict()` embeds expressions the same way.

Evaluation (`expr/eval.rs`) is row at a time against a `RowContext` (column values by `ColumnRef`). NULL
propagates, comparisons with NULL are NULL, a NULL predicate filters the row out, division by zero is NULL, `//` is
floor division.

## Functions

Builtins: a `Builtin` (name, module, params, required count, `strict`, `return_type`, `eval`) in a static table per
module, e.g. `functions/string.rs::FUNCTIONS`, registered in `functions/mod.rs::registry()`. `make_call()` binds
keyword arguments by name, types the call, and makes the result nullable when a `strict` function gets a nullable
argument (NULL arguments short-circuit to NULL). The Python side needs nothing more: the generated stub
`upper = _fn(__name__, 'upper')` in `pixeltable/functions/string.py` resolves to a `BuiltinFunction` (which calls
`_core.call_builtin`) as soon as the backend registers `pixeltable.functions.string.upper`, and `t.c.upper()` works
through `Expr.__getattr__` -> `MethodRef` (the module is chosen by the value's type, `method_module()`).

To add a builtin: implement `return_type` and `eval`, add the `Builtin` entry, rebuild. Aggregates and window
functions need a new trait (init/update/value) and a grouping stage in `exec` (not built yet).

Python UDFs (not built yet): `FuncRef::Udf(Arc<UdfRef>)` holds the callable and a descriptor (module + qualname, or
pickled bytes) that `pixeltable.func` produces and resolves when an expression is deserialized
(`FuncRef::from_json` has the hook). `FuncRef::call` is where evaluation calls back into Python (`Python::attach`,
convert args with `value::py::to_py`, convert the result with `value::py::coerce` against the declared return type).
The placeholder decorators in `python/pixeltable/func/__init__.py` already capture the callables.

## Catalog, storage and versioning

`Catalog` (in-memory) holds directories (`DirMd`), table metadata (`TableMd`) and, lazily, row data
(`Arc<TableData>`). `catalog::runtime::with_catalog(py, |cat| ...)` gives exclusive access; the catalog is loaded
from the store on first use, dropped by `reload_catalog()` and wiped by `reset_db()`.

Atomicity: each `with_catalog` call is atomic. `Catalog::checkpoint()` captures the in-memory state (maps of `Arc`s,
cheap); if the closure fails, `Catalog::finish()` restores it, rewrites `catalog.json` if the operation had already
committed, and removes the data files it wrote. Never delete files directly: use `defer_delete_*` (applied only
when the operation succeeds). An operation that spans several lock acquisitions (plan/compute/commit) is atomic
only in its commit step, so put every mutation into the commit step (see `AddComputedPlan::commit`, which also
performs the `if_exists='replace'` drop).

Locking rule: never run user Python code (UDFs, generators, pydantic serializers) while holding the catalog lock.
Operations that may do so are split into plan (lock) -> compute (no lock) -> commit (lock, re-check the schema
version): see `exec/insert.rs` and `exec/compute.rs`. Queries snapshot `Arc`s of metadata and data
(`exec::QueryInput::snapshot`) and execute without the lock.

Storage (`store/mod.rs`), format version 1, under `$PIXELTABLE_HOME/pxtrs/<db>/` (db = `PIXELTABLE_DB`, default
`pixeltable`):
- `catalog.json`: all directories and table metadata; atomically replaced on every mutation (the commit point);
- `tables/<tbl id>.v<version>.<seq>.bin`: MessagePack `TableData` (all row versions of a table) as of the version
  named in `TableMd.data_file`; written before `catalog.json`, superseded files deleted after.

Versioning: a data-versioned table gets a new version per mutation; a schema change also creates a schema version
equal to the new table version (`Catalog::bump_version`). Rows carry `[v_min, v_max)`; inserts add rows at the new
version, deletes/updates will close `v_max` (updates re-insert with the same rowid). Cells are indexed by column id,
so added columns read NULL for older rows and dropped columns keep their cells for older versions. Operational tables
(`_is_data_versioned=False`) bump the version only for schema changes. `get_table('t:3')` pins a handle to version 3
(queries read rows and schema as of 3).

Performance note: a mutation currently copies the table's whole `TableData` and rewrites its data file (O(rows)
per insert). The intended evolution is segment-based storage (`TableData` = list of `Arc` segments, one file per
segment; inserts add a segment, deletes/updates rewrite only the affected segments).

Views (`catalog/views.rs`) have no rows of their own yet: a query over a view scans its base (at the pinned version
for snapshots) with the view predicates applied (`ScanSource`), and the view's columns are its base's columns. To
add view columns, iterators or materialized snapshots, give views their own `TableData` and join on base rowids.

To add a catalog operation: implement it as a `Catalog` method in the matching `catalog/*.rs` file (validate, mutate,
`self.commit()`), expose it as a `#[pyfunction]` in `bindings/catalog.rs` or `bindings/table.rs` (convert arguments,
call inside `with_catalog`), register it in `bindings/mod.rs`, and forward to it from the Python method.

## Queries

`exec::QuerySpec` is the state of a `pxt.Query`; its `with_*` methods validate as upstream does. Execution:
scan visible rows (rowid order) -> view predicates and where clause -> stable sort (ASC nulls last, DESC nulls first)
-> offset/limit -> project. Unstored computed columns are evaluated on read. Results are a `ResultData` held in Rust
(`_core.ResultSet`); values convert to Python objects on access. Not built yet: joins, group by/aggregates, window
functions, distinct, sample, update/delete, JSON paths, array slices.
