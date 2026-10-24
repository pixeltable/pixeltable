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
  functions/        builtin registry (mod.rs: Builtin, FuncRef, make_call) + one file per module (string.rs, math.rs,
                    uuid.rs, image.rs (PIL), video.rs (PyAV metadata)); Python functions: signature.rs (binding
                    calls like inspect.Signature.bind), udf.rs (UDF/template calls), iterator.rs (iterator calls)
  media.rs          media cells: path/URL normalization, file cache, writing in-memory media, validation
  catalog/mod.rs    Catalog: dirs, tables (Arc<TableMd>), lazily loaded row data; commit(); lookups
  catalog/path.rs   Path parsing/validation/folding (upstream messages)
  catalog/table.rs  TableMd, ColumnMd, SchemaVersionMd, VersionMd, ViewMd, ResolvedColumn
  catalog/dirs.rs   directory ops, path-collision handling (if_exists), listings
  catalog/tables.rs create/get/drop table, add/drop/rename column, version bumps
  catalog/views.rs  views/snapshots (filter over a base, live or pinned), column resolution through bases
  catalog/indexes.rs B-tree and embedding index metadata, default indexes, drop/cascade
  catalog/runtime.rs the process-wide Catalog behind a mutex; reload/reset hooks; reserved column names
  store/mod.rs      files under $PIXELTABLE_HOME, atomic writes
  store/rows.rs     TableData / StoredRow with version ranges, cell errors
  exec/mod.rs       QuerySpec (select/where/order_by/limit, validation, JSON), QueryInput snapshot, execute(), count()
  exec/insert.rs    insert pipeline: InsertPlan -> build_rows -> commit
  exec/compute.rs   computed columns: eval_computed(), AddComputedPlan (plan -> compute -> commit)
  exec/compute_rows.rs Table.compute(): rows a table would hold for some input rows, nothing stored
  exec/index.rs     AddIndexPlan: embeddings of existing rows for a new embedding index
  bindings/         PyO3 glue only: types.rs (_core.ColumnType), expr.rs (_core.Expr, MethodRef), catalog.rs
                    (module-level functions), table.rs (tbl_* functions), query.rs (QuerySpec, ResultSet, cursor),
                    sources.rs (data sources: rows, iterators, pydantic models, queries; schema inference),
                    indexes.rs (index operations, similarity(), describe data),
                    func.rs (_core.Udf, _core.Iterator, _core.IteratorCall, the function registry)
python/pixeltable/
  __init__.py       the public names (same as upstream's __init__)
  exceptions.py     copy of upstream's public module
  type_system.py    public type classes; each instance holds `_core: _core.ColumnType`; type-hint interpretation
  types.py          TypedDicts (ColumnSpec, DirContents, TreeNode, ...), as upstream
  _globals.py       pxt.create_table() etc.: argument forwarding
  _table.py         Table/InsertableTable/View handles (id + pinned version), Dir
  _query.py         Query (wraps _core.QuerySpec), ResultSet (wraps _core.ResultSet), ResultCursor
  _exprs.py         Expr = _core.Expr
  func/             introspection of user functions: _signature.py (Signature from type hints), _function.py
                    (Function base, serialization, symbol resolution), _callable.py (@pxt.udf), _expr_template.py
                    (@pxt.expr_udf, using()), _iterator.py (@pxt.iterator), _udf.py (decorators, registry), ...
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
pixeltable-rs")`. User input must never panic (use checked arithmetic, char-safe slicing); as a safety net,
`with_catalog` turns a panic into an InternalError and rolls the operation back.

## Values and types

`Value` is Null | Bool | Int(i64) | Float(f64) | String | Timestamp(UTC micros) | Date(days since 1970-01-01) |
Uuid | Binary | Json(serde_json) | Array(Arc<NdArray>) | Media(path or URL) | PyObj (in flight only, never stored).
Media cells (`Value::Media`) hold an absolute local path or a URL
(`media::normalize_input()`); remote files are downloaded on first use into `$PIXELTABLE_HOME/file_cache/`
(`media::local_path()`). In-memory media (PIL images, bytes, images computed by builtins) are written to
`media/<db>/<tbl id>/<tbl id>_<col id>_<version>_<uuid>[.ext]` (JPEG, or WebP with transparency) by
`exec::compute::MediaTarget::materialize()` during insert/add_computed_column, outside the lock; the commit adopts the
files (`Catalog::adopt_media()`), so a failed operation deletes them. `Catalog::defer_delete_media_version()` deletes
the files of one table version (for revert). Write-validated media columns are validated in
`exec::insert::build_rows()` (PIL/PyAV, upstream messages); read-validated ones by `media::ReadValidation` in the query
context (an invalid file reads as NULL with a cell error). Column properties (`errortype`, `errormsg`, `fileurl`,
`localpath`) are `ExprKind::ColumnProperty` nodes; `RowContext::column_error()` supplies cell errors. A computed column
whose inputs have a cell error gets that error (`eval_computed()`). Timestamps: naive inputs are interpreted in the default time zone (`PIXELTABLE_TIME_ZONE`, `TZ`, the
system zone, else UTC) and results are returned as aware datetimes in it.

Nesting limits: JSON values hold at most `value::MAX_JSON_DEPTH` (500) levels of containers and Python-visible
expressions at most `expr::MAX_EXPR_DEPTH` (500) levels (`Expr::depth`, checked by `bindings::expr::wrap()` and
`from_object()`), so no recursive conversion, evaluation, serialization or drop can exhaust the stack. Recursive code
over Python input must carry a depth (see `value::py::json_from_py_at`). The store enforces the same limits on write
and read (below), so every value that inserts also reloads.

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
floor division. Integer arithmetic is checked: overflow raises Postgres' "bigint out of range" (as upstream's SQL
does), never wraps or panics.

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

## Python functions (UDFs, templates, iterators)

The boundary: `pixeltable.func` introspects user code (type hints -> `Signature`, the callables, template
expressions built by calling an `@pxt.expr_udf` body with `Variable` nodes) and creates the backend half once, on
first use: `Function._core_udf()` -> `_core.Udf(fn, spec)` (`bindings/func.rs`) -> `functions::udf::UdfDef`. Calling a
function (`Function.__call__` -> `Udf.call`) happens entirely in Rust (`udf::make_call`): `Signature::bind` reproduces
`inspect.Signature.bind` and its TypeError texts (upstream shows them verbatim), overloads are tried in declaration
order, `validate_args` type-checks, the result type is the declared (or conditional, computed by calling back
`Function._call_conditional_return_type`) return type, made nullable when a nullable argument meets a non-nullable
parameter. The node is a `FunctionCall` whose `FuncRef::Udf` holds a `UdfCall` (def, chosen signature, binding,
`is_method_call`, for templates the instantiated expression, and `validation_error` for stored calls that no longer
match the code). `Expr.fn` returns the resolved Python `Function`.

Evaluation (`udf::eval_call`): NULL arguments of non-nullable parameters yield NULL without calling; the Python
callable gets `value::py::to_py` values (batched UDFs get singleton batches; coroutines run via
`pixeltable.func._runtime`); the result goes through `value::py::coerce`. A Python exception propagates as
`Error::Python`, so `on_error='ignore'` records `errortype` = its class name and `errormsg` = `str(exc)`. Templates
evaluate their instantiated expression (`udf::substitute` replaces `Variable`s and re-binds nested calls).

Persistence: a UDF call serializes as `{"udf": Function.as_dict(), "is_method_call": ...}` (by path for module
functions, pickled for local ones; computed once per def). `udf::call_from_json` (from `expr/serde.rs`) resolves the
descriptor through `pixeltable.func._resolve_function` (an unresolvable path becomes an `InvalidFunction`), re-binds
the stored call pattern and checks the stored type against the current return type; failures become the call's
`validation_error` with upstream's texts (`udf::validation_errors(expr)` lists them). Computed columns reject
non-storable functions (`udf::validate_storable`, called by `validate_column_spec`). Module functions are registered
in `udf::register_function` (FUNCTION_ALREADY_EXISTS on duplicates); `is_method`/`is_property` UDFs are found by
`Expr.__getattr__` through `bindings::func::udf_attribute`.

Query templates (`@pxt.query`, `pxt.retrieval_udf()`): `UdfImpl::Query` holds the template `QuerySpec` (built by
calling the decorated function with `Variable`s); a call binds its arguments as literals (`QuerySpec::map_exprs`),
snapshots and executes the query and returns the rows as a JSON list. They serialize by value; their Python
`_from_dict()` must not touch the catalog (it runs under the catalog lock: deserializing a stored expression resolves
its functions in Python; `with_catalog` fails instead of deadlocking on re-entry).

Stored calls that no longer match the code: `get_table()` emits upstream's "no longer valid" `PixeltableWarning` once
per catalog load per table (`udf::load_warnings`), and inserting raises INVALID_STATE "Data cannot be inserted into
..." (`udf::check_valid_column`, also meant for update/recompute with op "updated in"/"computed for").

Iterators (`functions/iterator.rs`, for component views): `GeneratingFunction._core_iterator()` -> `_core.Iterator`
-> `IteratorDef`; calling it (`iterator::make_call`) binds/type-checks (UNSUPPORTED_OPERATION "Invalid iterator
arguments: ..."), runs the iterator's `validate()` on constant arguments and gets the output schema from Python
(`GeneratingFunction._check_call`), then fixes `outputs`: `pos` (not stored) followed by the fields with folded names
and `is_stored` (false for `unstored_cols`). Python wraps the result (`_core.IteratorCall`) in a
`GeneratingFunctionCall`; `bindings::func::iterator_call_from_py(obj)` gets the `Arc<IteratorCall>` back for
`create_view(iterator=...)`. The API a component view uses:
- `call.value_outputs()`: the output columns other than `pos`, in order; `call.arg_exprs()`: what it reads of the base;
- `call.eval_rows(ctx)`: runs the iterator for one base row (`ctx` resolves the base columns): one `Vec<Value>` per
  output row (values of `value_outputs()`), in `pos` order; user exceptions propagate as `Error::Python`;
- `call.seek_row(ctx, pos, stored)`: the outputs of row `pos` again (for unstored outputs), via `seek(pos, **stored)`;
- `call.to_json()` / `IteratorCall::from_json()` for `ViewMd`; `from_json` re-binds against the current code and
  sets `validation_error` (signature or output-schema changes, unresolvable iterator);
- `call.check_valid("View 'v'", "v", "updated in")`: upstream's INVALID_STATE error for an invalid stored call;
- `call.display()`: `iterator_call` in `get_metadata()`.

A running iterator is a `RowSource` (`next_row`, `seek`, `close`): Python iterators run through `PyRowSource`;
builtin iterators (`BuiltinIterator` entries in `iterator.rs::BUILTIN_ITERATORS`: `video_iterators.rs`
`frame_iterator`/`legacy_frame_iterator` over PyAV, `image_iterators.rs` `tile_iterator` over PIL) implement it in
Rust, and their Python objects (`func._iterator.BuiltinGeneratingFunction`, what the generated stubs resolve to) are
resolved by path like Python iterators. A NULL argument for a non-nullable parameter yields no rows. Test hooks:
`_core._iterator_rows(call, {col: value})` and `_core._iterator_seek(call, {col: value}, pos, stored)`.

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
  starts with a stamp (`catalog_id`, `generation`) that every commit advances;
- `tables/<tbl id>.v<version>.<seq>.bin`: MessagePack `TableData` (all row versions of a table) as of the version
  named in `TableMd.data_file`; written before `catalog.json`, superseded files deleted after. JSON cells are stored
  as JSON text;
- `catalog.lock`: the file lock below.

Writes go to a temp file, fsync, rename, fsync of the directory. JSON is read without serde_json's 128-level limit
but with the limit its writer enforced (`value::parse_json_bounded`: `MAX_JSON_DEPTH` for cells,
`store::MAX_CATALOG_DEPTH` for `catalog.json` and `from_dict()`).

Several processes may share a store. Every `with_catalog` call holds an exclusive OS file lock on `catalog.lock`
(`Store::lock()`, taken after the in-process mutex, detached from the interpreter while waiting) and compares the stamp
at the head of `catalog.json` (`Store::read_stamp()`, a 512-byte read) with the in-memory one; if another process
committed, the catalog and its cached row data are reloaded before the closure runs. A plan/compute/commit operation
therefore commits against the latest state (its commit step re-checks the schema version as usual).

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

## Data sources, compute(), indexes

Data sources (`bindings/sources.rs`, upstream `TableDataConduit`): `classify()` sorts a source into rows of dicts
(lists, materialized iterators), pydantic models (dumped before any catalog access, validated against the table as
upstream does) or a query (a `Table` or `Query`, executed in Rust and inserted value by value). `create_table(source=
...)` infers the schema from rows (`infer_schema_from_rows()`, with `schema_overrides`) or takes the query's schema,
creates the table, then inserts. `insert(return_rows=True)` reports the stored values (media by file URL).
`Table.compute()` (`exec/compute_rows.rs`) evaluates only the requested outputs and their dependencies (stored and
unstored computed columns alike) for rows of the root base table, applies view filters, and stores nothing.
Computed columns are evaluated with Python semantics (`RowContext::python_semantics()`: a division by zero raises
`ZeroDivisionError`), query expressions with SQL semantics (NULL). `UpdateStatus.cols_with_excs` names columns
`<table>.<column>` (`<table>.<column>:<index>` for index values).

Indexes (`catalog/indexes.rs`, `TableMd.indexes`): B-tree indexes are metadata only (scans are full scans);
`has_default_idxs` adds one to every eligible column (`push_column()`, `create_table()`). An embedding index stores the
embeddings of its column in a hidden stored computed column (`IndexKind::Embedding::val_col_id`, not listed in any
schema version, computed by inserts like other stored computed columns, named `<col>:<idx>` in returned rows); its
embedding functions are kept as `Function.as_dict()` descriptors per query modality. `col.similarity(...)` builds an
`ExprKind::Similarity` node that holds the embedding call of the query item (computed once per expression) and
compares it with the row's stored embedding (cosine similarity, inner product or L2 distance). Indexes on views are
not supported yet (views have no rows of their own).
