# pixeltable-rs

A functional clone of [Pixeltable](https://github.com/pixeltable/pixeltable) in which only the public API is Python and
the backend (catalog, storage, versioning, expression typing and evaluation, query planning and execution) is Rust,
built as a PyO3 extension with maturin. It aims to behave like Pixeltable as Pixeltable's own test suite observes it;
the internals are different (no Postgres, no SQLAlchemy).

- [PLAN.md](PLAN.md): goals, the Python/Rust boundary rules, phases and how the clone is validated.
- [ARCHITECTURE.md](ARCHITECTURE.md): the internals as built.
- [compat/README.md](compat/README.md): the harness that runs upstream Pixeltable tests against the clone.
  [compat/PASSING.txt](compat/PASSING.txt) lists the upstream tests that pass.

## Layout

| Path | Contents |
|---|---|
| `src/` | Rust crate `pxtcore`, built into `pixeltable._core` |
| `python/pixeltable/` | The public Python API: thin wrappers over `_core` |
| `compat/` | Upstream test files, copied byte for byte, with a replacement `conftest.py` and `utils.py` |
| `tests_rs/` | The clone's own tests |
| `tools/` | Generators (function stubs from upstream's source) |

## Concurrency of UDFs

Unlike upstream Pixeltable, which runs sync UDFs one call at a time, pixeltable-rs runs the calls of a UDF for many
rows concurrently:

- Calls of a **sync UDF** may run in parallel on a pool of worker threads, in any order. Calls that wait on I/O or
  release the GIL (network requests, `time.sleep`, numpy, PIL, torch) overlap; calls that would run no faster on
  several threads (pure-Python CPU-bound work, which holds the GIL, and short calls that contend for it) are detected
  and run one after the other, as upstream runs them.
- Calls of an **async UDF** are all in flight together on one event loop (up to `max_concurrency` per function), and
  functions with a `resource_pool` follow the pool's request rate and retry failed calls that look retryable, as
  upstream does.
- A **batched UDF** (`batch_size=...`) gets full batches, several in flight at once.
- Independent UDF calls of one operation (two computed columns filled by one insert, two calls in one expression) run
  concurrently.

Results are the same as with sequential execution: each value is placed in its row, an exception is recorded for the
row that raised it (a failed batch fails each of its rows), `on_error='abort'` reports the error of the first failing
row in row order (and calls for later rows are not started), and a `KeyboardInterrupt` aborts the operation (also one
raised by an async UDF). As in
upstream, every UDF call of an expression is made for every row, also where `&` or `|` would not need its result.

A UDF that is not thread-safe (it mutates shared state without a lock, or uses a library that must not be called
from several threads) declares `@pxt.udf(max_concurrency=1)`: its calls then run one at a time, in row order, in the
thread that runs the operation. `max_concurrency=n` caps the calls in flight at `n` for sync and async UDFs alike.

| Setting (environment variable / `[pixeltable]` key in `$PIXELTABLE_HOME/config.toml`) | Default |
|---|---|
| `PIXELTABLE_UDF_THREADS` / `udf_threads`: worker threads for sync UDFs; 1 runs every call in the calling thread, in row order (upstream's behavior) | 4 per CPU, 4 to 32 |
| `PIXELTABLE_UDF_ASYNC_CONCURRENCY` / `udf_async_concurrency`: calls of one async UDF in flight at once | 1024 |
| `<ENDPOINT>_RATE_LIMIT` / `[<endpoint>] rate_limit`: requests per minute of `resource_pool='request-rate:<endpoint>'` (`[<endpoint>.rate_limits] <model>` for `'request-rate:<endpoint>:<model>'`) | 600 |

On free-threaded CPython (3.13t, 3.14t: `uv venv --python 3.14t`, then the setup below) the extension runs without the
GIL, so pure-Python CPU-bound UDFs run in parallel too (a 1 ms pure-Python UDF over 1,000 rows: 3.5x faster on 4 cores
than with the GIL).

Builtin-only work (arithmetic, string functions, filters, grouping, joins, sorting, similarity) runs without the GIL and,
over many rows, on all cores, with results and errors as a single pass would produce them.

`bench/run.py` measures these paths; see [ARCHITECTURE.md](ARCHITECTURE.md), "Concurrency of user code", for how it
works.

## Performance

Measured with `bench/run.py` (a fresh process and `$PIXELTABLE_HOME` per benchmark) and `bench/run_storage.py`, the
builds run one after the other for each case, on a 4-core Linux VM that other jobs shared; medians of three runs
(storage: of three repeats; UDF work: one run). Upstream is Pixeltable 0.7.21.dev with its bundled Postgres, on the
same Python 3.13 and the same scripts (the upstream column is the time of upstream's own API).

Interactive queries and mutations on a 1M-row table (`a` int, `b` float, `g` int with 1,000 values, `s` string; benchmarks
`interactive_1m`, `builtin_1m`, `similarity_100k`):

| operation | upstream | pixeltable-rs | speedup | case |
|---|---|---|---|---|
| `head(10)` | 12.0 ms | 0.31 ms | 38x | `head10` |
| `select(a, s).limit(10)` | 11.9 ms | 0.29 ms | 41x | `limit10` |
| `where(a == k)` (no index), collect | 34.2 ms | 3.0 ms | 12x | `where_eq_noindex` |
| `where(g == k).select(s, b)`, collect (1,000 rows) | 53.1 ms | 3.4 ms | 15x | `where_g_collect` |
| `where((b > 0.5) & (g < 500)).count()` | 38.0 ms | 8.1 ms | 4.7x | `filter_1m` |
| `group_by(g)` with sum, mean, count | 93.7 ms | 41.1 ms | 2.3x | `groupby_1m` |
| `order_by(b).limit(10)` | 55.2 ms | 4.0 ms | 14x | `sort_1m` |
| join with a 1,000-row table on `g`, count | 69.4 ms | 22.7 ms | 3.1x | `join_1m` |
| `update()` of 1 row, predicate on a column without index | 100.7 ms | 3.8 ms | 26x | `update_1row_noindex` |
| `delete()` of a few rows, predicate without index | 66.1 ms | 4.1 ms | 16x | `delete_few_noindex` |
| reload, then first `head(10)` | 26.6 ms | 8.9 ms | 3.0x | `reload_first_head` |
| reload, then first `where(a == k).count()` | 51.4 ms | 17.3 ms | 3.0x | `reload_first_count` |
| reload, then first `where(g == k).select(s, b)` | 66.2 ms | 33.7 ms | 2.0x | `reload_first_where_collect` |
| similarity top 10 of 100k 64-d embeddings (exact scan; upstream: its HNSW index) | 14.7 ms | 4.9 ms | 3.0x | `similarity_100k` |
| `select(a * 2 + b / 3 - g)`, collect | 14.68 s | 167 ms | 88x | `arith_1m` |
| `select(s.upper() + '_x')`, collect | 15.88 s | 222 ms | 72x | `string_1m` |
| `create_table(source=DataFrame)` | 38.4 s | 4.0 s | 9.5x | `create_1m` |
| `add_computed_column(a * 2 + g)` | 34.9 s | 2.5 s | 14x | `add_column_1m` |
| `add_embedding_index` (numpy UDF), 100k rows | 33.7 s | 2.7 s | 12x | `embed_index_100k` |

Storage operations on a 100k-row table with a primary key (`run_storage.py`):

| operation | upstream | pixeltable-rs | speedup | metric |
|---|---|---|---|---|
| `where(id == k).collect()` (primary key) | 7.13 ms | 0.041 ms | 174x | `pk_lookup_ms` |
| `where(id == k).count()` | 10.5 ms | 0.12 ms | 87x | `pk_count_ms` |
| `where(s == v).collect()`, no index | 17.2 ms | 0.87 ms | 20x | `filter_lookup_ms` |
| `where(s == v).collect()`, B-tree index on `s` | 6.97 ms | 0.029 ms | 240x | `indexed_filter_lookup_ms` |
| `count()`, table loaded | 19.2 ms | 0.08 ms | 240x | `warm_count_ms` |
| reload, then first `count()` | 33.4 ms | 1.96 ms | 17x | `reload_first_count_ms` |
| reload, then first `where(id == k).collect()` | 21.8 ms | 1.55 ms | 14x | `reload_first_lookup_ms` |
| `update()` of 1% (a key range) | 49.8 ms | 4.73 ms | 11x | `update_1pct_ms` |
| `batch_update()` of 1% | 83.1 ms | 9.04 ms | 9.2x | `batch_update_1pct_ms` |
| `update()` of 1 row (by primary key) | 20.1 ms | 2.41 ms | 8.3x | `update_1row_ms` |
| `delete()` of 1% (a key range) | 16.2 ms | 3.27 ms | 4.9x | `delete_1pct_ms` |
| `delete()` of 1 row (by primary key) | 9.90 ms | 1.58 ms | 6.3x | `delete_1row_ms` |

UDF-heavy work (`bench/run.py`, one run each):

| operation | upstream | pixeltable-rs | speedup | case |
|---|---|---|---|---|
| insert, computed column with a 10 ms sleeping UDF, 1,000 rows | 10.30 s | 0.70 s | 15x | `sync_sleep_insert` |
| `select()` of the same UDF, 1,000 rows | 10.26 s | 0.70 s | 15x | `sync_sleep_select` |
| `add_computed_column()` of the same UDF, 1,000 rows | 10.33 s | 0.70 s | 15x | `add_column_sleep` |
| `compute()` of 1,000 rows with the same UDF | 10.24 s | 0.70 s | 15x | `compute_sleep` |
| component view (3 rows per base row) with the same UDF, 500 base rows | 15.41 s | 1.03 s | 15x | `component_view_sleep` |
| two independent columns with such UDFs, insert of 500 rows | 10.29 s | 0.70 s | 15x | `two_cols_insert` |
| async UDF awaiting `asyncio.sleep(0.05)`, insert of 1,000 rows | 0.153 s | 0.065 s | 2.4x | `async_sleep` |
| batched UDF (16 rows per batch) sleeping 20 ms per batch, 1,000 rows | 1.341 s | 0.209 s | 6.4x | `batched_sleep` |
| PIL resize of a 1024x1024 image, 1,000 rows | 11.38 s | 3.42 s | 3.3x | `pil_resize` |
| numpy 300x300 matrix product, 1,000 rows | 1.188 s | 0.416 s | 2.9x | `numpy_matmul` |
| pure-Python CPU-bound UDF (~1 ms, holds the GIL), 1,000 rows | 0.923 s | 0.894 s | 1.0x | `py_cpu_loop` |
| trivial pure-Python UDF, insert of 100,000 rows | 4.26 s | 0.161 s | 26x | `py_cpu_tiny` |
| trivial pure-Python UDF, `select()` of 100,000 rows | 3.16 s | 0.053 s | 59x | `py_cpu_tiny_select` |
| `group_by()` with a Python UDA, 100,000 rows in 100 groups | 1.107 s | 0.029 s | 38x | `uda_100k` |

What makes the interactive operations fast (see [ARCHITECTURE.md](ARCHITECTURE.md), "Speed without changing results"):
segment files are mapped, so loading a table reads only their headers; a scan visits a table's rows in key order by
address, without sorting or materializing them, and decodes only the storage chunks and columns it reads (numeric and
string columns of a chunk are kept in contiguous typed form, built straight from the encoded chunk); a limit stops the
scan; filters over typed columns are evaluated a chunk at a time, other expressions through compiled trees with borrowed
values; aggregation, joins and orderings by a number read typed columns; similarities are scored four embeddings at a
time with the same sums as one at a time.

Where pixeltable-rs is not faster, or behaves differently:

- Pure-Python CPU-bound UDFs (with the GIL) run one call at a time, as upstream runs them (`py_cpu_loop`).
- A query with a limit and a filter stops evaluating the filter once it has enough rows, as SQL's LIMIT stops a scan:
  an error in a later row is not raised, and user-function calls of the filter are not made for later rows.
- Rows are held in memory as decoded values (about 72 bytes per cell plus each row's own allocations), and each column a
  scan reads gets a typed copy of about 9 bytes per value per storage chunk read; upstream keeps its data in Postgres
  pages. A table much larger than memory is not a design target.
- Similarity search is an exact scan, linear in the number of rows (100k rows: 4.9 ms); upstream's approximate HNSW
  index grows sublinearly, so on tables of many millions of rows it answers faster, though not exactly.
- The first scan of a column after a table is loaded parses its storage chunks (the `reload_first_*` cases); later scans
  read the cached typed columns.

## Concurrent writers

Several threads or processes may change the same tables, with the outcome of running the operations one after the
other, as upstream's table locks give: no update is lost, a deleted row stays deleted, a column added while rows are
inserted gets values for those rows too. In one process, operations on a table (and the views over it) wait for each
other, as in upstream. Across processes, an operation that computed rows with user code while another process changed
what it read is computed again (a builtin-only operation commits atomically); after 100 such conflicts it fails with
`pxt.ConcurrencyError` (SERIALIZATION_FAILURE), as upstream's retry limit does. An operation whose own UDFs keep
changing the table it computes cannot complete and fails with `pxt.ConcurrencyError` (CONCURRENT_MODIFICATION).

## Setup

The harness copies tests from, and links test data into, an upstream Pixeltable checkout one directory up, so clone
this repository as `pixeltable-rs/` inside a Pixeltable checkout:

```sh
git clone https://github.com/pixeltable/pixeltable.git
cd pixeltable
git clone <this repository> pixeltable-rs
cd pixeltable-rs
uv venv .venv --python 3.13
VIRTUAL_ENV=$PWD/.venv uv pip install maturin pytest pytest-xdist pytest-timeout numpy pillow pandas pydantic \
    cloudpickle av beautifulsoup4 more-itertools requests httpx aiohttp tenacity filelock pyarrow
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
```

Run Python from `pixeltable-rs/` or anywhere outside the Pixeltable checkout's root, so that `import pixeltable`
finds the clone and not the upstream package.

## Tests

```sh
cd compat
../.venv/bin/python sync_tests.py        # copy the selected upstream test files
../.venv/bin/python check_unmodified.py  # verify the copies are byte-identical to upstream
../.venv/bin/python run.py               # run them; --check fails on any regression from PASSING.txt
cd .. && .venv/bin/python -m pytest tests_rs
PYO3_PYTHON=$PWD/.venv/bin/python cargo test --lib
```

## License

Apache License 2.0 (see [LICENSE](LICENSE)). The files in `compat/tests/` other than `conftest.py` and `utils.py`, and
parts of `python/pixeltable/exceptions.py`, are copied from Pixeltable; see [NOTICE](NOTICE).
