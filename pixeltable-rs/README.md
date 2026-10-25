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
