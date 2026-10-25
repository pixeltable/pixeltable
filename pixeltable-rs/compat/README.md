# pixeltable-rs compatibility harness

Runs upstream Pixeltable test files, copied byte for byte from `../../tests/`, against pixeltable-rs and reports
which tests pass. Only `tests/conftest.py` and `tests/utils.py` are replaced; they use the public API plus the two
test hooks in `pixeltable._testing`.

## Commands

All commands run from `pixeltable-rs/compat/` with any Python 3.11+ (only `run.py` needs the venv, and it finds it).

| Command | What it does |
|---|---|
| `python sync_tests.py [--prune]` | Copy the selected upstream test files and the `tests/` modules they import into `tests/`, write `manifest.txt` (sha256 per file), create the data symlinks. |
| `python check_unmodified.py` | Exit 1 if a copied file differs from upstream, the manifest is stale, or `tests/` holds a `.py` file that is neither a copy nor `conftest.py`/`utils.py`. |
| `python run.py [files] [-k EXPR] [-n N]` | Run the selected files (default all, 4 xdist workers), write `results/latest.json`, print a table per file. |
| `python run.py --update-passing` | Also rewrite `PASSING.txt` (sorted node ids of passing tests). |
| `python run.py --check` | Exit 1 if any test listed in `PASSING.txt` no longer passes (the regression gate). |
| `python classify.py [--explain]` | Write `public_api_tests.txt`, the public-API target set (see below). |

`run.py` options: files may be given as `test_dirs`, `dirs` or `test_dirs.py`; `-n 0` disables xdist; `--timeout S`
overrides the per-test timeout (120 s); `--stream` shows pytest's output instead of writing it to
`results/latest.pytest.log`; `--no-run` re-summarizes existing results; extra pytest arguments go after `--`
(`python run.py dirs -- -x`); `--python` picks another interpreter (used for the upstream baseline); `--passing`
uses another list than `PASSING.txt`.

With a subset of files, `--update-passing` and `--check` only touch entries of those files; with `-k`, `-m` or extra
pytest arguments, only entries of tests that actually ran. Exit status without `--check` is 0 unless pytest itself
failed (usage error, conftest import error, no tests collected).

Concurrent runs are isolated from each other (each pytest process gets its own `PIXELTABLE_HOME`), but they share
`results/latest.json` and its log; pass `--results results/<name>.json` for a run whose results must not be
overwritten.

`results/latest.json` holds, per node id: `outcome` (passed, failed, error, skipped, xfailed, xpassed), `reason`
(first line of the failure, at most 300 characters), `category` (a coarse cause: `internal-api` for a test that hit a
stub, `utils-definition`, `testing-hooks`, `timeout`, `crash`, `harness-helper`, `skip`, or `fail`) and `duration`;
plus collection errors and which modules were stubbed. Under xdist the per-test timeout uses pytest-timeout's thread
method, which stops a test stuck in native code by killing its worker; such a test shows up as `crash`.

## Selection

`pxt_compat/selection.py` lists the selected upstream test modules. `sync_tests.py` adds every module under `tests/`
that they import (`from .module_with_future_annotations import ...`, `import tests.test_function`) or run by path
(`tests/script_with_udf.py`), transitively, plus `tests/__init__.py`. Test data is not copied: `tests/data` and
`docs/resources` are symlinks into the upstream checkout, so `TESTS_DIR / 'data'`, relative paths like
`tests/data/images/...` (the tests run with `compat/` as the working directory, as upstream runs with the repository
root) and the `sample_file_server` fixture (which serves `compat/`) all resolve.

## How a run is isolated

- `compat/pytest.ini` makes `compat/` the rootdir, so upstream's `pyproject.toml` pytest settings are never read. It
  mirrors upstream's markers, the default `-m "not remote_api and not expensive and not very_expensive and not
  benchmark and not otel"` filter, `--strict-markers` and the filterwarnings entries; the two entries that name
  Pixeltable warning classes are added by the conftest, so that a missing class cannot keep pytest from starting.
- `run.py` runs pytest with `compat/` as the working directory and strips the upstream checkout from `PYTHONPATH`.
  The conftest asserts that `pixeltable.__file__` is in the interpreter's site-packages or in `pixeltable-rs/python`,
  never under the upstream `pixeltable/`, and that the `tests` package is `compat/tests`. Test modules are imported as
  `tests.test_x`, as upstream (some tests assert on names like `tests.test_function.evolving_udf`).
- Before `pixeltable` is imported, the conftest points `PIXELTABLE_HOME` (and `PIXELTABLE_CONFIG`) at a fresh
  temporary directory per process (per xdist worker), removed at the end (`PXT_COMPAT_KEEP_HOME=1` keeps it;
  `PXT_COMPAT_TMPDIR` picks the parent directory), removes `PIXELTABLE_API_KEY`, `PIXELTABLE_API_URL`,
  `PIXELTABLE_CLOUD_HOST`, `PIXELTABLE_DB*`, `PIXELTABLE_PGDATA` and the media-destination variables, and puts the
  interpreter's `bin/` first on `PATH` (a test runs `python tests/script_with_udf.py`).

## Test hooks (the contract with pixeltable-rs)

- `pixeltable._testing.reset_db()`: drop every table, directory and in-memory state; the next operation starts with an
  empty catalog. Called before every test that uses `uses_db` or `db_root` (directly or through `test_tbl` and the
  other table fixtures), and once by `init_env`. Upstream also resets the default time zone and current user here.
- `pixeltable._testing.reload_catalog()`: drop all in-memory catalog state; the next call reloads it from disk. Used by
  `utils.reload_catalog()` and `ReloadTester.run_reload_test()`.
- `pixeltable._testing.validate_store()`, called after each such test if present (upstream validates its store there):
  pixeltable-rs checks the store's files against the manifests, the catalog reloaded from disk against the one in
  memory, row version ranges and index consistency.

## Stubs for internal modules

Upstream test modules import internals at the top (`from pixeltable.env import Env`, `from pixeltable import exprs,
catalog, func`, `from pixeltable.func import Batch`, `from pixeltable.index.btree import BtreeIndex`, ...).
`pxt_compat/stubs.py`, installed by the conftest before anything imports pixeltable, keeps such a file importable so
that only the tests that use an internal fail:

1. It scans the copied test files for `pixeltable.*` imports. A name in `from pixeltable.x import name` counts as a
   module if the upstream source tree has `pixeltable/x/name.py` or `pixeltable/x/name/` (read from disk, not
   imported).
2. A `sys.meta_path` finder appended after all real finders supplies a stub module for an allow-listed module that no
   real finder can locate. A module pixeltable-rs provides is therefore never shadowed, and a name that exists as an
   attribute of its parent (`from pixeltable import udf`) is never turned into a module.
3. For `from pixeltable.x import Name` at the top of a test file where `pixeltable.x` exists but lacks `Name`, the
   module gets a PEP 562 `__getattr__` that answers exactly those names with a placeholder; real attributes are
   untouched and other missing names still raise `AttributeError`.
4. A placeholder is a class: it can be subclassed (`class CustomLegacyIterator(ComponentIterator)`), subscripted
   (`Batch[int]`), used in annotations and in `X | None`, and its attributes are placeholders. Calling or
   instantiating it, or using it in `isinstance`/`issubclass`, raises `InternalAPIUnavailable("internal API
   `pixeltable.env.Env.get` is not available in pixeltable-rs (stubbed by the compat harness)")`, which derives from
   `BaseException` so that neither `except Exception` in the package nor `pytest.raises(Exception)` swallows it.
   Private attributes of placeholders raise `AttributeError` (with the same message), to keep introspection away.

The pytest header and `results/latest.json` (`meta.stubs`) list what was stubbed. Every allow-listed module is
resolved when the conftest loads, so the list is the same in every process.

pixeltable-rs must still provide whatever a test file evaluates at import time beyond names: for instance
`@pxt.udf(batch_size=20) def f(vals: Batch[int]) -> Batch[int]` (test_table, test_view, test_function) needs a real
`pixeltable.func.Batch`, since a placeholder in the signature is not a type `pxt.udf` can accept, and a failing
module-level definition fails the whole file at collection.

## Where conftest.py and utils.py differ from upstream

Same fixture and helper names and behavior, except:

- Only the `local` database root exists. `pytest_generate_tests` keeps upstream's `is_data_versioned` axis exactly and
  its `db_roots` marker validation, but never parametrizes `db_root` (upstream leaves the node id alone when `local` is
  the only target, so node ids match upstream's local-only runs); a test whose marker excluded `local` would get an
  empty parameter set. No proxy daemon or hosted-database fixtures; `cloud_env_configured()` is always False.
- `init_env`/`uses_db`/`db_root` use `reset_db()` (upstream: `Env._init_env`, `clean_db()`, `Config`/`Env`/`FileCache`
  resets, `reload_catalog()`), and `validate_store()` (upstream: `catalog.validate_store()`).
- The session teardown asserts, as upstream's does, that no event loop or HTTP client is left open; upstream first
  retires its runtime (`reset_runtime()`), the harness calls the `reload_catalog()` hook, which in pixeltable-rs also
  closes the idle event loop of async UDFs.
- `test_tbl_exprs` builds `exprs.Literal`/`InlineDict`/`InlineList` as upstream does; without `pixeltable.exprs` it
  fails with the stub error (its three users are outside the target set anyway).
- `ReloadTester.run_reload_test()` calls `pxt.Query.from_dict(query.as_dict())` directly; upstream wraps it in
  `catalog.retry_loop()` (a transaction with retries).
- `assert_resultset_eq()`/`assert_columns_eq()` take column types from the public `ResultSet.schema` (type strings
  like `'Array[(10,), float32] | None'`) instead of the internal `ResultSet._schema`; types match when equal aside from
  ` | None` (upstream's `ColumnType.matches()`), and the comparer is chosen by the type's kind (Float, Array, Json,
  Image, Video, Audio, Document). `assert_columns_eq()` also accepts a `ColumnType` (its repr is used).
- `create_table_data()` reads column types from `get_metadata()['columns'][name]['type_']` instead of the internal
  `ColumnRef.col_type`. `make_tbl()` and `create_scalars_tbl()` use the public type hints (`pxt.String`,
  `pxt.Int | None`, ...) that normalize to the `ts.*Type(nullable=...)` instances upstream passes.
- `pxt_raises()` uses `pixeltable.exceptions._error_class(code)` (the code-to-class map that PLAN.md makes part of the
  contract) and falls back to `pxt.Error` plus the code check if it is missing.
- `skip_test_if_not_installed()` asks `importlib.util.find_spec()` (upstream: `Env.is_installed_package()`).
- `check_media_store_count()` counts the files under `$PIXELTABLE_HOME/media` that have the table id (from
  `get_metadata()['id']`, remembered per handle, since tests count after dropping the table) in their path, as 32 hex
  digits or hyphenated, in a directory name or as a file name prefix. That fits upstream's layout
  (`media/<id hex>/.../<id hex>_<column>_<version>_<unique>.<ext>`) and pixeltable-rs's (`media/<db>/<table id>/...`).
  Its `tbl_version` filter, which no selected test uses, needs upstream's file naming.
- `get_multimedia_commons_video_uris()` lists the public bucket with anonymous boto3 and skips without boto3 or
  network (upstream: internal `ObjectOps`).
- `assert_type_eq()` keeps `ts.ColumnType.normalize_type()`: `pixeltable.type_system` is public in pixeltable-rs.
- `capture_console_output()` captures every record of the `pixeltable` logger (upstream filters through its internal
  console handler); `generate_test_video()` uses ffmpeg's default encoder (upstream: `Env.default_video_encoder`).
  Neither is used by the selected files.
- Module-level UDFs and iterators in `utils.py` (`sleep`, `error`, `stock_price`, `DummyIterator`, `DummyIterator2`,
  `dummy_embedding`, `local_embedding`) and `ALL_DATATYPES_SCHEMA` are defined inside `try` blocks at module level (so
  their names stay `tests.utils.<name>`); if pixeltable-rs cannot define one yet, the name raises
  `UtilsDefinitionError` when used instead of breaking every test file, and the pytest header says so.
- Not available, because upstream implements them only with internals or infrastructure the harness lacks (none is
  used by the selected files; each raises `NotImplementedError` naming the reason): `list_store_indexes`,
  `get_temp_store_count`, `check_temp_store_count`, `reload_env`, `skip_test_if_no_client`, `skip_test_if_no_config`,
  `skip_test_if_no_pxt_credentials`, `home_bucket_uri`, `iceberg_catalog`, `aux_db_name`. Upstream's proxy/cloud
  helpers (`fetch_presigned`, `assert_video_bytes`, `assert_audio_bytes`, `_proxy_temp_store_count`) are omitted.
- `rerun()`/`rerun_on_network_error()` produce pytest-rerunfailures' `flaky` marker as upstream; the marker is
  registered in `pytest.ini` and has no effect unless that plugin is installed.

## Public-API target set

`classify.py` statically marks a test function as public-API-only unless it, or a helper it reaches within its module
(module functions and classes it names, methods it calls through `self`/the class), uses a name imported from an
internal module, reaches one through the package (`pxt.exec`, `pxt.func`, ...), accesses a private attribute
(`t._id`; `self._x`, dunders, `_repr_html_` and the module's own names excepted), accesses an attribute only internal
objects have (`col_type`, `signature`, `signatures`, `serialize`, `subexprs`, `py_fn`, `arity`, ...), or requests
`test_tbl_exprs`/`img_tbl_exprs`/`multi_img_tbl_exprs`. Tests deselected by default (expensive, very_expensive,
remote_api, benchmark, otel, skip, Hugging Face fixtures) are excluded too. Public means: the package's exported names,
`pixeltable.functions.*`, `io`, `iterators` (not `iterators.base`), `exceptions`, `types`, `type_system` (public in
pixeltable-rs per PLAN.md) and `pixeltable.func.Batch`. Annotations are ignored. The attribute list is a heuristic,
limited to names that are unambiguous in the selected files.

Current size: 310 of 383 test functions (70 use internals, 3 are deselected). `run.py` reports, per file, how many
collected test items (parametrizations counted separately) of the target set pass.

## Upstream baseline

`upstream_baseline/_testing.py` implements the two hooks with upstream internals (it mirrors upstream conftest's reset
and reload, and wraps `Query.from_dict()` in upstream's `retry_loop()` when called outside a transaction, as upstream's
own `ReloadTester` does). Running the harness against upstream Pixeltable separates harness problems from
pixeltable-rs gaps:

```sh
uv venv --python 3.13 /tmp/pxt-upstream
VIRTUAL_ENV=/tmp/pxt-upstream uv pip install /path/to/pixeltable pytest pytest-xdist pytest-timeout imagehash
cp upstream_baseline/_testing.py /tmp/pxt-upstream/lib/python3.13/site-packages/pixeltable/_testing.py
python run.py --python /tmp/pxt-upstream/bin/python --results results/upstream.json -- --no-rerun
```

Last baseline (2026-10-24, upstream at the branch's base commit 0cef701, 4 workers, about 6 minutes): 475 items,
453 passed, 12 failed, 4 errors, 6 skipped; public-API target set 384 of 395 items passed. Every non-passing test was
environmental or fails the same way under upstream's own conftest:
- 9 need pgvector, which the installed pixeltable-pgserver build lacked (`could not access file "$libdir/vector"`):
  test_exprs `test_ids`, `test_print`, `test_serialization`, `test_similarity`; test_function
  `test_udf_import_error`; test_table `test_computed_col_exceptions` (both variants), `test_repr`,
  `test_table_metadata`;
- 5 need S3 or external URLs: test_table `test_create_s3_image_table`, `test_create_video_table`, `test_video_url`,
  `test_video_urls`, `test_validate_external_url`;
- test_pixeltable `test_version`: a source install without git tags reports version 0.0.1;
- test_exprs `test_errored_rows_released[raise_rate_limited]` also fails under upstream's own conftest;
- skipped: torch, pycocotools and markitdown are not installed; `test_same_base_join` is marked skip upstream.

## Limitations

- Static stubs cover only names imported at the top of test files; an internal reached another way (e.g.
  `pxt.exec.CachePrefetchNode` inside a test) fails with a plain `AttributeError`, still per test.
- The public-API classification is static and heuristic (attribute names); it can miss internal use through a
  dynamically computed name, or exclude a test whose column happens to be called like an internal attribute.
- `check_media_store_count()` relies on a table's media files living under `$PIXELTABLE_HOME/media` with the table id in
  their path.
- Tests that need network access (S3, external URLs) or optional packages (torch, transformers, pycocotools,
  markitdown) fail or skip as they would upstream in the same environment.
