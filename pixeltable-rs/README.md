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
