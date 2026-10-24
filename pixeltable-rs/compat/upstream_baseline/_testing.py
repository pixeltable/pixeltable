"""pixeltable._testing for UPSTREAM Pixeltable: the harness's two hooks, implemented with upstream internals.

Not part of pixeltable-rs. Installed into a separate environment that has the upstream package, it lets the harness run
the copied tests against the real library, which separates harness problems from pixeltable-rs gaps (see
compat/README.md, "Upstream baseline"). It mirrors what upstream tests/conftest.py does around each test.
"""

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.metadata.schema import base_metadata
from pixeltable.runtime import get_runtime, reset_runtime
from pixeltable.utils.filecache import FileCache

_initialized = False


def _init() -> None:
    global _initialized  # noqa: PLW0603
    if _initialized:
        return
    Config.init(reinit=True, project_root=Config.get().project_root)
    Env._init_env(reinit_db=True)
    pxt.init()
    _initialized = True


def _quoted(names: set[str]) -> str:
    return ', '.join(f'"{name}"' for name in sorted(names))


def _clean_db() -> None:
    engine = Env.get().engine
    md_names = set(base_metadata.tables.keys())
    all_names = set(sql.inspect(engine).get_table_names())
    data_names = all_names - md_names
    existing_md_names = all_names & md_names
    with engine.connect() as conn:
        if data_names:
            conn.execute(sql.text(f'DROP TABLE IF EXISTS {_quoted(data_names)} CASCADE'))
        if existing_md_names:
            conn.execute(sql.text(f'TRUNCATE TABLE {_quoted(existing_md_names)} CASCADE'))
        conn.commit()


def reset_db() -> None:
    """Drop everything; start with an empty catalog (upstream conftest's _reset_catalog_state())."""
    _init()
    _clean_db()
    Config.init(reinit=True, project_root=Config.get().project_root)
    Env.get().default_time_zone = None
    Env.get().user = None
    reload_catalog()
    FileCache.get().validate()
    FileCache.get().set_capacity(10 << 30)
    FileCache.get().set_lease_seconds(Env.get()._file_cache_lease_s)


def reload_catalog() -> None:
    """Drop all in-memory catalog state; the next call reloads it from the store (upstream utils.reload_catalog())."""
    _init()
    reset_runtime()
    pxt.init()


def validate_store() -> None:
    """Upstream conftest's _validate_catalog_state()."""
    Env.get().user = None
    get_runtime().catalog.validate_store()


def _wrap_query_from_dict() -> None:
    """Upstream requires Query.from_dict() to run in a transaction; its own ReloadTester wraps the call in
    catalog.retry_loop(). The harness's ReloadTester calls the public pxt.Query.from_dict() directly, which pixeltable-rs
    must support, so for upstream the call is wrapped here when it is not already inside a transaction."""
    from pixeltable.catalog import retry_loop

    original = pxt.Query.from_dict.__func__  # type: ignore[attr-defined]

    def from_dict(cls: type[pxt.Query], d: dict) -> pxt.Query:
        if get_runtime().in_xact:
            return original(cls, d)
        return retry_loop()(lambda: original(cls, d))()

    pxt.Query.from_dict = classmethod(from_dict)  # type: ignore[assignment,method-assign]


_wrap_query_from_dict()
