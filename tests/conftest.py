import json
import logging
import os
import pathlib
import platform
import shutil
import sys
import uuid
from typing import Callable, Iterator

import pytest
import requests
import sqlalchemy as sql
import tenacity
from _pytest.config import Config as PytestConfig, argparsing
from filelock import FileLock
from sqlalchemy import text

import pixeltable as pxt
import pixeltable.utils.fault_injection as prod_fault_injection
import tests.fault_injection as test_fault_injection
from pixeltable import exprs, functions as pxtf
from pixeltable.config import Config
from pixeltable.env import LOG_FMT_STR, Env
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.metadata.schema import base_metadata
from pixeltable.runtime import get_runtime, reset_runtime
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.local_store import LocalStore, TempStore
from pixeltable.utils.sql import add_option_to_db_url

from .utils import (
    IN_CI,
    ReloadTester,
    create_all_datatypes_tbl,
    create_img_tbl,
    create_test_tbl,
    local_embedding,
    reload_catalog,
)

_logger = logging.getLogger('pixeltable_test')


DO_RERUN: bool


def pytest_addoption(parser: argparsing.Parser) -> None:
    parser.addoption('--no-rerun', action='store_true', default=False, help='Do not rerun any failed tests.')


def pytest_configure(config: PytestConfig) -> None:
    global DO_RERUN  # noqa: PLW0603
    DO_RERUN = not config.getoption('--no-rerun')


# Fixtures that download a Hugging Face model. Any test requesting one of these is implicitly a
# Hugging Face test and is marked 'very_expensive'.
_HF_FIXTURES = frozenset({'clip_embed', 'e5_embed', 'all_mpnet_embed'})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _HF_FIXTURES.intersection(getattr(item, 'fixturenames', ())):
            item.add_marker(pytest.mark.very_expensive)


def pytest_runtest_setup(item: pytest.Item) -> None:
    current_test = os.environ.get('PYTEST_CURRENT_TEST')
    _logger.info(f'Running Pixeltable test: {current_test}')
    pxtf.huggingface._model_cache.clear()
    pxtf.huggingface._processor_cache.clear()
    pxtf.vllm._model_cache.clear()


def pytest_runtest_teardown(item: pytest.Item) -> None:
    if IN_CI:
        _free_disk_space()
    current_test = os.environ.get('PYTEST_CURRENT_TEST')
    _logger.info(f'Finished Pixeltable test: {current_test}')


def _set_up_external_db_schema(worker_id: int | str) -> str:
    schema_name = f'test_{worker_id}_{uuid.uuid4().hex[:16]}'
    original_connect_str = os.environ['PIXELTABLE_DB_CONNECT_STR']

    # Create schema first
    engine = sql.create_engine(original_connect_str)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
            conn.commit()
    finally:
        engine.dispose()

    # Modify connection string with search_path for THIS worker process
    # Each worker process has its own os.environ, so this is process-safe
    modified_url = add_option_to_db_url(original_connect_str, f'-c search_path={schema_name},public')
    os.environ['PIXELTABLE_DB_CONNECT_STR'] = modified_url.render_as_string(hide_password=False)
    _logger.info(f'Created schema and configured connection string with search_path: {schema_name}')
    return schema_name


@pytest.fixture(scope='session')
def init_env(tmp_path_factory: pytest.TempPathFactory, worker_id: int) -> None:  # type: ignore[misc]
    os.chdir(os.path.dirname(os.path.dirname(__file__)))  # Project root directory

    # Set the relevant env vars for the test db.
    # We use a single shared pgserver instance, running in the "true" home directory ($PIXELTABLE_HOME/pgdata).
    # Each worker gets its own test db in this instance, along with its own home directory for everything else
    # (file cache, media store, etc).
    shared_home = pathlib.Path(os.environ.get('PIXELTABLE_HOME', str(pathlib.Path.home() / '.pixeltable')))
    home_dir = str(tmp_path_factory.mktemp('base') / '.pixeltable')
    os.environ['PIXELTABLE_HOME'] = home_dir
    os.environ['PIXELTABLE_CONFIG'] = str(shared_home / 'config.toml')
    os.environ['PIXELTABLE_DB'] = f'test_{worker_id}'
    os.environ['PIXELTABLE_PGDATA'] = str(shared_home / 'pgdata')
    os.environ['PIXELTABLE_API_URL'] = 'https://preprod-internal-api.pixeltable.com'
    os.environ['FIFTYONE_DATABASE_DIR'] = f'{home_dir}/.fiftyone'
    reinit_db = True
    schema_name = None
    if os.environ.get('PIXELTABLE_DB_CONNECT_STR') is not None:
        print('Using external database connection for test configuration')
        reinit_db = False
        schema_name = _set_up_external_db_schema(worker_id)

    for var in (
        'PIXELTABLE_HOME',
        'PIXELTABLE_CONFIG',
        'PIXELTABLE_DB',
        'PIXELTABLE_PGDATA',
        'PIXELTABLE_API_URL',
        'FIFTYONE_DATABASE_DIR',
        'PIXELTABLE_DB_CONNECT_STR',
    ):
        print(f'{var:25} = {os.environ.get(var)}')

    # Ensure the shared home directory exists.
    shared_home.mkdir(parents=True, exist_ok=True)

    # Initialize Pixeltable. If using multiple workers, they need to be initialized synchronously to ensure we
    # don't have several processes trying to initialize pgserver in parallel.
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / 'pxt-init.lock')):
        # We need to call `Env._init_env()` with `reinit_db=True`. This is because if a previous test run was
        # interrupted (e.g., by an inopportune Ctrl-C), there may be residual DB artifacts that interfere with
        # initialization.
        Env._init_env(reinit_db=reinit_db)
        pxt.init()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(LOG_FMT_STR))
    pxt_logger = logging.getLogger('pixeltable')
    pxt_logger.setLevel(logging.DEBUG)
    pxt_logger.addHandler(stdout_handler)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    yield
    FileCache.get().validate()

    # Cleanup: Drop schema on fixture teardown
    if schema_name:
        try:
            db_connect_str = os.environ.get('PIXELTABLE_DB_CONNECT_STR')
            if db_connect_str:
                engine = sql.create_engine(db_connect_str)
                try:
                    with engine.connect() as conn:
                        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
                        conn.commit()
                    _logger.info(f'Dropped test schema: {schema_name}')
                finally:
                    engine.dispose()
        except Exception as e:
            _logger.warning(f'Failed to cleanup test schema {schema_name}: {e}')


@pytest.fixture()
def fault_injection() -> Iterator[None]:
    """Enables fault injection"""
    orig_process_fault = prod_fault_injection.process_fault
    orig_create_fault_manager = prod_fault_injection.create_fault_manager

    # Monkey patch fault injection to product
    prod_fault_injection.process_fault = test_fault_injection.process_fault
    prod_fault_injection.create_fault_manager = test_fault_injection.create_fault_manager

    # Recreate runtime to pick up a fault manager
    reset_runtime()

    try:
        yield
    finally:
        prod_fault_injection.process_fault = orig_process_fault
        prod_fault_injection.create_fault_manager = orig_create_fault_manager
        reset_runtime()


@pytest.fixture(scope='function')
def uses_db(init_env: None, request: pytest.FixtureRequest) -> Iterator[None]:
    """Fixture for tests that interact with the underlying store (PosgreSQL or CockroachDB).
    Cleans up the database before the test, and validates it after the test.
    """
    # Clean the DB *before* reloading. This is because some tests
    # (such as test_migration.py) may leave the DB in a broken state.
    clean_db()
    Config.init({}, reinit=True)
    Env.get().default_time_zone = None
    Env.get().user = None
    reload_catalog()
    FileCache.get().validate()
    FileCache.get().set_capacity(10 << 30)  # 10 GiB

    yield

    Env.get().user = None
    get_runtime().catalog.validate_store()


@pytest.fixture(scope='session')
def proxy_daemon_db(init_env: None, worker_id: str) -> Iterator[str]:
    """A per-worker local proxy daemon, started once for the session and reused across tests.

    The db name is worker-scoped so parallel xdist workers don't share a catalog. start() is idempotent,
    so the per-test make_catalog_path fixture only resets the daemon's catalog rather than restarting the process.
    """
    # the proxy daemon serves over HTTP via fastapi/uvicorn (the serve extra); a minimal install omits them
    pytest.importorskip('fastapi')
    pytest.importorskip('uvicorn')
    from pixeltable.service import proxy_daemon

    db = f'testdb_{worker_id}'
    proxy_daemon.start(db)
    try:
        yield db
    finally:
        proxy_daemon.delete(db)


@pytest.fixture(scope='function', params=['local', 'proxy'])
def make_catalog_path(init_env: None, request: pytest.FixtureRequest) -> Iterator[Callable[[str], str]]:
    """Parameterized variant of uses_db: runs a test against both the in-process catalog and a delegated
    (proxied) catalog served by a local daemon.

    Yields a path-builder mapping a bare path to the active catalog: the identity for local, and the bare
    path prefixed with the daemon's pxt:// uri for proxy (with an empty path mapping to the catalog root).
    """
    clean_db()
    Config.init({}, reinit=True)
    Env.get().default_time_zone = None
    Env.get().user = None
    reload_catalog()
    FileCache.get().validate()
    FileCache.get().set_capacity(10 << 30)  # 10 GiB

    if request.param == 'proxy':
        from pixeltable.service import proxy_daemon

        db = request.getfixturevalue('proxy_daemon_db')
        proxy_daemon.reset(db)
        prefix = f'pxt://local:{db}'

        def p(path: str) -> str:
            return f'{prefix}/{path}' if path else prefix
    else:

        def p(path: str) -> str:
            return path

    yield p

    Env.get().user = None
    get_runtime().catalog.validate_store()


@pytest.fixture(scope='function')
def make_local_path(init_env: None) -> Iterator[Callable[[str], str]]:
    """Stand-in for make_catalog_path() for tests that fail in proxy mode."""
    clean_db()
    Config.init({}, reinit=True)
    Env.get().default_time_zone = None
    Env.get().user = None
    reload_catalog()
    FileCache.get().validate()
    FileCache.get().set_capacity(10 << 30)  # 10 GiB

    def p(path: str) -> str:
        return path

    yield p

    Env.get().user = None
    get_runtime().catalog.validate_store()


def _free_disk_space() -> None:
    assert IN_CI

    # In CI, we sometimes run into disk space issues. We try to mitigate this by clearing out various caches between
    # tests.

    # Clear the temp store and media dir
    try:
        TempStore.clear()
        LocalStore(Env.get().media_dir).clear()
        _logger.info('Cleared TempStore and media dir.')
    except PermissionError:
        # Sometimes this happens on Windows if a file is held open by a concurrent process.
        _logger.info('PermissionError trying to clear TempStore and media dir.')

    try:
        _clear_hf_caches()
    except ImportError:
        pass  # huggingface_hub not installed in this CI environment


def _clear_hf_caches() -> None:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HOME, HUGGINGFACE_HUB_CACHE

    assert IN_CI

    if pathlib.Path(HUGGINGFACE_HUB_CACHE).exists():
        try:
            # Scan the cache directory for all revisions of all models
            cache_info = scan_cache_dir()
            revisions_to_delete = [
                revision.commit_hash
                for repo in cache_info.repos
                # Keep around models that are used by multiple tests
                if repo.repo_id
                not in (
                    'openai/clip-vit-base-patch32',
                    'intfloat/e5-large-v2',
                    'sentence-transformers/all-mpnet-base-v2',
                )
                for revision in repo.revisions
            ]
            cache_info.delete_revisions(*revisions_to_delete).execute()
            _logger.info(f'Deleted {len(revisions_to_delete)} revision(s) from huggingface hub cache directory.')
        except (OSError, PermissionError) as exc:
            _logger.info(
                f'{type(exc).__name__} trying to clear huggingface hub cache directory: {HUGGINGFACE_HUB_CACHE}'
            )

    huggingface_xet_cache = pathlib.Path(HF_HOME) / 'xet'
    if huggingface_xet_cache.exists():
        try:
            shutil.rmtree(huggingface_xet_cache)
            huggingface_xet_cache.mkdir()
            _logger.info(f'Deleted xet cache directory: {huggingface_xet_cache}')
        except (OSError, PermissionError) as exc:
            _logger.info(
                f'{type(exc).__name__} trying to clear huggingface xet cache directory: {huggingface_xet_cache}'
            )


# Get all known metadata table names and cache them
_MD_TABLE_NAMES = set(base_metadata.tables.keys())


def clean_db(drop_md_tables: bool = False) -> None:
    engine = Env.get().engine
    inspector = sql.inspect(engine)
    all_table_names = set(inspector.get_table_names())
    data_table_names = all_table_names - _MD_TABLE_NAMES
    existing_md_names = all_table_names & _MD_TABLE_NAMES

    with engine.connect() as conn:
        # Drop data tables
        if data_table_names:
            table_names = ', '.join(f'"{t}"' for t in data_table_names)
            conn.execute(text(f'DROP TABLE IF EXISTS {table_names} CASCADE'))

        if existing_md_names:
            table_names = ', '.join(f'"{t}"' for t in existing_md_names)
            if drop_md_tables:
                # Drop existing metadata tables
                conn.execute(text(f'DROP TABLE IF EXISTS {table_names} CASCADE'))
            elif Env.get().is_using_cockroachdb:
                # CockroachDB sometimes rejects TRUNCATE when other in-flight statements are
                # dropping indexes on the same table; use DELETEs instead, in reverse FK
                # dependency order (children before parents). dirs' self-referential FK is fine
                # because FK constraints are checked at the end of each DELETE statement.
                for tbl in reversed(base_metadata.sorted_tables):
                    if tbl.name in existing_md_names:
                        conn.execute(tbl.delete())
            else:
                # Truncate existing metadata tables
                conn.execute(text(f'TRUNCATE TABLE {table_names} CASCADE'))
        conn.commit()


@pytest.fixture(scope='function')
def test_tbl(uses_db: None) -> pxt.Table:
    return create_test_tbl()


@pytest.fixture(scope='function')
def test_tbl_dual(make_catalog_path: Callable[[str], str]) -> pxt.Table:
    return create_test_tbl(make_catalog_path('test_tbl'))


@pytest.fixture(scope='function')
def reload_tester(init_env: None) -> ReloadTester:
    return ReloadTester()


@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: pxt.Table) -> list[exprs.Expr]:
    t = test_tbl
    return [
        t.c1,
        t.c7['*'].f1,
        exprs.Literal('test'),
        exprs.InlineDict(
            {'a': t.c1, 'b': t.c6.f1, 'c': 17, 'd': exprs.InlineDict({'e': t.c2}), 'f': exprs.InlineList([t.c3, t.c3])}
        ),
        exprs.InlineList([[t.c2, t.c2], [t.c2, t.c2]]),
        t.c2 > 5,
        t.c2 == None,
        ~(t.c2 > 5),
        (t.c2 > 5) & (t.c1 == 'test'),
        (t.c2 > 5) | (t.c1 == 'test'),
        pxtf.map(t.c7['*'].f5, lambda x: [x[3], x[2], x[1], x[0]]),
        t.c8[0, 1:],
        t.c2.isin([1, 2, 3]),
        t.c2.isin(t.c6.f5),
        t.c2.astype(pxt.Float),
        (t.c2 + 1).astype(pxt.Float),
        t.c2.apply(str),
        (t.c2 + 1).apply(str),
        t.c3.apply(str),
        t.c4.apply(str),
        t.c5.apply(str),
        t.c6.apply(str),
        t.c1.apply(json.loads),
        t.c8.errortype,
        t.c8.errormsg,
        pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3),
    ]


@pytest.fixture(scope='function')
def all_datatypes_tbl(uses_db: None) -> pxt.Table:
    return create_all_datatypes_tbl()


@pytest.fixture(scope='function')
def all_datatypes_tbl_dual(make_catalog_path: Callable[[str], str]) -> pxt.Table:
    return create_all_datatypes_tbl(name=make_catalog_path('all_datatype_tbl'))


@pytest.fixture(scope='function')
def img_tbl(uses_db: None) -> pxt.Table:
    return create_img_tbl('test_img_tbl')


@pytest.fixture(scope='function')
def img_tbl_dual(make_catalog_path: Callable[[str], str]) -> pxt.Table:
    return create_img_tbl(make_catalog_path('test_img_tbl'))


@pytest.fixture(scope='function')
def img_tbl_exprs(indexed_img_tbl: pxt.Table) -> list[exprs.Expr]:
    t = indexed_img_tbl
    return [
        t.img.width,
        t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        t.img.rotate(90).resize([224, 224]),
        t.img.fileurl,
        t.img.localpath,
        t.img.similarity(string='red truck', idx='img_idx0'),
    ]


@pytest.fixture(scope='function')
def multi_img_tbl_exprs(multi_idx_img_tbl: pxt.Table) -> list[exprs.Expr]:
    t = multi_idx_img_tbl
    return [t.img.similarity(string='red truck', idx='img_idx1'), t.img.similarity(string='red truck', idx='img_idx2')]


@pytest.fixture(scope='function')
def small_img_tbl(uses_db: None) -> pxt.Table:
    return create_img_tbl('small_img_tbl', num_rows=40)


@pytest.fixture(scope='function')
def small_img_tbl_dual(make_catalog_path: Callable[[str], str]) -> pxt.Table:
    return create_img_tbl(make_catalog_path('small_img_tbl'), num_rows=40)


@pytest.fixture(scope='function')
def indexed_img_tbl(uses_db: None, local_embed: pxt.Function) -> pxt.Table:
    t = create_img_tbl('indexed_img_tbl', num_rows=40)
    t.add_embedding_index(
        'img', idx_name='img_idx0', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


@pytest.fixture(scope='function')
def indexed_img_tbl_dual(make_catalog_path: Callable[[str], str], local_embed: pxt.Function) -> pxt.Table:
    t = create_img_tbl(make_catalog_path('indexed_img_tbl'), num_rows=40)
    t.add_embedding_index(
        'img', idx_name='img_idx0', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


@pytest.fixture(scope='function')
def multi_idx_img_tbl(uses_db: None, local_embed: pxt.Function) -> pxt.Table:
    t = create_img_tbl('multi_idx_img_tbl', num_rows=4)
    t.add_embedding_index(
        'img', idx_name='img_idx1', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    t.add_embedding_index(
        'img', idx_name='img_idx2', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


@pytest.fixture(scope='function')
def multi_idx_img_tbl_dual(make_catalog_path: Callable[[str], str], local_embed: pxt.Function) -> pxt.Table:
    t = create_img_tbl(make_catalog_path('multi_idx_img_tbl'), num_rows=4)
    t.add_embedding_index(
        'img', idx_name='img_idx1', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    t.add_embedding_index(
        'img', idx_name='img_idx2', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


# Deterministic, download-free embedding (see tests.utils.local_embedding). Same shape as clip. Use this instead
# of HF models fixtures, unless you intend to test the HF models specifically.
@pytest.fixture
def local_embed() -> pxt.Function:
    return local_embedding.using(dim=512)


# Parametrized embedding selectors for tests that assert real-model semantics:
# - the 'local_embed' param runs the full test body against the download-free embedding
# - the real-model param is marked 'very_expensive'
# Each fixture returns (embedding, is_dummy_model); skip model-dependent assertions when is_dummy_model.
@pytest.fixture(
    params=[
        pytest.param(False, id='local_embed'),
        pytest.param(True, id='clip_embed', marks=pytest.mark.very_expensive),
    ]
)
def clip_or_local(request: pytest.FixtureRequest) -> tuple[pxt.Function, bool]:
    if request.param:
        return request.getfixturevalue('clip_embed'), False
    return local_embedding.using(dim=512), True


@pytest.fixture(
    params=[
        pytest.param(False, id='local_embed'),
        pytest.param(True, id='mpnet_embed', marks=pytest.mark.very_expensive),
    ]
)
def mpnet_or_local(request: pytest.FixtureRequest) -> tuple[pxt.Function, bool]:
    if request.param:
        return request.getfixturevalue('all_mpnet_embed'), False
    return local_embedding.using(dim=512), True


# Fixtures for various reusable Huggingface models. We build in retries to guard against network issues.


def _retry_hf(fn: Callable) -> Callable:
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
        retry=tenacity.retry_if_exception_type(requests.ReadTimeout),
        reraise=True,
    )(fn)


@pytest.fixture(scope='session')
@_retry_hf
def clip_embed() -> pxt.Function:
    try:
        return clip.using(model_id='openai/clip-vit-base-patch32')
    except ImportError:
        return None  # Any time this happens, the test wil be skipped anyway.


@pytest.fixture(scope='session')
@_retry_hf
def e5_embed() -> pxt.Function:
    if IN_CI and platform.system() == 'Windows':
        pytest.skip('`sentence-transformers` crashes on Windows CI (memory pressure?)')
    try:
        return sentence_transformer.using(model_id='intfloat/e5-large-v2')
    except ImportError:
        return None


@pytest.fixture(scope='session')
@_retry_hf
def all_mpnet_embed() -> pxt.Function:
    try:
        return sentence_transformer.using(model_id='all-mpnet-base-v2')
    except ImportError:
        return None
