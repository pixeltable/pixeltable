import json
import logging
import os
import pathlib
import shutil
import uuid
from typing import Callable, Iterator

import pytest
import requests
import sqlalchemy as sql
import tenacity
from _pytest.config import Config as PytestConfig, argparsing
from filelock import FileLock
from sqlalchemy import orm, text

import pixeltable as pxt
from pixeltable import exprs, functions as pxtf
from pixeltable.catalog import Catalog
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.metadata import SystemInfo, create_system_info
from pixeltable.metadata.schema import Dir, Function, PendingTableOp, Table, TableSchemaVersion, TableVersion
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.local_store import LocalStore, TempStore
from pixeltable.utils.sql import add_option_to_db_url

from .utils import (
    IN_CI,
    ReloadTester,
    create_all_datatypes_tbl,
    create_img_tbl,
    create_test_tbl,
    reload_catalog,
    skip_test_if_not_installed,
)

_logger = logging.getLogger('pixeltable')


DO_RERUN: bool


def pytest_addoption(parser: argparsing.Parser) -> None:
    parser.addoption('--no-rerun', action='store_true', default=False, help='Do not rerun any failed tests.')


def pytest_configure(config: PytestConfig) -> None:
    global DO_RERUN  # noqa: PLW0603
    DO_RERUN = not config.getoption('--no-rerun')


def pytest_runtest_setup(item: pytest.Item) -> None:
    current_test = os.environ.get('PYTEST_CURRENT_TEST')
    _logger.info(f'Running Pixeltable test: {current_test}')
    pxtf.huggingface._model_cache.clear()
    pxtf.huggingface._processor_cache.clear()


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

    Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)

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


@pytest.fixture(scope='function')
def reset_db(init_env: None, request: pytest.FixtureRequest) -> Iterator[None]:
    """Fixture for tests that interact with the database (PosgreSQL or CockroachDB).
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

    if 'corrupts_db' in request.keywords:
        _logger.info('Skipping DB validation due to corrupts_db marker.')
        return

    Env.get().user = None
    Catalog.get().validate_store()


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
                if repo.repo_id not in ('openai/clip-vit-base-patch32', 'intfloat/e5-large-v2')
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


def clean_db(restore_md_tables: bool = True) -> None:
    engine = Env.get().engine

    # Drop all tables from the DB, including data tables. Dropping the data tables is necessary for certain tests,
    # such as test_db_migration, that may lead to UUID collisions if interrupted.
    sql_md = orm.declarative_base().metadata
    sql_md.reflect(engine)
    sql_md.drop_all(bind=engine)

    # The following lines may be uncommented as a replacement for the above, if one wishes to drop only metadata
    # tables for testing purposes.
    # SystemInfo.__table__.drop(engine, checkfirst=True)
    # TableSchemaVersion.__table__.drop(engine, checkfirst=True)
    # TableVersion.__table__.drop(engine, checkfirst=True)
    # Table.__table__.drop(engine, checkfirst=True)
    # Function.__table__.drop(engine, checkfirst=True)
    # Dir.__table__.drop(engine, checkfirst=True)

    if restore_md_tables:
        # Restore metadata tables and system info
        Dir.__table__.create(engine)
        Function.__table__.create(engine)
        Table.__table__.create(engine)
        TableVersion.__table__.create(engine)
        TableSchemaVersion.__table__.create(engine)
        PendingTableOp.__table__.create(engine)
        SystemInfo.__table__.create(engine)
        create_system_info(engine)


@pytest.fixture(scope='function')
def test_tbl(reset_db: None) -> pxt.Table:
    return create_test_tbl()


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
def all_datatypes_tbl(reset_db: None) -> pxt.Table:
    return create_all_datatypes_tbl()


@pytest.fixture(scope='function')
def img_tbl(reset_db: None) -> pxt.Table:
    return create_img_tbl('test_img_tbl')


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
def small_img_tbl(reset_db: None) -> pxt.Table:
    return create_img_tbl('small_img_tbl', num_rows=40)


@pytest.fixture(scope='function')
def indexed_img_tbl(reset_db: None, clip_embed: pxt.Function) -> pxt.Table:
    skip_test_if_not_installed('transformers')
    t = create_img_tbl('indexed_img_tbl', num_rows=40)
    t.add_embedding_index('img', idx_name='img_idx0', metric='cosine', image_embed=clip_embed, string_embed=clip_embed)
    return t


@pytest.fixture(scope='function')
def multi_idx_img_tbl(reset_db: None, clip_embed: pxt.Function) -> pxt.Table:
    skip_test_if_not_installed('transformers')
    t = create_img_tbl('multi_idx_img_tbl', num_rows=4)
    t.add_embedding_index('img', idx_name='img_idx1', metric='cosine', image_embed=clip_embed, string_embed=clip_embed)
    t.add_embedding_index('img', idx_name='img_idx2', metric='cosine', image_embed=clip_embed, string_embed=clip_embed)
    return t


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
