import json
import logging
import os
import pathlib
from typing import Callable, Iterator

import pytest
import requests
import tenacity
from filelock import FileLock
from sqlalchemy import orm

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import catalog, exprs, func
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.metadata import SystemInfo, create_system_info
from pixeltable.metadata.schema import Dir, Function, Table, TableSchemaVersion, TableVersion
from pixeltable.utils.filecache import FileCache

from .utils import (
    ReloadTester,
    create_all_datatypes_tbl,
    create_img_tbl,
    create_test_tbl,
    reload_catalog,
    skip_test_if_not_installed,
)

_logger = logging.getLogger('pixeltable')


@pytest.fixture(autouse=True)
def pxt_test_harness() -> Iterator[None]:
    current_test = os.environ.get('PYTEST_CURRENT_TEST')
    _logger.info(f'Running Pixeltable test: {current_test}')
    yield
    _logger.info(f'Finished Pixeltable test: {current_test}')


@pytest.fixture(scope='session')
def init_env(tmp_path_factory: pytest.TempPathFactory, worker_id: int) -> None:
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

    for var in ('PIXELTABLE_HOME', 'PIXELTABLE_CONFIG', 'PIXELTABLE_DB', 'PIXELTABLE_PGDATA'):
        print(f'{var:17} = {os.environ[var]}')

    # Ensure the shared home directory exists.
    shared_home.mkdir(parents=True, exist_ok=True)

    # Initialize Pixeltable. If using multiple workers, they need to be initialized synchronously to ensure we
    # don't have several processes trying to initialize pgserver in parallel.
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / 'pxt-init.lock')):
        # We need to call `Env._init_env()` with `reinit_db=True`. This is because if a previous test run was
        # interrupted (e.g., by an inopportune Ctrl-C), there may be residual DB artifacts that interfere with
        # initialization.
        Env._init_env(reinit_db=True)
        pxt.init()

    Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)


@pytest.fixture(scope='function')
def reset_db(init_env: None) -> None:
    from pixeltable.env import Env

    # Clean the DB *before* reloading. This is because some tests
    # (such as test_migration.py) may leave the DB in a broken state.
    clean_db()
    Env.get().default_time_zone = None
    Env.get().user = None
    # It'd be best to clear the tmp dir between tests, but this fails on Windows for unclear reasons.
    # Env.get().clear_tmp_dir()
    reload_catalog()
    FileCache.get().set_capacity(10 << 30)  # 10 GiB


def clean_db(restore_md_tables: bool = True) -> None:
    from pixeltable.env import Env

    # Drop all tables from the DB, including data tables. Dropping the data tables is necessary for certain tests,
    # such as test_db_migration, that may lead to UUID collisions if interrupted.
    engine = Env.get().engine
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
        SystemInfo.__table__.create(engine)
        create_system_info(engine)


@pytest.fixture(scope='function')
def test_tbl(reset_db: None) -> catalog.Table:
    return create_test_tbl()


@pytest.fixture(scope='function')
def reload_tester(init_env: None) -> ReloadTester:
    return ReloadTester()


@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: catalog.Table) -> list[exprs.Expr]:
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
def all_datatypes_tbl(reset_db: None) -> catalog.Table:
    return create_all_datatypes_tbl()


@pytest.fixture(scope='function')
def img_tbl(reset_db: None) -> catalog.Table:
    return create_img_tbl('test_img_tbl')


@pytest.fixture(scope='function')
def img_tbl_exprs(indexed_img_tbl: catalog.Table) -> list[exprs.Expr]:
    t = indexed_img_tbl
    return [
        t.img.width,
        t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        t.img.rotate(90).resize([224, 224]),
        t.img.fileurl,
        t.img.localpath,
        t.img.similarity('red truck', idx='img_idx0'),
    ]


@pytest.fixture(scope='function')
def multi_img_tbl_exprs(multi_idx_img_tbl: catalog.Table) -> list[exprs.Expr]:
    t = multi_idx_img_tbl
    return [t.img.similarity('red truck', idx='img_idx1'), t.img.similarity('red truck', idx='img_idx2')]


@pytest.fixture(scope='function')
def small_img_tbl(reset_db: None) -> catalog.Table:
    return create_img_tbl('small_img_tbl', num_rows=40)


@pytest.fixture(scope='function')
def indexed_img_tbl(reset_db: None, clip_embed: func.Function) -> pxt.Table:
    skip_test_if_not_installed('transformers')
    t = create_img_tbl('indexed_img_tbl', num_rows=40)
    t.add_embedding_index('img', idx_name='img_idx0', metric='cosine', image_embed=clip_embed, string_embed=clip_embed)
    return t


@pytest.fixture(scope='function')
def multi_idx_img_tbl(reset_db: None, clip_embed: func.Function) -> pxt.Table:
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
def clip_embed() -> func.Function:
    return clip.using(model_id='openai/clip-vit-base-patch32')


@pytest.fixture(scope='session')
@_retry_hf
def e5_embed() -> func.Function:
    return sentence_transformer.using(model_id='intfloat/e5-large-v2')


@pytest.fixture(scope='session')
@_retry_hf
def all_mpnet_embed() -> func.Function:
    return sentence_transformer.using(model_id='all-mpnet-base-v2')
