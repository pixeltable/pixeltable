import json
import logging
import os
import pathlib
from typing import List

import pytest
from filelock import FileLock

import pixeltable as pxt
import pixeltable.catalog as catalog
import pixeltable.functions as pxtf
from pixeltable import exprs
from pixeltable.env import Env
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.metadata import SystemInfo, create_system_info
from pixeltable.metadata.schema import Dir, Function, Table, TableSchemaVersion, TableVersion
from pixeltable.utils.filecache import FileCache

from .utils import create_all_datatypes_tbl, create_img_tbl, create_test_tbl, reload_catalog, skip_test_if_not_installed


@pytest.fixture(scope='session')
def init_env(tmp_path_factory, worker_id) -> None:
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

    # Ensure the shared home directory exists.
    shared_home.mkdir(parents=True, exist_ok=True)

    # Initialize Pixeltable. If using multiple workers, they need to be initialized synchronously to ensure we
    # don't have several processes trying to initialize pgserver in parallel.
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / 'pxt-init.lock')):
        pxt.init()

    Env.get().configure_logging(level=logging.DEBUG, to_stdout=True)


@pytest.fixture(scope='function')
def reset_db(init_env) -> None:
    from pixeltable.env import Env

    # Clean the DB *before* reloading. This is because some tests
    # (such as test_migration.py) may leave the DB in a broken state.
    clean_db()
    Env.get().default_time_zone = None
    reload_catalog()
    FileCache.get().set_capacity(10 << 30)  # 10 GiB


def clean_db(restore_tables: bool = True) -> None:
    from pixeltable.env import Env

    # UUID-named data tables will
    # not be cleaned. If in the future it is desirable to clean out data tables as well,
    # the commented lines may be used to drop ALL tables from the test db.
    # sql_md = declarative_base().metadata
    # sql_md.reflect(Env.get().engine)
    # sql_md.drop_all(bind=Env.get().engine)
    engine = Env.get().engine
    SystemInfo.__table__.drop(engine, checkfirst=True)
    TableSchemaVersion.__table__.drop(engine, checkfirst=True)
    TableVersion.__table__.drop(engine, checkfirst=True)
    Table.__table__.drop(engine, checkfirst=True)
    Function.__table__.drop(engine, checkfirst=True)
    Dir.__table__.drop(engine, checkfirst=True)
    if restore_tables:
        Dir.__table__.create(engine)
        Function.__table__.create(engine)
        Table.__table__.create(engine)
        TableVersion.__table__.create(engine)
        TableSchemaVersion.__table__.create(engine)
        SystemInfo.__table__.create(engine)
        create_system_info(engine)


@pytest.fixture(scope='function')
def test_tbl(reset_db) -> catalog.Table:
    return create_test_tbl()

@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: catalog.Table) -> List[exprs.Expr]:
    t = test_tbl
    return [
        t.c1,
        t.c7['*'].f1,
        exprs.Literal('test'),
        exprs.InlineDict({
            'a': t.c1, 'b': t.c6.f1, 'c': 17,
            'd': exprs.InlineDict({'e': t.c2}),
            'f': exprs.InlineList([t.c3, t.c3])
        }),
        exprs.InlineList([[t.c2, t.c2], [t.c2, t.c2]]),
        t.c2 > 5,
        t.c2 == None,
        ~(t.c2 > 5),
        (t.c2 > 5) & (t.c1 == 'test'),
        (t.c2 > 5) | (t.c1 == 'test'),
        t.c7['*'].f5 >> [R[3], R[2], R[1], R[0]],
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
def all_datatypes_tbl(reset_db) -> catalog.Table:
    return create_all_datatypes_tbl()

@pytest.fixture(scope='function')
def img_tbl(reset_db) -> catalog.Table:
    return create_img_tbl('test_img_tbl')

@pytest.fixture(scope='function')
def img_tbl_exprs(indexed_img_tbl: catalog.Table) -> List[exprs.Expr]:
    t = indexed_img_tbl
    return [
        t.img.width,
        t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        t.img.rotate(90).resize([224, 224]),
        t.img.fileurl,
        t.img.localpath,
        t.img.similarity('red truck'),
    ]

@pytest.fixture(scope='function')
def small_img_tbl(reset_db) -> catalog.Table:
    return create_img_tbl('small_img_tbl', num_rows=40)

@pytest.fixture(scope='function')
def indexed_img_tbl(reset_db) -> pxt.Table:
    skip_test_if_not_installed('transformers')
    t = create_img_tbl('indexed_img_tbl', num_rows=40)
    from .utils import clip_img_embed, clip_text_embed
    t.add_embedding_index('img', metric='cosine', image_embed=clip_img_embed, string_embed=clip_text_embed)
    return t
