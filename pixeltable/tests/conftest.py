import json
import logging
import os
import pathlib
from typing import List

import numpy as np
import pytest
import PIL.Image

import pixeltable as pxt
import pixeltable.catalog as catalog
from pixeltable import exprs
import pixeltable.functions as pxtf
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable.metadata import SystemInfo, create_system_info
from pixeltable.metadata.schema import TableSchemaVersion, TableVersion, Table, Function, Dir
from pixeltable.tests.utils import read_data_file, create_test_tbl, create_all_datatypes_tbl, skip_test_if_not_installed
from pixeltable.type_system import StringType, ImageType, FloatType


@pytest.fixture(scope='session')
def init_env(tmp_path_factory) -> None:
    from pixeltable.env import Env
    # set the relevant env vars for Client() to connect to the test db

    shared_home = pathlib.Path(os.environ.get('PIXELTABLE_HOME', str(pathlib.Path.home() / '.pixeltable')))
    home_dir = str(tmp_path_factory.mktemp('base') / '.pixeltable')
    os.environ['PIXELTABLE_HOME'] = home_dir
    os.environ['PIXELTABLE_CONFIG'] = str(shared_home / 'config.yaml')
    test_db = 'test'
    os.environ['PIXELTABLE_DB'] = test_db
    os.environ['PIXELTABLE_PGDATA'] = str(shared_home / 'pgdata')

    # ensure this home dir exits
    shared_home.mkdir(parents=True, exist_ok=True)
    # this also runs create_all()
    Env.get().set_up(echo=True)
    yield
    # leave db in place for debugging purposes

@pytest.fixture(scope='function')
def test_client(init_env) -> pxt.Client:
    # Clean the DB *before* instantiating a client object. This is because some tests
    # (such as test_migration.py) may leave the DB in a broken state, from which the
    # client is uninstantiable.
    clean_db()
    cl = pxt.Client(reload=True)
    cl.logging(level=logging.DEBUG, to_stdout=True)
    yield cl


def clean_db(restore_tables: bool = True) -> None:
    from pixeltable.env import Env
    # The logic from Client.reset_catalog() has been moved here, so that it
    # does not rely on instantiating a Client object. As before, UUID-named data tables will
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
def test_tbl(test_client: pxt.Client) -> catalog.Table:
    return create_test_tbl(test_client)

# @pytest.fixture(scope='function')
# def test_stored_fn(test_client: pxt.Client) -> pxt.Function:
#     @pxt.udf(return_type=pxt.IntType(), param_types=[pxt.IntType()])
#     def test_fn(x):
#         return x + 1
#     test_client.create_function('test_fn', test_fn)
#     return test_fn

@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: catalog.Table) -> List[exprs.Expr]:
#def test_tbl_exprs(test_tbl: catalog.Table, test_stored_fn: pxt.Function) -> List[exprs.Expr]:

    t = test_tbl
    return [
        t.c1,
        t.c7['*'].f1,
        exprs.Literal('test'),
        exprs.InlineDict({
            'a': t.c1, 'b': t.c6.f1, 'c': 17,
            'd': exprs.InlineDict({'e': t.c2}),
            'f': exprs.InlineArray((t.c3, t.c3))
        }),
        exprs.InlineArray([[t.c2, t.c2], [t.c2, t.c2]]),
        t.c2 > 5,
        t.c2 == None,
        ~(t.c2 > 5),
        (t.c2 > 5) & (t.c1 == 'test'),
        (t.c2 > 5) | (t.c1 == 'test'),
        t.c7['*'].f5 >> [R[3], R[2], R[1], R[0]],
        t.c8[0, 1:],
        t.c2.astype(FloatType()),
        (t.c2 + 1).astype(FloatType()),
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
def all_datatypes_tbl(test_client: pxt.Client) -> catalog.Table:
    return create_all_datatypes_tbl(test_client)

@pytest.fixture(scope='function')
def img_tbl(test_client: pxt.Client) -> catalog.Table:
    schema = {
        'img': ImageType(nullable=False),
        'category': StringType(nullable=False),
        'split': StringType(nullable=False),
    }
    # this table is not indexed in order to avoid the cost of computing embeddings
    tbl = test_client.create_table('test_img_tbl', schema)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    tbl.insert(rows)
    return tbl

@pytest.fixture(scope='function')
def img_tbl_exprs(img_tbl: catalog.Table) -> List[exprs.Expr]:
    img_t = img_tbl
    return [
        img_t.img.width,
        img_t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        img_t.img.rotate(90).resize([224, 224]),
        img_t.img.fileurl,
        img_t.img.localpath,
    ]

@pytest.fixture(scope='function')
def small_img_tbl(test_client: pxt.Client) -> catalog.Table:
    cl = test_client
    schema = {
        'img': ImageType(nullable=False),
        'category': StringType(nullable=False),
        'split': StringType(nullable=False),
    }
    tbl = cl.create_table('test_indexed_img_tbl', schema)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    # select output_rows randomly in the hope of getting a good sample of the available categories
    rng = np.random.default_rng(17)
    idxs = rng.choice(np.arange(len(rows)), size=40, replace=False)
    rows = [rows[i] for i in idxs]
    tbl.insert(rows)
    return tbl
