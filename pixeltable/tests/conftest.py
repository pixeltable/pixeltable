from typing import List
import numpy as np
import pandas as pd
import datetime
import os
import logging

import pytest

import pixeltable as pt
import pixeltable.catalog as catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, BoolType, TimestampType, ImageType, JsonType
from pixeltable.tests.utils import read_data_file, create_test_tbl
from pixeltable import exprs
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable import functions as ptf
import socket

def _get_test_db_port() -> int:
    # get an unused port for the test db
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port

@pytest.fixture(scope='session')
def init_env(tmp_path_factory) -> None:
    from pixeltable.env import Env
    # set the relevant env vars for Client() to connect to the test db
    home_dir = str(tmp_path_factory.mktemp('base') / '.pixeltable')
    os.environ['PIXELTABLE_HOME'] = home_dir
    test_db = 'test'
    os.environ['PIXELTABLE_DB'] = test_db
    os.environ['PIXELTABLE_DB_PORT'] =  str(_get_test_db_port())
    # this also runs create_all()
    Env.get().set_up(echo=True)
    logging.info('TestDB url is: %s', Env.get().db_url())
    yield
    # leave db in place for debugging purposes
    # but stop the server so the port is freed
    Env.get().stop_pg()

@pytest.fixture(scope='function')
def test_client(init_env) -> pt.Client:
    cl = pt.Client()
    cl.logging(level=logging.DEBUG)
    yield cl
    cl.reset_catalog()

@pytest.fixture(scope='function')
def test_tbl(test_client: pt.Client) -> catalog.Table:
    return create_test_tbl(test_client)

@pytest.fixture(scope='function')
def test_stored_fn(test_client: pt.Client) -> pt.Function:
    @pt.function(return_type=pt.IntType(), param_types=[pt.IntType()])
    def test_fn(x):
        return x + 1
    test_client.create_function('test_fn', test_fn)
    return test_fn

@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: catalog.Table, test_stored_fn: pt.Function) -> List[exprs.Expr]:
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
        t.c8.errortype,
        t.c8.errormsg,
        ptf.sum(t.c2, group_by=t.c4, order_by=t.c3),
        test_stored_fn(t.c2),
    ]

@pytest.fixture(scope='function')
def img_tbl(test_client: pt.Client) -> catalog.Table:
    cols = [
        catalog.Column('img', ImageType(nullable=False), indexed=False),
        catalog.Column('category', StringType(nullable=False)),
        catalog.Column('split', StringType(nullable=False)),
    ]
    # this table is not indexed in order to avoid the cost of computing embeddings
    tbl = test_client.create_table('test_img_tbl', cols)
    rows, col_names = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    tbl.insert(rows, columns=col_names)
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

# TODO: why does this not work with a session scope? (some user tables don't get created with create_all())
#@pytest.fixture(scope='session')
#def indexed_img_tbl(init_env: None) -> catalog.Table:
#    cl = pt.Client()
#    db = cl.create_db('test_indexed')
@pytest.fixture(scope='function')
def indexed_img_tbl(test_client: pt.Client) -> catalog.Table:
    cl = test_client
    cols = [
        catalog.Column('img', ImageType(nullable=False), indexed=True),
        catalog.Column('category', StringType(nullable=False)),
        catalog.Column('split', StringType(nullable=False)),
    ]
    tbl = cl.create_table('test_indexed_img_tbl', cols)
    rows, col_names = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    # select output_rows randomly in the hope of getting a good sample of the available categories
    rng = np.random.default_rng(17)
    idxs = rng.choice(np.arange(len(rows)), size=40, replace=False)
    rows = [rows[i] for i in idxs]
    tbl.insert(rows, columns=col_names)
    return tbl
