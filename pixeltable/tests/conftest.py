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
from pixeltable.tests.utils import read_data_file, make_tbl, create_table_data
from pixeltable import exprs
from pixeltable.exprs import RELATIVE_PATH_ROOT as R
from pixeltable import functions as ptf


@pytest.fixture(scope='session')
def init_env(tmp_path_factory) -> None:
    from pixeltable.env import Env
    # set the relevant env vars for Client() to connect to the test db
    home_dir = str(tmp_path_factory.mktemp('base') / '.pixeltable')
    os.environ['PIXELTABLE_HOME'] = home_dir
    test_db = 'test'
    os.environ['PIXELTABLE_DB'] = test_db
    # this also runs create_all()
    Env.get().set_up(echo=True)
    yield
    # leave db in place for debugging purposes


@pytest.fixture(scope='function')
def test_db(init_env) -> catalog.Db:
    cl = pt.Client()
    cl.logging(level=logging.DEBUG)
    db = cl.create_db(f'test')
    yield db
    cl.drop_db(db.name, force=True)


@pytest.fixture(scope='function')
def test_tbl(test_db: catalog.Db) -> catalog.Table:
    cols = [
        catalog.Column('c1', StringType(), nullable=False),
        catalog.Column('c1n', StringType(), nullable=True),
        catalog.Column('c2', IntType(), nullable=False),
        catalog.Column('c3', FloatType(), nullable=False),
        catalog.Column('c4', BoolType(), nullable=False),
        catalog.Column('c5', TimestampType(), nullable=False),
        catalog.Column('c6', JsonType(), nullable=False),
        catalog.Column('c7', JsonType(), nullable=False),
    ]
    t = test_db.create_table('test_tbl', cols)
    t.add_column(catalog.Column('c8', computed_with=[[1, 2, 3], [4, 5, 6]]))

    num_rows = 100
    d1 = {
        'f1': 'test string 1',
        'f2': 1,
        'f3': 1.0,
        'f4': True,
        'f5': [1.0, 2.0, 3.0, 4.0],
        'f6': {
            'f7': 'test string 2',
            'f8': [1.0, 2.0, 3.0, 4.0],
        },
    }
    d2 = [d1, d1]

    c1_data = [f'test string {i}' for i in range(num_rows)]
    c2_data = [i for i in range(num_rows)]
    c3_data = [float(i) for i in range(num_rows)]
    c4_data = [bool(i % 2) for i in range(num_rows)]
    c5_data = [datetime.datetime.now()] * num_rows
    c6_data = []
    for i in range(num_rows):
        d = {
            'f1': f'test string {i}',
            'f2': i,
            'f3': float(i),
            'f4': bool(i % 2),
            'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {
                'f7': 'test string 2',
                'f8': [1.0, 2.0, 3.0, 4.0],
            },
        }
        c6_data.append(d)

    c7_data = [d2] * num_rows
    data = {
        'c1': c1_data, 'c1n': [s if i % 10 != 0 else None for i, s in enumerate(c1_data)],
        'c2': c2_data, 'c3': c3_data, 'c4': c4_data, 'c5': c5_data, 'c6': c6_data, 'c7': c7_data}
    pd_df = pd.DataFrame(data=data)
    t.insert_pandas(pd_df)
    return t

@pytest.fixture(scope='function')
def test_stored_fn(test_db: catalog.Db) -> pt.Function:
    @pt.function(return_type=pt.IntType(), param_types=[pt.IntType()])
    def test_fn(x):
        return x + 1
    test_db.create_function('test_fn', test_fn)
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
def img_tbl(test_db: catalog.Db) -> catalog.Table:
    cols = [
        catalog.Column('img', ImageType(), nullable=False, indexed=False),
        catalog.Column('category', StringType(), nullable=False),
        catalog.Column('split', StringType(), nullable=False),
    ]
    # this table is not indexed in order to avoid the cost of computing embeddings
    tbl = test_db.create_table('test_img_tbl', cols)
    df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    tbl.insert_pandas(df)
    return tbl

@pytest.fixture(scope='function')
def img_tbl_exprs(img_tbl: catalog.Table) -> List[exprs.Expr]:
    img_t = img_tbl
    return [
        img_t.img.width,
        img_t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        img_t.img.rotate(90).resize([224, 224]),
    ]


# TODO: why does this not work with a session scope? (some user tables don't get created with create_all())
#@pytest.fixture(scope='session')
#def indexed_img_tbl(init_env: None) -> catalog.Table:
#    cl = pt.Client()
#    db = cl.create_db('test_indexed')
@pytest.fixture(scope='function')
def indexed_img_tbl(test_db: catalog.Db) -> catalog.Table:
    db = test_db
    cols = [
        catalog.Column('img', ImageType(), nullable=False, indexed=True),
        catalog.Column('category', StringType(), nullable=False),
        catalog.Column('split', StringType(), nullable=False),
    ]
    tbl = db.create_table('test_indexed_img_tbl', cols)
    df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    # select rows randomly in the hope of getting a good sample of the available categories
    rng = np.random.default_rng(17)
    idxs = rng.choice(np.arange(len(df)), size=40, replace=False)
    tbl.insert_pandas(df.iloc[idxs])
    return tbl
