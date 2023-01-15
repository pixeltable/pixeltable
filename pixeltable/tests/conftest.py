import numpy as np
import pandas as pd
import datetime

import pytest

import pixeltable as pt
import pixeltable.catalog as catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, BoolType, TimestampType, ImageType, JsonType
from pixeltable.tests.utils import read_data_file, make_tbl, create_table_data


@pytest.fixture(scope='session')
def init_db(tmp_path_factory) -> None:
    from pixeltable.env import Env
    # this also runs create_all()
    db_name = 'test'
    Env.get().set_up(tmp_path_factory.mktemp('base'), db_name=db_name, echo=True)
    yield
    # leave db in place for debugging purposes


@pytest.fixture(scope='function')
def test_db(init_db: None) -> pt.Db:
    cl = pt.Client()
    db = cl.create_db(f'test')
    yield db
    cl.drop_db(db.name, force=True)


@pytest.fixture(scope='function')
def test_tbl(test_db: pt.Db) -> catalog.Table:
    cols = [
        catalog.Column('c1', StringType(), nullable=False),
        catalog.Column('c2', IntType(), nullable=False),
        catalog.Column('c3', FloatType(), nullable=False),
        catalog.Column('c4', BoolType(), nullable=False),
        catalog.Column('c5', TimestampType(), nullable=False),
        catalog.Column('c6', JsonType(), nullable=False),
        catalog.Column('c7', JsonType(), nullable=False),
    ]
    t = test_db.create_table('test__tbl', cols)

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
    data = {'c1': c1_data, 'c2': c2_data, 'c3': c3_data, 'c4': c4_data, 'c5': c5_data, 'c6': c6_data, 'c7': c7_data}
    pd_df = pd.DataFrame(data=data)
    t.insert_pandas(pd_df)
    return t


@pytest.fixture(scope='function')
def img_tbl(test_db: pt.Db) -> catalog.Table:
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


# TODO: why does this not work with a session scope? (some user tables don't get created with create_all())
#@pytest.fixture(scope='session')
#def indexed_img_tbl(init_db: None) -> catalog.Table:
#    cl = pt.Client()
#    db = cl.create_db('test_indexed')
@pytest.fixture(scope='function')
def indexed_img_tbl(test_db: pt.Db) -> catalog.Table:
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
