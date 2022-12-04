import numpy as np

import pytest

import pixeltable as pt
import pixeltable.catalog as catalog
from pixeltable.type_system import StringType, ImageType
from pixeltable.tests.utils import read_data_file, make_tbl, create_table_data


@pytest.fixture(scope='session')
def init_db(tmp_path_factory) -> None:
    from pixeltable import env
    # this also runs create_all()
    env.init_env(tmp_path_factory.mktemp('base'), echo=True)
    yield
    env.teardown_env()


@pytest.fixture(scope='function')
def test_db(init_db: None) -> pt.Db:
    cl = pt.Client()
    db = cl.create_db(f'test')
    yield db
    cl.drop_db(db.name, force=True)


@pytest.fixture(scope='function')
def test_tbl(test_db: pt.Db) -> catalog.Table:
    t = make_tbl(test_db, 'test_tbl', ['c1', 'c2', 'c3', 'c4'])
    data = create_table_data(t)
    t.insert_pandas(data)
    return t


@pytest.fixture(scope='function')
def img_tbl(test_db: pt.Db) -> catalog.Table:
    cols = [
        catalog.Column('img', ImageType(), nullable=False),
        catalog.Column('category', StringType(), nullable=False),
        catalog.Column('split', StringType(), nullable=False),
    ]
    # this table is not indexed in order to avoid the cost of computing embeddings
    tbl = test_db.create_table('test_img_tbl', cols, indexed=False)
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
        catalog.Column('img', ImageType(), nullable=False),
        catalog.Column('category', StringType(), nullable=False),
        catalog.Column('split', StringType(), nullable=False),
    ]
    tbl = db.create_table('test_indexed_img_tbl', cols, indexed=True)
    df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    # select rows randomly in the hope of getting a good sample of the available categories
    rng = np.random.default_rng(17)
    idxs = rng.choice(np.arange(len(df)), size=40, replace=False)
    tbl.insert_pandas(df.iloc[idxs])
    return tbl
