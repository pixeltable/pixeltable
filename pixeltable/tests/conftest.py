import numpy as np

import pytest
import sqlalchemy as sql

import pixeltable as pt
import pixeltable.catalog as catalog
from pixeltable.type_system import StringType, ImageType
from pixeltable.tests.utils import read_data_file, make_tbl, create_table_data


def init_env(tmp_path) -> None:
    engine = sql.create_engine('sqlite:///:memory:', echo=True)
    from pixeltable import env
    env.init_env(tmp_path, engine)


@pytest.fixture(scope='function')
def test_env(tmp_path) -> None:
    init_env(tmp_path)


@pytest.fixture(scope='function')
def test_tbl(test_env) -> catalog.Table:
    cl = pt.Client()
    db = cl.create_db('test')
    t = make_tbl(db, 'test1', ['c1', 'c2', 'c3', 'c4'])
    data = create_table_data(t)
    t.insert_pandas(data)
    return t


@pytest.fixture(scope='session')
def test_img_tbl(tmp_path_factory) -> catalog.Table:
    init_env(tmp_path_factory.mktemp('base'))
    cl = pt.Client()
    db = cl.create_db('test')
    cols = [
        catalog.Column('img', ImageType(), nullable=False),
        catalog.Column('category', StringType(), nullable=False),
        catalog.Column('split', StringType(), nullable=False),
    ]
    tbl = db.create_table('test', cols)
    df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    # select rows randomly in the hope of getting a good sample of the available categories
    rng = np.random.default_rng(17)
    idxs = rng.choice(np.arange(len(df)), size=40, replace=False)
    tbl.insert_pandas(df.iloc[idxs])
    return tbl
