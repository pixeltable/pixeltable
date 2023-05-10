import pytest

import pixeltable as pt
from pixeltable import exceptions as exc


class TestClient:
    def test_create_db(self, init_env) -> None:
        cl = pt.Client()
        _ = cl.create_db('test')
        with pytest.raises(exc.Error):
            _ = cl.create_db('test')

        _ = cl.get_db('test')
        with pytest.raises(exc.Error):
            _ = cl.get_db('xyz')

        cl.drop_db('test', force=True)
        with pytest.raises(exc.Error):
            cl.drop_db('test', force=True)

    def test_list_functions(self, init_env) -> None:
        cl = pt.Client()
        _ = cl.list_functions()
        print(_)

