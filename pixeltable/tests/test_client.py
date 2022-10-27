import pytest

import pixeltable as pt
from pixeltable import exceptions as exc


class TestClient:
    def test_create_db(self, test_db: None) -> None:
        cl = pt.Client()
        _ = cl.create_db('test')
        with pytest.raises(exc.DuplicateNameError):
            _ = cl.create_db('test')

        _ = cl.get_db('test')
        with pytest.raises(exc.UnknownEntityError):
            _ = cl.get_db('xyz')

