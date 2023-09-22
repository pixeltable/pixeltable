import pytest

import pixeltable as pt
from pixeltable import exceptions as exc


class TestClient:
    def test_list_functions(self, init_env) -> None:
        cl = pt.Client()
        _ = cl.list_functions()
        print(_)

    def test_drop_table(self, test_tbl: pt.MutableTable) -> None:
        cl = pt.Client()
        t = cl.get_table('test_tbl')
        cl.drop_table('test_tbl')
        with pytest.raises(exc.Error):
            _ = cl.get_table('test_tbl')
        with pytest.raises(exc.Error):
            _ = t.show(1)

