import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs


class TestClient:

    def test_list_functions(self, init_env) -> None:
        cl = pxt.Client()
        _ = pxt.list_functions()
        print(_)

    def test_drop_table(self, test_tbl: pxt.Table) -> None:
        cl = pxt.Client()
        t = pxt.get_table('test_tbl')
        pxt.drop_table('test_tbl')
        with pytest.raises(excs.Error):
            _ = pxt.get_table('test_tbl')
        with pytest.raises(excs.Error):
            _ = t.show(1)

