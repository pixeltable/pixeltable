import pytest

import pixeltable as pt
from pixeltable import exceptions as exc


class TestClient:
    def test_list_functions(self, init_env) -> None:
        cl = pt.Client()
        _ = cl.list_functions()
        print(_)

