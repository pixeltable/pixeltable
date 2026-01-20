from typing import Iterator, TypedDict

import pixeltable as pxt
from tests.utils import ReloadTester


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def iterator1(x: int) -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(icol=i, scol=f'string {i}')


def test_iterator(uses_db: None, reload_tester: ReloadTester) -> None:
    t = pxt.create_table('tbl', schema={'input': pxt.Int})
    v = pxt.create_view('view', t, iterator=iterator1(t.input))
    t.insert([{'input': 3}, {'input': 5}])
    rs = reload_tester.run_query(v.order_by(v.input, v.pos))
    assert list(rs) == [
        {'input': 3, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
        {'input': 3, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
        {'input': 3, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
        {'input': 5, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
        {'input': 5, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
        {'input': 5, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
        {'input': 5, 'pos': 3, 'icol': 3, 'scol': 'string 3'},
        {'input': 5, 'pos': 4, 'icol': 4, 'scol': 'string 4'},
    ]

    reload_tester.run_reload_test()
