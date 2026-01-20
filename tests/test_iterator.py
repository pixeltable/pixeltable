from typing import Iterator, TypedDict

import pixeltable as pxt
from tests.utils import ReloadTester


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def simple_iterator(x: int) -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(icol=i, scol=f'string {i}')


@pxt.iterator
class class_based_iterator:
    x: int
    current: int

    def __init__(self, x: int):
        self.x = x
        self.current = 0

    def __next__(self) -> MyRow:
        if self.current >= self.x:
            raise StopIteration
        result = MyRow(icol=self.current, scol=f'string {self.current}')
        self.current += 1
        return result


def test_iterator(uses_db: None, reload_tester: ReloadTester) -> None:
    t = pxt.create_table('tbl', schema={'input': pxt.Int})

    for n, it in enumerate((simple_iterator, class_based_iterator)):
        v = pxt.create_view(f'view_{n}', t, iterator=it(t.input))
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
