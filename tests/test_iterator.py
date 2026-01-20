from typing import Iterator, TypedDict
import pixeltable as pxt


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def iterator1(x: int) -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(i, f'string {i}')


def test_iterator(uses_db: None) -> None:
    t = pxt.create_table('tbl', schema={'icol': pxt.Int})
    v = pxt.create_view('view', t, iterator=iterator1(t.icol))
    t.insert({'icol': 3})
    v.head()
