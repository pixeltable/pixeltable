from typing import Iterator, TypedDict
import pixeltable as pxt


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def iterator1(x: int) -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(i, f'string {i}')
