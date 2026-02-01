from typing import Any, Iterator, TypedDict

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from tests.utils import ReloadTester


class MyRow(TypedDict):
    icol: int
    scol: str


@pxt.iterator
def simple_iterator(x: int, str_text: str = 'string') -> Iterator[MyRow]:
    for i in range(x):
        yield MyRow(icol=i, scol=f'{str_text} {i}')


@simple_iterator.validate
def _(bound_args: dict[str, Any]) -> None:
    if 'x' in bound_args and bound_args['x'] < 0:
        raise pxt.Error('Parameter `x` must be non-negative.')
    if 'str_text' not in bound_args:
        raise pxt.Error('Parameter `str_text` must be a constant.')
    if not bound_args['str_text'].isidentifier():
        raise pxt.Error('Parameter `str_text` must be a valid identifier.')


@pxt.iterator
class class_based_iterator(pxt.PxtIterator[MyRow]):
    x: int
    str_text: str
    current: int

    def __init__(self, x: int, str_text: str = 'string') -> None:
        self.x = x
        self.str_text = str_text
        self.current = 0

    def __next__(self) -> MyRow:
        if self.current >= self.x:
            raise StopIteration
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}')
        self.current += 1
        return result


@class_based_iterator.validate
def _(bound_args: dict[str, Any]) -> None:
    if 'x' in bound_args and bound_args['x'] < 0:
        raise pxt.Error('Parameter `x` must be non-negative.')
    if 'str_text' not in bound_args:
        raise pxt.Error('Parameter `str_text` must be a constant.')
    if not bound_args['str_text'].isidentifier():
        raise pxt.Error('Parameter `str_text` must be a valid identifier.')


@pxt.iterator
class iterator_with_seek(pxt.PxtIterator[MyRow]):
    x: int
    str_text: str
    current: int

    def __init__(self, x: int, str_text: str = 'string') -> None:
        self.x = x
        self.str_text = str_text
        self.current = 0

    def __next__(self) -> MyRow:
        if self.current >= self.x:
            raise StopIteration
        result = MyRow(icol=self.current, scol=f'{self.str_text} {self.current}')
        self.current += 1
        return result

    def seek(self, pos: int, **kwargs: Any) -> None:
        assert kwargs['scol'] == f'{self.str_text} {pos}'
        self.current = pos


class TestIterator:
    def test_iterator(self, uses_db: None, reload_tester: ReloadTester) -> None:
        for n, it in enumerate((simple_iterator, class_based_iterator)):
            assert callable(it)
            t = pxt.create_table(f'tbl_{n}', schema={'input': pxt.Int})
            t.insert([{'input': 2}])
            v = pxt.create_view(f'view_{n}', t, iterator=it(t.input))
            t.insert([{'input': 3}, {'input': 5}])
            rs = reload_tester.run_query(v.order_by(v.input, v.pos))
            assert list(rs) == [
                {'input': 2, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 2, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 3, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 3, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 3, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'input': 5, 'pos': 0, 'icol': 0, 'scol': 'string 0'},
                {'input': 5, 'pos': 1, 'icol': 1, 'scol': 'string 1'},
                {'input': 5, 'pos': 2, 'icol': 2, 'scol': 'string 2'},
                {'input': 5, 'pos': 3, 'icol': 3, 'scol': 'string 3'},
                {'input': 5, 'pos': 4, 'icol': 4, 'scol': 'string 4'},
            ]

            # Test that the iterator-specific validator works at insertion time
            with pytest.raises(pxt.Error, match=r'Parameter `x` must be non-negative.'):
                t.insert([{'input': -1}])

            # Test that the iterator-specific validator works at iterator creation time
            with pytest.raises(pxt.Error, match=r'Parameter `x` must be non-negative.'):
                it(-1)
            with pytest.raises(pxt.Error, match=r'Parameter `str_text` must be a constant.'):
                it(t.input, str_text=pxtf.uuid.uuid7().to_string())
            with pytest.raises(pxt.Error, match=r'Parameter `str_text` must be a valid identifier.'):
                it(t.input, str_text='I am not a valid identifier!')

        reload_tester.run_reload_test()
