from typing import Generic, Iterable, Iterator, Optional, TypeVar

from .expr import Expr

T = TypeVar('T')


class ExprDict(Generic[T]):
    """
    A dictionary that maps Expr instances to values of type T.

    We cannot use dict[Expr, T] because Expr.__eq__() serves a different purpose than the default __eq__.
    """

    _data: dict[int, tuple[Expr, T]]

    def __init__(self, iterable: Optional[Iterable[tuple[Expr, T]]] = None):
        self._data = {}

        if iterable is not None:
            for key, value in iterable:
                self[key] = value

    def __setitem__(self, key: Expr, value: T) -> None:
        self._data[key.id] = (key, value)

    def __getitem__(self, key: Expr) -> T:
        return self._data[key.id][1]

    def __delitem__(self, key: Expr) -> None:
        del self._data[key.id]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Expr]:
        return (expr for expr, _ in self._data.values())

    def __contains__(self, key: Expr) -> bool:
        return key.id in self._data

    def get(self, key: Expr, default: Optional[T] = None) -> Optional[T]:
        item = self._data.get(key.id)
        return item[1] if item is not None else default

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> Iterator[Expr]:
        return iter(self)

    def values(self) -> Iterator[T]:
        return (value for _, value in self._data.values())

    def items(self) -> Iterator[tuple[Expr, T]]:
        return ((expr, value) for expr, value in self._data.values())
