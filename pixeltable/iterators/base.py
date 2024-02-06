from __future__ import annotations
from typing import Dict, Any, Tuple, List
from abc import abstractmethod, ABC

from pixeltable.type_system import ColumnType


class ComponentIterator(ABC):
    """Base class for iterators."""

    @classmethod
    @abstractmethod
    def input_schema(cls) -> Dict[str, ColumnType]:
        """Provide the Pixeltable types of the init() parameters

        The keys need to match the names of the init() parameters. This is equivalent to the parameters_types
        parameter of the @function decorator.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> Tuple[Dict[str, ColumnType], List[str]]:
        """Specify the dictionary returned by next() and a list of unstored column names

        Returns:
            a dictionary which is turned into a list of columns in the output table
            a list of unstored column names
        """
        raise NotImplementedError

    def __iter__(self) -> ComponentIterator:
        return self

    @abstractmethod
    def __next__(self) -> Dict[str, Any]:
        """Return the next element of the iterator as a dictionary or raise StopIteration"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the iterator and release all resources"""
        raise NotImplementedError

    @abstractmethod
    def set_pos(self, pos: int) -> None:
        """Set the iterator position to pos"""
        raise NotImplementedError
