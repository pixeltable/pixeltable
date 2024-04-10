import abc
from typing import Any

from pixeltable import Table


class Remote:

    @abc.abstractmethod
    def push(self, t: Table, col_mapping: dict[str, str]) -> None: ...

    @abc.abstractmethod
    def pull(self, t: Table, col_mapping: dict[str, str]) -> None: ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> None: ...
