import abc

from pixeltable import Table


class Remote:

    @abc.abstractmethod
    def push(self, t: Table, col_mapping: dict[str, str]) -> None: ...

    @abc.abstractmethod
    def pull(self, t: Table, col_mapping: dict[str, str]) -> None: ...
