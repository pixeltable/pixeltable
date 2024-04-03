import abc

from pixeltable import Table


class Remote:

    @abc.abstractmethod
    def push(self, t: Table) -> None: ...
