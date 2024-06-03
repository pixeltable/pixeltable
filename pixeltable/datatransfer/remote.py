from __future__ import annotations

import abc
from typing import Any

import pixeltable.type_system as ts
from pixeltable import Table


class Remote(abc.ABC):
    """
    Abstract base class that represents a remote data store. Subclasses of `Remote` provide
    functionality for synchronizing between Pixeltable tables and stateful remote stores.
    """

    @abc.abstractmethod
    def get_push_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Remote` expects to see in a data push.

        Returns:
            A `dict` mapping names of expected columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def get_pull_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Remote` provides in a data pull.

        Returns:
            A `dict` mapping names of provided columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def sync(self, t: Table, col_mapping: dict[str, str], push: bool, pull: bool) -> None:
        """
        Synchronizes the given [`Table`][pixeltable.Table] with this `Remote`. This method
        should generally not be called directly; instead, call
        [`t.sync_remotes()`][pixeltable.Table.sync_remotes].

        Args:
            t: The table to synchronize with this remote.
            col_mapping: A `dict` mapping columns in the Pixeltable table to push and/or pull columns in the remote
                store.
            push: If `True`, data from this table will be pushed to the remote during synchronization.
            pull: If `True`, data from this table will be pulled from the remote during synchronization.
        """

    @abc.abstractmethod
    def delete(self) -> None:
        """
        Deletes this `Remote`.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> Remote: ...


# A remote that cannot be synced, used mainly for testing.
class MockRemote(Remote):

    def __init__(self, name: str, push_cols: dict[str, ts.ColumnType], pull_cols: dict[str, ts.ColumnType]):
        self.name = name
        self.push_cols = push_cols
        self.pull_cols = pull_cols
        self.__is_deleted = False

    def get_push_columns(self) -> dict[str, ts.ColumnType]:
        return self.push_cols

    def get_pull_columns(self) -> dict[str, ts.ColumnType]:
        return self.pull_cols

    def sync(self, t: Table, col_mapping: dict[str, str], push: bool, pull: bool) -> NotImplemented:
        raise NotImplementedError()

    def delete(self) -> None:
        self.__is_deleted = True

    @property
    def is_deleted(self) -> bool:
        return self.__is_deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'push_cols': {k: v.as_dict() for k, v in self.push_cols.items()},
            'pull_cols': {k: v.as_dict() for k, v in self.pull_cols.items()}
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> Remote:
        return cls(
            name=md['name'],
            push_cols={k: ts.ColumnType.from_dict(v) for k, v in md['push_cols'].items()},
            pull_cols={k: ts.ColumnType.from_dict(v) for k, v in md['pull_cols'].items()}
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockRemote):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f'MockRemote `{self.name}`'
