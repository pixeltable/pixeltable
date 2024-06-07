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
    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Remote` expects to see in a data export.

        Returns:
            A `dict` mapping names of expected columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Remote` provides in a data import.

        Returns:
            A `dict` mapping names of provided columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def sync(self, t: Table, col_mapping: dict[str, str], export_data: bool, import_data: bool) -> None:
        """
        Synchronizes the given [`Table`][pixeltable.Table] with this `Remote`. This method
        should generally not be called directly; instead, call
        [`t.sync()`][pixeltable.Table.sync].

        Args:
            t: The table to synchronize with this remote.
            col_mapping: A `dict` mapping columns in the Pixeltable table to columns in the remote store.
            export_data: If `True`, data from this table will be exported to the remote during synchronization.
            import_data: If `True`, data from this table will be imported from the remote during synchronization.
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

    def __init__(self, name: str, export_cols: dict[str, ts.ColumnType], import_cols: dict[str, ts.ColumnType]):
        self.name = name
        self.export_cols = export_cols
        self.import_cols = import_cols
        self.__is_deleted = False

    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        return self.export_cols

    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        return self.import_cols

    def sync(self, t: Table, col_mapping: dict[str, str], export_data: bool, import_data: bool) -> NotImplemented:
        raise NotImplementedError()

    def delete(self) -> None:
        self.__is_deleted = True

    @property
    def is_deleted(self) -> bool:
        return self.__is_deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            # TODO Change in next schema version
            'name': self.name,
            'push_cols': {k: v.as_dict() for k, v in self.export_cols.items()},
            'pull_cols': {k: v.as_dict() for k, v in self.import_cols.items()}
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> Remote:
        return cls(
            name=md['name'],
            # TODO Change in next schema version
            export_cols={k: ts.ColumnType.from_dict(v) for k, v in md['push_cols'].items()},
            import_cols={k: ts.ColumnType.from_dict(v) for k, v in md['pull_cols'].items()}
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockRemote):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f'MockRemote `{self.name}`'
