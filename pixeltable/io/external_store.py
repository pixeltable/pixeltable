from __future__ import annotations

import abc
from typing import Any

import pixeltable.type_system as ts
from pixeltable import Table


class ExternalStore(abc.ABC):
    """
    Abstract base class that represents an external data store that is linked to a Pixeltable
    table. Subclasses of `ExternalStore` provide functionality for synchronizing between Pixeltable
    and stateful external stores.
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
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> ExternalStore: ...


class Project(ExternalStore, abc.ABC):

    @abc.abstractmethod
    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Project` expects to see in a data export.

        Returns:
            A `dict` mapping names of expected columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Project` provides in a data import.

        Returns:
            A `dict` mapping names of provided columns to their Pixeltable types.
        """

    @abc.abstractmethod
    def delete(self) -> None:
        """
        Deletes this `Project` and all associated (externally stored) data.
        """


# A project that cannot be synced, used mainly for testing.
class MockProject(Project):

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
            'name': self.name,
            'export_cols': {k: v.as_dict() for k, v in self.export_cols.items()},
            'import_cols': {k: v.as_dict() for k, v in self.import_cols.items()}
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> ExternalStore:
        return cls(
            name=md['name'],
            export_cols={k: ts.ColumnType.from_dict(v) for k, v in md['export_cols'].items()},
            import_cols={k: ts.ColumnType.from_dict(v) for k, v in md['import_cols'].items()}
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockProject):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f'MockProject `{self.name}`'
