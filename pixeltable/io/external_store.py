from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass
from typing import Any, Optional

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import Table


class ExternalStore(abc.ABC):
    """
    Abstract base class that represents an external data store that is linked to a Pixeltable
    table. Subclasses of `ExternalStore` provide functionality for synchronizing between Pixeltable
    and stateful external stores.
    """

    def __init__(self, name: str) -> None:
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @abc.abstractmethod
    def sync(self, t: Table, export_data: bool, import_data: bool) -> SyncStatus:
        """
        Called by `Table.sync()` to implement store-specific synchronization logic.
        """

    @abc.abstractmethod
    def get_table_columns(self) -> list[str]:
        """
        A list of all Pixeltable column names that are involved in this `ExternalStore`.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> ExternalStore: ...


class Project(ExternalStore, abc.ABC):

    def __init__(self, name: str, col_mapping: Optional[dict[str, str]]):
        super().__init__(name)
        self.col_mapping = col_mapping

    def get_table_columns(self) -> list[str]:
        return list(self.col_mapping.keys())

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

    @classmethod
    def validate_column_names(
            cls,
            table: Table,
            export_cols: dict[str, ts.ColumnType],
            import_cols: dict[str, ts.ColumnType],
            col_mapping: Optional[dict[str, str]]
    ) -> None:
        is_user_specified_col_mapping = col_mapping is not None
        if col_mapping is None:
            col_mapping = {col: col for col in itertools.chain(export_cols.keys(), import_cols.keys())}

        # Validate names
        t_cols = table.column_names()
        for t_col, r_col in col_mapping.items():
            if t_col not in t_cols:
                if is_user_specified_col_mapping:
                    raise excs.Error(
                        f'Column name `{t_col}` appears as a key in `col_mapping`, but Table `{table.get_name()}` '
                        'contains no such column.'
                    )
                else:
                    raise excs.Error(
                        f'Column `{t_col}` does not exist in Table `{table.get_name()}`. Either add a column `{t_col}`, '
                        f'or specify a `col_mapping` to associate a different column with the external field `{r_col}`.'
                    )
            if r_col not in export_cols and r_col not in import_cols:
                raise excs.Error(
                    f'Column name `{r_col}` appears as a value in `col_mapping`, but the external store '
                    f'configuration has no column `{r_col}`.'
                )
        # Validate column specs
        t_col_types = table.column_types()
        for t_col, r_col in col_mapping.items():
            t_col_type = t_col_types[t_col]
            if r_col in export_cols:
                # Validate that the table column can be assigned to the external column
                r_col_type = export_cols[r_col]
                if not r_col_type.is_supertype_of(t_col_type):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be exported to external column `{r_col}` (incompatible types; expecting `{r_col_type}`)'
                    )
            if r_col in import_cols:
                # Validate that the external column can be assigned to the table column
                if table.tbl_version_path.get_column(t_col).is_computed:
                    raise excs.Error(
                        f'Column `{t_col}` is a computed column, which cannot be populated from an external column'
                    )
                r_col_type = import_cols[r_col]
                if not t_col_type.is_supertype_of(r_col_type):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be imported from external column `{r_col}` (incompatible types; expecting `{r_col_type}`)'
                    )


@dataclass(frozen=True)
class SyncStatus:
    external_rows_created: int = 0
    external_rows_deleted: int = 0
    external_rows_updated: int = 0
    pxt_rows_updated: int = 0
    num_excs: int = 0

    def combine(self, other: 'SyncStatus') -> 'SyncStatus':
        return SyncStatus(
            external_rows_created=self.external_rows_created + other.external_rows_created,
            external_rows_deleted=self.external_rows_deleted + other.external_rows_deleted,
            external_rows_updated=self.external_rows_updated + other.external_rows_updated,
            pxt_rows_updated=self.pxt_rows_updated + other.pxt_rows_updated,
            num_excs=self.num_excs + other.num_excs
        )

    @classmethod
    def empty(cls) -> 'SyncStatus':
        return SyncStatus(0, 0, 0, 0, 0)


# A project that cannot be synced, used mainly for testing.
class MockProject(Project):

    def __init__(
            self,
            name: str,
            export_cols: dict[str, ts.ColumnType],
            import_cols: dict[str, ts.ColumnType],
            col_mapping: Optional[dict[str, str]]
    ):
        super().__init__(name, col_mapping)
        self.export_cols = export_cols
        self.import_cols = import_cols
        self.__is_deleted = False

    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        return self.export_cols

    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        return self.import_cols

    def sync(self, t: Table, export_data: bool, import_data: bool) -> NotImplemented:
        raise NotImplementedError()

    def delete(self) -> None:
        self.__is_deleted = True

    @property
    def is_deleted(self) -> bool:
        return self.__is_deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'col_mapping': self.col_mapping,
            'export_cols': {k: v.as_dict() for k, v in self.export_cols.items()},
            'import_cols': {k: v.as_dict() for k, v in self.import_cols.items()}
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> MockProject:
        return cls(
            name=md['name'],
            col_mapping=md['col_mapping'],
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
