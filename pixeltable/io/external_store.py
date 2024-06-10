from __future__ import annotations

import abc
import itertools
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

    @abc.abstractmethod
    def sync(self, t: Table, export_data: bool, import_data: bool) -> None:
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
    def get_table_columns(self) -> list[str]: ...

    @abc.abstractmethod
    def validate(self, table: Table) -> None: ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> ExternalStore: ...


class Project(ExternalStore, abc.ABC):

    def __init__(self, col_mapping: Optional[dict[str, str]]):
        self.__user_specified_col_mapping = col_mapping
        self.__col_mapping: Optional[dict[str, str]] = None

    @property
    def col_mapping(self) -> dict[str, str]:
        if self.__col_mapping is None:
            self.__col_mapping = self.__user_specified_col_mapping
            if self.__col_mapping is None:
                export_cols = self.get_export_columns()
                import_cols = self.get_import_columns()
                self.__col_mapping = {col: col for col in itertools.chain(export_cols.keys(), import_cols.keys())}
        return self.__col_mapping

    def get_table_columns(self) -> list[str]:
        return list(self.col_mapping.keys())

    def validate(self, table: Table):
        # Validate names
        t_cols = table.column_names()
        export_cols = self.get_export_columns()
        import_cols = self.get_import_columns()
        for t_col, r_col in self.col_mapping.items():
            if t_col not in t_cols:
                if self.__user_specified_col_mapping is not None:
                    raise excs.Error(
                        f'Column name `{t_col}` appears as a key in `col_mapping`, but Table `{table.get_name()}` '
                        'contains no such column.'
                    )
                else:
                    raise excs.Error(
                        f'Column `{t_col}` does not exist in Table `{table.get_name()}`. Either add a column `{t_col}`, '
                        f'or specify a `col_mapping` to associate a different column with the remote field `{r_col}`.'
                    )
            if r_col not in export_cols and r_col not in import_cols:
                raise excs.Error(
                    f'Column name `{r_col}` appears as a value in `col_mapping`, but the remote '
                    f'configuration has no column `{r_col}`.'
                )
        # Validate column specs
        t_col_types = table.column_types()
        for t_col, r_col in self.col_mapping.items():
            t_col_type = t_col_types[t_col]
            if r_col in export_cols:
                # Validate that the table column can be assigned to the remote column
                r_col_type = export_cols[r_col]
                if not r_col_type.is_supertype_of(t_col_type):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be exported to remote column `{r_col}` (incompatible types; expecting `{r_col_type}`)'
                    )
            if r_col in import_cols:
                # Validate that the remote column can be assigned to the table column
                if table.tbl_version_path.get_column(t_col).is_computed:
                    raise excs.Error(
                        f'Column `{t_col}` is a computed column, which cannot be populated from a remote column'
                    )
                r_col_type = import_cols[r_col]
                if not t_col_type.is_supertype_of(r_col_type):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be imported from remote column `{r_col}` (incompatible types; expecting `{r_col_type}`)'
                    )

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

    def __init__(
            self,
            name: str,
            export_cols: dict[str, ts.ColumnType],
            import_cols: dict[str, ts.ColumnType],
            col_mapping: Optional[dict[str, str]]
    ):
        self.name = name
        self.export_cols = export_cols
        self.import_cols = import_cols
        self.__is_deleted = False
        super().__init__(col_mapping)

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
