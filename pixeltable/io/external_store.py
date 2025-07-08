from __future__ import annotations

import abc
import itertools
import logging
from typing import Any, Optional

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import Column, Table
from pixeltable.catalog import ColumnHandle, TableVersion
from pixeltable.catalog.update_status import UpdateStatus

_logger = logging.getLogger('pixeltable')


class ExternalStore(abc.ABC):
    """
    Abstract base class that represents an external data store that is linked to a Pixeltable
    table. Subclasses of `ExternalStore` provide functionality for synchronizing between Pixeltable
    and stateful external stores.
    """

    __name: str

    def __init__(self, name: str) -> None:
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @abc.abstractmethod
    def link(self, tbl_version: TableVersion) -> None:
        """Creates store-specific metadata needed to implement sync()."""

    @abc.abstractmethod
    def unlink(self, tbl_version: TableVersion) -> None:
        """Removes store-specific metadata created in link()."""

    @abc.abstractmethod
    def get_local_columns(self) -> list[ColumnHandle]:
        """
        Gets a list of all local (Pixeltable) columns that are associated with this external store.
        """

    @abc.abstractmethod
    def sync(self, t: Table, export_data: bool, import_data: bool) -> UpdateStatus:
        """
        Called by `Table.sync()` to implement store-specific synchronization logic.
        """

    @abc.abstractmethod
    def as_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, md: dict[str, Any]) -> ExternalStore: ...


class Project(ExternalStore, abc.ABC):
    """
    An `ExternalStore` that represents a labeling project. Extends `ExternalStore` with a few
    additional capabilities specific to such projects.
    """

    _col_mapping: dict[ColumnHandle, str]  # col -> external col name
    stored_proxies: dict[ColumnHandle, ColumnHandle]  # original col -> proxy col

    def __init__(
        self,
        name: str,
        col_mapping: dict[ColumnHandle, str],
        stored_proxies: Optional[dict[ColumnHandle, ColumnHandle]],
    ):
        super().__init__(name)
        self._col_mapping = col_mapping

        # A mapping from original columns to proxy columns. A proxy column is an identical copy of a column that is
        # guaranteed to be stored; the Project will dynamically create and tear down proxy columns as needed. There
        # are two reasons this might happen:
        # (i) to force computed media data to be persisted; or
        # (ii) to force media data to be materialized in a particular location.
        # For each entry (k, v) in the dict, `v` is the stored proxy column for `k`. The proxy column `v` will
        # necessarily be a column of the table to which this project is linked, but `k` need not be; it might be a
        # column of a base table.
        # Note from aaron-siegel: This methodology is inefficient in the case where a table has many views with a high
        # proportion of overlapping rows, all proxying the same base column.
        if stored_proxies is None:
            self.stored_proxies: dict[ColumnHandle, ColumnHandle] = {}
        else:
            self.stored_proxies = stored_proxies

    def get_local_columns(self) -> list[ColumnHandle]:
        return list(self.col_mapping.keys())

    def link(self, tbl_version: TableVersion) -> None:
        # All of the media columns being linked need to either be stored computed columns, or else have stored proxies.
        # This ensures that the media in those columns resides in the media store.
        # First determine which columns (if any) need stored proxies, but don't have one yet.
        stored_proxies_needed: list[Column] = []
        for col_handle in self.col_mapping:
            col = col_handle.get()
            if col.col_type.is_media_type() and not (col.is_stored and col.is_computed):
                # If this column is already proxied in some other Project, use the existing proxy to avoid
                # duplication. Otherwise, we'll create a new one.
                for store in tbl_version.external_stores.values():
                    if isinstance(store, Project) and col_handle in store.stored_proxies:
                        self.stored_proxies[col_handle] = store.stored_proxies[col_handle]
                        break
                if col_handle not in self.stored_proxies:
                    # We didn't find it in an existing Project
                    stored_proxies_needed.append(col)

        if len(stored_proxies_needed) > 0:
            _logger.info(f'Creating stored proxies for columns: {[col.name for col in stored_proxies_needed]}')
            # Create stored proxies for columns that need one
            proxy_cols = [self.create_stored_proxy(col) for col in stored_proxies_needed]
            # Add the columns; this will also update table metadata.
            tbl_version.add_columns(proxy_cols, print_stats=False, on_error='ignore')
            self.stored_proxies.update(
                {col.handle: proxy_col.handle for col, proxy_col in zip(stored_proxies_needed, proxy_cols)}
            )

    def unlink(self, tbl_version: TableVersion) -> None:
        # Determine which stored proxies can be deleted. (A stored proxy can be deleted if it is not referenced by
        # any *other* external store for this table.)
        deletions_needed: set[ColumnHandle] = set(self.stored_proxies.values())
        for name, store in tbl_version.external_stores.items():
            if isinstance(store, Project) and name != self.name:
                deletions_needed = deletions_needed.difference(set(store.stored_proxies.values()))
        if len(deletions_needed) > 0:
            _logger.info(f'Removing stored proxies for columns: {[col.get().name for col in deletions_needed]}')
            tbl_version._drop_columns(col.get() for col in deletions_needed)
            self.stored_proxies.clear()

    def create_stored_proxy(self, col: Column) -> Column:
        """
        Creates a proxy column for the specified column. The proxy column will be created in the specified
        `TableVersion`.
        """
        from pixeltable import exprs

        assert (
            col.col_type.is_media_type() and not (col.is_stored and col.is_computed) and col not in self.stored_proxies
        )
        proxy_col = Column(
            name=None,
            # Force images in the proxy column to be materialized inside the media store, in a normalized format.
            # TODO(aaron-siegel): This is a temporary solution and it will be replaced by a proper `destination`
            #   parameter for computed columns. Among other things, this solution does not work for video or audio.
            #   Once `destination` is implemented, it can be replaced with a simple `ColumnRef`.
            computed_with=exprs.ColumnRef(col).apply(lambda x: x, col_type=col.col_type),
            stored=True,
        )
        return proxy_col

    @property
    def col_mapping(self) -> dict[ColumnHandle, str]:
        return self._col_mapping

    @abc.abstractmethod
    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        """
        Returns the names and Pixeltable types that this `Project` expects to see in a data export. The keys
        of the `dict` are the names of data fields in the external store, not Pixeltable columns.

        Returns:
            A `dict` mapping names of external data fields to their expected Pixeltable types.
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
    def validate_columns(
        cls,
        table: Table,
        export_cols: dict[str, ts.ColumnType],
        import_cols: dict[str, ts.ColumnType],
        col_mapping: Optional[dict[str, str]],
    ) -> dict[ColumnHandle, str]:
        """
        Verifies that the specified `col_mapping` is valid. In particular, checks that:
        (i) the keys of `col_mapping` are valid columns of the specified `Table`;
        (ii) the values of `col_mapping` are valid external columns (i.e., they appear in either `export_cols` or
            `import_cols`; and
        (iii) the Pixeltable types of the `col_mapping` keys are consistent with the expected types of the corresponding
            external (import or export) columns.
        If validation fails, an exception will be raised. If validation succeeds, a new mapping will be returned
        in which the Pixeltable column names are resolved to the corresponding `Column` objects.
        TODO: return columns as names or qualified ids
        """
        from pixeltable import exprs

        is_user_specified_col_mapping = col_mapping is not None
        if col_mapping is None:
            col_mapping = {col: col for col in itertools.chain(export_cols.keys(), import_cols.keys())}

        resolved_col_mapping: dict[ColumnHandle, str] = {}

        # Validate names
        t_cols = set(table._get_schema().keys())
        for t_col, ext_col in col_mapping.items():
            if t_col not in t_cols:
                if is_user_specified_col_mapping:
                    raise excs.Error(
                        f'Column name `{t_col}` appears as a key in `col_mapping`, but Table `{table._name}` '
                        'contains no such column.'
                    )
                else:
                    raise excs.Error(
                        f'Column `{t_col}` does not exist in Table `{table._name}`. Either add a column `{t_col}`, '
                        f'or specify a `col_mapping` to associate a different column with '
                        f'the external field `{ext_col}`.'
                    )
            if ext_col not in export_cols and ext_col not in import_cols:
                raise excs.Error(
                    f'Column name `{ext_col}` appears as a value in `col_mapping`, but the external store '
                    f'configuration has no column `{ext_col}`.'
                )
            col_ref = table[t_col]
            assert isinstance(col_ref, exprs.ColumnRef)
            resolved_col_mapping[col_ref.col.handle] = ext_col

        # Validate column specs
        t_col_types = table._get_schema()
        for t_col, ext_col in col_mapping.items():
            t_col_type = t_col_types[t_col]
            if ext_col in export_cols:
                # Validate that the table column can be assigned to the external column
                ext_col_type = export_cols[ext_col]
                if not ext_col_type.is_supertype_of(t_col_type, ignore_nullable=True):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be exported to external column `{ext_col}` '
                        f'(incompatible types; expecting `{ext_col_type}`)'
                    )
            if ext_col in import_cols:
                # Validate that the external column can be assigned to the table column
                if table._tbl_version_path.get_column(t_col).is_computed:
                    raise excs.Error(
                        f'Column `{t_col}` is a computed column, which cannot be populated from an external column'
                    )
                ext_col_type = import_cols[ext_col]
                if not t_col_type.is_supertype_of(ext_col_type, ignore_nullable=True):
                    raise excs.Error(
                        f'Column `{t_col}` cannot be imported from external column `{ext_col}` '
                        f'(incompatible types; expecting `{ext_col_type}`)'
                    )
        return resolved_col_mapping


class MockProject(Project):
    """A project that cannot be synced, used mainly for testing."""

    def __init__(
        self,
        name: str,
        export_cols: dict[str, ts.ColumnType],
        import_cols: dict[str, ts.ColumnType],
        col_mapping: dict[ColumnHandle, str],
        stored_proxies: Optional[dict[ColumnHandle, ColumnHandle]] = None,
    ):
        super().__init__(name, col_mapping, stored_proxies)
        self.export_cols = export_cols
        self.import_cols = import_cols
        self.__is_deleted = False

    @classmethod
    def create(
        cls,
        t: Table,
        name: str,
        export_cols: dict[str, ts.ColumnType],
        import_cols: dict[str, ts.ColumnType],
        col_mapping: Optional[dict[str, str]] = None,
    ) -> 'MockProject':
        col_mapping = cls.validate_columns(t, export_cols, import_cols, col_mapping)
        return cls(name, export_cols, import_cols, col_mapping)

    def get_export_columns(self) -> dict[str, ts.ColumnType]:
        return self.export_cols

    def get_import_columns(self) -> dict[str, ts.ColumnType]:
        return self.import_cols

    def sync(self, t: Table, export_data: bool, import_data: bool) -> UpdateStatus:
        raise NotImplementedError()

    def delete(self) -> None:
        self.__is_deleted = True

    @property
    def is_deleted(self) -> bool:
        return self.__is_deleted

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'export_cols': {k: v.as_dict() for k, v in self.export_cols.items()},
            'import_cols': {k: v.as_dict() for k, v in self.import_cols.items()},
            'col_mapping': [[k.as_dict(), v] for k, v in self.col_mapping.items()],
            'stored_proxies': [[k.as_dict(), v.as_dict()] for k, v in self.stored_proxies.items()],
        }

    @classmethod
    def from_dict(cls, md: dict[str, Any]) -> MockProject:
        return cls(
            md['name'],
            {k: ts.ColumnType.from_dict(v) for k, v in md['export_cols'].items()},
            {k: ts.ColumnType.from_dict(v) for k, v in md['import_cols'].items()},
            {ColumnHandle.from_dict(entry[0]): entry[1] for entry in md['col_mapping']},
            {ColumnHandle.from_dict(entry[0]): ColumnHandle.from_dict(entry[1]) for entry in md['stored_proxies']},
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockProject):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f'MockProject `{self.name}`'
