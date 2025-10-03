"""
RemoteTable and RemoteDir classes for representing remote objects.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from pixeltable import catalog, exceptions as excs, exprs
from pixeltable.catalog import Dir, Table
from pixeltable.catalog.table_version_path import TableVersionPath
from pixeltable.catalog.table_version_handle import TableVersionHandle
from pixeltable.catalog.column import Column
from pixeltable.catalog.table_version import TableVersion
from pixeltable.type_system import ColumnType
from pixeltable.utils.pydantic_mixin import PydanticSerializationMixin

_logger = logging.getLogger('pixeltable')


class RemoteTableVersion:
    """Remote version of TableVersion that doesn't require local catalog access."""
    
    def __init__(self, remote_metadata: dict[str, Any], handle: TableVersionHandle, name: str = "remote_table"):
        # Initialize with minimal metadata from remote server
        # This avoids the need for local catalog access
        self._remote_metadata = remote_metadata
        self.handle = handle  # Required by ColumnRef
        self.id = handle.id  # Required by ColumnRef._id_attrs()
        self.effective_version = handle.effective_version  # Required by ColumnRef._id_attrs()
        self.name = name  # Required by DataFrame._query_descriptor()
        self.cols_by_name: dict[str, Column] = {}
        self.cols_by_id: dict[int, Column] = {}
        self.cols: list[Column] = []
        self.include_base_columns = True
        self.is_validated = True  # Always consider remote metadata as validated
        self.is_component_view = False  # Default for remote tables
        
        # Parse remote metadata to create columns
        self._init_remote_columns()
    
    def _init_remote_columns(self) -> None:
        """Initialize columns from remote metadata."""
        # TODO: Parse actual remote metadata structure
        # For now, create dummy columns
        from pixeltable.type_system import StringType
        
        class RemoteColumn:
            def __init__(self, name: str, col_type: ColumnType, col_id: int, tbl):
                self.name = name
                self.col_type = col_type
                self.id = col_id
                self.tbl = tbl  # Reference to the table version
                self.is_stored = True  # Default for remote columns
        
        dummy_col = RemoteColumn("dummy_column", StringType(nullable=True), 1, self)
        self.cols_by_name["dummy_column"] = dummy_col
        self.cols_by_id[1] = dummy_col
        self.cols = [dummy_col]
    
    def is_iterator_column(self, col: Column) -> bool:
        """Check if column is an iterator column - default False for remote."""
        return False
    
    def num_rowid_columns(self) -> int:
        """Return number of rowid columns - default 1 for remote."""
        return 1


class RemoteTableVersionPath(TableVersionPath):
    """Remote version of TableVersionPath that doesn't require local catalog access."""
    
    def __init__(self, remote_path: str, remote_metadata: Optional[dict[str, Any]] = None):
        # Create a dummy TableVersionHandle for remote table
        dummy_handle = TableVersionHandle(uuid4(), None)
        super().__init__(dummy_handle)
        
        self._remote_path = remote_path
        self._remote_metadata = remote_metadata or {}
        self._cached_tbl_version: Optional[RemoteTableVersion] = None
    
    def refresh_cached_md(self) -> None:
        """Refresh cached metadata from remote server instead of local catalog."""
        if self._cached_tbl_version is not None:
            return  # Already cached
            
        # Create remote table version from remote metadata
        table_name = self._remote_path.split('/', maxsplit=1)[-1]  # Extract table name from path
        self._cached_tbl_version = RemoteTableVersion(self._remote_metadata, self.tbl_version, table_name)
    
    @property
    def remote_path(self) -> str:
        """Return the remote table path."""
        return self._remote_path


class RemoteTable(PydanticSerializationMixin):
    """
    Represents a remote table that supports column access and DataFrame creation.
    
    This class works exactly like catalog.Table but uses remote metadata instead of local catalog.
    It allows users to:
    - Access columns: remote_t.<column_name>
    - Create DataFrames: remote_t.<column>.head()
    - Run queries on remote tables and their base tables
    """

    def __init__(self, path: str, remote_metadata: Optional[dict[str, Any]] = None):
        """
        Initialize RemoteTable.
        
        Args:
            path: Remote table path (e.g., 'pxt://user/dir.table')
            remote_metadata: Optional metadata from remote server
        """
        self._remote_path = path
        self._name = path.split('/', maxsplit=1)[-1]  # Extract name from path
        self._remote_metadata = remote_metadata or {}
        
        # Create RemoteTableVersionPath that doesn't require local catalog access
        self._tbl_version_path = RemoteTableVersionPath(path, remote_metadata)

    @property
    def path(self) -> str:
        """Return the remote table path."""
        return self._remote_path

    @property
    def name(self) -> str:
        """Return the table name."""
        return self._name

    @property
    def is_remote(self) -> bool:
        """Return True to indicate this is a remote table."""
        return True

    @property
    def tbl_version_path(self) -> RemoteTableVersionPath:
        """Return the remote table version path."""
        return self._tbl_version_path

    def __repr__(self) -> str:
        return f"RemoteTable(path='{self.path}')"

    def __str__(self) -> str:
        return f"RemoteTable(path='{self.path}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RemoteTable):
            return False
        return self.path == other.path

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.path))

    def __getattr__(self, name: str) -> exprs.ColumnRef:
        """Return a ColumnRef for the given column name - exactly like catalog.Table."""
        col = self._tbl_version_path.get_column(name)
        if col is None:
            raise AttributeError(f'Column {name!r} unknown')
        return exprs.ColumnRef(col, reference_tbl=self._tbl_version_path)

    def __getitem__(self, name: str) -> exprs.ColumnRef:
        """Return a ColumnRef for the given column name - exactly like catalog.Table."""
        return getattr(self, name)

    def list_columns(self) -> list[str]:
        """List all column names in the remote table."""
        return [col.name for col in self._tbl_version_path.columns()]

    def get_metadata(self) -> dict[str, Any]:
        """Get basic metadata about the remote table."""
        return {
            'path': self.path,
            'name': self.name,
            'is_remote': True,
            'columns': self.list_columns()
        }

    def as_dict(self) -> dict:
        """Serialize RemoteTable to dictionary."""
        return {
            'path': self._remote_path,
            'name': self._name,
            'remote_metadata': self._remote_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RemoteTable':
        """Deserialize RemoteTable from dictionary."""
        return cls(
            path=d['path'],
            remote_metadata=d.get('remote_metadata')
        )


class RemoteDir(PydanticSerializationMixin):
    """Represents a remote directory that supports remote operations."""

    def __init__(self, path: str):
        """
        Initialize RemoteDir.
        
        Args:
            path: Remote directory path (e.g., 'pxt://user/dir')
        """
        self._remote_path = path
        self._name = path.split('/', maxsplit=1)[-1]  # Extract name from path

    @property
    def path(self) -> str:
        """Return the remote directory path."""
        return self._remote_path

    @property
    def name(self) -> str:
        """Return the directory name."""
        return self._name

    @property
    def is_remote(self) -> bool:
        """Return True to indicate this is a remote directory."""
        return True

    def __repr__(self) -> str:
        return f"RemoteDir(path='{self.path}')"

    def __str__(self) -> str:
        return f"RemoteDir(path='{self.path}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RemoteDir):
            return False
        return self.path == other.path

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.path))

    def as_dict(self) -> dict:
        """Serialize RemoteDir to dictionary."""
        return {
            'path': self._remote_path,
            'name': self._name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RemoteDir':
        """Deserialize RemoteDir from dictionary."""
        return cls(path=d['path'])
