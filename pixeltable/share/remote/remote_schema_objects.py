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
from pixeltable.type_system import ColumnType
from pixeltable.utils.pydantic_mixin import PydanticSerializationMixin

_logger = logging.getLogger('pixeltable')


class RemoteTable(PydanticSerializationMixin):
    """
    Represents a remote table that supports column access and DataFrame creation.
    
    This class is self-contained and allows users to:
    - Access columns: remote_t.<column_name>
    - Create DataFrames: remote_t.<column>.head()
    - Run queries on remote tables and their base tables
    """

    def __init__(self, path: str, tbl_version_path: Optional[TableVersionPath] = None):
        """
        Initialize RemoteTable.
        
        Args:
            path: Remote table path (e.g., 'pxt://user/dir.table')
            tbl_version_path: Optional TableVersionPath for the remote table
        """
        self._remote_path = path
        self._name = path.split('/', maxsplit=1)[-1]  # Extract name from path
        self._tbl_version_path = tbl_version_path
        
        # Cache for column metadata
        self._columns: dict[str, Column] = {}
        self._columns_loaded = False

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
    def tbl_version_path(self) -> Optional[TableVersionPath]:
        """Return the table version path if available."""
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

    def _load_column_metadata(self) -> None:
        """Load column metadata from remote server."""
        if self._columns_loaded:
            return
            
        # TODO: Implement actual remote metadata fetching
        # For now, create dummy columns for demonstration
        # In a real implementation, this would fetch schema from the remote server
        _logger.warning(f"RemoteTable._load_column_metadata() not implemented for {self.path}")
        
        # Create a minimal column-like object for testing
        # This avoids the complexity of creating a full Column with TableVersion
        from pixeltable.type_system import StringType
        
        class RemoteColumn:
            def __init__(self, name: str, col_type: ColumnType):
                self.name = name
                self.col_type = col_type
                self.id = 1  # Dummy ID
                self.tbl = None  # Will be set later if needed
        
        dummy_col = RemoteColumn("dummy_column", StringType(nullable=True))
        self._columns["dummy_column"] = dummy_col
        self._columns_loaded = True

    def _get_column(self, name: str) -> Optional[Any]:
        """Get column by name, loading metadata if needed."""
        self._load_column_metadata()
        return self._columns.get(name)

    def __getattr__(self, name: str) -> Any:
        """Return a ColumnRef for the given column name."""
        col = self._get_column(name)
        if col is None:
            raise AttributeError(f'Column {name!r} unknown in remote table {self.path}')
        
        # For now, create a simple object that can be used for DataFrame creation
        # TODO: Implement proper ColumnRef creation when remote metadata is available
        class RemoteColumnRef:
            def __init__(self, col, reference_tbl):
                self.col = col
                self.reference_tbl = reference_tbl
                self.col_type = col.col_type
            
            def _df(self) -> Any:
                """Create DataFrame from this remote column reference."""
                import pixeltable as pxt
                from pixeltable import plan
                return pxt.DataFrame(plan.FromClause([self.reference_tbl])).select(self)
            
            def head(self, *args: Any, **kwargs: Any) -> Any:
                """Return first few rows."""
                return self._df().head(*args, **kwargs)
            
            def show(self, *args: Any, **kwargs: Any) -> Any:
                """Show rows."""
                return self._df().show(*args, **kwargs)
            
            def __repr__(self) -> str:
                return f"RemoteColumnRef({self.col.name})"
        
        # Create ColumnRef with the remote table's version path as reference_tbl
        if self._tbl_version_path is not None:
            return RemoteColumnRef(col, self._tbl_version_path)
        else:
            # If no version path, create a dummy one for remote table
            # This allows DataFrame creation even without full metadata
            dummy_handle = TableVersionHandle(uuid4(), None)
            dummy_version_path = TableVersionPath(dummy_handle)
            return RemoteColumnRef(col, dummy_version_path)

    def __getitem__(self, name: str) -> Any:
        """Return a ColumnRef for the given column name."""
        return getattr(self, name)

    def list_columns(self) -> list[str]:
        """List all column names in the remote table."""
        self._load_column_metadata()
        return list(self._columns.keys())

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
            'tbl_version_path': self._tbl_version_path.as_dict() if self._tbl_version_path else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RemoteTable':
        """Deserialize RemoteTable from dictionary."""
        tbl_version_path = None
        if d.get('tbl_version_path'):
            tbl_version_path = TableVersionPath.from_dict(d['tbl_version_path'])
        
        return cls(
            path=d['path'],
            tbl_version_path=tbl_version_path
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
