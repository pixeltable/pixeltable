"""
RemoteTable and RemoteDir classes for representing remote objects.
"""

from typing import Any, Optional

from pixeltable.catalog import Dir, Table


class RemoteTable(Table):
    """Represents a remote table that supports remote operations."""

    def __init__(self, path: str):
        # Initialize with minimal required attributes
        self._remote_path = path
        self._name = path.split('/', 1)[-1]  # Extract name from path
        self._schema: dict[str, Any] = {}

    @property
    def path(self) -> str:
        return self._remote_path

    @property
    def is_remote(self) -> bool:
        """Return True to indicate this is a remote table."""
        return True

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

    # Override all Table methods to raise "unsupported" errors
    def select(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def insert(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def create_view(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def create_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def add_column(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def drop_column(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def rename_column(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def reorder_columns(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def add_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def drop_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def collect(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def show(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    def count(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Table operations are not supported on remote tables')

    # Implement abstract methods from Table
    def _display_name(self) -> str:
        """Return the display name for this remote table."""
        return f'RemoteTable({self.path})'

    def _effective_base_versions(self) -> list[Optional[int]]:  # type: ignore[override]
        """Return base versions - not applicable for remote tables."""
        return []

    def _get_base_table(self) -> 'Table':
        """Return base table - not applicable for remote tables."""
        raise NotImplementedError('Base table access not supported on remote tables')


class RemoteDir(Dir):
    """Represents a remote directory that supports remote operations."""

    def __init__(self, path: str):
        # Initialize with minimal required attributes
        self._remote_path = path
        self._name = path.split('/', 1)[-1]  # Extract name from path

    @property
    def path(self) -> str:
        return self._remote_path

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
