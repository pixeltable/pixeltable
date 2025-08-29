from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import UUID

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class ObjectStoreBase:
    def validate(self, error_col_name: str) -> Optional[str]:
        """Check the store configuration. Returns base URI if store is accessible.

        Args:
            error_col_name: a string of the form 'Column {name}: ' used when raising errors

        Returns:
            Base URI for the store. This value is stored in any Column attached to the store.
        """
        raise AssertionError

    def copy_local_file(self, col: Column, src_path: Path) -> str:
        """Copy a file associated with a Column to the store, returning the file's URL within the destination.

        Args:
            col: The Column to which the file belongs, used to generate the URI of the stored object.
            src_path: The Path to the local file

        Returns:
            The URI of the object in the store
        """
        raise AssertionError

    def move_local_file(self, col: Column, src_path: Path) -> Optional[str]:
        """Move a file associated with a Column to the store, returning the file's URL within the destination.

        Args:
            col: The Column to which the file belongs, used to generate the URI of the stored object.
            src_path: The Path to the local file

        Returns:
            The URI of the object in the store, None if the object cannot be moved to the store
        """
        return None

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        """Copies an object from the store to a local media file.

        Args:
            src_path: The URI of the object in the store
            dest_path: The desired Path to the local file
        """
        raise AssertionError

    def count(self, tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the number of objects in the store associated with the given tbl_id

        Args:
            tbl_id: Only count objects associated with a given table
            tbl_version: Only count objects associated with a specific table version

        Returns:
            Number of objects found with the specified criteria
        """
        raise AssertionError

    def delete(self, tbl_id: UUID, tbl_version: Optional[int] = None) -> Optional[int]:
        """Delete objects in the destination for a given table ID, table version.

        Args:
            tbl_id: Only count objects associated with a given table
            tbl_version: Only count objects associated with a specific table version

        Returns:
            Number of objects deleted or None if the store does not count deletions.
        """
        raise AssertionError

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects in the store.

        Args:
            return_uri: If True, returns a full URI for each object, otherwise just the path to the object.
            n_max: Maximum number of objects to list
        """
        raise AssertionError
