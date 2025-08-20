from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import UUID

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class MediaStoreBase:
    def count(self, tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Count the number of media objects for a given table ID."""
        raise NotImplementedError

    def copy_local_media_file(self, col: Column, src_path: Path) -> str:
        """Copy a local media file to the media store."""
        raise NotImplementedError

    def download_media_object(self, src_path: str, dest_path: Path) -> None:
        raise NotImplementedError

    def delete(self, tbl_id: UUID, tbl_version: Optional[int] = None) -> Optional[int]:
        """Delete media objects for a given table ID."""
        raise NotImplementedError

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """List media objects for a given table ID."""
        raise NotImplementedError
