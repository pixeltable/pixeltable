from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse
from uuid import UUID

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class MediaPath:
    PATTERN = re.compile(r'([0-9a-fA-F]+)_(\d+)_(\d+)_([0-9a-fA-F]+)')  # tbl_id, col_id, version, uuid

    @classmethod
    def media_table_prefix(cls, tbl_id: UUID) -> str:
        """Construct a unique unix-style prefix for a media table without leading/trailing slashes."""
        return f'{tbl_id.hex}'

    @classmethod
    def media_prefix_file_raw(
        cls, tbl_id: UUID, col_id: int, tbl_version: int, ext: Optional[str] = None
    ) -> tuple[str, str]:
        """Construct a unique unix-style prefix and filename for a persisted media file.
        The results are derived from table, col, and version specs.
        Returns:
            prefix: a unix-style prefix for the media file without leading/trailing slashes
            filename: a unique filename for the media file without leading slashes
        """
        table_prefix = cls.media_table_prefix(tbl_id)
        media_id_hex = uuid.uuid4().hex
        prefix = f'{table_prefix}/{media_id_hex[:2]}/{media_id_hex[:4]}'
        filename = f'{table_prefix}_{col_id}_{tbl_version}_{media_id_hex}{ext or ""}'
        return prefix, filename

    @classmethod
    def media_prefix_file(cls, col: Column, ext: Optional[str] = None) -> tuple[str, str]:
        """Construct a unique unix-style prefix and filename for a persisted media file.
        The results are derived from a Column specs.
        Returns:
            prefix: a unix-style prefix for the media file without leading/trailing slashes
            filename: a unique filename for the media file without leading slashes
        """
        assert col.tbl is not None, 'Column must be associated with a table'
        return cls.media_prefix_file_raw(col.tbl.id, col.id, col.tbl.version, ext=ext)

    @classmethod
    def parse_cloud_storage_uri(cls, uri: str) -> dict[str, Optional[str]]:
        """
        Parses a cloud storage URI into its scheme, bucket, prefix, and object name.

        Args:
            uri (str): The cloud storage URI (e.g., "gs://my-bucket/path/to/object.txt").

        Returns:
            dict: A dictionary containing 'scheme', 'bucket', 'prefix', and 'object_name'.
                Returns None for prefix and object_name if not applicable.
        """
        parsed_uri = urlparse(uri)

        scheme = parsed_uri.scheme
        bucket = parsed_uri.netloc
        path = parsed_uri.path.lstrip('/')  # Remove leading slash

        prefix = None
        object_name = None

        if path:
            if '/' in path:
                # If there are slashes in the path, it indicates a prefix and object name
                prefix_parts = path.rsplit('/', 1)
                prefix = prefix_parts[0]
                object_name = prefix_parts[1]
            else:
                # If no slashes, the entire path is the object name
                object_name = path

        return {'scheme': scheme, 'bucket': bucket, 'prefix': prefix, 'object_name': object_name}
