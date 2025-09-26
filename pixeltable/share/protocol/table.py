"""
Core table protocol for pixeltable operations.

This module contains the core table protocol structures that can be shared
between pixeltable core and cloud implementations.
"""

from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import AnyUrl, BaseModel

from pixeltable.metadata.schema import TableMd, TableSchemaVersionMd, TableVersionMd

from .common import PathUri, PathUriRequestModel, StorageDestination
from .operation_types import CoreOperationType


class TableMetadataEntry(BaseModel):
    """Entry containing table metadata information."""

    table_id: str
    table_md: TableMd
    table_version_md: TableVersionMd
    table_schema_version_md: TableSchemaVersionMd


class ListTableMetadataEntry(BaseModel):
    """List of table metadata entries."""

    tables: list[TableMetadataEntry]


class PublishSnapshotRequest(PathUriRequestModel):
    """Request to publish or push table snapshot."""

    operation_type: Literal[CoreOperationType.PUBLISH_SNAPSHOT] = CoreOperationType.PUBLISH_SNAPSHOT
    table_uri: PathUri  # If PathUri#is_uuid is true then its considered a push replica request
    pxt_version: str
    pxt_md_version: int
    md: ListTableMetadataEntry
    is_public: bool = False
    bucket_name: Optional[str] = None  # Optional bucket name, falls back to org's default bucket if not provided


class PublishSnapshotResponse(BaseModel):
    """Response from publishing a table snapshot."""

    upload_id: UUID
    destination: StorageDestination
    destination_uri: AnyUrl
    max_size: Optional[int] = None  # Maximum size that can be used by this snapshot, used for R2 home buckets


class FinalizeSnapshotRequest(PathUriRequestModel):
    """Request to finalize a table snapshot."""

    operation_type: Literal[CoreOperationType.FINALIZE_SNAPSHOT] = CoreOperationType.FINALIZE_SNAPSHOT
    table_uri: PathUri  # Use same table_uri that was given during publish snapshot request
    upload_id: UUID
    size: int
    sha256: str
    datafile: str
    rows: int
    preview_header: dict[str, str]
    preview_data: list[list[Any]]


class FinalizeSnapshotResponse(BaseModel):
    """Response from finalizing a table snapshot."""

    confirmed_table_uri: PathUri
    version: Optional[int] = None  # Version that was pushed to replica


class DeleteSnapshotRequest(PathUriRequestModel):
    """Request to delete a table replica."""

    operation_type: Literal[CoreOperationType.DELETE_SNAPSHOT] = CoreOperationType.DELETE_SNAPSHOT
    table_uri: PathUri
    version: Optional[int] = None  # Delete a version in replica


class DeleteSnapshotResponse(BaseModel):
    """Response from deleting a table snapshot."""

    table_uri: PathUri
    version: Optional[int] = None


class CloneSnapshotRequest(PathUriRequestModel):
    """Request to clone a table snapshot."""

    operation_type: Literal[CoreOperationType.CLONE_SNAPSHOT] = CoreOperationType.CLONE_SNAPSHOT
    table_uri: PathUri
    version: Optional[int] = None  # Clone a version in replica


class CloneSnapshotResponse(BaseModel):
    """Response from cloning a table snapshot."""

    table_uri: PathUri
    pxt_md_version: int
    destination: StorageDestination
    destination_uri: AnyUrl
    md: ListTableMetadataEntry
    version: Optional[int] = None
