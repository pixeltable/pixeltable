"""
Core table protocol for pixeltable operations.

This module contains the core table protocol structures that can be shared
between pixeltable core and cloud implementations.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import AnyUrl, BaseModel, field_validator

from pixeltable.catalog.table_version import TableVersionMd
from pixeltable.utils.object_stores import StorageTarget

from .common import PxtUri, RequestBaseModel
from .operation_types import ReplicaOperationType


class PublishRequest(RequestBaseModel):
    """Request to publish or push table replica."""

    operation_type: Literal[ReplicaOperationType.PUBLISH_REPLICA] = ReplicaOperationType.PUBLISH_REPLICA
    table_uri: PxtUri  # If PxtUri#id is not None then it's considered a push replica request
    pxt_version: str
    pxt_md_version: int
    md: list[TableVersionMd]
    is_public: bool = False
    bucket_name: str | None = None  # Optional bucket name, falls back to org's default bucket if not provided

    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request."""
        return self.table_uri

    @field_validator('md')
    @classmethod
    def validate_md_not_empty(cls, v: list[TableVersionMd]) -> list[TableVersionMd]:
        """Ensure md list has at least one element."""
        if len(v) == 0:
            raise ValueError('md list must contain at least one element')
        return v


class PublishResponse(BaseModel):
    """Response from publishing a table replica."""

    upload_id: UUID | None = None
    destination: StorageTarget | None = None
    destination_uri: AnyUrl | None = None
    max_size: int | None = None  # Maximum size that can be used by this replica, used for R2 home buckets
    table_uri: PxtUri | None = None  # If replica already exists, return the uri for it


class FinalizeRequest(RequestBaseModel):
    """Request to finalize a table replica."""

    operation_type: Literal[ReplicaOperationType.FINALIZE_REPLICA] = ReplicaOperationType.FINALIZE_REPLICA
    table_uri: PxtUri  # Use same table_uri that was given during publish replica request
    upload_id: UUID
    size: int
    sha256: str
    datafile: str
    row_count: int
    preview_header: dict[str, str]
    preview_data: list[list[Any]]

    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request."""
        return self.table_uri


class FinalizeResponse(BaseModel):
    """Response from finalizing a table replica."""

    confirmed_table_uri: PxtUri
    version: int | None = None  # Version that was pushed to replica


class DeleteRequest(RequestBaseModel):
    """Request to delete a table replica."""

    operation_type: Literal[ReplicaOperationType.DELETE_REPLICA] = ReplicaOperationType.DELETE_REPLICA
    table_uri: PxtUri
    version: int | None = None  # Delete a version in replica

    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request."""
        return self.table_uri


class DeleteResponse(BaseModel):
    """Response from deleting a table replica."""

    table_uri: PxtUri
    version: int | None = None


class ReplicateRequest(RequestBaseModel):
    """Request to clone a table replica."""

    operation_type: Literal[ReplicaOperationType.CLONE_REPLICA] = ReplicaOperationType.CLONE_REPLICA
    table_uri: PxtUri

    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request."""
        return self.table_uri


class ReplicateResponse(BaseModel):
    """Response from cloning a table replica."""

    table_uri: PxtUri
    pxt_md_version: int
    destination: StorageTarget
    destination_uri: AnyUrl
    md: list[TableVersionMd]
    version: int | None = None
