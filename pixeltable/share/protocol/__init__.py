"""
Pixeltable Core Protocol

This module contains the core protocol structures for pixeltable table operations
that can be shared between pixeltable core and cloud implementations.
"""

from .common import PathUri, PathUriRequestModel, StorageDestination
from .operation_types import CoreOperationType
from .table import (
    CloneSnapshotRequest,
    CloneSnapshotResponse,
    DeleteSnapshotRequest,
    DeleteSnapshotResponse,
    FinalizeSnapshotRequest,
    FinalizeSnapshotResponse,
    ListTableMetadataEntry,
    PublishSnapshotRequest,
    PublishSnapshotResponse,
    TableMetadataEntry,
)

__all__ = [
    'CloneSnapshotRequest',
    'CloneSnapshotResponse',
    'CoreOperationType',
    'DeleteSnapshotRequest',
    'DeleteSnapshotResponse',
    'FinalizeSnapshotRequest',
    'FinalizeSnapshotResponse',
    'ListTableMetadataEntry',
    'PathUri',
    'PathUriRequestModel',
    'PublishSnapshotRequest',
    'PublishSnapshotResponse',
    'StorageDestination',
    'TableMetadataEntry',
]
