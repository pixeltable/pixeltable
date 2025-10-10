"""
Pixeltable Core Protocol

This module contains the core protocol structures for pixeltable table operations
that can be shared between pixeltable core and cloud implementations.
"""

from .common import PxtUri, RequestBaseModel, StorageDestination
from .operation_types import ReplicaOperationType
from .replica import (
    DeleteRequest,
    DeleteResponse,
    FinalizeRequest,
    FinalizeResponse,
    PublishRequest,
    PublishResponse,
    ReplicateRequest,
    ReplicateResponse,
)

__all__ = [
    'DeleteRequest',
    'DeleteResponse',
    'FinalizeRequest',
    'FinalizeResponse',
    'PublishRequest',
    'PublishResponse',
    'PxtUri',
    'ReplicaOperationType',
    'ReplicateRequest',
    'ReplicateResponse',
    'RequestBaseModel',
    'StorageDestination',
]
