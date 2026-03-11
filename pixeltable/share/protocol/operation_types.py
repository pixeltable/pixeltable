"""
Operation types for pixeltable shared protocol.

This module defines operation types that are shared between
pixeltable core and cloud implementations.
"""

from __future__ import annotations

from enum import Enum


class ReplicaOperationType(str, Enum):
    """Replica operation types for table replica operations."""

    PUBLISH_REPLICA = 'publish_replica'
    FINALIZE_REPLICA = 'finalize_replica'
    CLONE_REPLICA = 'clone_replica'
    DELETE_REPLICA = 'delete_replica'

    def is_replica_operation(self) -> bool:
        """Check if operation is a replica operation."""
        return self in REPLICA_OPERATIONS


class PixeltableStoreOperationType(str, Enum):
    """Operation types for Pixeltable-managed storage (home buckets)."""

    GET_HOME_BUCKET_CREDENTIALS = 'get_home_bucket_credentials'


REPLICA_OPERATIONS: set[ReplicaOperationType] = {
    ReplicaOperationType.PUBLISH_REPLICA,
    ReplicaOperationType.FINALIZE_REPLICA,
    ReplicaOperationType.CLONE_REPLICA,
    ReplicaOperationType.DELETE_REPLICA,
}

PIXELTABLE_STORE_OPERATIONS: set[PixeltableStoreOperationType] = {
    PixeltableStoreOperationType.GET_HOME_BUCKET_CREDENTIALS,
}
