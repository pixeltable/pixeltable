"""
Replica operation types for pixeltable table replica operations.

This module defines the replica operation types that are shared between
pixeltable core and cloud implementations.
"""

from __future__ import annotations

from enum import Enum


class ReplicaOperationType(str, Enum):
    """Replica operation types for table replica operations."""

    # Table replica operations
    PUBLISH_REPLICA = 'publish_replica'
    FINALIZE_REPLICA = 'finalize_replica'
    CLONE_REPLICA = 'clone_replica'
    DELETE_REPLICA = 'delete_replica'

    def is_replica_operation(self) -> bool:
        """Check if operation is a replica operation."""
        return self in REPLICA_OPERATIONS


# Define the operation sets as module-level constants
REPLICA_OPERATIONS: set[ReplicaOperationType] = {
    ReplicaOperationType.PUBLISH_REPLICA,
    ReplicaOperationType.FINALIZE_REPLICA,
    ReplicaOperationType.CLONE_REPLICA,
    ReplicaOperationType.DELETE_REPLICA,
}
