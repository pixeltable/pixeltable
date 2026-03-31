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


class PixeltableStoreOperationType(str, Enum):
    """Operation types for Pixeltable-managed storage (home buckets)."""

    GET_HOME_BUCKET_CREDENTIALS = 'get_home_bucket_credentials'
    GET_PRESIGNED_URL = 'get_presigned_url'


REPLICA_OPERATIONS: frozenset[ReplicaOperationType] = frozenset(ReplicaOperationType)
PIXELTABLE_STORE_OPERATIONS: frozenset[PixeltableStoreOperationType] = frozenset(PixeltableStoreOperationType)
