"""
Core operation types for pixeltable table and namespace operations.

This module defines the core operation types that are shared between
pixeltable core and cloud implementations.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class CoreOperationType(str, Enum):
    """Core operation types for table and namespace operations."""

    # Table operations
    PUBLISH_SNAPSHOT = 'publish_snapshot'
    FINALIZE_SNAPSHOT = 'finalize_snapshot'
    CLONE_SNAPSHOT = 'clone_snapshot'
    DELETE_SNAPSHOT = 'delete_snapshot'

    # Catalog/namespace operations
    LIST_ALL = 'list_all'
    CREATE_DIR = 'create_dir'
    GET_TABLE = 'get_table'
    SET_TABLE_PUBLIC = 'set_table_public'

    def is_table_operation(self) -> bool:
        """Check if operation is a table operation."""
        return self in TABLE_OPERATIONS

    def is_namespace_operation(self) -> bool:
        """Check if operation is a namespace operation."""
        return self in NAMESPACE_OPERATIONS

    def is_worker_operation(self) -> bool:
        """Check if operation should be executed in worker (table or namespace operations)."""
        return self in WORKER_OPERATIONS


# Define the operation sets as module-level constants
TABLE_OPERATIONS: Set[CoreOperationType] = {
    CoreOperationType.PUBLISH_SNAPSHOT,
    CoreOperationType.FINALIZE_SNAPSHOT,
    CoreOperationType.CLONE_SNAPSHOT,
    CoreOperationType.DELETE_SNAPSHOT,
}

NAMESPACE_OPERATIONS: Set[CoreOperationType] = {
    CoreOperationType.LIST_ALL,
    CoreOperationType.CREATE_DIR,
    CoreOperationType.GET_TABLE,
    CoreOperationType.SET_TABLE_PUBLIC,
}

# Worker operations (both table and namespace operations)
WORKER_OPERATIONS: Set[CoreOperationType] = TABLE_OPERATIONS | NAMESPACE_OPERATIONS
