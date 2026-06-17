"""
Product-side counterpart of the fault injection framework. See tests/fault_injection.py for more info.
"""

from enum import Enum, auto
from typing import Any


class FaultLocation(Enum):
    """Instrumented locations in the codebase where faults can be injected."""

    CATALOG_CREATE_USER_AFTER_EXISTS_CHECK = auto()
    CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED = auto()
    CATALOG_FINALIZE_PENDING_OPS_NON_XACT = auto()
    CATALOG_LOAD_TBL_VERSION_BEFORE_INIT = auto()
    CATALOG_LOAD_VIEW_OP_EXEC = auto()
    IO_EXPORT_ICEBERG_WRITE_TEMP = auto()
    IO_EXPORT_ICEBERG_BEFORE_SWAP = auto()
    IO_EXPORT_ICEBERG_APPEND = auto()
    SCHEDULER_RATE_LIMITS_AEXEC = auto()
    TEST = auto()


def create_fault_manager() -> Any:
    """No op. The actual implementation is monkey-patched in tests only."""
    return None


def process_fault(loc: FaultLocation) -> None:
    """No op. The actual implementation is monkey-patched in tests only."""
    pass
