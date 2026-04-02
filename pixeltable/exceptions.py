"""Pixeltable exception hierarchy.

Exception classes carry structured error information (error_code, retryable, cause) for
programmatic error handling. Each exception class defines an ``http_status`` attribute for
use by REST integration layers (e.g., FastAPI).

HTTP Status Code Mapping
========================

Exception Class           HTTP  Error Codes
------------------------  ----  -------------------------------------------
Error                     500   INTERNAL_ERROR, MISSING_CREDENTIALS,
                                INVALID_CONFIGURATION
NotFoundError             404   COLUMN_NOT_FOUND, PATH_NOT_FOUND,
                                DIRECTORY_NOT_FOUND, INDEX_NOT_FOUND,
                                FUNCTION_NOT_FOUND, ROW_NOT_FOUND,
                                STORAGE_NOT_FOUND
AlreadyExistsError        409   COLUMN_ALREADY_EXISTS, PATH_ALREADY_EXISTS,
                                INDEX_ALREADY_EXISTS, FUNCTION_ALREADY_EXISTS
SchemaError               422   INVALID_COLUMN_NAME, INVALID_PATH,
                                INVALID_EXPRESSION, INVALID_TYPE,
                                INVALID_SCHEMA, INVALID_ARGUMENT,
                                INVALID_DATA_FORMAT, MISSING_REQUIRED,
                                TYPE_MISMATCH, CONSTRAINT_VIOLATION
OperationError            400   UNSUPPORTED_OPERATION, INVALID_STATE,
                                IMMUTABLE
AuthorizationError        403   INSUFFICIENT_PRIVILEGES
ExternalServiceError      502   PROVIDER_ERROR
RateLimitError            429   RATE_LIMITED
ServiceUnavailableError   503   DATABASE_UNAVAILABLE, STORE_UNAVAILABLE
ConcurrencyError          409   SERIALIZATION_FAILURE,
                                CONCURRENT_MODIFICATION
CancellationError         499   CANCELLED

Error Code Ranges
=================

Each error code group is assigned a stable numeric range with room for future
additions.  Ranges start at multiples of 1000 to avoid any confusion with HTTP
status codes.

    0xxx  — General / internal
    1xxx  — Not found
    2xxx  — Already exists
    3xxx  — Schema / validation
    4xxx  — Operation
    5xxx  — Authorization
    6xxx  — External service
    7xxx  — Service unavailable
    8xxx  — Concurrency
    9xxx  — Configuration
    10xxx — Cancellation
"""

from __future__ import annotations

import enum
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable import exprs


class ErrorCode(enum.Enum):
    """Machine-readable error codes for Pixeltable exceptions.

    Codes are assigned explicit stable values in grouped ranges.  New codes
    MUST be added at the end of their group and MUST NOT reuse or shift
    existing values.
    """

    # General (0xxx)
    INTERNAL_ERROR = 0

    # NotFoundError (1xxx)
    COLUMN_NOT_FOUND = 1000
    PATH_NOT_FOUND = 1001
    DIRECTORY_NOT_FOUND = 1002
    INDEX_NOT_FOUND = 1003
    FUNCTION_NOT_FOUND = 1004
    ROW_NOT_FOUND = 1005
    STORAGE_NOT_FOUND = 1006

    # AlreadyExistsError (2xxx)
    COLUMN_ALREADY_EXISTS = 2000
    PATH_ALREADY_EXISTS = 2001
    INDEX_ALREADY_EXISTS = 2002
    FUNCTION_ALREADY_EXISTS = 2003

    # SchemaError (3xxx)
    INVALID_COLUMN_NAME = 3000
    INVALID_PATH = 3001
    INVALID_EXPRESSION = 3002
    INVALID_TYPE = 3003
    INVALID_SCHEMA = 3004
    INVALID_ARGUMENT = 3005
    INVALID_DATA_FORMAT = 3006
    MISSING_REQUIRED = 3007
    TYPE_MISMATCH = 3008
    CONSTRAINT_VIOLATION = 3009

    # OperationError (4xxx)
    UNSUPPORTED_OPERATION = 4000
    INVALID_STATE = 4001
    IMMUTABLE = 4002

    # AuthorizationError (5xxx)
    INSUFFICIENT_PRIVILEGES = 5000

    # ExternalServiceError (6xxx)
    PROVIDER_ERROR = 6000
    RATE_LIMITED = 6001

    # ServiceUnavailableError (7xxx)
    DATABASE_UNAVAILABLE = 7000
    STORE_UNAVAILABLE = 7001

    # ConcurrencyError (8xxx)
    SERIALIZATION_FAILURE = 8000
    CONCURRENT_MODIFICATION = 8001

    # Configuration (9xxx) — used with base Error
    MISSING_CREDENTIALS = 9000
    INVALID_CONFIGURATION = 9001

    # CancellationError (10xxx)
    CANCELLED = 10000


class Error(Exception):
    """Base exception for user-facing Pixeltable errors."""

    http_status: int = 500
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    retryable: bool = False
    cause: Exception | None = None

    def __init__(
        self,
        message: str = '',
        *args: Any,
        error_code: ErrorCode | None = None,
        retryable: bool | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, *args)
        if error_code is not None:
            self.error_code = error_code
        if retryable is not None:
            self.retryable = retryable
        if cause is not None:
            self.cause = cause
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for REST error responses."""
        return {'error_code': self.error_code.name, 'message': str(self), 'retryable': self.retryable}


class NotFoundError(Error):
    """Resource not found."""

    http_status: int = 404
    error_code: ErrorCode = ErrorCode.PATH_NOT_FOUND


class AlreadyExistsError(Error):
    """Resource already exists."""

    http_status: int = 409
    error_code: ErrorCode = ErrorCode.PATH_ALREADY_EXISTS


class SchemaError(Error):
    """Invalid input, schema, or argument."""

    http_status: int = 422
    error_code: ErrorCode = ErrorCode.INVALID_ARGUMENT


class OperationError(Error):
    """Operation not supported or not valid for the current target/state."""

    http_status: int = 400
    error_code: ErrorCode = ErrorCode.UNSUPPORTED_OPERATION


class AuthorizationError(Error):
    """Caller lacks permission for the requested operation."""

    http_status: int = 403
    error_code: ErrorCode = ErrorCode.INSUFFICIENT_PRIVILEGES


class ExternalServiceError(Error):
    """An upstream provider or external store returned an error."""

    http_status: int = 502
    error_code: ErrorCode = ErrorCode.PROVIDER_ERROR
    provider: str | None = None
    status_code: int | None = None
    retry_after: float | None = None

    def __init__(
        self,
        message: str = '',
        *args: Any,
        error_code: ErrorCode | None = None,
        retryable: bool | None = None,
        cause: Exception | None = None,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, *args, error_code=error_code, retryable=retryable, cause=cause)
        if provider is not None:
            self.provider = provider
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.provider is not None:
            d['provider'] = self.provider
        if self.status_code is not None:
            d['status_code'] = self.status_code
        return d



class ServiceUnavailableError(Error):
    """Database, store, or other infrastructure is unreachable."""

    http_status: int = 503
    error_code: ErrorCode = ErrorCode.DATABASE_UNAVAILABLE
    retryable: bool = True


class ConcurrencyError(Error):
    """Serialization failure, deadlock, or concurrent modification conflict."""

    http_status: int = 409
    error_code: ErrorCode = ErrorCode.SERIALIZATION_FAILURE
    retryable: bool = True


class ExprEvalError(Exception):
    """
    Used during query execution to signal expr evaluation failures.

    NOT A USER-FACING EXCEPTION. All ExprEvalError instances need to be converted into Error instances.
    """

    expr: 'exprs.Expr'
    expr_msg: str
    exc: Exception
    exc_tb: TracebackType
    input_vals: list[Any]
    row_num: int

    def __init__(
        self,
        expr: 'exprs.Expr',
        expr_msg: str,
        exc: Exception,
        exc_tb: TracebackType,
        input_vals: list[Any],
        row_num: int,
    ) -> None:
        exct = type(exc)
        super().__init__(
            f'Expression evaluation failed with an error of type `{exct.__module__}.{exct.__qualname__}`:\n{expr}'
        )
        self.expr = expr
        self.expr_msg = expr_msg
        self.exc = exc
        self.exc_tb = exc_tb
        self.input_vals = input_vals
        self.row_num = row_num


class PixeltableWarning(Warning):
    pass


class PixeltableDeprecationWarning(DeprecationWarning, PixeltableWarning):
    pass
