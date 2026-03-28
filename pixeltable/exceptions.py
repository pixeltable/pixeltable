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
"""

from __future__ import annotations

import enum
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable import exprs


class ErrorCode(enum.Enum):
    """Machine-readable error codes for Pixeltable exceptions."""

    INTERNAL_ERROR = enum.auto()

    # NotFoundError codes
    COLUMN_NOT_FOUND = enum.auto()
    PATH_NOT_FOUND = enum.auto()
    DIRECTORY_NOT_FOUND = enum.auto()
    INDEX_NOT_FOUND = enum.auto()
    FUNCTION_NOT_FOUND = enum.auto()
    ROW_NOT_FOUND = enum.auto()
    STORAGE_NOT_FOUND = enum.auto()

    # AlreadyExistsError codes
    COLUMN_ALREADY_EXISTS = enum.auto()
    PATH_ALREADY_EXISTS = enum.auto()
    INDEX_ALREADY_EXISTS = enum.auto()
    FUNCTION_ALREADY_EXISTS = enum.auto()

    # SchemaError codes
    INVALID_COLUMN_NAME = enum.auto()
    INVALID_PATH = enum.auto()
    INVALID_EXPRESSION = enum.auto()
    INVALID_TYPE = enum.auto()
    INVALID_SCHEMA = enum.auto()
    INVALID_ARGUMENT = enum.auto()
    INVALID_DATA_FORMAT = enum.auto()
    MISSING_REQUIRED = enum.auto()
    TYPE_MISMATCH = enum.auto()
    CONSTRAINT_VIOLATION = enum.auto()

    # OperationError codes
    UNSUPPORTED_OPERATION = enum.auto()
    INVALID_STATE = enum.auto()
    IMMUTABLE = enum.auto()

    # AuthorizationError codes
    INSUFFICIENT_PRIVILEGES = enum.auto()

    # ExternalServiceError codes
    PROVIDER_ERROR = enum.auto()
    RATE_LIMITED = enum.auto()

    # ServiceUnavailableError codes
    DATABASE_UNAVAILABLE = enum.auto()
    STORE_UNAVAILABLE = enum.auto()

    # ConcurrencyError codes
    SERIALIZATION_FAILURE = enum.auto()
    CONCURRENT_MODIFICATION = enum.auto()

    # Configuration codes (used with base Error)
    MISSING_CREDENTIALS = enum.auto()
    INVALID_CONFIGURATION = enum.auto()

    # CancellationError codes
    CANCELLED = enum.auto()


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
        return {
            'error_code': self.error_code.name,
            'message': str(self),
            'retryable': self.retryable,
        }


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


class RateLimitError(ExternalServiceError):
    """Rate-limited by an upstream provider. Always retryable."""

    http_status: int = 429
    error_code: ErrorCode = ErrorCode.RATE_LIMITED
    retryable: bool = True
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
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            *args,
            error_code=error_code,
            retryable=retryable,
            cause=cause,
            provider=provider,
            status_code=status_code,
        )
        if retry_after is not None:
            self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.retry_after is not None:
            d['retry_after'] = self.retry_after
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


class CancellationError(Error):
    """Operation was cancelled."""

    http_status: int = 499
    error_code: ErrorCode = ErrorCode.CANCELLED


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
