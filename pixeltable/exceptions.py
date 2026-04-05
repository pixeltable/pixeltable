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
                          499   CANCELLED
NotFoundError             404   COLUMN_NOT_FOUND, PATH_NOT_FOUND,
                                DIRECTORY_NOT_FOUND, INDEX_NOT_FOUND,
                                FUNCTION_NOT_FOUND, ROW_NOT_FOUND,
                                STORAGE_NOT_FOUND
AlreadyExistsError        409   COLUMN_ALREADY_EXISTS, PATH_ALREADY_EXISTS,
                                INDEX_ALREADY_EXISTS, FUNCTION_ALREADY_EXISTS
RequestError              422   INVALID_COLUMN_NAME, INVALID_PATH,
                                INVALID_EXPRESSION, INVALID_TYPE,
                                INVALID_SCHEMA, INVALID_ARGUMENT,
                                INVALID_DATA_FORMAT, MISSING_REQUIRED,
                                TYPE_MISMATCH, CONSTRAINT_VIOLATION
                          400   UNSUPPORTED_OPERATION, INVALID_STATE,
                                IMMUTABLE
AuthorizationError        403   INSUFFICIENT_PRIVILEGES
ExternalServiceError      502   PROVIDER_ERROR
                          429   RATE_LIMITED
ServiceUnavailableError   503   DATABASE_UNAVAILABLE, STORE_UNAVAILABLE
ConcurrencyError          409   SERIALIZATION_FAILURE,
                                CONCURRENT_MODIFICATION

Error Code Ranges
=================

Each error code group is assigned a stable numeric range with room for future
additions.  Ranges start at multiples of 1000 to avoid any confusion with HTTP
status codes.

    0xxx  - Error (general / internal / configuration)
    1xxx  - NotFoundError
    2xxx  - AlreadyExistsError
    3xxx  - RequestError
    4xxx  - AuthorizationError
    5xxx  - ExternalServiceError
    6xxx  - ServiceUnavailableError
    7xxx  - ConcurrencyError
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

    # Error (0xxx)
    INTERNAL_ERROR = 0
    MISSING_CREDENTIALS = 1
    INVALID_CONFIGURATION = 2
    CANCELLED = 3

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

    # RequestError (3xxx)
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
    UNSUPPORTED_OPERATION = 3010
    INVALID_STATE = 3011
    IMMUTABLE = 3012

    # AuthorizationError (4xxx)
    INSUFFICIENT_PRIVILEGES = 4000

    # ExternalServiceError (5xxx)
    PROVIDER_ERROR = 5000
    RATE_LIMITED = 5001

    # ServiceUnavailableError (6xxx)
    DATABASE_UNAVAILABLE = 6000
    STORE_UNAVAILABLE = 6001

    # ConcurrencyError (7xxx)
    SERIALIZATION_FAILURE = 7000
    CONCURRENT_MODIFICATION = 7001


# Canonical mapping from every ErrorCode to its HTTP status code.
HTTP_STATUS: dict[ErrorCode, int] = {
    # Error (0xxx)
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.MISSING_CREDENTIALS: 500,
    ErrorCode.INVALID_CONFIGURATION: 500,
    ErrorCode.CANCELLED: 499,
    # NotFoundError (1xxx)
    ErrorCode.COLUMN_NOT_FOUND: 404,
    ErrorCode.PATH_NOT_FOUND: 404,
    ErrorCode.DIRECTORY_NOT_FOUND: 404,
    ErrorCode.INDEX_NOT_FOUND: 404,
    ErrorCode.FUNCTION_NOT_FOUND: 404,
    ErrorCode.ROW_NOT_FOUND: 404,
    ErrorCode.STORAGE_NOT_FOUND: 404,
    # AlreadyExistsError (2xxx)
    ErrorCode.COLUMN_ALREADY_EXISTS: 409,
    ErrorCode.PATH_ALREADY_EXISTS: 409,
    ErrorCode.INDEX_ALREADY_EXISTS: 409,
    ErrorCode.FUNCTION_ALREADY_EXISTS: 409,
    # RequestError (3xxx)
    ErrorCode.INVALID_COLUMN_NAME: 422,
    ErrorCode.INVALID_PATH: 422,
    ErrorCode.INVALID_EXPRESSION: 422,
    ErrorCode.INVALID_TYPE: 422,
    ErrorCode.INVALID_SCHEMA: 422,
    ErrorCode.INVALID_ARGUMENT: 422,
    ErrorCode.INVALID_DATA_FORMAT: 422,
    ErrorCode.MISSING_REQUIRED: 422,
    ErrorCode.TYPE_MISMATCH: 422,
    ErrorCode.CONSTRAINT_VIOLATION: 422,
    ErrorCode.UNSUPPORTED_OPERATION: 400,
    ErrorCode.INVALID_STATE: 400,
    ErrorCode.IMMUTABLE: 400,
    # AuthorizationError (4xxx)
    ErrorCode.INSUFFICIENT_PRIVILEGES: 403,
    # ExternalServiceError (5xxx)
    ErrorCode.PROVIDER_ERROR: 502,
    ErrorCode.RATE_LIMITED: 429,
    # ServiceUnavailableError (6xxx)
    ErrorCode.DATABASE_UNAVAILABLE: 503,
    ErrorCode.STORE_UNAVAILABLE: 503,
    # ConcurrencyError (7xxx)
    ErrorCode.SERIALIZATION_FAILURE: 409,
    ErrorCode.CONCURRENT_MODIFICATION: 409,
}


class Error(Exception):
    """Base exception for user-facing Pixeltable errors.

    http_status is derived automatically from the error code via
    HTTP_STATUS_BY_ERROR_CODE.  Subclasses set a default ``error_code``
    class attribute; callers can override per-instance.
    """

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    retryable: bool = False
    http_status: int

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
            self.__cause__ = cause
        self.http_status = HTTP_STATUS[self.error_code]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for REST error responses."""
        d: dict[str, Any] = {'error_code': self.error_code.name, 'message': str(self), 'retryable': self.retryable}
        if self.__cause__ is not None:
            d['cause'] = str(self.__cause__)
        return d


class NotFoundError(Error):
    """Resource not found."""

    error_code: ErrorCode = ErrorCode.PATH_NOT_FOUND


class AlreadyExistsError(Error):
    """Resource already exists."""

    error_code: ErrorCode = ErrorCode.PATH_ALREADY_EXISTS


class RequestError(Error):
    """Invalid request: bad input, schema violation, or unsupported operation.

    Covers both structural problems (invalid column name, type mismatch) and
    operational problems (unsupported operation, invalid state). Use error codes
    to distinguish the specific cause. HTTP status is derived automatically:
    schema/validation codes -> 422, operation codes -> 400.
    """

    error_code: ErrorCode = ErrorCode.INVALID_ARGUMENT


class AuthorizationError(Error):
    """Caller lacks permission for the requested operation."""

    error_code: ErrorCode = ErrorCode.INSUFFICIENT_PRIVILEGES


class ExternalServiceError(Error):
    """An upstream provider or external store returned an error."""

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

    error_code: ErrorCode = ErrorCode.DATABASE_UNAVAILABLE
    retryable: bool = True


class ConcurrencyError(Error):
    """Serialization failure, deadlock, or concurrent modification conflict."""

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
