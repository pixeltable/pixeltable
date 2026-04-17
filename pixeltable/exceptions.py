from __future__ import annotations

import enum
from types import TracebackType
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from pixeltable import exprs


class ErrorCode(enum.Enum):
    """
    Error codes for Pixeltable exceptions.

    Codes are assigned explicit stable values in grouped ranges.  Each code must map to a specific HTTP status code.
    New codes must be added at the end of their group and cannot reuse or shift existing values.
    """

    http_status: int
    is_retryable: bool

    def __new__(cls, value: int, http_status: int, is_retryable: bool) -> Self:
        obj = object.__new__(cls)
        obj._value_ = value
        obj.http_status = http_status
        obj.is_retryable = is_retryable
        return obj

    # Error (0xxx)
    INTERNAL_ERROR = 0, 500, False
    GENERIC_USER_ERROR = 1, 400, False

    # NotFoundError (1xxx)
    COLUMN_NOT_FOUND = 1000, 404, False
    PATH_NOT_FOUND = 1001, 404, False
    DIRECTORY_NOT_FOUND = 1002, 404, False
    INDEX_NOT_FOUND = 1003, 404, False
    FUNCTION_NOT_FOUND = 1004, 404, False
    ROW_NOT_FOUND = 1005, 404, False
    STORAGE_NOT_FOUND = 1006, 404, False

    # AlreadyExistsError (2xxx)
    COLUMN_ALREADY_EXISTS = 2000, 409, False
    PATH_ALREADY_EXISTS = 2001, 409, False
    INDEX_ALREADY_EXISTS = 2002, 409, False
    FUNCTION_ALREADY_EXISTS = 2003, 409, False

    # RequestError (3xxx)
    INVALID_COLUMN_NAME = 3000, 422, False
    INVALID_PATH = 3001, 422, False
    INVALID_EXPRESSION = 3002, 422, False
    INVALID_TYPE = 3003, 422, False
    INVALID_SCHEMA = 3004, 422, False
    INVALID_ARGUMENT = 3005, 422, False
    INVALID_DATA_FORMAT = 3006, 422, False
    MISSING_REQUIRED = 3007, 422, False
    TYPE_MISMATCH = 3008, 422, False
    CONSTRAINT_VIOLATION = 3009, 422, False
    UNSUPPORTED_OPERATION = 3010, 400, False
    INVALID_STATE = 3011, 400, False
    IMMUTABLE = 3012, 400, False
    INVALID_CONFIGURATION = 3013, 422, False

    # AuthorizationError (4xxx)
    INSUFFICIENT_PRIVILEGES = 4000, 403, False
    MISSING_CREDENTIALS = 4001, 403, False

    # ExternalServiceError (5xxx)
    PROVIDER_ERROR = 5000, 502, True
    RATE_LIMITED = 5001, 429, True
    PROVIDER_AUTH_ERROR = 5002, 401, False
    PROVIDER_BAD_REQUEST = 5003, 400, False
    PROVIDER_TIMEOUT = 5004, 504, True
    PROVIDER_OVERLOADED = 5005, 503, True

    # ServiceUnavailableError (6xxx)
    DATABASE_UNAVAILABLE = 6000, 503, True
    STORE_UNAVAILABLE = 6001, 503, True

    # ConcurrencyError (7xxx)
    SERIALIZATION_FAILURE = 7000, 409, True
    CONCURRENT_MODIFICATION = 7001, 409, True


class Error(Exception):
    """Base exception for user-facing Pixeltable errors.

    Every instance carries an explicit error_code. Retry-ability is a property of the
    error_code; retry_after is an optional per-instance hint (e.g., from a provider's
    Retry-After header) that any retryable error may carry.
    """

    error_code: ErrorCode
    retry_after: float | None = None

    def __init__(
        self,
        error_code: ErrorCode,
        message: str = '',
        cause: BaseException | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after
        if cause is not None:
            self.__cause__ = cause

    @property
    def http_status(self) -> int:
        return self.error_code.http_status

    @property
    def is_retryable(self) -> bool:
        return self.error_code.is_retryable

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for REST error responses."""
        d: dict[str, Any] = {'error_code': self.error_code.name, 'message': str(self), 'retryable': self.is_retryable}
        if self.__cause__ is not None:
            d['cause'] = str(self.__cause__)
        if self.retry_after is not None:
            d['retry_after'] = self.retry_after
        return d


class NotFoundError(Error):
    """Resource not found."""


class AlreadyExistsError(Error):
    """Resource already exists."""


class RequestError(Error):
    """Invalid request: bad input, schema violation, or unsupported operation.

    Covers both structural problems (invalid column name, type mismatch) and
    operational problems (unsupported operation, invalid state). Use error codes
    to distinguish the specific cause. HTTP status is derived automatically:
    schema/validation codes -> 422, operation codes -> 400.
    """


class AuthorizationError(Error):
    """Caller lacks permission for the requested operation."""


class ExternalServiceError(Error):
    """An upstream provider or external store returned an error."""

    provider: str | None = None
    provider_http_status_code: int | None = None

    def __init__(
        self,
        error_code: ErrorCode,
        message: str = '',
        *,
        cause: BaseException | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(error_code, message, cause=cause, retry_after=retry_after)
        self.provider = provider
        self.provider_http_status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.provider is not None:
            d['provider'] = self.provider
        if self.provider_http_status_code is not None:
            d['provider_http_status_code'] = self.provider_http_status_code
        return d


class ServiceUnavailableError(Error):
    """Database, store, or other infrastructure is unreachable."""


class ConcurrencyError(Error):
    """Serialization failure, deadlock, or concurrent modification conflict."""


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
