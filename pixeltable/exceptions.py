"""Pixeltable exception hierarchy.

Exception classes carry structured error information (error_code, retryable, cause) for
programmatic error handling. HTTP status code mapping is defined per exception class for
use by REST integration layers (e.g., FastAPI).

HTTP Status Code Mapping
========================

Exception Class       HTTP    Error Codes
--------------------  ------  -------------------------------------------
Error                 500     INTERNAL_ERROR
NotFoundError         404     COLUMN_NOT_FOUND, PATH_NOT_FOUND,
                              DIRECTORY_NOT_FOUND, INDEX_NOT_FOUND,
                              FUNCTION_NOT_FOUND, ROW_NOT_FOUND,
                              STORAGE_NOT_FOUND
AlreadyExistsError    409     COLUMN_ALREADY_EXISTS, PATH_ALREADY_EXISTS,
                              INDEX_ALREADY_EXISTS, FUNCTION_ALREADY_EXISTS
ValidationError       422     INVALID_COLUMN_NAME, INVALID_PATH,
                              INVALID_EXPRESSION, INVALID_TYPE,
                              INVALID_SCHEMA, INVALID_ARGUMENT,
                              INVALID_DATA_FORMAT, MISSING_REQUIRED,
                              TYPE_MISMATCH, CONSTRAINT_VIOLATION
OperationError        400     UNSUPPORTED_OPERATION, INVALID_STATE,
                              IMMUTABLE
AuthorizationError    403     INSUFFICIENT_PRIVILEGES
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

    # ValidationError codes
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


class Error(Exception):
    """Base exception for user-facing Pixeltable errors."""

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


class NotFoundError(Error):
    """Resource not found."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


class AlreadyExistsError(Error):
    """Resource already exists."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


class ValidationError(Error):
    """Invalid input, schema, or argument."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


class OperationError(Error):
    """Operation not supported or not valid for the current target/state."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


class AuthorizationError(Error):
    """Caller lacks permission for the requested operation."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


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
