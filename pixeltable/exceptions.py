from __future__ import annotations

import enum
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, NoReturn

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
    TABLE_NOT_FOUND = 1002, 404, False
    DIRECTORY_NOT_FOUND = 1003, 404, False
    INDEX_NOT_FOUND = 1004, 404, False
    FUNCTION_NOT_FOUND = 1005, 404, False
    ROW_NOT_FOUND = 1006, 404, False
    STORAGE_NOT_FOUND = 1007, 404, False
    SERVICE_NOT_FOUND = 1008, 404, False
    DEPLOYMENT_NOT_FOUND = 1009, 404, False

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
    INVALID_CONFIGURATION = 3013, 422, False
    NOT_BOUND = 3014, 400, False
    ALREADY_BOUND = 3015, 400, False
    SCHEMA_MISMATCH = 3016, 422, False
    FILE_CACHE_FULL = 3017, 507, False

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
    retry_after: float | None

    # Diagnostic text (e.g. an evaluation-environment stack trace) shown when the error is rendered locally,
    # but withheld from to_dict() so it never travels to a remote client.
    detail: str | None

    # Thousands digit of the ErrorCode values this class is allowed to carry.
    # The base Error class carries the 0xxx generic codes; each subclass narrows to its own group.
    _code_group: ClassVar[int] = 0

    def __init__(self, error_code: ErrorCode, message: str = '', *, retry_after: float | None = None) -> None:
        cls = type(self)
        # make sure we got an error code appropriate for this exception class
        assert error_code.value // 1000 == cls._code_group
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after
        self.detail = None

    @property
    def message(self) -> str:
        """The human-facing message, without any attached diagnostic detail."""
        return self.args[0] if len(self.args) > 0 else ''

    def __str__(self) -> str:
        return self.message if self.detail is None else f'{self.message}\n{self.detail}'

    @property
    def http_status(self) -> int:
        return self.error_code.http_status

    @property
    def is_retryable(self) -> bool:
        """
        If False, re-running the operation that caused this error without any changes is guaranteed to fail with the
        same error. If True, the error might be transient and the operation might succeed if retried.
        """
        return self.error_code.is_retryable

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for REST error responses."""
        d: dict[str, Any] = {
            'error_code': self.error_code.name,
            'message': self.message,
            'retryable': self.is_retryable,
        }
        # Only surface a cause that is itself a Pixeltable Error, and only its user-facing message: str() on an
        # arbitrary __cause__ (or an Error's detail) can leak internal types, stack traces, or filesystem paths.
        if isinstance(self.__cause__, Error):
            d['cause'] = self.__cause__.message
        if self.retry_after is not None:
            d['retry_after'] = self.retry_after
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Error':
        """Reconstruct an Error from to_dict() output."""
        code = ErrorCode[d['error_code']]
        subclass = _error_subclasses_by_group().get(code.value // 1000, Error)
        return subclass._reconstruct(code, d)

    @classmethod
    def _reconstruct(cls, error_code: ErrorCode, d: dict[str, Any]) -> 'Error':
        """Build an instance of this class from to_dict() output. Subclasses with extra serialized state
        override this to restore it. Bypasses __init__ (which may require args not on the wire)."""
        err = cls.__new__(cls)
        Error.__init__(err, error_code, d.get('message', ''), retry_after=d.get('retry_after'))
        err.detail = d.get('detail')
        return err


class NotFoundError(Error):
    """Resource not found."""

    _code_group = 1


class AlreadyExistsError(Error):
    """Resource already exists."""

    _code_group = 2


class RequestError(Error):
    """Invalid request: bad input, schema violation, or unsupported operation.

    Covers both structural problems (invalid column name, type mismatch) and
    operational problems (unsupported operation, invalid state). Use error codes
    to distinguish the specific cause. HTTP status is derived automatically:
    schema/validation codes -> 422, operation codes -> 400.
    """

    _code_group = 3


class AuthorizationError(Error):
    """Caller lacks permission for the requested operation."""

    _code_group = 4


class ExternalServiceError(Error):
    """An upstream provider or external store returned an error."""

    _code_group = 5

    provider: str | None = None
    provider_http_status_code: int | None = None

    def __init__(
        self,
        error_code: ErrorCode,
        message: str = '',
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(error_code, message, retry_after=retry_after)
        self.provider = provider
        self.provider_http_status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.provider is not None:
            d['provider'] = self.provider
        if self.provider_http_status_code is not None:
            d['provider_http_status_code'] = self.provider_http_status_code
        return d

    @classmethod
    def _reconstruct(cls, error_code: ErrorCode, d: dict[str, Any]) -> 'Error':
        err = super()._reconstruct(error_code, d)
        assert isinstance(err, ExternalServiceError)
        err.provider = d.get('provider')
        err.provider_http_status_code = d.get('provider_http_status_code')
        return err


class ServiceUnavailableError(Error):
    """Database, store, or other infrastructure is unreachable."""

    _code_group = 6


class ConcurrencyError(Error):
    """Serialization failure, deadlock, or concurrent modification conflict."""

    _code_group = 7


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


def raise_from_expr_eval_err(e: ExprEvalError) -> NoReturn:
    """Convert an ExprEvalError (internal) into a user-facing Error."""
    msg = f'In row {e.row_num} the {e.expr_msg} encountered exception {type(e.exc).__name__}:\n{e.exc}'
    if len(e.input_vals) > 0:
        input_msgs = [f"'{d}' = {d.col_type.print_value(e.input_vals[i])}" for i, d in enumerate(e.expr.dependencies())]
        msg += f'\nwith {", ".join(input_msgs)}'
    assert e.exc_tb is not None
    stack_trace = traceback.format_tb(e.exc_tb)
    # The stack frames belong to the (possibly non-local) evaluation environment, so they are carried as the
    # error's detail rather than folded into the message
    detail: str | None = None
    if len(stack_trace) > 2:
        # the exception happened in user code; frame 0 is ExprEvaluator and frame 1 is some expr's eval(),
        # so [-1:1:-1] drops those two and reverses the rest to put the most recent frame on top
        nl = '\n'
        detail = f'Stack:\n{nl.join(stack_trace[-1:1:-1])}'
    if isinstance(e.exc, Error):
        err: Error = type(e.exc)(e.exc.error_code, msg)
    else:
        err = RequestError(ErrorCode.UNSUPPORTED_OPERATION, msg)
    err.detail = detail
    raise err from e


class PixeltableWarning(Warning):
    pass


class PixeltableDeprecationWarning(DeprecationWarning, PixeltableWarning):
    pass


def table_was_dropped(identifier: Any = None) -> NotFoundError:
    """
    Creates an error to indicate that a table is not found (was dropped).

    identifier (optional) can be a table ID or name
    """
    msg = 'Table was dropped' if identifier is None else f'Table was dropped (no record found for {identifier})'
    return NotFoundError(ErrorCode.TABLE_NOT_FOUND, msg)


def is_table_not_found_error(e: BaseException) -> bool:
    """Returns True if the exception signals that a table was not found."""
    return isinstance(e, Error) and e.error_code == ErrorCode.TABLE_NOT_FOUND


def _error_subclasses_by_group() -> dict[int, type[Error]]:
    """Map each error-code group (thousands digit) to its Error subclass.from_dict."""

    def subclasses(c: type[Error]) -> Iterator[type[Error]]:
        yield c
        for sub in c.__subclasses__():
            yield from subclasses(sub)

    return {c._code_group: c for c in subclasses(Error)}
