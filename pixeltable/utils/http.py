import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from pathlib import Path
from random import random
from typing import Any

_logger = logging.getLogger('pixeltable')

_RETRIABLE_ERROR_INDICATORS = (
    'rate limit',
    'too many requests',
    '429',
    'quota exceeded',
    'throttled',
    'rate exceeded',
    'connection error',
    'timed out',
)
_RETRY_AFTER_PATTERNS = (
    r'retry after (\d+(?:\.\d+)?)\s*seconds?',
    r'try again in (\d+(?:\.\d+)?)\s*seconds?',
    r'wait (\d+(?:\.\d+)?)\s*seconds?',
    r'retry-after:\s*(\d+(?:\.\d+)?)',
)
_RETRIABLE_HTTP_STATUSES: dict[str, int] = {
    'TOO_MANY_REQUESTS': HTTPStatus.TOO_MANY_REQUESTS.value,
    'SERVICE_UNAVAILABLE': HTTPStatus.SERVICE_UNAVAILABLE.value,
    'REQUEST_TIMEOUT': HTTPStatus.REQUEST_TIMEOUT.value,
    'GATEWAY_TIMEOUT': HTTPStatus.GATEWAY_TIMEOUT.value,
}


def is_retriable_error(exc: Exception) -> tuple[bool, float | None]:
    """Attempts to guess if the exception indicates a retriable eror. If that is the case, returns True
    and the retry delay in seconds."""

    # Check for HTTP status TOO_MANY_REQUESTS in various exception classes.
    # We look for attributes that contain status codes, instead of checking the type of the exception,
    # in order to handle a wider variety of exception classes.
    err_md = _extract_error_metadata(exc)
    if (err_md is None or not err_md[0]) and hasattr(exc, 'response'):
        err_md = _extract_error_metadata(exc.response)

    if err_md is not None and err_md[0]:
        retry_after = err_md[1]
        return err_md[0], retry_after if retry_after is not None and retry_after >= 0 else None

    # Check common rate limit keywords in exception message
    error_msg = str(exc).lower()
    if any(indicator in error_msg for indicator in _RETRIABLE_ERROR_INDICATORS):
        retry_delay = _extract_retry_delay_from_message(error_msg)
        return True, retry_delay if retry_delay is not None and retry_delay >= 0 else None

    return False, None


def _extract_error_metadata(obj: Any) -> tuple[bool, float | None] | None:
    is_retriable: bool | None = None
    retry_delay: float | None = None
    for attr in ['status', 'code', 'status_code']:
        if hasattr(obj, attr):
            is_retriable = getattr(obj, attr) in _RETRIABLE_HTTP_STATUSES.values()
            is_retriable |= str(getattr(obj, attr)).upper() in _RETRIABLE_HTTP_STATUSES

    if hasattr(obj, 'headers'):
        retry_delay = _extract_retry_delay_from_headers(obj.headers)
        if retry_delay is not None:
            is_retriable = True

    return (is_retriable, retry_delay) if is_retriable is not None else None


def _extract_retry_delay_from_headers(headers: Any | None) -> float | None:
    """Extract retry delay from HTTP headers."""
    if headers is None:
        return None

    # convert headers to dict-like object for consistent access
    header_dict: dict
    if hasattr(headers, 'get'):
        header_dict = headers
    else:
        # headers are a list of tuples or other format
        try:
            header_dict = dict(headers)
        except (TypeError, ValueError):
            return None
    # normalize dict keys: lowercase and remove dashes
    header_dict = {k.lower().replace('-', ''): v for k, v in header_dict.items()}

    # check Retry-After header
    retry_after = header_dict.get('retryafter')
    if retry_after is not None:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass

    # check X-RateLimit-Reset (Unix timestamp)
    reset_time = header_dict.get('xratelimitreset')
    if reset_time is not None:
        try:
            reset_timestamp = float(reset_time)
            delay = max(0, reset_timestamp - time.time())
            return delay
        except (ValueError, TypeError):
            pass

    # check X-RateLimit-Reset-After (seconds from now)
    reset_after = header_dict.get('xratelimitresetafter')
    if reset_after is not None:
        try:
            return float(reset_after)
        except (ValueError, TypeError):
            pass

    return None


def _extract_retry_delay_from_message(msg: str) -> float | None:
    msg_lower = msg.lower()
    for pattern in _RETRY_AFTER_PATTERNS:
        match = re.search(pattern, msg_lower)
        if match is not None:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                continue
    return None


def exponential_backoff(attempt: int, base: float = 2.0, max_delay: float = 16.0) -> float:
    """Generates the retry delay using exponential backoff strategy with jitter. Attempt count starts from 0."""
    basic_delay = min(max_delay, base**attempt) / 2
    return basic_delay + random() * basic_delay


def fetch_url(url: str, allow_local_file: bool = False) -> Path:
    """
    Fetches a remote URL into the TempStore and returns its path.

    If `allow_local_file` is True, and the URL is a file:// URL or a local file path, then the local path is returned
    directly without copying. If `allow_local_file` is False, then an AssertionError is raised.
    """
    from .local_store import TempStore
    from .object_stores import ObjectOps

    _logger.debug(f'fetching url={url} thread_name={threading.current_thread().name}')
    parsed = urllib.parse.urlparse(url)

    if len(parsed.scheme) <= 1:
        # local file path (len(parsed.scheme) == 1 implies a Windows path with drive letter)
        assert allow_local_file
        return Path(url)

    path: Path | None = None
    if parsed.path:
        path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))

    if parsed.scheme == 'file':
        assert allow_local_file
        assert path is not None
        return path

    # preserve the file extension, if there is one
    tmp_path = TempStore.create_path(extension=(path.suffix if path else ''))

    _logger.debug(f'Downloading {url} to {tmp_path}')
    ObjectOps.copy_object_to_local_file(url, tmp_path)
    _logger.debug(f'Downloaded {url} to {tmp_path}')

    return tmp_path
