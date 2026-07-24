"""Tiny HTTP router used by the daemon's BaseHTTPRequestHandler dispatcher.

Each route registers a static path (eg `/api/tables/rows`) and a handler that takes a `Request`.
Lookup is an exact (method, path) match; catalog paths travel in the query string or body, not the URL.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

import pydantic

from pixeltable import exceptions as excs
from pixeltable_cli.utils import validate_path_shape

T = TypeVar('T', bound=pydantic.BaseModel)

# Handlers are uniformly typed so the dispatcher's table doesn't have to carry a per-route
# signature. Return type is intentionally broad: pydantic models, dicts, lists, None, or a
# RawResponse all flow through the same response serializer.
Handler = Callable[['Request'], Any]


@dataclass
class RawResponse:
    """Returned by handlers that need to control content-type/headers (e.g. CSV download)."""

    body: bytes
    content_type: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    status: int = 200


@dataclass
class Request:
    """Per-request inputs the dispatcher hands to a handler.

    Query values arrive as lists because urllib's parse_qs always returns lists; the typed
    accessors below collapse single-value params and validate range/type so each handler
    doesn't repeat that work.
    """

    query: dict[str, list[str]]
    body_bytes: bytes
    headers: dict[str, str] = field(default_factory=dict)
    resolved_paths: list[str] = field(default_factory=list)  # catalog paths resolved this request, for logging

    def query_str(self, name: str, default: str | None = None) -> str | None:
        vals = self.query.get(name)
        if vals is None or len(vals) == 0:
            return default
        return vals[0]

    @overload
    def query_int(self, name: str, *, default: None, ge: int | None = None, le: int | None = None) -> int | None: ...
    @overload
    def query_int(self, name: str, *, default: int, ge: int | None = None, le: int | None = None) -> int: ...
    def query_int(self, name: str, *, default: int | None, ge: int | None = None, le: int | None = None) -> int | None:
        raw = self.query_str(name)
        if raw is None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f"'{name}' must be an integer; got {raw!r}"
                ) from None
        if value is None:
            return None
        if ge is not None and value < ge:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"'{name}' must be >= {ge}; got {value}")
        if le is not None and value > le:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"'{name}' must be <= {le}; got {value}")
        return value

    def query_bool(self, name: str, default: bool = False) -> bool:
        # Match what FastAPI/Pydantic v2 accepts: '1'/'0', 'true'/'false', 'yes'/'no', 'on'/'off'.
        raw = self.query_str(name)
        if raw is None:
            return default
        low = raw.lower()
        if low in ('1', 'true', 'yes', 'on'):
            return True
        if low in ('0', 'false', 'no', 'off'):
            return False
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"'{name}' must be a boolean; got {raw!r}")

    def query_list(self, name: str) -> list[str]:
        return list(self.query.get(name, []))

    def body(self, model_cls: type[T]) -> T:
        if len(self.body_bytes) == 0:
            raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'request body required')
        try:
            return model_cls.model_validate_json(self.body_bytes)
        except pydantic.ValidationError as e:
            msgs = [str(err.get('msg', '')).removeprefix('Value error, ') for err in e.errors()]
            detail = '; '.join(m for m in msgs if m != '') or 'invalid request body'
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, detail) from None

    def resolve_path(self, path: str) -> str:
        """Resolve a catalog path against this request's session working directory, then shape-validate it.

        As a CLI convention:
        - a pxt:// URI is an absolute hosted path, used as-is;
        - a leading '/' marks an absolute path from the catalog root -- the '/' is stripped and any working
          directory is ignored;
        - anything else is relative and is taken under the session's working directory when one is set (the
          empty path resolves to the working directory itself).
        """
        from . import daemon  # module-level import would be circular: daemon -> http_server -> router

        if not path.startswith('pxt://'):
            if path.startswith('/'):
                path = path[1:]  # drop the leading '/'
            else:
                wd = daemon.get_wd(self.headers.get('x-pxt-session'))
                if wd is not None:
                    path = wd if path == '' else f'{wd}/{path}'
        if path != '':
            err = validate_path_shape(path)
            if err is not None:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, err)
        self.resolved_paths.append(path)
        return path


class Router:
    """Decorator-based route table keyed by (method, path) for exact-match lookup."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], Handler] = {}

    def get(self, path: str) -> Callable[[Handler], Handler]:
        return self._register('GET', path)

    def post(self, path: str) -> Callable[[Handler], Handler]:
        return self._register('POST', path)

    def _register(self, method: str, path: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            assert (method, path) not in self._routes, f'duplicate route {method} {path}'
            self._routes[method, path] = fn
            return fn

        return decorator

    def match(self, method: str, url_path: str) -> Handler | None:
        return self._routes.get((method, url_path))
