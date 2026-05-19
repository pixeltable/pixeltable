"""Tiny HTTP router used by the daemon's BaseHTTPRequestHandler dispatcher.

Each route registers a regex (built from a FastAPI-style pattern) and a handler
that takes a `Request`. Lookup walks the route list in registration order and
returns the first match; this is what lets `/api/tables/{path:path}/rows` win over
the catch-all `/api/tables/{path:path}` describe route.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pydantic

from pixeltable import exceptions as excs

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

    path_params: dict[str, str]
    query: dict[str, list[str]]
    body_bytes: bytes

    # --- query accessors --------------------------------------------------------------------

    def query_str(self, name: str, default: str | None = None) -> str | None:
        vals = self.query.get(name)
        if vals is None or len(vals) == 0:
            return default
        return vals[0]

    def query_int(self, name: str, default: int | None = None, *, ge: int | None = None, le: int | None = None) -> int:
        raw = self.query_str(name)
        if raw is None:
            if default is None:
                raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, f"missing required query parameter '{name}'")
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f"'{name}' must be an integer; got {raw!r}"
                ) from None
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

    # --- body --------------------------------------------------------------------------------

    def body(self, model_cls: type[T]) -> T:
        if len(self.body_bytes) == 0:
            raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'request body required')
        try:
            return model_cls.model_validate_json(self.body_bytes)
        except pydantic.ValidationError as e:
            msgs = [str(err.get('msg', '')).removeprefix('Value error, ') for err in e.errors()]
            detail = '; '.join(m for m in msgs if m != '') or 'invalid request body'
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, detail) from None


@dataclass(frozen=True)
class _Route:
    method: str
    pattern: re.Pattern[str]
    handler: Handler


class Router:
    """Decorator-based route table. Lookup walks routes in registration order."""

    def __init__(self) -> None:
        self._routes: list[_Route] = []

    def get(self, pattern: str) -> Callable[[Handler], Handler]:
        return self._register('GET', pattern)

    def post(self, pattern: str) -> Callable[[Handler], Handler]:
        return self._register('POST', pattern)

    def _register(self, method: str, pattern: str) -> Callable[[Handler], Handler]:
        regex = _compile_pattern(pattern)

        def decorator(fn: Handler) -> Handler:
            self._routes.append(_Route(method=method, pattern=regex, handler=fn))
            return fn

        return decorator

    def match(self, method: str, url_path: str) -> tuple[Handler, dict[str, str]] | None:
        for r in self._routes:
            if r.method != method:
                continue
            m = r.pattern.fullmatch(url_path)
            if m is not None:
                return r.handler, m.groupdict()
        return None


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """FastAPI-style `{name}` and `{name:path}` placeholders -> a regex.

    `{name}` matches one URL segment (no slashes); `{name:path}` greedily matches any
    non-empty run including slashes. fullmatch is used at lookup time so an explicit
    trailing anchor isn't necessary in the output.
    """
    out_parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == '{':
            close = pattern.index('}', i)
            name, _, ptype = pattern[i + 1 : close].partition(':')
            if ptype == 'path':
                out_parts.append(f'(?P<{name}>.+)')
            elif ptype == '':
                out_parts.append(f'(?P<{name}>[^/]+)')
            else:
                raise ValueError(f'unknown path converter {ptype!r} in pattern {pattern!r}')
            i = close + 1
        else:
            out_parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile(''.join(out_parts))
