"""Dependency-free instrumentation hooks.

Core pixeltable reports spans (timed, nested units of work) and discrete events through this module;
subscribers (e.g. the OTEL bridge in `pixeltable.otel`) translate them into a telemetry backend. With no
subscribers registered every call is a near-free no-op, but hot loops should still guard with
`if hooks.active():` before building attribute dicts (or pass `attrs` as a callable).

Span levels mirror logging levels: spans declared below the configured threshold (default INFO) are not
emitted; their descendants are parented to the nearest emitted ancestor. Only operation spans
(`set_current=True`) may be roots; any other span started without an ambient ancestor is suppressed.

Parentage: spans started with `set_current=True` become the ambient parent (a ContextVar, so it propagates
into asyncio tasks); other spans parent to the ambient span unless an explicit `parent` handle is passed.
`set_current=True` requires that the span is ended on the same thread/context it was started on.
`capture_context()`/`restore_context()`/`exit_context()` carry the ambient state (including subscriber
state) across explicit thread handoffs.

Spans should cover contiguous units of real computation (a UDF call, a DB insert batch, model loading) or
serve as structural containers (operation and exec-node spans). A CPU work span must not contain a
`yield`/`await`, which would let it cover unrelated interleaved work; awaiting an external call inside a
span is fine (the request really is in flight).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable, Iterator, Union

_logger = logging.getLogger('pixeltable.hooks')

TRACE = 5
DEBUG = 10
INFO = 20

Attrs = Union[dict[str, Any], Callable[[], dict[str, Any]], None]


class Subscriber:
    """Base class for instrumentation subscribers; all methods are optional no-ops.

    Methods are invoked synchronously at the call site; implementations must be fast and should not raise
    (exceptions are caught and logged, never propagated to the host operation).

    Attribute dicts may contain keys starting with `_`; these carry raw Python objects (e.g. a UDF result)
    for subscribers to consume and must not be exported as telemetry attributes.
    """

    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        """Start a span and return a token that is passed back to on_span_end().

        parent_token is this subscriber's token for the parent span, or None for a root span.
        """
        return None

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        pass

    def on_event(self, name: str, attrs: dict[str, Any] | None) -> None:
        pass

    def capture_context(self) -> Any:
        """Snapshot subscriber-specific ambient state; called on the spawning thread."""
        return None

    def restore_context(self, ctx: Any) -> Any:
        """Restore a capture_context() snapshot on a worker thread; the result is passed to exit_context()."""
        return None

    def exit_context(self, token: Any) -> None:
        pass


class SpanHandle:
    """Opaque handle for an emitted span; created by span_start() and consumed by span_end()."""

    __slots__ = ('cv_token', 'pending_attrs', 'subs', 'tokens')

    def __init__(self, subs: tuple[Subscriber, ...], tokens: tuple[Any, ...]) -> None:
        self.subs = subs
        self.tokens = tokens
        self.cv_token: Token[SpanHandle | None] | None = None
        self.pending_attrs: dict[str, Any] | None = None


@dataclasses.dataclass(slots=True)
class _PassthroughHandle:
    """Stand-in for a level-suppressed span: span_end() ignores it, children parent to `target`."""

    target: SpanHandle | None


AnySpanHandle = Union[SpanHandle, _PassthroughHandle]

_registry_lock = threading.Lock()
_SUBSCRIBERS: tuple[Subscriber, ...] = ()
_span_level = INFO
_current_span: ContextVar[SpanHandle | None] = ContextVar('pxt_current_span', default=None)
_logged_error_keys: set[tuple[int, str]] = set()


def subscribe(subscriber: Subscriber) -> None:
    global _SUBSCRIBERS  # noqa: PLW0603
    with _registry_lock:
        if subscriber not in _SUBSCRIBERS:
            _SUBSCRIBERS = (*_SUBSCRIBERS, subscriber)


def unsubscribe(subscriber: Subscriber) -> None:
    global _SUBSCRIBERS  # noqa: PLW0603
    with _registry_lock:
        _SUBSCRIBERS = tuple(s for s in _SUBSCRIBERS if s is not subscriber)


def active() -> bool:
    return bool(_SUBSCRIBERS)


def set_span_level(level: int) -> None:
    """Set the global span emission threshold; spans with level < threshold are suppressed."""
    global _span_level  # noqa: PLW0603
    _span_level = level


def current_span() -> SpanHandle | None:
    return _current_span.get()


def _log_subscriber_error(subscriber: Subscriber, method: str, exc: Exception) -> None:
    # warn once per (subscriber, method), then drop to debug to avoid log storms from per-row failures
    key = (id(subscriber), method)
    level = logging.DEBUG if key in _logged_error_keys else logging.WARNING
    _logged_error_keys.add(key)
    _logger.log(level, f'instrumentation subscriber {type(subscriber).__name__}.{method}() failed: {exc!r}')


def _resolve_attrs(attrs: Attrs) -> dict[str, Any] | None:
    if not callable(attrs):
        return attrs
    try:
        return attrs()
    except Exception as e:
        _logger.debug(f'attribute callable failed: {e!r}')
        return None


def span_start(
    name: str, *, level: int = INFO, parent: AnySpanHandle | None = None, set_current: bool = False, attrs: Attrs = None
) -> AnySpanHandle | None:
    subs = _SUBSCRIBERS
    if not subs:
        return None
    if isinstance(parent, _PassthroughHandle):
        parent = parent.target
    if parent is None:
        parent = _current_span.get()
    # only operation spans (set_current=True) may be roots: spans reported from inside an operation that
    # carries no span (eg, a bare query) are suppressed rather than emitted as orphan roots
    if level < _span_level or (parent is None and not set_current):
        return _PassthroughHandle(parent)
    attrs = _resolve_attrs(attrs)
    tokens: list[Any] = []
    for s in subs:
        parent_token: Any = None
        if parent is not None and s in parent.subs:
            # a subscriber that registered after the parent started has no parent token (its span is a root)
            parent_token = parent.tokens[parent.subs.index(s)]
        try:
            tokens.append(s.on_span_start(name, parent_token, attrs, set_current))
        except Exception as e:
            _log_subscriber_error(s, 'on_span_start', e)
            tokens.append(None)
    handle = SpanHandle(subs, tuple(tokens))
    if set_current:
        handle.cv_token = _current_span.set(handle)
    return handle


def span_end(handle: AnySpanHandle | None, *, exc: BaseException | None = None, attrs: Attrs = None) -> None:
    if not isinstance(handle, SpanHandle):
        return
    if handle.cv_token is not None:
        try:
            _current_span.reset(handle.cv_token)
        except Exception as e:
            _logger.debug(f'failed to reset ambient span: {e!r}')
        handle.cv_token = None
    attrs = _resolve_attrs(attrs)
    if handle.pending_attrs is not None:
        attrs = {**handle.pending_attrs, **(attrs or {})}
    for s, token in zip(handle.subs, handle.tokens, strict=True):
        try:
            s.on_span_end(token, exc, attrs)
        except Exception as e:
            _log_subscriber_error(s, 'on_span_end', e)


def add_attrs(handle: AnySpanHandle | None, **attrs: Any) -> None:
    """Attach attrs ('pxt.'-prefixed, None values skipped) to be reported when the span ends.

    Argument computation is eager; call sites should guard with `if handle is not None:` (or
    `hooks.active()`) when the values are expensive to produce.
    """
    if not isinstance(handle, SpanHandle):
        return
    if handle.pending_attrs is None:
        handle.pending_attrs = {}
    handle.pending_attrs.update({f'pxt.{k}': v for k, v in attrs.items() if v is not None})


def emit(name: str, attrs: Attrs = None) -> None:
    subs = _SUBSCRIBERS
    if not subs:
        return
    attrs = _resolve_attrs(attrs)
    for s in subs:
        try:
            s.on_event(name, attrs)
        except Exception as e:
            _log_subscriber_error(s, 'on_event', e)


@contextlib.contextmanager
def span(
    name: str, *, level: int = INFO, parent: AnySpanHandle | None = None, set_current: bool = False, **attrs: Any
) -> Iterator[AnySpanHandle | None]:
    """Context-manager sugar for a lexical-block span.

    Keyword attrs get a 'pxt.' prefix; None values are skipped.
    """
    if not _SUBSCRIBERS:
        yield None
        return
    handle = span_start(
        name,
        level=level,
        parent=parent,
        set_current=set_current,
        attrs={f'pxt.{k}': v for k, v in attrs.items() if v is not None},
    )
    try:
        yield handle
    except BaseException as e:
        span_end(handle, exc=e)
        raise
    else:
        span_end(handle)


@dataclasses.dataclass(slots=True)
class _CtxSnapshot:
    subs: tuple[Subscriber, ...]
    span: SpanHandle | None
    sub_ctxs: tuple[Any, ...]


_CtxToken = tuple[_CtxSnapshot, Token[Union[SpanHandle, None]], tuple[Any, ...]]


def capture_context() -> _CtxSnapshot | None:
    """Snapshot the ambient instrumentation state; call on the spawning thread."""
    subs = _SUBSCRIBERS
    if not subs:
        return None
    sub_ctxs: list[Any] = []
    for s in subs:
        try:
            sub_ctxs.append(s.capture_context())
        except Exception as e:
            _log_subscriber_error(s, 'capture_context', e)
            sub_ctxs.append(None)
    return _CtxSnapshot(subs, _current_span.get(), tuple(sub_ctxs))


def restore_context(snapshot: _CtxSnapshot | None) -> _CtxToken | None:
    """Restore a capture_context() snapshot; call on the worker thread. The result is passed to exit_context()."""
    if snapshot is None:
        return None
    cv_token = _current_span.set(snapshot.span)
    sub_tokens: list[Any] = []
    for s, ctx in zip(snapshot.subs, snapshot.sub_ctxs, strict=True):
        try:
            sub_tokens.append(s.restore_context(ctx))
        except Exception as e:
            _log_subscriber_error(s, 'restore_context', e)
            sub_tokens.append(None)
    return (snapshot, cv_token, tuple(sub_tokens))


def exit_context(token: _CtxToken | None) -> None:
    if token is None:
        return
    snapshot, cv_token, sub_tokens = token
    for s, t in zip(snapshot.subs, sub_tokens, strict=True):
        try:
            s.exit_context(t)
        except Exception as e:
            _log_subscriber_error(s, 'exit_context', e)
    try:
        _current_span.reset(cv_token)
    except Exception as e:
        _logger.debug(f'failed to reset ambient span: {e!r}')
