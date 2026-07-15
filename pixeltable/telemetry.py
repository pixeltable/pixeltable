"""Dependency-free instrumentation hooks.

Core pixeltable reports spans (timed, nested units of work), discrete events attached to spans, and
metrics through this module; subscribers (e.g. the bridge in `opentelemetry-instrumentation-pixeltable`)
translate them into a telemetry backend. Metric instruments are declared once at module level
(`_rows = telemetry.counter(...)`) and recorded into where the measurement happens
(`_rows.add(n, table=path)`). With no subscribers
registered every call is a near-free no-op, but hot loops should still guard with `if telemetry.active():`
before building attribute dicts (or pass `attrs` as a callable).

Spans should cover contiguous units of real computation (a UDF call, a DB insert batch, model loading) or
serve as structural containers (operation and exec-node spans). A CPU work span must not contain a
`yield`/`await`, which would let it cover unrelated interleaved work; awaiting an external call inside a
span is fine (the request really is in flight).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import logging
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable, Hashable, Iterator, TypeVar, cast
from weakref import WeakKeyDictionary

_logger = logging.getLogger(__name__)

TRACE = 5
DEBUG = 10
INFO = 20

HookAttrs = dict[str, Any] | Callable[[], dict[str, Any]] | None

F = TypeVar('F', bound=Callable[..., Any])


class Subscriber(Hashable):
    """Base class for instrumentation subscribers; all methods are optional no-ops.

    Methods are invoked synchronously at the call site; implementations must be fast and should not raise
    (exceptions are caught and logged, never propagated to the host operation).

    Attribute dicts may contain keys starting with `_`; these carry raw Python objects (e.g. a UDF result)
    for subscribers to consume and must not be exported as telemetry attributes.
    """

    __hash__ = object.__hash__

    def on_span_start(self, name: str, parent_token: Any, attrs: dict[str, Any] | None, set_current: bool) -> Any:
        """Start a span and return a token that is passed back to on_span_end().

        parent_token is this subscriber's token for the parent span, or None for a root span.
        """
        return None

    def on_span_end(self, token: Any, exc: BaseException | None, attrs: dict[str, Any] | None) -> None:
        pass

    def on_event(self, token: Any, name: str, attrs: dict[str, Any]) -> None:
        """Attach a discrete event to the span identified by token (this subscriber's on_span_start() return)."""
        pass

    def on_counter_add(self, counter: Counter, value: int | float, attrs: dict[str, Any]) -> None:
        pass

    def on_histogram_record(self, histogram: Histogram, value: int | float, attrs: dict[str, Any]) -> None:
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

    # one handle is allocated per emitted span, which at DEBUG level means per row and per UDF cell;
    # slots drops the per-instance __dict__, cutting allocation cost and memory on that hot path
    __slots__ = ('cv_token', 'pending_attrs', 'subs', 'tokens')

    def __init__(self, subs: tuple[Subscriber, ...], tokens: tuple[Any, ...]) -> None:
        self.subs = subs
        self.tokens = tokens
        self.cv_token: Token[SpanHandle | None] | None = None
        self.pending_attrs: dict[str, Any] | None = None


# the ambient span; a ContextVar rather than a thread-local so it propagates into asyncio tasks and
# concurrent tasks on one thread don't see each other's spans
_current_span: ContextVar[SpanHandle | None] = ContextVar('pxt_current_span', default=None)
# the innermost enclosing spanned() span, exposed via func_span() for add_attrs() call sites
_func_span: ContextVar[SpanHandle | None] = ContextVar('pxt_func_span', default=None)


class SubscriberRegistry:
    """Process-global registry of instrumentation subscribers, plus the span-level threshold.

    Singleton accessed via SubscriberRegistry.get(). Subscribers are pushed in by external callers (e.g. an
    instrumentor), possibly before Pixeltable initializes, and cannot be re-derived, so the registry lives
    for the lifetime of the process. The span-emission API (span_start/span_end/emit/span/active/
    set_span_level/...) is module-level.
    """

    _instance: SubscriberRegistry | None = None
    # one lock guards both singleton creation and the read-modify-write mutations of _subscribers /
    # _logged_error_methods; after creation get() takes the lock-free fast path, so registration never
    # contends with it (_span_level is a plain scalar store and needs no lock)
    _lock: threading.Lock = threading.Lock()

    _subscribers: tuple[Subscriber, ...]
    _span_level: int
    _logged_error_methods: WeakKeyDictionary[Subscriber, set[str]]

    @classmethod
    def get(cls) -> SubscriberRegistry:
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = SubscriberRegistry()
        return cls._instance

    def __init__(self) -> None:
        assert self._instance is None, (
            'SubscriberRegistry is a singleton; use SubscriberRegistry.get() to access the instance'
        )
        self._subscribers = ()
        self._span_level = INFO
        self._logged_error_methods = WeakKeyDictionary()

    def subscribe(self, subscriber: Subscriber) -> None:
        """Register a subscriber to receive instrumentation callbacks; idempotent."""
        with self._lock:
            if all(s is not subscriber for s in self._subscribers):
                self._subscribers = (*self._subscribers, subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove a previously registered subscriber; a no-op if it isn't registered."""
        with self._lock:
            self._subscribers = tuple(s for s in self._subscribers if s is not subscriber)

    def _log_subscriber_error(self, subscriber: Subscriber, method: str, exc: Exception) -> None:
        # warn once per (subscriber, method), then drop to debug to avoid log storms from per-row failures
        with self._lock:
            logged = self._logged_error_methods.setdefault(subscriber, set())
            level = logging.DEBUG if method in logged else logging.WARNING
            logged.add(method)
        _logger.log(level, f'instrumentation subscriber {type(subscriber).__name__}.{method}() failed: {exc!r}')


def active() -> bool:
    """True if any subscriber is registered; use to guard computation that is only needed for telemetry."""
    return len(SubscriberRegistry.get()._subscribers) > 0


def set_span_level(level: int) -> None:
    """Set the global span emission threshold; spans with level < threshold are suppressed."""
    SubscriberRegistry.get()._span_level = level


def current_span() -> SpanHandle | None:
    return _current_span.get()


def _resolve_attrs(attrs: HookAttrs) -> dict[str, Any] | None:
    if not callable(attrs):
        return attrs
    try:
        return attrs()
    except Exception as e:
        _logger.debug(f'attribute callable failed: {e!r}')
        return None


def span_start(
    name: str,
    *,
    level: int = INFO,
    parent: SpanHandle | None = None,
    set_current: bool = False,
    attrs: HookAttrs = None,
) -> SpanHandle | None:
    """Start a span and return its handle; None if no subscribers are registered or the span is suppressed.

    Spans below the level threshold are suppressed, as are non-set_current spans with no ambient ancestor;
    descendants of a suppressed span parent to the ambient span. set_current=True makes the span the
    ambient parent (a ContextVar, so it propagates into asyncio tasks) and requires that span_end() runs
    on the same thread/context; use capture_context()/restore_context()/exit_context() for explicit
    thread handoffs.
    """
    env = SubscriberRegistry.get()
    subs = env._subscribers
    if not subs:
        return None
    if parent is None:
        parent = _current_span.get()
    # only operation spans (set_current=True) may be roots: spans reported from inside an operation that
    # carries no span (eg, a bare query) are suppressed rather than emitted as orphan roots. suppressed
    # spans return None; their descendants fall back to the ambient span via the parent lookup above
    if level < env._span_level or (parent is None and not set_current):
        return None
    attrs = _resolve_attrs(attrs)
    tokens: list[Any] = []
    for s in subs:
        parent_token: Any = None
        parent_idx = (
            next((i for i, parent_sub in enumerate(parent.subs) if parent_sub is s), None)
            if parent is not None
            else None
        )
        if parent_idx is not None:
            # a subscriber that registered after the parent started has no parent token (its span is a root)
            parent_token = parent.tokens[parent_idx]
        try:
            tokens.append(s.on_span_start(name, parent_token, attrs, set_current))
        except Exception as e:
            env._log_subscriber_error(s, 'on_span_start', e)
            tokens.append(None)
    handle = SpanHandle(subs, tuple(tokens))
    if set_current:
        handle.cv_token = _current_span.set(handle)
    return handle


def span_end(handle: SpanHandle | None, *, exc: BaseException | None = None, attrs: HookAttrs = None) -> None:
    """End a span started with span_start(); a no-op for None handles."""
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
            SubscriberRegistry.get()._log_subscriber_error(s, 'on_span_end', e)


def add_attrs(handle: SpanHandle | None, **attrs: Any) -> None:
    """Attach attrs ('pxt.'-prefixed, None values skipped) to be reported when the span ends.

    Argument computation is eager; call sites should guard with `if handle is not None:` (or
    `telemetry.active()`) when the values are expensive to produce.
    """
    if not isinstance(handle, SpanHandle):
        return
    if handle.pending_attrs is None:
        handle.pending_attrs = {}
    handle.pending_attrs.update({f'pxt.{k}': v for k, v in attrs.items() if v is not None})


def emit(handle: SpanHandle | None, name: str, **attrs: Any) -> None:
    """Attach a discrete event to the given span; a no-op for None handles.

    Keyword attrs get a 'pxt.' prefix; None values are skipped.
    """
    if not isinstance(handle, SpanHandle):
        return
    prefixed = {f'pxt.{k}': v for k, v in attrs.items() if v is not None}
    for s, token in zip(handle.subs, handle.tokens, strict=True):
        try:
            s.on_event(token, name, prefixed)
        except Exception as e:
            SubscriberRegistry.get()._log_subscriber_error(s, 'on_event', e)


class Counter:
    """A monotonic count of things; declare once at module level with counter().

    `name`/`unit` identify the backend instrument (unit is a UCUM code, e.g. '{row}', 'By', 's').
    """

    __slots__ = ('name', 'unit')

    def __init__(self, name: str, unit: str) -> None:
        self.name = name
        self.unit = unit

    def add(self, value: int | float, **attrs: Any) -> None:
        """Add value to the counter; keyword attrs get a 'pxt.' prefix, None values are skipped.

        Attrs become metric dimensions (one time series per distinct value combination), so only pass
        low-cardinality values (table path, UDF name); ids, urls, and versions belong on spans.
        """
        env = SubscriberRegistry.get()
        subs = env._subscribers
        if not subs:
            return
        prefixed = {f'pxt.{k}': v for k, v in attrs.items() if v is not None}
        for s in subs:
            try:
                s.on_counter_add(self, value, prefixed)
            except Exception as e:
                env._log_subscriber_error(s, 'on_counter_add', e)


class Histogram:
    """A distribution of per-observation values; declare once at module level with histogram().

    Record individual observations only (a call's latency, a file's size); recording pre-aggregated
    sums produces meaningless percentiles. `boundaries` advises the backend's bucket layout for
    backends whose defaults don't fit the recorded value range.
    """

    __slots__ = ('boundaries', 'name', 'unit')

    def __init__(self, name: str, unit: str, boundaries: tuple[float, ...] | None = None) -> None:
        self.name = name
        self.unit = unit
        self.boundaries = boundaries

    def record(self, value: int | float, **attrs: Any) -> None:
        """Record one observation; keyword attrs get a 'pxt.' prefix, None values are skipped.

        The same low-cardinality rule as Counter.add() applies to attrs.
        """
        env = SubscriberRegistry.get()
        subs = env._subscribers
        if not subs:
            return
        prefixed = {f'pxt.{k}': v for k, v in attrs.items() if v is not None}
        for s in subs:
            try:
                s.on_histogram_record(self, value, prefixed)
            except Exception as e:
                env._log_subscriber_error(s, 'on_histogram_record', e)


def counter(name: str, unit: str = '') -> Counter:
    """Declare a counter instrument; module-level, once per metric name."""
    return Counter(name, unit)


def histogram(name: str, unit: str = '', boundaries: tuple[float, ...] | None = None) -> Histogram:
    """Declare a histogram instrument; module-level, once per metric name."""
    return Histogram(name, unit, boundaries)


@contextlib.contextmanager
def span(
    name: str, *, level: int = INFO, parent: SpanHandle | None = None, set_current: bool = False, **attrs: Any
) -> Iterator[SpanHandle | None]:
    """Context-manager sugar for a lexical-block span.

    Keyword attrs get a 'pxt.' prefix; None values are skipped.
    """
    if not SubscriberRegistry.get()._subscribers:
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


def func_span() -> SpanHandle | None:
    """Handle of the innermost enclosing spanned() span, for use with add_attrs().

    None if there is no enclosing spanned() function or its span was suppressed (add_attrs() accepts None).
    """
    return _func_span.get()


def spanned(name: str, *, level: int = INFO, set_current: bool = False) -> Callable[[F], F]:
    """Decorator form of span() for a function whose entire body is one span.

    The span's handle isn't lexically available inside the function; use `add_attrs(func_span(), ...)` to
    attach attributes to it.
    """

    def decorator(fn: F) -> F:
        # a generator/coroutine function returns immediately, which would end the span before any work runs
        assert not inspect.iscoroutinefunction(fn) and not inspect.isgeneratorfunction(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not SubscriberRegistry.get()._subscribers:
                return fn(*args, **kwargs)
            with span(name, level=level, set_current=set_current) as handle:
                token = _func_span.set(handle)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _func_span.reset(token)

        return cast(F, wrapper)

    return decorator


@dataclasses.dataclass(slots=True)
class CtxSnapshot:
    """Ambient instrumentation state captured by capture_context() for handoff to a worker thread."""

    subs: tuple[Subscriber, ...]
    span: SpanHandle | None
    sub_ctxs: tuple[Any, ...]


_CtxToken = tuple[CtxSnapshot, Token[SpanHandle | None], tuple[Any, ...]]


def capture_context() -> CtxSnapshot | None:
    """Snapshot the ambient instrumentation state; call on the spawning thread."""
    subs = SubscriberRegistry.get()._subscribers
    if not subs:
        return None
    sub_ctxs: list[Any] = []
    for s in subs:
        try:
            sub_ctxs.append(s.capture_context())
        except Exception as e:
            SubscriberRegistry.get()._log_subscriber_error(s, 'capture_context', e)
            sub_ctxs.append(None)
    return CtxSnapshot(subs, _current_span.get(), tuple(sub_ctxs))


def restore_context(snapshot: CtxSnapshot | None) -> _CtxToken | None:
    """Restore a capture_context() snapshot; call on the worker thread. The result is passed to exit_context()."""
    if snapshot is None:
        return None
    cv_token = _current_span.set(snapshot.span)
    sub_tokens: list[Any] = []
    for s, ctx in zip(snapshot.subs, snapshot.sub_ctxs, strict=True):
        try:
            sub_tokens.append(s.restore_context(ctx))
        except Exception as e:
            SubscriberRegistry.get()._log_subscriber_error(s, 'restore_context', e)
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
            SubscriberRegistry.get()._log_subscriber_error(s, 'exit_context', e)
    try:
        _current_span.reset(cv_token)
    except Exception as e:
        _logger.debug(f'failed to reset ambient span: {e!r}')
