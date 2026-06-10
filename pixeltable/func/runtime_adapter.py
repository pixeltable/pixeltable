from __future__ import annotations

import atexit
import logging
import os
import sys
import sysconfig
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import cloudpickle  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .callable_function import CallableFunction

_logger = logging.getLogger('pixeltable')

# Resource pool prefix used to route GPU-hinted UDFs to the Modal runtime. Shared between CallableFunction
# (which assigns the pool from a `gpu=` hint) and ModalAdapter/ModalScheduler (which match and execute it),
# so the encoding is defined in exactly one place.
MODAL_RESOURCE_POOL_PREFIX = 'modal:'


def modal_resource_pool(gpu: str) -> str:
    """Return the resource pool id for a UDF that should run on Modal with the given GPU spec."""
    return f'{MODAL_RESOURCE_POOL_PREFIX}{gpu}'


class RuntimeAdapter(ABC):
    """Executes Pixeltable functions on an external runtime."""

    @classmethod
    @abstractmethod
    def matches_resource_pool(cls, resource_pool: str) -> bool:
        """Return True if this adapter handles the given resource pool."""
        raise NotImplementedError

    @abstractmethod
    def invoke(self, fn: CallableFunction, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        """Invoke a scalar function call."""
        raise NotImplementedError

    @abstractmethod
    def invoke_batch(self, fn: CallableFunction, args: Sequence[Any], kwargs: dict[str, Any]) -> list[Any]:
        """Invoke a batched function call."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the adapter. Idempotent; default no-op."""


# Process-global cache of runtime adapters, keyed by resource pool. A single adapter instance per pool is shared
# between the dataflow executor (ModalScheduler) and synchronous/query-time calls (CallableFunction.exec), so the
# remote runtime (e.g. a Modal app/image) is set up once per process and reused, rather than rebuilt per executor run
# or per query.
_adapter_cache: dict[str, RuntimeAdapter] = {}
_adapter_lock = threading.Lock()


def _adapter_classes() -> list[type[RuntimeAdapter]]:
    # Imported lazily to avoid an import cycle: modal_adapter imports from this module.
    from .modal_adapter import ModalAdapter

    return [ModalAdapter]


def get_runtime_adapter(resource_pool: str | None) -> RuntimeAdapter | None:
    """Return the shared RuntimeAdapter for the given resource pool, or None if no external runtime handles it.

    Adapters are created lazily and cached for the lifetime of the process. Thread-safe.
    """
    if resource_pool is None:
        return None
    cached = _adapter_cache.get(resource_pool)
    if cached is not None:
        return cached
    with _adapter_lock:
        cached = _adapter_cache.get(resource_pool)
        if cached is not None:
            return cached
        for adapter_cls in _adapter_classes():
            if adapter_cls.matches_resource_pool(resource_pool):
                adapter = adapter_cls(resource_pool)  # type: ignore[call-arg]
                _adapter_cache[resource_pool] = adapter
                return adapter
    return None


def close_all_adapters() -> None:
    """Close and forget all cached runtime adapters. Best-effort; safe to call multiple times."""
    with _adapter_lock:
        for adapter in _adapter_cache.values():
            try:
                adapter.close()
            except Exception as exc:
                _logger.debug(f'Error closing runtime adapter: {exc}')
        _adapter_cache.clear()


# Release external-runtime resources (e.g. open Modal app contexts) at interpreter shutdown. Idempotent and a no-op
# when no adapters were created.
atexit.register(close_all_adapters)


# Serializes the global cloudpickle register/unregister_pickle_by_value() state change so that scoped by-value
# pickling (below) doesn't race with unrelated cloudpickle usage on other threads.
_pickle_lock = threading.Lock()


def _is_user_module(mod: Any) -> bool:
    """True if `mod` is user code (not stdlib/site-packages/__main__) and should be pickled by value.

    UDFs frequently reference sibling module-level helpers/constants. cloudpickle pickles such references by module
    path, which fails to resolve wherever the user's module is not installed (e.g. a Modal container, or a fresh
    process reloading the catalog). Registering the user's own module for pickle-by-value makes those siblings travel
    with the UDF. We deliberately exclude __main__ (cloudpickle already pickles it by value) and installed packages
    (present via pip, and pickling them by value would be wasteful/incorrect).
    """
    name = getattr(mod, '__name__', None)
    if name is None or name == '__main__':
        return False
    file = getattr(mod, '__file__', None)
    if not file:
        return False
    real = os.path.realpath(file)
    paths = sysconfig.get_paths()
    for key in ('stdlib', 'platstdlib', 'purelib', 'platlib'):
        base = paths.get(key)
        if base and real.startswith(os.path.realpath(base) + os.sep):
            return False
    return True


def dumps_by_value(py_fn: Any) -> bytes:
    """cloudpickle.dumps `py_fn`, temporarily registering its own (user) module for pickle-by-value so that sibling
    module-level helpers/constants are serialized inline rather than by an unresolvable module path.

    The register/dump/unregister is done under `_pickle_lock` so the global cloudpickle state change is scoped to this
    dump and does not leak into unrelated pickling. Shared by Modal argument serialization and catalog `to_store()`,
    so a UDF that references sibling helpers behaves identically on the remote runtime and on catalog reload.
    """
    mod = sys.modules.get(getattr(py_fn, '__module__', '') or '')
    register = mod is not None and _is_user_module(mod)
    with _pickle_lock:
        if register:
            cloudpickle.register_pickle_by_value(mod)
        try:
            return cloudpickle.dumps(py_fn)
        finally:
            if register:
                cloudpickle.unregister_pickle_by_value(mod)
