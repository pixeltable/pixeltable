from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import threading
from contextlib import ExitStack
from typing import Any, Sequence

import cloudpickle  # type: ignore[import-untyped]

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.config import Config

from .callable_function import CallableFunction
from .runtime_adapter import MODAL_RESOURCE_POOL_PREFIX, RuntimeAdapter, dumps_by_value


class _FilePayload:
    """Wire wrapper for a file-backed media argument (video/audio/document).

    The remote container does not have access to the client's local filesystem, so file-path-backed media is sent
    as inline bytes and materialized to a temporary file on the remote side, where the UDF receives a valid local
    path. This is the simplest correct strategy; for very large media a Modal volume or URL hand-off would be more
    efficient, but those require additional infrastructure and are out of scope here.
    """

    __slots__ = ('data', 'suffix')

    def __init__(self, suffix: str, data: bytes):
        self.suffix = suffix
        self.data = data

    def materialize(self) -> str:
        """Write the payload to a temp file in the remote container and return its path."""
        fd, path = tempfile.mkstemp(suffix=self.suffix)
        with os.fdopen(fd, 'wb') as f:
            f.write(self.data)
        return path


def _decode_media(value: Any) -> Any:
    """Recursively materialize any _FilePayload values (including inside batched lists) into local file paths."""
    if isinstance(value, _FilePayload):
        return value.materialize()
    if isinstance(value, list):
        return [_decode_media(v) for v in value]
    return value


def _invoke_pickled(pickled_fn: bytes, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Modal entrypoint: unpickle and execute the UDF inside the remote container."""
    fn = cloudpickle.loads(pickled_fn)
    args = [_decode_media(a) for a in args]
    kwargs = {k: _decode_media(v) for k, v in kwargs.items()}
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def _encode_media_scalar(col_type: ts.ColumnType | None, value: Any) -> Any:
    """Encode a single value for the wire.

    File-backed media (video/audio/document) is passed to UDFs as a local file path; we replace it with inline
    bytes so the remote can reconstruct the file. Images are passed as in-memory PIL objects, which cloudpickle
    (used by Modal for argument serialization) handles directly, so they need no special treatment.
    """
    if value is None or col_type is None:
        return value
    is_file_media = col_type.is_video_type() or col_type.is_audio_type() or col_type.is_document_type()
    if is_file_media and isinstance(value, str) and os.path.isfile(value):
        with open(value, 'rb') as f:
            data = f.read()
        return _FilePayload(os.path.splitext(value)[1], data)
    return value


class ModalAdapter(RuntimeAdapter):
    """RuntimeAdapter implementation backed by Modal serverless functions."""

    RESOURCE_POOL_PREFIX = MODAL_RESOURCE_POOL_PREFIX
    APP_NAME = 'pixeltable-runtime-adapter'
    # Always present in the remote image so the entrypoint can unpickle and run the UDF.
    BASE_PIP_PACKAGES = ('cloudpickle', 'pixeltable')

    gpu: str
    _exit_stack: ExitStack
    _lock: threading.Lock
    # Remote Modal functions, keyed by image spec (image, apt, pip). One Modal app/image is built and its context
    # entered exactly once per distinct spec (not per row), so UDFs with different dependencies get their own image.
    _remote_fns: dict[tuple[str | None, tuple[str, ...], tuple[str, ...]], Any]
    # Cache of cloudpickle-serialized py_fns, keyed by id(fn). Pickling is expensive and the same UDF is invoked for
    # many rows, so we pickle each function once. The CallableFunction is held in the value to keep it alive and
    # guard against id() reuse after garbage collection.
    _pickled_cache: dict[int, tuple[CallableFunction, bytes]]

    def __init__(self, resource_pool: str):
        if not self.matches_resource_pool(resource_pool):
            raise excs.Error(f'Not a Modal resource pool: {resource_pool}')
        self.gpu = resource_pool.removeprefix(self.RESOURCE_POOL_PREFIX)
        self._exit_stack = ExitStack()
        self._lock = threading.Lock()
        self._remote_fns = {}
        self._pickled_cache = {}

    @classmethod
    def matches_resource_pool(cls, resource_pool: str) -> bool:
        return resource_pool.startswith(cls.RESOURCE_POOL_PREFIX)

    def invoke(self, fn: CallableFunction, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        # Mirrors CallableFunction.exec() for the non-batched case: the remote simply calls py_fn(*args, **kwargs).
        enc_args, enc_kwargs = self._encode_media(fn, list(args), dict(kwargs), batched=False)
        return self._invoke_remote(fn, enc_args, enc_kwargs)

    def invoke_batch(self, fn: CallableFunction, args: Sequence[Any], kwargs: dict[str, Any]) -> list[Any]:
        assert fn.is_batched
        # Mirror CallableFunction.exec_batch()/aexec_batch(): split constant vs batched kwargs here (where the
        # CallableFunction's signature is available), so the remote entrypoint stays a plain py_fn(*args, **kwargs).
        constant_kwargs, batched_kwargs = fn.create_batch_kwargs(dict(kwargs))
        combined_kwargs = {**constant_kwargs, **batched_kwargs}
        enc_args, enc_kwargs = self._encode_media(fn, list(args), combined_kwargs, batched=True)
        result = self._invoke_remote(fn, enc_args, enc_kwargs)
        assert isinstance(result, list)
        return result

    @staticmethod
    def _encode_media(
        fn: CallableFunction, args: list[Any], kwargs: dict[str, Any], batched: bool
    ) -> tuple[list[Any], dict[str, Any]]:
        """Replace file-backed media arguments with inline payloads, using the signature to identify media params.

        For batched calls, a batched parameter's value is a per-row list (encode each element); a constant
        parameter's value is a scalar.
        """
        params_by_pos = fn.signature.parameters_by_pos
        params_by_name = fn.signature.parameters

        def encode(col_type: ts.ColumnType | None, value: Any, is_list: bool) -> Any:
            if is_list and isinstance(value, list):
                return [_encode_media_scalar(col_type, v) for v in value]
            return _encode_media_scalar(col_type, value)

        enc_args = []
        for i, value in enumerate(args):
            param = params_by_pos[i] if i < len(params_by_pos) else None
            col_type = param.col_type if param is not None else None
            enc_args.append(encode(col_type, value, is_list=batched and (param is None or param.is_batched)))

        enc_kwargs = {}
        for k, value in kwargs.items():
            param = params_by_name.get(k)
            col_type = param.col_type if param is not None else None
            enc_kwargs[k] = encode(col_type, value, is_list=batched and param is not None and param.is_batched)

        return enc_args, enc_kwargs

    def _pickled_fn(self, fn: CallableFunction) -> bytes:
        key = id(fn)
        entry = self._pickled_cache.get(key)
        if entry is None or entry[0] is not fn:
            entry = (fn, self._dumps_by_value(fn.py_fn))
            self._pickled_cache[key] = entry
        return entry[1]

    def _dumps_by_value(self, py_fn: Any) -> bytes:
        """Serialize the UDF with sibling helpers inlined. Delegates to the shared `dumps_by_value` so Modal argument
        serialization and catalog `to_store()` behave identically. Results are cached by the caller (once per UDF)."""
        return dumps_by_value(py_fn)

    def close(self) -> None:
        """Exit all Modal app contexts that were entered. Best-effort and idempotent."""
        with self._lock:
            self._exit_stack.close()
            self._remote_fns.clear()

    def _invoke_remote(self, fn: CallableFunction, args: list[Any], kwargs: dict[str, Any]) -> Any:
        remote_fn = self._ensure_remote(fn)
        return remote_fn.remote(self._pickled_fn(fn), args, kwargs)

    def _ensure_remote(self, fn: CallableFunction) -> Any:
        """Return the remote Modal function for the UDF's image spec, building and entering its app context once.

        Idempotent and thread-safe: invoked from worker threads (via asyncio.to_thread), so concurrent first
        invocations for the same spec must build and enter the app context only once.
        """
        key = (fn.image, tuple(fn.apt) if fn.apt else (), tuple(fn.pip) if fn.pip else ())
        remote_fn = self._remote_fns.get(key)
        if remote_fn is not None:
            return remote_fn
        with self._lock:
            remote_fn = self._remote_fns.get(key)
            if remote_fn is not None:
                return remote_fn
            self._configure_auth()
            modal = self._modal()
            app = modal.App(self.APP_NAME)
            image = self._build_image(modal, fn.image, fn.apt, fn.pip)
            function_kwargs: dict[str, Any] = {'gpu': self.gpu, 'image': image}
            remote_fn = app.function(**function_kwargs)(_invoke_pickled)
            # Enter the app context and keep it open for the adapter's lifetime (not per row).
            self._exit_stack.enter_context(app.run())
            self._remote_fns[key] = remote_fn
            return remote_fn

    @classmethod
    def _build_image(cls, modal: Any, image: str | None, apt: list[str] | None, pip: list[str] | None) -> Any:
        base = modal.Image.from_registry(image) if image is not None else modal.Image.debian_slim()
        if apt:
            base = base.apt_install(*apt)
        return base.pip_install(*cls.BASE_PIP_PACKAGES, *(pip or []))

    @staticmethod
    def _configure_auth() -> None:
        """Route Modal credentials through Pixeltable's Config (env vars + $PIXELTABLE_HOME/config.toml), consistent
        with other providers. This is additive: if credentials are configured in Pixeltable, we surface them via
        Modal's expected env vars; otherwise Modal falls back to its own token file (`modal token set`).
        """
        # Config reads either the MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars or the [modal] section of config.toml.
        for key, modal_env_var in (('token_id', 'MODAL_TOKEN_ID'), ('token_secret', 'MODAL_TOKEN_SECRET')):
            value = Config.get().get_string_value(key, section='modal')
            if value is not None and modal_env_var not in os.environ:
                os.environ[modal_env_var] = value

    @staticmethod
    def _modal() -> Any:
        try:
            import modal
        except ImportError as exc:
            raise excs.Error(
                'The Modal SDK is required for `@pxt.udf(gpu=...)`; install it with `pip install modal`.'
            ) from exc
        return modal
