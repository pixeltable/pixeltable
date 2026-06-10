from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Sequence
from uuid import UUID

import cloudpickle  # type: ignore[import-untyped]

import pixeltable.exceptions as excs
from pixeltable.runtime import get_runtime

from .function import Function
from .runtime_adapter import RuntimeAdapter, dumps_by_value, get_runtime_adapter, modal_resource_pool
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import exprs


def _run_maybe_off_loop(call: Callable[[], Any]) -> Any:
    """Run a blocking external-runtime call, hopping to a worker thread if an event loop is already running.

    The synchronous query-time path (e.g. embedding-index similarity) can reach `exec()` from inside a coroutine
    running on the executor's event loop. Some external SDKs (Modal) bridge their sync API to async via their own
    event loop, which can conflict when invoked from a thread that already has a running loop. Off-loading to a fresh
    thread keeps that bridge on a clean thread. When no loop is running (the common direct-call case), we call inline.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return call()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(call).result()


class CallableFunction(Function):
    """Pixeltable Function backed by a Python Callable.

    CallableFunctions come in two flavors:
    - references to lambdas and functions defined in notebooks, which are pickled and serialized to the store
    - functions that are defined in modules are serialized via the default mechanism
    """

    py_fns: list[Callable]
    self_name: str | None
    batch_size: int | None
    gpu: str | None
    # Modal runtime image hints (only meaningful when `gpu` is set). `image` is a base container image reference;
    # `apt` is a list of system packages (apt) and `pip` a list of pip packages to install in the remote image.
    image: str | None
    apt: list[str] | None
    pip: list[str] | None

    def __init__(
        self,
        signatures: list[Signature],
        py_fns: list[Callable],
        self_path: str | None = None,
        self_name: str | None = None,
        batch_size: int | None = None,
        is_method: bool = False,
        is_property: bool = False,
        is_deterministic: bool = True,
        gpu: str | None = None,
        image: str | None = None,
        apt: list[str] | None = None,
        pip: list[str] | None = None,
    ):
        assert len(signatures) > 0
        assert len(signatures) == len(py_fns)
        if self_path is None and len(signatures) > 1:
            raise excs.Error('Multiple signatures are only allowed for module UDFs (not locally defined UDFs)')
        self.py_fns = py_fns
        self.self_name = self_name
        self.batch_size = batch_size
        self.gpu = gpu
        self.image = image
        self.apt = apt
        self.pip = pip
        self.__doc__ = self.py_fns[0].__doc__
        super().__init__(
            signatures,
            self_path=self_path,
            is_method=is_method,
            is_property=is_property,
            is_deterministic=is_deterministic,
        )
        if gpu is not None:
            # Routing to Modal is intrinsic to the function: deriving it here (rather than in the @udf decorator)
            # ensures that reloaded/deserialized functions (via from_store()) route to Modal identically.
            pool = modal_resource_pool(gpu)
            self.resource_pool(lambda: pool)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        assert len(self.py_fns) > signature_idx
        self.py_fns = [self.py_fns[signature_idx]]

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.py_fn)

    @property
    def _runtime_adapter(self) -> RuntimeAdapter | None:
        """The external runtime adapter for this UDF, or None if it runs in-process.

        Keying on `self.gpu` keeps the common (pool-less) path a single attribute check; only GPU-hinted UDFs
        consult the shared adapter registry. This makes synchronous execution (e.g. embedding-index query-time
        embedding) route to the external runtime identically to the async dataflow path.
        """
        if self.gpu is None:
            return None
        return get_runtime_adapter(modal_resource_pool(self.gpu))

    def comment(self) -> str | None:
        return inspect.getdoc(self.py_fns[0])

    @property
    def py_fn(self) -> Callable:
        assert not self.is_polymorphic
        return self.py_fns[0]

    async def aexec(self, *args: Any, **kwargs: Any) -> Any:
        assert not self.is_polymorphic
        assert self.is_async
        if self.is_batched:
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result = await self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        else:
            return await self.py_fn(*args, **kwargs)

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        assert not self.is_polymorphic
        adapter = self._runtime_adapter
        if self.is_batched:
            if adapter is not None:
                # Mirror the local batched-singleton path on the remote runtime: pack every argument into a
                # singleton list and let invoke_batch (via create_batch_kwargs) split constant vs batched kwargs.
                batched_args = [[arg] for arg in args]
                singleton_kwargs = {k: [v] for k, v in kwargs.items()}
                batch_result = _run_maybe_off_loop(lambda: adapter.invoke_batch(self, batched_args, singleton_kwargs))
                assert len(batch_result) == 1
                return batch_result[0]
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result: list[Any]
            if inspect.iscoroutinefunction(self.py_fn):
                result = get_runtime().run_coro(self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs))
            else:
                result = self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        elif adapter is not None:
            # Query-time embedding (EmbeddingIndex.similarity_clause) reaches here while a read transaction is open;
            # this is one remote round-trip inside a read-only xact (no row locks), which is acceptable.
            return _run_maybe_off_loop(lambda: adapter.invoke(self, args, kwargs))
        elif inspect.iscoroutinefunction(self.py_fn):
            return get_runtime().run_coro(self.py_fn(*args, **kwargs))
        else:
            return self.py_fn(*args, **kwargs)

    async def aexec_batch(self, *args: Any, **kwargs: Any) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        assert self.is_async
        assert not self.is_polymorphic
        # Unpack the constant parameters
        constant_kwargs, batched_kwargs = self.create_batch_kwargs(kwargs)
        return await self.py_fn(*args, **constant_kwargs, **batched_kwargs)

    def exec_batch(self, args: list[Any], kwargs: dict[str, Any]) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        assert not self.is_polymorphic
        assert not self.is_async
        adapter = self._runtime_adapter
        if adapter is not None:
            return _run_maybe_off_loop(lambda: adapter.invoke_batch(self, args, kwargs))
        # Unpack the constant parameters
        constant_kwargs, batched_kwargs = self.create_batch_kwargs(kwargs)
        return self.py_fn(*args, **constant_kwargs, **batched_kwargs)

    def create_batch_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
        """Converts kwargs containing lists into constant and batched kwargs in the format expected by a batched udf."""
        constant_param_names = [p.name for p in self.signature.constant_parameters]
        constant_kwargs = {k: v[0] for k, v in kwargs.items() if k in constant_param_names}
        batched_kwargs = {k: v for k, v in kwargs.items() if k not in constant_param_names}
        return constant_kwargs, batched_kwargs

    def get_batch_size(self, *args: Any, **kwargs: Any) -> int | None:
        return self.batch_size

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def overload(self, fn: Callable) -> CallableFunction:
        if self.self_path is None:
            raise excs.Error('`overload` can only be used with module UDFs (not locally defined UDFs)')
        if self.is_method or self.is_property:
            raise excs.Error('`overload` cannot be used with `is_method` or `is_property`')
        if self._has_resolved_fns:
            raise excs.Error('New `overload` not allowed after the UDF has already been called')
        if self._conditional_return_type is not None:
            raise excs.Error('New `overload` not allowed after a conditional return type has been specified')
        sig = Signature.create(fn)
        self.signatures.append(sig)
        self.py_fns.append(fn)
        return self

    def _as_dict(self) -> dict:
        if self.self_path is None:
            # this is not a module function
            assert not self.is_method and not self.is_property
            from .function_registry import FunctionRegistry

            id = FunctionRegistry.get().create_stored_function(self)
            return {'id': id.hex}
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'id' in d:
            from .function_registry import FunctionRegistry

            return FunctionRegistry.get().get_stored_function(UUID(hex=d['id']))
        return super()._from_dict(d)

    def to_store(self) -> tuple[dict, bytes]:
        assert not self.is_polymorphic  # multi-signature UDFs not allowed for stored fns
        md = {
            'signature': self.signature.as_dict(),
            'batch_size': self.batch_size,
            'gpu': self.gpu,
            'image': self.image,
            'apt': self.apt,
            'pip': self.pip,
        }
        # Pickle by value so a UDF that references sibling module-level helpers reloads in a process that lacks its
        # defining module (consistent with how the same UDF is serialized for the external runtime).
        return md, dumps_by_value(self.py_fn)

    @classmethod
    def from_store(cls, name: str | None, md: dict, binary_obj: bytes) -> Function:
        py_fn = cloudpickle.loads(binary_obj)
        assert callable(py_fn)
        sig = Signature.from_dict(md['signature'])
        batch_size = md['batch_size']
        gpu = md.get('gpu')
        image = md.get('image')
        apt = md.get('apt')
        pip = md.get('pip')
        return CallableFunction(
            [sig], [py_fn], self_name=name, batch_size=batch_size, gpu=gpu, image=image, apt=apt, pip=pip
        )

    def validate_call(self, bound_args: dict[str, 'exprs.Expr']) -> None:
        from pixeltable import exprs

        super().validate_call(bound_args)
        if self.is_batched:
            signature = self.signatures[0]
            for param in signature.constant_parameters:
                # Check that constant parameters map to constant arguments. It's ok for the argument to be a Variable,
                # since in that case the FunctionCall is part of an unresolved template; the check will be done again
                # when the template is fully resolved.
                if param.name in bound_args and not isinstance(bound_args[param.name], (exprs.Literal, exprs.Variable)):
                    raise ValueError(f'{self.display_name}(): parameter {param.name} must be a constant value')

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'
