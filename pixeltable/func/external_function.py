import inspect
from typing import List, Dict, Any, Optional, Callable
import abc

from .function import Function
from .function_md import FunctionMd


class ExternalFunction(Function):
    """Base class for functions that are evaluated outside of RowBuilder"""
    @abc.abstractmethod
    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        """Return the batch size for the given arguments, or None if the batch size is unknown.
        args/kwargs might be empty
        """
        raise NotImplementedError

    @abc.abstractmethod
    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        """Invoke the function for the given batch and return a batch of results"""
        raise NotImplementedError


class ExplicitExternalFunction(ExternalFunction):

    def __init__(self, md: FunctionMd, batch_size: Optional[int], invoker_fn: Callable, constant_params: List[str], display_name: str):
        super().__init__(md, display_name=display_name, py_signature=inspect.signature(invoker_fn))
        self.batch_size = batch_size
        self.invoker_fn = invoker_fn
        self.constant_params = constant_params

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return self.batch_size

    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        """Invoke the function for the given batch and return a batch of results"""
        kwargs = {k: v[0] for k, v in kwarg_batches.items() if k in self.constant_params}
        kwarg_batches = {k: v for k, v in kwarg_batches.items() if k not in self.constant_params}
        return self.invoker_fn(*arg_batches, **kwargs, **kwarg_batches)
