import inspect
from typing import List, Dict, Any, Optional, Callable
import abc

from .function import Function
from .signature import Signature


class BatchedFunction(Function):
    """Base class for functions that can run on batches"""

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


class ExplicitBatchedFunction(BatchedFunction):
    """
    A `BatchedFunction` that is defined by a signature and an explicit python
    `Callable`.
    """
    def __init__(self, signature: Signature, batch_size: Optional[int], invoker_fn: Callable, self_path: str):
        super().__init__(signature=signature, py_signature=inspect.signature(invoker_fn), self_path=self_path)
        self.batch_size = batch_size
        self.invoker_fn = invoker_fn

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return self.batch_size

    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        """Invoke the function for the given batch and return a batch of results"""
        constant_param_names = [p.name for p in self.signature.constant_parameters]
        kwargs = {k: v[0] for k, v in kwarg_batches.items() if k in constant_param_names}
        kwarg_batches = {k: v for k, v in kwarg_batches.items() if k not in constant_param_names}
        return self.invoker_fn(*arg_batches, **kwargs, **kwarg_batches)

    def validate_call(self, bound_args: Dict[str, Any]) -> None:
        """Verify constant parameters"""
        import pixeltable.exprs as exprs
        for param in self.signature.constant_parameters:
            if param.name in bound_args and isinstance(bound_args[param.name], exprs.Expr):
                raise ValueError(
                    f'{self.display_name}(): '
                    f'parameter {param.name} must be a constant value, not a Pixeltable expression'
                )
