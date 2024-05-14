import inspect
from typing import List, Dict, Any, Optional, Callable

from .function import Function
from .signature import Signature


class BatchedFunction(Function):
    """A Pixeltable `Function` that accepts batches of data as input. The batches are orchestrated by
    Pixeltable so that each batch is processed as a unit.
    """

    def __init__(self, signature: Signature, batch_size: Optional[int], invoker_fn: Callable, self_path: str):
        super().__init__(signature=signature, py_signature=inspect.signature(invoker_fn), self_path=self_path)
        self.batch_size = batch_size
        self.invoker_fn = invoker_fn

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return self.batch_size

    def invoke(self, arg_batches: list[list[Any]], kwarg_batches: dict[str, list[Any]]) -> list[Any]:
        """Invoke the function for the given batch and return a batch of results"""
        constant_param_names = [p.name for p in self.signature.constant_parameters]
        kwargs = {k: v[0] for k, v in kwarg_batches.items() if k in constant_param_names}
        kwarg_batches = {k: v for k, v in kwarg_batches.items() if k not in constant_param_names}
        return self.invoker_fn(*arg_batches, **kwargs, **kwarg_batches)

    def validate_call(self, bound_args: dict[str, Any]) -> None:
        """Verify constant parameters"""
        import pixeltable.exprs as exprs
        for param in self.signature.constant_parameters:
            if param.name in bound_args and isinstance(bound_args[param.name], exprs.Expr):
                raise ValueError(
                    f'{self.display_name}(): '
                    f'parameter {param.name} must be a constant value, not a Pixeltable expression'
                )

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        arg_batches = [[arg] for arg in args]
        kwarg_batches = {k: [v] for k, v in kwargs.items()}
        batch_result = self.invoke(arg_batches, kwarg_batches)
        assert len(batch_result) == 1
        return batch_result[0]
