from typing import List, Dict, Any, Optional
import abc

from .function import Function


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
