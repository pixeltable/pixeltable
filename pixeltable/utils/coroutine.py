from typing import Any, Coroutine, TypeVar

from pixeltable.runtime import get_runtime

T = TypeVar('T')

# TODO This is a temporary hack to be able to run async UDFs in contexts that are not properly handled by the existing
#   scheduler logic (e.g., as an embedding function as part of a similarity lookup). Once the scheduler is fully
#   general, it can be removed.


def run_coroutine_synchronously(coroutine: Coroutine[Any, Any, T]) -> T:
    """
    Runs the given coroutine synchronously, even if called in the context of a running event loop.
    """
    return get_runtime().event_loop.run_until_complete(coroutine)
