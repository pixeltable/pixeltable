import asyncio
import threading
from typing import Any, Coroutine, TypeVar

from pixeltable.env import Env

T = TypeVar('T')

# TODO This is a temporary hack to be able to run async UDFs in contexts that are not properly handled by the existing
#   scheduler logic (e.g., as an embedding function as part of a similarity lookup). Once the scheduler is fully
#   general, it can be removed.


def run_coroutine_synchronously(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """
    Runs the given coroutine synchronously, even if called in the context of a running event loop.
    """
    loop = Env.get().event_loop

    if threading.current_thread() is threading.main_thread():
        return loop.run_until_complete(coroutine)
    else:
        # Not in main thread, use run_coroutine_threadsafe
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout)
