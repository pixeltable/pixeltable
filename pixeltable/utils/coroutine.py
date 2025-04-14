import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

T = TypeVar('T')


# TODO This is a temporary hack to be able to run async UDFs in contexts that are not properly handled by the existing
#   scheduler logic (e.g., as an embedding function as part of a similarity lookup). Once the scheduler is fully
#   general, it can be removed.


def run_coroutine_synchronously(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """
    Runs the given coroutine synchronously, even if called in the context of a running event loop.
    """

    def run_in_new_loop() -> T:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop; just call `asyncio.run()`
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
