import logging
import sys
from typing import Any, Callable, Optional, TypeVar

R = TypeVar('R')


def _is_in_exception() -> bool:
    """
    Check if code is currently executing within an exception context.
    """
    current_exception = sys.exc_info()[1]
    return current_exception is not None


def run_cleanup_on_exception(cleanup_func: Callable[..., R], *args: Any, **kwargs: Any) -> Optional[R]:
    """
    Runs cleanup only when running in exception context.

    The function `run_cleanup_on_exception()` should be used to clean up resources when an operation fails.
    This is typically done using a try, except, and finally block, with the resource cleanup logic placed within
    the except block. However, this pattern may not handle KeyboardInterrupt exceptions.
    To ensure that resources are always cleaned up at least once when an exception or KeyboardInterrupt occurs,
    create an idempotent function for cleaning up resources and pass it to the `run_cleanup_on_exception()` function
    from the finally block.
    """
    if _is_in_exception():
        return run_cleanup(cleanup_func, *args, raise_error=False, **kwargs)
    return None


def run_cleanup(cleanup_func: Callable[..., R], *args: Any, raise_error: bool = True, **kwargs: Any) -> Optional[R]:
    """
    Runs a cleanup function. If interrupted, retry cleanup.
    The `run_cleanup()` function ensures that the `cleanup_func()` function executes at least once.
    If the `cleanup_func()` is interrupted during execution, it will be retried.

    Args:
        cleanup_func: an idempotent function
        raise_error: raise an exception if an error occurs during cleanup.
    """
    try:
        logging.debug(f'Running cleanup function: {cleanup_func.__name__!r}')
        return cleanup_func(*args, **kwargs)
    except KeyboardInterrupt as interrupt:
        # Save original exception and re-attempt cleanup
        original_exception = interrupt
        logging.debug(f'Cleanup {cleanup_func.__name__!r} interrupted, retrying')
        try:
            return cleanup_func(*args, **kwargs)
        except Exception as e:
            # Suppress this exception
            logging.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e}')
        raise KeyboardInterrupt from original_exception
    except Exception as e:
        logging.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e}')
        if raise_error:
            raise e
    return None
