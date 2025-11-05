import logging
from typing import Any, Callable, TypeVar

R = TypeVar('R')

logger = logging.getLogger('pixeltable')


def run_cleanup(cleanup_func: Callable[..., R], *args: Any, raise_error: bool = True, **kwargs: Any) -> R | None:
    """
    Runs a cleanup function. If interrupted, retry cleanup.
    The `run_cleanup()` function ensures that the `cleanup_func()` function executes at least once.
    If the `cleanup_func()` is interrupted during execution, it will be retried.

    Args:
        cleanup_func: an idempotent function
        raise_error: raise an exception if an error occurs during cleanup.
    """
    try:
        logger.debug(f'Running cleanup function: {cleanup_func.__name__!r}')
        return cleanup_func(*args, **kwargs)
    except KeyboardInterrupt as interrupt:
        # Save original exception and re-attempt cleanup
        original_exception = interrupt
        logger.debug(f'Cleanup {cleanup_func.__name__!r} interrupted, retrying')
        try:
            return cleanup_func(*args, **kwargs)
        except Exception as e:
            # Suppress this exception
            logger.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e.__class__}: {e}')
        raise KeyboardInterrupt from original_exception
    except Exception as e:
        logger.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e.__class__}: {e}')
        if raise_error:
            raise e
    return None
