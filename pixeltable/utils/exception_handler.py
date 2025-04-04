import logging
import sys
from typing import Callable, TypeVar

R = TypeVar('R')


def _is_in_exception():
    """
    Returns true when running in exception context.
    """
    current_exception = sys.exc_info()[1]
    return current_exception is not None


def run_cleanup_on_exception(cleanup_func: Callable[..., R], *args, **kwargs):
    """
    Runs cleanup only when running in exception context.
    """
    if _is_in_exception():
        run_cleanup(cleanup_func, *args, raise_error=False, **kwargs)


def run_cleanup(cleanup_func: Callable[..., R], *args, raise_error=True, **kwargs):
    """
    Runs a cleanup function and if interrupted will try to run it again.
    Args:
        cleanup_func: cleanup function to run must be idempotent.
        raise_error: raise an exception if an error occurs during cleanup.
    """
    try:
        logging.debug(f'Running cleanup function: {cleanup_func.__name__!r}')
        cleanup_func(*args, **kwargs)
    except KeyboardInterrupt as interrupt:
        # Save original exception and re-attempt cleanup
        original_exception = interrupt
        logging.debug(f'Cleanup {cleanup_func.__name__!r} interrupted, retrying')
        try:
            cleanup_func(*args, **kwargs)
        except Exception as e:
            # Suppress this exception
            logging.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e}')
        raise KeyboardInterrupt from original_exception
    except Exception as e:
        logging.error(f'Cleanup {cleanup_func.__name__!r} failed with exception {e}')
        if raise_error:
            raise e
