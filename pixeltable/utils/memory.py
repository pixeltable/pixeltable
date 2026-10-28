"""Drop cached models and return freed heap pages to the OS."""

from __future__ import annotations

import ctypes
import gc
import platform


def return_heap_to_os() -> None:
    """Collect garbage and, on glibc, return freed heap pages to the OS.

    glibc keeps freed arenas unless malloc_trim() runs.
    """
    gc.collect()
    if platform.libc_ver()[0] == 'glibc':
        ctypes.CDLL('libc.so.6').malloc_trim(0)


def release_process_memory() -> None:
    """Drop cached models and return freed heap to the OS."""
    from pixeltable.functions.util import clear_model_caches

    clear_model_caches()
    return_heap_to_os()
