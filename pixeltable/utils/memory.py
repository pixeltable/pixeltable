"""Release process memory held by cached models and the glibc heap.

The test suite calls this between tests, and the proxy daemon's test-only /release_memory
endpoint calls it in the daemon process, so a worker and its daemon both return RAM to the OS.
"""

from __future__ import annotations

import ctypes
import gc
import platform
from typing import Any


def clear_model_caches() -> None:
    """Drop the in-process function-module model caches.

    llama.cpp instances are closed first so their native allocations are released; the other
    caches are plain dicts of Python objects.
    """
    from pixeltable.functions import huggingface, llama_cpp, vllm, whisper, whisperx, yolox

    llama_cpp.cleanup()
    caches: tuple[dict[Any, Any], ...] = (
        huggingface._model_cache,
        huggingface._processor_cache,
        vllm._model_cache,
        whisper._model_cache,
        whisperx._model_cache,
        whisperx._alignment_model_cache,
        whisperx._diarization_model_cache,
        yolox._model_cache,
        yolox._processor_cache,
    )
    for cache in caches:
        cache.clear()


def return_heap_to_os() -> None:
    """Collect garbage and, on glibc, return freed heap pages to the OS.

    glibc otherwise keeps freed arenas in the process, which is what exhausted ubuntu-large
    after tests that allocate large numpy arrays (PXT-1716).
    """
    gc.collect()
    if platform.libc_ver()[0] == 'glibc':
        ctypes.CDLL('libc.so.6').malloc_trim(0)


def release_process_memory() -> None:
    """Drop cached models and return freed heap to the OS."""
    clear_model_caches()
    return_heap_to_os()
