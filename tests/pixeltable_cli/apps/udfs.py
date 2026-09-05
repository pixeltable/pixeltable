"""Udfs the corpus applications and the CLI tests share."""

import numpy as np

import pixeltable as pxt


@pxt.udf
def fail_on_zero(x: int) -> int:
    """Raise for x=0, so that a computed column over this stores an error for that row."""
    if x == 0:
        raise ValueError('fail')
    return x


@pxt.udf
def trivial_embed(s: str) -> pxt.Array[(8,), np.float32]:
    """Embed to a fixed vector, so that an index can be built without downloading a model."""
    return np.zeros(8, dtype=np.float32)


@pxt.udf
def dummy_embedding(text: str) -> pxt.Array[(32,), np.float32]:
    """Embed to a fixed 32 dimensions, deterministic in the text's length."""
    arr = np.zeros((32,), dtype=np.float32)
    arr[len(text) % 32] = 1
    return arr
