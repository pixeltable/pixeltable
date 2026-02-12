"""
Pixeltable UDFs for array/vector operations.
"""

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf
def identity(vec: pxt.Array[(None,), pxt.Float]) -> pxt.Array[(None,), pxt.Float]:
    """
    Identity function for array/vector inputs. Returns the input unchanged.

    Used as the embedding function for array columns in embedding indexes when you want
    to do similarity search directly on stored vectors.
    """
    return vec


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
