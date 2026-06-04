"""
Pixeltable UDFs for `ArrayType`.

Example:

>>> import pixeltable as pxt
>>> t = pxt.get_table(...)
>>> t.select(t.array_col.to_list()).collect()
"""

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def to_list(self: pxt.Array) -> pxt.Json:
    """
    Convert an array to a nested Python list.

    Equivalent to numpy's [`ndarray.tolist()`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html).
    Useful for exporting to systems that lack a fixed-shape tensor type (e.g. Iceberg).

    Example:

        >>> t.select(t.array_col.to_list()).collect()
    """
    return self.tolist()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
