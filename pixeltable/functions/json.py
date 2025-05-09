"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `JsonType`.

Example:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf

t = pxt.get_table(...)
t.select(pxtf.json.make_list(t.json_col)).collect()
```
"""

from typing import Any

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.uda
class make_list(pxt.Aggregator):
    """
    Collects arguments into a list.
    """

    def __init__(self) -> None:
        self.output: list[Any] = []

    def update(self, obj: pxt.Json) -> None:
        if obj is None:
            return
        self.output.append(obj)

    def value(self) -> list[Any]:
        return self.output


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
