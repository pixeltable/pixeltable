"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `JsonType`.

Example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(pxt.functions.json.make_list()).collect()
```
"""

from typing import Any

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.utils.code import local_public_names


@pxt.uda(
    update_types=[pxt.JsonType(nullable=True)],
    value_type=pxt.JsonType(),
    requires_order_by=False,
    allows_window=False,
)
class make_list(pxt.Aggregator):
    """
    Collects arguments into a list.
    """
    def __init__(self):
        self.output: list[Any] = []

    def update(self, obj: Any) -> None:
        if obj is None:
            return
        self.output.append(obj)

    def value(self) -> list[Any]:
        return self.output


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
