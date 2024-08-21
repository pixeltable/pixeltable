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

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.utils.code import local_public_names


@func.uda(
    update_types=[ts.JsonType(nullable=True)],
    value_type=ts.JsonType(),
    requires_order_by=False,
    allows_window=False,
)
class make_list(func.Aggregator):
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
