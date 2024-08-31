"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `AudioType`.

Example:
```python
import pixeltable as pxt
import pixeltable.functions as pxtf

t = pxt.get_table(...)
t.select(pxtf.audio.get_metadata()).collect()
```
"""

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.utils.code import local_public_names


@func.udf(return_type=ts.JsonType(nullable=False), param_types=[ts.AudioType(nullable=False)], is_method=True)
def get_metadata(video: str) -> dict:
    """
    Gets various metadata associated with a video file and returns it as a dictionary.
    """
    import pixeltable.functions as pxtf
    return pxtf.video._get_metadata(video)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
