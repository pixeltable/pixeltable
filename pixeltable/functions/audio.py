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

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def get_metadata(audio: pxt.Audio) -> dict:
    """
    Gets various metadata associated with an audio file and returns it as a dictionary.
    """
    return pxt.functions.video._get_metadata(audio)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
