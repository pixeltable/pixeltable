from typing import Any

from pixeltable.type_system import StringType
import pixeltable.func as func


@func.udf(return_type=StringType(), param_types=[StringType()])
def str_format(format_str: str, *args: Any, **kwargs: Any) -> str:
    """ Return a formatted version of format_str, using substitutions from args and kwargs:
    - {<int>} will be replaced by the corresponding element in args
    - {<key>} will be replaced by the corresponding value in kwargs
    """
    return format_str.format(*args, **kwargs)
