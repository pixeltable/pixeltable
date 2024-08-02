from typing import Any

import pixeltable.func as func
from pixeltable.type_system import StringType
from pixeltable.utils.code import local_public_names


@func.udf(return_type=StringType(), param_types=[StringType()], member_access='method')
def str_format(format_str: str, *args: Any, **kwargs: Any) -> str:
    """Return a formatted version of format_str, using substitutions from args and kwargs:
    - {<int>} will be replaced by the corresponding element in args
    - {<key>} will be replaced by the corresponding value in kwargs
    """
    return format_str.format(*args, **kwargs)


@func.udf
def contains(string: str, substr: str) -> bool:
    return substr in string


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
