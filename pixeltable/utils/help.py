from typing import Any

import pixeltable.func as func


def help(obj: Any) -> None:
    """Returns help text for the given object."""
    if isinstance(obj, func.Function):
        print(obj.help_str())
    else:
        print(__builtins__.help(obj))
