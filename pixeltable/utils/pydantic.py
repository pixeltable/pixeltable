import typing
from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Any, Union

import pydantic


def is_json_convertible(model: type[pydantic.BaseModel]) -> bool:
    """
    Determine if instances of a Pydantic model can be converted to valid JSON
    based on the type hints of its fields.
    """
    type_hints = typing.get_type_hints(model)
    return all(_type_is_json_convertible(field_type) for field_type in type_hints.values())


def _type_is_json_convertible(type_hint: Any) -> bool:
    """
    Recursively check if a type hint represents a JSON-compatible type.

    TODO: also allow ndarrays and PIL.Image.Image, once we support those within json structures.
    """
    if type_hint is type(None):
        return True
    if type_hint is Any:
        return False

    if type_hint in (str, int, float, bool, datetime):
        return True

    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return all(isinstance(member.value, (str, int, float, bool, type(None))) for member in type_hint)

    if isinstance(type_hint, type) and issubclass(type_hint, pydantic.BaseModel):
        return is_json_convertible(type_hint)

    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if origin in (Union, UnionType):
        return all(_type_is_json_convertible(arg) for arg in args)

    if origin in (list, tuple):
        return all(_type_is_json_convertible(arg) for arg in args) if len(args) > 0 else False

    if origin is dict:
        if len(args) != 2:
            # we can't tell what this is
            return False
        key_type, value_type = args
        # keys must be strings, values must be json-convertible
        return key_type is str and _type_is_json_convertible(value_type)

    # Literal types are json-convertible if their values are
    if origin is typing.Literal:
        return all(isinstance(val, (str, int, float, bool, type(None))) for val in args)

    return False
