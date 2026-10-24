"""Type-hint introspection helpers used by the backend (pydantic models at the data-source boundary)."""

from __future__ import annotations

import datetime
import enum
import typing
from types import UnionType
from typing import Any, Union

import pydantic


def is_json_convertible(model: type[pydantic.BaseModel]) -> bool:
    """True if instances of the model convert to valid JSON, judging by the type hints of its fields."""
    type_hints = typing.get_type_hints(model)
    return all(_type_is_json_convertible(field_type) for field_type in type_hints.values())


def _type_is_json_convertible(type_hint: Any) -> bool:
    if type_hint is type(None):
        return True
    if type_hint is Any:
        return False
    if type_hint in (str, int, float, bool, datetime.datetime):
        return True
    if isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
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
            return False
        key_type, value_type = args
        return key_type is str and _type_is_json_convertible(value_type)
    if origin is typing.Literal:
        return all(isinstance(val, (str, int, float, bool, type(None))) for val in args)
    return False


def resolve_embedding_fn(embed_fn: Any, expected_type: str) -> Any:
    """The unary resolution of `embed_fn` whose (single required) parameter has type `expected_type` (a
    `ColumnType.Type` name such as 'STRING'), or None (upstream `EmbeddingIndex._resolve_embedding_fn()`)."""
    for resolved_fn in embed_fn._resolved_fns:
        sig = resolved_fn.signature
        if len(sig.parameters) == 0 or len(sig.required_parameters) > 1:
            continue
        first = sig.parameters_by_pos[0]
        if first.col_type is None or first.col_type.type_enum.name != expected_type:
            continue
        if len(sig.parameters) == 1:
            return resolved_fn
        if not all(p.has_default() for p in sig.parameters_by_pos[1:]):
            continue
        return resolved_fn.using(**{p.name: p.default_value for p in sig.parameters_by_pos[1:]})
    return None
