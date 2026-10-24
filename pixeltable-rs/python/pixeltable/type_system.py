"""Pixeltable column types.

The public type classes mirror upstream `pixeltable.type_system`. Every `ColumnType` instance holds a Rust
`pixeltable._core.ColumnType` (`_core`), which computes nullability, matching, the supertype algebra, `repr()`,
serialization and value validation/coercion. This module keeps what is inherently Python: interpreting type hints
(`pxt.Int | None`, `list[int]`, TypedDicts, pydantic models) and the `pxt.Int`-style aliases.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import types
import typing
import uuid
import warnings
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping, Sequence, Union, is_typeddict

import numpy as np
import PIL.Image
import pydantic
from typing_extensions import TypeForm, TypeIs

import pixeltable.exceptions as excs
from pixeltable import _core


def is_type_form(x: object) -> TypeIs[TypeForm]:
    """
    Return True if `x` is a bare class such as `int`, or a subscripted form such as `list[int]`, `int | None`, or
    `Annotated[int, ...]` (the form the `pxt.Int`-style aliases take).
    """
    return isinstance(x, type) or typing.get_origin(x) is not None


class ColumnType:
    @enum.unique
    class Type(enum.Enum):
        STRING = 0
        INT = 1
        FLOAT = 2
        BOOL = 3
        TIMESTAMP = 4
        JSON = 5
        ARRAY = 6
        IMAGE = 7
        VIDEO = 8
        AUDIO = 9
        DOCUMENT = 10
        DATE = 11
        UUID = 12
        BINARY = 13

        # exprs that don't evaluate to a computable value in Pixeltable, such as an Image member function
        INVALID = 255

    scalar_json_types: ClassVar[set[Type]] = {Type.STRING, Type.INT, Type.FLOAT, Type.BOOL}
    scalar_types: ClassVar[set[Type]] = scalar_json_types | {Type.TIMESTAMP, Type.DATE, Type.UUID}
    numeric_types: ClassVar[set[Type]] = {Type.INT, Type.FLOAT}

    _core: _core.ColumnType

    def __init__(self, t: Type, nullable: bool = False):
        self._core = _core.ColumnType.make(t.value, nullable)

    @classmethod
    def _from_core(cls, core: _core.ColumnType) -> ColumnType:
        """Wrap a core type in the matching subclass."""
        obj = object.__new__(_CLASSES_BY_NAME[core.class_name])
        obj._core = core
        return obj

    def has_supertype(self) -> bool:
        return self._core.has_supertype()

    @property
    def nullable(self) -> bool:
        return self._core.nullable

    @property
    def _type(self) -> Type:
        return ColumnType.Type(self._core.type_enum)

    @property
    def type_enum(self) -> Type:
        return self._type

    def copy(self, nullable: bool) -> ColumnType:
        if nullable == self.nullable:
            return self
        return ColumnType._from_core(self._core.copy(nullable))

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def serialize_list(cls, type_list: list[ColumnType]) -> str:
        return json.dumps([t.as_dict() for t in type_list])

    def as_dict(self) -> dict:
        return self._core.as_dict()

    def _as_dict(self) -> dict:
        d = self._core.as_dict()
        del d['_classname']
        return d

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        return ColumnType.from_dict({'_classname': cls.__name__, **d})

    @classmethod
    def deserialize(cls, type_str: str) -> ColumnType:
        return cls.from_dict(json.loads(type_str))

    @classmethod
    def deserialize_list(cls, type_list_str: str) -> list[ColumnType]:
        return [cls.from_dict(d) for d in json.loads(type_list_str)]

    @classmethod
    def from_dict(cls, type_dict: dict) -> ColumnType:
        return ColumnType._from_core(_core.ColumnType.from_dict(type_dict))

    @classmethod
    def make_type(cls, t: Type) -> ColumnType:
        return ColumnType._from_core(_core.ColumnType.make(t.value, False))

    def __repr__(self) -> str:
        return repr(self._core)

    def _to_base_str(self) -> str:
        return repr(self._core.copy(False))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColumnType) and self._core.equals(other._core)

    def __hash__(self) -> int:
        return hash(self._core.hash_value())

    def is_supertype_of(self, other: ColumnType, ignore_nullable: bool = False, strict_json: bool = True) -> bool:
        return self._core.is_supertype_of(other._core, ignore_nullable, strict_json)

    def matches(self, other: ColumnType) -> bool:
        """Two types match if they're equal, aside from nullability"""
        return self._core.matches(other._core)

    def supertype(self, other: ColumnType, for_inference: bool = False) -> ColumnType | None:
        """The most specific type that is a supertype of both `self` and `other`."""
        result = self._core.supertype(other._core, for_inference)
        return None if result is None else ColumnType._from_core(result)

    @classmethod
    def common_supertype(cls, types: Iterable[ColumnType], for_inference: bool = False) -> ColumnType | None:
        result = _core.ColumnType.common_supertype([t._core for t in types], for_inference)
        return None if result is None else ColumnType._from_core(result)

    @classmethod
    def infer_literal_type(
        cls, val: Any, nullable: bool = False, infer_json_type_schema: bool = True
    ) -> ColumnType | None:
        result = _core.ColumnType.infer_literal_type(val)
        if result is None:
            return None
        col_type = ColumnType._from_core(result)
        if val is None:
            return col_type
        if col_type.is_json_type() and not infer_json_type_schema:
            return JsonType(nullable=nullable)
        return col_type.copy(nullable=nullable)

    @classmethod
    def infer_common_literal_type(cls, vals: Iterable[Any]) -> ColumnType | None:
        """The most specific type that is a supertype of all literals in `vals`, or None."""
        inferred_type: ColumnType | None = None
        for val in vals:
            val_type = cls.infer_literal_type(val)
            if val_type is None:
                return None
            inferred_type = val_type if inferred_type is None else inferred_type.supertype(val_type, for_inference=True)
            if inferred_type is None:
                return None
            if not inferred_type.has_supertype():
                return inferred_type
        return inferred_type

    @classmethod
    def from_python_type(
        cls, t: TypeForm, allow_builtin_types: bool = True, infer_pydantic_json: bool = False
    ) -> ColumnType | None:
        """
        Convert a Python type into a Pixeltable `ColumnType` instance.

        As in Python and Pydantic, a bare type such as `pxt.Int` is non-nullable; the nullable form is
        `pxt.Int | None`.
        """
        origin = typing.get_origin(t)
        type_args = typing.get_args(t)
        if origin in (typing.Union, types.UnionType):
            if len(type_args) == 2 and type(None) in type_args:
                underlying_py_type = type_args[0] if type_args[1] is type(None) else type_args[1]
                underlying = cls.from_python_type(
                    underlying_py_type, allow_builtin_types=allow_builtin_types, infer_pydantic_json=infer_pydantic_json
                )
                if underlying is not None:
                    return underlying.copy(nullable=True)
        elif origin is Required:
            assert len(type_args) == 1
            warnings.warn(
                '`Required[T]` is deprecated and will be removed in a future version; use `T` instead. '
                'Bare types such as `pxt.Int` are now non-nullable by default; use `pxt.Int | None` for a '
                'nullable column.',
                excs.PixeltableDeprecationWarning,
                stacklevel=2,
            )
            underlying = cls.from_python_type(type_args[0], allow_builtin_types=allow_builtin_types)
            if underlying is None:
                return None
            return underlying.copy(nullable=False)
        elif origin is typing.Required or origin is typing.NotRequired:
            assert len(type_args) == 1
            return cls.from_python_type(type_args[0], allow_builtin_types=allow_builtin_types)
        elif origin is typing.Annotated:
            origin = type_args[0]
            parameters = type_args[1]
            if isinstance(parameters, ColumnType):
                return parameters.copy(nullable=False)
        else:
            origin = origin or t
            if isinstance(origin, type):
                if issubclass(origin, _PxtType):
                    return origin.as_col_type(nullable=False)
                if is_typeddict(origin):
                    return cls.__from_typed_dict(origin)
                if issubclass(origin, pydantic.BaseModel):
                    return cls.__from_pydantic_model_type(origin)

            if allow_builtin_types:
                if origin is Literal and len(type_args) > 0:
                    return cls.infer_common_literal_type(type_args)
                if infer_pydantic_json and isinstance(t, type) and issubclass(t, enum.Enum):
                    return cls.infer_common_literal_type(member.value for member in t)
                if infer_pydantic_json and t is Path:
                    return StringType(nullable=False)
                if t is str:
                    return StringType(nullable=False)
                if t is int:
                    return IntType(nullable=False)
                if t is float:
                    return FloatType(nullable=False)
                if t is bool:
                    return BoolType(nullable=False)
                if t is datetime.datetime:
                    return TimestampType(nullable=False)
                if t is datetime.date:
                    return DateType(nullable=False)
                if t is uuid.UUID:
                    return UUIDType(nullable=False)
                if t is bytes:
                    return BinaryType(nullable=False)
                if t is PIL.Image.Image:
                    return ImageType(nullable=False)
                if origin is tuple:
                    return cls.__from_tuple_type(type_args)
                if isinstance(origin, type) and issubclass(origin, Sequence):
                    return cls.__from_list_type(type_args)
                if isinstance(origin, type) and issubclass(origin, Mapping):
                    return JsonType(nullable=False)

        return None

    @classmethod
    def __from_tuple_type(cls, type_args: tuple) -> JsonType:
        if len(type_args) == 0:
            return JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False)
        variadic_type = None
        if len(type_args) > 0 and type_args[-1] is Ellipsis:
            if len(type_args) == 1:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE, 'Invalid type schema: tuple with only `...` is not allowed'
                )
            variadic_type = type_args[-2]
            type_args = type_args[:-2]
        if Ellipsis in type_args:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE, 'Invalid type schema: `...` allowed only in last position'
            )
        return JsonType(
            JsonType.TypeSchema(
                type_spec=[cls.__from_python_type_or_exc(type_arg) for type_arg in type_args],
                variadic_type=cls.__from_python_type_or_exc(variadic_type) if variadic_type is not None else None,
            ),
            nullable=False,
        )

    @classmethod
    def __from_list_type(cls, type_args: tuple) -> JsonType:
        if len(type_args) > 1:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE,
                'Invalid type schema: `list` or `Sequence` must have at most one type argument',
            )
        if len(type_args) == 0 or type_args[0] is Any:
            return JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False)
        return JsonType(
            JsonType.TypeSchema([], variadic_type=cls.__from_python_type_or_exc(type_args[0])), nullable=False
        )

    @classmethod
    def __from_python_type_or_exc(cls, t: TypeForm | None) -> ColumnType:
        col_type = cls.from_python_type(t)
        if col_type is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE, f'Python type found in type argument is not a valid Pixeltable type: {t!r}'
            )
        return col_type

    @classmethod
    def __from_typed_dict(cls, t: type) -> JsonType:
        try:
            field_types = typing.get_type_hints(t, include_extras=True)
        except NameError as e:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE, f'Cannot resolve the field types of TypedDict `{t.__name__}`: {e}'
            ) from e
        type_spec: dict[str, ColumnType] = {}
        optional_keys = set(getattr(t, '__optional_keys__', frozenset()))
        for key, value in field_types.items():
            qualifier = typing.get_origin(value)
            if qualifier is typing.NotRequired:
                optional_keys.add(key)
            elif qualifier is typing.Required:
                optional_keys.discard(key)
            col_type = cls.from_python_type(value)
            if col_type is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE,
                    f'Field {key!r} in TypedDict `{t.__name__}` is not a valid Pixeltable type: {value!r}',
                )
            type_spec[key] = col_type
        return JsonType(
            JsonType.TypeSchema(type_spec=type_spec, optional_keys=frozenset(optional_keys)),
            nullable=False,
            pretty_print_name=t.__name__,
        )

    @classmethod
    def __from_pydantic_model_type(cls, t: type[pydantic.BaseModel]) -> JsonType:
        fields: dict[str, ColumnType] = {}
        for name, info in t.model_fields.items():
            col_type = cls.from_python_type(info.annotation)
            if col_type is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE,
                    f'Field {name!r} in Pydantic model `{t.__name__}` '
                    f'is not a valid Pixeltable type: {info.annotation!r}',
                )
            fields[name] = col_type
        optional_keys = frozenset(name for name, info in t.model_fields.items() if not info.is_required())
        return JsonType(
            JsonType.TypeSchema(type_spec=fields, optional_keys=optional_keys),
            nullable=False,
            pretty_print_name=t.__name__,
        )

    @classmethod
    def normalize_type(cls, t: ColumnType | TypeForm, allow_builtin_types: bool = True) -> ColumnType:
        """Convert any type recognizable by Pixeltable to its corresponding ColumnType."""
        if isinstance(t, ColumnType):
            return t
        col_type = cls.from_python_type(t, allow_builtin_types)
        if col_type is None:
            cls.__raise_exc_for_invalid_type(t)
        return col_type

    __TYPE_SUGGESTIONS: ClassVar[list[tuple[type, str]]] = [
        (str, 'pxt.String'),
        (bool, 'pxt.Bool'),
        (int, 'pxt.Int'),
        (float, 'pxt.Float'),
        (datetime.datetime, 'pxt.Timestamp'),
        (datetime.date, 'pxt.Date'),
        (uuid.UUID, 'pxt.UUID'),
        (PIL.Image.Image, 'pxt.Image'),
        (bytes, 'pxt.Binary'),
        (Sequence, 'pxt.Json'),
        (Mapping, 'pxt.Json'),
    ]

    @classmethod
    def __raise_exc_for_invalid_type(cls, t: TypeForm) -> None:
        for builtin_type, suggestion in cls.__TYPE_SUGGESTIONS:
            if t is builtin_type or (isinstance(t, type) and issubclass(t, builtin_type)):
                name = t.__name__ if t.__module__ == 'builtins' else f'{t.__module__}.{t.__name__}'
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE,
                    f'Standard Python type `{name}` cannot be used here; use `{suggestion}` instead',
                )
        raise excs.RequestError(excs.ErrorCode.INVALID_TYPE, f'Unknown type: {t}')

    @classmethod
    def from_json_schema(cls, schema: dict[str, Any]) -> ColumnType | None:
        py_type = cls.__json_schema_to_py_type(schema)
        return cls.from_python_type(py_type) if py_type is not None else None

    @classmethod
    def __json_schema_to_py_type(cls, schema: dict[str, Any]) -> TypeForm | None:
        if 'type' in schema:
            if schema['type'] == 'null':
                return type(None)
            if schema['type'] == 'string':
                return str
            if schema['type'] == 'integer':
                return int
            if schema['type'] == 'number':
                return float
            if schema['type'] == 'boolean':
                return bool
            if schema['type'] in ('array', 'object'):
                return list
        elif 'anyOf' in schema:
            subscripts: Any = tuple(cls.__json_schema_to_py_type(subschema) for subschema in schema['anyOf'])
            if all(subscript is not None for subscript in subscripts):
                return Union[subscripts]
        return None

    def validate_literal(self, val: Any) -> None:
        """Raise TypeError if val is not a valid literal for this type"""
        self._core.validate_literal(val)

    def validate_media(self, val: Any) -> None:
        """Raise if val is not a valid media file for this type (not implemented yet in pixeltable-rs)."""
        if self.is_media_type():
            raise NotImplementedError('media validation is not implemented yet in pixeltable-rs')

    def create_literal(self, val: Any) -> Any:
        """Create a literal of this type from val or raise TypeError if not possible"""
        return self._core.create_literal(val)

    def print_value(self, val: Any) -> str:
        if self.is_string_type() or self.is_uuid_type():
            return f"'{val}'"
        return str(val)

    def is_scalar_type(self) -> bool:
        return self._type in self.scalar_types

    def is_scalar_json_type(self) -> bool:
        return self._type in self.scalar_json_types

    def is_numeric_type(self) -> bool:
        return self._type in self.numeric_types

    def is_invalid_type(self) -> bool:
        return self._type == self.Type.INVALID

    def is_string_type(self) -> bool:
        return self._type == self.Type.STRING

    def is_int_type(self) -> bool:
        return self._type == self.Type.INT

    def is_float_type(self) -> bool:
        return self._type == self.Type.FLOAT

    def is_bool_type(self) -> bool:
        return self._type == self.Type.BOOL

    def is_timestamp_type(self) -> bool:
        return self._type == self.Type.TIMESTAMP

    def is_date_type(self) -> bool:
        return self._type == self.Type.DATE

    def is_uuid_type(self) -> bool:
        return self._type == self.Type.UUID

    def is_json_type(self) -> bool:
        return self._type == self.Type.JSON

    def is_array_type(self) -> bool:
        return self._type == self.Type.ARRAY

    def is_binary_type(self) -> bool:
        return self._type == self.Type.BINARY

    def is_image_type(self) -> bool:
        return self._type == self.Type.IMAGE

    def is_video_type(self) -> bool:
        return self._type == self.Type.VIDEO

    def is_audio_type(self) -> bool:
        return self._type == self.Type.AUDIO

    def is_document_type(self) -> bool:
        return self._type == self.Type.DOCUMENT

    def is_media_type(self) -> bool:
        return self.is_image_type() or self.is_video_type() or self.is_audio_type() or self.is_document_type()

    def to_json_schema(self) -> dict[str, Any]:
        if self.nullable:
            return {'anyOf': [self._to_json_schema(), {'type': 'null'}]}
        return self._to_json_schema()

    def _to_json_schema(self) -> dict[str, Any]:
        raise excs.RequestError(excs.ErrorCode.INVALID_TYPE, f'Pixeltable type {self} is not a valid JSON type')

    def to_pydantic_type(self) -> Any:
        """The nominal Python type used to represent values of this column type in a Pydantic field type."""
        py_type: Any = {
            self.Type.STRING: str,
            self.Type.INT: int,
            self.Type.FLOAT: float,
            self.Type.BOOL: bool,
            self.Type.TIMESTAMP: datetime.datetime,
            self.Type.DATE: datetime.date,
            self.Type.UUID: uuid.UUID,
            self.Type.BINARY: bytes,
            self.Type.JSON: Any,
            self.Type.ARRAY: list,
            self.Type.IMAGE: str,
            self.Type.VIDEO: str,
            self.Type.AUDIO: str,
            self.Type.DOCUMENT: str,
        }[self._type]
        if self.nullable:
            return typing.Optional[py_type]
        return py_type

    @classmethod
    def from_np_dtype(cls, dtype: np.dtype, nullable: bool) -> ColumnType | None:
        """Return the pixeltable type corresponding to a simple numpy dtype"""
        if np.issubdtype(dtype, np.integer):
            return IntType(nullable=nullable)
        if np.issubdtype(dtype, np.floating):
            return FloatType(nullable=nullable)
        if dtype == np.bool_:
            return BoolType(nullable=nullable)
        if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.character):
            return StringType(nullable=nullable)
        if np.issubdtype(dtype, np.datetime64):
            unit, _ = np.datetime_data(dtype)
            if unit in ('D', 'M', 'Y'):
                return DateType(nullable=nullable)
            return TimestampType(nullable=nullable)
        return None


class InvalidType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INVALID, nullable=nullable)


class StringType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.STRING, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'string'}


class IntType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INT, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'integer'}


class FloatType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.FLOAT, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'number'}


class BoolType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.BOOL, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'boolean'}


class TimestampType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.TIMESTAMP, nullable=nullable)


class DateType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.DATE, nullable=nullable)


class UUIDType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.UUID, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'string', 'format': 'uuid'}


class BinaryType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.BINARY, nullable=nullable)


class JsonType(ColumnType):
    # pretty_print_name preserves the name of the `TypedDict` used to define this `JsonType` (if any), for docs.
    pretty_print_name: str | None

    def __init__(
        self, type_schema: JsonType.TypeSchema | None = None, nullable: bool = False, pretty_print_name: str | None = None
    ):
        if type_schema is None:
            self._core = _core.ColumnType.make(self.Type.JSON.value, nullable)
        elif isinstance(type_schema.type_spec, list):
            self._core = _core.ColumnType.json_list(
                [t._core for t in type_schema.type_spec],
                type_schema.variadic_type._core if type_schema.variadic_type is not None else None,
                nullable,
            )
        else:
            self._core = _core.ColumnType.json_dict(
                [(k, t._core) for k, t in type_schema.type_spec.items()], sorted(type_schema.optional_keys), nullable
            )
        self.pretty_print_name = pretty_print_name

    @property
    def type_schema(self) -> JsonType.TypeSchema | None:
        s = self._core.json_schema()
        if s is None:
            return None
        kind, spec, variadic, optional_keys = s
        if kind == 'list':
            return JsonType.TypeSchema(
                type_spec=[ColumnType._from_core(t) for t in spec],
                variadic_type=ColumnType._from_core(variadic) if variadic is not None else None,
            )
        return JsonType.TypeSchema(
            type_spec={k: ColumnType._from_core(t) for k, t in spec}, optional_keys=frozenset(optional_keys)
        )

    @classmethod
    def from_json_type_arg(cls, json_type_arg: Any) -> JsonType:
        """The JsonType for a type argument `T` of `Json[T]` (see upstream for the accepted forms)."""
        if json_type_arg is None:
            return JsonType()
        result = cls.from_json_type_arg_r(json_type_arg)
        if not isinstance(result, JsonType):
            raise ValueError(
                f'Invalid type schema: type argument does not represent a valid JSON type: {json_type_arg}'
            )
        return result

    @classmethod
    def from_json_type_arg_r(cls, json_type_arg: Any) -> ColumnType:
        if isinstance(json_type_arg, list):
            if len(json_type_arg) != 1:
                raise ValueError(
                    f'Invalid type schema: expected a single-item list; got a list of length {len(json_type_arg)}'
                )
            return JsonType(JsonType.TypeSchema(type_spec=[], variadic_type=cls.from_json_type_arg_r(json_type_arg[0])))
        if isinstance(json_type_arg, tuple):
            variadic_type = None
            if len(json_type_arg) > 0 and json_type_arg[-1] is Ellipsis:
                if len(json_type_arg) == 1:
                    raise ValueError('Invalid type schema: tuple with only `...` is not allowed')
                variadic_type = json_type_arg[-2]
                fixed_types = json_type_arg[:-2]
            else:
                fixed_types = json_type_arg
            if Ellipsis in fixed_types or variadic_type is Ellipsis:
                raise ValueError('Invalid type schema: `...` allowed only in last position')
            return JsonType(
                JsonType.TypeSchema(
                    type_spec=[cls.from_json_type_arg_r(item) for item in fixed_types],
                    variadic_type=cls.from_json_type_arg_r(variadic_type) if variadic_type is not None else None,
                )
            )
        if isinstance(json_type_arg, dict):
            type_spec: dict[str, ColumnType] = {}
            for key, value in json_type_arg.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f'Invalid type schema: expected keys of type `str`; got type `{type(key).__name__}`'
                    )
                type_spec[key] = cls.from_json_type_arg_r(value)
            return JsonType(JsonType.TypeSchema(type_spec))
        result = ColumnType.from_python_type(json_type_arg)
        if result is None:
            raise ValueError(f'Invalid type schema: type argument contains an invalid Pixeltable type: {json_type_arg}')
        return result

    def copy(self, nullable: bool) -> ColumnType:
        if nullable == self.nullable:
            return self
        return ColumnType._from_core(self._core.copy(nullable))

    def array_element_type(self) -> ColumnType | None:
        """The element type of a json array type."""
        schema = self.type_schema
        if schema is None or not isinstance(schema.type_spec, list):
            return None
        if schema.variadic_type is not None:
            return schema.variadic_type
        if len(schema.type_spec) > 0:
            return ColumnType.common_supertype(schema.type_spec)
        return None

    def _to_json_schema(self) -> dict[str, Any]:
        schema = self.type_schema
        return {} if schema is None else schema.to_json_schema()

    @classmethod
    def from_literal(cls, val: Any, nullable: bool = False) -> JsonType | None:
        t = ColumnType.infer_literal_type(list(val) if isinstance(val, tuple) else val, nullable=nullable)
        return t if isinstance(t, JsonType) else None

    @dataclasses.dataclass(frozen=True)
    class TypeSchema:
        """
        The type schema of a JsonType: a fixed-length tuple (`type_spec` a list, possibly with a `variadic_type`
        tail), or a typed dictionary (`type_spec` a dict, with `optional_keys`).
        """

        type_spec: list[ColumnType] | dict[str, ColumnType]
        variadic_type: ColumnType | None = None
        optional_keys: frozenset[str] = frozenset()

        def to_json_schema(self) -> dict[str, Any]:
            json_schema: dict[str, Any]
            if isinstance(self.type_spec, list):
                json_schema = {'type': 'array'}
                if len(self.type_spec) > 0:
                    json_schema['prefixItems'] = [t.to_json_schema() for t in self.type_spec]
                json_schema['items'] = False if self.variadic_type is None else self.variadic_type.to_json_schema()
            else:
                json_schema = {
                    'type': 'object',
                    'properties': {k: t.to_json_schema() for k, t in self.type_spec.items()},
                    'additionalProperties': False,
                }
                required = [k for k in self.type_spec if k not in self.optional_keys]
                if len(required) > 0:
                    json_schema['required'] = required
            return json_schema


ARRAY_SUPPORTED_NUMPY_DTYPES = [
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.str_,
]

_PXT_DTYPE_NAMES = {
    ColumnType.Type.INT: 'int64',
    ColumnType.Type.FLOAT: 'float32',
    ColumnType.Type.BOOL: 'bool',
    ColumnType.Type.STRING: 'str',
}


class ArrayType(ColumnType):
    def __init__(
        self,
        shape: tuple[int | None, ...] | None = None,
        dtype: ColumnType | np.dtype | None = None,
        nullable: bool = False,
    ):
        assert shape is None or dtype is not None, (shape, dtype)  # cannot specify a shape without a dtype
        dtype_name: str | None
        if dtype is None:
            dtype_name = None
        elif isinstance(dtype, np.dtype):
            if dtype.type == np.str_:
                dtype_name = 'str'
            elif dtype not in ARRAY_SUPPORTED_NUMPY_DTYPES:
                raise ValueError(f'Unsupported dtype: {dtype}')
            else:
                dtype_name = dtype.name
        elif isinstance(dtype, ColumnType):
            dtype_name = _PXT_DTYPE_NAMES.get(dtype._type)
            if dtype_name is None:
                raise ValueError(f'Unsupported dtype: {dtype}')
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        self._core = _core.ColumnType.array(list(shape) if shape is not None else None, dtype_name, nullable)

    @property
    def shape(self) -> tuple[int | None, ...] | None:
        return self._core.shape

    @property
    def dtype(self) -> np.dtype | None:
        name = self._core.dtype_name
        if name is None:
            return None
        return np.dtype(np.str_) if name == 'str' else np.dtype(name)

    def subscript_type(self, index: tuple[int | slice, ...]) -> ColumnType:
        """The type of `self[index]` under numpy basic indexing."""
        idx = [i if isinstance(i, int) else (i.start, i.stop, i.step) for i in index]
        return ColumnType._from_core(self._core.subscript_type(idx))

    @classmethod
    def from_literal(cls, val: np.ndarray, nullable: bool = False) -> ArrayType | None:
        if val.dtype.type not in ARRAY_SUPPORTED_NUMPY_DTYPES:
            return None
        return cls(val.shape, dtype=val.dtype, nullable=nullable)

    def _to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {'type': 'array'}
        if self.dtype == np.str_:
            schema.update({'items': {'type': 'str'}})
        elif self.dtype is not None:
            schema.update({'items': {'type': str(self.dtype)}})
        return schema


class ImageType(ColumnType):
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        size: tuple[int, int] | None = None,
        mode: str | None = None,
        nullable: bool = False,
    ):
        assert not (width is not None and size is not None)
        assert not (height is not None and size is not None)
        if size is not None:
            width, height = size
        self._core = _core.ColumnType.image(width, height, mode, nullable)

    @property
    def width(self) -> int | None:
        return self._core.width

    @property
    def height(self) -> int | None:
        return self._core.height

    @property
    def mode(self) -> str | None:
        return self._core.mode

    @property
    def size(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return (self.width, self.height)


class VideoType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.VIDEO, nullable=nullable)


class AudioType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.AUDIO, nullable=nullable)


class DocumentType(ColumnType):
    @enum.unique
    class DocumentFormat(enum.Enum):
        HTML = 0
        MD = 1
        PDF = 2
        XML = 3
        TXT = 4
        PPTX = 5
        DOCX = 6
        XLSX = 7

        @classmethod
        def from_extension(cls, ext: str) -> DocumentType.DocumentFormat | None:
            if ext in ('.htm', '.html'):
                return cls.HTML
            if ext == '.md':
                return cls.MD
            if ext == '.pdf':
                return cls.PDF
            if ext == '.xml':
                return cls.XML
            if ext == '.txt':
                return cls.TXT
            if ext in ('.pptx', '.ppt'):
                return cls.PPTX
            if ext in ('.docx', '.doc'):
                return cls.DOCX
            if ext in ('.xlsx', '.xls'):
                return cls.XLSX
            return None

    def __init__(self, nullable: bool = False, doc_formats: str | None = None):
        self._core = _core.ColumnType.document(doc_formats, nullable)

    @property
    def doc_formats(self) -> str | None:
        return self._core.doc_formats


_CLASSES_BY_NAME: dict[str, type[ColumnType]] = {
    'InvalidType': InvalidType,
    'StringType': StringType,
    'IntType': IntType,
    'FloatType': FloatType,
    'BoolType': BoolType,
    'TimestampType': TimestampType,
    'DateType': DateType,
    'UUIDType': UUIDType,
    'BinaryType': BinaryType,
    'JsonType': JsonType,
    'ArrayType': ArrayType,
    'ImageType': ImageType,
    'VideoType': VideoType,
    'AudioType': AudioType,
    'DocumentType': DocumentType,
}


T = typing.TypeVar('T')


class _PxtType:
    """
    Base class for the Pixeltable type-hint family (`pxt.Json`, `pxt.Array`, `pxt.Image`, ...). These classes are
    not instantiable; they resolve to `ColumnType`s via `ColumnType.from_python_type()`.
    """

    def __init__(self) -> None:
        raise TypeError(f'Type `{type(self)}` cannot be instantiated.')

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        raise NotImplementedError()


class Required(_PxtType, typing.Generic[T]):
    """Deprecated marker: bare types such as `pxt.Int` are non-nullable, so `Required[T]` is equivalent to `T`."""

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_TYPE,
            'Bare `Required` is not a valid type; use `Required[T]` for some valid type `T`.',
        )


String = typing.Annotated[str, StringType(nullable=False)]
Int = typing.Annotated[int, IntType(nullable=False)]
Float = typing.Annotated[float, FloatType(nullable=False)]
Bool = typing.Annotated[bool, BoolType(nullable=False)]
Timestamp = typing.Annotated[datetime.datetime, TimestampType(nullable=False)]
Date = typing.Annotated[datetime.date, DateType(nullable=False)]
UUID = typing.Annotated[uuid.UUID, UUIDType(nullable=False)]
Binary = typing.Annotated[bytes, BinaryType(nullable=False)]


class Json(_PxtType):
    def __class_getitem__(cls, item: Any) -> TypeForm:
        """`item` must be a valid Pixeltable JSON type specifier (see `JsonType.from_json_type_arg()`)."""
        return typing.Annotated[Any, JsonType.from_json_type_arg(item)]

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return JsonType(nullable=nullable)


class Array(np.ndarray, _PxtType):
    def __class_getitem__(cls, item: Any) -> TypeForm:  # type: ignore[override]
        """
        `item` must be a tuple with at most two elements (in any order): an optional shape tuple of `int | None`s,
        and a dtype (`ColumnType | np.dtype`). Examples: `Array[(3, None, 2), pxt.Float]`, `Array[np.bool]`.
        """
        params = item if isinstance(item, tuple) else (item,)
        shape: tuple | None = None
        dtype: ColumnType | np.dtype | None = None
        if not any(typing.get_origin(param) is typing.Annotated or isinstance(param, type) for param in params):
            raise TypeError('Array type parameter must include a dtype.')
        for param in params:
            if isinstance(param, tuple):
                if not all(n is None or (isinstance(n, int) and n >= 1) for n in param):
                    raise TypeError(f'Invalid Array type parameter: {param}')
                if shape is not None:
                    raise TypeError(f'Duplicate Array type parameter: {param}')
                shape = param
            elif typing.get_origin(param) is typing.Annotated or isinstance(param, type):
                if dtype is not None:
                    raise TypeError(f'Duplicate Array type parameter: {param}')
                if isinstance(param, type) and param in ARRAY_SUPPORTED_NUMPY_DTYPES:
                    dtype = np.dtype(param)
                else:
                    dtype = ColumnType.normalize_type(param, allow_builtin_types=False)
            else:
                raise TypeError(f'Invalid Array type parameter: {param}')
        return typing.Annotated[np.ndarray, ArrayType(shape=shape, dtype=dtype, nullable=False)]

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return ArrayType(nullable=nullable)


class Image(PIL.Image.Image, _PxtType):
    def __class_getitem__(cls, item: Any) -> TypeForm:
        """`item`: a `(width, height)` tuple and/or a mode string, e.g. `Image[(300, 300), 'RGB']`."""
        if isinstance(item, tuple) and all(n is None or isinstance(n, int) for n in item):
            params = (item,)
        elif isinstance(item, tuple):
            params = item
        else:
            params = (item,)
        size: tuple | None = None
        mode: str | None = None
        for param in params:
            if isinstance(param, tuple):
                if (
                    len(param) != 2
                    or not isinstance(param[0], (int, type(None)))
                    or not isinstance(param[1], (int, type(None)))
                ):
                    raise TypeError(f'Invalid Image type parameter: {param}')
                if size is not None:
                    raise TypeError(f'Duplicate Image type parameter: {param}')
                size = param
            elif isinstance(param, str):
                if param not in PIL.Image.MODES:
                    raise TypeError(f'Invalid Image type parameter: {param!r}')
                if mode is not None:
                    raise TypeError(f'Duplicate Image type parameter: {param!r}')
                mode = param
            else:
                raise TypeError(f'Invalid Image type parameter: {param}')
        return typing.Annotated[PIL.Image.Image, ImageType(size=size, mode=mode, nullable=False)]

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return ImageType(nullable=nullable)


class Video(str, _PxtType):
    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return VideoType(nullable=nullable)


class Audio(str, _PxtType):
    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return AudioType(nullable=nullable)


class Document(str, _PxtType):
    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return DocumentType(nullable=nullable)


ALL_PIXELTABLE_TYPES = (String, Bool, Int, Float, Timestamp, Json, Array, Image, Video, Audio, Document, Date, UUID, Binary)
