from __future__ import annotations

import abc
import dataclasses
import datetime
import enum
import io
import itertools
import json
import types
import typing
import urllib.request
import uuid
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping, Sequence, Union

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip

import av
import numpy as np
import PIL.Image
import pydantic
import sqlalchemy as sql
from typing_extensions import _AnnotatedAlias

import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.utils import parse_local_file_path


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

        @classmethod
        def supertype(
            cls,
            type1: 'ColumnType.Type' | None,
            type2: 'ColumnType.Type' | None,
            # we need to pass this in because we can't easily append it as a class member
            common_supertypes: dict[tuple['ColumnType.Type', 'ColumnType.Type'], 'ColumnType.Type'],
        ) -> 'ColumnType.Type' | None:
            if type1 == type2:
                return type1
            t = common_supertypes.get((type1, type2))
            if t is not None:
                return t
            t = common_supertypes.get((type2, type1))
            if t is not None:
                return t
            return None

    scalar_json_types: ClassVar[set[Type]] = {Type.STRING, Type.INT, Type.FLOAT, Type.BOOL}
    scalar_types: ClassVar[set[Type]] = scalar_json_types | {Type.TIMESTAMP, Type.DATE, Type.UUID}
    numeric_types: ClassVar[set[Type]] = {Type.INT, Type.FLOAT}
    common_supertypes: ClassVar[dict[tuple[Type, Type], Type]] = {
        (Type.BOOL, Type.INT): Type.INT,
        (Type.BOOL, Type.FLOAT): Type.FLOAT,
        (Type.INT, Type.FLOAT): Type.FLOAT,
    }

    def __init__(self, t: Type, nullable: bool = False):
        self._type = t
        self._nullable = nullable

    def has_supertype(self) -> bool:
        return True

    @property
    def nullable(self) -> bool:
        return self._nullable

    @property
    def type_enum(self) -> Type:
        return self._type

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

    def copy(self, nullable: bool) -> ColumnType:
        # Default implementation calls unary initializer
        if nullable == self.nullable:
            return self
        else:
            return self.__class__(nullable=nullable)  # type: ignore[call-arg]

    @classmethod
    def serialize_list(cls, type_list: list[ColumnType]) -> str:
        return json.dumps([t.as_dict() for t in type_list])

    def as_dict(self) -> dict:
        return {'_classname': self.__class__.__name__, **self._as_dict()}

    def _as_dict(self) -> dict:
        return {'nullable': self.nullable}

    @classmethod
    def deserialize(cls, type_str: str) -> ColumnType:
        type_dict = json.loads(type_str)
        return cls.from_dict(type_dict)

    @classmethod
    def deserialize_list(cls, type_list_str: str) -> list[ColumnType]:
        type_dict_list = json.loads(type_list_str)
        return [cls.from_dict(type_dict) for type_dict in type_dict_list]

    @classmethod
    def from_dict(cls, type_dict: dict) -> ColumnType:
        assert '_classname' in type_dict
        type_class = globals()[type_dict['_classname']]
        return type_class._from_dict(type_dict)

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        """
        Default implementation: simply invoke c'tor
        """
        assert 'nullable' in d
        return cls(nullable=d['nullable'])  # type: ignore[call-arg]

    @classmethod
    def make_type(cls, t: Type) -> ColumnType:
        match t:
            case cls.Type.STRING:
                return StringType()
            case cls.Type.INT:
                return IntType()
            case cls.Type.FLOAT:
                return FloatType()
            case cls.Type.BOOL:
                return BoolType()
            case cls.Type.TIMESTAMP:
                return TimestampType()
            case cls.Type.JSON:
                return JsonType()
            case cls.Type.ARRAY:
                return ArrayType()
            case cls.Type.IMAGE:
                return ImageType()
            case cls.Type.VIDEO:
                return VideoType()
            case cls.Type.AUDIO:
                return AudioType()
            case cls.Type.DOCUMENT:
                return DocumentType()
            case cls.Type.DATE:
                return DateType()
            case cls.Type.UUID:
                return UUIDType()
            case cls.Type.BINARY:
                return BinaryType()
            case _:
                raise AssertionError(t)

    def __repr__(self) -> str:
        return self._to_str(as_schema=False)

    def _to_str(self, as_schema: bool) -> str:
        base_str = self._to_base_str()
        if as_schema:
            return base_str if self.nullable else f'Required[{base_str}]'
        else:
            return f'{base_str} | None' if self.nullable else base_str

    def _to_base_str(self) -> str:
        """
        String representation of this type, disregarding nullability. Default implementation is to camel-case
        the type name; subclasses may override.
        """
        return self._type.name[0] + self._type.name[1:].lower()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColumnType) and self.matches(other) and self.nullable == other.nullable

    def __hash__(self) -> int:
        return hash((self._type, self.nullable))

    def is_supertype_of(self, other: ColumnType, ignore_nullable: bool = False, strict_json: bool = True) -> bool:
        self_ = self
        if not strict_json and self.is_json_type():
            # strict_json is turned off; erase the type schema
            self_ = JsonType(nullable=self_.nullable)
        if ignore_nullable:
            supertype = self_.supertype(other)
            if supertype is None:
                return False
            return supertype.matches(self_)
        else:
            return self_.supertype(other) == self_

    def matches(self, other: ColumnType) -> bool:
        """Two types match if they're equal, aside from nullability"""
        # Default: just compare base types (this works for all types whose only parameter is nullable)
        return self._type == other._type

    def supertype(self, other: ColumnType, for_inference: bool = False) -> ColumnType | None:
        """
        Returns the most specific type that is a supertype of both `self` and `other`.

        If `for_inference=True`, then we disallow certain type relationships that are technically correct, but may
        be confusing for schema inference during data imports.
        """
        if self == other:
            return self
        if self.matches(other):
            return self.copy(nullable=(self.nullable or other.nullable))

        if self.is_invalid_type():
            return other.copy(nullable=(self.nullable or other.nullable))
        if other.is_invalid_type():
            return self.copy(nullable=(self.nullable or other.nullable))

        if self.is_scalar_type() and other.is_scalar_type():
            t = self.Type.supertype(self._type, other._type, self.common_supertypes)
            if t is not None:
                return self.make_type(t).copy(nullable=(self.nullable or other.nullable))

        # If we see a mix of JSON and/or JSON-compatible scalar types, resolve to JSON.
        # (For JSON+JSON, we return None to allow JsonType to handle merging the type schemas.)
        if not for_inference and (
            (self.is_json_type() and other.is_scalar_json_type())
            or (self.is_scalar_json_type() and other.is_json_type())
            or (self.is_scalar_json_type() and other.is_scalar_json_type())
        ):
            return JsonType(nullable=(self.nullable or other.nullable))

        return None

    @classmethod
    def infer_literal_type(cls, val: Any, nullable: bool = False) -> ColumnType | None:
        if val is None:
            return InvalidType(nullable=True)
        if isinstance(val, str):
            return StringType(nullable=nullable)
        if isinstance(val, bool):
            # We have to check bool before int, because isinstance(b, int) is True if b is a Python bool
            return BoolType(nullable=nullable)
        if isinstance(val, int):
            return IntType(nullable=nullable)
        if isinstance(val, float):
            return FloatType(nullable=nullable)
        # When checking types of dates / timestamps, be aware that a datetime is also a date,
        # but a date is not a datetime. So check for datetime first.
        if isinstance(val, datetime.datetime):
            return TimestampType(nullable=nullable)
        if isinstance(val, datetime.date):
            return DateType(nullable=nullable)
        if isinstance(val, uuid.UUID):
            return UUIDType(nullable=nullable)
        if isinstance(val, bytes):
            return BinaryType(nullable=nullable)
        if isinstance(val, PIL.Image.Image):
            return ImageType(width=val.width, height=val.height, mode=val.mode, nullable=nullable)
        if isinstance(val, np.ndarray):
            return ArrayType.from_literal(val, nullable=nullable)
        if isinstance(val, (list, tuple, dict, pydantic.BaseModel)):
            return JsonType.from_literal(val, nullable=nullable)
        return None

    @classmethod
    def infer_common_literal_type(cls, vals: Iterable[Any]) -> ColumnType | None:
        """
        Returns the most specific type that is a supertype of all literals in `vals`. If no such type
        exists, returns None.

        Args:
            vals: A collection of literals.
        """
        inferred_type: ColumnType | None = None
        for val in vals:
            val_type = cls.infer_literal_type(val)
            if inferred_type is None:
                inferred_type = val_type
            else:
                inferred_type = inferred_type.supertype(val_type, for_inference=True)
            if inferred_type is None:
                return None
            if not inferred_type.has_supertype():
                return inferred_type
        return inferred_type

    @classmethod
    def from_python_type(
        cls,
        t: type | _GenericAlias,
        nullable_default: bool = False,
        allow_builtin_types: bool = True,
        infer_pydantic_json: bool = False,
    ) -> ColumnType | None:
        """
        Convert a Python type into a Pixeltable `ColumnType` instance.

        Args:
            t: The Python type.
            nullable_default: If True, then the returned `ColumnType` will be nullable unless it is marked as
                `Required`.
            allow_builtin_types: If True, then built-in types such as `str`, `int`, `float`, etc., will be
                allowed (as in UDF definitions). If False, then only Pixeltable types such as `pxt.String`,
                `pxt.Int`, etc., will be allowed (as in schema definitions). `Optional` and `Required`
                designations will be allowed regardless.
            infer_pydantic_json: If True, accepts an extended set of built-ins (eg, Enum, Path) and returns the type to
                which pydantic.BaseModel.model_dump(mode='json') serializes it.
        """
        origin = typing.get_origin(t)
        type_args = typing.get_args(t)
        if origin in (typing.Union, types.UnionType):
            # Check if `t` has the form T | None.
            if len(type_args) == 2 and type(None) in type_args:
                # `t` is a type of the form T | None (equivalently, T | None or None | T).
                # We treat it as the underlying type but with nullable=True.
                underlying_py_type = type_args[0] if type_args[1] is type(None) else type_args[1]
                underlying = cls.from_python_type(
                    underlying_py_type, allow_builtin_types=allow_builtin_types, infer_pydantic_json=infer_pydantic_json
                )
                if underlying is not None:
                    return underlying.copy(nullable=True)
        elif origin is Required:
            assert len(type_args) == 1
            return cls.from_python_type(
                type_args[0], nullable_default=False, allow_builtin_types=allow_builtin_types
            ).copy(nullable=False)
        elif origin is typing.Annotated:
            origin = type_args[0]
            parameters = type_args[1]
            if isinstance(parameters, ColumnType):
                return parameters.copy(nullable=nullable_default)
        else:
            # It's something other than T | None, Required[T], or an explicitly annotated type.
            # for non-generic types, get_origin returns None, so we use the type itself as the origin
            origin = origin or t
            if isinstance(origin, type):
                if issubclass(origin, _PxtType):
                    # We always allow Pixeltable types
                    return origin.as_col_type(nullable=nullable_default)

                if getattr(origin, '__orig_bases__', None) == (typing.TypedDict,):
                    # We always allow TypedDicts
                    assert isinstance(origin, type)
                    return cls.__from_typed_dict(nullable_default, origin)

                if issubclass(origin, pydantic.BaseModel):
                    # We always allow Pydantic models
                    return cls.__from_pydantic_model_type(nullable_default, origin)

            # Everything else is allowed only if allow_builtin_types=True
            if allow_builtin_types:
                if origin is Literal and len(type_args) > 0:
                    literal_type = cls.infer_common_literal_type(type_args)
                    if literal_type is None:
                        return None
                    return literal_type.copy(nullable=(literal_type.nullable or nullable_default))
                if infer_pydantic_json and isinstance(t, type) and issubclass(t, enum.Enum):
                    literal_type = cls.infer_common_literal_type(member.value for member in t)
                    if literal_type is None:
                        return None
                    return literal_type.copy(nullable=(literal_type.nullable or nullable_default))
                if infer_pydantic_json and t is Path:
                    return StringType(nullable=nullable_default)
                if t is str:
                    return StringType(nullable=nullable_default)
                if t is int:
                    return IntType(nullable=nullable_default)
                if t is float:
                    return FloatType(nullable=nullable_default)
                if t is bool:
                    return BoolType(nullable=nullable_default)
                if t is datetime.datetime:
                    return TimestampType(nullable=nullable_default)
                if t is datetime.date:
                    return DateType(nullable=nullable_default)
                if t is uuid.UUID:
                    return UUIDType(nullable=nullable_default)
                if t is bytes:
                    return BinaryType(nullable=nullable_default)
                if t is PIL.Image.Image:
                    return ImageType(nullable=nullable_default)
                if origin is tuple:
                    return cls.__from_tuple_type(nullable_default, type_args)
                if isinstance(origin, type) and issubclass(origin, Sequence):
                    return cls.__from_list_type(nullable_default, type_args)
                if isinstance(origin, type) and issubclass(origin, Mapping):
                    # dict or Mapping that's not a TypedDict subclass; treat it is untyped JSON.
                    return JsonType(nullable=nullable_default)

        return None

    @classmethod
    def __from_tuple_type(cls, nullable_default: bool, subscripts: tuple) -> JsonType:
        # It's a type hint of the form `tuple[T1, T2, T3]` or `tuple[T, ...]`.
        # Technically this logic will also work for semi-variadic tuples (`tuple[T1, T2, ...]`) but Python
        # doesn't allow that syntax.
        if len(subscripts) == 0:
            return JsonType(nullable=nullable_default)  # treat unparameterized tuple as untyped JSON
        variadic_type = None
        if len(subscripts) > 0 and subscripts[-1] is Ellipsis:
            if len(subscripts) == 1:
                raise excs.Error('Invalid type schema: tuple with only `...` is not allowed')
            variadic_type = subscripts[-2]
            subscripts = subscripts[:-2]

        if Ellipsis in subscripts:
            raise excs.Error('Invalid type schema: `...` allowed only in last position')

        return JsonType(
            JsonType.TypeSchema(
                type_spec=[cls.from_python_type(subscript) for subscript in subscripts],
                variadic_type=cls.from_python_type(variadic_type) if variadic_type is not None else None,
            ),
            nullable=nullable_default,
        )

    @classmethod
    def __from_list_type(cls, nullable_default: bool, subscripts: tuple) -> JsonType:
        if len(subscripts) == 0:
            return JsonType(nullable=nullable_default)  # treat unparameterized list as untyped JSON
        if len(subscripts) > 1:
            raise excs.Error('Invalid type schema: `list` or `Sequence` must have at most one type argument')
        return JsonType(
            JsonType.TypeSchema(type_spec=[], variadic_type=cls.from_python_type(subscripts[0])),
            nullable=nullable_default,
        )

    @classmethod
    def __from_typed_dict(cls, nullable_default: bool, t: type) -> JsonType:
        # It's a subclass of `TypedDict`.
        return JsonType(
            JsonType.TypeSchema(
                type_spec={key: cls.from_python_type(value) for key, value in t.__annotations__.items()},
                optional_keys=list(getattr(t, '__optional_keys__', [])),
            ),
            nullable=nullable_default,
        )

    @classmethod
    def __from_pydantic_model_type(cls, nullable_default: bool, t: type[pydantic.BaseModel]) -> JsonType:
        fields: dict[str, ColumnType] = {}
        optional_keys: list[str] = []
        for name, info in t.model_fields.items():
            fields[name] = cls.from_python_type(info.annotation)
            if not info.is_required():
                optional_keys.append(name)
        return JsonType(JsonType.TypeSchema(type_spec=fields, optional_keys=optional_keys), nullable=nullable_default)

    @classmethod
    def normalize_type(
        cls, t: ColumnType | type | _AnnotatedAlias, nullable_default: bool = False, allow_builtin_types: bool = True
    ) -> ColumnType:
        """
        Convert any type recognizable by Pixeltable to its corresponding ColumnType.
        """
        if isinstance(t, ColumnType):
            return t
        col_type = cls.from_python_type(t, nullable_default, allow_builtin_types)
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
    def __raise_exc_for_invalid_type(cls, t: type | _AnnotatedAlias) -> None:
        for builtin_type, suggestion in cls.__TYPE_SUGGESTIONS:
            if t is builtin_type or (isinstance(t, type) and issubclass(t, builtin_type)):
                name = t.__name__ if t.__module__ == 'builtins' else f'{t.__module__}.{t.__name__}'
                raise excs.Error(f'Standard Python type `{name}` cannot be used here; use `{suggestion}` instead')
        raise excs.Error(f'Unknown type: {t}')

    @classmethod
    def from_json_schema(cls, schema: dict[str, Any]) -> ColumnType | None:
        # We first express the JSON schema as a Python type, and then convert it to a Pixeltable type.
        # TODO: Is there a meaningful fallback if one of these operations fails? (Maybe another use case for a pxt Any
        #     type?)
        py_type = cls.__json_schema_to_py_type(schema)
        return cls.from_python_type(py_type) if py_type is not None else None

    @classmethod
    def __json_schema_to_py_type(cls, schema: dict[str, Any]) -> type | _GenericAlias | None:
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
            subscripts = tuple(cls.__json_schema_to_py_type(subschema) for subschema in schema['anyOf'])
            if all(subscript is not None for subscript in subscripts):
                return Union[subscripts]

        return None

    def validate_literal(self, val: Any) -> None:
        """Raise TypeError if val is not a valid literal for this type"""
        if val is None:
            if not self.nullable:
                raise TypeError('Expected non-None value')
            else:
                return
        self._validate_literal(val)

    def validate_media(self, val: Any) -> None:
        """
        Raise TypeError if val is not a path to a valid media file (or a valid in-memory byte sequence) for this type
        """
        if self.is_media_type():
            raise NotImplementedError(f'validate_media() not implemented for {self.__class__.__name__}')

    def _validate_file_path(self, val: Any) -> None:
        """Raises TypeError if not a valid local file path or not a path/byte sequence"""
        if isinstance(val, str):
            path = parse_local_file_path(val)
            if path is not None and not path.is_file():
                raise TypeError(f'File not found: {path}')
        elif not isinstance(val, bytes):
            raise TypeError(f'expected file path or bytes, got {type(val)}')

    @abc.abstractmethod
    def _validate_literal(self, val: Any) -> None:
        """Raise TypeError if val is not a valid literal for this type"""

    def _create_literal(self, val: Any) -> Any:
        """Create a literal of this type from val, including any needed conversions.
        val is guaranteed to be non-None"""
        return val

    def create_literal(self, val: Any) -> Any:
        """Create a literal of this type from val or raise TypeError if not possible"""
        if val is not None:
            val = self._create_literal(val)

        self.validate_literal(val)
        return val

    def print_value(self, val: Any) -> str:
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
        # types that refer to external media files
        return self.is_image_type() or self.is_video_type() or self.is_audio_type() or self.is_document_type()

    def supports_file_offloading(self) -> bool:
        # types that can be offloaded to file-based storage via a CellMaterializationNode
        return self.is_array_type() or self.is_json_type() or self.is_binary_type()

    @classmethod
    @abc.abstractmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        """
        Return corresponding SQLAlchemy type.
        """

    def to_json_schema(self) -> dict[str, Any]:
        if self.nullable:
            return {'anyOf': [self._to_json_schema(), {'type': 'null'}]}
        else:
            return self._to_json_schema()

    def _to_json_schema(self) -> dict[str, Any]:
        raise excs.Error(f'Pixeltable type {self} is not a valid JSON type')

    @classmethod
    def from_np_dtype(cls, dtype: np.dtype, nullable: bool) -> ColumnType | None:
        """
        Return pixeltable type corresponding to a given simple numpy dtype
        """
        if np.issubdtype(dtype, np.integer):
            return IntType(nullable=nullable)

        if np.issubdtype(dtype, np.floating):
            return FloatType(nullable=nullable)

        if dtype == np.bool_:
            return BoolType(nullable=nullable)

        if np.issubdtype(dtype, np.str_):
            return StringType(nullable=nullable)

        if np.issubdtype(dtype, np.character):
            return StringType(nullable=nullable)

        if np.issubdtype(dtype, np.datetime64):
            unit, _ = np.datetime_data(dtype)
            if unit in ('D', 'M', 'Y'):
                return DateType(nullable=nullable)
            else:
                return TimestampType(nullable=nullable)

        return None


class InvalidType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INVALID, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.types.NullType()

    def print_value(self, val: Any) -> str:
        return str(val)

    def _validate_literal(self, val: Any) -> None:
        raise AssertionError()


class StringType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.STRING, nullable=nullable)

    def has_supertype(self) -> bool:
        return not self.nullable

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.String()

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'string'}

    def print_value(self, val: Any) -> str:
        return f"'{val}'"

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError(f'Expected string, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        # Replace null byte within python string with space to avoid issues with Postgres.
        # Use a space to avoid merging words.
        # TODO(orm): this will also be an issue with JSON inputs, would space still be a good replacement?
        if isinstance(val, str) and '\x00' in val:
            return val.replace('\x00', ' ')
        return val


class IntType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INT, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.BigInteger()

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'integer'}

    def _validate_literal(self, val: Any) -> None:
        # bool is a subclass of int, so we need to check for it
        # explicitly first
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f'Expected int, got {val.__class__.__name__}')


class FloatType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.FLOAT, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.Float()

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'number'}

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, float):
            raise TypeError(f'Expected float, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, int):
            return float(val)
        return val


class BoolType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.BOOL, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.Boolean()

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'boolean'}

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, bool):
            raise TypeError(f'Expected bool, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, int):
            return bool(val)
        return val


class TimestampType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.TIMESTAMP, nullable=nullable)

    def has_supertype(self) -> bool:
        return not self.nullable

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.TIMESTAMP(timezone=True)

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, datetime.datetime):
            raise TypeError(f'Expected datetime.datetime, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, str):
            return datetime.datetime.fromisoformat(val)
        # Place naive timestamps in the default time zone
        if isinstance(val, datetime.datetime) and val.tzinfo is None:
            return val.replace(tzinfo=Env.get().default_time_zone)
        return val


class DateType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.DATE, nullable=nullable)

    def has_supertype(self) -> bool:
        return not self.nullable

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.Date()

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, datetime.date):
            raise TypeError(f'Expected datetime.date, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, str):
            return datetime.datetime.fromisoformat(val).date()
        if isinstance(val, datetime.date):
            return val
        return val


class UUIDType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.UUID, nullable=nullable)

    def has_supertype(self) -> bool:
        return not self.nullable

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.UUID(as_uuid=True)

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'string', 'format': 'uuid'}

    def print_value(self, val: Any) -> str:
        return f"'{val}'"

    def _to_base_str(self) -> str:
        return 'UUID'

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, uuid.UUID):
            raise TypeError(f'Expected uuid.UUID, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, str):
            return uuid.UUID(val)
        return val


class BinaryType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.BINARY, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.LargeBinary()

    def _to_base_str(self) -> str:
        return 'Binary'

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, bytes):
            raise TypeError(f'Expected `bytes`, got `{val.__class__.__name__}`')


class JsonType(ColumnType):
    type_schema: TypeSchema | None

    def __init__(self, type_schema: TypeSchema | None = None, nullable: bool = False):
        super().__init__(self.Type.JSON, nullable=nullable)
        self.type_schema = type_schema

    @classmethod
    def from_json_type_arg(cls, json_subscript: Any) -> JsonType:
        """
        Constructs a JsonType instance from a subscript, i.e., a value `T` that appears in a type hint of the form
        `Json[T]`. It must be one of the following:

        - A Python type that represents a JsonType instance: either a `TypedDict` subclass, a `pydantic.BaseModel`
            subclass, or a type hint of the form `list[T]` or `tuple[T1, T2, ...]` (ellipsis optional), where all of
            the contained types are valid Pixeltable types. In all such cases, the interpretation is identical to
            the same type appearing without the `Json[]` wrapper.
        - A "convenience" list, tuple, or dictionary, directly specifying the type schema. The values/elements of
            the list, tuple, or dictionary may themselves be either valid types or convenience structures.
        """
        if json_subscript is None:
            return JsonType()  # untyped JSON

        if isinstance(json_subscript, list):
            # Convenience list, such as `[Int]`, interpreted as a pure-variadic tuple.
            if len(json_subscript) != 1:
                raise excs.Error(
                    f'Invalid type schema: expected a single-item list; got a list of length {len(json_subscript)}'
                )
            return JsonType(JsonType.TypeSchema(type_spec=[], variadic_type=cls.from_json_type_arg(json_subscript[0])))

        if isinstance(json_subscript, tuple):
            # Convenience tuple, such as `(String, Int, Float)` or `(String, Int, ...)`. A single ellipsis is
            # allowed only in last position.
            variadic_type = None
            if len(json_subscript) > 0 and json_subscript[-1] is Ellipsis:
                if len(json_subscript) == 1:
                    raise ValueError('Invalid type schema: tuple with only `...` is not allowed')
                variadic_type = json_subscript[-2]
                fixed_types = json_subscript[:-2]

            if Ellipsis in fixed_types:
                raise excs.Error('Invalid type schema: `...` allowed only in last position')

            return JsonType(
                JsonType.TypeSchema(
                    type_spec=[cls.from_json_type_arg(item) for item in fixed_types],
                    variadic_type=cls.from_json_type_arg(variadic_type),
                )
            )

        if isinstance(json_subscript, dict):
            # Convenience dictionary; all keys are assumed to be required.
            type_spec: dict[str, ColumnType] = {}
            for key, value in json_subscript.items():
                if not isinstance(key, str):
                    raise excs.Error(
                        f'Invalid type schema: expected keys of type `str`; got type `{type(key).__name__}`'
                    )
                type_spec[key] = cls.from_json_type_arg(value)
            return JsonType(JsonType.TypeSchema(type_spec))

        # Anything else: Convert it to a ColumnType instance in the usual fashion.
        result = ColumnType.from_python_type(json_subscript)
        if not isinstance(result, JsonType):
            raise excs.Error(
                f'Invalid type schema: type argument does not represent a valid JSON type: {json_subscript}'
            )
        return result

    def copy(self, nullable: bool) -> ColumnType:
        return JsonType(type_schema=self.type_schema, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return isinstance(other, JsonType) and self.type_schema == other.type_schema

    def _to_json_schema(self) -> dict[str, Any]:
        if self.type_schema is None:
            return {}
        else:
            return self.type_schema.to_json_schema()

    def _as_dict(self) -> dict:
        result = super()._as_dict()
        if self.type_schema is not None:
            result.update({'type_schema': self.type_schema.as_dict()})
        return result

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        type_schema = d.get('type_schema')
        return cls(
            type_schema=JsonType.TypeSchema.from_dict(type_schema) if type_schema is not None else None,
            nullable=d['nullable'],
        )

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.dialects.postgresql.JSONB()

    def print_value(self, val: Any) -> str:
        val_type = self.infer_literal_type(val)
        if val_type is None:
            return super().print_value(val)
        if val_type == self:
            return str(val)
        return val_type.print_value(val)

    @classmethod
    def from_literal(
        cls, val: list | tuple | dict[str, Any] | pydantic.BaseModel, nullable: bool = False
    ) -> JsonType | None:
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, pydantic.BaseModel):
            val = val.model_dump()

        if not cls.__is_valid_json(val):
            return None

        type_schema: JsonType.TypeSchema
        if isinstance(val, (list, tuple)):
            type_schema = JsonType.TypeSchema(type_spec=[cls.infer_literal_type(item) for item in val])
        else:
            type_schema = JsonType.TypeSchema(
                type_spec={key: cls.infer_literal_type(value) for key, value in val.items()}
            )

        return JsonType(type_schema, nullable)

    def _validate_literal(self, val: Any) -> None:
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, pydantic.BaseModel):
            val = val.model_dump()
        if not self.__is_valid_json(val):
            raise TypeError(f'That literal is not a valid Pixeltable JSON object: {val}')
        if self.type_schema is not None:
            self.type_schema.validate_literal(val)

    @classmethod
    def __is_valid_json(cls, val: Any) -> bool:
        if val is None or isinstance(val, (str, int, float, bool, np.ndarray, PIL.Image.Image, bytes)):
            return True
        if isinstance(val, (list, tuple)):
            return all(cls.__is_valid_json(v) for v in val)
        if isinstance(val, dict):
            return all(isinstance(k, str) and cls.__is_valid_json(v) for k, v in val.items())
        return False

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, pydantic.BaseModel):
            return val.model_dump()
        return val

    def supertype(self, other: ColumnType, for_inference: bool = False) -> JsonType | None:
        # Try using the (much faster) supertype logic in ColumnType first. That will work if, for example, the types
        # are identical except for nullability. If that doesn't work and both types are JsonType, then we will need to
        # merge their schemas.
        basic_supertype = super().supertype(other)
        if basic_supertype is not None:
            assert isinstance(basic_supertype, JsonType)
            return basic_supertype

        if not isinstance(other, JsonType):
            return None

        return JsonType(
            type_schema=self.__superschema(self.type_schema, other.type_schema),
            nullable=(self.nullable or other.nullable),
        )

    @classmethod
    def __superschema(cls, a: TypeSchema | None, b: TypeSchema | None) -> TypeSchema | None:
        if a == b:
            return a
        if a is None or b is None:
            return None

        if isinstance(a.type_spec, list) and isinstance(b.type_spec, list):
            variadic_type: ColumnType | None = None
            if a.variadic_type is not None:
                variadic_type = a.variadic_type
                if b.variadic_type is not None:
                    variadic_type = variadic_type.supertype(b.variadic_type)
                    if variadic_type is None:
                        return None  # incompatible variadic types
            else:
                variadic_type = b.variadic_type

            joined_list: list[ColumnType] = []
            for item_a, item_b in itertools.zip_longest(a.type_spec, b.type_spec):
                if item_a is None:
                    assert isinstance(item_b, ColumnType)
                    # We're past the end of a's type args. The supertype will still be valid as long as the type of
                    # item_b can be incorporated into the variadic type. This may result in a variadic parameter in
                    # the supertype where none exists in either subtype.
                    if variadic_type is None:
                        variadic_type = item_b
                    else:
                        variadic_type = variadic_type.supertype(item_b)
                        if variadic_type is None:
                            return None  # existing variadic type is incompatible with the extra item_b
                elif item_b is None:
                    assert isinstance(item_a, ColumnType)
                    # Same thing in reverse
                    if variadic_type is None:
                        variadic_type = item_a
                    else:
                        variadic_type = variadic_type.supertype(item_a)
                        if variadic_type is None:
                            return None  # existing variadic type is incompatible with the extra item_a
                else:
                    item_supertype = item_a.supertype(item_b)
                    if item_supertype is None:
                        return None  # incompatible types in this position
                    joined_list.append(item_supertype)

            return JsonType.TypeSchema(type_spec=joined_list, variadic_type=variadic_type)

        if isinstance(a.type_spec, dict) and isinstance(b.type_spec, dict):
            joined_dict: dict[str, ColumnType] = {}
            optional_keys: list[str] = []
            for key, item_a in a.type_spec.items():
                if key in b.type_spec:
                    item_supertype = item_a.supertype(b.type_spec[key])
                    if item_supertype is None:
                        return None  # incompatible types for this key
                    else:
                        joined_dict[key] = item_supertype
                        # key is in both a and b, so optionality is derived from a and b
                        if key in a.optional_keys or key in b.optional_keys:
                            optional_keys.append(key)
                else:
                    joined_dict[key] = item_a
                    # key is only in a, so it's optional in the supertype, regardless of its status in a
                    optional_keys.append(key)
            for key, item_b in b.type_spec.items():
                if key not in a.type_spec:
                    joined_dict[key] = item_b
                    optional_keys.append(key)

            return JsonType.TypeSchema(type_spec=joined_dict, optional_keys=optional_keys)

        return None

    def _to_base_str(self) -> str:
        if self.type_schema is None:
            return 'Json'
        else:
            return f'Json[{self.type_schema!r}]'

    @dataclasses.dataclass(frozen=True)
    class TypeSchema:
        type_spec: list[ColumnType] | dict[str, ColumnType]
        variadic_type: ColumnType | None = None
        optional_keys: list[str] = dataclasses.field(default_factory=list)

        def __post_init__(self) -> None:
            assert self.variadic_type is None or isinstance(self.variadic_type, ColumnType), self.variadic_type

        def validate_literal(self, val: Any) -> None:
            if isinstance(self.type_spec, list):
                if not isinstance(val, list):
                    raise TypeError(f'Expected a list; got `{val.__class__.__name__}`: {val!r}')
                if len(val) < len(self.type_spec):
                    qualifier = 'exactly' if self.variadic_type is None else 'at least'
                    raise TypeError(
                        f'Too few items in list: expected {qualifier} {len(self.type_spec)}; got {len(val)}'
                    )
                for i, item in enumerate(val):
                    if i < len(self.type_spec):
                        expected_type = self.type_spec[i]
                    elif self.variadic_type is not None:
                        expected_type = self.variadic_type
                    else:
                        raise TypeError(
                            f'Too many items in list: expected exactly {len(self.type_spec)}; got {len(val)}'
                        )
                    expected_type.validate_literal(item)
            else:
                if not isinstance(val, dict):
                    raise TypeError(f'Expected a dict; got `{val.__class__.__name__}`: {val!r}')
                for key, expected_type in self.type_spec.items():
                    if key in val:
                        expected_type.validate_literal(val[key])
                    elif key not in self.optional_keys:
                        raise TypeError(f'Missing required key: {key!r}')
                for key in val:
                    if key not in self.type_spec:
                        raise TypeError(f'Unexpected key: {key}')

        def to_json_schema(self) -> dict[str, Any]:
            if isinstance(self.type_spec, list):
                prefix_items_schema = [t.to_json_schema() for t in self.type_spec]
                if self.variadic_type is None:
                    return {'type': 'array', 'prefixItems': prefix_items_schema, 'items': False}
                else:
                    items_schema = self.variadic_type.to_json_schema()
                    return {'type': 'array', 'prefixItems': prefix_items_schema, 'items': items_schema}
            else:
                properties = {k: t.to_json_schema() for k, t in self.type_spec.items()}
                required = [k for k in self.type_spec if k not in self.optional_keys]
                schema: dict[str, Any] = {'type': 'object', 'properties': properties, 'additionalProperties': False}
                if len(required) > 0:
                    schema['required'] = required
                return schema

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, JsonType.TypeSchema)
                and self.type_spec == other.type_spec
                and self.variadic_type == other.variadic_type
                and set(self.optional_keys) == set(other.optional_keys)
            )

        def __hash__(self) -> int:
            type_spec_hash = (
                hash(tuple(self.type_spec))
                if isinstance(self.type_spec, list)
                else hash(frozenset(self.type_spec.items()))
            )
            return hash((type_spec_hash, self.variadic_type, frozenset(self.optional_keys)))

        def __repr__(self) -> str:
            if isinstance(self.type_spec, list):
                reprs = [repr(t) for t in self.type_spec]
                if self.variadic_type is not None:
                    reprs.append(f'{self.variadic_type!r}, ...')
                return f'({", ".join(reprs)})'
            else:
                r = repr(self.type_spec)
                if len(self.optional_keys) > 0:
                    r += f', optional_keys={self.optional_keys}'
                return r

        def as_dict(self) -> dict[str, Any]:
            type_spec_d: list | dict
            if isinstance(self.type_spec, list):
                type_spec_d = [t.as_dict() for t in self.type_spec]
            else:
                type_spec_d = {k: t.as_dict() for k, t in self.type_spec.items()}
            return {
                'type_spec': type_spec_d,
                'variadic_type': self.variadic_type.as_dict() if self.variadic_type is not None else None,
                'optional_keys': self.optional_keys,
            }

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> JsonType.TypeSchema:
            type_spec_d = d['type_spec']
            type_spec: list | dict
            if isinstance(type_spec_d, list):
                type_spec = [ColumnType.from_dict(t) for t in type_spec_d]
            else:
                assert isinstance(type_spec_d, dict)
                type_spec = {k: ColumnType.from_dict(t) for k, t in type_spec_d.items()}
            variadic_type_d = d['variadic_type']
            return cls(
                type_spec=type_spec,
                variadic_type=ColumnType.from_dict(variadic_type_d) if variadic_type_d is not None else None,
                optional_keys=d['optional_keys'],
            )


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


class ArrayType(ColumnType):
    pxt_dtype_to_numpy_dtype: ClassVar[dict[ColumnType.Type, np.dtype]] = {
        ColumnType.Type.INT: np.dtype(np.int64),
        ColumnType.Type.FLOAT: np.dtype(np.float32),
        ColumnType.Type.BOOL: np.dtype(np.bool_),
        ColumnType.Type.STRING: np.dtype(np.str_),
    }

    shape: tuple[int | None, ...] | None
    dtype: np.dtype | None

    def __init__(
        self,
        shape: tuple[int | None, ...] | None = None,
        dtype: ColumnType | np.dtype | None = None,
        nullable: bool = False,
    ):
        super().__init__(self.Type.ARRAY, nullable=nullable)
        assert shape is None or dtype is not None, (shape, dtype)  # cannot specify a shape without a dtype
        self.shape = shape
        if dtype is None:
            self.dtype = None
        elif isinstance(dtype, np.dtype):
            # Numpy string has some specifications (endianness, max length, encoding) that we don't support, so we just
            # strip them out.
            if dtype.type == np.str_:
                self.dtype = np.dtype(np.str_)
            else:
                if dtype not in ARRAY_SUPPORTED_NUMPY_DTYPES:
                    raise ValueError(f'Unsupported dtype: {dtype}')
                self.dtype = dtype
        elif isinstance(dtype, ColumnType):
            self.dtype = self.pxt_dtype_to_numpy_dtype.get(dtype._type, None)
            if self.dtype is None:
                raise ValueError(f'Unsupported dtype: {dtype}')
            assert self.dtype in ARRAY_SUPPORTED_NUMPY_DTYPES
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')

    def copy(self, nullable: bool) -> ColumnType:
        return ArrayType(self.shape, self.dtype, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return isinstance(other, ArrayType) and self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash((self._type, self.nullable, self.shape, self.dtype))

    def supertype(self, other: ColumnType, for_inference: bool = False) -> ArrayType | None:
        basic_supertype = super().supertype(other)
        if basic_supertype is not None:
            assert isinstance(basic_supertype, ArrayType)
            return basic_supertype

        if not isinstance(other, ArrayType):
            return None

        # Supertype has dtype only if dtypes are identical. We can change this behavior to consider casting rules or
        # something else if there's demand for it.
        if self.dtype != other.dtype:
            return ArrayType(nullable=(self.nullable or other.nullable))
        super_dtype = self.dtype

        # Determine the shape of the supertype
        super_shape: tuple[int | None, ...] | None
        if self.shape is None or other.shape is None or len(self.shape) != len(other.shape):
            super_shape = None
        else:
            super_shape = tuple(n1 if n1 == n2 else None for n1, n2 in zip(self.shape, other.shape))
        return ArrayType(super_shape, super_dtype, nullable=(self.nullable or other.nullable))

    def _as_dict(self) -> dict:
        result = super()._as_dict()
        shape_as_list = None if self.shape is None else list(self.shape)
        result.update(shape=shape_as_list)

        if self.dtype is None:
            result.update(numpy_dtype=None)
        elif self.dtype == np.str_:
            # str(np.str_) would be something like '<U', but since we don't support the string specifications, just use
            # 'str' instead to avoid confusion.
            result.update(numpy_dtype='str')
        else:
            result.update(numpy_dtype=str(self.dtype))
        return result

    def _to_base_str(self) -> str:
        if self.shape is None and self.dtype is None:
            return 'Array'
        if self.shape is None:
            return f'Array[{self.dtype.name}]'
        assert self.dtype is not None
        return f'Array[{self.shape}, {self.dtype.name}]'

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        assert 'numpy_dtype' in d
        dtype = None if d['numpy_dtype'] is None else np.dtype(d['numpy_dtype'])
        assert 'shape' in d
        shape = None if d['shape'] is None else tuple(d['shape'])
        return cls(shape, dtype, nullable=d['nullable'])

    @classmethod
    def from_literal(cls, val: np.ndarray, nullable: bool = False) -> ArrayType | None:
        assert isinstance(val, np.ndarray)
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

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {val.__class__.__name__}')

        # If column type has a dtype, check if it matches
        if self.dtype == np.str_:
            if val.dtype.type != np.str_:
                raise TypeError(f'Expected numpy.ndarray of dtype {self.dtype}, got numpy.ndarray of dtype {val.dtype}')
        elif self.dtype is not None and self.dtype != val.dtype:
            raise TypeError(f'Expected numpy.ndarray of dtype {self.dtype}, got numpy.ndarray of dtype {val.dtype}')

        # Check that the dtype is one of the supported types
        if val.dtype.type != np.str_ and val.dtype not in ARRAY_SUPPORTED_NUMPY_DTYPES:
            raise TypeError(f'Unsupported dtype {val.dtype}')

        # If a shape is specified, check that there's a match
        if self.shape is not None:
            if len(val.shape) != len(self.shape):
                raise TypeError(
                    f'Expected numpy.ndarray({self.shape}, dtype={self.dtype}), '
                    f'got numpy.ndarray({val.shape}, dtype={val.dtype})'
                )
            # check that the shapes are compatible
            for n1, n2 in zip(val.shape, self.shape):
                assert n1 is not None  # `val` must have a concrete shape
                if n2 is None:
                    continue  # wildcard
                if n1 != n2:
                    raise TypeError(
                        f'Expected numpy.ndarray({self.shape}, dtype={self.dtype}), '
                        f'got numpy.ndarray({val.shape}, dtype={val.dtype})'
                    )

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, (list, tuple)):
            # map python float to whichever numpy float is
            # declared for this type, rather than assume float64
            return np.array(val, dtype=self.dtype)
        return val

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.LargeBinary()


class ImageType(ColumnType):
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        size: tuple[int, int] | None = None,
        mode: str | None = None,
        nullable: bool = False,
    ):
        # TODO: does it make sense to specify only width or height?
        super().__init__(self.Type.IMAGE, nullable=nullable)
        assert not (width is not None and size is not None)
        assert not (height is not None and size is not None)
        if size is not None:
            self.width = size[0]
            self.height = size[1]
        else:
            self.width = width
            self.height = height
        self.mode = mode

    def copy(self, nullable: bool) -> ColumnType:
        return ImageType(self.width, self.height, mode=self.mode, nullable=nullable)

    def _to_base_str(self) -> str:
        params = []
        if self.width is not None or self.height is not None:
            params.append(f'({self.width}, {self.height})')
        if self.mode is not None:
            params.append(repr(self.mode))
        if len(params) == 0:
            params_str = ''
        else:
            params_str = f'[{", ".join(params)}]'
        return f'Image{params_str}'

    def matches(self, other: ColumnType) -> bool:
        return (
            isinstance(other, ImageType)
            and self.width == other.width
            and self.height == other.height
            and self.mode == other.mode
        )

    def __hash__(self) -> int:
        return hash((self._type, self.nullable, self.size, self.mode))

    def supertype(self, other: ColumnType, for_inference: bool = False) -> ImageType | None:
        basic_supertype = super().supertype(other)
        if basic_supertype is not None:
            assert isinstance(basic_supertype, ImageType)
            return basic_supertype

        if not isinstance(other, ImageType):
            return None

        width = self.width if self.width == other.width else None
        height = self.height if self.height == other.height else None
        mode = self.mode if self.mode == other.mode else None
        return ImageType(width=width, height=height, mode=mode, nullable=(self.nullable or other.nullable))

    @property
    def size(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return (self.width, self.height)

    def _as_dict(self) -> dict:
        result = super()._as_dict()
        result.update(width=self.width, height=self.height, mode=self.mode)
        return result

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        assert 'width' in d
        assert 'height' in d
        assert 'mode' in d
        return cls(width=d['width'], height=d['height'], mode=d['mode'], nullable=d['nullable'])

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.String()

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, str) and val.startswith('data:'):
            # try parsing this as a `data:` URL, and if successful, decode the image immediately
            try:
                with urllib.request.urlopen(val) as response:
                    b = response.read()
                img = PIL.Image.open(io.BytesIO(b))
                img.load()
                return img
            except Exception as exc:
                error_msg_val = val if len(val) < 50 else val[:50] + '...'
                raise excs.Error(f'data URL could not be decoded into a valid image: {error_msg_val}') from exc
        return val

    def _validate_literal(self, val: Any) -> None:
        if isinstance(val, PIL.Image.Image):
            return
        self._validate_file_path(val)

    def validate_media(self, val: Any) -> None:
        assert isinstance(val, str)
        try:
            _ = PIL.Image.open(val)
        except PIL.UnidentifiedImageError:
            raise excs.Error(f'Not a valid image: {val}') from None


class VideoType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.VIDEO, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        # stored as a file path
        return sql.String()

    def _validate_literal(self, val: Any) -> None:
        self._validate_file_path(val)

    def validate_media(self, val: Any) -> None:
        assert isinstance(val, str)
        try:
            with av.open(val, 'r') as fh:
                if len(fh.streams.video) == 0:
                    raise excs.Error(f'Not a valid video: {val}')
                # decode a few frames to make sure it's playable
                # TODO: decode all frames? but that's very slow
                num_decoded = 0
                for frame in fh.decode(video=0):
                    _ = frame.to_image()
                    num_decoded += 1
                    if num_decoded == 10:
                        break
                if num_decoded < 2:
                    # this is most likely an image file
                    raise excs.Error(f'Not a valid video: {val}')
        except av.FFmpegError:
            raise excs.Error(f'Not a valid video: {val}') from None


class AudioType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.AUDIO, nullable=nullable)

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        # stored as a file path
        return sql.String()

    def _validate_literal(self, val: Any) -> None:
        self._validate_file_path(val)

    def validate_media(self, val: Any) -> None:
        try:
            with av.open(val) as container:
                if len(container.streams.audio) == 0:
                    raise excs.Error(f'No audio stream in file: {val}')
                audio_stream = container.streams.audio[0]

                # decode everything to make sure it's playable
                # TODO: is there some way to verify it's a playable audio file other than decoding all of it?
                for packet in container.demux(audio_stream):
                    for _ in packet.decode():
                        pass
        except av.FFmpegError as e:
            raise excs.Error(f'Not a valid audio file: {val}\n{e}') from None


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
        def from_extension(cls, ext: str) -> 'DocumentType.DocumentFormat' | None:
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
        super().__init__(self.Type.DOCUMENT, nullable=nullable)
        self.doc_formats = doc_formats
        if doc_formats is not None:
            type_strs = doc_formats.split(',')
            for type_str in type_strs:
                if not hasattr(self.DocumentFormat, type_str):
                    raise ValueError(f'Invalid document type: {type_str}')
            self._doc_formats = [self.DocumentFormat[type_str.upper()] for type_str in type_strs]
        else:
            self._doc_formats = list(self.DocumentFormat)

    def copy(self, nullable: bool) -> ColumnType:
        return DocumentType(doc_formats=self.doc_formats, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return isinstance(other, DocumentType) and self._doc_formats == other._doc_formats

    def __hash__(self) -> int:
        return hash((self._type, self.nullable, self._doc_formats))

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        # stored as a file path
        return sql.String()

    def _validate_literal(self, val: Any) -> None:
        self._validate_file_path(val)

    def validate_media(self, val: Any) -> None:
        assert isinstance(val, str)
        from pixeltable.utils.documents import get_document_handle

        _ = get_document_handle(val)


T = typing.TypeVar('T')


class Required(typing.Generic[T]):
    """
    Marker class to indicate that a column is non-nullable in a schema definition. This has no meaning as a type hint,
    and is intended only for schema declarations.
    """

    pass


String = typing.Annotated[str, StringType(nullable=False)]
Int = typing.Annotated[int, IntType(nullable=False)]
Float = typing.Annotated[float, FloatType(nullable=False)]
Bool = typing.Annotated[bool, BoolType(nullable=False)]
Timestamp = typing.Annotated[datetime.datetime, TimestampType(nullable=False)]
Date = typing.Annotated[datetime.date, DateType(nullable=False)]
UUID = typing.Annotated[uuid.UUID, UUIDType(nullable=False)]
Binary = typing.Annotated[bytes, BinaryType(nullable=False)]


class _PxtType:
    """
    Base class for the Pixeltable type-hint family. Subclasses of this class are meant to be used as type hints, both
    in schema definitions and in UDF signatures. Whereas `ColumnType`s are instantiable and carry semantic information
    about the Pixeltable type system, `_PxtType` subclasses are purely for convenience: they are not instantiable and
    must be resolved to a `ColumnType` (by calling `ColumnType.from_python_type()`) in order to do anything meaningful
    with them.

    `_PxtType` subclasses can be specialized (as type hints) with type parameters; for example:
    `Image[(300, 300), 'RGB']`. The specialized forms resolve to `typing.Annotated` instances whose annotation is a
    `ColumnType`.
    """

    def __init__(self) -> None:
        raise TypeError(f'Type `{type(self)}` cannot be instantiated.')

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        raise NotImplementedError()


class Json(_PxtType):
    def __class_getitem__(cls, item: Any) -> _AnnotatedAlias:
        """
        `item` (the type subscript) must be a `dict` representing a valid JSON Schema.
        """
        return typing.Annotated[Any, JsonType.from_json_type_arg(item)]

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return JsonType(nullable=nullable)


class Array(np.ndarray, _PxtType):
    def __class_getitem__(cls, item: Any) -> _AnnotatedAlias:
        """
        `item` (the type subscript) must be a tuple with at most two elements (in any order):
        - An optional tuple of `int | None`s, specifying the shape of the array
        - A type (`ColumnType | np.dtype`), specifying the dtype of the array
        Examples:
        * Array[(3, None, 2), pxt.Float]
        * Array[(4, 4), np.uint8]
        * Array[np.bool]
        """
        params = item if isinstance(item, tuple) else (item,)
        shape: tuple | None = None
        dtype: ColumnType | np.dtype | None = None
        if not any(isinstance(param, (type, _AnnotatedAlias)) for param in params):
            raise TypeError('Array type parameter must include a dtype.')
        for param in params:
            if isinstance(param, tuple):
                if not all(n is None or (isinstance(n, int) and n >= 1) for n in param):
                    raise TypeError(f'Invalid Array type parameter: {param}')
                if shape is not None:
                    raise TypeError(f'Duplicate Array type parameter: {param}')
                shape = param
            elif isinstance(param, (type, _AnnotatedAlias)):
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
    def __class_getitem__(cls, item: Any) -> _AnnotatedAlias:
        """
        `item` (the type subscript) must be one of the following, or a tuple containing either or both in any order:
        - A 2-tuple of `int`s, specifying the size of the image
        - A string, specifying the mode of the image
        Example: Image[(300, 300), 'RGB']
        """
        if isinstance(item, tuple) and all(n is None or isinstance(n, int) for n in item):
            # It's a tuple of the form (width, height)
            params = (item,)
        elif isinstance(item, tuple):
            # It's a compound tuple (multiple parameters)
            params = item
        else:
            # Not a tuple (single arg)
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


ALL_PIXELTABLE_TYPES = (
    String,
    Bool,
    Int,
    Float,
    Timestamp,
    Json,
    Array,
    Image,
    Video,
    Audio,
    Document,
    Date,
    UUID,
    Binary,
)
