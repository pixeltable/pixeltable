from __future__ import annotations

import abc
import datetime
import enum
import io
import json
import typing
import urllib.parse
import urllib.request
from typing import Any, ClassVar, Iterable, Literal, Mapping, Optional, Sequence, Union

import av
import jsonschema
import jsonschema.protocols
import jsonschema.validators
import numpy as np
import PIL.Image
import pydantic
import sqlalchemy as sql
from typing_extensions import _AnnotatedAlias

import pixeltable.exceptions as excs
from pixeltable.utils import parse_local_file_path

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip


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

        # exprs that don't evaluate to a computable value in Pixeltable, such as an Image member function
        INVALID = 255

        @classmethod
        def supertype(
            cls,
            type1: Optional['ColumnType.Type'],
            type2: Optional['ColumnType.Type'],
            # we need to pass this in because we can't easily append it as a class member
            common_supertypes: dict[tuple['ColumnType.Type', 'ColumnType.Type'], 'ColumnType.Type'],
        ) -> Optional['ColumnType.Type']:
            if type1 == type2:
                return type1
            t = common_supertypes.get((type1, type2))
            if t is not None:
                return t
            t = common_supertypes.get((type2, type1))
            if t is not None:
                return t
            return None

    @enum.unique
    class DType(enum.Enum):
        """
        Base type used in images and arrays
        """

        BOOL = (0,)
        INT8 = (1,)
        INT16 = (2,)
        INT32 = (3,)
        INT64 = (4,)
        UINT8 = (5,)
        UINT16 = (6,)
        UINT32 = (7,)
        UINT64 = (8,)
        FLOAT16 = (9,)
        FLOAT32 = (10,)
        FLOAT64 = 11

    scalar_types: ClassVar[set[Type]] = {Type.STRING, Type.INT, Type.FLOAT, Type.BOOL, Type.TIMESTAMP, Type.DATE}
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
        assert t != cls.Type.INVALID
        if t == cls.Type.STRING:
            return StringType()
        if t == cls.Type.INT:
            return IntType()
        if t == cls.Type.FLOAT:
            return FloatType()
        if t == cls.Type.BOOL:
            return BoolType()
        if t == cls.Type.TIMESTAMP:
            return TimestampType()
        if t == cls.Type.ARRAY:
            return ArrayType()
        if t == cls.Type.JSON:
            return JsonType()
        if t == cls.Type.IMAGE:
            return ImageType()
        if t == cls.Type.VIDEO:
            return VideoType()
        if t == cls.Type.AUDIO:
            return AudioType()
        if t == cls.Type.DOCUMENT:
            return DocumentType()
        if t == cls.Type.DATE:
            return DateType()

    def __repr__(self) -> str:
        return self._to_str(as_schema=False)

    def _to_str(self, as_schema: bool) -> str:
        base_str = self._to_base_str()
        if as_schema:
            return base_str if self.nullable else f'Required[{base_str}]'
        else:
            return f'Optional[{base_str}]' if self.nullable else base_str

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

    def is_supertype_of(self, other: ColumnType, ignore_nullable: bool = False) -> bool:
        if ignore_nullable:
            supertype = self.supertype(other)
            if supertype is None:
                return False
            return supertype.matches(self)
        else:
            return self.supertype(other) == self

    def matches(self, other: ColumnType) -> bool:
        """Two types match if they're equal, aside from nullability"""
        # Default: just compare base types (this works for all types whose only parameter is nullable)
        return self._type == other._type

    def supertype(self, other: ColumnType) -> Optional[ColumnType]:
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
            return None

        return None

    @classmethod
    def infer_literal_type(cls, val: Any, nullable: bool = False) -> Optional[ColumnType]:
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
        if isinstance(val, PIL.Image.Image):
            return ImageType(width=val.width, height=val.height, mode=val.mode, nullable=nullable)
        if isinstance(val, np.ndarray):
            col_type = ArrayType.from_literal(val, nullable=nullable)
            if col_type is not None:
                return col_type
        # this could still be json-serializable
        if isinstance(val, (list, tuple, dict, np.ndarray, pydantic.BaseModel)):
            try:
                JsonType().validate_literal(val)
                return JsonType(nullable=nullable)
            except TypeError:
                return None
        return None

    @classmethod
    def infer_common_literal_type(cls, vals: Iterable[Any]) -> Optional[ColumnType]:
        """
        Returns the most specific type that is a supertype of all literals in `vals`. If no such type
        exists, returns None.

        Args:
            vals: A collection of literals.
        """
        inferred_type: Optional[ColumnType] = None
        for val in vals:
            val_type = cls.infer_literal_type(val)
            if inferred_type is None:
                inferred_type = val_type
            else:
                inferred_type = inferred_type.supertype(val_type)
            if inferred_type is None:
                return None
            if not inferred_type.has_supertype():
                return inferred_type
        return inferred_type

    @classmethod
    def from_python_type(
        cls, t: Union[type, _GenericAlias], nullable_default: bool = False, allow_builtin_types: bool = True
    ) -> Optional[ColumnType]:
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
        """
        origin = typing.get_origin(t)
        type_args = typing.get_args(t)
        if origin is typing.Union:
            # Check if `t` has the form Optional[T].
            if len(type_args) == 2 and type(None) in type_args:
                # `t` is a type of the form Optional[T] (equivalently, Union[T, None] or Union[None, T]).
                # We treat it as the underlying type but with nullable=True.
                underlying_py_type = type_args[0] if type_args[1] is type(None) else type_args[1]
                underlying = cls.from_python_type(underlying_py_type, allow_builtin_types=allow_builtin_types)
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
            # It's something other than Optional[T], Required[T], or an explicitly annotated type.
            if origin is not None:
                # Discard type parameters to ensure that parameterized types such as `list[T]`
                # are correctly mapped to Pixeltable types.
                t = origin
            if isinstance(t, type) and issubclass(t, _PxtType):
                return t.as_col_type(nullable=nullable_default)
            elif allow_builtin_types:
                if t is Literal and len(type_args) > 0:
                    literal_type = cls.infer_common_literal_type(type_args)
                    if literal_type is None:
                        return None
                    return literal_type.copy(nullable=(literal_type.nullable or nullable_default))
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
                if t is PIL.Image.Image:
                    return ImageType(nullable=nullable_default)
                if isinstance(t, type) and issubclass(t, (Sequence, Mapping, pydantic.BaseModel)):
                    return JsonType(nullable=nullable_default)
        return None

    @classmethod
    def normalize_type(
        cls,
        t: Union[ColumnType, type, _AnnotatedAlias],
        nullable_default: bool = False,
        allow_builtin_types: bool = True,
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
        (PIL.Image.Image, 'pxt.Image'),
        (Sequence, 'pxt.Json'),
        (Mapping, 'pxt.Json'),
    ]

    @classmethod
    def __raise_exc_for_invalid_type(cls, t: Union[type, _AnnotatedAlias]) -> None:
        for builtin_type, suggestion in cls.__TYPE_SUGGESTIONS:
            if t is builtin_type or (isinstance(t, type) and issubclass(t, builtin_type)):
                name = t.__name__ if t.__module__ == 'builtins' else f'{t.__module__}.{t.__name__}'
                raise excs.Error(f'Standard Python type `{name}` cannot be used here; use `{suggestion}` instead')
        raise excs.Error(f'Unknown type: {t}')

    @classmethod
    def from_json_schema(cls, schema: dict[str, Any]) -> Optional[ColumnType]:
        # We first express the JSON schema as a Python type, and then convert it to a Pixeltable type.
        # TODO: Is there a meaningful fallback if one of these operations fails? (Maybe another use case for a pxt Any
        #     type?)
        py_type = cls.__json_schema_to_py_type(schema)
        return cls.from_python_type(py_type) if py_type is not None else None

    @classmethod
    def __json_schema_to_py_type(cls, schema: dict[str, Any]) -> Union[type, _GenericAlias, None]:
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

    def is_json_type(self) -> bool:
        return self._type == self.Type.JSON

    def is_array_type(self) -> bool:
        return self._type == self.Type.ARRAY

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
        if isinstance(val, datetime.datetime):
            return val
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


class JsonType(ColumnType):
    json_schema: Optional[dict[str, Any]]
    __validator: Optional[jsonschema.protocols.Validator]

    def __init__(self, json_schema: Optional[dict[str, Any]] = None, nullable: bool = False):
        super().__init__(self.Type.JSON, nullable=nullable)
        self.json_schema = json_schema
        if json_schema is None:
            self.__validator = None
        else:
            validator_cls = jsonschema.validators.validator_for(json_schema)
            validator_cls.check_schema(json_schema)
            self.__validator = validator_cls(json_schema)

    def copy(self, nullable: bool) -> ColumnType:
        return JsonType(json_schema=self.json_schema, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return isinstance(other, JsonType) and self.json_schema == other.json_schema

    def _as_dict(self) -> dict:
        result = super()._as_dict()
        if self.json_schema is not None:
            result.update({'json_schema': self.json_schema})
        return result

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        return cls(json_schema=d.get('json_schema'), nullable=d['nullable'])

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.dialects.postgresql.JSONB()

    def _to_json_schema(self) -> dict[str, Any]:
        if self.json_schema is None:
            return {}
        return self.json_schema

    def print_value(self, val: Any) -> str:
        val_type = self.infer_literal_type(val)
        if val_type is None:
            return super().print_value(val)
        if val_type == self:
            return str(val)
        return val_type.print_value(val)

    def _validate_literal(self, val: Any) -> None:
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, pydantic.BaseModel):
            val = val.model_dump()
        if not self.__is_valid_json(val):
            raise TypeError(f'That literal is not a valid Pixeltable JSON object: {val}')
        if self.__validator is not None:
            self.__validator.validate(val)

    @classmethod
    def __is_valid_json(cls, val: Any) -> bool:
        if val is None or isinstance(val, (str, int, float, bool)):
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

    def supertype(self, other: ColumnType) -> Optional[JsonType]:
        # Try using the (much faster) supertype logic in ColumnType first. That will work if, for example, the types
        # are identical except for nullability. If that doesn't work and both types are JsonType, then we will need to
        # merge their schemas.
        basic_supertype = super().supertype(other)
        if basic_supertype is not None:
            assert isinstance(basic_supertype, JsonType)
            return basic_supertype

        if not isinstance(other, JsonType):
            return None

        if self.json_schema is None or other.json_schema is None:
            return JsonType(nullable=(self.nullable or other.nullable))

        superschema = self.__superschema(self.json_schema, other.json_schema)

        return JsonType(
            json_schema=(None if len(superschema) == 0 else superschema), nullable=(self.nullable or other.nullable)
        )

    @classmethod
    def __superschema(cls, a: dict[str, Any], b: dict[str, Any]) -> Optional[dict[str, Any]]:
        # Defining a general type hierarchy over all JSON schemas would be a challenging problem. In order to keep
        # things manageable, we only define a hierarchy among "conforming" schemas, which provides enough generality
        # for the most important use cases (unions for type inference, validation of inline exprs). A schema is
        # considered to be conforming if either:
        # (i) it is a scalar (string, integer, number, boolean) or dictionary (object) type; or
        # (ii) it is an "anyOf" schema of one of the above types and the exact schema {'type': 'null'}.
        # Conforming schemas are organized into a type hierarchy in an internally consistent way. Nonconforming
        # schemas are allowed, but they are isolates in the type hierarchy: a nonconforming schema has no proper
        # subtypes, and its only proper supertype is an unconstrained JsonType().
        #
        # There is some subtlety in the handling of nullable fields. Nullable fields are represented in JSON
        # schemas as (for example) {'anyOf': [{'type': 'string'}, {'type': 'null'}]}. When finding the supertype
        # of schemas that might be nullable, we first unpack the 'anyOf's, find the supertype of the underlyings,
        # then reapply the 'anyOf' if appropriate. The top-level schema (i.e., JsonType.json_schema) is presumed
        # to NOT be in this form (since nullability is indicated by the `nullable` field of the JsonType object),
        # so this subtlety is applicable only to types that occur in subfields.
        #
        # There is currently no special handling of lists; distinct schemas with type 'array' will union to the
        # generic {'type': 'array'} schema. This could be a TODO item if there is a need for it in the future.

        if a == b:
            return a

        if 'properties' in a and 'properties' in b:
            a_props = a['properties']
            b_props = b['properties']
            a_req = a.get('required', [])
            b_req = b.get('required', [])
            super_props = {}
            super_req = []
            for key, a_prop_schema in a_props.items():
                if key in b_props:  # in both a and b
                    prop_schema = cls.__superschema_with_nulls(a_prop_schema, b_props[key])
                    super_props[key] = prop_schema
                    if key in a_req and key in b_req:
                        super_req.append(key)
                else:  # in a but not b
                    # Add it to the supertype schema as optional (regardless of its status in a)
                    super_props[key] = a_prop_schema
            for key, b_prop_schema in b_props.items():
                if key not in a_props:  # in b but not a
                    super_props[key] = b_prop_schema
            schema = {'type': 'object', 'properties': super_props}
            if len(super_req) > 0:
                schema['required'] = super_req
            return schema

        a_type = a.get('type')
        b_type = b.get('type')

        if a_type in ('string', 'integer', 'number', 'boolean', 'object', 'array') and a_type == b_type:
            # a and b both have the same type designation, but are not identical. This can happen if
            # (for example) they have validators or other attributes that differ. In this case, we
            # generalize to {'type': t}, where t is their shared type, with no other qualifications.
            return {'type': a_type}

        return {}  # Unresolvable type conflict; the supertype is an unrestricted JsonType.

    @classmethod
    def __superschema_with_nulls(cls, a: dict[str, Any], b: dict[str, Any]) -> Optional[dict[str, Any]]:
        a, a_nullable = cls.__unpack_null_from_schema(a)
        b, b_nullable = cls.__unpack_null_from_schema(b)

        result = cls.__superschema(a, b)
        if len(result) > 0 and (a_nullable or b_nullable):
            # if len(result) == 0, then null is implicitly accepted; otherwise, we need to explicitly allow it
            return {'anyOf': [result, {'type': 'null'}]}
        return result

    @classmethod
    def __unpack_null_from_schema(cls, s: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if 'anyOf' in s and len(s['anyOf']) == 2 and {'type': 'null'} in s['anyOf']:
            try:
                return next(s for s in s['anyOf'] if s != {'type': 'null'}), True
            except StopIteration:
                pass
        return s, False

    def _to_base_str(self) -> str:
        if self.json_schema is None:
            return 'Json'
        elif 'title' in self.json_schema:
            return f'Json[{self.json_schema["title"]}]'
        else:
            return f'Json[{self.json_schema}]'


class ArrayType(ColumnType):
    shape: Optional[tuple[Optional[int], ...]]
    pxt_dtype: Optional[ColumnType]
    dtype: Optional[ColumnType.Type]

    def __init__(
        self,
        shape: Optional[tuple[Optional[int], ...]] = None,
        dtype: Optional[ColumnType] = None,
        nullable: bool = False,
    ):
        super().__init__(self.Type.ARRAY, nullable=nullable)
        assert shape is None or dtype is not None, (shape, dtype)  # cannot specify a shape without a dtype
        assert (
            dtype is None
            or dtype.is_int_type()
            or dtype.is_float_type()
            or dtype.is_bool_type()
            or dtype.is_string_type()
        )

        self.shape = shape
        self.pxt_dtype = dtype  # we need this for copy() and __str__()
        self.dtype = None if dtype is None else dtype._type

    def copy(self, nullable: bool) -> ColumnType:
        return ArrayType(self.shape, self.pxt_dtype, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return isinstance(other, ArrayType) and self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash((self._type, self.nullable, self.shape, self.dtype))

    def supertype(self, other: ColumnType) -> Optional[ArrayType]:
        basic_supertype = super().supertype(other)
        if basic_supertype is not None:
            assert isinstance(basic_supertype, ArrayType)
            return basic_supertype

        if not isinstance(other, ArrayType):
            return None

        super_dtype = self.Type.supertype(self.dtype, other.dtype, self.common_supertypes)
        if super_dtype is None:
            # if the dtypes are incompatible, then the supertype is a fully general array
            return ArrayType(nullable=(self.nullable or other.nullable))
        super_shape: Optional[tuple[Optional[int], ...]]
        if self.shape is None or other.shape is None or len(self.shape) != len(other.shape):
            super_shape = None
        else:
            super_shape = tuple(n1 if n1 == n2 else None for n1, n2 in zip(self.shape, other.shape))
        return ArrayType(super_shape, self.make_type(super_dtype), nullable=(self.nullable or other.nullable))

    def _as_dict(self) -> dict:
        result = super()._as_dict()
        shape_as_list = None if self.shape is None else list(self.shape)
        dtype_value = None if self.dtype is None else self.dtype.value
        result.update(shape=shape_as_list, dtype=dtype_value)
        return result

    def _to_base_str(self) -> str:
        if self.shape is None and self.dtype is None:
            return 'Array'
        if self.shape is None:
            return f'Array[{self.pxt_dtype}]'
        assert self.dtype is not None
        return f'Array[{self.shape}, {self.pxt_dtype}]'

    @classmethod
    def _from_dict(cls, d: dict) -> ColumnType:
        assert 'shape' in d
        assert 'dtype' in d
        shape = None if d['shape'] is None else tuple(d['shape'])
        dtype = None if d['dtype'] is None else cls.make_type(cls.Type(d['dtype']))
        return cls(shape, dtype, nullable=d['nullable'])

    @classmethod
    def from_np_dtype(cls, dtype: np.dtype, nullable: bool) -> Optional[ColumnType]:
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
            if unit in ['D', 'M', 'Y']:
                return DateType(nullable=nullable)
            else:
                return TimestampType(nullable=nullable)

        return None

    @classmethod
    def from_literal(cls, val: np.ndarray, nullable: bool = False) -> Optional[ArrayType]:
        # determine our dtype
        assert isinstance(val, np.ndarray)
        pxttype: Optional[ColumnType] = cls.from_np_dtype(val.dtype, nullable)
        if pxttype is None:
            return None
        return cls(val.shape, dtype=pxttype, nullable=nullable)

    def is_valid_literal(self, val: np.ndarray) -> bool:
        if not isinstance(val, np.ndarray):
            return False

        # If a dtype is specified, check that there's a match
        if self.dtype is not None and not np.issubdtype(val.dtype, self.numpy_dtype()):
            return False

        # If no dtype is specified, we still need to check that the dtype is one of the supported types
        if self.dtype is None and not any(
            np.issubdtype(val.dtype, ndtype) for ndtype in [np.int64, np.float32, np.bool_, np.str_]
        ):
            return False

        # If a shape is specified, check that there's a match
        if self.shape is not None:
            if len(val.shape) != len(self.shape):
                return False
            # check that the shapes are compatible
            for n1, n2 in zip(val.shape, self.shape):
                assert n1 is not None  # `val` must have a concrete shape
                if n2 is None:
                    continue  # wildcard
                if n1 != n2:
                    return False

        return True

    def _to_json_schema(self) -> dict[str, Any]:
        return {'type': 'array', 'items': self.pxt_dtype._to_json_schema()}

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {val.__class__.__name__}')
        if not self.is_valid_literal(val):
            if self.shape is not None:
                raise TypeError(
                    f'Expected numpy.ndarray({self.shape}, dtype={self.numpy_dtype()}), '
                    f'got numpy.ndarray({val.shape}, dtype={val.dtype})'
                )
            elif self.dtype is not None:
                raise TypeError(
                    f'Expected numpy.ndarray of dtype {self.numpy_dtype()}, got numpy.ndarray of dtype {val.dtype}'
                )
            else:
                raise TypeError(f'Unsupported dtype for numpy.ndarray: {val.dtype}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, (list, tuple)):
            # map python float to whichever numpy float is
            # declared for this type, rather than assume float64
            return np.array(val, dtype=self.numpy_dtype())
        return val

    @classmethod
    def to_sa_type(cls) -> sql.types.TypeEngine:
        return sql.LargeBinary()

    def numpy_dtype(self) -> Optional[np.dtype]:
        if self.dtype is None:
            return None
        if self.dtype == self.Type.INT:
            return np.dtype(np.int64)
        if self.dtype == self.Type.FLOAT:
            return np.dtype(np.float32)
        if self.dtype == self.Type.BOOL:
            return np.dtype(np.bool_)
        if self.dtype == self.Type.STRING:
            return np.dtype(np.str_)
        raise AssertionError(self.dtype)


class ImageType(ColumnType):
    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        size: Optional[tuple[int, int]] = None,
        mode: Optional[str] = None,
        nullable: bool = False,
    ):
        """
        TODO: does it make sense to specify only width or height?
        """
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

    def supertype(self, other: ColumnType) -> Optional[ImageType]:
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
    def size(self) -> Optional[tuple[int, int]]:
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

        @classmethod
        def from_extension(cls, ext: str) -> Optional['DocumentType.DocumentFormat']:
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
            return None

    def __init__(self, nullable: bool = False, doc_formats: Optional[str] = None):
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
        if not isinstance(item, dict):
            raise TypeError('Json type parameter must be a dict')

        # The JsonType initializer will validate the JSON Schema.
        return typing.Annotated[Any, JsonType(json_schema=item, nullable=False)]

    @classmethod
    def as_col_type(cls, nullable: bool) -> ColumnType:
        return JsonType(nullable=nullable)


class Array(np.ndarray, _PxtType):
    def __class_getitem__(cls, item: Any) -> _AnnotatedAlias:
        """
        `item` (the type subscript) must be a tuple with exactly two elements (in any order):
        - A tuple of `Optional[int]`s, specifying the shape of the array
        - A type, specifying the dtype of the array
        Example: Array[(3, None, 2), pxt.Float]
        """
        params = item if isinstance(item, tuple) else (item,)
        shape: Optional[tuple] = None
        dtype: Optional[ColumnType] = None
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
        size: Optional[tuple] = None
        mode: Optional[str] = None
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


ALL_PIXELTABLE_TYPES = (String, Bool, Int, Float, Timestamp, Json, Array, Image, Video, Audio, Document)
