from __future__ import annotations

import abc
import datetime
import enum
import json
import typing
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import av  # type: ignore
import numpy as np
import PIL.Image
import sqlalchemy as sql

from pixeltable import exceptions as excs


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

        # exprs that don't evaluate to a computable value in Pixeltable, such as an Image member function
        INVALID = 255

        @classmethod
        def supertype(
                cls, type1: 'ColumnType.Type', type2: 'ColumnType.Type',
                # we need to pass this in because we can't easily append it as a class member
                common_supertypes: Dict[Tuple['ColumnType.Type', 'ColumnType.Type'], 'ColumnType.Type']
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
        BOOL = 0,
        INT8 = 1,
        INT16 = 2,
        INT32 = 3,
        INT64 = 4,
        UINT8 = 5,
        UINT16 = 6,
        UINT32 = 7,
        UINT64 = 8,
        FLOAT16 = 9,
        FLOAT32 = 10,
        FLOAT64 = 11

    scalar_types = {Type.STRING, Type.INT, Type.FLOAT, Type.BOOL, Type.TIMESTAMP}
    numeric_types = {Type.INT, Type.FLOAT}
    common_supertypes: dict[tuple[Type, Type], Type] = {
        (Type.BOOL, Type.INT): Type.INT,
        (Type.BOOL, Type.FLOAT): Type.FLOAT,
        (Type.INT, Type.FLOAT): Type.FLOAT,
    }

    def __init__(self, t: Type, nullable: bool = False):
        self._type = t
        self._nullable = nullable

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
            return self.__class__(nullable=nullable)

    @classmethod
    def serialize_list(cls, type_list: List[ColumnType]) -> str:
        return json.dumps([t.as_dict() for t in type_list])

    def as_dict(self) -> Dict:
        return {
            '_classname': self.__class__.__name__,
            **self._as_dict(),
        }

    def _as_dict(self) -> Dict:
        return {'nullable': self.nullable}

    @classmethod
    def deserialize(cls, type_str: str) -> ColumnType:
        type_dict = json.loads(type_str)
        return cls.from_dict(type_dict)

    @classmethod
    def deserialize_list(cls, type_list_str: str) -> List[ColumnType]:
        type_dict_list = json.loads(type_list_str)
        return [cls.from_dict(type_dict) for type_dict in type_dict_list]

    @classmethod
    def from_dict(cls, type_dict: Dict) -> ColumnType:
        assert '_classname' in type_dict
        type_class = globals()[type_dict['_classname']]
        return type_class._from_dict(type_dict)

    @classmethod
    def _from_dict(cls, d: Dict) -> ColumnType:
        """
        Default implementation: simply invoke c'tor
        """
        assert 'nullable' in d
        return cls(nullable=d['nullable'])  # type: ignore[call-arg]

    @classmethod
    def make_type(cls, t: Type) -> ColumnType:
        assert t != cls.Type.INVALID and t != cls.Type.ARRAY
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

    def __str__(self) -> str:
        return self._type.name.lower()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColumnType) and self.matches(other) and self.nullable == other.nullable

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
            return other
        if other.is_invalid_type():
            return self

        if self.is_scalar_type() and other.is_scalar_type():
            t = self.Type.supertype(self._type, other._type, self.common_supertypes)
            if t is not None:
                return self.make_type(t).copy(nullable=(self.nullable or other.nullable))
            return None

        return None

    @classmethod
    def infer_literal_type(cls, val: Any, nullable: bool = False) -> Optional[ColumnType]:
        if isinstance(val, str):
            return StringType(nullable=nullable)
        if isinstance(val, bool):
            # We have to check bool before int, because isinstance(b, int) is True if b is a Python bool
            return BoolType(nullable=nullable)
        if isinstance(val, int):
            return IntType(nullable=nullable)
        if isinstance(val, float):
            return FloatType(nullable=nullable)
        if isinstance(val, datetime.datetime):
            return TimestampType(nullable=nullable)
        if isinstance(val, PIL.Image.Image):
            return ImageType(width=val.width, height=val.height, mode=val.mode, nullable=nullable)
        if isinstance(val, np.ndarray):
            col_type = ArrayType.from_literal(val, nullable=nullable)
            if col_type is not None:
                return col_type
            # this could still be json-serializable
        if isinstance(val, dict) or isinstance(val, list) or isinstance(val, np.ndarray):
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
        return inferred_type

    @classmethod
    def from_python_type(cls, t: type) -> Optional[ColumnType]:
        if typing.get_origin(t) is typing.Union:
            union_args = typing.get_args(t)
            if union_args[1] is type(None):
                # `t` is a type of the form Optional[T] (equivalently, Union[T, None]).
                # We treat it as the underlying type but with nullable=True.
                underlying = cls.from_python_type(union_args[0])
                if underlying is not None:
                    underlying._nullable = True
                    return underlying
        else:
            # Discard type parameters to ensure that parameterized types such as `list[T]`
            # are correctly mapped to Pixeltable types.
            base = typing.get_origin(t)
            if base is None:
                # No type parameters; the base type is just `t` itself
                base = t
            if base is str:
                return StringType()
            if base is int:
                return IntType()
            if base is float:
                return FloatType()
            if base is bool:
                return BoolType()
            if base is datetime.datetime:
                return TimestampType()
            if issubclass(base, Sequence) or issubclass(base, Mapping):
                return JsonType()
            if issubclass(base, PIL.Image.Image):
                return ImageType()
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
            parsed = urllib.parse.urlparse(val)
            if parsed.scheme != '' and parsed.scheme != 'file':
                return
            path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
            if not path.is_file():
                raise TypeError(f'File not found: {str(path)}')
        else:
            if not isinstance(val, bytes):
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

    @abc.abstractmethod
    def to_sa_type(self) -> sql.types.TypeEngine:
        """
        Return corresponding SQLAlchemy type.
        """
        pass


class InvalidType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INVALID, nullable=nullable)

    def to_sa_type(self) -> sql.types.TypeEngine:
        assert False

    def print_value(self, val: Any) -> str:
        assert False

    def _validate_literal(self, val: Any) -> None:
        assert False


class StringType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.STRING, nullable=nullable)

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.String()

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

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.BigInteger()

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, int):
            raise TypeError(f'Expected int, got {val.__class__.__name__}')


class FloatType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.FLOAT, nullable=nullable)

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.Float()

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

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.Boolean()

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

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.TIMESTAMP()

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, datetime.datetime):
            raise TypeError(f'Expected datetime.datetime, got {val.__class__.__name__}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, str):
            return datetime.datetime.fromisoformat(val)
        return val


class JsonType(ColumnType):
    # TODO: type_spec also needs to be able to express lists
    def __init__(self, type_spec: Optional[Dict[str, ColumnType]] = None, nullable: bool = False):
        super().__init__(self.Type.JSON, nullable=nullable)
        self.type_spec = type_spec

    def copy(self, nullable: bool) -> ColumnType:
        return JsonType(self.type_spec, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return other._type == self.Type.JSON and self.type_spec == other.type_spec

    def supertype(self, other: ColumnType) -> Optional[JsonType]:
        if not isinstance(other, JsonType):
            return None
        if self.type_spec is None:
            # we don't have a type spec and can accept anything accepted by other
            return JsonType(nullable=(self.nullable or other.nullable))
        if other.type_spec is None:
            # we have a type spec but other doesn't
            return JsonType(nullable=(self.nullable or other.nullable))

        # we both have type specs; the supertype's type spec is the union of the two
        type_spec: dict[str, ColumnType] = {}
        type_spec.update(self.type_spec)
        for other_field_name, other_field_type in other.type_spec.items():
            if other_field_name not in type_spec:
                type_spec[other_field_name] = other_field_type
            else:
                # both type specs have this field
                field_type = type_spec[other_field_name].supertype(other_field_type)
                if field_type is None:
                    # conflicting types
                    return JsonType(nullable=(self.nullable or other.nullable))
                type_spec[other_field_name] = field_type
        return JsonType(type_spec, nullable=(self.nullable or other.nullable))

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        if self.type_spec is not None:
            type_spec_dict = {field_name: field_type.serialize() for field_name, field_type in self.type_spec.items()}
            result.update({'type_spec': type_spec_dict})
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> ColumnType:
        type_spec = None
        if 'type_spec' in d:
            type_spec = {
                field_name: cls.deserialize(field_type_dict) for field_name, field_type_dict in d['type_spec'].items()
            }
        return cls(type_spec, nullable=d['nullable'])

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.dialects.postgresql.JSONB()

    def print_value(self, val: Any) -> str:
        val_type = self.infer_literal_type(val)
        if val_type is None:
            return super().print_value(val)
        if val_type == self:
            return str(val)
        return val_type.print_value(val)

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, dict) and not isinstance(val, list):
            raise TypeError(f'Expected dict or list, got {val.__class__.__name__}')
        try:
            _ = json.dumps(val)
        except TypeError as e:
            raise TypeError(f'Expected JSON-serializable object, got {val}')

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, tuple):
            val = list(val)
        return val


class ArrayType(ColumnType):
    def __init__(self, shape: tuple[Union[int, None], ...], dtype: ColumnType, nullable: bool = False):
        super().__init__(self.Type.ARRAY, nullable=nullable)
        self.shape = shape
        assert dtype.is_int_type() or dtype.is_float_type() or dtype.is_bool_type() or dtype.is_string_type()
        self.pxt_dtype = dtype
        self.dtype = dtype._type

    def copy(self, nullable: bool) -> ColumnType:
        return ArrayType(self.shape, self.pxt_dtype, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return other._type == self.Type.ARRAY and self.shape == other.shape and self.dtype == other.dtype

    def supertype(self, other: ColumnType) -> Optional[ArrayType]:
        if not isinstance(other, ArrayType):
            return None
        if len(self.shape) != len(other.shape):
            return None
        base_type = self.Type.supertype(self.dtype, other.dtype, self.common_supertypes)
        if base_type is None:
            return None
        shape = [n1 if n1 == n2 else None for n1, n2 in zip(self.shape, other.shape)]
        return ArrayType(tuple(shape), self.make_type(base_type), nullable=(self.nullable or other.nullable))

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        result.update(shape=list(self.shape), dtype=self.dtype.value)
        return result

    def __str__(self) -> str:
        return f'{self._type.name.lower()}({self.shape}, dtype={self.dtype.name})'

    @classmethod
    def _from_dict(cls, d: Dict) -> ColumnType:
        assert 'shape' in d
        assert 'dtype' in d
        shape = tuple(d['shape'])
        dtype = cls.make_type(cls.Type(d['dtype']))
        return cls(shape, dtype, nullable=d['nullable'])

    @classmethod
    def from_literal(cls, val: np.ndarray, nullable: bool = False) -> Optional[ArrayType]:
        # determine our dtype
        assert isinstance(val, np.ndarray)
        if np.issubdtype(val.dtype, np.integer):
            dtype: ColumnType = IntType()
        elif np.issubdtype(val.dtype, np.floating):
            dtype = FloatType()
        elif val.dtype == np.bool_:
            dtype = BoolType()
        elif val.dtype == np.str_:
            dtype = StringType()
        else:
            return None
        return cls(val.shape, dtype=dtype, nullable=nullable)

    def is_valid_literal(self, val: np.ndarray) -> bool:
        if not isinstance(val, np.ndarray):
            return False
        if len(val.shape) != len(self.shape):
            return False
        # check that the shapes are compatible
        for n1, n2 in zip(val.shape, self.shape):
            if n1 is None:
                return False
            if n2 is None:
                # wildcard
                continue
            if n1 != n2:
                return False
        return val.dtype == self.numpy_dtype()

    def _validate_literal(self, val: Any) -> None:
        if not isinstance(val, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {val.__class__.__name__}')
        if not self.is_valid_literal(val):
            raise TypeError((
                f'Expected ndarray({self.shape}, dtype={self.numpy_dtype()}), '
                f'got ndarray({val.shape}, dtype={val.dtype})'))

    def _create_literal(self, val: Any) -> Any:
        if isinstance(val, (list,tuple)):
            # map python float to whichever numpy float is
            # declared for this type, rather than assume float64
            return np.array(val, dtype=self.numpy_dtype())
        return val

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.LargeBinary()

    def numpy_dtype(self) -> np.dtype:
        if self.dtype == self.Type.INT:
            return np.dtype(np.int64)
        if self.dtype == self.Type.FLOAT:
            return np.dtype(np.float32)
        if self.dtype == self.Type.BOOL:
            return np.dtype(np.bool_)
        if self.dtype == self.Type.STRING:
            return np.dtype(np.str_)
        assert False


class ImageType(ColumnType):
    def __init__(
            self, width: Optional[int] = None, height: Optional[int] = None, size: Optional[Tuple[int, int]] = None,
            mode: Optional[str] = None, nullable: bool = False
    ):
        """
        TODO: does it make sense to specify only width or height?
        """
        super().__init__(self.Type.IMAGE, nullable=nullable)
        assert not(width is not None and size is not None)
        assert not(height is not None and size is not None)
        if size is not None:
            self.width = size[0]
            self.height = size[1]
        else:
            self.width = width
            self.height = height
        self.mode = mode

    def copy(self, nullable: bool) -> ColumnType:
        return ImageType(self.width, self.height, mode=self.mode, nullable=nullable)

    def __str__(self) -> str:
        if self.width is not None or self.height is not None or self.mode is not None:
            params_str = ''
            if self.width is not None:
                params_str = f'width={self.width}'
            if self.height is not None:
                if len(params_str) > 0:
                    params_str += ', '
                params_str += f'height={self.height}'
            if self.mode is not None:
                if len(params_str) > 0:
                    params_str += ', '
                params_str += f'mode={self.mode}'
            params_str = f'({params_str})'
        else:
            params_str = ''
        return f'{self._type.name.lower()}{params_str}'

    def matches(self, other: ColumnType) -> bool:
        return (
            other._type == self.Type.IMAGE
            and self.width == other.width
            and self.height == other.height
            and self.mode == other.mode
        )

    def supertype(self, other: ColumnType) -> Optional[ImageType]:
        if not isinstance(other, ImageType):
            return None
        width = self.width if self.width == other.width else None
        height = self.height if self.height == other.height else None
        mode = self.mode if self.mode == other.mode else None
        return ImageType(width=width, height=height, mode=mode, nullable=(self.nullable or other.nullable))

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        if self.width is None or self.height is None:
            return None
        return (self.width, self.height)

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        result.update(width=self.width, height=self.height, mode=self.mode)
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> ColumnType:
        assert 'width' in d
        assert 'height' in d
        assert 'mode' in d
        return cls(width=d['width'], height=d['height'], mode=d['mode'], nullable=d['nullable'])

    def to_sa_type(self) -> sql.types.TypeEngine:
        return sql.String()

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

    def to_sa_type(self) -> sql.types.TypeEngine:
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
        except av.AVError:
            raise excs.Error(f'Not a valid video: {val}') from None


class AudioType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.AUDIO, nullable=nullable)

    def to_sa_type(self) -> sql.types.TypeEngine:
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
        except av.AVError as e:
            raise excs.Error(f'Not a valid audio file: {val}\n{e}') from None


class DocumentType(ColumnType):
    @enum.unique
    class DocumentFormat(enum.Enum):
        HTML = 0
        MD = 1
        PDF = 2

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
            self._doc_formats = [t for t in self.DocumentFormat]

    def copy(self, nullable: bool) -> ColumnType:
        return DocumentType(doc_formats=self.doc_formats, nullable=nullable)

    def matches(self, other: ColumnType) -> bool:
        return other._type == self.Type.DOCUMENT and self._doc_formats == other._doc_formats

    def to_sa_type(self) -> sql.types.TypeEngine:
        # stored as a file path
        return sql.String()

    def _validate_literal(self, val: Any) -> None:
        self._validate_file_path(val)

    def validate_media(self, val: Any) -> None:
        assert isinstance(val, str)
        from pixeltable.utils.documents import get_document_handle
        try:
            dh = get_document_handle(val)
            if dh is None:
                raise excs.Error(f'Not a recognized document format: {val}')
        except Exception as e:
            raise excs.Error(f'Not a recognized document format: {val}') from None
