from __future__ import annotations

import abc
import datetime
import enum
import io
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, Callable, List, Union

import cv2
import nos
import numpy as np
from PIL import Image

from pixeltable import exceptions as exc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import tensorflow as tf
import PIL.Image
import sqlalchemy as sql



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

        # exprs that don't evaluate to a computable value in Pixeltable, such as an Image member function
        INVALID = 9

        def to_tf(self) -> 'tf.dtypes.DType':
            import tensorflow as tf
            if self == self.STRING:
                return tf.string
            if self == self.INT:
                return tf.int64
            if self == self.FLOAT:
                return tf.float32
            if self == self.BOOL:
                return tf.bool
            raise TypeError(f'Cannot convert {self} to TensorFlow')

        @classmethod
        def supertype(
                cls, type1: 'Type', type2: 'Type',
                # we need to pass this in because we can't easily append it as a class member
                common_supertypes: Dict[Tuple['Type', 'Type'], 'Type']
        ) -> Optional['Type']:
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
    common_supertypes: Dict[Tuple[Type, Type], Type] = {
        (Type.BOOL, Type.INT): Type.INT,
        (Type.BOOL, Type.FLOAT): Type.FLOAT,
        (Type.INT, Type.FLOAT): Type.FLOAT,
    }

    def __init__(self, t: Type, nullable: bool = False):
        self._type = t
        self.nullable = nullable

    @property
    def type_enum(self) -> Type:
        return self._type

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

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
        return cls(nullable=d['nullable'])

    @classmethod
    def from_nos(cls, type_info: nos.common.spec.ObjectTypeInfo, ignore_shape: bool = False) -> ColumnType:
        """Convert ObjectTypeInfo to ColumnType"""
        if type_info.base_spec() is None:
            if type_info.base_type() == str:
                return StringType()
            if type_info.base_type() == int:
                return IntType()
            if type_info.base_type() == float:
                return FloatType()
            if type_info.base_type() == bool:
                return BoolType()
            else:
                raise exc.Error(f'Cannot convert {type_info} to ColumnType')
        elif isinstance(type_info.base_spec(), nos.common.ImageSpec):
            size = None
            if not ignore_shape and type_info.base_spec().shape is not None:
                size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
            # TODO: set mode
            return ImageType(size=size)
        elif isinstance(type_info.base_spec(), nos.common.TensorSpec):
            return ArrayType(shape=type_info.base_spec().shape, dtype=FloatType())
        else:
            raise exc.Error(f'Cannot convert {type_info} to ColumnType')


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

    def __str__(self) -> str:
        return self._type.name.lower()

    def __eq__(self, other: object) -> bool:
        return self.matches(other) and self.nullable == other.nullable

    def is_supertype_of(self, other: ColumnType) -> bool:
        if type(self) != type(other):
            return False
        if self.matches(other):
            return True
        return self._is_supertype_of(other)

    @abc.abstractmethod
    def _is_supertype_of(self, other: ColumnType) -> bool:
        return False

    def matches(self, other: object) -> bool:
        """Two types match if they're equal, aside from nullability"""
        assert isinstance(other, ColumnType)
        if type(self) != type(other):
            return False
        for member_var in vars(self).keys():
            if member_var == 'nullable':
                continue
            if getattr(self, member_var) != getattr(other, member_var):
                return False
        return True

    @classmethod
    def supertype(cls, type1: ColumnType, type2: ColumnType) -> Optional[ColumnType]:
        if type1 == type2:
            return type1

        if type1.is_invalid_type():
            return type2
        if type2.is_invalid_type():
            return type1

        if type1.is_scalar_type() and type2.is_scalar_type():
            t = cls.Type.supertype(type1._type, type2._type, cls.common_supertypes)
            if t is not None:
                return cls.make_type(t)
            return None

        if type1._type == type2._type:
            return cls._supertype(type1, type2)

        return None

    @classmethod
    @abc.abstractmethod
    def _supertype(cls, type1: ColumnType, type2: ColumnType) -> Optional[ColumnType]:
        """
        Class-specific implementation of determining the supertype. type1 and type2 are from the same subclass of
        ColumnType.
        """
        pass

    @classmethod
    def infer_literal_type(cls, val: Any) -> Optional[ColumnType]:
        if isinstance(val, str):
            return StringType()
        if isinstance(val, int):
            return IntType()
        if isinstance(val, float):
            return FloatType()
        if isinstance(val, bool):
            return BoolType()
        if isinstance(val, datetime.datetime) or isinstance(val, datetime.date):
            return TimestampType()
        if isinstance(val, np.ndarray):
            col_type = ArrayType.from_literal(val)
            if col_type is not None:
                return col_type
            # this could still be json-serializable
        if isinstance(val, dict) or isinstance(val, np.ndarray):
            try:
                JsonType().validate_literal(val)
                return JsonType()
            except TypeError:
                return None
        return None

    @abc.abstractmethod
    def validate_literal(self, val: Any) -> None:
        """Raise TypeError if val is not a valid literal for this type"""
        pass

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

    @abc.abstractmethod
    def to_sql(self) -> str:
        """
        Return corresponding Postgres type.
        """
        pass

    @abc.abstractmethod
    def to_sa_type(self) -> Any:
        """
        Return corresponding SQLAlchemy type.
        return type Any: there doesn't appear to be a superclass for the sqlalchemy types
        """
        assert self._type != self.Type.INVALID
        if self._type == self.Type.STRING:
            return sql.String
        if self._type == self.Type.INT:
            return sql.Integer
        if self._type == self.Type.FLOAT:
            return sql.Float
        if self._type == self.Type.BOOL:
            return sql.Boolean
        if self._type == self.Type.TIMESTAMP:
            return sql.TIMESTAMP
        if self._type == self.Type.IMAGE:
            # the URL
            return sql.String
        if self._type == self.Type.JSON:
            return sql.dialects.postgresql.JSONB
        if self._type == self.Type.ARRAY:
            return sql.VARBINARY
        assert False

    @staticmethod
    def no_conversion(v: Any) -> Any:
        """
        Special return value of conversion_fn() that indicates that no conversion is necessary.
        Should not be called
        """
        assert False

    def conversion_fn(self, target: ColumnType) -> Optional[Callable[[Any], Any]]:
        """
        Return Callable that converts a column value of type self to a value of type 'target'.
        Returns None if conversion isn't possible.
        """
        return None

    @abc.abstractmethod
    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        pass


class InvalidType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INVALID, nullable=nullable)

    def to_sql(self) -> str:
        assert False

    def to_sa_type(self) -> Any:
        assert False

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        raise TypeError(f'Invalid type cannot be converted to Tensorflow')

    def print_value(self, val: Any) -> str:
        assert False

    def validate_literal(self, val: Any) -> None:
        assert False

class StringType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.STRING, nullable=nullable)

    def conversion_fn(self, target: ColumnType) -> Optional[Callable[[Any], Any]]:
        if not target.is_timestamp_type():
            return None
        def convert(val: str) -> Optional[datetime]:
            try:
                dt = datetime.datetime.fromisoformat(val)
                return dt
            except ValueError:
                return None
        return convert

    def to_sql(self) -> str:
        return 'VARCHAR'

    def to_sa_type(self) -> str:
        return sql.String

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        import tensorflow as tf
        return tf.TensorSpec(shape=(), dtype=tf.string)

    def print_value(self, val: Any) -> str:
        return f"'{val}'"

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError(f'Expected string, got {val}')


class IntType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.INT, nullable=nullable)

    def to_sql(self) -> str:
        return 'INTEGER'

    def to_sa_type(self) -> str:
        return sql.Integer

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        # TODO: how to specify the correct int subtype?
        import tensorflow as tf
        return tf.TensorSpec(shape=(), dtype=tf.int64)

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, int):
            raise TypeError(f'Expected int, got {val}')


class FloatType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.FLOAT, nullable=nullable)

    def to_sql(self) -> str:
        return 'FLOAT'

    def to_sa_type(self) -> str:
        return sql.Float

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        import tensorflow as tf
        # TODO: how to specify the correct float subtype?
        return tf.TensorSpec(shape=(), dtype=tf.float32)

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, float):
            raise TypeError(f'Expected float, got {val}')


class BoolType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.BOOL, nullable=nullable)

    def to_sql(self) -> str:
        return 'BOOLEAN'

    def to_sa_type(self) -> str:
        return sql.Boolean

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        import tensorflow as tf
        # TODO: how to specify the correct int subtype?
        return tf.TensorSpec(shape=(), dtype=tf.bool)

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, bool):
            raise TypeError(f'Expected bool, got {val}')


class TimestampType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.TIMESTAMP, nullable=nullable)

    def to_sql(self) -> str:
        return 'INTEGER'

    def to_sa_type(self) -> str:
        return sql.TIMESTAMP

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        raise TypeError(f'Timestamp type cannot be converted to Tensorflow')

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, datetime.datetime) and not isinstance(val, datetime.date):
            raise TypeError(f'Expected datetime.datetime or datetime.date, got {val}')


class JsonType(ColumnType):
    # TODO: type_spec also needs to be able to express lists
    def __init__(self, type_spec: Optional[Dict[str, ColumnType]] = None, nullable: bool = False):
        super().__init__(self.Type.JSON, nullable=nullable)
        self.type_spec = type_spec

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

    def to_sql(self) -> str:
        return 'JSONB'

    def to_sa_type(self) -> str:
        return sql.dialects.postgresql.JSONB

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        if self.type_spec is None:
            raise TypeError(f'Cannot convert {self.__class__.__name__} with missing type spec to TensorFlow')
        return {k: v.to_tf() for k, v in self.type_spec.items()}

    def print_value(self, val: Any) -> str:
        val_type = self.infer_literal_type(val)
        if val_type == self:
            return str(val)
        return val_type.print_value(val)

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, dict) and not isinstance(val, list):
            raise TypeError(f'Expected dict or list, got {val}')
        try:
            _ = json.dumps(val)
        except TypeError as e:
            raise TypeError(f'Expected JSON-serializable object, got {val}')


class ArrayType(ColumnType):
    def __init__(
            self, shape: Tuple[Union[int, None], ...], dtype: ColumnType, nullable: bool = False):
        super().__init__(self.Type.ARRAY, nullable=nullable)
        self.shape = shape
        assert dtype.is_int_type() or dtype.is_float_type() or dtype.is_bool_type() or dtype.is_string_type()
        self.dtype = dtype._type

    def _supertype(cls, type1: ArrayType, type2: ArrayType) -> Optional[ArrayType]:
        if len(type1.shape) != len(type2.shape):
            return None
        base_type = ColumnType.supertype(type1.dtype, type2.dtype)
        if base_type is None:
            return None
        shape = [n1 if n1 == n2 else None for n1, n2 in zip(type1.shape, type2.shape)]
        return ArrayType(tuple(shape), base_type)

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
    def from_literal(cls, val: np.ndarray) -> Optional[ArrayType]:
        # determine our dtype
        assert isinstance(val, np.ndarray)
        if np.issubdtype(val.dtype, np.integer):
            dtype = IntType()
        elif np.issubdtype(val.dtype, np.floating):
            dtype = FloatType()
        elif val.dtype == np.bool_:
            dtype = BoolType()
        elif val.dtype == np.str_:
            dtype = StringType()
        else:
            return None
        return cls(val.shape, dtype=dtype, nullable=True)

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

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {val}')
        if not self.is_valid_literal(val):
            raise TypeError((
                f'Expected ndarray({self.shape}, dtype={self.numpy_dtype()}), '
                f'got ndarray({val.shape}, dtype={val.dtype})'))

    def to_sql(self) -> str:
        return 'BYTEA'

    def to_sa_type(self) -> str:
        return sql.LargeBinary

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        import tensorflow as tf
        return tf.TensorSpec(shape=self.shape, dtype=self.dtype.to_tf())

    def numpy_dtype(self) -> np.dtype:
        if self.dtype == self.Type.INT:
            return np.dtype(np.int32)
        if self.dtype == self.Type.FLOAT:
            return np.dtype(np.float32)
        if self.dtype == self.Type.BOOL:
            return np.dtype(np.bool_)
        if self.dtype == self.Type.STRING:
            return np.dtype(np.str_)


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

    def _is_supertype_of(self, other: ImageType) -> bool:
        if self.mode != other.mode:
            return False
        if self.width is None and self.height is None:
            return True
        if self.width != other.width and self.height != other.height:
            return False

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        if self.width is None or self.height is None:
            return None
        return (self.width, self.height)

    @property
    def num_channels(self) -> Optional[int]:
        return None if self.mode is None else self.mode.num_channels()

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

    def conversion_fn(self, target: ColumnType) -> Optional[Callable[[Any], Any]]:
        if not target.is_image_type():
            return None
        assert isinstance(target, ImageType)
        if (target.width is None) != (target.height is None):
            # we can't resize only one dimension
            return None
        if (target.width == self.width or target.width is None) \
            and (target.height == self.height or target.height is None) \
            and (target.mode == self.mode or target.mode is None):
            # nothing to do
            return self.no_conversion
        def convert(img: PIL.Image.Image) -> PIL.Image.Image:
            if self.width != target.width or self.height != target.height:
                img = img.resize((target.width, target.height))
            if self.mode != target.mode:
                img = img.convert(target.mode.to_pil())
            return img
        return convert

    def to_sql(self) -> str:
        return 'VARCHAR'

    def to_sa_type(self) -> str:
        return sql.String

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        import tensorflow as tf
        return tf.TensorSpec(shape=(self.height, self.width, self.num_channels), dtype=tf.uint8)

    def validate_literal(self, val: Any) -> None:
        # make sure file path points to a valid image file or binary is a valid image
        if not isinstance(val, (str, bytes)):
            raise TypeError(f'Expected file path or binary, got {val}')

        if isinstance(val, bytes):
            try:
                _ = Image.open(io.BytesIO(val))
            except PIL.UnidentifiedImageError:
                raise TypeError(f'Binary is not a valid image: {val}')
        elif isinstance(val, str):
            try:
                _ = Image.open(val)
            except FileNotFoundError:
                raise TypeError(f'File not found: {val}')
            except PIL.UnidentifiedImageError:
                raise TypeError(f'File is not a valid image: {val}')
        else:
            assert False


class VideoType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(self.Type.VIDEO, nullable=nullable)

    def to_sql(self) -> str:
        # stored as a file path
        return 'VARCHAR'

    def to_sa_type(self) -> str:
        return sql.String

    def to_tf(self) -> Union['tf.TypeSpec', Dict[str, 'tf.TypeSpec']]:
        assert False

    def validate_literal(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError(f'Expected file path, got {val}')
        path = Path(val)
        if not path.is_file():
            raise TypeError(f'File not found: {val}')
        cap = cv2.VideoCapture(val)
        # TODO: this succeeds for image files; figure out how to verify it's a video
        success = cap.isOpened()
        cap.release()
        if not success:
            raise TypeError(f'File is not a valid video: {val}')
