import abc
from typing import Any, Optional, Tuple, Dict, Callable, List, Union
import enum
import datetime
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
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

        def to_tf(self) -> tf.dtypes.DType:
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
                # we need to pass this in because we can't easily add it as a class member
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

    def __init__(self, t: Type):
        self._type = t

    @property
    def type_enum(self) -> Type:
        return self._type

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def serialize_list(cls, type_list: List['ColumnType']) -> str:
        return json.dumps([t.as_dict() for t in type_list])

    def as_dict(self) -> Dict:
        return {
            '_classname': self.__class__.__name__,
            **self._as_dict(),
        }

    def _as_dict(self) -> Dict:
        return {}

    @classmethod
    def deserialize(cls, type_str: str) -> 'ColumnType':
        type_dict = json.loads(type_str)
        return cls.from_dict(type_dict)

    @classmethod
    def deserialize_list(cls, type_list_str: str) -> List['ColumnType']:
        type_dict_list = json.loads(type_list_str)
        return [cls.from_dict(type_dict) for type_dict in type_dict_list]

    @classmethod
    def from_dict(cls, type_dict: Dict) -> 'ColumnType':
        assert '_classname' in type_dict
        type_class = globals()[type_dict['_classname']]
        return type_class._from_dict(type_dict)

    @classmethod
    def _from_dict(cls, d: Dict) -> 'ColumnType':
        """
        Default implementation: simply invoke c'tor without arguments
        """
        return cls()

    @classmethod
    def make_type(cls, t: Type) -> 'ColumnType':
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
        assert isinstance(other, ColumnType)
        if False and type(self) != type(other):
            return False
        for member_var in vars(self).keys():
            if getattr(self, member_var) != getattr(other, member_var):
                return False
        return True

    @classmethod
    def supertype(cls, type1: 'ColumnType', type2: 'ColumnType') -> Optional['ColumnType']:
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
    def _supertype(cls, type1: 'ColumnType', type2: 'ColumnType') -> Optional['ColumnType']:
        """
        Class-specific implementation of determining the supertype. type1 and type2 are from the same subclass of
        ColumnType.
        """
        pass


    @classmethod
    def get_value_type(cls, val: Any) -> 'ColumnType':
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

    def conversion_fn(self, target: 'ColumnType') -> Optional[Callable[[Any], Any]]:
        """
        Return Callable that converts a column value of type self to a value of type 'target'.
        Returns None if conversion isn't possible.
        """
        return None

    @abc.abstractmethod
    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        pass


class InvalidType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.INVALID)

    def to_sql(self) -> str:
        assert False

    def to_sa_type(self) -> Any:
        assert False

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        raise TypeError(f'Invalid type cannot be converted to Tensorflow')


class StringType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.STRING)

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

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        return tf.TensorSpec(shape=(), dtype=tf.string)


class IntType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.INT)

    def to_sql(self) -> str:
        return 'INTEGER'

    def to_sa_type(self) -> str:
        return sql.Integer

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        # TODO: how to specify the correct int subtype?
        return tf.TensorSpec(shape=(), dtype=tf.int64)


class FloatType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.FLOAT)

    def to_sql(self) -> str:
        return 'FLOAT'

    def to_sa_type(self) -> str:
        return sql.Float

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        # TODO: how to specify the correct float subtype?
        return tf.TensorSpec(shape=(), dtype=tf.float32)


class BoolType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.BOOL)

    def to_sql(self) -> str:
        return 'BOOLEAN'

    def to_sa_type(self) -> str:
        return sql.Boolean

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        # TODO: how to specify the correct int subtype?
        return tf.TensorSpec(shape=(), dtype=tf.bool)


class TimestampType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.TIMESTAMP)

    def to_sql(self) -> str:
        return 'INTEGER'

    def to_sa_type(self) -> str:
        return sql.TIMESTAMP

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        raise TypeError(f'Timestamp type cannot be converted to Tensorflow')


class JsonType(ColumnType):
    # TODO: type_spec also needs to be able to express lists
    def __init__(self, type_spec: Optional[Dict[str, ColumnType]] = None):
        super().__init__(self.Type.JSON)
        self.type_spec = type_spec

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        if self.type_spec is not None:
            type_spec_dict = {field_name: field_type.serialize() for field_name, field_type in self.type_spec.items()}
            result.update({'type_spec': type_spec_dict})
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> 'ColumnType':
        type_spec = None
        if 'type_spec' in d:
            type_spec = {
                field_name: cls.deserialize(field_type_dict) for field_name, field_type_dict in d['type_spec'].items()
            }
        return cls(type_spec)

    def to_sql(self) -> str:
        return 'JSONB'

    def to_sa_type(self) -> str:
        return sql.dialects.postgresql.JSONB

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        if self.type_spec is None:
            raise TypeError(f'Cannot convert {self.__class__.__name__} with missing type spec to TensorFlow')
        return {k: v.to_tf() for k, v in self.type_spec.items()}


class ArrayType(ColumnType):
    def __init__(
            self, shape: Tuple[Union[int, None], ...], dtype: ColumnType.Type):
        super().__init__(self.Type.ARRAY)
        self.shape = shape
        self.dtype = dtype

    def _supertype(cls, type1: 'ArrayType', type2: 'ArrayType') -> Optional['ArrayType']:
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
        return f'{self.__class__.__name__}({self.shape}, dtype={self.dtype.name})'

    @classmethod
    def _from_dict(cls, d: Dict) -> 'ColumnType':
        assert 'shape' in d
        assert 'dtype' in d
        shape = tuple(d['shape'])
        dtype = cls.Type(d['dtype'])
        return cls(shape, dtype)

    def to_sql(self) -> str:
        return 'BYTEA'

    def to_sa_type(self) -> str:
        return sql.VARBINARY

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        return tf.TensorSpec(shape=self.shape, dtype=self.dtype.to_tf())


class ImageType(ColumnType):
    @enum.unique
    class Mode(enum.Enum):
        L = 0,
        RGB = 1

        @classmethod
        def from_pil(cls, pil_mode: str) -> 'Mode':
            if pil_mode == 'L':
                return cls.L
            if pil_mode == 'RGB':
                return cls.RGB

        def to_pil(self) -> str:
            return self.name

        def num_channels(self) -> int:
            return len(self.name)

    def __init__(
            self, width: Optional[int] = None, height: Optional[int] = None, size: Optional[Tuple[int, int]] = None,
            mode: Optional[Mode] = None
    ):
        """
        TODO: does it make sense to specify only width or height?
        """
        super().__init__(self.Type.IMAGE)
        assert not(width is not None and size is not None)
        assert not(height is not None and size is not None)
        if size is not None:
            self.width = size[0]
            self.height = size[1]
        else:
            self.width = width
            self.height = height
        self.mode = mode

    @property
    def num_channels(self) -> Optional[int]:
        return None if self.mode is None else self.mode.num_channels()

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        result.update(width=self.width, height=self.height, mode=self.mode.value if self.mode is not None else None)
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> 'ColumnType':
        assert 'width' in d
        assert 'height' in d
        assert 'mode' in d
        mode_val = d['mode']
        return cls(width=d['width'], height=d['height'], mode=cls.Mode(mode_val) if mode_val is not None else None)

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

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        return tf.TensorSpec(shape=(self.height, self.width, self.num_channels), dtype=tf.uint8)


class VideoType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.VIDEO)

    def _as_dict(self) -> Dict:
        result = super()._as_dict()
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> 'ColumnType':
        return cls()

    def to_sql(self) -> str:
        # stored as a file path
        return 'VARCHAR'

    def to_sa_type(self) -> str:
        return sql.String

    def to_tf(self) -> Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]:
        assert False
