from typing import Any, Optional, Tuple, Dict
import enum

import sqlalchemy as sql


class ColumnType:
    @enum.unique
    class Type(enum.Enum):
        INVALID = 0
        STRING = 1
        INT = 2
        FLOAT = 3
        BOOL = 4
        TIMESTAMP = 5
        IMAGE = 6
        DICT = 7
        ARRAY = 8

    def __init__(self, t: Type):
        self._type = t

    @property
    def type_enum(self) -> Type:
        return self._type

    def serialize(self) -> Dict:
        return {'type': self._type.value}

    #@classmethod
    #def deserialize(cls, d: Dict) -> 'ColumnType':
        #return None

    @classmethod
    def make_type(cls, t: Type) -> 'ColumnType':
        """
        TODO: replace with deserialize(d: Dict)
        """
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
        if t == cls.Type.IMAGE:
            return ImageType()
        if t == cls.Type.DICT:
            return DictType()
        if t == cls.Type.ARRAY:
            return ArrayType()

    def __str__(self) -> str:
        return self.Type.name.lower()

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other) and self._type == other._type

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

    def is_image_type(self) -> bool:
        return self._type == self.Type.IMAGE

    def is_dict_type(self) -> bool:
        return self._type == self.Type.DICT

    def is_array_type(self) -> bool:
        return self._type == self.Type.ARRAY

    def to_sql(self) -> str:
        """
        Return corresponding SQL type.
        """
        assert self._type != self.Type.INVALID
        if self._type == self.Type.STRING:
            return 'VARCHAR'
        if self._type == self.Type.INT:
            return 'INTEGER'
        if self._type == self.Type.FLOAT:
            return 'FLOAT'
        if self._type == self.Type.BOOL:
            return 'BOOLEAN'
        if self._type == self.Type.TIMESTAMP:
            return 'INTEGER'
        if self._type == self.Type.IMAGE:
            # the URL
            return 'VARCHAR'
        if self._type == self.Type.DICT:
            return 'VARCHAR'
        if self._type == self.Type.ARRAY:
            return 'BLOB'
        assert False

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
        if self._type == self.Type.DICT:
            return sql.String
        if self._type == self.Type.ARRAY:
            return sql.VARBINARY
        assert False


class InvalidType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.INVALID)

class StringType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.STRING)


class IntType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.INT)


class FloatType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.FLOAT)


class BoolType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.BOOL)


class TimestampType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.TIMESTAMP)


class ImageType(ColumnType):
    def __init__(
            self, width: Optional[int] = None, height: Optional[int] = None, size: Optional[Tuple[int, int]] = None):
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

    def serialize(self) -> Dict:
        result = super().serialize()
        result.update({'width': self.width, 'height': self.height})
        return result


class DictType(ColumnType):
    def __init__(self):
        super().__init__(self.Type.DICT)


class ArrayType(ColumnType):
    """
    TODO: enum Dtype, dtype in ctor
    """
    def __init__(self, shape: Optional[Tuple[int, ...]] = None):
        super().__init__(self.Type.ARRAY)
        self.shape = shape
