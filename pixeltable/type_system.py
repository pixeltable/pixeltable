from typing import Any
import enum

import sqlalchemy as sql


# this doesn't work, breaks with
# TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of
# all its bases
# class ColumnType(abc.ABC):
@enum.unique
class ColumnType(enum.Enum):
    STRING = 0
    INT = 1
    FLOAT = 2
    BOOL = 3
    TIMESTAMP = 4
    IMAGE = 5
    DICT = 6
    VECTOR = 7

    def __str__(self) -> str:
        return self.name.lower()

    def to_sa_type(self) -> Any:
        """
        return type Any: there doesn't appear to be a superclass for the sqlalchemy types
        """
        if self == self.STRING:
            return sql.String
        if self == self.INT:
            return sql.Integer
        if self == self.FLOAT:
            return sql.Float
        if self == self.BOOL:
            return sql.Boolean
        if self == self.TIMESTAMP:
            return sql.TIMESTAMP
        if self == self.IMAGE:
            # the URL
            return sql.String
        if self == self.DICT:
            return sql.String
        if self == self.VECTOR:
            return sql.VARBINARY
        assert False

    def to_sql(self) -> str:
        if self == self.STRING:
            return 'VARCHAR'
        if self == self.INT:
            return 'INTEGER'
        if self == self.FLOAT:
            return 'FLOAT'
        if self == self.BOOL:
            return 'BOOLEAN'
        if self == self.TIMESTAMP:
            return 'INTEGER'
        if self == self.IMAGE:
            # the URL
            return 'VARCHAR'
        if self == self.DICT:
            return 'VARCHAR'
        if self == self.VECTOR:
            return 'BLOB'
        assert False
