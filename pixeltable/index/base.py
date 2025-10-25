from __future__ import annotations

import abc
from typing import Any

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
import pixeltable.type_system as ts


class IndexBase(abc.ABC):
    """
    Internal interface used by the catalog and runtime system to interact with indices:
    - types and expressions needed to create and populate the index value column
    - creating/dropping the index
    This doesn't cover querying the index, which is dependent on the index semantics and handled by
    the specific subclass.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    @abc.abstractmethod
    def create_value_expr(self, c: catalog.Column) -> exprs.Expr:
        """
        Validates that the index can be created on column c and returns an expression that computes the index value.
        """
        pass

    @abc.abstractmethod
    def records_value_errors(self) -> bool:
        """True if index_value_expr() can raise errors"""
        pass

    @abc.abstractmethod
    def get_index_sa_type(self, value_col_type: ts.ColumnType) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        pass

    @abc.abstractmethod
    def sa_index(self, index_name: str, index_value_col: catalog.Column) -> sql.Index:
        """Return a sqlalchemy Index instance"""
        pass

    @abc.abstractmethod
    def drop_index(self, index_name: str, index_value_col: catalog.Column) -> None:
        """Drop the index on the index value column"""
        pass

    @classmethod
    @abc.abstractmethod
    def display_name(cls) -> str:
        pass

    @abc.abstractmethod
    def as_dict(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: dict) -> IndexBase:
        pass
