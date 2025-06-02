from __future__ import annotations

import abc
from typing import Any

import sqlalchemy as sql

from pixeltable import catalog, exprs


class IndexBase(abc.ABC):
    """
    Internal interface used by the catalog and runtime system to interact with indices:
    - types and expressions needed to create and populate the index value column
    - creating/dropping the index
    This doesn't cover querying the index, which is dependent on the index semantics and handled by
    the specific subclass.
    """

    @abc.abstractmethod
    def __init__(self, c: catalog.Column, **kwargs: Any):
        pass

    @abc.abstractmethod
    def index_value_expr(self) -> exprs.Expr:
        """Return expression that computes the value that goes into the index"""
        pass

    @abc.abstractmethod
    def records_value_errors(self) -> bool:
        """True if index_value_expr() can raise errors"""
        pass

    @abc.abstractmethod
    def index_sa_type(self) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        pass

    @abc.abstractmethod
    def create_index(self, index_name: str, index_value_col: catalog.Column) -> None:
        """Create the index on the index value column"""
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
    def from_dict(cls, c: catalog.Column, d: dict) -> IndexBase:
        pass
