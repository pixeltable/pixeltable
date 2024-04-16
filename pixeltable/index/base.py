from __future__ import annotations

import abc
from typing import Any

import sqlalchemy as sql

import pixeltable.catalog as catalog


class IndexBase(abc.ABC):
    """
    Internal interface used by the catalog and runtime system to interact with indices:
    - types and expressions needed to create and populate the index value column
    - creating/dropping the index
    - TODO: translating queries into sqlalchemy predicates
    """
    @abc.abstractmethod
    def __init__(self, c: catalog.Column, **kwargs: Any):
        pass

    @abc.abstractmethod
    def index_value_expr(self) -> 'pixeltable.exprs.Expr':
        """Return expression that computes the value that goes into the index"""
        pass

    @abc.abstractmethod
    def index_sa_type(self) -> sql.sqltypes.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        pass

    @abc.abstractmethod
    def create_index(self, index_name: str, index_value_col: catalog.Column, conn: sql.engine.Connection) -> None:
        """Create the index on the index value column"""
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
