from __future__ import annotations

from typing import Any, NoReturn

import sqlalchemy as sql

from pixeltable import type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class Variable(Expr):
    """An expr parameter, needed for ExprTemplateFunctions

    A Variable has a name and type and needs to have been replaced by an actual expression before evaluation.
    """

    _bound_val: Any

    def __init__(self, name: str, col_type: ts.ColumnType):
        super().__init__(col_type)
        self.name = name
        self.id = self._create_id()

    @property
    def _bind_name(self) -> str:
        # 'pxt:' prefix makes this a non-identifier so it can't collide with SQLAlchemy's
        # auto-generated bind names (param_1, c2_1, etc.) when the same bound_args dict is passed
        # to execute()
        return f'pxt:{self.name}'

    def prepare(self, args: dict[str, Any], bound_args: dict[str, Any]) -> None:
        super().prepare(args, bound_args)
        self._bound_val = args[self.name]
        bound_args[self._bind_name] = self._bound_val

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('name', self.name)]

    def default_column_name(self) -> NoReturn:
        raise NotImplementedError()

    def _equals(self, other: Variable) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Variable('{self.name}')"

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        # types that don't round-trip cleanly through SQLAlchemy parameter binding (eg, arrays bound to
        # LargeBinary expect serialized bytes, not the raw Python value, and media types hold PIL/file
        # refs) are evaluated in Python instead
        if self.col_type.supports_file_offloading() or self.col_type.is_media_type():
            return None
        return sql.bindparam(self._bind_name, type_=self.col_type.to_sa_type())

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = self._bound_val

    def _as_dict(self) -> dict:
        return {'name': self.name, 'type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, _: list[Expr], tbl_versions: Any = None) -> Variable:
        return cls(d['name'], ts.ColumnType.from_dict(d['type']))
