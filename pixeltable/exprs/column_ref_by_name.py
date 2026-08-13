from __future__ import annotations

from typing import Any

import pixeltable.type_system as ts
from pixeltable import exceptions as excs

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder


class ColumnRefByName(Expr):
    """
    A placeholder for a ColumnRef, referenced by column name, that gets substituted with an actual ColumnRef during
    table creation or binding.
    """

    name: str

    # the model declaring this column, when this placeholder stands for one of its attributes; it identifies the
    # shape an expression method resolves against, and is deliberately not part of this expression's identity
    model_cls: Any

    # col_type is what keeps the enclosing expression well-typed until this placeholder is substituted away; it says
    # nothing about which column is being named, and _equals() compares only the name
    _id_includes_col_type = False

    def __init__(self, name: str, col_type: ts.ColumnType | None = None) -> None:
        super().__init__(col_type if col_type is not None else ts.InvalidType())
        self.name = name
        self.model_cls = None
        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'ColumnRefByName({self.name!r})'

    def __str__(self) -> str:
        # Render as a bare column name, identically to the ColumnRef this placeholder stands in for.
        return self.name

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('name', self.name)]

    def _equals(self, other: ColumnRefByName) -> bool:
        return self.name == other.name

    def default_column_name(self) -> str | None:
        return self.name

    @property
    def is_column_ref(self) -> bool:
        return True

    def similarity(self, item: Any = None, **kwargs: Any) -> Expr:
        """The similarity score between this column's values and the given item.

        Resolves the embedding index against the shape the declaring model describes, so no table is needed.
        """
        from .column_ref import ColumnRef

        if self.model_cls is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'similarity(): column {self.name!r} is not a model attribute, so it has no index to resolve against.',
            )
        path = self.model_cls.table_path()
        col_md = path.get_column_md_by_name(self.name)
        assert col_md is not None, self.name
        return ColumnRef(col_md).similarity(item, _tbl_path=path, **kwargs)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise AssertionError('this should never be called')

    def __getattr__(self, item: str) -> Any:
        if item in ('errortype', 'errormsg', 'fileurl', 'localpath'):
            from .column_property_ref import ColumnPropertyRef

            prop = ColumnPropertyRef.Property[item.upper()]
            return ColumnPropertyRef(self, prop)
        return super().__getattr__(item)

    def _as_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'col_type': self.col_type.as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, _components: list[Expr], _tbl_versions: Any = None) -> ColumnRefByName:
        return cls(d['name'], ts.ColumnType.from_dict(d['col_type']))
