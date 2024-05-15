from __future__ import annotations

from typing import Optional, List, Any, Dict, Tuple

import PIL
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.func as func
import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .function_call import FunctionCall
from .row_builder import RowBuilder


# TODO: this doesn't dig up all attrs for actual jpeg images
def _create_pil_attr_info() -> Dict[str, ts.ColumnType]:
    # create random Image to inspect for attrs
    img = PIL.Image.new('RGB', (100, 100))
    # we're only interested in public attrs (including properties)
    result: Dict[str, ts.ColumnType] = {}
    for name in [name for name in dir(img) if not callable(getattr(img, name)) and not name.startswith('_')]:
        if getattr(img, name) is None:
            continue
        if isinstance(getattr(img, name), str):
            result[name] = ts.StringType()
        if isinstance(getattr(img, name), int):
            result[name] = ts.IntType()
        if getattr(img, name) is dict:
            result[name] = ts.JsonType()
    return result


class ImageMemberAccess(Expr):
    """
    Access of either an attribute or function member of PIL.Image.Image.
    Ex.: tbl.img_col_ref.rotate(90), tbl.img_col_ref.width
    TODO: remove this class and use FunctionCall instead (attributes to be replaced by functions)
    """
    attr_info = _create_pil_attr_info()

    def __init__(self, member_name: str, caller: Expr):
        if member_name in self.attr_info:
            super().__init__(self.attr_info[member_name])
        else:
            candidates = func.FunctionRegistry.get().get_type_methods(member_name, ts.ColumnType.Type.IMAGE)
            if len(candidates) == 0:
                raise excs.Error(f'Unknown Image member: {member_name}')
            if len(candidates) > 1:
                raise excs.Error(f'Ambiguous Image method: {member_name}')
            self.img_method = candidates[0]
            super().__init__(ts.InvalidType())  # requires FunctionCall to return value
        self.member_name = member_name
        self.components = [caller]
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return self.member_name.replace('.', '_')

    @property
    def _caller(self) -> Expr:
        return self.components[0]

    def __str__(self) -> str:
        return f'{self._caller}.{self.member_name}'

    def _as_dict(self) -> Dict:
        return {'member_name': self.member_name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'member_name' in d
        assert len(components) == 1
        return cls(d['member_name'], components[0])

    def __call__(self, *args, **kwargs) -> FunctionCall:
        result = self.img_method(*[self._caller, *args], **kwargs)
        result.is_method_call = True
        return result

    def _equals(self, other: ImageMemberAccess) -> bool:
        return self.member_name == other.member_name

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('member_name', self.member_name)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        caller_val = data_row[self._caller.slot_idx]
        try:
            data_row[self.slot_idx] = getattr(caller_val, self.member_name)
        except AttributeError:
            data_row[self.slot_idx] = None
