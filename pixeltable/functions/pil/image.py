import PIL.Image

from pixeltable.catalog import Function
from pixeltable.type_system import ColumnType


alpha_composite = Function(
    'alpha_composite', PIL.Image.alpha_composite, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE])
blend = Function('blend', PIL.Image.blend, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE, ColumnType.FLOAT])
composite = Function(
    'composite', PIL.Image.composite, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE, ColumnType.IMAGE])
