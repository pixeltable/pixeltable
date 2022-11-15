import PIL.Image

from pixeltable.functions import Function
from pixeltable.type_system import ColumnType


alpha_composite = Function(PIL.Image.alpha_composite, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE])
blend = Function(PIL.Image.blend, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE, ColumnType.FLOAT])
composite = Function(PIL.Image.composite, ColumnType.IMAGE, [ColumnType.IMAGE, ColumnType.IMAGE, ColumnType.IMAGE])
