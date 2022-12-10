from pixeltable.utils import clip
from pixeltable.type_system import StringType, ImageType, ArrayType, Function, ColumnType


encode_image = Function(clip.encode_image, ArrayType((512,), ColumnType.Type.FLOAT), [ImageType()])
encode_text = Function(clip.encode_text, ArrayType((512,), ColumnType.Type.FLOAT), [StringType()])
