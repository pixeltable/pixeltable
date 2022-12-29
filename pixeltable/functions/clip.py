from pixeltable.utils import clip
from pixeltable.type_system import StringType, ImageType, ArrayType, Function, ColumnType


encode_image = Function(ArrayType((512,), ColumnType.Type.FLOAT), [ImageType()], eval_fn=clip.encode_image)
encode_text = Function(ArrayType((512,), ColumnType.Type.FLOAT), [StringType()], eval_fn=clip.encode_text)
