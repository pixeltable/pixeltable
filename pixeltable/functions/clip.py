from pixeltable.type_system import StringType, ImageType, ArrayType, ColumnType
from pixeltable.function import Function


encode_image = Function(
    ArrayType((512,), ColumnType.Type.FLOAT), [ImageType()],
    module_name='pixeltable.utils.clip', eval_symbol='encode_image')
encode_text = Function(
    ArrayType((512,), ColumnType.Type.FLOAT), [StringType()],
    module_name = 'pixeltable.utils.clip', eval_symbol = 'encode_text')
