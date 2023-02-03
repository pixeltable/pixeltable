from pixeltable.type_system import StringType, ImageType, ArrayType, ColumnType
from pixeltable.function import Function


encode_image = Function.make_library_function(
    ArrayType((512,), ColumnType.Type.FLOAT), [ImageType()], 'pixeltable.utils.clip', 'encode_image')
encode_text = Function.make_library_function(
    ArrayType((512,), ColumnType.Type.FLOAT), [StringType()], 'pixeltable.utils.clip', 'encode_text')
