from pixeltable.type_system import StringType, ImageType, ArrayType, Function, ColumnType


encode_image = Function(
    ArrayType((512,), ColumnType.Type.FLOAT), [ImageType()],
    module_name='pixeltable.utils.clip', symbol='encode_image')
encode_text = Function(
    ArrayType((512,), ColumnType.Type.FLOAT), [StringType()],
    module_name = 'pixeltable.utils.clip', symbol = 'encode_text')
