from pixeltable.type_system import StringType, ImageType, ArrayType, FloatType
from pixeltable.function import Function, FunctionRegistry


encode_image = Function.make_library_function(
    ArrayType((512,), FloatType()), [ImageType()], 'pixeltable.utils.clip', 'encode_image')
FunctionRegistry.get().register_function(__name__, 'encode_image', encode_image)
encode_text = Function.make_library_function(
    ArrayType((512,), FloatType()), [StringType()], 'pixeltable.utils.clip', 'encode_text')
FunctionRegistry.get().register_function(__name__, 'encode_text', encode_text)
