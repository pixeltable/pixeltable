from typing import Any, Tuple
import tensorflow as tf

from  pixeltable.type_system import ColumnType

TF_DTYPES = {
    ColumnType.DType.INT8: tf.int8,
    ColumnType.DType.INT16: tf.int16,
    ColumnType.DType.INT32: tf.int32,
    ColumnType.DType.INT64: tf.int64,
    ColumnType.DType.UINT8: tf.uint8,
    ColumnType.DType.UINT16: tf.uint16,
    ColumnType.DType.UINT32: tf.uint32,
    ColumnType.DType.UINT64: tf.uint64,
}

def col_type_to_tf_type(t: ColumnType) -> Tuple[tf.dtypes.DType, Tuple[int, ...]]:
    """
    Returns (dtype, shape)
    """
    assert not t.is_invalid_type()
    assert not t.is_dict_type()
    assert not t.is_timestamp_type()
    if t.is_string_type():
        return tf.string, ()
    if t.is_int_type():
        return tf.int64, ()
    if t.is_float_type():
        return tf.float64, ()
    if t.is_bool_type():
        return tf.bool, ()
    if t.is_image_type():
        # TODO: deal with formats other than rgb
        return tf.uint8, (t.height, t.width, t.num_channels)
    if t.is_array_type():
        return TF_DTYPES[t.dtype], t.shape

def to_tensor(v: Any, t: ColumnType) -> tf.Tensor:
    if t.is_scalar_type():
        return tf.convert_to_tensor(v)
    if t.is_image_type():
        # convert PIL.Image.Image to tf.Tensor
        return tf.cast(tf.keras.utils.img_to_array(v), tf.uint8)
    if t.is_array_type():
        return tf.convert_to_tensor(v, dtype=TF_DTYPES[t.dtype])
