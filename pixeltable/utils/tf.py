from typing import Any, Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from  pixeltable.type_system import ColumnType, JsonType

TF_DTYPES = {
    # ColumnType.DType.INT8: tf.int8,
    # ColumnType.DType.INT16: tf.int16,
    # ColumnType.DType.INT32: tf.int32,
    # ColumnType.DType.INT64: tf.int64,
    # ColumnType.DType.UINT8: tf.uint8,
    # ColumnType.DType.UINT16: tf.uint16,
    # ColumnType.DType.UINT32: tf.uint32,
    # ColumnType.DType.UINT64: tf.uint64,
    ColumnType.Type.INT: tf.int64,
    ColumnType.Type.FLOAT: tf.float32
}

def to_tensor(v: Any, t: ColumnType) -> tf.Tensor:
    if t.is_scalar_type():
        return tf.convert_to_tensor(v)
    if t.is_image_type():
        # convert PIL.Image.Image to tf.Tensor
        return tf.cast(tf.keras.utils.img_to_array(v), tf.uint8)
    if t.is_array_type():
        return tf.convert_to_tensor(v, dtype=TF_DTYPES[t.dtype])
    if t.is_json_type():
        assert isinstance(v, dict)
        assert isinstance(t, JsonType)
        assert t.type_spec is not None
        return {key: to_tensor(val, t.type_spec[key]) for key, val in v.items()}
