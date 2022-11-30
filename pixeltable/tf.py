from typing import Any, Tuple, List, Generator
import tensorflow as tf

import pixeltable.exceptions as exc
from pixeltable.dataframe import DataFrame
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, TimestampType, ImageType, DictType, ArrayType

_SUPPORTED_TYPES = {
    ColumnType.Type.STRING, ColumnType.Type.INT, ColumnType.Type.FLOAT, ColumnType.Type.BOOL,
    ColumnType.Type.IMAGE, ColumnType.Type.ARRAY
}

_TF_DTYPES = {
    ColumnType.DType.INT8: tf.int8,
    ColumnType.DType.INT16: tf.int16,
    ColumnType.DType.INT32: tf.int32,
    ColumnType.DType.INT64: tf.int64,
    ColumnType.DType.UINT8: tf.uint8,
    ColumnType.DType.UINT16: tf.uint16,
    ColumnType.DType.UINT32: tf.uint32,
    ColumnType.DType.UINT64: tf.uint64,
}

def _col_type_to_tf_type(t: ColumnType) -> Tuple[tf.dtypes.DType, Tuple[int, ...]]:
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
        return _TF_DTYPES[t.dtype], t.shape

def _to_tf_data(o: Any, t: ColumnType) -> Any:
    if t.is_scalar_type():
        return o
    if t.is_image_type():
        # convert PIL.Image.Image to tf.Tensor
        return tf.cast(tf.keras.utils.img_to_array(o), tf.uint8)
    if t.is_array_type():
        return tf.convert_to_tensor(o, dtype=_TF_DTYPES[t.dtype])

def to_dataset(df: DataFrame) -> tf.data.Dataset:
    """
    Convert df result set to tf.data.Dataset containing a tuple, each element of which is a result column.
    """
    result_set = df.show(0)
    tf_output_types: List[tf.dtypes.DType] = []
    tf_output_shapes: List[Tuple[int, ...]] = []
    for i, col_type in enumerate(result_set.col_types):
        if col_type.type_enum not in _SUPPORTED_TYPES:
            raise exc.OperationalError(
                f'Type of column {result_set.col_names[i]} cannot be represented in Tensorflow: {col_type.type_enum}')
        dtype, shape = _col_type_to_tf_type(col_type)
        tf_output_types.append(dtype)
        tf_output_shapes.append(shape)

    def f() -> Generator[Tuple[Any, ...], None, None]:
        for row in result_set.rows:
            tf_row = [_to_tf_data(val, result_set.col_types[i]) for i, val in enumerate(row)]
            yield tuple(tf_row)
    return tf.data.Dataset.from_generator(f, output_types=tuple(tf_output_types), output_shapes=tuple(tf_output_shapes))