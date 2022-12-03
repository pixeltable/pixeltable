from typing import Any, Tuple, List, Generator
import tensorflow as tf

import pixeltable.exceptions as exc
from pixeltable.dataframe import DataFrame
from pixeltable.type_system import ColumnType
import pixeltable.utils.tf as tf_utils

_SUPPORTED_TYPES = {
    ColumnType.Type.STRING, ColumnType.Type.INT, ColumnType.Type.FLOAT, ColumnType.Type.BOOL,
    ColumnType.Type.IMAGE, ColumnType.Type.ARRAY
}

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
        dtype, shape = tf_utils.col_type_to_tf_type(col_type)
        tf_output_types.append(dtype)
        tf_output_shapes.append(shape)

    def f() -> Generator[Tuple[Any, ...], None, None]:
        for row in result_set.rows:
            tf_row = [tf_utils.to_tensor(val, result_set.col_types[i]) for i, val in enumerate(row)]
            yield tuple(tf_row)
    return tf.data.Dataset.from_generator(f, output_types=tuple(tf_output_types), output_shapes=tuple(tf_output_shapes))