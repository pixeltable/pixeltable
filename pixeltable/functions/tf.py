from typing import Callable, Any, Optional, Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


from pixeltable.type_system import ColumnType, ArrayType
import pixeltable.utils.tf as tf_utils
from pixeltable.function import Function

def TFModelFunction(
        model: tf.keras.Model, param_type: ColumnType, output_shape: Optional[Tuple[int, ...]] = None,
        preprocess: Optional[Callable] = None) -> Function:
    def predict(arg: Any) -> Any:
        tensor = tf_utils.to_tensor(arg, param_type)
        tensor = tf.expand_dims(tensor, axis=0)
        if preprocess is not None:
            tensor = preprocess(tensor)
        output = model.predict(tensor)
        return output[0]
    return Function(ArrayType(output_shape, dtype=ColumnType.DType.FLOAT32), [param_type], eval_fn=predict)