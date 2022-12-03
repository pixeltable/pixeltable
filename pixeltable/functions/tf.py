from typing import Callable, Any, Optional
import tensorflow as tf


from pixeltable.type_system import ColumnType, ArrayType, Function
import pixeltable.utils.tf as tf_utils

def TFModelFunction(model: tf.keras.Model, param_type: ColumnType, preprocess: Optional[Callable] = None) -> Function:
    def predict(arg: Any) -> Any:
        tensor = tf_utils.to_tensor(arg, param_type)
        tensor = tf.expand_dims(tensor, axis=0)
        if preprocess is not None:
            tensor = preprocess(tensor)
        output = model.predict(tensor)
        return output[0]
    return Function(predict, ArrayType, [param_type])