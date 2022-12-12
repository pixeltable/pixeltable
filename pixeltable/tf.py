from typing import Any, Tuple, List, Generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from pixeltable.dataframe import DataFrame
import pixeltable.utils.tf as tf_utils


def to_dataset(df: DataFrame) -> tf.data.Dataset:
    """
    Convert df result set to tf.data.Dataset.
    """
    result_set = df.show(0)
    # construct output signature
    if len(result_set.col_types) == 1 and result_set.col_types[0].is_json_type():
        # we're only getting one column back and it's a dict: make the dataset type a dict
        type_spec = result_set.col_types[0].to_tf()
    else:
        tf_types = [t.to_tf() for t in result_set.col_types]
        type_spec = tuple(tf_types)

    def f() -> Generator[Tuple[Any, ...], None, None]:
        if isinstance(type_spec, dict):
            json_type = result_set.col_types[0]
            for row in result_set.rows:
                res = {k: tf_utils.to_tensor(v, json_type.type_spec[k]) for k, v in row[0].items()}
                yield res
        else:
            for row in result_set.rows:
                tf_row = [tf_utils.to_tensor(val, result_set.col_types[i]) for i, val in enumerate(row)]
                yield tuple(tf_row)
    return tf.data.Dataset.from_generator(f, output_signature=type_spec)