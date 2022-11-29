import tensorflow as tf

from pixeltable.dataframe import DataFrame

def to_dataset(df: DataFrame) -> tf.data.Dataset:
    """
    Convert df result set to tf.data.Dataset containing a tuple, each element of which is a result column.
    """
    slices = []