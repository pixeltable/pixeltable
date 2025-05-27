import sqlalchemy as sql

from pixeltable.func.udf import udf

@udf
def sample_key(seed: int, *key_fields: int) -> str:
    """
    Create a sample key from the given seed and key fields.

    Args:
        seed: The seed value.
        rowids: The rowids to include in the sample key.

    Returns:
        A string key for each row
    """
    raise NotImplementedError('SampleKey creation is not implemented in python.')


@sample_key.to_sql
def _(seed: sql.ColumnElement, *key_fields: sql.ColumnElement) -> sql.ColumnElement:
    from pixeltable.exec.sql_node import SqlSampleNode

    return SqlSampleNode.key_sql_expr(seed, key_fields)
