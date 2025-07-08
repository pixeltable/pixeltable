from __future__ import annotations

import typing
from typing import Any, Optional, Union

import pixeltable as pxt
import pixeltable.type_system as ts

if typing.TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]


# note, there are many more types. we allow overrides in the schema_overrides parameter
# to handle cases where the appropriate type is not yet mapped, or to override this mapping.
# https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Value
_hf_to_pxt: dict[str, ts.ColumnType] = {
    'bool': ts.BoolType(nullable=True),
    'int8': ts.IntType(nullable=True),
    'int16': ts.IntType(nullable=True),
    'int32': ts.IntType(nullable=True),
    'int64': ts.IntType(nullable=True),
    'uint8': ts.IntType(nullable=True),
    'uint16': ts.IntType(nullable=True),
    'uint32': ts.IntType(nullable=True),
    'uint64': ts.IntType(nullable=True),
    'float16': ts.FloatType(nullable=True),
    'float32': ts.FloatType(nullable=True),
    'float64': ts.FloatType(nullable=True),
    'string': ts.StringType(nullable=True),
    'large_string': ts.StringType(nullable=True),
    'timestamp[s]': ts.TimestampType(nullable=True),
    'timestamp[ms]': ts.TimestampType(nullable=True),  # HF dataset iterator converts timestamps to datetime.datetime
    'timestamp[us]': ts.TimestampType(nullable=True),
    'date32': ts.DateType(nullable=True),
    'date64': ts.DateType(nullable=True),
}


def _to_pixeltable_type(feature_type: Any, nullable: bool) -> Optional[ts.ColumnType]:
    """Convert a huggingface feature type to a pixeltable ColumnType if one is defined."""
    import datasets

    if isinstance(feature_type, datasets.ClassLabel):
        # enum, example: ClassLabel(names=['neg', 'pos'], id=None)
        return ts.StringType(nullable=nullable)
    elif isinstance(feature_type, datasets.Value):
        # example: Value(dtype='int64', id=None)
        pt = _hf_to_pxt.get(feature_type.dtype, None)
        return pt.copy(nullable=nullable) if pt is not None else None
    elif isinstance(feature_type, datasets.Sequence):
        # example: cohere wiki. Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None)
        dtype = _to_pixeltable_type(feature_type.feature, nullable)
        if dtype is None:
            return None
        if dtype.is_int_type() or dtype.is_float_type() or dtype.is_bool_type() or dtype.is_string_type():
            length = feature_type.length if feature_type.length != -1 else None
            return ts.ArrayType(shape=(length,), dtype=dtype, nullable=nullable)
        else:
            # Sequence of dicts must be cast as Json
            return ts.JsonType(nullable=nullable)
    elif isinstance(feature_type, datasets.Image):
        return ts.ImageType(nullable=nullable)
    elif isinstance(feature_type, dict):
        return ts.JsonType(nullable=nullable)
    else:
        return None


def _get_hf_schema(dataset: Union[datasets.Dataset, datasets.DatasetDict]) -> datasets.Features:
    """Get the schema of a huggingface dataset as a dictionary."""
    import datasets

    first_dataset = dataset if isinstance(dataset, datasets.Dataset) else next(iter(dataset.values()))
    return first_dataset.features


def huggingface_schema_to_pxt_schema(
    hf_schema: datasets.Features, schema_overrides: dict[str, Any], primary_key: list[str]
) -> dict[str, Optional[ts.ColumnType]]:
    """Generate a pixeltable schema from a huggingface dataset schema.
    Columns without a known mapping are mapped to None
    """
    pixeltable_schema = {
        column_name: _to_pixeltable_type(feature_type, column_name not in primary_key)
        if column_name not in schema_overrides
        else schema_overrides[column_name]
        for column_name, feature_type in hf_schema.items()
    }
    return pixeltable_schema


def import_huggingface_dataset(
    table_path: str,
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    **kwargs: Any,
) -> pxt.Table:
    """Create a new base table from a Huggingface dataset, or dataset dict with multiple splits.
        Requires `datasets` library to be installed.

    Args:
        table_path: Path to the table.
        dataset: Huggingface [`datasets.Dataset`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset)
            or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict)
            to insert into the table.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
            name `name` will be given type `type`, instead of being inferred from the `Dataset` or `DatasetDict`.
            The keys in `schema_overrides` should be the column names of the `Dataset` or `DatasetDict` (whether or not
            they are valid Pixeltable identifiers).
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        kwargs: Additional arguments to pass to `create_table`.
            An argument of `column_name_for_split` must be provided if the source is a DatasetDict.
            This column name will contain the split information. If None, no split information will be stored.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        table_path, source=dataset, schema_overrides=schema_overrides, primary_key=primary_key, extra_args=kwargs
    )
