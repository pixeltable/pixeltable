import datasets
from typing import Union, Optional, List, Dict, Any
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
import math
import logging
import pixeltable
import random

_logger = logging.getLogger(__name__)

# use 100MB as the batch size limit for loading a huggingface dataset into pixeltable.
# The primary goal is to bound memory use, regardless of dataset size.
# Second goal is to limit overhead. 100MB is presumed to be reasonable for a lot of storage systems.
_K_BATCH_SIZE_BYTES = 100_000_000

# note, there are many more types. we allow overrides in the schema_override parameter
# to handle cases where the appropriate type is not yet mapped, or to override this mapping.
# https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Value
_hf_to_pxt: Dict[str, ts.ColumnType] = {
    'int32': ts.IntType(nullable=True),  # pixeltable widens to big int
    'int64': ts.IntType(nullable=True),
    'bool': ts.BoolType(nullable=True),
    'float32': ts.FloatType(nullable=True),
    'string': ts.StringType(nullable=True),
    'timestamp[s]': ts.TimestampType(nullable=True),
    'timestamp[ms]': ts.TimestampType(nullable=True),  # HF dataset iterator converts timestamps to datetime.datetime
}

def _to_pixeltable_type(
    feature_type: Union[datasets.ClassLabel, datasets.Value, datasets.Sequence],
) -> Optional[ts.ColumnType]:
    """Convert a huggingface feature type to a pixeltable ColumnType if one is defined."""
    if isinstance(feature_type, datasets.ClassLabel):
        # enum, example: ClassLabel(names=['neg', 'pos'], id=None)
        return ts.StringType(nullable=True)
    elif isinstance(feature_type, datasets.Value):
        # example: Value(dtype='int64', id=None)
        return _hf_to_pxt.get(feature_type.dtype, None)
    elif isinstance(feature_type, datasets.Sequence):
        # example: cohere wiki. Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None)
        dtype = _to_pixeltable_type(feature_type.feature)
        length = feature_type.length if feature_type.length != -1 else None
        return ts.ArrayType(shape=(length,), dtype=dtype)
    else:
        return None

def _get_hf_schema(dataset: Union[datasets.Dataset, datasets.DatasetDict]) -> datasets.Features:
    """Get the schema of a huggingface dataset as a dictionary."""
    first_dataset = dataset if isinstance(dataset, datasets.Dataset) else next(iter(dataset.values()))
    return first_dataset.features

def huggingface_schema_to_pixeltable_schema(
    hf_dataset: Union[datasets.Dataset, datasets.DatasetDict],
) -> Dict[str, Optional[ts.ColumnType]]:
    """Generate a pixeltable schema from a huggingface dataset schema.
    Columns without a known mapping are mapped to None
    """
    hf_schema = _get_hf_schema(hf_dataset)
    pixeltable_schema = {
        column_name: _to_pixeltable_type(feature_type) for column_name, feature_type in hf_schema.items()
    }
    return pixeltable_schema

def import_huggingface_dataset(
    cl: 'pixeltable.Client',
    table_path: str,
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    *,
    column_name_for_split: Optional[str],
    schema_override: Optional[Dict[str, Any]],
    **kwargs,
) -> 'pixeltable.InsertableTable':
    """See `pixeltable.Client.import_huggingface_dataset` for documentation"""
    if table_path in cl.list_tables():
        raise excs.Error(f'table {table_path} already exists')

    if not isinstance(dataset, (datasets.Dataset, datasets.DatasetDict)):
        raise excs.Error(f'`type(dataset)` must be `datasets.Dataset` or `datasets.DatasetDict`. Got {type(dataset)=}')

    if isinstance(dataset, datasets.Dataset):
        # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
        raw_name = dataset.split._name
        split_name = raw_name.split('[')[0] if raw_name is not None else None
        dataset_dict = {split_name: dataset}
    else:
        dataset_dict = dataset

    pixeltable_schema = huggingface_schema_to_pixeltable_schema(dataset)
    if schema_override is not None:
        pixeltable_schema.update(schema_override)

    if column_name_for_split is not None:
        if column_name_for_split in pixeltable_schema:
            raise excs.Error(
                f'Column name `{column_name_for_split}` already exists in dataset schema; provide a different `column_name_for_split`'
            )
        pixeltable_schema[column_name_for_split] = ts.StringType(nullable=True)

    for field, column_type in pixeltable_schema.items():
        if column_type is None:
            raise excs.Error(f'Could not infer pixeltable type for feature `{field}` in huggingface dataset')

    if isinstance(dataset, datasets.Dataset):
        # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
        raw_name = dataset.split._name
        split_name = raw_name.split('[')[0] if raw_name is not None else None
        dataset_dict = {split_name: dataset}
    elif isinstance(dataset, datasets.DatasetDict):
        dataset_dict = dataset
    else:
        raise excs.Error(f'`type(dataset)` must be `datasets.Dataset` or `datasets.DatasetDict`. Got {type(dataset)=}')

    # extract all class labels from the dataset to translate category ints to strings
    hf_schema = _get_hf_schema(dataset)
    categorical_features = {
        feature_name: feature_type.names
        for (feature_name, feature_type) in hf_schema.items()
        if isinstance(feature_type, datasets.ClassLabel)
    }

    try:
        # random tmp name
        tmp_name = f'{table_path}_tmp_{random.randint(0, 100000000)}'
        tab = cl.create_table(tmp_name, pixeltable_schema, **kwargs)

        def _translate_row(row: Dict[str, Any], split_name: str) -> Dict[str, Any]:
            output_row = row.copy()
            # map all class labels to strings
            for field, values in categorical_features.items():
                output_row[field] = values[row[field]]
            # add split name to row
            if column_name_for_split is not None:
                output_row[column_name_for_split] = split_name
            return output_row

        for split_name, split_dataset in dataset_dict.items():
            num_batches = split_dataset.size_in_bytes / _K_BATCH_SIZE_BYTES
            tuples_per_batch = math.ceil(split_dataset.num_rows / num_batches)
            assert tuples_per_batch > 0

            batch = []
            for row in split_dataset:
                batch.append(_translate_row(row, split_name))
                if len(batch) >= tuples_per_batch:
                    tab.insert(batch)
                    batch = []
            # last batch
            if len(batch) > 0:
                tab.insert(batch)

    except Exception as e:
        _logger.error(f'Error while inserting dataset into table: {tmp_name}')
        raise e

    cl.move(tmp_name, table_path)
    return cl.get_table(table_path)
