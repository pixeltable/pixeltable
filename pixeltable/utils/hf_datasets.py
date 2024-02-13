import datasets
from typing import Union, Optional, List, Dict, Any
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
import math
import logging

_logger = logging.getLogger(__name__)

# https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Value
_hf_to_pt : Dict[str, ts.ColumnType] = {
    'uint32' : ts.IntType(nullable=True),
    'uint64' : ts.IntType(nullable=True),
    'int32' : ts.IntType(nullable=True),
    'int64' : ts.IntType(nullable=True),
    'bool' : ts.BoolType(nullable=True),
    'float32' : ts.FloatType(nullable=True),
    'string' : ts.StringType(nullable=True),
    'timestamp[s]' : ts.TimestampType(nullable=True),
    'timestamp[ms]' : ts.TimestampType(nullable=True),
}

def create_table_from_huggingface_dataset(self : 'pixeltable.Client',
                                            path_str: str,
                                            dataset : Union['datasets.Dataset', 'datasets.DatasetDict'],
                                            split_column_name : Optional[str],
                                            explain_only : bool,
                                            primary_key: Union[str, List[str]],
                                            num_retained_versions: int) -> Optional['catalog.InsertableTable']:
    """ See pixeltable.Client.create_table_from_huggingface_dataset for documentation
    """

    dataset_dict : Dict[str, datasets.Dataset] = {}
    if isinstance(dataset, datasets.Dataset):
        # get split string while removing any slice notation
        if dataset.split._name is not None:
            # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
            # in those cases, we only only want to keep the "train" part
            split_name = dataset.split._name.split('[')[0]
        else:
            split_name = None

        if split_name is None and split_column_name is not None:
            print('no split name found in dataset, ignoring')
            split_column_name = None

        dataset_dict = {split_name:dataset}
    elif isinstance(dataset, datasets.DatasetDict):
        for column_name in dataset:
            dataset_dict[column_name] = dataset[column_name]
    else:
        raise excs.Error(f'type(dataset) must be datasets.Dataset or datasets.DatasetDict. Got {type(dataset)}')

    # get the schema from the first split
    dataset_features = next(iter(dataset_dict.values())).features

    pixeltable_schema : Dict[str, ts.ColumnType] = {}
    categorical_features : Dict[str, datasets.ClassLabel] = {}
    for column_name,feature_type in dataset_features.items():
        if isinstance(feature_type, datasets.ClassLabel):
            # enum, example: ClassLabel(names=['neg', 'pos'], id=None)"
            pixeltable_schema[column_name] = ts.StringType(nullable=True)
            categorical_features[column_name] = feature_type.names
        elif isinstance(feature_type, datasets.Value):
            # example: Value(dtype='int64', id=None)
            if feature_type.dtype in _hf_to_pt:
                dest_type = _hf_to_pt[feature_type.dtype]
                pixeltable_schema[column_name] = dest_type
            else:
                raise excs.Error(f'unsupported value type {feature_type} for column {column_name}')
        else:
            raise excs.Error(f'unsupported feature type {feature_type} for column {column_name}')


    if split_column_name is not None:
        if split_column_name in pixeltable_schema:
            raise excs.Error(f'split column name {split_column_name} already exists in dataset schema, use a different name')
        pixeltable_schema[split_column_name] = ts.StringType(nullable=False)

    print('mapping huggingface features to pixeltable schema:', pixeltable_schema)
    if explain_only:
        return None

    tab = self.create_table(path_str, pixeltable_schema, primary_key, num_retained_versions)

    try:
        target_bytes_per_batch = 20_000_000 # 20MB
        for split_name, split_dataset in dataset_dict.items():
            num_batches = split_dataset.size_in_bytes / target_bytes_per_batch
            tuples_per_batch = math.ceil(split_dataset.num_rows / num_batches)

            batch : List[Dict[str,Any]] = []
            for row in split_dataset:
                row[split_column_name] = split_name
                # map all class labels to strings
                for column_name, categorical_map in categorical_features.items():
                    row[column_name] = categorical_map[row[column_name]]

                batch.append(row)
                if len(batch) > tuples_per_batch:
                    tab.insert(batch)
                    batch = []

            # final batch
            tab.insert(batch)
    except Exception as e:
        _logger.error(f'Error while inserting dataset into table: {e}')
        self.drop_table(path_str, force=True, ignore_errors=True)
        raise e

    return tab
