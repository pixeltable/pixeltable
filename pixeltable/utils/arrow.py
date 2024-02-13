import pyarrow as pa
from typing import Any, Dict, Iterator, Iterable, Optional, Union, List
import pixeltable.type_system as ts
import pathlib
import logging

_logger = logging.getLogger(__name__)

_pa_to_pt : Dict[pa.DataType, ts.ColumnType] = {
        pa.string() : ts.StringType(nullable=True),
        pa.timestamp('us') : ts.TimestampType(nullable=True),
        pa.bool_() : ts.BoolType(nullable=True),
        pa.uint8() : ts.IntType(nullable=True),
        pa.int8() : ts.IntType(nullable=True),
        pa.uint32() : ts.IntType(nullable=True),
        pa.uint64() : ts.IntType(nullable=True),
        pa.int32() : ts.IntType(nullable=True),
        pa.int64() : ts.IntType(nullable=True),
        pa.float32() : ts.FloatType(nullable=True),
}

_pt_to_pa : Dict[type, pa.DataType] = {
        ts.StringType : pa.string(),
        ts.TimestampType : pa.timestamp('us'), # postgres timestamp is microseconds
        ts.BoolType : pa.bool_(),
        ts.IntType : pa.int64(),
        ts.FloatType : pa.float32(),
        ts.JsonType : pa.string(), # TODO(orm) pa.struct() is possible
        ts.ImageType : pa.binary(), # inline image
        ts.AudioType : pa.string(), # path
        ts.VideoType : pa.string(), # path
        ts.DocumentType : pa.string(), # path
}

def get_pixeltable_type(arrow_type : pa.DataType) -> ts.ColumnType:
    if arrow_type in _pa_to_pt:
        return _pa_to_pt[arrow_type]
    elif isinstance(arrow_type, pa.FixedShapeTensorType):
        assert arrow_type.value_type in _pa_to_pt, f'unknown value type {arrow_type.value_type} for array'
        return ts.ArrayType(shape=arrow_type.shape, dtype=get_pixeltable_type(arrow_type.value_type))
    else:
        assert False, f'default conversion for {arrow_type} not implemented'

def get_arrow_type(pixeltable_type : ts.ColumnType) -> pa.DataType:
    if pixeltable_type.__class__ in _pt_to_pa:
        return _pt_to_pa[pixeltable_type.__class__]
    elif isinstance(pixeltable_type, ts.ArrayType):
        return pa.fixed_shape_tensor(pa.from_numpy_dtype(pixeltable_type.numpy_dtype()), pixeltable_type.shape)
    else:
        assert False, f'default conversion for {pixeltable_type} not implemented'

def get_pixeltable_schema(arrow_schema : pa.Schema) -> Dict[str, ts.ColumnType]:
    return {field.name: get_pixeltable_type(field.type) for field in arrow_schema}

def get_arrow_schema(pixeltable_schema : Dict[str, Any]) -> pa.Schema:
    return pa.schema({name: get_arrow_type(dtype) for name, dtype in pixeltable_schema.items()})

def to_pydict(batch : pa.RecordBatch) -> Dict[str, Iterable[Any]]:
    """ Convert a RecordBatch to a dictionary of lists, unlike pa.lib.RecordBatch.to_pydict,
        this function will not convert numpy arrays to lists, and will preserve the original numpy dtype.
    """
    out = {}
    for k, name in enumerate(batch.schema.names):
        col = batch.column(k)
        if isinstance(col.type, pa.FixedShapeTensorType):
            # treat array columns as numpy arrays to easily preserve numpy type
            out[name] = col.to_numpy(zero_copy_only=False)
        else:
            # for the rest, use pydict to preserve python types
            out[name] = col.to_pylist()

    return out

def iter_tuples(batch : pa.RecordBatch) -> Iterator[Dict[str,Any]]:
    """ Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup """
    pydict = to_pydict(batch)
    n = len(pydict[list(pydict.keys())[0]])
    for i in range(n):
        yield {col_name: values[i] for col_name, values in pydict.items()}


def create_table_from_parquet(cl : 'pixeltable.Client',
                              path_str: str,
                              parquet_path: str,
                              explain_only : bool = False,
                              primary_key: Union[str, List[str]] = [],
                              num_retained_versions: int = 10) -> Optional['catalog.InsertableTable']:
    """ See pixeltable.Client.create_table_from_parquet for documentation
    """

    input_path = pathlib.Path(parquet_path).expanduser()
    parquet_dataset = pa.parquet.ParquetDataset(input_path)
    arrow_schema = parquet_dataset.schema
    base_schema = get_pixeltable_schema(arrow_schema)

    print('mapping parquet columns to pixeltable schema:', base_schema)
    if explain_only:
        return None

    tab = cl.create_table(path_str, base_schema, primary_key, num_retained_versions)

    try:
        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches():
                dict_batch = [d for d in iter_tuples(batch)]
                tab.insert(dict_batch)
    except Exception as e:
        _logger.error(f'Error while inserting Parquet file into table: {e}')
        cl.drop_table(path_str, force=True, ignore_errors=True)
        raise e

    return tab