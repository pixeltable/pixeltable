from __future__ import annotations

import enum
import json
import logging
import math
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, cast

import numpy as np
import pandas as pd
import PIL
from pyarrow.parquet import ParquetDataset

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.io.pandas import _df_check_primary_key_values, _df_row_to_pxt_row, df_infer_schema
from pixeltable.utils import parse_local_file_path

from .utils import normalize_schema_names

_logger = logging.getLogger('pixeltable')


if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]

    from pixeltable.globals import RowData, TableDataSource


class TableDataConduitFormat(str, enum.Enum):
    """Supported formats for TableDataConduit"""

    JSON = 'json'
    CSV = 'csv'
    EXCEL = 'excel'
    PARQUET = 'parquet'

    @classmethod
    def is_valid(cls, x: Any) -> bool:
        if isinstance(x, str):
            return x.lower() in [c.value for c in cls]
        return False


@dataclass
class TableDataConduit:
    source: 'TableDataSource'
    source_format: str | None = None
    source_column_map: dict[str, str] | None = None
    if_row_exists: Literal['update', 'ignore', 'error'] = 'error'
    pxt_schema: dict[str, ts.ColumnType] | None = None
    src_schema_overrides: dict[str, ts.ColumnType] | None = None
    src_schema: dict[str, ts.ColumnType] | None = None
    pxt_pk: list[str] | None = None
    src_pk: list[str] | None = None
    valid_rows: RowData | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    reqd_col_names: set[str] = field(default_factory=set)
    computed_col_names: set[str] = field(default_factory=set)

    total_rows: int = 0  # total number of rows emitted via valid_row_batch Iterator

    _K_BATCH_SIZE_BYTES = 100_000_000  # 100 MB

    def check_source_format(self) -> None:
        assert self.source_format is None or TableDataConduitFormat.is_valid(self.source_format)

    def __post_init__(self) -> None:
        """If no extra_fields were provided, initialize to empty dict"""
        if self.extra_fields is None:
            self.extra_fields = {}

    @classmethod
    def is_rowdata_structure(cls, d: TableDataSource) -> bool:
        if not isinstance(d, list) or len(d) == 0:
            return False
        return all(isinstance(row, dict) for row in d)

    def is_direct_query(self) -> bool:
        return isinstance(self.source, pxt.Query) and self.source_column_map is None

    def normalize_pxt_schema_types(self) -> None:
        for name, coltype in self.pxt_schema.items():
            self.pxt_schema[name] = ts.ColumnType.normalize_type(coltype)

    def infer_schema(self) -> dict[str, ts.ColumnType]:
        raise NotImplementedError

    def valid_row_batch(self) -> Iterator[RowData]:
        raise NotImplementedError

    def prepare_for_insert_into_table(self) -> None:
        if self.source is None:
            return
        raise NotImplementedError

    def add_table_info(self, table: pxt.Table) -> None:
        """Add information about the table into which we are inserting data"""
        assert isinstance(table, pxt.Table)
        self.pxt_schema = table._get_schema()
        self.pxt_pk = table._tbl_version.get().primary_key
        for col in table._tbl_version_path.columns():
            if col.is_required_for_insert:
                self.reqd_col_names.add(col.name)
            if col.is_computed:
                self.computed_col_names.add(col.name)
        self.src_pk = []

    # Check source columns : required, computed, unknown
    def check_source_columns_are_insertable(self, columns: Iterable[str]) -> None:
        col_name_set: set[str] = set()
        for col_name in columns:  # FIXME
            mapped_col_name = self.source_column_map.get(col_name, col_name)
            col_name_set.add(mapped_col_name)
            if mapped_col_name not in self.pxt_schema:
                raise excs.Error(f'Unknown column name {mapped_col_name}')
            if mapped_col_name in self.computed_col_names:
                raise excs.Error(f'Value for computed column {mapped_col_name}')
        missing_cols = self.reqd_col_names - col_name_set
        if len(missing_cols) > 0:
            raise excs.Error(f'Missing required column(s) ({", ".join(missing_cols)})')


class QueryTableDataConduit(TableDataConduit):
    pxt_query: pxt.Query = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'QueryTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        if isinstance(tds.source, pxt.Table):
            t.pxt_query = tds.source.select()
        else:
            assert isinstance(tds.source, pxt.Query)
            t.pxt_query = tds.source
        return t

    def infer_schema(self) -> dict[str, ts.ColumnType]:
        self.pxt_schema = self.pxt_query.schema
        self.pxt_pk = self.src_pk
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.pxt_query.schema.keys())


class RowDataTableDataConduit(TableDataConduit):
    raw_rows: RowData | None = None
    disable_mapping: bool = True
    batch_count: int = 0

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'RowDataTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        if isinstance(tds.source, Iterator):
            # Instantiate the iterator to get the raw rows here
            t.raw_rows = list(tds.source)
        elif TYPE_CHECKING:
            t.raw_rows = cast(RowData, tds.source)
        else:
            t.raw_rows = tds.source
        t.batch_count = 0
        return t

    def infer_schema(self) -> dict[str, ts.ColumnType]:
        from .datarows import _infer_schema_from_rows

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.src_schema = _infer_schema_from_rows(self.raw_rows, self.src_schema_overrides, self.src_pk)
            self.pxt_schema, self.pxt_pk, self.source_column_map = normalize_schema_names(
                self.src_schema, self.src_pk, self.src_schema_overrides, self.disable_mapping
            )
            self.normalize_pxt_schema_types()
        else:
            raise NotImplementedError()

        self.prepare_for_insert_into_table()
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        # Converting rows to insertable format is not needed, misnamed columns and types
        # are errors in the incoming row format
        if self.source_column_map is None:
            self.source_column_map = {}
        self.valid_rows = [self._translate_row(row) for row in self.raw_rows]

        self.batch_count = 1 if self.raw_rows is not None else 0

    def _translate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise excs.Error(f'row {row} is not a dictionary')

        col_names: set[str] = set()
        output_row: dict[str, Any] = {}
        for col_name, val in row.items():
            mapped_col_name = self.source_column_map.get(col_name, col_name)
            col_names.add(mapped_col_name)
            if mapped_col_name not in self.pxt_schema:
                raise excs.Error(f'Unknown column name {mapped_col_name} in row {row}')
            if mapped_col_name in self.computed_col_names:
                raise excs.Error(f'Value for computed column {mapped_col_name} in row {row}')
            # basic sanity checks here
            try:
                checked_val = self.pxt_schema[mapped_col_name].create_literal(val)
            except TypeError as e:
                msg = str(e)
                raise excs.Error(f'Error in column {col_name}: {msg[0].lower() + msg[1:]}\nRow: {row}') from e
            output_row[mapped_col_name] = checked_val
        missing_cols = self.reqd_col_names - col_names
        if len(missing_cols) > 0:
            raise excs.Error(f'Missing required column(s) ({", ".join(missing_cols)}) in row {row}')
        return output_row

    def valid_row_batch(self) -> Iterator[RowData]:
        if self.batch_count > 0:
            self.batch_count -= 1
            yield self.valid_rows


class PandasTableDataConduit(TableDataConduit):
    pd_df: pd.DataFrame = None
    batch_count: int = 0

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> PandasTableDataConduit:
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        assert isinstance(tds.source, pd.DataFrame)
        t.pd_df = tds.source
        t.batch_count = 0
        return t

    def infer_schema_part1(self) -> tuple[dict[str, ts.ColumnType], list[str]]:
        """Return inferred schema, inferred primary key, and source column map"""
        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.src_schema = df_infer_schema(self.pd_df, self.src_schema_overrides, self.src_pk)
            inferred_schema, inferred_pk, self.source_column_map = normalize_schema_names(
                self.src_schema, self.src_pk, self.src_schema_overrides, False
            )
            return inferred_schema, inferred_pk
        else:
            raise NotImplementedError()

    def infer_schema(self) -> dict[str, ts.ColumnType]:
        self.pxt_schema, self.pxt_pk = self.infer_schema_part1()
        self.normalize_pxt_schema_types()
        _df_check_primary_key_values(self.pd_df, self.src_pk)
        self.prepare_insert()
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        _, inferred_pk = self.infer_schema_part1()
        assert len(inferred_pk) == 0
        self.prepare_insert()

    def prepare_insert(self) -> None:
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.pd_df.columns)
        # Convert all rows to insertable format
        self.valid_rows = [
            _df_row_to_pxt_row(row, self.src_schema, self.source_column_map) for row in self.pd_df.itertuples()
        ]
        self.batch_count = 1

    def valid_row_batch(self) -> Iterator[RowData]:
        if self.batch_count > 0:
            self.batch_count -= 1
            yield self.valid_rows


class CSVTableDataConduit(TableDataConduit):
    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'PandasTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        assert isinstance(t.source, str)
        t.source = pd.read_csv(t.source, **t.extra_fields)
        return PandasTableDataConduit.from_tds(t)


class ExcelTableDataConduit(TableDataConduit):
    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'PandasTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        assert isinstance(t.source, str)
        t.source = pd.read_excel(t.source, **t.extra_fields)
        return PandasTableDataConduit.from_tds(t)


class JsonTableDataConduit(TableDataConduit):
    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> RowDataTableDataConduit:
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        assert isinstance(t.source, str)

        path = parse_local_file_path(t.source)
        if path is None:  # it's a URL
            # TODO: This should read from S3 as well.
            contents = urllib.request.urlopen(t.source).read()
        else:
            with open(path, 'r', encoding='utf-8') as fp:
                contents = fp.read()
        rows = json.loads(contents, **t.extra_fields)
        t.source = rows
        t2 = RowDataTableDataConduit.from_tds(t)
        t2.disable_mapping = False
        return t2


class HFTableDataConduit(TableDataConduit):
    """
    TODO:
    - use set_format('arrow') and convert ChunkedArrays to PIL.Image.Image instead of going through numpy, which is slow
    """

    column_name_for_split: str | None = None
    categorical_features: dict[str, dict[int, str]]
    dataset_dict: dict[str, datasets.Dataset] = None
    hf_schema_source: dict[str, Any] = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'HFTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        import datasets

        assert isinstance(tds.source, (datasets.Dataset, datasets.DatasetDict))
        if 'column_name_for_split' in t.extra_fields:
            t.column_name_for_split = t.extra_fields['column_name_for_split']

        first_dataset = tds.source if isinstance(tds.source, datasets.Dataset) else next(iter(tds.source.values()))
        # we want to handle these feature types as numpy arrays
        numpy_feature_types = (datasets.Sequence, datasets.Image, datasets.Audio, datasets.Video, datasets.Array2D, datasets.Array3D, datasets.Array4D, datasets.Array5D)
        numpy_columns = [ name for name, feature in first_dataset.features.items() if isinstance(feature, numpy_feature_types) ]
        dict_columns = [ name for name, feature in first_dataset.features.items() if isinstance(feature, dict) ]
        numpy_columns += dict_columns
        if len(numpy_columns) > 0:
            source = tds.source.with_format(type='numpy', columns=numpy_columns, output_all_columns=True)
        else:
            source = tds.source

        if isinstance(source, datasets.Dataset):
            # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
            raw_name = source.split._name
            split_name = raw_name.split('[')[0] if raw_name is not None else None
            t.dataset_dict = {split_name: source}
        else:
            assert isinstance(source, datasets.DatasetDict)
            t.dataset_dict = source
        return t

    @classmethod
    def is_applicable(cls, tds: TableDataConduit) -> bool:
        try:
            import datasets

            return (isinstance(tds.source_format, str) and tds.source_format.lower() == 'huggingface') or isinstance(
                tds.source, (datasets.Dataset, datasets.DatasetDict)
            )
        except ImportError:
            return False

    def infer_schema_part1(self) -> tuple[dict[str, ts.ColumnType], list[str]]:
        from pixeltable.io.hf_datasets import _get_hf_schema, huggingface_schema_to_pxt_schema

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.hf_schema_source = _get_hf_schema(self.source)
            self.src_schema = huggingface_schema_to_pxt_schema(
                self.hf_schema_source, self.src_schema_overrides, self.src_pk
            )

            # Add the split column to the schema if requested
            if self.column_name_for_split is not None:
                if self.column_name_for_split in self.src_schema:
                    raise excs.Error(
                        f'Column name `{self.column_name_for_split}` already exists in dataset schema;'
                        f'provide a different `column_name_for_split`'
                    )
                self.src_schema[self.column_name_for_split] = ts.StringType(nullable=True)

            inferred_schema, inferred_pk, self.source_column_map = normalize_schema_names(
                self.src_schema, self.src_pk, self.src_schema_overrides, True
            )
            return inferred_schema, inferred_pk
        else:
            raise NotImplementedError()

    def infer_schema(self) -> dict[str, Any]:
        self.pxt_schema, self.pxt_pk = self.infer_schema_part1()
        self.normalize_pxt_schema_types()
        self.prepare_insert()
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        _, inferred_pk = self.infer_schema_part1()
        assert len(inferred_pk) == 0
        self.prepare_insert()

    def prepare_insert(self) -> None:
        import datasets

        # extract all class labels from the dataset to translate category ints to strings
        self.categorical_features = {
            feature_name: feature_type.names
            for (feature_name, feature_type) in self.hf_schema_source.items()
            if isinstance(feature_type, datasets.ClassLabel)
        }
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.hf_schema_source.keys())

    def _translate_row(self, row: dict[str, Any], split_name: str, features: datasets.Features) -> dict[str, Any]:
        output_row: dict[str, Any] = {}
        for col_name, val in row.items():
            # translate category ints to strings
            new_val = self.categorical_features[col_name][val] if col_name in self.categorical_features else val
            mapped_col_name = self.source_column_map.get(col_name, col_name)

            new_val = self._translate_val(new_val, features[col_name])
            output_row[mapped_col_name] = new_val

        # add split name to output row
        if self.column_name_for_split is not None:
            output_row[self.column_name_for_split] = split_name
        return output_row

    def _translate_val(self, val: Any, feature: datasets.Feature) -> Any:
        """Convert numpy scalars to Python types and images to PIL.Image.Image"""
        import datasets

        if isinstance(feature, datasets.Value):
            if isinstance(val, (np.generic, np.ndarray)):
                # a scalar, which we want as a standard Python type
                assert np.ndim(val) == 0
                return val.item()
            else:
                # a standard Python object
                return val
        elif isinstance(feature, datasets.Sequence):
            assert np.ndim(val) > 0
            return val
        elif isinstance(feature, datasets.Image):
            return PIL.Image.fromarray(val)
        elif isinstance(feature, dict):
            assert isinstance(val, dict)
            return {k: self._translate_val(v, feature[k]) for k, v in val.items()}
        else:
            return val

    def valid_row_batch(self) -> Iterator[RowData]:
        for split_name, split_dataset in self.dataset_dict.items():
            num_batches = split_dataset.size_in_bytes / self._K_BATCH_SIZE_BYTES
            tuples_per_batch = math.ceil(split_dataset.num_rows / num_batches)
            assert tuples_per_batch > 0

            batch = []
            for row in split_dataset:
                batch.append(self._translate_row(row, split_name, split_dataset.features))
                if len(batch) >= tuples_per_batch:
                    yield batch
                    batch = []
            # last batch
            if len(batch) > 0:
                yield batch


class FastHFImporter(TableDataConduit):
    """
    Fast HuggingFace dataset importer using Arrow format.
    Uses with_format('arrow') to iterate over Arrow record batches directly,
    avoiding per-row format conversions.
    """

    column_name_for_split: str | None = None
    categorical_features: dict[str, dict[int, str]]
    dataset_dict: dict[str, 'datasets.Dataset'] = None
    hf_schema_source: dict[str, Any] = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'FastHFImporter':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        import datasets

        assert isinstance(tds.source, (datasets.Dataset, datasets.DatasetDict))
        if 'column_name_for_split' in t.extra_fields:
            t.column_name_for_split = t.extra_fields['column_name_for_split']

        # Set Arrow format for efficient batch iteration
        source = tds.source.with_format('arrow')

        if isinstance(source, datasets.Dataset):
            raw_name = source.split._name
            split_name = raw_name.split('[')[0] if raw_name is not None else None
            t.dataset_dict = {split_name: source}
        else:
            assert isinstance(source, datasets.DatasetDict)
            t.dataset_dict = dict(source)
        return t

    @classmethod
    def is_applicable(cls, tds: TableDataConduit) -> bool:
        try:
            import datasets

            return (isinstance(tds.source_format, str) and tds.source_format.lower() == 'huggingface') or isinstance(
                tds.source, (datasets.Dataset, datasets.DatasetDict)
            )
        except ImportError:
            return False

    def infer_schema_part1(self) -> tuple[dict[str, ts.ColumnType], list[str]]:
        from pixeltable.io.hf_datasets import _get_hf_schema, huggingface_schema_to_pxt_schema

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            if self.src_pk is None:
                self.src_pk = []
            self.hf_schema_source = _get_hf_schema(self.source)
            self.src_schema = huggingface_schema_to_pxt_schema(
                self.hf_schema_source, self.src_schema_overrides, self.src_pk
            )

            # Add the split column to the schema if requested
            if self.column_name_for_split is not None:
                if self.column_name_for_split in self.src_schema:
                    raise excs.Error(
                        f'Column name `{self.column_name_for_split}` already exists in dataset schema;'
                        f'provide a different `column_name_for_split`'
                    )
                self.src_schema[self.column_name_for_split] = ts.StringType(nullable=True)

            inferred_schema, inferred_pk, self.source_column_map = normalize_schema_names(
                self.src_schema, self.src_pk, self.src_schema_overrides, True
            )
            return inferred_schema, inferred_pk
        else:
            raise NotImplementedError()

    def infer_schema(self) -> dict[str, Any]:
        self.pxt_schema, self.pxt_pk = self.infer_schema_part1()
        self.normalize_pxt_schema_types()
        self.prepare_insert()
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        _, inferred_pk = self.infer_schema_part1()
        assert len(inferred_pk) == 0
        self.prepare_insert()

    def prepare_insert(self) -> None:
        import datasets

        # Extract all class labels from the dataset to translate category ints to strings
        self.categorical_features = {
            feature_name: feature_type.names
            for (feature_name, feature_type) in self.hf_schema_source.items()
            if isinstance(feature_type, datasets.ClassLabel)
        }
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.hf_schema_source.keys())

    def _convert_value(self, val: Any, feature: Any) -> Any:
        """Convert Arrow values to Pixeltable-compatible types."""
        import datasets

        if val is None:
            return None

        if isinstance(feature, datasets.ClassLabel):
            # Convert integer label to string name
            return feature.names[val]

        elif isinstance(feature, datasets.Value):
            # Arrow already converts to Python types
            return val

        elif isinstance(feature, datasets.Sequence):
            # Convert to numpy array
            return np.array(val)

        elif isinstance(feature, datasets.Image):
            # Decode from Arrow binary format
            return self._decode_image(val)

        elif isinstance(feature, datasets.Audio):
            # Audio is stored as dict with 'array', 'path', 'sampling_rate'
            return self._decode_audio(val, feature)

        elif isinstance(feature, dict):
            # Recursively handle nested structures (e.g., audio dict with Sequence inside)
            if isinstance(val, dict):
                return {k: self._convert_value(v, feature[k]) for k, v in val.items()}
            return val

        elif isinstance(feature, list):
            # List of dicts → keep as-is (JsonType)
            return val

        else:
            return val

    def _decode_image(self, val: Any) -> PIL.Image.Image:
        """Decode Arrow image data to PIL.Image."""
        import io as _io

        if isinstance(val, dict) and 'bytes' in val:
            return PIL.Image.open(_io.BytesIO(val['bytes']))
        elif isinstance(val, bytes):
            return PIL.Image.open(_io.BytesIO(val))
        elif isinstance(val, dict) and 'path' in val:
            return PIL.Image.open(val['path'])
        return val

    def _decode_audio(self, val: Any, feature: Any) -> dict:
        """Convert audio dict, ensuring array is numpy."""
        import datasets

        if isinstance(val, dict):
            result = {}
            # Get the inner feature types from the Audio feature
            # Audio features have an implicit structure: {'array': Sequence, 'path': Value, 'sampling_rate': Value}
            for k, v in val.items():
                if k == 'array':
                    # Convert array to numpy
                    result[k] = np.array(v) if not isinstance(v, np.ndarray) else v
                else:
                    result[k] = v
            return result
        return val

    def valid_row_batch(self) -> Iterator['RowData']:
        for split_name, split_dataset in self.dataset_dict.items():
            features = split_dataset.features

            # Calculate batch size based on dataset size
            num_batches = max(1, split_dataset.size_in_bytes / self._K_BATCH_SIZE_BYTES)
            tuples_per_batch = int(math.ceil(split_dataset.num_rows / num_batches))
            assert tuples_per_batch > 0

            # Iterate using Arrow format - slicing returns Arrow tables
            for start_idx in range(0, split_dataset.num_rows, tuples_per_batch):
                end_idx = min(start_idx + tuples_per_batch, split_dataset.num_rows)
                # With arrow format, slicing returns a pa.Table
                arrow_batch = split_dataset[start_idx:end_idx]
                pydict = arrow_batch.to_pydict()  # Fast columnar → dict conversion
                batch_size = end_idx - start_idx
                rows = []

                for i in range(batch_size):
                    row = {}
                    for col_name in pydict:
                        raw_val = pydict[col_name][i]
                        feature = features[col_name]

                        # Apply categorical label conversion if needed
                        if col_name in self.categorical_features:
                            new_val = self.categorical_features[col_name][raw_val]
                        else:
                            new_val = self._convert_value(raw_val, feature)

                        mapped_col_name = self.source_column_map.get(col_name, col_name)
                        row[mapped_col_name] = new_val

                    # Add split name to output row
                    if self.column_name_for_split is not None:
                        row[self.column_name_for_split] = split_name

                    rows.append(row)

                yield rows


class ParquetTableDataConduit(TableDataConduit):
    pq_ds: ParquetDataset | None = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'ParquetTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)

        from pyarrow import parquet

        assert isinstance(tds.source, str)
        input_path = Path(tds.source).expanduser()
        t.pq_ds = parquet.ParquetDataset(str(input_path))
        return t

    def infer_schema_part1(self) -> tuple[dict[str, ts.ColumnType], list[str]]:
        from pixeltable.utils.arrow import to_pxt_schema

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.src_schema = to_pxt_schema(self.pq_ds.schema, self.src_schema_overrides, self.src_pk)
            inferred_schema, inferred_pk, self.source_column_map = normalize_schema_names(
                self.src_schema, self.src_pk, self.src_schema_overrides
            )
            return inferred_schema, inferred_pk
        else:
            raise NotImplementedError()

    def infer_schema(self) -> dict[str, ts.ColumnType]:
        self.pxt_schema, self.pxt_pk = self.infer_schema_part1()
        self.normalize_pxt_schema_types()
        self.prepare_insert()
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        _, inferred_pk = self.infer_schema_part1()
        assert len(inferred_pk) == 0
        self.prepare_insert()

    def prepare_insert(self) -> None:
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.pq_ds.schema.names)
        self.total_rows = 0

    def valid_row_batch(self) -> Iterator[RowData]:
        from pixeltable.utils.arrow import iter_tuples2

        try:
            for fragment in self.pq_ds.fragments:
                for batch in fragment.to_batches():
                    dict_batch = list(iter_tuples2(batch, self.source_column_map, self.pxt_schema))
                    self.total_rows += len(dict_batch)
                    yield dict_batch
        except Exception as e:
            _logger.error(f'Error after inserting {self.total_rows} rows from Parquet file into table: {e}')
            raise e


class UnkTableDataConduit(TableDataConduit):
    """Source type is not known at the time of creation"""

    def specialize(self) -> TableDataConduit:
        if isinstance(self.source, (pxt.Table, pxt.Query)):
            return QueryTableDataConduit.from_tds(self)
        if isinstance(self.source, pd.DataFrame):
            return PandasTableDataConduit.from_tds(self)
        if HFTableDataConduit.is_applicable(self):
            return HFTableDataConduit.from_tds(self)
        if self.source_format == 'csv' or (isinstance(self.source, str) and '.csv' in self.source.lower()):
            return CSVTableDataConduit.from_tds(self)
        if self.source_format == 'excel' or (isinstance(self.source, str) and '.xls' in self.source.lower()):
            return ExcelTableDataConduit.from_tds(self)
        if self.source_format == 'json' or (isinstance(self.source, str) and '.json' in self.source.lower()):
            return JsonTableDataConduit.from_tds(self)
        if self.source_format == 'parquet' or (
            isinstance(self.source, str) and any(s in self.source.lower() for s in ['.parquet', '.pq', '.parq'])
        ):
            return ParquetTableDataConduit.from_tds(self)
        if (
            self.is_rowdata_structure(self.source)
            # An Iterator as a source is assumed to produce rows
            or isinstance(self.source, Iterator)
        ):
            return RowDataTableDataConduit.from_tds(self)

        raise excs.Error(f'Unsupported data source type: {type(self.source)}')
