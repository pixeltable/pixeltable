from __future__ import annotations

import enum
import json
import logging
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, cast

import numpy as np
import pandas as pd
import PIL
import PIL.Image
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
    import pyarrow as pa

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


# class HFTableDataConduit2(TableDataConduit):
#     """
#     TODO:
#     - use set_format('arrow') and convert ChunkedArrays to PIL.Image.Image instead of going through numpy, whichisslow
#     """
#
#     column_name_for_split: str | None = None
#     categorical_features: dict[str, dict[int, str]]
#     dataset_dict: dict[str, datasets.Dataset] = None
#     hf_schema_source: dict[str, Any] = None
#
#     @classmethod
#     def from_tds(cls, tds: TableDataConduit) -> 'HFTableDataConduit':
#         tds_fields = {f.name for f in fields(tds)}
#         kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
#         t = cls(**kwargs)
#         import datasets
#
#         assert isinstance(tds.source, (datasets.Dataset, datasets.DatasetDict))
#         if 'column_name_for_split' in t.extra_fields:
#             t.column_name_for_split = t.extra_fields['column_name_for_split']
#
#         first_dataset = tds.source if isinstance(tds.source, datasets.Dataset) else next(iter(tds.source.values()))
#         # we want to handle these feature types as numpy arrays
#         numpy_feature_types = (
#             datasets.Sequence,
#             datasets.Image,
#             datasets.Audio,
#             datasets.Video,
#             datasets.Array2D,
#             datasets.Array3D,
#             datasets.Array4D,
#             datasets.Array5D,
#         )
#         numpy_columns = [
#             name for name, feature in first_dataset.features.items() if isinstance(feature, numpy_feature_types)
#         ]
#         dict_columns = [name for name, feature in first_dataset.features.items() if isinstance(feature, dict)]
#         numpy_columns += dict_columns
#         if len(numpy_columns) > 0:
#             source = tds.source.with_format(type='numpy', columns=numpy_columns, output_all_columns=True)
#         else:
#             source = tds.source
#
#         if isinstance(source, datasets.Dataset):
#             # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
#             raw_name = source.split._name
#             split_name = raw_name.split('[')[0] if raw_name is not None else None
#             t.dataset_dict = {split_name: source}
#         else:
#             assert isinstance(source, datasets.DatasetDict)
#             t.dataset_dict = source
#         return t
#
#     @classmethod
#     def is_applicable(cls, tds: TableDataConduit) -> bool:
#         try:
#             import datasets
#
#             return (isinstance(tds.source_format, str) and tds.source_format.lower() == 'huggingface') or isinstance(
#                 tds.source, (datasets.Dataset, datasets.DatasetDict)
#             )
#         except ImportError:
#             return False
#
#     def infer_schema_part1(self) -> tuple[dict[str, ts.ColumnType], list[str]]:
#         from pixeltable.io.hf_datasets import _get_hf_schema, huggingface_schema_to_pxt_schema
#
#         if self.source_column_map is None:
#             if self.src_schema_overrides is None:
#                 self.src_schema_overrides = {}
#             self.hf_schema_source = _get_hf_schema(self.source)
#             self.src_schema = huggingface_schema_to_pxt_schema(
#                 self.hf_schema_source, self.src_schema_overrides, self.src_pk
#             )
#
#             # Add the split column to the schema if requested
#             if self.column_name_for_split is not None:
#                 if self.column_name_for_split in self.src_schema:
#                     raise excs.Error(
#                         f'Column name `{self.column_name_for_split}` already exists in dataset schema;'
#                         f'provide a different `column_name_for_split`'
#                     )
#                 self.src_schema[self.column_name_for_split] = ts.StringType(nullable=True)
#
#             inferred_schema, inferred_pk, self.source_column_map = normalize_schema_names(
#                 self.src_schema, self.src_pk, self.src_schema_overrides, True
#             )
#             return inferred_schema, inferred_pk
#         else:
#             raise NotImplementedError()
#
#     def infer_schema(self) -> dict[str, Any]:
#         self.pxt_schema, self.pxt_pk = self.infer_schema_part1()
#         self.normalize_pxt_schema_types()
#         self.prepare_insert()
#         return self.pxt_schema
#
#     def prepare_for_insert_into_table(self) -> None:
#         _, inferred_pk = self.infer_schema_part1()
#         assert len(inferred_pk) == 0
#         self.prepare_insert()
#
#     def prepare_insert(self) -> None:
#         import datasets
#
#         # extract all class labels from the dataset to translate category ints to strings
#         self.categorical_features = {
#             feature_name: feature_type.names
#             for (feature_name, feature_type) in self.hf_schema_source.items()
#             if isinstance(feature_type, datasets.ClassLabel)
#         }
#         if self.source_column_map is None:
#             self.source_column_map = {}
#         self.check_source_columns_are_insertable(self.hf_schema_source.keys())
#
#     def _translate_row(self, row: dict[str, Any], split_name: str, features: datasets.Features) -> dict[str, Any]:
#         output_row: dict[str, Any] = {}
#         for col_name, val in row.items():
#             # translate category ints to strings
#             new_val = self.categorical_features[col_name][val] if col_name in self.categorical_features else val
#             mapped_col_name = self.source_column_map.get(col_name, col_name)
#
#             new_val = self._translate_val(new_val, features[col_name])
#             output_row[mapped_col_name] = new_val
#
#         # add split name to output row
#         if self.column_name_for_split is not None:
#             output_row[self.column_name_for_split] = split_name
#         return output_row
#
#     def _translate_val(self, val: Any, feature: datasets.Feature) -> Any:
#         """Convert numpy scalars to Python types and images to PIL.Image.Image"""
#         import datasets
#
#         if isinstance(feature, datasets.Value):
#             if isinstance(val, (np.generic, np.ndarray)):
#                 # a scalar, which we want as a standard Python type
#                 assert np.ndim(val) == 0
#                 return val.item()
#             else:
#                 # a standard Python object
#                 return val
#         elif isinstance(feature, datasets.Sequence):
#             assert np.ndim(val) > 0
#             return val
#         elif isinstance(feature, datasets.Image):
#             return PIL.Image.fromarray(val)
#         elif isinstance(feature, dict):
#             assert isinstance(val, dict)
#             return {k: self._translate_val(v, feature[k]) for k, v in val.items()}
#         else:
#             return val
#
#     def valid_row_batch(self) -> Iterator[RowData]:
#         for split_name, split_dataset in self.dataset_dict.items():
#             num_batches = split_dataset.size_in_bytes / self._K_BATCH_SIZE_BYTES
#             tuples_per_batch = math.ceil(split_dataset.num_rows / num_batches)
#             assert tuples_per_batch > 0
#
#             batch = []
#             for row in split_dataset:
#                 batch.append(self._translate_row(row, split_name, split_dataset.features))
#                 if len(batch) >= tuples_per_batch:
#                     yield batch
#                     batch = []
#             # last batch
#             if len(batch) > 0:
#                 yield batch


class HFTableDataConduit(TableDataConduit):
    # class FastHFImporter(TableDataConduit):
    """
    Fast HuggingFace dataset importer using direct Arrow table access.
    Uses dataset.data.slice() with natural chunk boundaries for zero-copy iteration,
    and processes columns individually to avoid to_pydict() overhead.
    """

    column_name_for_split: str | None = None
    categorical_features: dict[str, dict[int, str]]
    dataset_dict: dict[str, 'datasets.Dataset'] = None
    hf_schema_source: dict[str, Any] = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> HFTableDataConduit:
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        import datasets

        assert isinstance(tds.source, (datasets.Dataset, datasets.DatasetDict))
        if 'column_name_for_split' in t.extra_fields:
            t.column_name_for_split = t.extra_fields['column_name_for_split']

        # Store dataset directly - we'll access .data for Arrow tables
        if isinstance(tds.source, datasets.Dataset):
            split_name: str | None = None
            if tds.source.split is not None:
                raw_name = tds.source.split._name
                split_name = raw_name.split('[')[0] if raw_name is not None else None
            t.dataset_dict = {split_name: tds.source}
        else:
            assert isinstance(tds.source, datasets.DatasetDict)
            t.dataset_dict = dict(tds.source)
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
                self.src_schema, self.src_pk, self.src_schema_overrides
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

    def _convert_column(self, column: 'pa.ChunkedArray', feature: object, chunk_size: int) -> list:
        """
        Convert an Arrow column to a list of Python values based on HF feature type.
        Handles all feature types at the column level, recursing for structs.
        Returns a list of length chunk_size with converted values.
        """
        import datasets

        # return scalars as Python scalars
        if isinstance(feature, datasets.Value):
            return column.to_pylist()

        # ClassLabel: int -> string name
        if isinstance(feature, datasets.ClassLabel):
            values = column.to_pylist()
            return [feature.names[v] if v is not None else None for v in values]

        # check for list of dict before Sequence, which could contain array data
        is_list_of_dict = isinstance(feature, (datasets.Sequence, datasets.LargeList)) and isinstance(
            feature.feature, dict
        )
        if is_list_of_dict:
            return column.to_pylist()

        # array data
        if isinstance(feature, datasets.Sequence):
            arr = column.to_numpy(zero_copy_only=False)
            result = []
            for i in range(chunk_size):
                val = arr[i]
                assert not isinstance(val, dict)
                # convert object array of arrays (e.g., multi-channel audio) to proper ndarray
                if (
                    isinstance(val, np.ndarray)
                    and val.dtype == object
                    and len(val) > 0
                    and isinstance(val[0], np.ndarray)
                ):
                    val = np.stack(list(val))
                result.append(val)
            return result

        # Image: decode from bytes
        if isinstance(feature, datasets.Image):
            values = column.to_pylist()
            return [self._decode_image(v) for v in values]

        if isinstance(feature, datasets.Audio):
            # Audio can have different Arrow schemas depending on how it was stored:
            # - Decoded: struct<array: list<float>, path: string, sampling_rate: int32>
            # - Raw: struct<bytes: binary, path: string>
            # Build feature dict from actual Arrow schema
            import pyarrow as pa

            arrow_type = column.type
            if isinstance(arrow_type, pa.StructType):
                audio_features = {}
                for i in range(arrow_type.num_fields):
                    field = arrow_type.field(i)
                    if field.name == 'array':
                        audio_features['array'] = datasets.Sequence(feature=datasets.Value('float32'))
                    elif field.name == 'bytes':
                        audio_features['bytes'] = datasets.Value('binary')
                    elif field.name == 'path':
                        audio_features['path'] = datasets.Value('string')
                    elif field.name == 'sampling_rate':
                        audio_features['sampling_rate'] = datasets.Value('int32')
                return self._convert_struct_column(column, audio_features, chunk_size)
            # Fallback to default assumed structure
            return self._convert_struct_column(
                column,
                {
                    'array': datasets.Sequence(feature=datasets.Value('float32')),
                    'path': datasets.Value('string'),
                    'sampling_rate': datasets.Value('int32'),
                },
                chunk_size,
            )

        if isinstance(feature, dict):
            return self._convert_struct_column(column, feature, chunk_size)

        if isinstance(feature, list):
            return column.to_pylist()

        # Array2D, Array3D, etc.: multi-dimensional fixed-shape arrays
        # These have known fixed shapes, so we can reshape the flat storage directly
        # instead of recursively stacking nested object arrays (much faster)
        if isinstance(feature, (datasets.Array2D, datasets.Array3D, datasets.Array4D, datasets.Array5D)):
            return self._extract_array_feature(column, feature.shape, chunk_size)

        return column.to_pylist()

    def _convert_struct_column(
        self, column: 'pa.ChunkedArray', feature: dict[str, object], chunk_size: int
    ) -> list[dict[str, Any]]:
        """
        Convert a StructArray column to a list of dicts by recursively
        converting each field.
        """
        import pyarrow.compute as pc

        # Initialize result dicts
        results: list[dict[str, Any]] = [{} for _ in range(chunk_size)]

        # Process each field recursively
        for field_name, field_feature in feature.items():
            # Use pyarrow.compute.struct_field for ChunkedArray support
            field_column = pc.struct_field(column, field_name)
            field_values = self._convert_column(field_column, field_feature, chunk_size)

            for i, val in enumerate(field_values):
                results[i][field_name] = val

        return results

    @staticmethod
    def _extract_array_feature(
        column: 'pa.ChunkedArray', shape: tuple[int, ...], chunk_size: int
    ) -> list[np.ndarray]:
        """Extract fixed-shape arrays efficiently by reshaping flat storage.

        HuggingFace Array2D/3D/etc features are stored as nested ListArrays in Arrow
        (e.g., list<list<list<float>>>). The standard to_numpy() returns nested object
        arrays that require expensive recursive stacking. Since these features have
        known fixed shapes, we can instead extract the flat innermost values and
        reshape directly, which is ~100-1000x faster.
        """
        # Get the extension array (avoid combine_chunks overhead when possible)
        if column.num_chunks == 1:
            arr = column.chunks[0]
        else:
            arr = column.combine_chunks()

        # Navigate through nested ListArray storage to innermost values
        storage = arr.storage
        vals = storage.values
        while hasattr(vals, 'values'):
            vals = vals.values

        # Reshape flat array using the known fixed shape
        flat_arr = vals.to_numpy()
        full_shape = (chunk_size,) + tuple(shape)
        reshaped = flat_arr.reshape(full_shape)

        # Return as list of array views (shares memory with reshaped)
        return list(reshaped)

    @staticmethod
    def _decode_image(val: dict[str, Any] | bytes | None) -> PIL.Image.Image | None:
        """Decode Arrow image data to PIL.Image."""
        import io as _io

        if val is None:
            return None
        if isinstance(val, dict) and 'bytes' in val:
            return PIL.Image.open(_io.BytesIO(val['bytes']))
        elif isinstance(val, bytes):
            return PIL.Image.open(_io.BytesIO(val))
        elif isinstance(val, dict) and 'path' in val:
            return PIL.Image.open(val['path'])
        raise AssertionError(f'Unexpected image data type: {type(val)}')

    def valid_row_batch(self) -> Iterator['RowData']:
        for split_name, split_dataset in self.dataset_dict.items():
            features = split_dataset.features
            table = split_dataset.data  # Access underlying Arrow table (public API)

            # Get chunk boundaries from first column's ChunkedArray
            first_column = table.column(0)

            offset = 0
            for chunk in first_column.chunks:
                chunk_size = len(chunk)
                # Zero-copy slice using existing chunk boundaries
                batch = table.slice(offset, chunk_size)

                # Pre-create empty row dicts
                rows: list[dict[str, Any]] = [{} for _ in range(chunk_size)]

                # Add split column if needed
                if self.column_name_for_split is not None:
                    for row in rows:
                        row[self.column_name_for_split] = split_name

                # Process each column using recursive conversion
                for col_idx, col_name in enumerate(batch.schema.names):
                    feature = features[col_name]
                    mapped_col_name = self.source_column_map.get(col_name, col_name)
                    column = batch.column(col_idx)

                    # Convert entire column at once
                    values = self._convert_column(column, feature, chunk_size)

                    for i, val in enumerate(values):
                        rows[i][mapped_col_name] = val

                offset += chunk_size
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
