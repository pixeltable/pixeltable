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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pat
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

    _K_BATCH_SIZE_BYTES = 256 * 2**20

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
    """HuggingFace dataset importer"""

    column_name_for_split: str | None = None
    categorical_features: dict[str, dict[int, str]]
    dataset_dict: dict[str, 'datasets.Dataset'] = None  # key: split name
    hf_schema_source: dict[str, Any] = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> HFTableDataConduit:
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        import datasets

        assert isinstance(tds.source, cls._get_dataset_classes())
        if 'column_name_for_split' in t.extra_fields:
            t.column_name_for_split = t.extra_fields['column_name_for_split']

        if isinstance(tds.source, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            tds.source = tds.source.with_format('arrow')

        if isinstance(tds.source, (datasets.Dataset, datasets.IterableDataset)):
            split_name = str(tds.source.split) if tds.source.split is not None else None
            t.dataset_dict = {split_name: tds.source}
        else:
            assert isinstance(tds.source, (datasets.DatasetDict, datasets.IterableDatasetDict))
            t.dataset_dict = dict(tds.source)

        # Disable auto-decoding for Audio and Image columns, we want to write the bytes directly to temp files
        for ds_split_name, dataset in list(t.dataset_dict.items()):
            for col_name, feature in dataset.features.items():
                if isinstance(feature, (datasets.Audio, datasets.Image)):
                    t.dataset_dict[ds_split_name] = t.dataset_dict[ds_split_name].cast_column(
                        col_name, feature.__class__(decode=False)
                    )
        return t

    @classmethod
    def _get_dataset_classes(cls) -> tuple[type, ...]:
        import datasets

        return (datasets.Dataset, datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict)

    @classmethod
    def is_applicable(cls, tds: TableDataConduit) -> bool:
        try:
            return (isinstance(tds.source_format, str) and tds.source_format.lower() == 'huggingface') or isinstance(
                tds.source, cls._get_dataset_classes()
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

    def _convert_column(self, column: 'pa.ChunkedArray', feature: object) -> list:
        """
        Convert an Arrow column to a list of Python values based on HF feature type.
        Handles all feature types at the column level, recursing for structs.
        Returns a list of length chunk_size.
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

        # array data represented as a (possibly nested) sequence of numerical data: convert to numpy arrays
        if self._is_sequence_of_numerical(feature):
            arr = column.to_numpy(zero_copy_only=False)
            result: list = []
            for i in range(len(column)):
                val = arr[i]
                assert not isinstance(val, dict)  # we dealt with list of dicts earlier
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

        if isinstance(feature, (datasets.Audio, datasets.Image)):
            # Audio/Image is stored in Arrow as struct<bytes: binary, path: string>

            from pixeltable.utils.local_store import TempStore

            arrow_type = column.type
            if not pa.types.is_struct(arrow_type):
                raise pxt.Error(f'Expected struct type for Audio column, got {arrow_type}')
            field_names = {field.name for field in arrow_type}
            if 'bytes' not in field_names or 'path' not in field_names:
                raise pxt.Error(f"Audio struct missing required fields 'bytes' and/or 'path', has: {field_names}")

            bytes_column = pc.struct_field(column, 'bytes')
            path_column = pc.struct_field(column, 'path')

            bytes_list = bytes_column.to_pylist()
            path_list = path_column.to_pylist()

            result = []
            for bytes, path in zip(bytes_list, path_list):
                if bytes is None:
                    result.append(None)
                    continue
                # we want to preserve the extension from the original path
                ext = Path(path).suffix if path is not None else None
                temp_path = TempStore.create_path(extension=ext)
                temp_path.write_bytes(bytes)
                result.append(str(temp_path))
            return result

        if isinstance(feature, dict):
            return self._convert_struct_column(column, feature)

        if isinstance(feature, list):
            return column.to_pylist()

        # Array<N>D: multi-dimensional fixed-shape arrays
        if isinstance(feature, (datasets.Array2D, datasets.Array3D, datasets.Array4D, datasets.Array5D)):
            return self._convert_array_feature(column, feature.shape)

        return column.to_pylist()

    def _is_sequence_of_numerical(self, feature: object) -> bool:
        """Returns True if feature is a (nested) Sequence of numerical values."""
        import datasets

        if not isinstance(feature, datasets.Sequence):
            return False
        if isinstance(feature.feature, datasets.Sequence):
            return self._is_sequence_of_numerical(feature.feature)

        pa_type = feature.feature.pa_type
        return pa_type is not None and (pat.is_integer(pa_type) or pat.is_floating(pa_type))

    def _convert_struct_column(self, column: 'pa.ChunkedArray', feature: dict[str, object]) -> list[dict[str, Any]]:
        """
        Convert a StructArray column to a list of dicts by recursively
        converting each field.
        """

        results: list[dict[str, Any]] = [{} for _ in range(len(column))]
        for field_name, field_feature in feature.items():
            field_column = pc.struct_field(column, field_name)
            field_values = self._convert_column(field_column, field_feature)

            for i, val in enumerate(field_values):
                results[i][field_name] = val

        return results

    def _convert_array_feature(self, column: 'pa.ChunkedArray', shape: tuple[int, ...]) -> list[np.ndarray]:
        arr: pa.ExtensionArray
        # TODO: can we get multiple chunks here?
        if column.num_chunks == 1:
            arr = column.chunks[0]  # type: ignore[assignment]
        else:
            arr = column.combine_chunks()  # type: ignore[assignment]

        # an Array<N>D feature is stored in Arrow as a list<list<...<dtype>>>; we want to peel off the outer lists
        # to get to contiguous storage and then reshape that
        storage = arr.storage
        vals = storage.values
        while hasattr(vals, 'values'):
            vals = vals.values
        flat_arr = vals.to_numpy()
        chunk_shape = (len(column), *shape)
        reshaped = flat_arr.reshape(chunk_shape)

        # Return as list of array views (shares memory with reshaped)
        return list(reshaped)

    def valid_row_batch(self) -> Iterator['RowData']:
        import datasets

        for split_name, split_dataset in self.dataset_dict.items():
            features = split_dataset.features
            if isinstance(split_dataset, datasets.Dataset):
                table = split_dataset.data  # the underlying Arrow table
                yield from self._process_arrow_table(table, split_name, features)
            else:
                # we're getting batches of Arrow tables, since we did set_format('arrow');
                # use a trial batch to determine the target batch size
                first_batch = next(split_dataset.iter(batch_size=16))
                bytes_per_row = int(first_batch.nbytes / len(first_batch))
                batch_size = self._K_BATCH_SIZE_BYTES // bytes_per_row
                yield from self._process_arrow_table(first_batch, split_name, features)
                for batch in split_dataset.skip(16).iter(batch_size=batch_size):
                    yield from self._process_arrow_table(batch, split_name, features)

    def _process_arrow_table(self, table: 'pa.Table', split_name: str, features: dict[str, Any]) -> Iterator[RowData]:
        # get chunk boundaries from first column's ChunkedArray
        first_column = table.column(0)
        offset = 0
        for chunk in first_column.chunks:
            chunk_size = len(chunk)
            # zero-copy slice using existing chunk boundaries
            batch = table.slice(offset, chunk_size)

            # we assemble per-row dicts by from lists of per-column values
            rows: list[dict[str, Any]] = [{} for _ in range(chunk_size)]
            if self.column_name_for_split is not None:
                for row in rows:
                    row[self.column_name_for_split] = split_name

            for col_idx, col_name in enumerate(batch.schema.names):
                feature = features[col_name]
                mapped_col_name = self.source_column_map.get(col_name, col_name)
                column = batch.column(col_idx)
                values = self._convert_column(column, feature)
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

        assert isinstance(tds.source, str)
        input_path = Path(tds.source).expanduser()
        t.pq_ds = pa.parquet.ParquetDataset(str(input_path))
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
