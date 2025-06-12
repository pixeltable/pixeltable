from __future__ import annotations

import enum
import json
import logging
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional, Union, cast

import pandas as pd
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
    source: TableDataSource
    source_format: Optional[str] = None
    source_column_map: Optional[dict[str, str]] = None
    if_row_exists: Literal['update', 'ignore', 'error'] = 'error'
    pxt_schema: Optional[dict[str, Any]] = None
    src_schema_overrides: Optional[dict[str, Any]] = None
    src_schema: Optional[dict[str, Any]] = None
    pxt_pk: Optional[list[str]] = None
    src_pk: Optional[list[str]] = None
    valid_rows: Optional[RowData] = None
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

    def is_direct_df(self) -> bool:
        return isinstance(self.source, pxt.DataFrame) and self.source_column_map is None

    def normalize_pxt_schema_types(self) -> None:
        for name, coltype in self.pxt_schema.items():
            self.pxt_schema[name] = ts.ColumnType.normalize_type(coltype)

    def infer_schema(self) -> dict[str, Any]:
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


class DFTableDataConduit(TableDataConduit):
    pxt_df: pxt.DataFrame = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'DFTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        assert isinstance(tds.source, pxt.DataFrame)
        t.pxt_df = tds.source
        return t

    def infer_schema(self) -> dict[str, Any]:
        self.pxt_schema = self.pxt_df.schema
        self.pxt_pk = self.src_pk
        return self.pxt_schema

    def prepare_for_insert_into_table(self) -> None:
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.pxt_df.schema.keys())


class RowDataTableDataConduit(TableDataConduit):
    raw_rows: Optional[RowData] = None
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

    def infer_schema(self) -> dict[str, Any]:
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

    def infer_schema_part1(self) -> tuple[dict[str, Any], list[str]]:
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

    def infer_schema(self) -> dict[str, Any]:
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
    hf_ds: Optional[Union[datasets.Dataset, datasets.DatasetDict]] = None
    column_name_for_split: Optional[str] = None
    categorical_features: dict[str, dict[int, str]]
    hf_schema: dict[str, Any] = None
    dataset_dict: dict[str, datasets.Dataset] = None
    hf_schema_source: dict[str, Any] = None

    @classmethod
    def from_tds(cls, tds: TableDataConduit) -> 'HFTableDataConduit':
        tds_fields = {f.name for f in fields(tds)}
        kwargs = {k: v for k, v in tds.__dict__.items() if k in tds_fields}
        t = cls(**kwargs)
        import datasets

        assert isinstance(tds.source, (datasets.Dataset, datasets.DatasetDict))
        t.hf_ds = tds.source
        if 'column_name_for_split' in t.extra_fields:
            t.column_name_for_split = t.extra_fields['column_name_for_split']
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

    def infer_schema_part1(self) -> tuple[dict[str, Any], list[str]]:
        from pixeltable.io.hf_datasets import _get_hf_schema, huggingface_schema_to_pxt_schema

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.hf_schema_source = _get_hf_schema(self.hf_ds)
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

        if isinstance(self.source, datasets.Dataset):
            # when loading an hf dataset partially, dataset.split._name is sometimes the form "train[0:1000]"
            raw_name = self.source.split._name
            split_name = raw_name.split('[')[0] if raw_name is not None else None
            self.dataset_dict = {split_name: self.source}
        else:
            assert isinstance(self.source, datasets.DatasetDict)
            self.dataset_dict = self.source

        # extract all class labels from the dataset to translate category ints to strings
        self.categorical_features = {
            feature_name: feature_type.names
            for (feature_name, feature_type) in self.hf_schema_source.items()
            if isinstance(feature_type, datasets.ClassLabel)
        }
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.hf_schema_source.keys())

    def _translate_row(self, row: dict[str, Any], split_name: str) -> dict[str, Any]:
        output_row: dict[str, Any] = {}
        for col_name, val in row.items():
            # translate category ints to strings
            new_val = self.categorical_features[col_name][val] if col_name in self.categorical_features else val
            mapped_col_name = self.source_column_map.get(col_name, col_name)

            # Convert values to the appropriate type if needed
            try:
                checked_val = self.pxt_schema[mapped_col_name].create_literal(new_val)
            except TypeError as e:
                msg = str(e)
                raise excs.Error(f'Error in column {col_name}: {msg[0].lower() + msg[1:]}\nRow: {row}') from e
            output_row[mapped_col_name] = checked_val

        # add split name to output row
        if self.column_name_for_split is not None:
            output_row[self.column_name_for_split] = split_name
        return output_row

    def valid_row_batch(self) -> Iterator[RowData]:
        for split_name, split_dataset in self.dataset_dict.items():
            num_batches = split_dataset.size_in_bytes / self._K_BATCH_SIZE_BYTES
            tuples_per_batch = math.ceil(split_dataset.num_rows / num_batches)
            assert tuples_per_batch > 0

            batch = []
            for row in split_dataset:
                batch.append(self._translate_row(row, split_name))
                if len(batch) >= tuples_per_batch:
                    yield batch
                    batch = []
            # last batch
            if len(batch) > 0:
                yield batch


class ParquetTableDataConduit(TableDataConduit):
    pq_ds: Optional[ParquetDataset] = None

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

    def infer_schema_part1(self) -> tuple[dict[str, Any], list[str]]:
        from pixeltable.utils.arrow import ar_infer_schema

        if self.source_column_map is None:
            if self.src_schema_overrides is None:
                self.src_schema_overrides = {}
            self.src_schema = ar_infer_schema(self.pq_ds.schema, self.src_schema_overrides, self.src_pk)
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
        if self.source_column_map is None:
            self.source_column_map = {}
        self.check_source_columns_are_insertable(self.pq_ds.schema.names)
        self.total_rows = 0

    def valid_row_batch(self) -> Iterator[RowData]:
        from pixeltable.utils.arrow import iter_tuples2

        try:
            for fragment in self.pq_ds.fragments:  # type: ignore[attr-defined]
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
        if isinstance(self.source, pxt.DataFrame):
            return DFTableDataConduit.from_tds(self)
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
