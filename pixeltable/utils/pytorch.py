import io
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.utils.data
import math
from pathlib import Path
import PIL.Image
import json
from typing import Dict, Tuple, Generator, Any
import datetime

from pixeltable.type_system import ColumnType
from pixeltable.utils.parquet import get_part_metadata
import numpy as np

def _cumsum(lst):
    acc = [0]
    for x in lst:
        acc.append(acc[-1] + x)
    return acc

class PixeltablePytorchDataset(torch.utils.data.IterableDataset):
    """
    PyTorch dataset interface for pixeltable data.
    NB. This class must inherit from torch.utils.data.IterableDataset for it
    to work with torch.utils.data.DataLoader.
    """
    def __init__(
        self,
        path: Path,
        image_format: str,
    ):
        """
        Args:
            path: path to directory containing parquet files
            image_format: 'np' or 'pt'. 'np' is RGB uint8 array,
                        'pt' is result of torchvision.transforms.ToTensor()
        """
        super().__init__()

        self.path = path
        self.image_format = image_format
        assert image_format in ["np", "pt"]
        column_type_path = path / '.pixeltable.column_types.json'
        assert column_type_path.exists(), f"missing {column_type_path}"
        with column_type_path.open() as f:
            column_types = json.load(f)
        self.column_types = {k: ColumnType.from_dict(v) for k, v in column_types.items()}
        self.part_metadata = get_part_metadata(self.path)
        self._totals = _cumsum([x[1] for x in self.part_metadata])

    def _get_start_position(self, row_start: int) -> Tuple[int, int]:
        """
        Returns the starting parquet file and row within that file for a given 'global' row number.
        based on the individual sizes of each part
        """
        assert row_start >= self._totals[0]
        assert row_start < self._totals[-1]

        prev_acc = 0
        for i, acc in enumerate(self._totals[1:], start=1):
            if acc > row_start:
                return (i - 1, row_start - prev_acc)
            prev_acc = acc

        assert False, "unreachable"

    def _unmarshall(self, k: str, v: Any) -> Any:
        if self.column_types[k].is_image_type():
            assert isinstance(v, bytes)
            im = PIL.Image.open(io.BytesIO(v))
            arr = np.array(im) # will copy data to guarantee "WRITEABLE" flag assertion below.
            assert arr.flags["WRITEABLE"]

            if self.image_format == "np":
                return arr

            assert self.image_format == "pt"
            import torchvision  # pylint: disable = import-outside-toplevel

            # use arr instead of im in ToTensor() to guarantee array input
            # to torch.from_numpy is writable. Using im is a suspected cause of
            # https://github.com/pixeltable/pixeltable/issues/69
            return torchvision.transforms.ToTensor()(arr)
        elif self.column_types[k].is_json_type():
            assert isinstance(v, str)
            return json.loads(v)
        elif self.column_types[k].is_array_type():
            assert isinstance(v, np.ndarray)
            # WRITEABLE is required for torch collate function, or undefined behavior
            if not v.flags["WRITEABLE"]:
                vout = v.copy()
                assert vout.flags["WRITEABLE"]
                return vout
            else:
                return v
        elif self.column_types[k].is_timestamp_type():
            # pytorch default collation only supports numeric types
            assert isinstance(v, datetime.datetime)
            return v.timestamp()
        else:
            assert not isinstance(v, np.ndarray) # all array outputs should be handled above
            return v

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start_row = 0
            end_row = self._totals[-1]
        else:  # in a worker process
            num_workers = [
                math.floor(self._totals[-1] / float(worker_info.num_workers))
                for _ in range(worker_info.num_workers)
            ]
            assert self._totals[-1] - sum(num_workers) < worker_info.num_workers
            for i in range(self._totals[-1] - sum(num_workers)):
                num_workers[i] += 1

            assert sum(num_workers) == self._totals[-1]
            start_rows = _cumsum(num_workers)
            start_row = start_rows[worker_info.id]
            end_row = start_rows[worker_info.id + 1]

        if start_row == end_row:
            return iter([])  # type: ignore
        else:
            return self._iter_range(start_row, end_row)

    def _iter_range(self, start_row : int, end_row : int) -> Generator[Dict[str, Any], None, None]:
        (part_no, iter_start) = self._get_start_position(start_row)
        total = end_row - start_row

        acc = 0
        part_pos = part_no
        iter_pos = iter_start

        def _to_column_dict(tab : pa.Table) -> Dict[str, Any]:
            column_dict = {}
            for k in tab.column_names:
                if self.column_types[k].is_array_type():
                    # treat array columns as numpy arrays to easily preserve numpy type
                    column_dict[k] = tab.column(k).to_numpy()
                else:
                    # for the rest, use pydict to preserve python types
                    column_dict[k] = tab.column(k).to_pylist()
            return column_dict

        tab: pa.Table = pq.read_table(self.path / self.part_metadata[part_no][0])
        column_dict = _to_column_dict(tab)
        assert tab.num_rows == self.part_metadata[part_no][1]

        while True:
            while iter_pos < tab.num_rows and acc < total:
                next_tup = {}
                for col_name, col_vals in column_dict.items():
                    raw_val = col_vals[iter_pos]
                    next_tup[col_name] = self._unmarshall(col_name, raw_val)

                yield next_tup
                acc += 1
                iter_pos += 1

            if acc == total:
                break

            # move on to next part
            part_pos += 1
            assert part_pos < len(self.part_metadata)
            iter_pos = 0
            tab = pq.read_table(self.path / self.part_metadata[part_pos][0])
            column_dict = _to_column_dict(tab)