import io
import pyarrow as pa
import pyarrow.parquet
import torch
import torch.utils.data
from pathlib import Path
import PIL.Image
import json
from typing import Dict, Iterator, Any
import datetime

from pixeltable.type_system import ColumnType
import numpy as np

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
        self.part_metadata = pa.parquet.ParquetDataset(path).files

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
            if not v.flags["WRITEABLE"]:
                v = v.copy()
            assert v.flags["WRITEABLE"]
            return v
        elif self.column_types[k].is_timestamp_type():
            # pytorch default collation only supports numeric types
            assert isinstance(v, datetime.datetime)
            return v.timestamp()
        else:
            assert not isinstance(v, np.ndarray) # all array outputs should be handled above
            return v

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import pixeltable.utils.arrow as arrow
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            part_list = range(len(self.part_metadata))
        else:
            part_list = [ i for i in part_list if (i % worker_info.num_workers) == worker_info.id ]

        for part_no in part_list:
            pqf = pa.parquet.ParquetFile(self.part_metadata[part_no])
            for batch in pqf.iter_batches():
                for tup in arrow.iter_tuples(batch):
                    yield {k: self._unmarshall(k, v) for k, v in tup.items()}