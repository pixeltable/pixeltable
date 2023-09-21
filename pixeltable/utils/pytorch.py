import PIL.Image
import io
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import math
from pathlib import Path
from ..type_system import ColumnType
from typing import List, Dict, Tuple

def _cumsum(lst):
    acc = [0]
    for x in lst:
        acc.append(acc[-1] + x)
    return acc

class PixeltablePytorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, column_types : Dict[str,ColumnType], parquet_path : Path, image_format : str, 
                 part_metadata : List[Tuple[str, int]]):
        ''' reads parquet sequentially.
            return_image: 'np',  'pt'. 
            'pt' output is result of torchvision.transforms.ToTensor(), with float values between 0 and 1
            'np' is RGB uint8 array
        '''
        super().__init__()
        self.column_types = column_types
        self.pqpath = parquet_path
        self.image_format = image_format
        assert image_format in ['np', 'pt']
        self.part_metadata = part_metadata

        self._totals = _cumsum([x[1] for x in part_metadata])

    def _get_start_position(self, row_start) -> (int, int):
        ''' infers the starting position of the start block within part_metadata based on the sizes '''
        assert row_start >= self._totals[0]
        assert row_start < self._totals[-1]

        prev_acc = 0
        for (i, acc) in enumerate(self._totals[1:], start=1):
            if acc > row_start:
                return (i-1, row_start - prev_acc)
            prev_acc = acc

        assert False, 'unreachable'

    def _unmarshall(self, k, v):
        if not self.column_types[k].is_image_type():
            return v
        
        if isinstance(v, str):
            pil =  PIL.Image.open(v)
        elif isinstance(v, bytes):
            pil =  PIL.Image.open(io.BytesIO(v))
        else:
            assert False, f'unknown image type {type(v)}'

        if self.image_format == 'np':
            import numpy as np
            return np.array(pil)
        elif self.image_format == 'pt':
            import torchvision
            return torchvision.transforms.ToTensor()(pil)
        else:
            assert False, f'unknown image repr {self.image_format}'
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start_row = 0
            end_row = self._totals[-1]
        else:  # in a worker process
            num_workers =  [math.floor(self._totals[-1] / float(worker_info.num_workers)) for _ in range(worker_info.num_workers)]
            assert self._totals[-1] - sum(num_workers) < worker_info.num_workers
            for i in range(self._totals[-1] - sum(num_workers)):
                num_workers[i] += 1
            
            assert sum(num_workers) == self._totals[-1]
            start_rows = _cumsum(num_workers)
            start_row = start_rows[worker_info.id]                        
            end_row = start_rows[worker_info.id + 1]

            assert end_row >= start_row
        
        if start_row == end_row:
            return iter([])
        else:
            return self._iter_range(start_row, end_row)        

    def _iter_range(self, start_row, end_row):
        (part_no, iter_start) = self._get_start_position(start_row)
        total = end_row - start_row
        
        acc = 0

        part_pos = part_no
        iter_pos = iter_start
        tab : pa.Table = pq.read_table(self.pqpath / self.part_metadata[part_no][0])
        cols = tab.to_pydict()
        assert tab.num_rows == self.part_metadata[part_no][1]

        while True:
            while iter_pos < tab.num_rows and acc < total:
                d = {k:cols[k][iter_pos] for k in cols}
                # represent image as higher level types based on dataset schema
                ansd = {k:self._unmarshall(k, v) for (k,v) in d.items()}
                yield ansd
                acc += 1
                iter_pos += 1
            
            if acc == total:
                break

            # move to next part
            part_pos += 1
            assert part_pos < len(self.part_metadata)
            iter_pos = 0
            tab = pq.read_table(self.pqpath / self.part_metadata[part_pos][0])
            cols = tab.to_pydict()