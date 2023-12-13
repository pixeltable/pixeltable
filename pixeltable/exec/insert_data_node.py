from typing import List, Dict, Any, Optional
import urllib
import logging
import os

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
import pixeltable.env as env
from pixeltable.utils.imgstore import ImageStore


_logger = logging.getLogger('pixeltable')

class InsertDataNode(ExecNode):
    """Outputs in-memory data as a row batch of a particular table"""
    def __init__(
            self, tbl: catalog.TableVersion, rows: List[List[Any]], row_column_pos: Dict[str, int],
            row_builder: exprs.RowBuilder, input_cols: List[exprs.ColumnSlotIdx], start_row_id: int,
    ):
        super().__init__(row_builder, [], [], None)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.input_rows = rows
        self.row_column_pos = row_column_pos  # col name -> idx of col in self.input_rows
        self.input_cols = input_cols
        self.start_row_id = start_row_id
        self.has_returned_data = False
        self.output_rows: Optional[DataRowBatch] = None

        # TODO: remove this with component views
        self.boto_client: Optional[Any] = None

    def _open(self) -> None:
        """Create row batch and populate with self.data"""

        for info in self.input_cols:
            assert info.col.name in self.row_column_pos

        # before anything, convert any literal images within the input rows into references
        # copy the input rows to avoid indirectly modifying the argument
        _input_rows = [row.copy() for row in self.input_rows]
        for info in self.input_cols:
            if info.col.col_type.is_image_type():
                col_idx = self.row_column_pos[info.col.name]
                for row_idx, input_row in enumerate(_input_rows):
                    val = input_row[col_idx]
                    if isinstance(val, bytes):
                        # we will save literal to a file here and use this path as the new value
                        valpath = str(ImageStore.get_path(self.tbl.id, info.col.id, self.tbl.version))
                        open(valpath, 'wb').write(val)
                        input_row[col_idx] = valpath

        self.input_rows = _input_rows

        self.output_rows = DataRowBatch(self.tbl, self.row_builder, len(self.input_rows))
        for info in self.input_cols:
            col_idx = self.row_column_pos[info.col.name]
            for row_idx, input_row in enumerate(self.input_rows):
                self.output_rows[row_idx][info.slot_idx] = input_row[col_idx]

        self.output_rows.set_row_ids([self.start_row_id + i for i in range(len(self.output_rows))])
        self.ctx.num_rows = len(self.output_rows)

    def _get_local_path(self, url: str) -> str:
        """Returns local path for url"""
        if url is None:
            return None
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == '' or parsed.scheme == 'file':
            # local file path
            return parsed.path
        if parsed.scheme == 's3':
            from pixeltable.utils.s3 import get_client
            if self.boto_client is None:
                self.boto_client = get_client()
            tmp_path = str(env.Env.get().tmp_dir / os.path.basename(parsed.path))
            self.boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), tmp_path)
            return tmp_path
        assert False, f'Unsupported URL scheme: {parsed.scheme}'

    def __next__(self) -> DataRowBatch:
        if self.has_returned_data:
            raise StopIteration
        self.has_returned_data = True
        _logger.debug(f'InsertDataNode: created row batch with {len(self.output_rows)} output_rows')
        return self.output_rows

