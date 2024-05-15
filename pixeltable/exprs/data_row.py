from __future__ import annotations

import io
import urllib.parse
import urllib.request
from typing import Optional, List, Any, Tuple

import sqlalchemy as sql
import pgvector.sqlalchemy
import PIL
import numpy as np


class DataRow:
    """
    Encapsulates all data and execution state needed by RowBuilder and DataRowBatch:
    - state for in-memory computation
    - state for storing the data
    This is not meant to be a black-box abstraction.

    In-memory representations by column type:
    - StringType: str
    - IntType: int
    - FloatType: float
    - BoolType: bool
    - TimestampType: datetime.datetime
    - JsonType: json-serializable object
    - ArrayType: numpy.ndarray
    - ImageType: PIL.Image.Image
    - VideoType: local path if available, otherwise url
    """
    def __init__(self, size: int, img_slot_idxs: List[int], media_slot_idxs: List[int], array_slot_idxs: List[int]):
        self.vals: List[Any] = [None] * size  # either cell values or exceptions
        self.has_val = [False] * size
        self.excs: List[Optional[Exception]] = [None] * size

        # control structures that are shared across all DataRows in a batch
        self.img_slot_idxs = img_slot_idxs
        self.media_slot_idxs = media_slot_idxs  # all media types aside from image
        self.array_slot_idxs = array_slot_idxs

        # the primary key of a store row is a sequence of ints (the number is different for table vs view)
        self.pk: Optional[Tuple[int, ...]] = None

        # file_urls:
        # - stored url of file for media in vals[i]
        # - None if vals[i] is not media type
        # - not None if file_paths[i] is not None
        self.file_urls: List[Optional[str]] = [None] * size

        # file_paths:
        # - local path of media file in vals[i]; points to the file cache if file_urls[i] is remote
        # - None if vals[i] is not a media type or if there is no local file yet for file_urls[i]
        self.file_paths: List[Optional[str]] = [None] * size

    def clear(self) -> None:
        size = len(self.vals)
        self.vals = [None] * size
        self.has_val = [False] * size
        self.excs = [None] * size
        self.pk = None
        self.file_urls = [None] * size
        self.file_paths = [None] * size

    def copy(self, target: DataRow) -> None:
        """Create a copy of the contents of this DataRow in target
        The copy shares the cell values, but not the control structures (eg, self.has_val), because these
        need to be independently updateable.
        """
        target.vals = self.vals.copy()
        target.has_val = self.has_val.copy()
        target.excs = self.excs.copy()
        target.pk = self.pk
        target.file_urls = self.file_urls.copy()
        target.file_paths = self.file_paths.copy()

    def set_pk(self, pk: Tuple[int, ...]) -> None:
        self.pk = pk

    def has_exc(self, slot_idx: int) -> bool:
        return self.excs[slot_idx] is not None

    def get_exc(self, slot_idx: int) -> Exception:
        assert self.has_val[slot_idx] is False
        assert self.excs[slot_idx] is not None
        return self.excs[slot_idx]

    def set_exc(self, slot_idx: int, exc: Exception) -> None:
        assert self.excs[slot_idx] is None
        self.excs[slot_idx] = exc

        if self.has_val[slot_idx]:
            # eg. during validation, where contents of file is found invalid
            self.has_val[slot_idx] = False
            self.vals[slot_idx] = None
            self.file_paths[slot_idx] = None
            self.file_urls[slot_idx] = None

    def __getitem__(self, index: object) -> Any:
        """Returns in-memory value, ie, what is needed for expr evaluation"""
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index], index

        if self.file_urls[index] is not None and index in self.img_slot_idxs:
            # if we need to load this from a file, it should have been materialized locally
            assert self.file_paths[index] is not None
            if self.vals[index] is None:
                self.vals[index] = PIL.Image.open(self.file_paths[index])
                self.vals[index].load()

        return self.vals[index]

    def get_stored_val(self, index: object, sa_col_type: Optional[sql.types.TypeEngine] = None) -> Any:
        """Return the value that gets stored in the db"""
        assert self.excs[index] is None
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index]

        if self.file_urls[index] is not None and (index in self.img_slot_idxs or index in self.media_slot_idxs):
            # if this is an image or other media type we want to store, we should have a url
            return self.file_urls[index]

        if self.vals[index] is not None and index in self.array_slot_idxs:
            assert isinstance(self.vals[index], np.ndarray)
            np_array = self.vals[index]
            if sa_col_type is not None and isinstance(sa_col_type, pgvector.sqlalchemy.Vector):
                return np_array
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

        return self.vals[index]

    def __setitem__(self, idx: object, val: Any) -> None:
        """Assign in-memory cell value
        This allows overwriting
        """
        assert self.excs[idx] is None

        if (idx in self.img_slot_idxs or idx in self.media_slot_idxs) and isinstance(val, str):
            # this is either a local file path or a URL
            parsed = urllib.parse.urlparse(val)
            # Determine if this is a local file or a remote URL. If the scheme length is <= 1,
            # we assume it's a local file. (This is because a Windows path will be interpreted
            # by urllib as a URL with scheme equal to the drive letter.)
            if len(parsed.scheme) <= 1 or parsed.scheme == 'file':
                # local file path
                assert self.file_urls[idx] is None and self.file_paths[idx] is None
                if len(parsed.scheme) <= 1:
                    self.file_urls[idx] = urllib.parse.urljoin('file:', urllib.request.pathname2url(val))
                    self.file_paths[idx] = val
                else:
                    self.file_urls[idx] = val
                    # Wrap the path in a url2pathname() call to ensure proper handling on Windows.
                    self.file_paths[idx] = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
            else:
                # URL
                assert self.file_urls[idx] is None
                self.file_urls[idx] = val

            if idx in self.media_slot_idxs:
                self.vals[idx] = self.file_paths[idx] if self.file_paths[idx] is not None else self.file_urls[idx]
        elif idx in self.array_slot_idxs and isinstance(val, bytes):
            self.vals[idx] = np.load(io.BytesIO(val))
        else:
            self.vals[idx] = val
        self.has_val[idx] = True

    def set_file_path(self, idx: object, path: str) -> None:
        """Augment an existing url with a local file path"""
        assert self.has_val[idx]
        assert idx in self.img_slot_idxs or idx in self.media_slot_idxs
        self.file_paths[idx] = path
        if idx in self.media_slot_idxs:
            self.vals[idx] = path

    def flush_img(self, index: object, filepath: Optional[str] = None) -> None:
        """Discard the in-memory value and save it to a local file, if filepath is not None"""
        if self.vals[index] is None:
            return
        assert self.excs[index] is None
        if self.file_paths[index] is None:
            if filepath is not None:
                # we want to save this to a file
                self.file_paths[index] = filepath
                self.file_urls[index] = urllib.parse.urljoin('file:', urllib.request.pathname2url(filepath))
                self.vals[index].save(filepath, format='JPEG')
            else:
                # we discard the content of this cell
                self.has_val[index] = False
        else:
            # we already have a file for this image, nothing left to do
            pass
        self.vals[index] = None

