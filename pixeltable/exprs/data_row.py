from __future__ import annotations

import datetime
import io
import urllib.parse
import urllib.request
from typing import Any, Optional

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL
import PIL.Image
import sqlalchemy as sql

from pixeltable import env


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

    vals: list[Any]
    has_val: list[bool]
    excs: list[Optional[Exception]]

    # control structures that are shared across all DataRows in a batch
    img_slot_idxs: list[int]
    media_slot_idxs: list[int]
    array_slot_idxs: list[int]

    # the primary key of a store row is a sequence of ints (the number is different for table vs view)
    pk: Optional[tuple[int, ...]]

    # file_urls:
    # - stored url of file for media in vals[i]
    # - None if vals[i] is not media type
    # - not None if file_paths[i] is not None
    file_urls: list[Optional[str]]

    # file_paths:
    # - local path of media file in vals[i]; points to the file cache if file_urls[i] is remote
    # - None if vals[i] is not a media type or if there is no local file yet for file_urls[i]
    file_paths: list[Optional[str]]

    def __init__(self, size: int, img_slot_idxs: list[int], media_slot_idxs: list[int], array_slot_idxs: list[int]):
        self.vals = [None] * size
        self.has_val = [False] * size
        self.excs = [None] * size
        self.img_slot_idxs = img_slot_idxs
        self.media_slot_idxs = media_slot_idxs
        self.array_slot_idxs = array_slot_idxs
        self.pk = None
        self.file_urls = [None] * size
        self.file_paths = [None] * size

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

    def set_pk(self, pk: tuple[int, ...]) -> None:
        self.pk = pk

    def has_exc(self, slot_idx: Optional[int] = None) -> bool:
        if slot_idx is not None:
            return self.excs[slot_idx] is not None
        return any(exc is not None for exc in self.excs)

    def get_exc(self, slot_idx: Optional[int] = None) -> Optional[Exception]:
        if slot_idx is not None:
            return self.excs[slot_idx]
        for exc in self.excs:
            if exc is not None:
                return exc
        assert False

    def set_exc(self, slot_idx: int, exc: Exception) -> None:
        assert self.excs[slot_idx] is None
        self.excs[slot_idx] = exc

        # an exception means the value is None
        self.has_val[slot_idx] = True
        self.vals[slot_idx] = None
        self.file_paths[slot_idx] = None
        self.file_urls[slot_idx] = None

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: object) -> Any:
        """Returns in-memory value, ie, what is needed for expr evaluation"""
        assert isinstance(index, int)
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index], index

        if self.file_urls[index] is not None and index in self.img_slot_idxs:
            # if we need to load this from a file, it should have been materialized locally
            # TODO this fails if the url was instantiated dynamically using astype()
            assert self.file_paths[index] is not None
            if self.vals[index] is None:
                self.vals[index] = PIL.Image.open(self.file_paths[index])
                self.vals[index].load()

        return self.vals[index]

    def get_stored_val(self, index: int, sa_col_type: Optional[sql.types.TypeEngine] = None) -> Any:
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

        # for JSON columns, we need to store None as an explicit NULL, otherwise it stores a json 'null'
        if self.vals[index] is None and sa_col_type is not None and isinstance(sa_col_type, sql.JSON):
            return sql.sql.null()

        if isinstance(self.vals[index], datetime.datetime) and self.vals[index].tzinfo is None:
            # if the datetime is naive, cast it to the default time zone
            return self.vals[index].replace(tzinfo=env.Env.get().default_time_zone)

        return self.vals[index]

    def __setitem__(self, idx: object, val: Any) -> None:
        """Assign in-memory cell value
        This allows overwriting
        """
        assert isinstance(idx, int)
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

    def set_file_path(self, idx: int, path: str) -> None:
        """Augment an existing url with a local file path"""
        assert self.has_val[idx]
        assert idx in self.img_slot_idxs or idx in self.media_slot_idxs
        self.file_paths[idx] = path
        if idx in self.media_slot_idxs:
            self.vals[idx] = path

    def flush_img(self, index: int, filepath: Optional[str] = None) -> None:
        """Discard the in-memory value and save it to a local file, if filepath is not None"""
        if self.vals[index] is None:
            return
        assert self.excs[index] is None
        if self.file_paths[index] is None:
            if filepath is not None:
                # we want to save this to a file
                self.file_paths[index] = filepath
                self.file_urls[index] = urllib.parse.urljoin('file:', urllib.request.pathname2url(filepath))
                image = self.vals[index]
                assert isinstance(image, PIL.Image.Image)
                # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
                # In that case, use WebP instead.
                format = 'webp' if image.has_transparency_data else 'jpeg'
                image.save(filepath, format=format)
            else:
                # we discard the content of this cell
                self.has_val[index] = False
        else:
            # we already have a file for this image, nothing left to do
            pass
        self.vals[index] = None

    @property
    def rowid(self) -> tuple[int, ...]:
        return self.pk[:-1]

    @property
    def v_min(self) -> int:
        return self.pk[-1]
