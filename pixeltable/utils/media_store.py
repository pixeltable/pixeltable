from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from uuid import UUID

import PIL.Image

from pixeltable import env, exceptions as excs
from pixeltable.utils.media_path import MediaPath

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class MediaStore:
    """
    Utilities to manage media files stored in a local filesystem directory.

    Media file names are a composite of: table id, column id, tbl_version, new uuid:
    the table id/column id/tbl_version are redundant but useful for identifying all files for a table
    or all files created for a particular version of a table
    """

    __base_dir: Path

    def __init__(self, base_dir: Path):
        """Initialize a MediaStore with a base directory."""
        assert isinstance(base_dir, Path), 'Base directory must be a Path instance.'
        self.__base_dir = base_dir

    @staticmethod
    def validate_destination(col_name: str, dest: Union[str, Path]) -> str:
        """Convert a Column destination parameter to a URI, else raise errors."""
        if isinstance(dest, Path):
            dest_path = dest
        elif not isinstance(dest, str):
            raise excs.Error(f'Column {col_name}: "destination" must be a string or path, got {dest!r}')
        else:
            # Check if it's already a valid URI scheme
            valid_schemes: set[str] = set()  # {'s3', 'gs', 'azure'}
            parsed = urllib.parse.urlparse(dest)
            if parsed.scheme:
                # For file:// URIs, check if it points to a directory
                # len(parsed.scheme) == 1 handles Windows drive letters like C:\
                if parsed.scheme.lower() == 'file' or len(parsed.scheme) == 1:
                    path_str = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
                    dest_path = Path(path_str)
                elif parsed.scheme.lower() in valid_schemes:
                    # If it's a valid scheme, treat it as a URI and do no further checking
                    return dest
                else:
                    raise excs.Error(
                        f'Column {col_name}: "destination" must be a valid URI to a supported destination, got {dest}'
                    )
            else:
                # If no scheme, treat as local file path
                dest_path = Path(dest)

        # Check if path exists and validate it's a directory
        if not dest_path.exists():
            raise excs.Error(f'Column {col_name}: "destination" does not exist: {dest}')
        if not dest_path.is_dir():
            raise excs.Error(f'Column {col_name}: "destination" must be a directory, not a file: {dest}')

        # Check if path is absolute
        if dest_path.is_absolute():
            # Convert to file URI
            return dest_path.as_uri()

        # For relative paths, convert to absolute first
        try:
            absolute_path = dest_path.resolve()
            return absolute_path.as_uri()
        except (OSError, ValueError) as e:
            raise excs.Error(f'Column {col_name}: Invalid destination path: {dest}. Error: {str(e)!r}') from None

    @staticmethod
    def file_url_to_path(url: str) -> Optional[Path]:
        """Convert a file:// URI to a Path object with support for Windows UNC paths."""
        assert isinstance(url, str), type(url)
        parsed = urllib.parse.urlparse(url)

        # Verify it's a file scheme
        # We should never be passed a local file path here. The "len > 1" ensures that Windows
        # file paths aren't mistaken for URLs with a single-character scheme.
        assert len(parsed.scheme) > 1, url
        if parsed.scheme.lower() != 'file':
            return None

        path_str = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        return Path(path_str)

    @classmethod
    def get(cls, base_uri: Optional[str] = None) -> MediaStore:
        """Get a MediaStoreFile instance for the specified base URI, or the environment's media_dir if None."""
        if base_uri is None:
            return MediaStore(env.Env.get().media_dir)
        base_path = cls.file_url_to_path(base_uri)
        if base_path is None:
            raise excs.Error(f"URI must have 'file' scheme: '{base_uri}'")
        return MediaStore(base_path)

    @classmethod
    def _save_binary_media_file(cls, file_data: bytes, dest_path: Path, format: Optional[str]) -> Path:
        """Save a media binary data to a file in a MediaStore. format is ignored for binary data."""
        assert isinstance(file_data, bytes)
        with open(dest_path, 'wb') as f:
            f.write(file_data)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage
        return dest_path

    @classmethod
    def _save_pil_image_file(cls, image: PIL.Image.Image, dest_path: Path, format: Optional[str]) -> Path:
        """Save a PIL Image to a file in a MediaStore with the specified format."""
        if dest_path.suffix != f'.{format}':
            dest_path = dest_path.with_name(f'{dest_path.name}.{format}')

        with open(dest_path, 'wb') as f:
            image.save(f, format=format)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage
        return dest_path

    def _prepare_media_path_raw(self, tbl_id: UUID, col_id: int, tbl_version: int, ext: Optional[str] = None) -> Path:
        """
        Construct a new, unique Path name for a persisted media file, and create the parent directory
        for the new Path if it does not already exist. The Path will reside in
        the environment's media_dir.
        """
        prefix, filename = MediaPath.media_prefix_file_raw(tbl_id, col_id, tbl_version, ext)
        parent = self.__base_dir / Path(prefix)
        parent.mkdir(parents=True, exist_ok=True)
        return parent / filename

    def _prepare_media_path(self, col: Column, ext: Optional[str] = None) -> Path:
        """
        Construct a new, unique Path name for a persisted media file, and create the parent directory
        for the new Path if it does not already exist. The Path will reside in
        the environment's media_dir.
        """
        prefix, filename = MediaPath.media_prefix_file(col, ext)
        parent = self.__base_dir / Path(prefix)
        parent.mkdir(parents=True, exist_ok=True)
        return parent / filename

    def contains_path(self, file_path: Path) -> bool:
        """Return True if the given path refers to a file managed by this MediaStore, else False."""
        return str(file_path).startswith(str(self.__base_dir))

    def resolve_url(self, file_url: Optional[str]) -> Optional[Path]:
        """Return path if the given url refers to a file managed by this MediaStore, else None.

        Args:
            file_url: URL to check

        Returns:
            If the url is a managed file, return a Path() to the file, None, otherwise
        """
        if file_url is None:
            return None
        file_path = self.file_url_to_path(file_url)
        if file_path is None:
            return None
        if not str(file_path).startswith(str(self.__base_dir)):
            # not a tmp file
            return None
        return file_path

    def relocate_local_media_file(self, src_path: Path, col: Column) -> str:
        """Relocate a local file to a MediaStore, and return its new URL"""
        dest_path = self._prepare_media_path(col, ext=src_path.suffix)
        src_path.rename(dest_path)
        new_file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
        _logger.debug(f'Media Storage: moved {src_path} to {new_file_url}')
        return new_file_url

    def copy_local_media_file(self, src_path: Path, col: Column) -> str:
        """Copy a local file to a MediaStore, and return its new URL"""
        dest_path = self._prepare_media_path(col, ext=src_path.suffix)
        shutil.copy2(src_path, dest_path)
        new_file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
        _logger.debug(f'Media Storage: copied {src_path} to {new_file_url}')
        return new_file_url

    def save_media_object(self, data: bytes | PIL.Image.Image, col: Column, format: Optional[str]) -> tuple[Path, str]:
        """Save a media data object to a file in a MediaStore
        Returns:
            dest_path: Path to the saved media file
            url: URL of the saved media file
        """
        assert col.col_type.is_media_type(), f'MediaStore: request to store non media_type Column {col.name}'
        dest_path = self._prepare_media_path(col)
        if isinstance(data, bytes):
            dest_path = self._save_binary_media_file(data, dest_path, format)
        elif isinstance(data, PIL.Image.Image):
            dest_path = self._save_pil_image_file(data, dest_path, format)
        else:
            raise ValueError(f'Unsupported media object type: {type(data)}')
        new_file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
        return dest_path, new_file_url

    def delete(self, tbl_id: UUID, tbl_version: Optional[int] = None) -> None:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version."""
        assert tbl_id is not None
        table_prefix = MediaPath.media_table_prefix(tbl_id)
        if tbl_version is None:
            # Remove the entire folder for this table id.
            path = self.__base_dir / table_prefix
            if path.exists():
                shutil.rmtree(path)
        else:
            # Remove only the elements for the specified tbl_version.
            paths = glob.glob(
                str(self.__base_dir / table_prefix) + f'/**/{table_prefix}_*_{tbl_version}_*', recursive=True
            )
            for p in paths:
                os.remove(p)

    def count(self, tbl_id: Optional[UUID], tbl_version: Optional[int] = None) -> int:
        """
        Return number of files for given tbl_id.
        """
        if tbl_id is None:
            paths = glob.glob(str(self.__base_dir / '*'), recursive=True)
        elif tbl_version is None:
            table_prefix = MediaPath.media_table_prefix(tbl_id)
            paths = glob.glob(str(self.__base_dir / table_prefix) + f'/**/{table_prefix}_*', recursive=True)
        else:
            table_prefix = MediaPath.media_table_prefix(tbl_id)
            paths = glob.glob(
                str(self.__base_dir / table_prefix) + f'/**/{table_prefix}_*_{tbl_version}_*', recursive=True
            )
        # Filter out directories, only count files
        return len([p for p in paths if not os.path.isdir(p)])

    def stats(self) -> list[tuple[UUID, int, int, int]]:
        paths = glob.glob(str(self.__base_dir) + '/**', recursive=True)
        # key: (tbl_id, col_id), value: (num_files, size)
        d: dict[tuple[UUID, int], list[int]] = defaultdict(lambda: [0, 0])
        for p in paths:
            if not os.path.isdir(p):
                matched = re.match(MediaPath.PATTERN, Path(p).name)
                assert matched is not None
                tbl_id, col_id = UUID(hex=matched[1]), int(matched[2])
                file_info = os.stat(p)
                t = d[tbl_id, col_id]
                t[0] += 1
                t[1] += file_info.st_size
        result = [(tbl_id, col_id, num_files, size) for (tbl_id, col_id), (num_files, size) in d.items()]
        result.sort(key=lambda e: e[3], reverse=True)
        return result


class TempStore:
    """
    A temporary store for files of data that are not yet persisted to their destination(s).
    A destination is typically either a MediaStore (local persisted files) or a cloud object store.

    The TempStore class has no internal state. It provides functionality to manage temporary files
    in the env.Env.get().tmp_dir directory.
    It reuses some of the MediaStore functionality to create unique file names and save objects.
    """

    @classmethod
    def _tmp_dir(cls) -> Path:
        """Returns the path to the temporary directory where files are stored."""
        from pixeltable import env

        return env.Env.get().tmp_dir

    @classmethod
    def count(cls, tbl_id: Optional[UUID] = None, tbl_version: Optional[int] = None) -> int:
        return MediaStore(cls._tmp_dir()).count(tbl_id, tbl_version)

    @classmethod
    def contains_path(cls, file_path: Path) -> bool:
        return MediaStore(cls._tmp_dir()).contains_path(file_path)

    @classmethod
    def resolve_url(cls, file_url: Optional[str]) -> Optional[Path]:
        return MediaStore(cls._tmp_dir()).resolve_url(file_url)

    @classmethod
    def save_media_object(cls, data: bytes | PIL.Image.Image, col: Column, format: Optional[str]) -> tuple[Path, str]:
        return MediaStore(cls._tmp_dir()).save_media_object(data, col, format)

    @classmethod
    def delete_media_file(cls, file_path: Path) -> None:
        """Delete a media object from the temporary store."""
        assert file_path is not None, 'Object path must be provided'
        assert file_path.exists(), f'Object path does not exist: {file_path}'
        assert cls.contains_path(file_path), f'Object path must be in the TempStore: {file_path}'
        file_path.unlink()
        _logger.debug(f'Media Storage: deleted {file_path}')

    @classmethod
    def create_path(cls, tbl_id: Optional[UUID] = None, extension: str = '') -> Path:
        """Return a new, unique Path located in the temporary store.
        If tbl_id is provided, the path name will be similar to a MediaStore path based on the tbl_id.
        If tbl_id is None, a random UUID will be used to create the path."""
        if tbl_id is not None:
            return MediaStore(cls._tmp_dir())._prepare_media_path_raw(tbl_id, 0, 0, extension)
        return cls._tmp_dir() / f'{uuid.uuid4()}{extension}'
