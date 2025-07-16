import glob
import os
import re
import shutil
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import UUID

import PIL.Image

from pixeltable import env


class MediaStore:
    """
    Utilities to manage media files stored in Env.media_dir

    Media file names are a composite of: table id, column id, tbl_version, new uuid:
    the table id/column id/tbl_version are redundant but useful for identifying all files for a table
    or all files created for a particular version of a table
    """

    pattern = re.compile(r'([0-9a-fA-F]+)_(\d+)_(\d+)_([0-9a-fA-F]+)')  # tbl_id, col_id, version, uuid

    @classmethod
    def _media_dir(cls) -> Path:
        """Returns the media directory path."""
        return env.Env.get().media_dir

    @classmethod
    def _tmp_dir(cls) -> Path:
        """Returns the temporary directory path."""
        return env.Env.get().tmp_dir

    @classmethod
    def _prepare_media_path(cls, tbl_id: UUID, col_id: int, tbl_version: int, ext: Optional[str] = None) -> Path:
        """
        Construct a new, unique Path name for a persisted media file, and create the parent directory
        for the new Path if it does not already exist. The Path will reside in
        the environment's media_dir.
        """
        id_hex = uuid.uuid4().hex
        parent = cls._media_dir() / tbl_id.hex / id_hex[:2] / id_hex[:4]
        parent.mkdir(parents=True, exist_ok=True)
        return parent / f'{tbl_id.hex}_{col_id}_{tbl_version}_{id_hex}{ext or ""}'

    @classmethod
    def is_tmp_url(cls, file_url: Optional[str]) -> tuple[bool, Path]:
        """Check if the given url is a tmp file.

        Args:
            file_url: URL of the tmp media file to check

        Returns:
            True if the file_url is a tmp file, False otherwise
        """
        if file_url is None:
            return False, Path()
        assert isinstance(file_url, str), type(file_url)
        parsed = urllib.parse.urlparse(file_url)
        # We should never be passed a local file path here. The "len > 1" ensures that Windows
        # file paths aren't mistaken for URLs with a single-character scheme.
        assert len(parsed.scheme) > 1, file_url
        if parsed.scheme != 'file':
            # remote url
            return False, Path()
        src_path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        pxt_tmp_dir = str(cls._tmp_dir())
        return src_path.startswith(pxt_tmp_dir), Path(src_path)

    @classmethod
    def move_tmp_media_file(cls, file_url: Optional[str], tbl_id: UUID, col_id: int, tbl_version: int) -> Optional[str]:
        """If the given url is a tmp file, move it into the MediaStore, and return new url
        If it is not a tmp file in the tmp_dir, return the original url.

        Args:
            file_url: URL of the tmp media file to move
            tbl_id: Table ID to associate with the media file
            col_id: Column ID to associate with the media file
            tbl_version: tbl_version number to associate with the media file

        Returns:
            URL of the media final location of the file
        """
        is_tmp, src_path = cls.is_tmp_url(file_url)
        if not is_tmp:
            return file_url
        new_file_url = cls.relocate_local_media_file(src_path, tbl_id, col_id, tbl_version)
        return new_file_url

    @classmethod
    def relocate_local_media_file(cls, src_path: Path, tbl_id: UUID, col_id: int, tbl_version: int) -> str:
        """Relocate a local file to the MediaStore, and return its new URL"""
        dest_path = cls._prepare_media_path(tbl_id, col_id, tbl_version, ext=src_path.suffix)
        src_path.rename(dest_path)
        return urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))

    @classmethod
    def save_media_object(
        cls, data: bytes | PIL.Image.Image, tbl_id: UUID, col_id: int, tbl_version: int
    ) -> tuple[Path, str]:
        """Save a media binary data to a file in the MediaStore"""
        dest_path = cls._prepare_media_path(tbl_id, col_id, tbl_version)
        if isinstance(data, bytes):
            dest_path = cls._save_media_file(data, dest_path)
        elif isinstance(data, PIL.Image.Image):
            dest_path = cls._save_image_file(data, dest_path)
        else:
            raise ValueError(f'Unsupported media object type: {type(data)}')
        url = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
        return dest_path, url

    @classmethod
    def _save_media_file(cls, file_data: bytes, dest_path: Path) -> Path:
        """Save a media binary data to a file in the MediaStore"""
        assert isinstance(file_data, bytes)
        with open(dest_path, 'wb') as f:
            f.write(file_data)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage
        return dest_path

    @classmethod
    def _save_image_file(cls, image: PIL.Image.Image, dest_path: Path) -> Path:
        """Save a PIL Image to a file in the MediaStore"""
        assert isinstance(image, PIL.Image.Image)
        # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
        # In that case, use WebP instead.
        format = 'webp' if image.has_transparency_data else 'jpeg'
        t_path = str(dest_path)
        if not t_path.endswith(f'.{format}'):
            t_path += f'.{format}'
        dest_path = Path(t_path)

        with open(dest_path, 'wb') as f:
            image.save(f, format=format)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage
        return dest_path

    @classmethod
    def delete(cls, tbl_id: UUID, tbl_version: Optional[int] = None) -> None:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version."""
        assert tbl_id is not None
        if tbl_version is None:
            # Remove the entire folder for this table id.
            path = cls._media_dir() / tbl_id.hex
            if path.exists():
                shutil.rmtree(path)
        else:
            # Remove only the elements for the specified tbl_version.
            paths = glob.glob(
                str(cls._media_dir() / tbl_id.hex) + f'/**/{tbl_id.hex}_*_{tbl_version}_*', recursive=True
            )
            for p in paths:
                os.remove(p)

    @classmethod
    def count(cls, tbl_id: UUID) -> int:
        """
        Return number of files for given tbl_id.
        """
        paths = glob.glob(str(cls._media_dir() / tbl_id.hex) + f'/**/{tbl_id.hex}_*', recursive=True)
        return len(paths)

    @classmethod
    def stats(cls) -> list[tuple[UUID, int, int, int]]:
        paths = glob.glob(str(cls._media_dir()) + '/**', recursive=True)
        # key: (tbl_id, col_id), value: (num_files, size)
        d: dict[tuple[UUID, int], list[int]] = defaultdict(lambda: [0, 0])
        for p in paths:
            if not os.path.isdir(p):
                matched = re.match(cls.pattern, Path(p).name)
                assert matched is not None
                tbl_id, col_id = UUID(hex=matched[1]), int(matched[2])
                file_info = os.stat(p)
                t = d[tbl_id, col_id]
                t[0] += 1
                t[1] += file_info.st_size
        result = [(tbl_id, col_id, num_files, size) for (tbl_id, col_id), (num_files, size) in d.items()]
        result.sort(key=lambda e: e[3], reverse=True)
        return result
