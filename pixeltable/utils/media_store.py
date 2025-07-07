import glob
import os
import re
import shutil
import urllib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import UUID

from pixeltable.env import Env


class MediaStore:
    """
    Utilities to manage media files stored in Env.media_dir

    Media file names are a composite of: table id, column id, version, uuid:
    the table id/column id/version are redundant but useful for identifying all files for a table
    or all files created for a particular version of a table
    """

    pattern = re.compile(r'([0-9a-fA-F]+)_(\d+)_(\d+)_([0-9a-fA-F]+)')  # tbl_id, col_id, version, uuid

    @classmethod
    def prepare_media_path(cls, tbl_id: UUID, col_id: int, version: int, ext: Optional[str] = None) -> Path:
        """
        Construct a new, unique Path name for a persisted media file, and create the parent directory
        for the new Path if it does not already exist. The Path will reside in
        the environment's media_dir.
        """
        id_hex = uuid.uuid4().hex
        parent = Env.get().media_dir / tbl_id.hex / id_hex[:2] / id_hex[:4]
        parent.mkdir(parents=True, exist_ok=True)
        return parent / f'{tbl_id.hex}_{col_id}_{version}_{id_hex}{ext or ""}'

    @classmethod
    def move_tmp_media_file(cls, file_url: Optional[str], tbl_id: UUID, col_id: int, v_min: int) -> Optional[str]:
        """Move a tmp media file with given url into the MediaStore, and return new url
        If it is not a tmp file in the tmp_dir, return the original url.

        Args:
            file_url: URL of the tmp media file to move
            tbl_id: Table ID to associate with the media file
            col_id: Column ID to associate with the media file
            v_min: Version number to associate with the media file

        Returns:
            URL of the media final location of the file
        """
        if file_url is None:
            return None
        assert isinstance(file_url, str), type(file_url)
        pxt_tmp_dir = str(Env.get().tmp_dir)
        parsed = urllib.parse.urlparse(file_url)
        # We should never be passed a local file path here. The "len > 1" ensures that Windows
        # file paths aren't mistaken for URLs with a single-character scheme.
        assert len(parsed.scheme) > 1, file_url
        if parsed.scheme != 'file':
            # remote url
            return file_url
        file_path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        if not file_path.startswith(pxt_tmp_dir):
            # not a tmp file
            return file_url
        new_file_url = cls.relocate_local_media_file(Path(file_path), tbl_id, col_id, v_min)
        return new_file_url

    @classmethod
    def relocate_local_media_file(cls, src_path: Path, tbl_id: UUID, col_id: int, tbl_version: int) -> str:
        dest_path = MediaStore.prepare_media_path(tbl_id, col_id, tbl_version, ext=src_path.suffix)
        src_path.rename(dest_path)
        return urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))

    @classmethod
    def delete(cls, tbl_id: UUID, version: Optional[int] = None) -> None:
        """Delete all files belonging to tbl_id. If version is not None, delete
        only those files belonging to the specified version."""
        assert tbl_id is not None
        if version is None:
            # Remove the entire folder for this table id.
            path = Env.get().media_dir / tbl_id.hex
            if path.exists():
                shutil.rmtree(path)
        else:
            # Remove only the elements for the specified version.
            paths = glob.glob(str(Env.get().media_dir / tbl_id.hex) + f'/**/{tbl_id.hex}_*_{version}_*', recursive=True)
            for p in paths:
                os.remove(p)

    @classmethod
    def count(cls, tbl_id: UUID) -> int:
        """
        Return number of files for given tbl_id.
        """
        paths = glob.glob(str(Env.get().media_dir / tbl_id.hex) + f'/**/{tbl_id.hex}_*', recursive=True)
        return len(paths)

    @classmethod
    def stats(cls) -> list[tuple[UUID, int, int, int]]:
        paths = glob.glob(str(Env.get().media_dir) + '/**', recursive=True)
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
