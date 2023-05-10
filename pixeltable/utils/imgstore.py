import time
import glob
import os
import re
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from collections import defaultdict

from pixeltable.env import Env


class ImageStore:
    """
    Utilities to manage images stored in Env.img_dir
    """
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_\d+.jpg')  # tbl_id, col_id, version, rowid

    @classmethod
    def get_path(cls, tbl_id: int, col_id: int, version: int, rowid: int) -> Path:
        return Env.get().img_dir / f'{tbl_id}_{col_id}_{version}_{rowid}.jpg'

    @classmethod
    def add(cls, tbl_id: int, col_id: int, version: int, rowid: int, file_path: str) -> Path:
        """
        Move file to store
        """
        new_path = cls.get_path(tbl_id, col_id, version, rowid)
        os.rename(file_path, str(new_path))
        return new_path

    @classmethod
    def delete(cls, tbl_id: int, v_min: Optional[int] = None) -> None:
        """
        Delete all images belonging to tbl_id. If v_min is given, only deletes images with version >= v_min
        """
        assert tbl_id is not None
        paths = glob.glob(str(Env.get().img_dir / f'{tbl_id}_*_*_*.jpg'))
        if v_min is not None:
            paths = [
                Env.get().img_dir / matched[0]
                for matched in [re.match(cls.pattern, Path(p).name) for p in paths] if int(matched[3]) >= v_min
            ]
        for p in paths:
            os.remove(p)

    @classmethod
    def count(cls, tbl_id: int) -> int:
        """
        Return number of images for given tbl_id.
        """
        paths = glob.glob(str(Env.get().img_dir / f'{tbl_id}_*_*_*.jpg'))
        return len(paths)

    @classmethod
    def stats(cls) -> List[Tuple[int, int, int, int]]:
        paths = glob.glob(str(Env.get().img_dir / '*.jpg'))
        d: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])
        for p in paths:
            matched = re.match(cls.pattern, Path(p).name)
            assert matched is not None
            tbl_id, col_id = int(matched[1]), int(matched[2])
            file_info = os.stat(p)
            t = d[(tbl_id, col_id)]
            t[0] += 1
            t[1] += file_info.st_size
        result = [(tbl_id, col_id, num_files, size) for (tbl_id, col_id), (num_files, size) in d.items()]
        result.sort(key=lambda e: e[3], reverse=True)
        return result
