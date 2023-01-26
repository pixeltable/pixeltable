from typing import Optional
from pathlib import Path
import glob

from pixeltable.env import Env


def is_computed_img_path(path: str) -> bool:
    try:
        _ = Path(path).relative_to(Env.get().img_dir)
        return True
    except ValueError:
        return False

def get_computed_img_path(tbl_id: int, col_id: int, version: int, rowid: int) -> str:
    return Env.get().img_dir / f'img_{tbl_id}_{col_id}_{version}_{rowid}.jpg'

def get_extracted_frame_path(tbl_id: int, video_col_id: int, version: int, offset: int) -> str:
    return Env.get().img_dir / f'frame_{tbl_id}_{video_col_id}_{version}_{offset}'

def computed_imgs(
        tbl_id: Optional[int] = None, col_id: Optional[int] = None, version: Optional[int] = None) -> int:
    path = f'{Env.get().img_dir}/img_'
    path += f'{tbl_id}_' if tbl_id is not None else '*_'
    path += f'{col_id}_' if col_id is not None else '*_'
    path += f'{version}_' if version is not None else '*_'
    path += '*'
    names = glob.glob(path)
    return names

def computed_img_count(
        tbl_id: Optional[int] = None, col_id: Optional[int] = None, version: Optional[int] = None) -> int:
    return len(computed_imgs(tbl_id=tbl_id, col_id=col_id, version=version))

def extracted_frames(tbl_id: Optional[int] = None, version: Optional[int] = None) -> int:
    path = f'{Env.get().img_dir}/frame_'
    path += f'{tbl_id}_' if tbl_id is not None else '*_'
    path += '*_'  # video_col_id
    path += f'{version}_' if version is not None else '*_'
    path += '*_'  # offset
    path += '*'  # running frame index
    names = glob.glob(path)
    return names

def extracted_frame_count(tbl_id: Optional[int] = None, version: Optional[int] = None) -> int:
    return len(extracted_frames(tbl_id, version))
