import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pixeltable.exceptions as excs


@contextmanager
def transactional_directory(folder_path: Path, replace: bool = False) -> Generator[Path, Any, Any]:
    """
    Args:
        folder_path: path to the folder we want to create
        replace: if True, an existing folder at `folder_path` is replaced on commit instead of raising an error;
            it is removed only after the new contents have been written successfully

    Yields:
        A pathlib.Path to a hidden temporary folder, which can be used to accumulate changes.
        If everything succeeds, the changes are committed via an atomic move operation upon
        exiting the 'with' block (os.replace)
        If an exception occurred, no changes are visible in the original folder.

    Example:
        folder_path = pathlib.Path("path/to/folder")
        with transactional_folder(folder_path) as temp_folder:
            (temp_folder / "subfolder1").mkdir()
            (temp_folder / "subfolder2").mkdir()
    """
    if folder_path.exists() and not replace:
        raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Folder {folder_path} already exists')

    tmp_folder = folder_path.parent / f'.tmp_{folder_path.name}'
    # Remove the temporary folder if it already exists, eg if the previous run crashed
    shutil.rmtree(str(tmp_folder), ignore_errors=True)
    tmp_folder.mkdir(parents=True)
    yield tmp_folder
    # If everything succeeds, `commit' the changes by moving the temporary folder
    if replace and folder_path.exists():
        old_folder = folder_path.parent / f'.old_{folder_path.name}_{uuid.uuid4().hex}'
        folder_path.rename(old_folder)
        tmp_folder.rename(folder_path)
        shutil.rmtree(str(old_folder))
    else:
        tmp_folder.rename(folder_path)
