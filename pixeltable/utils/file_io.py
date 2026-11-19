import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator


@contextmanager
def atomic_write(file_path: Path, **open_kwargs: Any) -> Iterator[IO[str]]:
    """Open a tempfile alongside `file_path`; rename it onto `file_path` on clean exit, delete on error.

    `open_kwargs` are forwarded to `os.fdopen` (e.g. `mode`, `encoding`, `newline`).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(file_path.parent), prefix=f'.{file_path.name}.', suffix='.tmp')
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, **open_kwargs) as f:
            yield f
        os.replace(tmp_path, file_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
