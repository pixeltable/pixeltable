import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
import threading
import logging

_logger = logging.getLogger('pixeltable')

def print_perf_counter_delta(delta: float) -> str:
    """Prints a performance counter delta in a human-readable format.

    Args:
        delta: delta in seconds

    Returns:
        Human-readable string
    """
    if delta < 1e-6:
        return f'{delta * 1e9:.2f} ns'
    elif delta < 1e-3:
        return f'{delta * 1e6:.2f} us'
    elif delta < 1:
        return f'{delta * 1e3:.2f} ms'
    else:
        return f'{delta:.2f} s'


def sha256sum(path: Path | str) -> str:
    """
    Compute the SHA256 hash of a file.
    """
    if isinstance(path, str):
        path = Path(path)

    h = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(h.block_size):
            h.update(chunk)

    return h.hexdigest()


def parse_local_file_path(file_or_url: str) -> Path | None:
    """
    Parses a string that may be either a URL or a local file path.

    If the string is a local file path or a file-scheme URL (file://), then a Path object will be returned.
    Otherwise, None will be returned.
    """
    parsed = urllib.parse.urlparse(file_or_url)
    if len(parsed.scheme) <= 1:
        # We're using `urlparse` to help distinguish file paths from URLs. If there is no scheme, then it's a file path.
        # If there's a single-character scheme, we also interpret this as a file path; this insures that drive letters
        # on Windows pathnames are correctly handled.
        return Path(file_or_url).absolute()
    elif parsed.scheme == 'file':
        return Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
    else:
        return None


def fetch_url(url: str, allow_local_file: bool = False) -> Path:
    """
    Fetches a remote URL into the TempStore and returns its path.

    If `allow_local_file` is True, and the URL is a file:// URL or a local file path, then the local path is returned
    directly without copying. If `allow_local_file` is False, then an AssertionError is raised.
    """
    from .local_store import TempStore
    from .object_stores import ObjectOps

    _logger.debug(f'fetching url={url} thread_name={threading.current_thread().name}')
    parsed = urllib.parse.urlparse(url)

    # Sanity check to ensure we're not being passed a Windows filename and misinterpreting it as a URL
    assert len(parsed.scheme) > 1
    assert allow_local_file or parsed.scheme != 'file'

    path: Path | None = None
    if parsed.path:
        path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed.path)))
    if parsed.scheme == 'file':
        assert path is not None
        return path

    # preserve the file extension, if there is one
    tmp_path = TempStore.create_path(extension=(path.suffix if path else ''))

    _logger.debug(f'Downloading {url} to {tmp_path}')
    ObjectOps.copy_object_to_local_file(url, tmp_path)
    _logger.debug(f'Downloaded {url} to {tmp_path}')

    return tmp_path
