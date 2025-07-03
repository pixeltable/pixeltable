from __future__ import annotations

import logging
from typing import Iterator

from pixeltable import exceptions as excs

from .globals import is_valid_path

_logger = logging.getLogger('pixeltable')


class Path:
    def __init__(self, path: str, empty_is_valid: bool = False, allow_system_paths: bool = False):
        if not is_valid_path(path, empty_is_valid, allow_system_paths):
            raise excs.Error(f"Invalid path format: '{path}'")
        self.components = path.split('.')

    @property
    def len(self) -> int:
        return 0 if self.is_root else len(self.components)

    @property
    def name(self) -> str:
        assert len(self.components) > 0
        return self.components[-1]

    @property
    def is_root(self) -> bool:
        return not self.components[0]

    @property
    def is_system_path(self) -> bool:
        return self.components[0].startswith('_')

    @property
    def parent(self) -> Path:
        if len(self.components) == 1:
            if self.is_root:
                return self
            else:
                return Path('', empty_is_valid=True, allow_system_paths=True)
        else:
            return Path('.'.join(self.components[:-1]), allow_system_paths=True)

    def append(self, name: str) -> Path:
        if self.is_root:
            return Path(name, allow_system_paths=True)
        else:
            return Path(f'{self}.{name}', allow_system_paths=True)

    def is_ancestor(self, other: Path, is_parent: bool = False) -> bool:
        """
        True if self as an ancestor path of other.
        """
        if self.len >= other.len or other.is_root:
            return False
        if self.is_root and (other.len == 1 or not is_parent):
            return True
        is_prefix = self.components == other.components[: self.len]
        return is_prefix and (self.len == (other.len - 1) or not is_parent)

    def ancestors(self) -> Iterator[Path]:
        """
        Return all ancestors of this path in top-down order including root.
        If this path is for the root directory, which has no parent, then None is returned.
        """
        if self.is_root:
            return
        else:
            for i in range(0, len(self.components)):
                yield Path('.'.join(self.components[0:i]), empty_is_valid=True)

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        return '.'.join(self.components)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Path) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: Path) -> bool:
        return str(self) < str(other)
