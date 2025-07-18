from __future__ import annotations

import logging
from typing import Iterator, Optional

from pixeltable import exceptions as excs

from .globals import is_valid_identifier

_logger = logging.getLogger('pixeltable')


class Path:
    components: list[str]
    version: Optional[int]

    def __init__(
        self,
        path: str,
        empty_is_valid: bool = False,
        allow_system_paths: bool = False,
        allow_versioned_path: bool = False,
    ):
        if ':' in path:
            parts = path.split(':')
            if len(parts) != 2:
                raise excs.Error(f'Invalid path: {path}')
            try:
                self.components = parts[0].split('.')
                self.version = int(parts[1])
            except ValueError:
                raise excs.Error(f'Invalid path: {path}') from None
        else:
            self.components = path.split('.')
            self.version = None

        if (self.is_root and not empty_is_valid) or not (
            self.is_root or all(is_valid_identifier(c, allow_system_paths) for c in self.components)
        ):
            raise excs.Error(f'Invalid path: {path}')

        if not allow_versioned_path and self.version is not None:
            raise excs.Error(f'Versioned path not allowed here: {path}')

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
    def is_versioned(self) -> bool:
        return self.version is not None

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
            for i in range(len(self.components)):
                yield Path('.'.join(self.components[0:i]), empty_is_valid=True)

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        base_path = '.'.join(self.components)
        if self.version is not None:
            return f'{base_path}:{self.version}'
        else:
            return base_path

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Path) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: Path) -> bool:
        return str(self) < str(other)
