from __future__ import annotations

import logging
from typing import NamedTuple

from pixeltable import exceptions as excs

from .globals import is_valid_identifier

_logger = logging.getLogger('pixeltable')


class Path(NamedTuple):
    components: list[str]
    version: int | None = None

    @classmethod
    def parse(
        cls,
        path: str,
        allow_empty_path: bool = False,
        allow_system_path: bool = False,
        allow_versioned_path: bool = False,
    ) -> Path:
        components: list[str]
        version: int | None
        if ':' in path:
            parts = path.split(':')
            if len(parts) != 2:
                raise excs.Error(f'Invalid path: {path}')
            try:
                components = parts[0].split('.')
                version = int(parts[1])
            except ValueError:
                raise excs.Error(f'Invalid path: {path}') from None
        else:
            components = path.split('.')
            version = None

        if components == [''] and not allow_empty_path:
            raise excs.Error(f'Invalid path: {path}')

        if components != [''] and not all(
            is_valid_identifier(c, allow_system_identifiers=allow_system_path, allow_hyphens=True) for c in components
        ):
            raise excs.Error(f'Invalid path: {path}')

        if version is not None and not allow_versioned_path:
            raise excs.Error(f'Versioned path not allowed here: {path}')

        assert len(components) > 0
        return Path(components, version)

    @property
    def len(self) -> int:
        return 0 if self.is_root else len(self.components)

    @property
    def name(self) -> str:
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
            return ROOT_PATH  # Includes the case of the root path, which is its own parent.
        else:
            return Path(self.components[:-1])

    def append(self, name: str) -> Path:
        if self.is_root:
            return Path([name])
        else:
            return Path([*self.components, name])

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

    def ancestors(self) -> list[Path]:
        """
        Return all proper ancestors of this path in top-down order including root.
        If this path is for the root directory, which has no parent, then None is returned.
        """
        if self.is_root:
            return []
        else:
            return [Path(self.components[:i]) if i > 0 else ROOT_PATH for i in range(len(self.components))]

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        base = '.'.join(self.components)
        if self.version is not None:
            return f'{base}:{self.version}'
        else:
            return base

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Path) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


ROOT_PATH = Path([''])
