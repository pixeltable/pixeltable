from __future__ import annotations

import dataclasses
import re

from pixeltable import exceptions as excs

from .globals import is_valid_identifier

# pxt://<org>[:<db>][/<rest>] — org is a required slug, db an optional slug, rest the in-db path part
# (which may carry a trailing :version). The org:db colon lives in the netloc and never collides with
# the version colon, which is in rest.
_URI_RE = re.compile(r'^pxt://(?P<org>[^:/]+)(?::(?P<db>[^/]+))?(?:/(?P<rest>.*))?$')

# Pixeltable web URLs that denote the same resource as a pxt:// URI.
_URL_PREFIXES = (
    'https://www.pixeltable.com/t/',
    'http://www.pixeltable.com/t/',
    'https://pixeltable.com/t/',
    'http://pixeltable.com/t/',
)


@dataclasses.dataclass(frozen=True, slots=True, order=True, kw_only=True)
class Path:
    """A table/dir location, in the in-process catalog or in a remote/proxied catalog.

    Identifies the catalog it lives in (org/db) plus the path within that catalog (components) and an
    optional version. Org/db is None for local paths.

    Construct via parse() or from_components(), which apply context-specific rules.
    """

    org: str | None = None  # None => in-process catalog (catalog_uri ''); a slug => remote/proxied catalog
    db: str | None = None  # database within the org; always None when org is None, optional otherwise
    components: tuple[str, ...] = ()  # the empty tuple denotes the catalog root
    version: int | None = None

    def __post_init__(self) -> None:
        if self.db is not None and self.org is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_PATH, f'Path specifies a database ({self.db!r}) but no organization'
            )
        if self.org is not None and not is_valid_identifier(self.org, allow_hyphens=True):
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid organization name: {self.org!r}')
        if self.db is not None and not is_valid_identifier(self.db, allow_hyphens=True):
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid database name: {self.db!r}')
        if self.version is not None and self.version < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Version must be non-negative: {self.version}')
        # the root is the empty tuple; every component of a non-root path must be a valid identifier
        if not all(is_valid_identifier(c, allow_hyphens=True) for c in self.components):
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {"/".join(self.components)}')

    @classmethod
    def parse(cls, path: str, *, allow_empty_path: bool = False, allow_versioned_path: bool = False) -> Path:
        normalized = cls._normalize(path)

        org: str | None = None
        db: str | None = None
        if normalized.startswith('pxt://'):
            m = _URI_RE.match(normalized)
            if m is None:
                raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'URI must have an organization: {path!r}')
            org = m.group('org')
            db = m.group('db')
            path_part = m.group('rest') or ''
        else:
            path_part = normalized

        # Extract the trailing :version, if present.
        version: int | None = None
        if ':' in path_part:
            parts = path_part.split(':')
            if len(parts) != 2:
                raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {path}')
            path_part = parts[0]
            try:
                version = int(parts[1])
            except ValueError:
                raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {path}') from None

        # Split the in-db path part into components. Slash-separated is canonical;
        # dotted form is accepted only when no slashes are present (backward compatibility).
        components: tuple[str, ...]
        if '/' in path_part:
            components = tuple(path_part.split('/'))
        elif '.' in path_part:
            components = tuple(path_part.split('.'))
        else:
            components = (path_part,) if path_part else ()

        if len(components) == 0 and not allow_empty_path:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {path}')
        # component identifier validation is enforced by __post_init__ at construction
        if version is not None and not allow_versioned_path:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Versioned path not allowed here: {path}')

        return cls(org=org, db=db, components=components, version=version)

    @classmethod
    def from_components(
        cls, components: tuple[str, ...], *, version: int | None = None, org: str | None = None, db: str | None = None
    ) -> Path:
        return cls(org=org, db=db, components=tuple(components), version=version)

    @classmethod
    def is_pxt_uri(cls, s: str) -> bool:
        """Return True if the string is a pxt:// URI or a recognized Pixeltable web URL."""
        return s.startswith('pxt://') or any(s.startswith(p) for p in _URL_PREFIXES)

    @classmethod
    def _normalize(cls, s: str) -> str:
        """Rewrite a recognized Pixeltable web URL to its canonical pxt:// form; otherwise unchanged."""
        for prefix in _URL_PREFIXES:
            if s.startswith(prefix):
                return 'pxt://' + s[len(prefix) :]
        return s

    @property
    def is_local(self) -> bool:
        """True if this path lives in the in-process catalog (direct execution, no proxy)."""
        return self.org is None

    @property
    def catalog_uri(self) -> Path:
        """The catalog this path lives in, as a path (org/db, no components). ROOT_PATH for the in-process catalog."""
        return Path(org=self.org, db=self.db)

    @property
    def uri(self) -> str:
        """The catalog this path lives in, as a URI string. Empty string for the in-process catalog."""
        if self.is_local:
            return ''
        return f'pxt://{self.org}' if self.db is None else f'pxt://{self.org}:{self.db}'

    @property
    def len(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        return self.components[-1]

    @property
    def is_root(self) -> bool:
        return len(self.components) == 0

    @property
    def parent(self) -> Path:
        # the root (empty components) is its own parent: ()[:-1] == ()
        return dataclasses.replace(self, components=self.components[:-1], version=None)

    def append(self, name: str) -> Path:
        return dataclasses.replace(self, components=(*self.components, name), version=None)

    def is_ancestor(self, other: Path, is_parent: bool = False) -> bool:
        """
        True if self as an ancestor path of other.
        """
        if self.org != other.org or self.db != other.db:
            return False
        if self.len >= other.len or other.is_root:
            return False
        if self.is_root and (other.len == 1 or not is_parent):
            return True
        is_prefix = self.components == other.components[: self.len]
        return is_prefix and (self.len == (other.len - 1) or not is_parent)

    def ancestors(self) -> list[Path]:
        """
        Return all proper ancestors of this path in top-down order including root.
        If this path is for the root directory, which has no parent, then an empty list is returned.
        """
        if self.is_root:
            return []
        return [
            dataclasses.replace(self, components=self.components[:i], version=None) for i in range(len(self.components))
        ]

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        base = '/'.join(self.components)
        if self.version is not None:
            base = f'{base}:{self.version}'
        if self.org is None:
            return base
        return self.uri if base == '' else f'{self.uri}/{base}'


ROOT_PATH = Path()
