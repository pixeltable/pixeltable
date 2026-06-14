from __future__ import annotations

import re
from typing import NamedTuple

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


class Path(NamedTuple):
    """A table/dir location, in the in-process catalog or in a remote/proxied catalog.

    Identifies the catalog it lives in (org/db) plus the path within that catalog (components) and an
    optional version.

    org is None for the in-process catalog: direct execution against the user's own Postgres, no
    proxy, catalog_uri ''. When org is set, the catalog is reached over RPC; org[:db] names it. The
    org slug 'local' is reserved for a local proxy daemon (e.g. pxt://local:testdb), which is still a
    proxied catalog, distinct from the in-process one.

    Construct via parse() or from_components(), never positionally.
    """

    org: str | None = None  # None => in-process catalog (catalog_uri ''); a slug => remote/proxied catalog
    db: str | None = None  # database within the org; always None when org is None, optional otherwise
    components: tuple[str, ...] = ('',)  # ('',) denotes the catalog root
    version: int | None = None

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
            if version < 0:
                raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Version must be non-negative: {path!r}')

        # Split the in-db path part into components (dotted form accepted for backward compatibility).
        components: tuple[str, ...]
        if '.' in path_part:
            components = tuple(path_part.split('.'))
        elif '/' in path_part:
            components = tuple(path_part.split('/'))
        else:
            components = (path_part,) if path_part else ('',)

        if components == ('',) and not allow_empty_path:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {path}')
        if components != ('',) and not all(is_valid_identifier(c, allow_hyphens=True) for c in components):
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Invalid path: {path}')
        if version is not None and not allow_versioned_path:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Versioned path not allowed here: {path}')

        assert len(components) > 0
        return cls(org=org, db=db, components=components, version=version)

    @classmethod
    def from_components(
        cls, components: tuple[str, ...], *, version: int | None = None, org: str | None = None, db: str | None = None
    ) -> Path:
        if version is not None and version < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_PATH, f'Version must be non-negative: {version}')
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
    def catalog_uri(self) -> str:
        """The catalog this path lives in, as a URI string. Empty string for the in-process catalog."""
        if self.is_local:
            return ''
        return f'pxt://{self.org}' if self.db is None else f'pxt://{self.org}:{self.db}'

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
    def parent(self) -> Path:
        if len(self.components) == 1:
            # Includes the case of the root path, which is its own parent.
            return self._replace(components=('',), version=None)
        return self._replace(components=self.components[:-1], version=None)

    def append(self, name: str) -> Path:
        components = (name,) if self.is_root else (*self.components, name)
        return self._replace(components=components, version=None)

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
            self._replace(components=self.components[:i] if i > 0 else ('',), version=None)
            for i in range(len(self.components))
        ]

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        base = '/'.join(self.components)
        if self.version is not None:
            base = f'{base}:{self.version}'
        if self.org is None:
            return base
        return f'{self.catalog_uri}/{base}'


ROOT_PATH = Path()
