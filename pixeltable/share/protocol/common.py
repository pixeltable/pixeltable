from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, field_validator, model_validator

# Protocol version for replica operations. Used by both client and server
# to determine request/response format and maintain backward compatibility.
PROTOCOL_VERSION = 1


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


class PxtUri(BaseModel):
    """Pixeltable URI model for pxt:// URIs with validation and parsing."""

    uri: str  # The full URI string

    # Parsed components
    org: str  # Organization slug from the URI
    db: str | None  # Database slug from the URI (optional)
    path: str | None = None  # The table or directory path (None if using UUID)
    id: UUID | None = None  # The table UUID (None if using path)
    version: int | None = None  # Optional version number parsed from URI (format: identifier:<version>)

    @field_validator('version')
    @classmethod
    def validate_version(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError('Version must be a non-negative integer.')
        return v

    def __init__(self, uri: str | dict | None = None, **kwargs: Any) -> None:
        # Handle dict input directly (from JSON deserialization or explicit dict)
        if isinstance(uri, dict):
            # Dict input goes directly to Pydantic, which will call parse_uri
            kwargs.update(uri)
        elif uri is not None:
            # Validate that uri is a string when passed as positional argument
            if not isinstance(uri, str):
                raise ValueError(f'Invalid data type for PxtUri: expected str or dict, got {type(uri)}')
            kwargs['uri'] = uri
        super().__init__(**kwargs)

    @model_validator(mode='before')
    @classmethod
    def parse_uri(cls, data: Any) -> dict:
        # Handle case where data is already a string (from JSON deserialization)
        if isinstance(data, str):
            uri = data
        elif isinstance(data, dict):
            uri = data.get('uri')
            if uri is None:
                raise ValueError('URI must be provided in dict with "uri" key')
            if not isinstance(uri, str):
                raise ValueError(f'URI in dict must be a string, got {type(uri)}')
        else:
            raise ValueError(f'Invalid data type for PxtUri: expected str or dict, got {type(data)}')

        return {'uri': uri, **cls._parse_and_validate_uri(uri)}

    def __str__(self) -> str:
        """Return the URI string."""
        return self.uri

    @classmethod
    def _parse_and_validate_uri(cls, uri: str) -> dict:
        """Parse and validate a URI string, return parsed components."""
        if not uri.startswith('pxt://'):
            raise ValueError('URI must start with pxt://')

        parsed = urlparse(uri)
        if parsed.scheme != 'pxt':
            raise ValueError('URI must use pxt:// scheme')

        if not parsed.netloc:
            raise ValueError('URI must have an organization')

        # Parse netloc for org and optional db
        netloc_parts = parsed.netloc.split(':')
        org = netloc_parts[0]
        if not org:
            raise ValueError('URI must have an organization')

        db = netloc_parts[1] if len(netloc_parts) > 1 else None

        # Allow root path (/) as valid, but reject missing path
        if parsed.path is None:
            raise ValueError('URI must have a path')

        # Get path and remove leading slash (but keep empty string for root path)
        # path will be '/' for root directory or '/path/to/table' for regular paths
        path_part = parsed.path.lstrip('/') if parsed.path else ''

        # Handle version parsing (format: identifier:version)
        identifier, version = path_part, None
        if path_part and ':' in path_part:
            parts = path_part.rsplit(':', 1)
            if len(parts) == 2:
                try:
                    version_int = int(parts[1])
                except ValueError:
                    raise ValueError(f'Invalid table version {parts[1]!r} in uri: {uri}') from None
                else:
                    if version_int < 0:
                        raise ValueError('Version must be a non-negative integer.') from None
                    identifier, version = parts[0], version_int

        # Parse identifier into either a path string or UUID
        path: str | None = None
        id: UUID | None = None
        if identifier and is_valid_uuid(identifier):
            id = UUID(identifier)
        else:
            path = identifier or ''

        return {'org': org, 'db': db, 'path': path, 'id': id, 'version': version}

    @classmethod
    def from_components(
        cls,
        org: str,
        path: str | None = None,
        id: UUID | None = None,
        db: str | None = None,
        version: int | None = None,
    ) -> PxtUri:
        """Construct a PxtUri from its components."""
        if path is None and id is None:
            raise ValueError('Either path or id must be provided')
        if path is not None and id is not None:
            raise ValueError('Cannot specify both path and id')
        if version is not None and version < 0:
            raise ValueError('Version must be a non-negative integer.')

        # Build the URI string from components
        netloc = org if db is None else f'{org}:{db}'

        # Use path or UUID as identifier
        if id is not None:
            identifier = str(id)
        elif path is not None:
            # Path is already in URI format (slash-separated)
            identifier = path or ''
        else:
            identifier = ''

        path_part = f'{identifier}:{version}' if version is not None else identifier
        uri = f'pxt://{netloc}/{path_part}'
        return cls(uri=uri)


class RequestBaseModel(BaseModel, ABC):
    """Abstract base model for protocol requests that must have a PxtUri."""

    @abstractmethod
    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request. Must be implemented by subclasses."""
        pass
