from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, model_validator


class StorageDestination(str, Enum):
    """Storage destination types for table snapshots."""

    S3 = 's3'
    R2 = 'r2'
    GCS = 'gcs'


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
    org_slug: str  # Organization slug from the URI
    db_slug: Optional[str]  # Database slug from the URI (optional)
    table_identifier: str  # The table identifier (path or UUID)
    is_uuid: bool  # True if table_identifier is a UUID, False if it's a path

    def __init__(self, uri: str | None = None, **kwargs: Any) -> None:
        if uri is not None:
            # If uri is provided as positional argument, use it
            kwargs['uri'] = uri
        super().__init__(**kwargs)

    @model_validator(mode='before')
    @classmethod
    def parse_uri(cls, data: dict) -> dict:  # Type as dict directly
        uri = data['uri']  # KeyError if missing is correct behavior
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
            raise ValueError('URI must have a hostname (org_slug)')

        # Parse netloc for org_slug and optional db_slug
        netloc_parts = parsed.netloc.split(':')
        org_slug = netloc_parts[0]
        if not org_slug:
            raise ValueError('URI must have an org_slug')

        db_slug = netloc_parts[1] if len(netloc_parts) > 1 else None

        # Allow root path (/) as valid, but reject missing path
        if parsed.path is None or parsed.path == '':
            raise ValueError('URI must have a path')

        # Get path and remove leading slash (but keep empty string for root path)
        table_identifier = parsed.path.lstrip('/') if parsed.path else ''

        # Determine if the path part is a UUID
        is_uuid = is_valid_uuid(table_identifier) if table_identifier else False

        return {'org_slug': org_slug, 'db_slug': db_slug, 'table_identifier': table_identifier, 'is_uuid': is_uuid}

    @classmethod
    def from_components(cls, org_slug: str, table_identifier: str, db_slug: Optional[str] = None) -> PxtUri:
        """Construct a PxtUri from its components."""
        # Build the URI string from components
        netloc = org_slug if db_slug is None else f'{org_slug}:{db_slug}'
        uri = f'pxt://{netloc}/{table_identifier}'
        return cls(uri=uri)


class RequestBaseModel(BaseModel, ABC):
    """Abstract base model for protocol requests that must have a PxtUri."""

    @abstractmethod
    def get_pxt_uri(self) -> PxtUri:
        """Get the PxtUri from this request. Must be implemented by subclasses."""
        pass
