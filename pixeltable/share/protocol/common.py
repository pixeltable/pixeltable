import re
import uuid
from enum import Enum
from typing import Any, NamedTuple, Optional

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

# Regex to handle both table paths and UUIDs
PATTERN_PXT_URI = re.compile(r'^pxt://([a-zA-Z0-9-]+)(?::([a-zA-Z0-9-]+))?(/.*)?$')


class StorageDestination(str, Enum):
    """Storage destination types for table snapshots."""

    S3 = 's3'
    R2 = 'r2'
    GCS = 'gcs'


class PathUriComponents(NamedTuple):
    """Components of a parsed PathUri."""

    org_slug: str
    db_slug: Optional[str]
    path_or_uuid: str
    is_uuid: bool  # New field to indicate if path_or_uuid is a UUID


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def validate_pxt_uri(v: str) -> str:
    """Validate pxt:// URI format with support for both paths and UUIDs."""
    if not v.startswith('pxt://'):
        raise ValueError('URI must start with pxt://')

    match = PATTERN_PXT_URI.match(v)
    if not match:
        raise ValueError('Invalid URI format. Expected pxt://<org_slug>[:<db_slug>]/<table_path_or_uuid>')

    org_slug, _, path_part = match.groups()

    if not org_slug:
        raise ValueError('URI must have an org_slug')

    if path_part:
        # Remove leading slash and validate the path/UUID part
        clean_path = path_part.lstrip('/')
        if not clean_path:
            raise ValueError('URI must have a path or UUID after the slash')

        # Both hierarchical paths and UUIDs are valid
        # UUID validation is optional here - we'll identify it in parsing

    return v


class PathUri(str):
    """Custom URI type for pxt:// URIs with validation."""

    def __new__(cls, value: str) -> 'PathUri':
        validate_pxt_uri(value)
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.with_info_after_validator_function(
            cls._validate, core_schema.str_schema(), serialization=core_schema.to_string_ser_schema()
        )

    @classmethod
    def _validate(cls, v: str, info: Any) -> 'PathUri':
        if isinstance(v, str):
            return cls(v)
        raise ValueError(f'Expected string, got {type(v)}')

    def is_uuid(self) -> bool:
        """Check if this URI uses UUID format."""
        return PathUriParser.is_uuid_uri(str(self))

    def is_path(self) -> bool:
        """Check if this URI uses hierarchical path format."""
        return not self.is_uuid()

    def get_uuid(self) -> Optional[str]:
        """Get table UUID if this is a UUID URI, None otherwise."""
        return PathUriParser.get_table_uuid(str(self))

    def get_path(self) -> Optional[str]:
        """Get table path if this is a hierarchical URI, None otherwise."""
        return PathUriParser.get_table_path(str(self))

    def get_org_slug(self) -> str:
        """Extract the org_slug from this URI."""
        return PathUriParser.parse(str(self)).org_slug

    def get_db_slug(self) -> Optional[str]:
        """Extract the db_slug from this URI."""
        return PathUriParser.parse(str(self)).db_slug

    def get_components(self) -> PathUriComponents:
        """Get all parsed components of this URI."""
        return PathUriParser.parse(str(self))

    def get_identifier(self) -> str:
        """Get the table identifier (either UUID or path)."""
        components = self.get_components()
        return components.path_or_uuid


class PathUriParser:
    """Helper class to parse PathUri strings into components."""

    @staticmethod
    def parse(uri: str) -> PathUriComponents:
        """Parse a PathUri string into its components."""
        match = PATTERN_PXT_URI.match(uri)
        if not match:
            raise ValueError('Invalid URI format. Expected pxt://<org_slug>[:<db_slug>]/<table_path_or_uuid>')

        org_slug, db_slug, path_part = match.groups()
        path_or_uuid = path_part.lstrip('/') if path_part else ''

        # Safely strip whitespace, handling None values
        org_slug = org_slug.strip() if org_slug else ''
        db_slug = db_slug.strip() if db_slug else None

        # Determine if the path part is a UUID
        is_uuid = is_valid_uuid(path_or_uuid) if path_or_uuid else False

        return PathUriComponents(org_slug, db_slug, path_or_uuid, is_uuid)

    @staticmethod
    def get_table_path(uri: str) -> Optional[str]:
        """Get table path if this is a hierarchical URI, None if it's a UUID."""
        components = PathUriParser.parse(uri)
        return components.path_or_uuid if not components.is_uuid else None

    @staticmethod
    def get_table_uuid(uri: str) -> Optional[str]:
        """Get table UUID if this is a UUID URI, None if it's a hierarchical path."""
        components = PathUriParser.parse(uri)
        return components.path_or_uuid if components.is_uuid else None

    @staticmethod
    def is_uuid_uri(uri: str) -> bool:
        """Check if this URI uses UUID format."""
        return PathUriParser.parse(uri).is_uuid


class PathUriRequestModel(BaseModel):
    """Base model for requests that need path URI for slug identification."""

    def get_org_slug(self) -> Optional[str]:
        """Extract the org_slug from a table_uri if present."""
        if hasattr(self, 'table_uri'):
            return PathUriParser.parse(self.table_uri).org_slug
        if hasattr(self, 'dir_uri'):
            return PathUriParser.parse(self.dir_uri).org_slug
        return None

    def get_db_slug(self) -> Optional[str]:
        """Extract the db_slug from a table_uri if present."""
        if hasattr(self, 'table_uri'):
            db_slug = PathUriParser.parse(self.table_uri).db_slug
            return db_slug if db_slug else None
        if hasattr(self, 'dir_uri'):
            db_slug = PathUriParser.parse(self.dir_uri).db_slug
            return db_slug if db_slug else None
        return None

    def get_table_path(self) -> Optional[str]:
        """Extract the table path if this is a hierarchical URI."""
        if hasattr(self, 'table_uri'):
            return PathUriParser.get_table_path(self.table_uri)
        return None

    def get_table_uuid(self) -> Optional[str]:
        """Extract the table UUID if this is a UUID URI."""
        if hasattr(self, 'table_uri'):
            return PathUriParser.get_table_uuid(self.table_uri)
        return None

    def is_uuid_uri(self) -> bool:
        """Check if the table_uri uses UUID format."""
        if hasattr(self, 'table_uri'):
            return PathUriParser.is_uuid_uri(self.table_uri)
        return False
