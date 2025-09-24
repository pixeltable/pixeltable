"""Dataclass field documentation section generator."""

import dataclasses
from typing import Any
from section_base import SectionBase


class DataclassSection(SectionBase):
    """Generate documentation sections for dataclass fields."""

    def can_handle(self, obj: Any) -> bool:
        """Check if the object is a dataclass."""
        return dataclasses.is_dataclass(obj)

    def generate_section(self, obj: Any, name: str) -> str:
        """Generate dataclass fields documentation section.

        Args:
            obj: The dataclass to document
            name: The name of the dataclass

        Returns:
            Markdown string for the fields section
        """
        fields = dataclasses.fields(obj)
        if not fields:
            return ""

        content = "## Attributes\n\n"

        for field in fields:
            content += f"### `{field.name}`\n\n"

            # Format type
            type_str = self._format_type(field.type)
            content += f"**Type:** `{type_str}`\n\n"

            # Show default value if present
            if field.default is not dataclasses.MISSING:
                content += f"**Default:** `{field.default!r}`\n\n"
            elif field.default_factory is not dataclasses.MISSING:
                content += "**Default:** Factory function\n\n"

            # Field metadata
            if field.metadata:
                content += f"**Metadata:** {field.metadata}\n\n"

            # Check if field is required (no default)
            is_required = (field.default is dataclasses.MISSING and
                          field.default_factory is dataclasses.MISSING)
            if is_required:
                content += "**Required:** This field must be provided at initialization\n\n"

            # Try to extract field documentation
            field_doc = self._extract_field_doc(obj, field.name)
            if field_doc:
                content += f"{self._escape_mdx(field_doc)}\n\n"

        return content

    def _extract_field_doc(self, obj: Any, field_name: str) -> str | None:
        """Try to extract documentation for a specific field.

        Args:
            obj: The dataclass
            field_name: Name of the field

        Returns:
            Field documentation if found, None otherwise
        """
        import inspect

        doc = inspect.getdoc(obj)
        if not doc:
            return None

        # Look for field documentation in various formats
        import re
        # Try "Attributes:" section format
        pattern = rf'^\s*{re.escape(field_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|^\s*$)'
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)

        if match:
            field_doc = match.group(1).strip()
            # Clean up multiple spaces and newlines
            field_doc = ' '.join(field_doc.split())
            return field_doc

        return None