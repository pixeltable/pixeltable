"""TypedDict field documentation section generator."""

from typing import Any
from section_base import SectionBase


class TypedDictSection(SectionBase):
    """Generate documentation sections for TypedDict fields."""

    def can_handle(self, obj: Any) -> bool:
        """Check if the object is a TypedDict."""
        # TypedDicts have a special metaclass
        return hasattr(obj, '__annotations__') and type(obj).__name__ == '_TypedDictMeta'

    def generate_section(self, obj: Any, name: str) -> str:
        """Generate TypedDict fields documentation section.

        Args:
            obj: The TypedDict class to document
            name: The name of the TypedDict

        Returns:
            Markdown string for the fields section
        """
        if not hasattr(obj, '__annotations__'):
            return ''

        content = '## Attributes\n\n'

        # Get annotations
        annotations = obj.__annotations__

        # Get required and optional keys
        required_keys = getattr(obj, '__required_keys__', set())
        optional_keys = getattr(obj, '__optional_keys__', set())

        # If we don't have explicit required/optional, assume all are required
        if not required_keys and not optional_keys:
            required_keys = set(annotations.keys())

        # Sort fields alphabetically
        for field_name in sorted(annotations.keys()):
            field_type = annotations[field_name]

            # Determine if required or optional
            is_required = field_name in required_keys
            is_optional = field_name in optional_keys

            content += f'### `{field_name}`\n\n'

            # Format type
            type_str = self._format_type(field_type)
            content += f'**Type:** `{type_str}`\n\n'

            # Add required/optional indicator
            if is_optional:
                content += '**Optional:** This field is optional\n\n'
            elif is_required:
                content += '**Required:** This field is required\n\n'

            # Try to get field documentation from docstring
            # TypedDicts don't typically have per-field docs, but we could
            # parse them from the class docstring if formatted consistently
            field_doc = self._extract_field_doc(obj, field_name)
            if field_doc:
                content += f'{self._escape_mdx(field_doc)}\n\n'

        return content

    def _extract_field_doc(self, obj: Any, field_name: str) -> str | None:
        """Try to extract documentation for a specific field.

        This looks for field descriptions in the class docstring,
        typically in a format like:
            Attributes:
                field_name: Description of field

        Args:
            obj: The TypedDict class
            field_name: Name of the field

        Returns:
            Field documentation if found, None otherwise
        """
        import inspect

        doc = inspect.getdoc(obj)
        if not doc:
            return None

        # Simple pattern matching for field documentation
        # Look for "field_name:" or "field_name :" patterns
        import re

        pattern = rf'^\s*{re.escape(field_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|$)'
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)

        if match:
            field_doc = match.group(1).strip()
            # Clean up multiple spaces and newlines
            field_doc = ' '.join(field_doc.split())
            return field_doc

        return None
