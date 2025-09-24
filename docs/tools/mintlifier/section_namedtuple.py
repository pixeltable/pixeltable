"""NamedTuple field documentation section generator."""

from typing import Any
from section_base import SectionBase


class NamedTupleSection(SectionBase):
    """Generate documentation sections for NamedTuple fields."""

    def can_handle(self, obj: Any) -> bool:
        """Check if the object is a NamedTuple."""
        # NamedTuples have a _fields attribute
        return hasattr(obj, '_fields') and hasattr(obj, '_field_defaults')

    def generate_section(self, obj: Any, name: str) -> str:
        """Generate NamedTuple fields documentation section.

        Args:
            obj: The NamedTuple class to document
            name: The name of the NamedTuple

        Returns:
            Markdown string for the fields section
        """
        if not hasattr(obj, '_fields'):
            return ""

        content = "## Attributes\n\n"

        # Get field information
        fields = obj._fields
        defaults = getattr(obj, '_field_defaults', {})
        types = getattr(obj, '__annotations__', {})

        # Document each field
        for field_name in fields:
            content += f"### `{field_name}`\n\n"

            # Format type if available
            if field_name in types:
                type_str = self._format_type(types[field_name])
                content += f"**Type:** `{type_str}`\n\n"

            # Show default value if present
            if field_name in defaults:
                content += f"**Default:** `{defaults[field_name]!r}`\n\n"

            # Note that NamedTuples are immutable
            content += "**Immutable:** This field cannot be modified after creation\n\n"

            # Try to extract field documentation
            field_doc = self._extract_field_doc(obj, field_name)
            if field_doc:
                content += f"{self._escape_mdx(field_doc)}\n\n"

        # Add note about tuple behavior
        content += "## Usage\n\n"
        content += f"`{name}` is a NamedTuple and supports:\n\n"
        content += "- Accessing fields by name: `instance.field_name`\n"
        content += "- Accessing fields by index: `instance[0]`\n"
        content += "- Unpacking: `field1, field2 = instance`\n"
        content += "- Iteration: `for value in instance`\n\n"

        return content

    def _extract_field_doc(self, obj: Any, field_name: str) -> str | None:
        """Try to extract documentation for a specific field.

        Args:
            obj: The NamedTuple class
            field_name: Name of the field

        Returns:
            Field documentation if found, None otherwise
        """
        import inspect

        doc = inspect.getdoc(obj)
        if not doc:
            return None

        # Look for field documentation
        import re
        pattern = rf'^\s*{re.escape(field_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|^\s*$)'
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)

        if match:
            field_doc = match.group(1).strip()
            # Clean up multiple spaces and newlines
            field_doc = ' '.join(field_doc.split())
            return field_doc

        return None