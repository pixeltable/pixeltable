"""Standard class attribute documentation section generator."""

import inspect
from typing import Any
from section_base import SectionBase


class AttributesSection(SectionBase):
    """Generate documentation sections for standard class attributes."""

    def can_handle(self, obj: Any) -> bool:
        """Check if this is a standard class (not TypedDict, dataclass, etc)."""
        # This is the fallback handler, so it can handle anything
        # But it returns False for special types that have their own handlers
        import dataclasses

        # Don't handle TypedDict
        if hasattr(obj, '__annotations__') and type(obj).__name__ == '_TypedDictMeta':
            return False

        # Don't handle dataclass
        if dataclasses.is_dataclass(obj):
            return False

        # Don't handle NamedTuple
        return not (hasattr(obj, '_fields') and hasattr(obj, '_field_defaults'))

    def generate_section(self, obj: Any, name: str) -> str:
        """Generate class attributes documentation section.

        Args:
            obj: The class to document
            name: The name of the class

        Returns:
            Markdown string for the attributes section
        """
        # Get class attributes (excluding methods and properties)
        attributes = []

        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue

            try:
                attr_value = getattr(obj, attr_name)
                # Check if it's a class attribute (not instance method or property)
                if (not callable(attr_value) and not isinstance(attr_value, property) and
                        hasattr(obj, attr_name) and not hasattr(obj.__init__, attr_name)):
                    attributes.append((attr_name, attr_value))
            except AttributeError:
                continue

        if not attributes:
            return ""

        content = "## Class Attributes\n\n"

        for attr_name, attr_value in sorted(attributes):
            content += f"### `{attr_name}`\n\n"

            # Try to get type from annotations
            if hasattr(obj, '__annotations__') and attr_name in obj.__annotations__:
                type_str = self._format_type(obj.__annotations__[attr_name])
                content += f"**Type:** `{type_str}`\n\n"

            # Show the value
            content += f"**Value:** `{attr_value!r}`\n\n"

            # Try to extract attribute documentation
            attr_doc = self._extract_attr_doc(obj, attr_name)
            if attr_doc:
                content += f"{self._escape_mdx(attr_doc)}\n\n"

        return content

    def _extract_attr_doc(self, obj: Any, attr_name: str) -> str | None:
        """Try to extract documentation for a specific attribute.

        Args:
            obj: The class
            attr_name: Name of the attribute

        Returns:
            Attribute documentation if found, None otherwise
        """
        doc = inspect.getdoc(obj)
        if not doc:
            return None

        # Look for attribute documentation in docstring
        import re
        # Try various documentation patterns
        patterns = [
            # "Attributes:" section format
            rf'Attributes:.*?^\s*{re.escape(attr_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|^\s*$)',
            # Direct "attr_name:" format
            rf'^\s*{re.escape(attr_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|^\s*$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)
            if match:
                attr_doc = match.group(1).strip()
                # Clean up multiple spaces and newlines
                attr_doc = ' '.join(attr_doc.split())
                return attr_doc

        return None
