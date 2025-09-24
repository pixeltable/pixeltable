"""Base class for documentation section generators."""

from abc import ABC, abstractmethod
from typing import Any
import re


class SectionBase(ABC):
    """Base class for generating documentation sections within pages."""

    def __init__(self, show_errors: bool = True):
        """Initialize the section generator.

        Args:
            show_errors: Whether to show error messages in documentation
        """
        self.show_errors = show_errors

    @abstractmethod
    def can_handle(self, obj: Any) -> bool:
        """Check if this section generator can handle the given object.

        Args:
            obj: The object to check

        Returns:
            True if this generator can handle the object
        """
        pass

    @abstractmethod
    def generate_section(self, obj: Any, name: str) -> str:
        """Generate documentation section for the object.

        Args:
            obj: The object to document
            name: The name of the object

        Returns:
            Markdown string for the section
        """
        pass

    def _escape_mdx(self, text: str) -> str:
        """Escape text for MDX format."""
        if not text:
            return ''

        # Escape braces for MDX
        text = text.replace('{', '\\{').replace('}', '\\}')

        # Convert URLs in angle brackets to markdown links
        text = re.sub(r'<(https?://[^>]+)>', r'[\1](\1)', text)

        # Handle other angle brackets
        text = re.sub(r'<([^>]+)>', r'`\1`', text)

        return text

    def _format_type(self, type_annotation: Any) -> str:
        """Format a type annotation for display.

        Args:
            type_annotation: The type to format

        Returns:
            Formatted type string
        """
        if type_annotation is None:
            return "Any"

        # Handle string annotations
        if isinstance(type_annotation, str):
            return type_annotation

        # Get the string representation
        type_str = str(type_annotation)

        # Clean up common patterns
        type_str = type_str.replace('typing.', '')
        type_str = type_str.replace('NoneType', 'None')

        return type_str