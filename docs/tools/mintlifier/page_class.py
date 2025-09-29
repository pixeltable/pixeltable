"""Class documentation page generator."""

import inspect
from pathlib import Path
from typing import Optional, List, Any
from page_base import PageBase
from docstring_parser import parse as parse_docstring
from section_method import MethodSectionGenerator
from section_typeddict import TypedDictSection
from section_dataclass import DataclassSection
from section_namedtuple import NamedTupleSection
from section_attributes import AttributesSection


class ClassPageGenerator(PageBase):
    """Generate documentation pages for Python classes."""

    def __init__(
        self,
        output_dir: Path,
        version: str = 'main',
        show_errors: bool = True,
        github_repo: str = 'pixeltable/pixeltable',
        github_package_path: str = 'pixeltable',
    ):
        """Initialize with output directory, version, and error display setting."""
        super().__init__(output_dir, version, show_errors, github_repo, github_package_path)
        # Initialize method generator for inline sections
        self.method_gen = MethodSectionGenerator(show_errors)

        # Initialize attribute section handlers
        self.attribute_handlers = [
            TypedDictSection(show_errors),
            DataclassSection(show_errors),
            NamedTupleSection(show_errors),
            AttributesSection(show_errors),  # Fallback handler
        ]

    def generate_page(
        self, class_path: str, parent_groups: List[str], item_type: str, opml_children: List[str] | None = None
    ) -> Optional[dict]:
        """Generate class documentation page and return navigation structure.

        Args:
            class_path: Full path to class
            parent_groups: Parent group hierarchy
            item_type: Type of item (class)
            opml_children: If specified, only document these methods
        """
        # Store parent groups for use in method generation
        self.current_parent_groups = parent_groups
        self._generated_methods = []  # Track generated method pages

        # Parse class path
        parts = class_path.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]

        # Import and get class
        try:
            import importlib

            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                return self._generate_error_page(
                    class_name, parent_groups, f'Class {class_name} not found in {module_path}'
                )
            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                return self._generate_error_page(class_name, parent_groups, f'{class_name} is not a class')
        except ImportError as e:
            return self._generate_error_page(class_name, parent_groups, str(e))

        # Store children for use in method documentation
        self.opml_children = opml_children

        # Build documentation
        content = self._build_frontmatter(cls, class_name, class_path)
        content += self._build_class_documentation(cls, class_name, class_path)

        # Write file
        class_page = self._write_mdx_file(class_name, parent_groups, content)

        # Build navigation structure for class
        # Methods are now inline, so classes don't need to be groups anymore
        # Just return the class page directly
        return class_page

    def _build_frontmatter(self, cls: type, name: str, full_path: str) -> str:
        """Build MDX frontmatter."""
        doc = inspect.getdoc(cls)
        parsed = parse_docstring(doc) if doc else None
        description = ''
        if parsed and parsed.short_description:
            description = self._escape_yaml(parsed.short_description[:200])

        # Use full path for title, but just class name for sidebar
        class_name = full_path.split('.')[-1]
        return f"""---
title: "{full_path}"
sidebarTitle: "{class_name}"
icon: "square-c"
---
"""

    def _build_class_documentation(self, cls: type, name: str, full_path: str) -> str:
        """Build complete class documentation."""
        content = ''

        # Get docstring
        doc = inspect.getdoc(cls)
        if doc:
            # Add the full docstring as the class description
            content += f'\n{self._escape_mdx(doc)}\n\n'
        else:
            if self.show_errors:
                content += f'\n## ⚠️ No Documentation\n\n'
                content += f'<Warning>\nDocumentation for `{name}` is not available.\n</Warning>\n\n'

        # Skip constructor per Marcel's feedback - no longer documenting __init__

        # Add GitHub link
        github_link = self._get_github_link(cls)
        if github_link:
            content += f'<a href="{github_link}" target="_blank">View source on GitHub</a>\n\n'

        # Document class methods
        content += self._document_methods(cls, full_path)

        # Document all attributes (properties, fields, etc.) in a unified section
        content += self._document_all_attributes(cls)
        return content

    # Constructor documentation removed per Marcel's feedback

    def _document_methods(self, cls: type, full_path: str) -> str:
        """Document class methods inline."""
        content = ''

        # Get methods to document
        methods = []

        if self.opml_children:
            # Only document specified methods
            for method_name in self.opml_children:
                if hasattr(cls, method_name):
                    obj = getattr(cls, method_name)
                    if inspect.ismethod(obj) or inspect.isfunction(obj):
                        methods.append((method_name, obj))
                else:
                    print(f'        ⚠️ Method {method_name} not found in {full_path}')
        else:
            # Document all public methods
            for name, obj in inspect.getmembers(cls):
                # Skip private methods except special methods we want to document
                if name.startswith('_') and name not in ['__call__', '__enter__', '__exit__']:
                    continue
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    # Skip __init__ per Marcel's feedback
                    methods.append((name, obj))

        if not methods:
            return content

        content += '## Methods\n\n'

        # Get class name for inline sections
        class_name = full_path.split('.')[-1]

        # Generate inline method documentation
        for method_name, method in sorted(methods):
            # Generate inline section (skips __init__ internally)
            method_section = self.method_gen.generate_section(method, method_name, class_name)
            content += method_section

        # Clear the generated methods list since we're not creating separate pages
        self._generated_methods = []

        return content

    def _document_all_attributes(self, cls: type) -> str:
        """Document all attributes (properties, fields, etc.) in a unified section."""
        import dataclasses

        content = ''
        all_attributes = []

        # Collect properties
        for name, obj in inspect.getmembers(cls):
            # Skip private attributes
            if name.startswith('_'):
                continue
            if isinstance(obj, property):
                doc = inspect.getdoc(obj.fget) if obj.fget else None
                attr_type = 'property'
                if obj.fset:
                    attr_type = 'property (read/write)'
                else:
                    attr_type = 'property (read-only)'
                all_attributes.append((name, attr_type, doc, None))

        # Collect dataclass fields if applicable
        if dataclasses.is_dataclass(cls):
            for field in dataclasses.fields(cls):
                if not field.name.startswith('_'):
                    # Try to get field documentation from class docstring
                    field_doc = self._extract_field_doc(cls, field.name)
                    field_type = self._format_type(field.type) if field.type else 'Any'
                    default = (
                        field.default
                        if field.default != dataclasses.MISSING
                        else field.default_factory
                        if field.default_factory != dataclasses.MISSING
                        else None
                    )
                    all_attributes.append((field.name, f'field: {field_type}', field_doc, default))

        # Collect TypedDict fields
        elif hasattr(cls, '__annotations__') and type(cls).__name__ == '_TypedDictMeta':
            annotations = cls.__annotations__
            required_keys = getattr(cls, '__required_keys__', set())
            optional_keys = getattr(cls, '__optional_keys__', set())

            for field_name, field_type in annotations.items():
                if not field_name.startswith('_'):
                    field_doc = self._extract_field_doc(cls, field_name)
                    type_str = self._format_type(field_type)
                    if field_name in optional_keys:
                        type_str = f'{type_str} (optional)'
                    elif field_name in required_keys:
                        type_str = f'{type_str} (required)'
                    all_attributes.append((field_name, f'field: {type_str}', field_doc, None))

        # Collect NamedTuple fields
        elif hasattr(cls, '_fields') and hasattr(cls, '_field_defaults'):
            fields = cls._fields
            defaults = getattr(cls, '_field_defaults', {})
            types = getattr(cls, '__annotations__', {})

            for field_name in fields:
                if not field_name.startswith('_'):
                    field_doc = self._extract_field_doc(cls, field_name)
                    type_str = self._format_type(types.get(field_name, 'Any'))
                    default = defaults.get(field_name)
                    all_attributes.append((field_name, f'field: {type_str}', field_doc, default))

        if not all_attributes:
            return content

        content += '## Attributes\n\n'

        for i, (attr_name, attr_type, doc, default) in enumerate(sorted(all_attributes)):
            # Add separator between items (but not before the first one)
            if i > 0:
                content += '---\n\n'

            # Determine the label based on attr_type
            if 'property' in attr_type:
                label = 'property'
            elif 'field' in attr_type:
                label = 'field'
            else:
                label = 'attribute'

            content += f'### `{attr_name}` *{label}*\n\n'

            # Add documentation or warning
            if doc:
                parsed = parse_docstring(doc) if '\n' in doc else None
                if parsed and parsed.short_description:
                    content += f'{self._escape_mdx(parsed.short_description)}\n\n'
                else:
                    content += f'{self._escape_mdx(doc)}\n\n'
            elif self.show_errors:
                content += '⚠️ **No documentation**\n\n'

            # Add type information
            content += f'**Type:** {attr_type}\n\n'

            # Add default value if present
            if default is not None:
                if callable(default):
                    content += f'**Default:** Factory function\n\n'
                else:
                    content += f'**Default:** `{default!r}`\n\n'

        return content

    def _extract_field_doc(self, cls: type, field_name: str) -> str | None:
        """Extract documentation for a specific field from class docstring."""
        doc = inspect.getdoc(cls)
        if not doc:
            return None

        # Simple pattern matching for field documentation
        import re

        pattern = rf'^\s*{re.escape(field_name)}\s*:\s*(.+?)(?=^\s*\w+\s*:|^\s*$)'
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)

        if match:
            field_doc = match.group(1).strip()
            # Clean up multiple spaces and newlines
            field_doc = ' '.join(field_doc.split())
            return field_doc

        return None

    def _generate_error_page(self, class_name: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when class can't be loaded."""
        content = f"""---
title: "{class_name}"
icon: "triangle-exclamation"
---

## ⚠️ Error Loading Class

<Warning>
Failed to load class `{class_name}`:

```
{error}
```
</Warning>
"""
        return self._write_mdx_file(class_name, parent_groups, content)
