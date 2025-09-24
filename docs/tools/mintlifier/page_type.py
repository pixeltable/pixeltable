"""Type documentation page generator for Pixeltable data types."""

import inspect
import importlib
from pathlib import Path
from typing import Optional, List
from page_base import PageBase
from docstring_parser import parse as parse_docstring


class TypePageGenerator(PageBase):
    """Generate documentation pages for Pixeltable types (Image, Video, etc.)."""
    
    def __init__(self, output_dir: Path, version: str = "main", show_errors: bool = True,
                 github_repo: str = "pixeltable/pixeltable", github_package_path: str = "pixeltable"):
        """Initialize with output directory, version, and error display setting."""
        super().__init__(output_dir, version, show_errors, github_repo, github_package_path)
    
    def generate_page(self, type_path: str, parent_groups: List[str], item_type: str, opml_children: List[str] = None) -> Optional[str]:
        """Generate type documentation page.
        
        Types are documented differently from classes - they show:
        - The type description
        - How to use them in schemas
        - Any special properties or methods (but not as detailed as classes)
        
        Args:
            type_path: Full path to type (e.g., 'pixeltable.Image')
            parent_groups: Parent group hierarchy
            item_type: Type of item ('type')
            opml_children: Not used for types
        """
        try:
            parts = type_path.split('.')
            module_path = '.'.join(parts[:-1])
            type_name = parts[-1]
            
            module = importlib.import_module(module_path)
            if not hasattr(module, type_name):
                return self._generate_error_page(type_name, parent_groups, f"Type {type_name} not found")
            
            type_cls = getattr(module, type_name)
            # Check if it's a class OR a type alias (like String, Int, etc.)
            # These are typing._AnnotatedAlias objects in Pixeltable
            if not (inspect.isclass(type_cls) or hasattr(type_cls, '__class__')):
                return self._generate_error_page(type_name, parent_groups, f"{type_name} is not a type")
                
        except ImportError as e:
            return self._generate_error_page(type_name, parent_groups, str(e))
        
        # Build documentation
        content = self._build_frontmatter(type_cls, type_name, type_path)
        content += self._build_type_documentation(type_cls, type_name, type_path)
        
        # Write file and return path
        return self._write_mdx_file(type_name.lower(), parent_groups, content)
    
    def _build_frontmatter(self, type_cls, type_name: str, full_path: str) -> str:
        """Build MDX frontmatter for type documentation."""
        # Try to get docstring - might not work for type aliases
        try:
            doc = inspect.getdoc(type_cls) if hasattr(type_cls, '__doc__') else None
        except:
            doc = None
        parsed = parse_docstring(doc) if doc else None
        
        # Extract short description for frontmatter
        description = ""
        if parsed and parsed.short_description:
            # Clean up the description - remove extra line breaks and clean whitespace
            desc_text = parsed.short_description[:200]
            desc_text = ' '.join(desc_text.split())  # Normalize whitespace
            description = self._escape_yaml(desc_text)
        elif not description:
            description = f"Pixeltable column type for {type_name.lower()} data"
        
        # Types get circle-t icon
        return f"""---
title: "{full_path}"
sidebarTitle: "{type_name}"
description: "{description}"
icon: "circle-t"
---

"""
    
    def _build_type_documentation(self, type_cls, type_name: str, full_path: str) -> str:
        """Build complete type documentation."""
        content = ""
        
        # Get docstring - handle type aliases gracefully
        try:
            doc = inspect.getdoc(type_cls) if hasattr(type_cls, '__doc__') else None
        except:
            doc = None
        parsed = parse_docstring(doc) if doc else None
        
        # Add long description if available (short is in frontmatter)
        if parsed and parsed.long_description:
            content += f"{self._escape_mdx(parsed.long_description)}\n\n"
        elif not doc:
            # No documentation available
            if self.show_errors:
                content += "## ⚠️ No Documentation\n\n"
                content += f"<Warning>\nDocumentation for type `{full_path}` is not available.\n</Warning>\n\n"
        
        # Add GitHub link (may not work for type aliases)
        try:
            github_link = self._get_github_link(type_cls)
            if github_link:
                content += f"<a href=\"{github_link}\" target=\"_self\">View source on GitHub</a>\n\n"
        except:
            # Type aliases might not have inspectable source
            pass
        
        # Add usage section
        content += self._add_usage_section(type_name)
        
        # Add properties section if applicable
        content += self._add_properties_section(type_name)
        
        # Add see also section for related functions
        content += self._add_see_also_section(type_name)
        
        return content
    
    def _add_usage_section(self, type_name: str) -> str:
        """Add usage examples for the type."""
        content = "## Usage\n\n"
        
        # Type-specific usage examples
        if type_name == 'Image':
            content += """Create a table with an `Image` column:

```python
import pixeltable as pxt

table = pxt.create_table('images', {
    'img': pxt.Image,
    'caption': pxt.String
})

# Insert images from files or URLs
table.insert({
    'img': '/path/to/image.jpg',
    'caption': 'A sample image'
})
```
"""
        elif type_name == 'Video':
            content += """Create a table with a `Video` column:

```python
import pixeltable as pxt

table = pxt.create_table('videos', {
    'video': pxt.Video,
    'title': pxt.String
})

# Insert videos from files or URLs
table.insert({
    'video': '/path/to/video.mp4',
    'title': 'A sample video'
})
```
"""
        elif type_name == 'Audio':
            content += """Create a table with an `Audio` column:

```python
import pixeltable as pxt

table = pxt.create_table('audio_files', {
    'audio': pxt.Audio,
    'transcript': pxt.String
})
```
"""
        elif type_name == 'Document':
            content += """Create a table with a `Document` column:

```python
import pixeltable as pxt

table = pxt.create_table('documents', {
    'doc': pxt.Document,
    'summary': pxt.String
})

# Documents can be PDF, HTML, Markdown, etc.
table.insert({
    'doc': '/path/to/document.pdf',
    'summary': 'Document summary'
})
```
"""
        elif type_name == 'Array':
            content += """Create a table with an `Array` column:

```python
import pixeltable as pxt

table = pxt.create_table('embeddings', {
    'text': pxt.String,
    'embedding': pxt.Array[(768,), pxt.Float]
})
```
"""
        elif type_name == 'Json':
            content += """Create a table with a `Json` column:

```python
import pixeltable as pxt

table = pxt.create_table('metadata', {
    'id': pxt.String,
    'data': pxt.Json
})

# Insert JSON data
table.insert({
    'id': 'item1',
    'data': {'key': 'value', 'nested': {'data': 123}}
})
```
"""
        elif type_name in ['String', 'Int', 'Float', 'Bool', 'Timestamp']:
            content += f"""Create a table with a `{type_name}` column:

```python
import pixeltable as pxt

table = pxt.create_table('data', {{
    'value': pxt.{type_name},
    'label': pxt.String
}})
```
"""
        else:
            # Generic example
            content += f"""```python
import pixeltable as pxt

# Use {type_name} in a table schema
table = pxt.create_table('my_table', {{
    'column': pxt.{type_name}
}})
```
"""
        
        content += "\n"
        return content
    
    def _add_properties_section(self, type_name: str) -> str:
        """Add properties section if the type has notable properties."""
        content = ""
        
        # For media types, mention error handling
        if type_name in ['Image', 'Video', 'Audio', 'Document']:
            content += "## Media Validation\n\n"
            content += f"When a column has type `{type_name}`, Pixeltable validates the media files. "
            content += "If validation fails, you can access error information:\n\n"
            content += f"- `table.{type_name.lower()}_column.errortype` - The type of error\n"
            content += f"- `table.{type_name.lower()}_column.errormsg` - The error message\n\n"
        
        # For Array type, mention shape specification
        if type_name == 'Array':
            content += "## Array Shapes\n\n"
            content += "Arrays can have specified shapes and data types:\n\n"
            content += "```python\n"
            content += "# Fixed shape array\n"
            content += "pxt.Array[(768,), pxt.Float]  # 768-dimensional float array\n"
            content += "pxt.Array[(224, 224, 3), pxt.Int]  # 224x224x3 image array\n\n"
            content += "# Variable shape array\n"
            content += "pxt.Array[pxt.Float]  # Float array of any shape\n"
            content += "```\n\n"
        
        return content
    
    def _add_see_also_section(self, type_name: str) -> str:
        """Add see also section with related functions."""
        content = ""
        
        related = {
            'Image': ['pixeltable.functions.image', 'pixeltable.functions.vision'],
            'Video': ['pixeltable.functions.video', 'pixeltable.iterators.FrameIterator'],
            'Audio': ['pixeltable.functions.audio', 'pixeltable.iterators.AudioSplitter'],
            'Document': ['pixeltable.iterators.DocumentSplitter'],
            'String': ['pixeltable.functions.string'],
            'Timestamp': ['pixeltable.functions.timestamp'],
            'Json': ['pixeltable.functions.json'],
        }
        
        if type_name in related:
            content += "## See Also\n\n"
            for item in related[type_name]:
                # Convert to documentation link format
                link_parts = item.split('.')
                if 'iterators' in item:
                    # Link to iterator page
                    class_name = link_parts[-1]
                    content += f"- [{class_name}](../iterators/{self._camel_to_kebab(class_name)})\n"
                elif 'functions' in item:
                    # Link to functions module
                    module_name = link_parts[-1]
                    content += f"- [{module_name} functions](../media-processing/{module_name})\n"
            content += "\n"
        
        return content
    
    def _camel_to_kebab(self, name: str) -> str:
        """Convert CamelCase to kebab-case."""
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('-')
            result.append(char.lower())
        return ''.join(result)
    
    def _generate_error_page(self, type_name: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when type can't be loaded."""
        content = f"""---
title: "{type_name}"
description: "Type documentation unavailable"
icon: "triangle-exclamation"
---

## ⚠️ Error Loading Type

<Warning>
Failed to load type `{type_name}`:

```
{error}
```

This may be because the type is not properly installed or there was an import error.
</Warning>
"""
        
        # Write file and return path
        return self._write_mdx_file(type_name.lower(), parent_groups, content)