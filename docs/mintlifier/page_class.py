"""Class documentation page generator."""

import inspect
from pathlib import Path
from typing import Optional, List, Any
from page_base import PageBase
from docstring_parser import parse as parse_docstring
from page_method import MethodPageGenerator


class ClassPageGenerator(PageBase):
    """Generate documentation pages for Python classes."""
    
    def __init__(self, output_dir: Path, version: str = "main", show_errors: bool = True):
        """Initialize with output directory, version, and error display setting."""
        super().__init__(output_dir, version, show_errors)
        # Initialize method generator
        self.method_gen = MethodPageGenerator(output_dir, version, show_errors)
    
    def generate_page(self, class_path: str, parent_groups: List[str], item_type: str, opml_children: List[str] = None) -> Optional[dict]:
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
                return self._generate_error_page(class_name, parent_groups, f"Class {class_name} not found in {module_path}")
            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                return self._generate_error_page(class_name, parent_groups, f"{class_name} is not a class")
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
        if self._generated_methods:
            # Class with methods - return as a group/folder
            # Class page comes first, then methods directly (no Methods subgroup)
            # Use full path for group name in menu
            return {
                "group": f"class|{class_path}",  # Add class| prefix for class groups  
                "pages": [class_page] + self._generated_methods
            }
        else:
            # Class with no methods - just return the page
            return class_page
    
    def _build_frontmatter(self, cls: type, name: str, full_path: str) -> str:
        """Build MDX frontmatter."""
        doc = inspect.getdoc(cls)
        parsed = parse_docstring(doc) if doc else None
        description = ""
        if parsed and parsed.short_description:
            description = self._escape_yaml(parsed.short_description[:200])
        
        # Use full path for title, but just class name for sidebar
        class_name = full_path.split('.')[-1]
        return f"""---
title: "{full_path}"
sidebarTitle: "{class_name}"
description: "{description or 'Class documentation'}"
icon: "square-c"
---
"""
    
    def _build_class_documentation(self, cls: type, name: str, full_path: str) -> str:
        """Build complete class documentation."""
        content = ""
        
        # Get docstring
        doc = inspect.getdoc(cls)
        if doc:
            parsed = parse_docstring(doc)
            # Skip short description (it's in frontmatter), only add long description
            if parsed.long_description:
                content += f"\n{self._escape_mdx(parsed.long_description)}\n\n"
        else:
            if self.show_errors:
                content += f"\n## ⚠️ No Documentation\n\n"
                content += f"<Warning>\nDocumentation for `{name}` is not available.\n</Warning>\n\n"
        
        # Add constructor signature
        content += self._document_constructor(cls, name)
        
        # Add GitHub link
        github_link = self._get_github_link(cls)
        if github_link:
            content += f"<a href=\"{github_link}\" target=\"_blank\">View source on GitHub</a>\n\n"
        
        # Document class methods
        content += self._document_methods(cls, full_path)
        
        # Document properties
        content += self._document_properties(cls)
        
        # Document class attributes
        content += self._document_class_attributes(cls)
        
        return content
    
    def _document_constructor(self, cls: type, name: str) -> str:
        """Document class constructor."""
        content = "## Constructor\n\n"
        
        try:
            sig = inspect.signature(cls.__init__)
            content += f"```python\n{self._format_signature(name, sig)}\n```\n\n"
            
            # Get constructor docstring
            init_doc = inspect.getdoc(cls.__init__)
            if init_doc and init_doc != inspect.getdoc(cls):
                parsed = parse_docstring(init_doc)
                
                # Add parameters
                if parsed.params:
                    content += "### Parameters\n\n"
                    for param in parsed.params:
                        content += f"#### `{param.arg_name}`\n"
                        if param.type_name:
                            content += f"**Type:** `{param.type_name}`\n\n"
                        if param.description:
                            content += f"{self._escape_mdx(param.description)}\n\n"
                
                # Add returns
                if parsed.returns:
                    content += "### Returns\n\n"
                    if parsed.returns.type_name:
                        content += f"**Type:** `{parsed.returns.type_name}`\n\n"
                    if parsed.returns.description:
                        content += f"{self._escape_mdx(parsed.returns.description)}\n\n"
        
        except (ValueError, TypeError):
            content += f"```python\n{name}()\n```\n\n"
        
        return content
    
    def _document_methods(self, cls: type, full_path: str) -> str:
        """Document class methods - just list them as links."""
        content = ""
        
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
                    print(f"        ⚠️ Method {method_name} not found in {full_path}")
        else:
            # Document all public methods
            for name, obj in inspect.getmembers(cls):
                # Skip private methods except special methods we want to document
                if name.startswith('_') and name not in ['__init__', '__call__', '__enter__', '__exit__']:
                    continue
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    if name != '__init__':  # Skip init, already documented
                        methods.append((name, obj))
        
        if not methods:
            return content
        
        content += "## Methods\n\n"
        
        # Get class name for parent groups
        class_name = full_path.split('.')[-1]
        method_parent_groups = self.current_parent_groups + [class_name]
        
        # Just list methods as links and generate their pages
        for method_name, method in sorted(methods):
            # Generate the method page
            method_path = f"{full_path}.{method_name}"
            print(f"        📄 Generating method: {method_name}")
            method_result = self.method_gen.generate_page(method_path, method_parent_groups, 'method')
            
            # Track the generated method page
            if method_result:
                if isinstance(method_result, dict):
                    self._generated_methods.append(method_result["page"])
                else:
                    self._generated_methods.append(method_result)
            
            # Get short description if available
            doc = inspect.getdoc(method)
            if doc:
                parsed = parse_docstring(doc)
                if parsed.short_description:
                    content += f"- [{method_name}](./{method_name}) - {self._escape_mdx(parsed.short_description)}\n"
                else:
                    content += f"- [{method_name}](./{method_name})\n"
            else:
                content += f"- [{method_name}](./{method_name})\n"
        
        content += "\n"
        return content
    
    def _document_properties(self, cls: type) -> str:
        """Document class properties."""
        content = ""
        properties = []
        
        for name, obj in inspect.getmembers(cls):
            # Skip private properties
            if name.startswith('_'):
                continue
            if isinstance(obj, property):
                properties.append((name, obj))
        
        if not properties:
            return content
        
        content += "## Properties\n\n"
        
        for prop_name, prop in sorted(properties):
            content += f"### {prop_name}\n\n"
            
            doc = inspect.getdoc(prop.fget) if prop.fget else None
            if doc:
                parsed = parse_docstring(doc)
                if parsed.short_description:
                    content += f"{self._escape_mdx(parsed.short_description)}\n\n"
            
            content += f"**Type:** Property (read"
            if prop.fset:
                content += "/write"
            content += ")\n\n"
        
        return content
    
    def _document_class_attributes(self, cls: type) -> str:
        """Document class-level attributes."""
        content = ""
        
        # Get class attributes (excluding methods and properties)
        attributes = []
        for name in dir(cls):
            if name.startswith('_'):
                continue
            
            try:
                obj = getattr(cls, name)
                if not callable(obj) and not isinstance(obj, property):
                    # Check if it's a class attribute (not instance)
                    if hasattr(cls, name) and not hasattr(cls.__init__, name):
                        attributes.append((name, obj))
            except AttributeError:
                continue
        
        if not attributes:
            return content
        
        content += "## Class Attributes\n\n"
        
        for attr_name, attr_value in sorted(attributes):
            content += f"### {attr_name}\n\n"
            content += f"```python\n{attr_name} = {repr(attr_value)}\n```\n\n"
        
        return content
    
    def _generate_error_page(self, class_name: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when class can't be loaded."""
        content = f"""---
title: "{class_name}"
description: "Class documentation unavailable"
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