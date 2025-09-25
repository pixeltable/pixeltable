"""Module documentation page generator."""

import inspect
import importlib
from pathlib import Path
from typing import Optional, List, Any
from page_base import PageBase
from docstring_parser import parse as parse_docstring
from section_function import FunctionSectionGenerator
from page_class import ClassPageGenerator


class ModulePageGenerator(PageBase):
    """Generate documentation pages for Python modules."""

    def __init__(self, output_dir: Path, version: str = "main", show_errors: bool = True,
                 github_repo: str = "pixeltable/pixeltable", github_package_path: str = "pixeltable",
                 internal_blacklist: List[str] = None):
        """Initialize with output directory, version, and error display setting."""
        super().__init__(output_dir, version, show_errors, github_repo, github_package_path)
        # Initialize child generators
        self.function_gen = FunctionSectionGenerator(show_errors)
        self.class_gen = ClassPageGenerator(output_dir, version, show_errors, github_repo, github_package_path)
        # Store internal blacklist
        self.internal_blacklist = set(internal_blacklist) if internal_blacklist else set()
    
    def generate_page(self, module_path: str, parent_groups: List[str], item_type: str, opml_children: List[str] = None) -> Optional[dict]:
        """Generate module documentation page.
        
        Args:
            module_path: Full path to module
            parent_groups: Parent group hierarchy
            item_type: Type of item (module)
            opml_children: If specified, only document these children
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            return self._generate_error_page(module_path, parent_groups, str(e))
        
        # Get module info
        module_name = module_path.split('.')[-1]
        docstring = inspect.getdoc(module) or ""
        
        # Check for explicit children - either from OPML or module attribute
        children = opml_children or self._get_module_children(module)
        
        # Build page content
        content = self._build_frontmatter(module_path, docstring)
        
        if not docstring:
            if self.show_errors:
                content += f"\n## ⚠️ No Documentation\n\n"
                content += f"<Warning>\nDocumentation for `{module_path}` is not available.\n</Warning>\n\n"
        else:
            # Add the full docstring as the module description
            content += f"\n{self._escape_mdx(docstring)}\n\n"

        # Add GitHub link for the module
        github_link = self._get_github_link(module)
        if github_link:
            content += f"<a href=\"{github_link}\" target=\"_blank\">View source on GitHub</a>\n\n"

        # Store parent groups for use in child generation
        self.current_parent_groups = parent_groups
        
        # Add module contents
        if children:
            content += self._document_children(module, children, module_path)
        else:
            content += self._document_all_public(module, module_path, parent_groups)
        
        # Write file and build navigation structure
        module_page = self._write_mdx_file(module_name, parent_groups, content)
        
        # Build navigation structure
        nav_children = []
        
        # Add class structures first (no Classes group)
        # Classes are already groups/folders themselves
        if hasattr(self, '_generated_classes') and self._generated_classes:
            nav_children.extend(self._generated_classes)
            
        # Functions are now inline, so no navigation entries needed
        
        # Return navigation structure for module
        # Return module page and children as a flat list, not a group
        if nav_children:
            # Module with children - return as a flat list
            # Module page comes first, then classes (no grouping)
            return [module_page] + nav_children
        else:
            # Module with no children - just return the page
            return module_page
    
    def _get_module_children(self, module: Any) -> Optional[List[str]]:
        """Check if module has explicit __children__ attribute."""
        if hasattr(module, '__children__'):
            return module.__children__
        return None
    
    def _document_children(self, module: Any, children: List[str], module_path: str) -> str:
        """Document only specified children."""
        content = ""
        
        # Get module name for parent groups
        module_name = module_path.split('.')[-1]
        parent_groups = self.current_parent_groups if hasattr(self, 'current_parent_groups') else []
        
        # Separate children by type
        classes_to_generate = []
        functions_to_generate = []
        
        for child_name in children:
            if not hasattr(module, child_name):
                content += f"<Warning>Child `{child_name}` not found in module</Warning>\n\n"
                continue
            
            child = getattr(module, child_name)
            if inspect.isclass(child):
                classes_to_generate.append((child_name, child))
            elif inspect.isfunction(child) or (hasattr(child, '__class__') and 'CallableFunction' in child.__class__.__name__):
                functions_to_generate.append((child_name, child))
            else:
                # For other types, just document inline
                content += self._document_item(child, child_name, module_path)
        
        # Generate class pages
        if classes_to_generate:
            content += "## Classes\n\n"
            self._generated_classes = []
            class_parent_groups = parent_groups + [module_name]
            
            for name, obj in sorted(classes_to_generate):
                # Generate the class page
                class_path = f"{module_path}.{name}"
                print(f"      📄 Generating class: {name}")
                class_result = self.class_gen.generate_page(class_path, class_parent_groups, 'class')
                
                if class_result:
                    if isinstance(class_result, dict):
                        self._generated_classes.append(class_result)
                    else:
                        self._generated_classes.append(class_result)
                
                # Add link to the list
                content += f"- [{name}](./{name.lower()})\n"
            content += "\n"
        
        # Document functions inline (no separate pages)
        if functions_to_generate:
            content += "## Functions\n\n"
            
            for name, obj in sorted(functions_to_generate):
                # Generate inline documentation
                func_section = self.function_gen.generate_section(obj, name, module_path)
                content += func_section
            
            # Clear the generated functions list since we're not creating separate pages
            self._generated_functions = []
        
        return content
    
    def _document_all_public(self, module: Any, module_path: str, parent_groups: List[str]) -> str:
        """Document all public items in module."""
        content = ""

        # Get all public items
        items = []
        for name in dir(module):
            if name.startswith('_'):
                continue

            # Skip blacklisted internal classes
            if name in self.internal_blacklist:
                continue
            
            try:
                obj = getattr(module, name)
                # For Pixeltable UDFs, they may have different module origins
                # but should still be documented if they're public in this module
                obj_module = getattr(obj, '__module__', None)
                
                # Include if:
                # 1. It's from this module
                # 2. It's from a submodule (e.g., FrameIterator from pixeltable.iterators.video)
                # 3. It's a Pixeltable CallableFunction (UDF) in a functions module
                # 4. It's not an imported standard library item
                if obj_module:
                    is_from_this_module = obj_module == module.__name__
                    # Include items from submodules (for iterators imported into parent module)
                    is_from_submodule = obj_module.startswith(module.__name__ + '.')
                    # Only include UDFs if we're documenting a functions module
                    is_pixeltable_udf = ('functions' in module_path and 
                                       'pixeltable' in obj_module and
                                       hasattr(obj, '__class__') and 
                                       'CallableFunction' in obj.__class__.__name__)
                    is_stdlib = obj_module.startswith('builtins') or not '.' in obj_module
                    
                    if not (is_from_this_module or is_from_submodule or is_pixeltable_udf) or is_stdlib:
                        continue
                
                items.append((name, obj))
            except AttributeError:
                continue
        
        # Group by type
        classes = [(n, o) for n, o in items if inspect.isclass(o)]
        # Include regular functions and Pixeltable CallableFunction objects
        functions = [(n, o) for n, o in items if inspect.isfunction(o) or 
                     (hasattr(o, '__class__') and 'CallableFunction' in o.__class__.__name__)]
        constants = [(n, o) for n, o in items if not inspect.isclass(o) and not inspect.isfunction(o) and 
                     not inspect.ismodule(o) and not (hasattr(o, '__class__') and 'CallableFunction' in o.__class__.__name__)]
        
        # Document classes - list with links and generate their pages
        if classes:
            content += "## Classes\n\n"
            module_name = module_path.split('.')[-1]
            class_parent_groups = parent_groups + [module_name]
            
            # Track generated class pages
            self._generated_classes = []
            
            for name, obj in sorted(classes):
                # Generate the class page
                class_path = f"{module_path}.{name}"
                print(f"      📄 Generating class: {name}")
                class_result = self.class_gen.generate_page(class_path, class_parent_groups, 'class')
                
                if class_result:
                    # Class generator returns full navigation structure
                    if isinstance(class_result, dict):
                        self._generated_classes.append(class_result)
                    else:
                        self._generated_classes.append(class_result)
                
                # Add link to the list
                content += f"- [{name}](./{name.lower()})\n"
            content += "\n"
        
        # Document functions inline (no separate pages)
        if functions:
            content += "## Functions\n\n"
            
            for name, obj in sorted(functions):
                # Generate inline documentation
                func_section = self.function_gen.generate_section(obj, name, module_path)
                content += func_section
            
            # Clear the generated functions list since we're not creating separate pages
            self._generated_functions = []
        
        # Document constants
        if constants:
            content += "### Constants\n\n"
            for name, obj in sorted(constants):
                content += f"#### `{name}`\n\n"
                content += f"```python\n{name} = {repr(obj)}\n```\n\n"
        
        return content
    
    def _document_item(self, obj: Any, name: str, module_path: str) -> str:
        """Document a single item."""
        if inspect.isclass(obj):
            return self._document_class_summary(obj, name, module_path)
        elif inspect.isfunction(obj):
            return self._document_function_summary(obj, name, module_path)
        else:
            return f"### `{name}`\n\n```python\n{name} = {repr(obj)}\n```\n\n"
    
    def _document_class_summary(self, cls: type, name: str, module_path: str) -> str:
        """Generate class summary documentation."""
        content = f"### {name}\n\n"
        
        # Add signature
        try:
            sig = inspect.signature(cls.__init__)
            content += f"```python\n{self._format_signature(name, sig)}\n```\n\n"
        except (ValueError, TypeError):
            content += f"```python\nclass {name}\n```\n\n"
        
        # Add description
        doc = inspect.getdoc(cls)
        if doc:
            parsed = parse_docstring(doc)
            if parsed.short_description:
                content += f"{self._escape_mdx(parsed.short_description)}\n\n"
        
        # Add link to full documentation
        content += f"[→ Full documentation]({self._get_class_doc_link(module_path, name)})\n\n"
        
        return content
    
    def _document_function_summary(self, func: Any, name: str, module_path: str) -> str:
        """Generate function summary documentation."""
        content = f"### {name}\n\n"
        
        # Handle Pixeltable CallableFunction objects
        if hasattr(func, '__class__') and 'CallableFunction' in func.__class__.__name__:
            # Try to get the signature from the wrapped function
            if hasattr(func, 'sig'):
                sig = func.sig
                content += f"```python\n{name}{sig}\n```\n\n"
            elif hasattr(func, '__wrapped__'):
                try:
                    sig = inspect.signature(func.__wrapped__)
                    content += f"```python\n{self._format_signature(name, sig)}\n```\n\n"
                except:
                    content += f"```python\n{name}(...)\n```\n\n"
            else:
                content += f"```python\n{name}(...)\n```\n\n"
            
            # Try to get docstring
            doc = None
            if hasattr(func, '__doc__'):
                doc = func.__doc__
            elif hasattr(func, '__wrapped__') and hasattr(func.__wrapped__, '__doc__'):
                doc = func.__wrapped__.__doc__
        else:
            # Regular function
            try:
                sig = inspect.signature(func)
                content += f"```python\n{self._format_signature(f'{module_path}.{name}', sig)}\n```\n\n"
            except (ValueError, TypeError):
                content += f"```python\n{module_path}.{name}(...)\n```\n\n"
            
            doc = inspect.getdoc(func)
        
        # Add description
        if doc:
            parsed = parse_docstring(doc)
            if parsed.short_description:
                content += f"{self._escape_mdx(parsed.short_description)}\n\n"
        
        # Add GitHub link if possible
        source_func = func.__wrapped__ if hasattr(func, '__wrapped__') else func
        github_link = self._get_github_link(source_func)
        if github_link:
            content += f"<a href=\"{github_link}\" target=\"_blank\">View source on GitHub</a>\n\n"
        
        return content
    
    def _build_frontmatter(self, module_path: str, docstring: str) -> str:
        """Build MDX frontmatter."""
        parsed = parse_docstring(docstring) if docstring else None
        description = ""
        if parsed and parsed.short_description:
            description = self._escape_yaml(parsed.short_description[:200])
        
        # Use full module path for title, but just module name for sidebar
        module_name = module_path.split('.')[-1]
        return f"""---
title: "{module_path}"
sidebarTitle: "{module_name}"
icon: "square-m"
---
"""
    
    def _generate_error_page(self, module_path: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when module can't be imported."""
        module_name = module_path.split('.')[-1]
        content = f"""---
title: "{module_name}"
icon: "triangle-exclamation"
---

## ⚠️ Import Error

<Warning>
Failed to import module `{module_path}`:

```
{error}
```
</Warning>
"""
        return self._write_mdx_file(module_name, parent_groups, content)
    
    def _get_class_doc_link(self, module_path: str, class_name: str) -> str:
        """Get link to class documentation page."""
        # This would be generated by the class page generator
        path_parts = module_path.split('.')
        if len(path_parts) > 2:
            # Remove 'pixeltable.functions' prefix for cleaner URLs
            if path_parts[0] == 'pixeltable' and path_parts[1] == 'functions':
                group = path_parts[2]
                return f"../{class_name.lower()}"
        return f"./{class_name.lower()}"