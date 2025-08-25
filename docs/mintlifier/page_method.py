"""Method documentation page generator."""

import inspect
import importlib
from pathlib import Path
from typing import Optional, List, Any
from page_base import PageBase
from docstring_parser import parse as parse_docstring


class MethodPageGenerator(PageBase):
    """Generate documentation pages for class methods."""
    
    def generate_page(self, method_path: str, parent_groups: List[str], item_type: str) -> Optional[str]:
        """Generate method documentation page."""
        # Parse method path (e.g., pixeltable.Table.insert)
        parts = method_path.split('.')
        if len(parts) < 3:
            return self._generate_error_page(method_path, parent_groups, "Invalid method path")
        
        method_name = parts[-1]
        class_name = parts[-2]
        module_path = '.'.join(parts[:-2])
        
        # Import and get method
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                return self._generate_error_page(method_name, parent_groups, f"Class {class_name} not found in {module_path}")
            
            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                return self._generate_error_page(method_name, parent_groups, f"{class_name} is not a class")
            
            if not hasattr(cls, method_name):
                return self._generate_error_page(method_name, parent_groups, f"Method {method_name} not found in {class_name}")
            
            method = getattr(cls, method_name)
            if not callable(method):
                return self._generate_error_page(method_name, parent_groups, f"{method_name} is not callable")
            
        except ImportError as e:
            return self._generate_error_page(method_name, parent_groups, str(e))
        
        # Build documentation
        content = self._build_frontmatter(method, method_name, class_name, method_path)
        content += self._build_method_documentation(method, method_name, class_name, method_path)
        
        # Write file and return path
        return self._write_mdx_file(method_name, parent_groups, content)
    
    def _build_frontmatter(self, method: Any, name: str, class_name: str, full_path: str) -> str:
        """Build MDX frontmatter."""
        doc = inspect.getdoc(method)
        parsed = parse_docstring(doc) if doc else None
        description = ""
        if parsed and parsed.short_description:
            description = self._escape_yaml(parsed.short_description[:200])
        
        # Use consistent circle-m icon for all methods
        icon = "circle-m"
        
        # Use full path for title, but truncated method name for sidebar
        sidebar_title = self._truncate_sidebar_title(name)
        return f"""---
title: "{full_path}"
sidebarTitle: "{sidebar_title}"
description: "{description or 'Method documentation'}"
icon: "{icon}"
---
"""
    
    def _build_method_documentation(self, method: Any, name: str, class_name: str, full_path: str) -> str:
        """Build complete method documentation."""
        content = ""
        
        # Check for documentation
        doc = inspect.getdoc(method)
        if not doc:
            if self.show_errors:
                content += "## ⚠️ No Documentation\n\n"
                content += f"<Warning>\nDocumentation for `{full_path}` is not available.\n</Warning>\n\n"
        else:
            parsed = parse_docstring(doc)
            
            # Check for unparsed examples
            if self.show_errors and ('Examples:' in doc or 'Example:' in doc):
                if not hasattr(parsed, 'examples') or not parsed.examples:
                    content += "## ⚠️ Documentation Issues\n\n"
                    content += "<Warning>\n- Examples section exists in docstring but was not parsed by docstring_parser\n</Warning>\n\n"
            
            # Add long description only (short description is in frontmatter)
            if parsed.long_description:
                content += f"{self._escape_mdx(parsed.long_description)}\n\n"
        
        # Add signature
        content += self._document_signature(method, name, class_name)
        
        # Add GitHub link
        github_link = self._get_github_link(method)
        if github_link:
            content += f"<a href=\"{github_link}\" target=\"_self\">View source on GitHub</a>\n\n"
        
        # Add parameters
        if doc:
            content += self._document_parameters(method, doc)
        
        # Add returns
        if doc:
            content += self._document_returns(doc)
        
        # Add raises
        if doc:
            content += self._document_raises(doc)
        
        # Add examples
        if doc:
            content += self._document_examples(doc)
        
        # Add note about parent class
        content += self._add_parent_class_note(class_name)
        
        return content
    
    def _document_signature(self, method: Any, name: str, class_name: str) -> str:
        """Document method signature."""
        content = "## Signature\n\n```python\n"
        
        try:
            sig = inspect.signature(method)
            # Format as instance method
            if 'self' in sig.parameters:
                # Remove self from signature for display
                params = list(sig.parameters.values())[1:]
                new_sig = sig.replace(parameters=params)
                content += f"{name}{new_sig}\n"
            else:
                # Class method or static method
                content += f"{class_name}.{self._format_signature(name, sig)}\n"
        except (ValueError, TypeError):
            content += f"{name}(...)\n"
        
        content += "```\n\n"
        return content
    
    def _document_parameters(self, method: Any, doc: str) -> str:
        """Document method parameters."""
        parsed = parse_docstring(doc)
        if not parsed.params:
            return ""
        
        content = "## Parameters\n\n"
        
        # Get signature for default values
        try:
            sig = inspect.signature(method)
            params_with_defaults = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.default != inspect.Parameter.empty:
                    params_with_defaults[param_name] = param.default
        except (ValueError, TypeError):
            params_with_defaults = {}
        
        for param in parsed.params:
            if param.arg_name == 'self':
                continue
            
            # Determine if required
            required = param.arg_name not in params_with_defaults and not param.is_optional
            
            content += f'<ParamField path="{param.arg_name}" type="{param.type_name or "any"}" {"required" if required else ""}>\n'
            
            if param.description:
                desc = self._escape_mdx(param.description)
                content += f"  {desc}\n"
            
            content += "</ParamField>\n\n"
        
        return content
    
    def _document_returns(self, doc: str) -> str:
        """Document method returns."""
        parsed = parse_docstring(doc)
        if not parsed.returns:
            return ""
        
        content = "## Returns\n\n"
        
        if parsed.returns.description:
            content += f"{self._escape_mdx(parsed.returns.description)}\n\n"
        
        if parsed.returns.type_name:
            content += f"**Type:** `{parsed.returns.type_name}`\n\n"
        
        return content
    
    def _document_raises(self, doc: str) -> str:
        """Document exceptions raised."""
        parsed = parse_docstring(doc)
        if not parsed.raises:
            return ""
        
        content = "## Raises\n\n"
        
        for exc in parsed.raises:
            content += f"### `{exc.type_name or 'Exception'}`\n\n"
            if exc.description:
                content += f"{self._escape_mdx(exc.description)}\n\n"
        
        return content
    
    def _document_examples(self, doc: str) -> str:
        """Try to extract and document examples."""
        # Similar to function examples
        if 'Examples:' not in doc and 'Example:' not in doc:
            return ""
        
        content = ""
        lines = doc.split('\n')
        in_examples = False
        example_lines = []
        
        for line in lines:
            if 'Examples:' in line or 'Example:' in line:
                in_examples = True
                continue
            elif in_examples:
                if line.strip() and not line.startswith(' ') and ':' in line:
                    break
                example_lines.append(line)
        
        if example_lines:
            content += "## Examples\n\n"
            min_indent = min(len(line) - len(line.lstrip()) 
                           for line in example_lines if line.strip())
            cleaned = [line[min_indent:] if len(line) > min_indent else line 
                      for line in example_lines]
            
            example_text = '\n'.join(cleaned).strip()
            if '>>>' in example_text or 'import' in example_text or '=' in example_text:
                content += f"```python\n{example_text}\n```\n\n"
            else:
                content += f"{self._escape_mdx(example_text)}\n\n"
        
        return content
    
    def _add_parent_class_note(self, class_name: str) -> str:
        """Add note about parent class."""
        content = "\n---\n\n"
        content += f"<Note>\nThis method belongs to the [`{class_name}`](../{class_name}) class.\n</Note>\n"
        return content
    
    def _generate_error_page(self, method_name: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when method can't be loaded."""
        content = f"""---
title: "{method_name}"
description: "Method documentation unavailable"
icon: "triangle-exclamation"
---

## ⚠️ Error Loading Method

<Warning>
Failed to load method `{method_name}`:

```
{error}
```
</Warning>
"""
        return self._write_mdx_file(method_name, parent_groups, content)