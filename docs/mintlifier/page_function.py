"""Function documentation page generator."""

import inspect
import importlib
from pathlib import Path
from typing import Optional, List, Any
from page_base import PageBase
from docstring_parser import parse as parse_docstring


class FunctionPageGenerator(PageBase):
    """Generate documentation pages for standalone functions."""
    
    def generate_page(self, function_path: str, parent_groups: List[str], item_type: str) -> Optional[str]:
        """Generate function documentation page."""
        # Parse function path
        parts = function_path.split('.')
        module_path = '.'.join(parts[:-1])
        func_name = parts[-1]
        
        # Import and get function
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, func_name):
                return self._generate_error_page(func_name, parent_groups, f"Function {func_name} not found in {module_path}")
            func = getattr(module, func_name)
            if not callable(func):
                return self._generate_error_page(func_name, parent_groups, f"{func_name} is not callable")
        except ImportError as e:
            return self._generate_error_page(func_name, parent_groups, str(e))
        
        # Build documentation
        content = self._build_frontmatter(func, func_name, function_path)
        content += self._build_function_documentation(func, func_name, function_path)
        
        # Write file and return path
        return self._write_mdx_file(func_name, parent_groups, content)
    
    def _build_frontmatter(self, func: Any, name: str, full_path: str) -> str:
        """Build MDX frontmatter."""
        doc = inspect.getdoc(func)
        parsed = parse_docstring(doc) if doc else None
        description = ""
        if parsed and parsed.short_description:
            description = self._escape_yaml(parsed.short_description[:200])
        
        # Check if it's a UDF (has @pxt.udf decorator or is a CallableFunction)
        icon = "circle-f"  # default function icon
        if hasattr(func, '__class__') and 'CallableFunction' in func.__class__.__name__:
            # This is a UDF - use circle-u icon
            icon = "circle-u"
        
        # Use full path for title, but truncated function name for sidebar
        sidebar_title = self._truncate_sidebar_title(name)
        return f"""---
title: "{full_path}"
sidebarTitle: "{sidebar_title}"
description: "{description or 'Function documentation'}"
icon: "{icon}"
---
"""
    
    def _build_function_documentation(self, func: Any, name: str, full_path: str) -> str:
        """Build complete function documentation."""
        content = ""
        
        # Check for documentation issues
        doc = inspect.getdoc(func)
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
        content += self._document_signature(func, full_path)
        
        # Add GitHub link
        github_link = self._get_github_link(func)
        if github_link:
            content += f"<a href=\"{github_link}\" target=\"_self\">View source on GitHub</a>\n\n"
        
        # Add parameters
        if doc:
            content += self._document_parameters(func, doc)
        
        # Add returns
        if doc:
            content += self._document_returns(doc)
        
        # Add raises
        if doc:
            content += self._document_raises(doc)
        
        # Add examples (if we can extract them)
        if doc:
            content += self._document_examples(doc)
        
        return content
    
    def _document_signature(self, func: Any, full_path: str) -> str:
        """Document function signature."""
        content = "## Signature\n\n```python\n"
        
        try:
            # Check if it's a polymorphic function FIRST (before accessing .signature which throws)
            if hasattr(func, 'is_polymorphic') and func.is_polymorphic:
                # Show ALL signatures for polymorphic functions
                if hasattr(func, 'signatures'):
                    content += f"# Polymorphic function with {len(func.signatures)} signatures:\n\n"
                    for i, sig in enumerate(func.signatures, 1):
                        content += f"# Signature {i}:\n"
                        content += f"{full_path}{sig}\n"
                        if i < len(func.signatures):
                            content += "\n"
                else:
                    content += f"{full_path}(...) # Polymorphic function\n"
            elif hasattr(func, 'signature') and func.signature:
                # Pixeltable CallableFunction stores signature as a string
                sig_str = str(func.signature)
                content += f"{full_path}{sig_str}\n"
            else:
                # Fall back to standard introspection
                sig = inspect.signature(func)
                content += f"{self._format_signature(full_path, sig)}\n"
        except (ValueError, TypeError):
            content += f"{full_path}(...)\n"
        
        content += "```\n\n"
        return content
    
    def _document_parameters(self, func: Any, doc: str) -> str:
        """Document function parameters."""
        parsed = parse_docstring(doc)
        if not parsed.params:
            return ""
        
        content = "## Args\n\n"
        content += "<Note>\nParameter optional/required status may not be accurate if docstring doesn't specify defaults.\n</Note>\n\n"
        
        # Get signature for default values and type annotations
        params_with_defaults = {}
        params_with_types = {}
        try:
            # For polymorphic functions, document ALL parameter variants
            if hasattr(func, 'is_polymorphic') and func.is_polymorphic and hasattr(func, 'signatures'):
                # Collect all unique parameters from all signatures
                all_params = {}
                for sig in func.signatures:
                    if hasattr(sig, 'parameters'):
                        for param_name, param in sig.parameters.items():
                            if param_name not in all_params:
                                # Store first occurrence of each param
                                all_params[param_name] = param
                                if hasattr(param, 'col_type'):
                                    params_with_types[param_name] = str(param.col_type)
                                if hasattr(param, 'default') and param.default is not None:
                                    params_with_defaults[param_name] = param.default
            # For regular Pixeltable functions, parse the signature string
            elif hasattr(func, 'signature') and func.signature:
                # Parse signature string like "(audio: Audio) -> Json"
                sig_str = str(func.signature)
                # Extract parameters from the string
                if '(' in sig_str and ')' in sig_str:
                    params_str = sig_str[sig_str.index('(')+1:sig_str.index(')')]
                    if params_str:
                        for param in params_str.split(','):
                            param = param.strip()
                            if ':' in param:
                                name, type_str = param.split(':', 1)
                                params_with_types[name.strip()] = type_str.strip()
            else:
                # Standard introspection
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    if param.default != inspect.Parameter.empty:
                        params_with_defaults[param_name] = param.default
                    if param.annotation != inspect.Parameter.empty:
                        params_with_types[param_name] = param.annotation
        except (ValueError, TypeError):
            pass
        
        for param in parsed.params:
            # Determine if required
            required = param.arg_name not in params_with_defaults and not param.is_optional
            
            # Get type from annotation or docstring
            type_str = param.type_name or "string"
            if param.arg_name in params_with_types:
                type_annotation = params_with_types[param.arg_name]
                type_str = self._format_type_annotation(type_annotation)
            
            content += f'<ParamField path="{param.arg_name}" type="{type_str}" {"required" if required else ""}>\n'
            
            if param.description:
                desc = self._escape_mdx(param.description)
                # Handle multi-line descriptions with bullet points
                if '\n-' in desc or '\n*' in desc:
                    content += f"  {desc}\n"
                else:
                    content += f"  {desc}\n\n"
            
            # Add structure documentation for complex types
            if param.arg_name == 'messages' and 'list' in type_str.lower():
                content += self._document_messages_structure()
            elif param.arg_name == 'tools' and 'list' in type_str.lower():
                content += self._document_tools_structure()
            elif param.arg_name == 'system' and 'list' in type_str.lower():
                content += self._document_system_structure()
            
            content += "</ParamField>\n\n"
        
        return content
    
    def _format_type_annotation(self, annotation) -> str:
        """Format a type annotation into a readable string."""
        if annotation is None:
            return "any"
        
        # Handle string annotations
        if isinstance(annotation, str):
            return annotation
        
        # Get the string representation
        type_str = str(annotation)
        
        # Clean up common patterns
        type_str = type_str.replace('typing.', '')
        type_str = type_str.replace('Optional[', 'optional<')
        type_str = type_str.replace('List[', 'list<')
        type_str = type_str.replace('Dict[', 'dict<')
        type_str = type_str.replace(']', '>')
        
        # Simplify list[dict[str, Any]] to array
        if 'list<dict<str,' in type_str.lower():
            return "array"
        
        return type_str
    
    def _document_messages_structure(self) -> str:
        """Document the structure of messages parameter."""
        return """
  **Message Structure:**
  ```json
  [
    {
      "role": "user" | "assistant" | "system",
      "content": "string" | [{"text": "string"}] | [{"image": "base64"}]
    }
  ]
  ```
  
"""
    
    def _document_tools_structure(self) -> str:
        """Document the structure of tools parameter."""
        return """
  **Tool Structure:**
  ```json
  [
    {
      "name": "string",
      "description": "string",
      "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
      }
    }
  ]
  ```
  
"""
    
    def _document_system_structure(self) -> str:
        """Document the structure of system parameter."""
        return """
  **System Message Structure:**
  ```json
  [
    {
      "text": "string"
    }
  ]
  ```
  
"""
    
    def _document_returns(self, doc: str) -> str:
        """Document function returns."""
        parsed = parse_docstring(doc)
        if not parsed.returns:
            return ""
        
        content = "## Returns\n\n"
        content += '<ResponseField name="return" type="any" required>\n'
        
        if parsed.returns.description:
            content += f"  {self._escape_mdx(parsed.returns.description)}\n\n"
        
        content += "</ResponseField>\n\n"
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
        # Look for Examples section manually since parser might miss it
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
                # Stop at next section
                if line.strip() and not line.startswith(' ') and ':' in line:
                    break
                example_lines.append(line)
        
        if example_lines:
            content += "## Examples\n\n"
            # Clean up indentation
            min_indent = min(len(line) - len(line.lstrip()) 
                           for line in example_lines if line.strip())
            cleaned = [line[min_indent:] if len(line) > min_indent else line 
                      for line in example_lines]
            
            # Format as code block if it looks like code
            example_text = '\n'.join(cleaned).strip()
            if '>>>' in example_text or 'import' in example_text or '=' in example_text:
                content += f"```python\n{example_text}\n```\n\n"
            else:
                content += f"{self._escape_mdx(example_text)}\n\n"
        
        return content
    
    def _generate_error_page(self, func_name: str, parent_groups: List[str], error: str) -> str:
        """Generate error page when function can't be loaded."""
        content = f"""---
title: "{func_name}"
description: "Function documentation unavailable"
icon: "triangle-exclamation"
---

## ⚠️ Error Loading Function

<Warning>
Failed to load function `{func_name}`:

```
{error}
```
</Warning>
"""
        return self._write_mdx_file(func_name, parent_groups, content)