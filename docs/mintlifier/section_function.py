"""Generate documentation sections for functions within module pages."""

import inspect
import importlib
from typing import Any, List, Optional
from docstring_parser import parse as parse_docstring
from page_base import PageBase


class FunctionSectionGenerator(PageBase):
    """Generate documentation sections for functions within module pages."""
    
    def __init__(self, show_errors: bool = True):
        """Initialize the section generator."""
        # Don't initialize PageBase with output_dir since we're not writing files
        self.show_errors = show_errors
    
    def generate_section(self, func: Any, func_name: str, module_path: str) -> str:
        """Generate function documentation section for inline use.
        
        Args:
            func: The function object to document
            func_name: Name of the function
            module_path: Full module path (e.g., 'pixeltable.functions.image')
            
        Returns:
            Markdown string for the function documentation
        """
        full_path = f"{module_path}.{func_name}"
        
        # Build section content (no frontmatter for inline sections)
        content = f"### `{func_name}()`\n\n"
        
        # Add description
        doc = inspect.getdoc(func) or ""
        if doc:
            parsed = parse_docstring(doc)
            if parsed and parsed.short_description:
                content += f"{self._escape_mdx(parsed.short_description)}\n\n"
            if parsed.long_description:
                content += f"{self._escape_mdx(parsed.long_description)}\n\n"
        
        # Add signature  
        content += self._document_signature(func, func_name)
        
        # Add parameters
        if doc:
            content += self._document_parameters(func, doc)
        
        # Add returns
        if doc:
            parsed = parse_docstring(doc)
            if parsed and parsed.returns:
                content += self._document_returns(parsed)
        
        # Add examples
        if doc:
            parsed = parse_docstring(doc) 
            if parsed and hasattr(parsed, 'examples') and parsed.examples:
                content += self._document_examples(parsed)
        
        return content
    
    def _document_signature(self, func: Any, func_name: str) -> str:
        """Document function signature."""
        content = "**Signature:**\n\n```python\n"
        
        try:
            # Check if it's a polymorphic function FIRST (before accessing .signature which throws)
            if hasattr(func, 'is_polymorphic') and func.is_polymorphic:
                # Show ALL signatures for polymorphic functions
                if hasattr(func, 'signatures'):
                    for i, sig in enumerate(func.signatures, 1):
                        if len(func.signatures) > 1:
                            content += f"# Signature {i}:\n"
                        content += f"{func_name}{sig}\n"
                        if i < len(func.signatures):
                            content += "\n"
                else:
                    content += f"{func_name}(...) # Polymorphic function\n"
            elif hasattr(func, 'signature') and func.signature:
                # Pixeltable CallableFunction stores signature as a string
                sig_str = str(func.signature)
                content += f"{func_name}{sig_str}\n"
            else:
                # Fall back to standard introspection
                sig = inspect.signature(func)
                content += f"{func_name}{sig}\n"
        except (ValueError, TypeError):
            content += f"{func_name}(...)\n"
        
        content += "```\n\n"
        return content
    
    def _document_parameters(self, func: Any, doc: str) -> str:
        """Document function parameters."""
        parsed = parse_docstring(doc)
        if not parsed or not parsed.params:
            return ""
        
        content = "**Parameters:**\n\n"
        
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
            # Format parameter
            type_str = params_with_types.get(param.arg_name, param.type_name or "Any")
            default = params_with_defaults.get(param.arg_name)
            
            content += f"- **`{param.arg_name}`** "
            if type_str:
                content += f"(*{type_str}*)"
            if default is not None:
                content += f" = `{default}`"
            content += f": {self._escape_mdx(param.description) if param.description else 'No description'}\n"
        
        content += "\n"
        return content
    
    def _document_returns(self, parsed) -> str:
        """Document return value."""
        if not parsed.returns:
            return ""
        
        content = "**Returns:**\n\n"
        
        return_type = parsed.returns.type_name or "Any"
        return_desc = parsed.returns.description or "Return value"
        
        content += f"- *{return_type}*: {self._escape_mdx(return_desc)}\n\n"
        return content
    
    def _document_examples(self, parsed) -> str:
        """Document examples."""
        if not hasattr(parsed, 'examples') or not parsed.examples:
            return ""
        
        content = "**Examples:**\n\n"
        
        for example in parsed.examples:
            if example.description:
                content += f"{self._escape_mdx(example.description)}\n\n"
            if example.snippet:
                content += f"```python\n{example.snippet}\n```\n\n"
        
        return content