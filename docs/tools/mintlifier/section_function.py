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
        
        # Build section content with elegant visual separation
        content = "\n---\n\n"  # Beautiful horizontal divider
        content += f"### `{func_name}()`\n\n"
        
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
        
        # Add examples using docstring_parser
        if doc:
            parsed = parse_docstring(doc)
            examples_meta = [m for m in parsed.meta if m.args and 'examples' in m.args[0].lower()]
            if examples_meta:
                content += self._format_examples_from_meta(examples_meta)
        
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
                        # Format signature with line breaks after commas for readability
                        sig_str = str(sig) if sig else "(...)"
                        formatted_sig = self._format_signature(sig_str)
                        content += f"{func_name}{formatted_sig}\n"
                        if i < len(func.signatures):
                            content += "\n"
                else:
                    content += f"{func_name}(...) # Polymorphic function\n"
            elif hasattr(func, 'signature') and func.signature:
                # Pixeltable CallableFunction stores signature as a string
                sig_str = str(func.signature)
                # Format signature with line breaks after commas for readability
                formatted_sig = self._format_signature(sig_str)
                content += f"{func_name}{formatted_sig}\n"
            else:
                # Fall back to standard introspection
                sig = inspect.signature(func)
                # Format signature with line breaks after commas for readability
                formatted_sig = self._format_signature(str(sig))
                content += f"{func_name}{formatted_sig}\n"
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
    
    def _wrap_code_line(self, line: str) -> str:
        """Wrap code lines beautifully at logical break points."""
        # Handle function calls with parameters
        if '(' in line and ')' in line:
            # Find the function name and opening paren
            func_start = line[:line.index('(') + 1]
            params = line[line.index('(') + 1:line.rindex(')')]
            func_end = line[line.rindex(')'):]
            
            # Split parameters at commas
            param_parts = []
            depth = 0
            current = []
            
            for char in params:
                if char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    param_parts.append(''.join(current).strip())
                    current = []
                    continue
                current.append(char)
            
            if current:
                param_parts.append(''.join(current).strip())
            
            # Reconstruct with wrapping
            if len(param_parts) > 1:
                wrapped = func_start + '\n'
                for i, param in enumerate(param_parts):
                    wrapped += '    ' + param
                    if i < len(param_parts) - 1:
                        wrapped += ',\n'
                wrapped += '\n' + func_end
                return wrapped
        
        return line
    
    def _run_doctest(self, code: str, module_context: dict = None) -> dict:
        """
        Stub method for running doctest examples and validating their output.

        This would use Python's doctest module to execute the code and verify output.
        For now, this is just a stub for future implementation.

        Args:
            code: The doctest code string (with >>> and ... prompts)
            module_context: Dict of globals/locals to run the code in (e.g., imported modules)

        Returns:
            Dict with:
                - 'passed': bool indicating if the test passed
                - 'output': actual output from running the code
                - 'expected': expected output from the doctest
                - 'error': any error message if the test failed

        Implementation notes:
            1. Parse the doctest string to extract code and expected output
            2. Create a temporary module or namespace
            3. Execute the code using doctest.run_docstring_examples() or similar
            4. Compare actual vs expected output
            5. Return results

        Example usage:
            result = self._run_doctest(
                ">>> 2 + 2\\n4",
                module_context={'numpy': numpy}
            )
            if result['passed']:
                print("Doctest passed!")
        """
        # TODO: Implement actual doctest execution
        # For now, just return a stub response
        return {
            'passed': True,
            'output': '',
            'expected': '',
            'error': None,
            'warning': 'Doctest execution not yet implemented'
        }

    def _extract_doctest_examples(self, text: str) -> list:
        """
        Extract doctest examples from text, similar to docstring_parser's Numpy example parser's approach.

        Returns a list of dicts with 'description', 'code', and 'output' keys.
        """
        import re

        examples = []
        lines = text.split('\n')
        current_example = {'description': '', 'code': '', 'output': ''}
        in_code = False
        code_lines = []

        for i, line in enumerate(lines):
            # Check if line starts a doctest
            if line.strip().startswith('>>>'):
                # If we were collecting output, save the previous example
                if current_example['code']:
                    current_example['code'] = '\n'.join(code_lines)
                    examples.append(current_example)
                    current_example = {'description': '', 'code': '', 'output': ''}
                    code_lines = []

                in_code = True
                # Remove the >>> prompt and add to code
                code_line = line.strip()[3:].lstrip()
                code_lines.append(code_line)

            elif line.strip().startswith('...'):
                # Continuation of previous code line
                if in_code:
                    # Remove the ... prompt and add to code
                    code_line = line.strip()[3:].lstrip()
                    code_lines.append(code_line)

            elif in_code:
                # This is either output or the end of the code block
                if line.strip() == '':
                    # Empty line might mean end of example
                    # Check if next line is another >>> or regular text
                    if i + 1 < len(lines) and not lines[i+1].strip().startswith(('>>>', '...')):
                        in_code = False
                        current_example['code'] = '\n'.join(code_lines)
                        code_lines = []
                else:
                    # This is output from the previous command
                    current_example['output'] += line + '\n'

            else:
                # Regular descriptive text
                if not in_code and line.strip():
                    current_example['description'] += line.strip() + ' '

        # Don't forget the last example
        if code_lines:
            current_example['code'] = '\n'.join(code_lines)
            examples.append(current_example)
        elif current_example['description']:
            examples.append(current_example)

        return examples

    def _format_examples_from_meta(self, examples_meta) -> str:
        """Format examples from parsed meta using improved doctest extraction."""
        if not examples_meta:
            return ""

        content = "\n**Example:**\n\n"
        content += "<Tip>\n\n"

        for meta in examples_meta:
            if meta.description:
                # Use docstring_parser's Numpy example parser's approach to extract doctest examples
                examples = self._extract_doctest_examples(meta.description)

                for example in examples:
                    if example['description']:
                        content += f"{example['description']}\n"

                    if example['code']:
                        # Format the code without the >>> and ... prompts
                        content += "```python\n"
                        content += example['code']
                        content += "\n```\n"

                    if example['output']:
                        # Optionally show output as a comment or separate block
                        # For now, we'll skip output to keep it clean
                        pass

                    content += "\n"
        
        content += "\n</Tip>\n\n"
        return content
    
    def _extract_and_format_examples(self, doc: str) -> str:
        """Extract and format examples from docstring."""
        # Find the Examples section - use simpler pattern that works!
        import re
        pattern = r'Examples?:(.*?)$'
        match = re.search(pattern, doc, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return ""
        
        examples_text = match.group(2).strip()
        if not examples_text:
            return ""
        
        # Format with glamour but subtle headers!
        content = "\n**Example:**\n\n"
        content += "<Tip>\n\n"
        
        # Clean up the example text
        lines = examples_text.split('\n')
        in_code_block = False
        formatted_lines = []
        
        for line in lines:
            # Check if this looks like code (starts with >>> or has significant indentation)
            if line.strip().startswith('>>>') or (line and line[0] == ' ' * 4):
                if not in_code_block:
                    formatted_lines.append('\n```python')
                    in_code_block = True
                # Clean up the >>> prompt if present
                clean_line = line.replace('>>>', '').replace('...', '   ')
                formatted_lines.append(clean_line)
            else:
                if in_code_block:
                    formatted_lines.append('```\n')
                    in_code_block = False
                if line.strip():  # Only add non-empty lines outside code blocks
                    formatted_lines.append(line.strip())
        
        if in_code_block:
            formatted_lines.append('```')
        
        content += '\n'.join(formatted_lines)
        content += "\n\n</Tip>\n\n"
        
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
    
    def _format_signature(self, sig_str: str) -> str:
        """Format signature with line breaks after commas - ALWAYS."""
        # ALWAYS add line breaks after commas for consistency
        # But preserve commas inside type annotations like Dict[str, Any]
        result = []
        depth = 0
        current = []
        
        for char in sig_str:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 1:  # Only break at top-level commas
                current.append(char)
                result.append(''.join(current))
                result.append('\n    ')  # Indent continuation
                current = []
                continue
                
            current.append(char)
        
        if current:
            result.append(''.join(current))
            
        return ''.join(result)