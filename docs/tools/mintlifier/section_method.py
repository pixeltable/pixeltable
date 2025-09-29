"""Generate documentation sections for methods within class pages."""

import inspect
from typing import Any, List, Optional
from docstring_parser import parse as parse_docstring
from page_base import PageBase


class MethodSectionGenerator(PageBase):
    """Generate documentation sections for methods within class pages."""

    def __init__(self, show_errors: bool = True):
        """Initialize the section generator."""
        # Don't initialize PageBase with output_dir since we're not writing files
        self.show_errors = show_errors

    def generate_section(self, method: Any, method_name: str, class_name: str) -> str:
        """Generate method documentation section for inline use.

        Args:
            method: The method object to document
            method_name: Name of the method
            class_name: Name of the parent class

        Returns:
            Markdown string for the method documentation
        """
        # Skip constructors as per Marcel's feedback
        if method_name == '__init__':
            return ''

        # Build section content with elegant visual separation
        content = '\n---\n\n'  # Beautiful horizontal divider
        content += f'### `{method_name}()`\n\n'

        # Add description
        doc = inspect.getdoc(method) or ''
        if doc:
            parsed = parse_docstring(doc)
            if parsed and parsed.short_description:
                content += f'{self._escape_mdx(parsed.short_description)}\n\n'
            if parsed.long_description:
                content += f'{self._escape_mdx(parsed.long_description)}\n\n'

        # Add signature
        content += self._document_signature(method, method_name)

        # Add parameters
        if doc:
            content += self._document_parameters(method, doc)

        # Add returns
        if doc:
            parsed = parse_docstring(doc)
            if parsed and parsed.returns:
                content += self._document_returns(parsed)

        # Add examples using docstring_parser meta
        if doc:
            parsed = parse_docstring(doc)
            examples_meta = [m for m in parsed.meta if m.args and 'examples' in m.args[0].lower()]
            if examples_meta:
                content += self._format_examples_from_meta(examples_meta)

        return content

    def _format_signature(self, sig_str: str) -> str:
        """Format method signature - delegates to base class."""
        return super()._format_signature(sig_str, default_name='method')

    def _document_signature(self, method: Any, method_name: str) -> str:
        """Document method signature."""
        content = '**Signature:**\n\n```python\n'

        try:
            # Check for Pixeltable's custom signature attribute first
            if hasattr(method, 'signature') and method.signature:
                # Pixeltable CallableFunction stores signature as a string
                sig_str = str(method.signature)
                # For methods, remove 'self:' from signature display
                if sig_str.startswith('(self:'):
                    sig_str = '(' + sig_str[6:]
                    if sig_str.startswith('(,'):
                        sig_str = '(' + sig_str[2:]  # Remove leading comma
                # Format signature with line breaks after commas for readability
                formatted_sig = self._format_signature(sig_str)
                content += f'{method_name}{formatted_sig}\n'
            else:
                # Standard introspection
                sig = inspect.signature(method)
                # Remove self from signature for display
                params = list(sig.parameters.values())
                if params and params[0].name == 'self':
                    params = params[1:]
                new_sig = sig.replace(parameters=params)
                # Format signature with line breaks after commas for readability
                formatted_sig = self._format_signature(str(new_sig))
                content += f'{method_name}{formatted_sig}\n'
        except (ValueError, TypeError):
            content += f'{method_name}(...)\n'

        content += '```\n\n'
        return content

    def _format_examples_from_meta(self, examples_meta) -> str:
        """Format examples from parsed meta - beautiful and consistent!"""
        if not examples_meta:
            return ''

        # Format with subtle headers like Parameters
        content = '\n**Example:**\n\n'

        for meta in examples_meta:
            if meta.description:
                # Clean up the description and format code blocks
                lines = meta.description.split('\n')
                formatted = []
                in_code = False
                code_lines = []

                # Process lines elegantly
                for i, line in enumerate(lines):
                    # Detect code lines (>>> or indented)
                    if line.strip().startswith('>>>') or line.strip().startswith('...'):
                        if not in_code:
                            in_code = True
                        # Remove prompt markers and collect code
                        clean = line.replace('>>>', '').replace('...', '   ').rstrip()
                        code_lines.append(clean)
                    elif in_code and line.strip() == '':
                        # Empty line might end code block
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith(('>>>', '...')):
                            # End of code block - format it with ruff
                            code_text = '\n'.join(code_lines)
                            formatted_code = self._format_code_with_ruff(code_text)
                            formatted.append('```python')
                            formatted.append(formatted_code)
                            formatted.append('```')
                            in_code = False
                            code_lines = []
                    else:
                        if in_code:
                            # End of code block - format it with ruff
                            code_text = '\n'.join(code_lines)
                            formatted_code = self._format_code_with_ruff(code_text)
                            formatted.append('```python')
                            formatted.append(formatted_code)
                            formatted.append('```')
                            in_code = False
                            code_lines = []
                        if line.strip():
                            formatted.append(line.strip())

                if in_code:
                    # Format remaining code
                    code_text = '\n'.join(code_lines)
                    formatted_code = self._format_code_with_ruff(code_text)
                    formatted.append('```python')
                    formatted.append(formatted_code)
                    formatted.append('```')

                content += '\n'.join(formatted) + '\n'

        return content

    def _document_parameters(self, method: Any, doc: str) -> str:
        """Document method parameters."""
        parsed = parse_docstring(doc)
        if not parsed or not parsed.params:
            return ''

        content = '**Parameters:**\n\n'

        # Get signature for default values
        params_with_defaults = {}
        params_with_types = {}
        try:
            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.default != inspect.Parameter.empty:
                    params_with_defaults[param_name] = param.default
                if param.annotation != inspect.Parameter.empty:
                    params_with_types[param_name] = param.annotation
        except (ValueError, TypeError):
            pass

        for param in parsed.params:
            # Skip self parameter
            if param.arg_name == 'self':
                continue

            # Format parameter
            type_str = params_with_types.get(param.arg_name, param.type_name or 'Any')
            # Clean up type strings that break MDX
            if type_str:
                import re

                # Convert to string first if it's not already
                type_str = str(type_str)
                # Remove <class '...'> format
                type_str = re.sub(r"<class '([^']+)'>", r'\1', type_str)
            default = params_with_defaults.get(param.arg_name)

            content += f'- **`{param.arg_name}`** '
            if type_str:
                content += f'(*{type_str}*)'
            if default is not None:
                content += f' = `{default}`'
            content += f': {self._escape_mdx(param.description) if param.description else "No description"}\n'

        content += '\n'
        return content

    def _document_examples(self, parsed) -> str:
        """Document examples."""
        if not hasattr(parsed, 'examples') or not parsed.examples:
            return ''

        content = '**Examples:**\n\n'

        for example in parsed.examples:
            if example.description:
                content += f'{self._escape_mdx(example.description)}\n\n'
            if example.snippet:
                content += f'```python\n{example.snippet}\n```\n\n'

        return content
