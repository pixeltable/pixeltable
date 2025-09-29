"""Base class for all page generators."""

import re
import inspect
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Any, List
from docstring_parser import parse as parse_docstring

try:
    import pypandoc

    HAS_PANDOC = True
except ImportError:
    HAS_PANDOC = False


class PageBase:
    """Base page generator with common MDX functionality."""

    def __init__(
        self,
        output_dir: Path,
        version: str = 'main',
        show_errors: bool = True,
        github_repo: str = 'pixeltable/pixeltable',
        github_package_path: str = 'pixeltable',
    ):
        """Initialize with output directory, version, and error display setting."""
        self.output_dir = output_dir
        self.version = version
        self.show_errors = show_errors
        self.github_repo = github_repo
        self.github_package_path = github_package_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_signature_with_ruff(self, sig_str: str, func_name: str = 'func') -> str:
        """Format signature with line breaks after commas.

        This removes extra quotes and ensures consistent formatting with line breaks after commas.
        """
        # Remove quotes from type annotations first
        sig_str = self._remove_type_quotes(sig_str)

        # Now manually format with line breaks after commas
        return self._format_signature_manual(sig_str)

    def _format_signature_manual(self, sig_str: str) -> str:
        """Format signature with line breaks after commas at the top level."""
        if '(' not in sig_str:
            return sig_str

        # Split into parts: before params, params, return type
        open_paren = sig_str.index('(')
        close_paren = sig_str.rindex(')')

        params_str = sig_str[open_paren + 1 : close_paren].strip()
        after_close = sig_str[close_paren + 1 :].strip()

        if not params_str:
            # Empty params
            return f'(){after_close}'

        # Split parameters by comma, respecting nested brackets
        params = self._split_params(params_str)

        if len(params) == 0:
            return f'(){after_close}'
        elif len(params) == 1:
            # Single parameter - no line breaks needed
            return f'({params[0]}){after_close}'
        else:
            # Multiple parameters - break after each comma
            formatted_params = ',\n    '.join(param.strip() for param in params)
            return f'(\n    {formatted_params}\n){after_close}'

    def _split_params(self, params_str: str) -> list:
        """Split parameter string by commas, respecting nested brackets."""
        params = []
        current = []
        depth = 0
        in_string = False
        string_char = None

        for i, char in enumerate(params_str):
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                elif char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    params.append(''.join(current).strip())
                    current = []
                    continue
            elif char == string_char and (i == 0 or params_str[i - 1] != '\\'):
                in_string = False
                string_char = None

            current.append(char)

        if current:
            params.append(''.join(current).strip())

        return params

    def _remove_type_quotes(self, sig_str: str) -> str:
        """Remove quotes around type annotations while preserving default value strings."""
        import re

        # Pattern to match type annotations with quotes: : "Type" or : 'Type'
        # This handles both single and double quotes
        # Match : followed by optional whitespace, then quoted string
        # But not = followed by quoted string (that's a default value)

        # Replace : "Type" with : Type (after colon, remove quotes)
        sig_str = re.sub(r':\s*"([^"]+)"', r': \1', sig_str)
        sig_str = re.sub(r":\s*'([^']+)'", r': \1', sig_str)

        # Also handle -> "ReturnType" and -> 'ReturnType'
        sig_str = re.sub(r'->\s*"([^"]+)"', r'-> \1', sig_str)
        sig_str = re.sub(r"->\s*'([^']+)'", r'-> \1', sig_str)

        return sig_str

    def _format_code_with_ruff(self, code: str) -> str:
        """Format a code snippet using ruff.

        This ensures consistent formatting for code examples in documentation.
        """
        try:
            # Create a temporary Python file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            try:
                # Run ruff format on the temp file
                result = subprocess.run(['ruff', 'format', temp_path], capture_output=True, text=True, timeout=5)

                # Read back the formatted content
                with open(temp_path, 'r') as f:
                    formatted = f.read()

                return formatted.strip()
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
        except Exception:
            # If ruff fails, return original
            return code

    def generate_page(self, module_path: str, parent_groups: List[str], item_type: str) -> Optional[str]:
        """Generate documentation page. Override in subclasses."""
        raise NotImplementedError('Subclasses must implement generate_page')

    def _write_mdx_file(self, name: str, parent_groups: List[str], content: str) -> str:
        """Write MDX content to file and return relative path."""
        # Always write to flat structure in output_dir (no subdirectories)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        filename = f'{self._sanitize_path(name)}.mdx'
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            f.write(content)

        # Return path for docs.json (just the filename, no subdirectories)
        return self._build_docs_json_path(parent_groups, name)

    def _build_docs_json_path(self, parent_groups: List[str], name: str) -> str:
        """Build the path for docs.json (includes docs/sdk/latest prefix)."""
        base_path = 'docs/sdk/latest'
        # Always return flat structure path (no subdirectories)
        return f'{base_path}/{self._sanitize_path(name)}'

    def _build_nav_structure(self, page_path: str, children: List = None, group_name: str = None) -> dict:
        """Build navigation structure for this page.

        Args:
            page_path: Path to the main page
            children: Optional list of child pages or groups
            group_name: If provided, create a group containing the page and children

        Returns:
            Dict with navigation structure or string for simple page
        """
        if children:
            # When there are children, we need a group structure
            if group_name:
                # Create a group with the page as first item, then children
                return {'group': group_name, 'pages': [page_path] + children}
            else:
                # Return just the children with the main page first
                return [page_path] + children
        return page_path  # Simple page, no children

    def _build_nav_group(self, group_name: str, pages: List) -> dict:
        """Build a navigation group.

        Args:
            group_name: Name of the group
            pages: List of pages in the group

        Returns:
            Dict with group structure
        """
        return {'group': group_name, 'pages': pages}

    def _create_warning_page(self, name: str, message: str, icon: str = 'triangle-exclamation') -> str:
        """Create a warning page when documentation is missing."""
        return f"""---
title: "{name}"
description: "Documentation unavailable"
icon: "{icon}"
---

## ⚠️ {message}

<Warning>
Documentation for `{name}` is not available.
</Warning>"""

    def _format_type(self, type_annotation: Any) -> str:
        """Convert Python type annotations to clean strings for MDX documentation."""
        if type_annotation is None:
            return 'Any'

        # Handle string annotations
        if isinstance(type_annotation, str):
            return type_annotation

        # Handle class types (like <class 'str'>)
        if hasattr(type_annotation, '__module__') and hasattr(type_annotation, '__name__'):
            # For built-in types, just use the name
            if type_annotation.__module__ == 'builtins':
                return type_annotation.__name__
            # For other types, include the module
            return f'{type_annotation.__module__}.{type_annotation.__name__}'

        # Get the string representation
        type_str = str(type_annotation)

        # Clean up common patterns that break MDX
        # Remove <class '...'> format
        import re

        type_str = re.sub(r"<class '([^']+)'>", r'\1', type_str)

        # Clean up typing module references
        type_str = type_str.replace('typing.', '')
        type_str = type_str.replace('NoneType', 'None')

        return type_str

    def _get_github_link(self, obj: Any) -> Optional[str]:
        """Get GitHub link to source code."""
        try:
            # For modules, use __name__ to get the module path
            if inspect.ismodule(obj):
                # Convert module name to file path (e.g., pixeltable.functions.openai -> pixeltable/functions/openai.py)
                module_path = obj.__name__.replace('.', '/') + '.py'
                # Check if it's an __init__ file
                source_file = inspect.getsourcefile(obj)
                if source_file and source_file.endswith('__init__.py'):
                    module_path = obj.__name__.replace('.', '/') + '/__init__.py'
                return f'https://github.com/{self.github_repo}/blob/{self.version}/{module_path}#L0'

            # For other objects (functions, classes), get the source location
            source_file = inspect.getsourcefile(obj)
            if not source_file:
                return None

            source_lines, line_number = inspect.getsourcelines(obj)

            # Get the module name from the object
            module = inspect.getmodule(obj)
            if module:
                # Convert module name to file path
                module_path = module.__name__.replace('.', '/') + '.py'
                # Check if it's an __init__ file
                if source_file.endswith('__init__.py'):
                    module_path = module.__name__.replace('.', '/') + '/__init__.py'
                return f'https://github.com/{self.github_repo}/blob/{self.version}/{module_path}#L{line_number}'

            return None
        except (TypeError, OSError):
            return None

    def _format_signature(self, name: str, sig: inspect.Signature) -> str:
        """Format function/method signature with line breaks using ruff."""
        sig_str = str(sig)

        # Use ruff to format the signature
        formatted = self._format_signature_with_ruff(sig_str, name)

        # Add function name back if it was stripped
        if not formatted.startswith('('):
            return formatted
        else:
            return f'{name}{formatted}'

    def _split_parameters(self, params_str: str) -> List[str]:
        """Split parameter string handling nested brackets."""
        params = []
        current = []
        depth = 0
        in_string = False
        quote_char = None

        for char in params_str:
            if not in_string:
                if char in '"\'':
                    quote_char = char
                    in_string = True
                elif char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    params.append(''.join(current).strip())
                    current = []
                    continue
            elif char == quote_char and (not current or current[-1] != '\\'):
                in_string = False

            current.append(char)

        if current:
            params.append(''.join(current).strip())

        return params

    def _escape_mdx(self, text: str) -> str:
        """Escape text for MDX format."""
        if not text:
            return ''

        if HAS_PANDOC:
            try:
                # Use pypandoc for conversion
                escaped = pypandoc.convert_text(text, 'gfm', format='commonmark', extra_args=['--wrap=none'])

                # MDX-specific escaping
                escaped = escaped.replace('{', '\\{').replace('}', '\\}')

                # Convert URLs in angle brackets to markdown links
                escaped = re.sub(r'<(https?://[^>]+)>', r'[\1](\1)', escaped)
                escaped = re.sub(r'<(ftp://[^>]+)>', r'[\1](\1)', escaped)
                escaped = re.sub(r'<(mailto:[^>]+)>', r'[\1](\1)', escaped)

                # Handle non-URL angle brackets
                escaped = re.sub(r'<(?!https?://|ftp://|mailto:)([^>]+)>', r'`\1`', escaped)

                # Handle Sphinx/RST directives like :data:`Quantize.MEDIANCUT`
                escaped = re.sub(r':data:`([^`]+)`', r'`\1`', escaped)
                escaped = re.sub(r':(?:py:)?(?:func|class|meth|attr|mod):`([^`]+)`', r'`\1`', escaped)

                # Fix escaped markdown links like \[`Table`\]\[pixeltable.Table\]
                escaped = re.sub(r'\\\\\[`([^`]+)`\\\\\]\\\\\[([^\]]+)\\\\\]', r'[`\1`](\2)', escaped)
                escaped = re.sub(r'\\\[`([^`]+)`\\\]\\\[([^\]]+)\\\]', r'[`\1`](\2)', escaped)

                return escaped
            except Exception:
                pass

        # Fallback: manual escaping
        # Handle Sphinx/RST directives
        text = re.sub(r':data:`([^`]+)`', r'`\1`', text)
        text = re.sub(r':(?:py:)?(?:func|class|meth|attr|mod):`([^`]+)`', r'`\1`', text)

        # Escape braces for MDX
        text = text.replace('{', '\\{').replace('}', '\\}')

        # Convert URLs in angle brackets to markdown links
        text = re.sub(r'<(https?://[^>]+)>', r'[\1](\1)', text)
        text = re.sub(r'<(ftp://[^>]+)>', r'[\1](\1)', text)
        text = re.sub(r'<(mailto:[^>]+)>', r'[\1](\1)', text)

        # Handle other angle brackets
        text = re.sub(r'<([^>]+)>', r'`\1`', text)

        return text

    def _sanitize_path(self, text: str) -> str:
        """Convert text to valid file path."""
        return text.lower().replace(' ', '-').replace('/', '-').replace('.', '-')

    def _escape_yaml(self, text: str) -> str:
        """Escape text for YAML frontmatter."""
        if not text:
            return ''
        return text.replace('"', "'")

    def _truncate_sidebar_title(self, title: str, max_length: int = 23) -> str:
        """Truncate sidebar title if too long to prevent menu squishing."""
        if len(title) <= max_length:
            return title
        # Clean truncation at 23 characters, no indicator
        return title[:max_length]

    def _format_signature(self, sig_str: str, default_name: str = 'func') -> str:
        """Format signature using consistent formatting with line breaks after commas.

        Args:
            sig_str: The signature string to format
            default_name: Default function name if none is found ('func' or 'method')
        """
        # Extract function name if present
        if '(' in sig_str:
            paren_idx = sig_str.index('(')
            if paren_idx > 0 and sig_str[:paren_idx].replace('_', '').replace('.', '').isalnum():
                func_name = sig_str[:paren_idx]
                sig_part = sig_str[paren_idx:]
                formatted = self._format_signature_with_ruff(sig_part, func_name)
                if formatted.startswith('('):
                    return f'{func_name}{formatted}'
                return formatted

        # No function name, just format the signature
        return self._format_signature_with_ruff(sig_str, default_name)

    def _document_returns(self, parsed) -> str:
        """Document return value - common to both methods and functions."""
        if not parsed.returns:
            return ''

        content = '**Returns:**\n\n'

        return_type = parsed.returns.type_name or 'Any'
        return_desc = parsed.returns.description or 'Return value'

        content += f'- *{return_type}*: {self._escape_mdx(return_desc)}\n\n'
        return content
