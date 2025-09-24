"""Base class for all page generators."""

import re
import inspect
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

    def __init__(self, output_dir: Path, version: str = "main", show_errors: bool = True,
                 github_repo: str = "pixeltable/pixeltable", github_package_path: str = "pixeltable"):
        """Initialize with output directory, version, and error display setting."""
        self.output_dir = output_dir
        self.version = version
        self.show_errors = show_errors
        self.github_repo = github_repo
        self.github_package_path = github_package_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_page(self, module_path: str, parent_groups: List[str], item_type: str) -> Optional[str]:
        """Generate documentation page. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate_page")
    
    def _write_mdx_file(self, name: str, parent_groups: List[str], content: str) -> str:
        """Write MDX content to file and return relative path."""
        # Always write to flat structure in output_dir (no subdirectories)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        filename = f"{self._sanitize_path(name)}.mdx"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            f.write(content)

        # Return path for docs.json (just the filename, no subdirectories)
        return self._build_docs_json_path(parent_groups, name)
    
    def _build_docs_json_path(self, parent_groups: List[str], name: str) -> str:
        """Build the path for docs.json (includes docs/sdk/latest prefix)."""
        base_path = "docs/sdk/latest"
        # Always return flat structure path (no subdirectories)
        return f"{base_path}/{self._sanitize_path(name)}"
    
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
                return {
                    "group": group_name,
                    "pages": [page_path] + children
                }
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
        return {
            "group": group_name,
            "pages": pages
        }
    
    def _create_warning_page(self, name: str, message: str, icon: str = "triangle-exclamation") -> str:
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
                return f"https://github.com/{self.github_repo}/blob/{self.version}/{module_path}#L0"

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
                return f"https://github.com/{self.github_repo}/blob/{self.version}/{module_path}#L{line_number}"

            return None
        except (TypeError, OSError):
            return None
    
    def _format_signature(self, name: str, sig: inspect.Signature) -> str:
        """Format function/method signature with line breaks."""
        sig_str = str(sig)
        
        # Check for return type
        return_type = ""
        if ' -> ' in sig_str:
            sig_part, return_type = sig_str.rsplit(' -> ', 1)
            return_type = f" -> {return_type}"
        else:
            sig_part = sig_str
        
        # Format with line breaks if long
        full_sig = f"{name}{sig_part}{return_type}"
        if len(full_sig) <= 100:
            return full_sig
        
        # Parse parameters for multi-line format
        if sig_part.startswith('(') and sig_part.endswith(')'):
            params_str = sig_part[1:-1]
            if params_str:
                params = self._split_parameters(params_str)
                if len(params) > 1:
                    formatted_params = ',\n    '.join(params)
                    return f"{name}(\n    {formatted_params}\n){return_type}"
        
        return full_sig
    
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
                escaped = pypandoc.convert_text(
                    text,
                    'gfm',
                    format='commonmark',
                    extra_args=['--wrap=none']
                )
                
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