#!/usr/bin/env python3
"""Generate a compact codebase index for Claude Code.

Parses all Python files under pixeltable/ using AST to extract:
- Classes (with base classes, key methods, line numbers)
- Top-level functions (with signatures, line numbers)
- Module-level docstrings

Output: CODEBASE_INDEX.md — a compact reference file that Claude Code
can read to navigate the codebase without expensive exploration.

Usage:
    python tool/generate_codebase_index.py
    python tool/generate_codebase_index.py --root pixeltable --output CODEBASE_INDEX.md
"""

import argparse
import ast
from pathlib import Path


def get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a compact function signature."""
    args = []

    # defaults are right-aligned across posonlyargs + args combined
    all_positional = node.args.posonlyargs + node.args.args
    defaults_offset = len(all_positional) - len(node.args.defaults)

    for i, arg in enumerate(all_positional):
        if arg.arg in {'self', 'cls'}:
            continue
        annotation = ''
        if arg.annotation:
            annotation = f': {ast.unparse(arg.annotation)}'
        default = ''
        default_idx = i - defaults_offset
        if 0 <= default_idx < len(node.args.defaults):
            default = f' = {ast.unparse(node.args.defaults[default_idx])}'
        args.append(f'{arg.arg}{annotation}{default}')
        # Emit the `/` marker after the last positional-only arg
        if node.args.posonlyargs and i == len(node.args.posonlyargs) - 1:
            args.append('/')

    if node.args.vararg:
        args.append(f'*{node.args.vararg.arg}')
    if node.args.kwonlyargs:
        if not node.args.vararg:
            args.append('*')
        for j, kw in enumerate(node.args.kwonlyargs):
            annotation = f': {ast.unparse(kw.annotation)}' if kw.annotation else ''
            default = ''
            if j < len(node.args.kw_defaults) and node.args.kw_defaults[j]:
                default = f' = {ast.unparse(node.args.kw_defaults[j])}'
            args.append(f'{kw.arg}{annotation}{default}')
    if node.args.kwarg:
        args.append(f'**{node.args.kwarg.arg}')

    ret = ''
    if node.returns:
        ret = f' -> {ast.unparse(node.returns)}'

    sig = ', '.join(args)
    # Truncate very long signatures
    if len(sig) > 120:
        sig = sig[:117] + '...'
    return f'({sig}){ret}'


def get_first_docstring_line(node: ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef | ast.Module) -> str:
    """Get the first line of a docstring, if present."""
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.strip().split('\n')[0]
        if len(first_line) > 100:
            first_line = first_line[:97] + '...'
        return first_line
    return ''


def get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names."""
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(f'@{dec.id}')
        elif isinstance(dec, ast.Attribute):
            decorators.append(f'@{ast.unparse(dec)}')
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                decorators.append(f'@{dec.func.id}')
            elif isinstance(dec.func, ast.Attribute):
                decorators.append(f'@{ast.unparse(dec.func)}')
    return decorators


def analyze_file(filepath: Path) -> dict | None:
    """Analyze a single Python file and return its structure."""
    try:
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return None

    module_doc = get_first_docstring_line(tree)

    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith('_'):
                continue

            bases = [ast.unparse(b) for b in node.bases]
            base_str = f'({", ".join(bases)})' if bases else ''
            doc = get_first_docstring_line(node)
            decorators = get_decorators(node)

            # Extract public methods (non-dunder, non-private)
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith('_') and not item.name.startswith('__'):
                        continue
                    if item.name in ('__init__', '__repr__', '__str__', '__hash__', '__eq__'):
                        continue
                    method_dec = get_decorators(item)
                    method_sig = get_signature(item)
                    method_doc = get_first_docstring_line(item)
                    methods.append(
                        {
                            'name': item.name,
                            'line': item.lineno,
                            'signature': method_sig,
                            'decorators': method_dec,
                            'doc': method_doc,
                        }
                    )

            classes.append(
                {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': base_str,
                    'doc': doc,
                    'decorators': decorators,
                    'methods': methods,
                }
            )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('_'):
                continue
            doc = get_first_docstring_line(node)
            decorators = get_decorators(node)
            sig = get_signature(node)
            functions.append(
                {'name': node.name, 'line': node.lineno, 'signature': sig, 'decorators': decorators, 'doc': doc}
            )

    if not classes and not functions and not module_doc:
        return None

    return {'module_doc': module_doc, 'classes': classes, 'functions': functions}


def generate_index(root: Path) -> str:
    """Generate the full codebase index as markdown."""
    lines = [
        '# Codebase Index',
        '',
        '<!-- Auto-generated by tool/generate_codebase_index.py — do not edit manually -->',
        '',
        'Compact reference of all public classes and functions in the codebase.',
        'Use this to quickly locate code without reading full files.',
        '',
    ]

    py_files = sorted(root.rglob('*.py'))

    # Group files by directory, preserving order of first appearance
    dir_files: dict[str, list[tuple[Path, Path]]] = {}
    for filepath in py_files:
        parts = filepath.parts
        if any(p in ('__pycache__', '.git', 'node_modules') for p in parts):
            continue
        rel = filepath.relative_to(root.parent)
        file_dir = str(rel.parent)
        dir_files.setdefault(file_dir, []).append((filepath, rel))

    for file_dir, files in dir_files.items():
        # Collect analyzed results for this directory, skipping empty files
        dir_entries = []
        for filepath, rel in files:
            result = analyze_file(filepath)
            if result is not None:
                dir_entries.append((rel, result))

        if not dir_entries:
            continue

        lines.append(f'## {file_dir}/')
        lines.append('')

        for rel, result in dir_entries:
            # File header
            file_label = f'### {rel.name}'
            if result['module_doc']:
                file_label += f' — {result["module_doc"]}'
            lines.append(file_label)
            lines.append('')

            # Classes
            for cls in result['classes']:
                dec_str = ' '.join(cls['decorators'])
                if dec_str:
                    dec_str = f' {dec_str}'
                doc_str = f' — {cls["doc"]}' if cls['doc'] else ''
                lines.append(f'- **class {cls["name"]}{cls["bases"]}** (L{cls["line"]}){dec_str}{doc_str}')

                for method in cls['methods']:
                    mdec = ' '.join(method['decorators'])
                    if mdec:
                        mdec = f' {mdec}'
                    mdoc = f' — {method["doc"]}' if method['doc'] else ''
                    lines.append(f'  - `{method["name"]}{method["signature"]}`{mdec} (L{method["line"]}){mdoc}')

            # Top-level functions
            for fn in result['functions']:
                dec_str = ' '.join(fn['decorators'])
                if dec_str:
                    dec_str = f' {dec_str}'
                doc_str = f' — {fn["doc"]}' if fn['doc'] else ''
                lines.append(f'- `{fn["name"]}{fn["signature"]}`{dec_str} (L{fn["line"]}){doc_str}')

            lines.append('')

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate codebase index for Claude Code')
    parser.add_argument('--root', type=Path, default=Path('pixeltable'), help='Root directory to analyze')
    parser.add_argument('--output', type=Path, default=Path('.claude/CODEBASE_INDEX.md'), help='Output file path')
    args = parser.parse_args()

    if not args.root.exists():
        print(f'Error: {args.root} does not exist')
        raise SystemExit(1)

    index = generate_index(args.root)
    args.output.write_text(index, encoding='utf-8')

    # Stats
    line_count = len(index.split('\n'))
    size_kb = len(index.encode('utf-8')) / 1024
    print(f'Generated {args.output} ({line_count} lines, {size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
