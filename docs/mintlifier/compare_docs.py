#!/usr/bin/env python3
"""
Compare mkdocs vs mintlifier documentation coverage
"""

import os
from pathlib import Path
import json

def get_mkdocs_api_items():
    """Get all API items from mkdocs structure"""
    mkdocs_api = Path("/Users/lux/repos/pixeltable-docs/docs/api/pixeltable")
    items = set()
    
    # Check main pixeltable.md for listed functions
    pixeltable_md = mkdocs_api / "pixeltable.md"
    if pixeltable_md.exists():
        content = pixeltable_md.read_text()
        # Extract function names from markdown links
        import re
        pattern = r'\[`pxt\.([^`]+)`\]'
        matches = re.findall(pattern, content)
        for match in matches:
            items.add(f"pixeltable.{match}")
    
    # Check functions directory
    functions_dir = mkdocs_api / "functions"
    if functions_dir.exists():
        for file in functions_dir.glob("*.md"):
            module_name = file.stem
            items.add(f"pixeltable.functions.{module_name}")
    
    # Check for DataFrame and Table
    if (mkdocs_api / "data-frame.md").exists():
        items.add("pixeltable.DataFrame")
    if (mkdocs_api / "table.md").exists():
        items.add("pixeltable.Table")
    
    return items

def get_mintlifier_items():
    """Get all items from mintlifier output"""
    mintlify_sdk = Path("/Users/lux/repos/pixeltable-docs/docs/mintlify/docs/sdk/latest")
    items = set()
    
    def scan_directory(path, prefix=""):
        for item in path.iterdir():
            if item.is_dir():
                # Build module path
                module_part = item.name
                new_prefix = f"{prefix}.{module_part}" if prefix else module_part
                
                # Check for module.mdx file
                module_mdx = item / f"{module_part}.mdx"
                if not module_mdx.exists():
                    # Try parent name
                    parent_name = item.parent.name
                    module_mdx = item / f"{parent_name}.mdx"
                
                # Add the module/class/function
                if prefix == "core-api" and module_part == "pixeltable":
                    items.add("pixeltable")
                    # Scan subdirectory for functions/classes
                    for subitem in item.iterdir():
                        if subitem.is_file() and subitem.suffix == ".mdx":
                            name = subitem.stem
                            items.add(f"pixeltable.{name}")
                        elif subitem.is_dir():
                            # This is a class with methods
                            class_name = subitem.name
                            items.add(f"pixeltable.{class_name}")
                            for method_file in subitem.glob("*.mdx"):
                                method_name = method_file.stem
                                items.add(f"pixeltable.{class_name}.{method_name}")
                
                # Recursively scan
                scan_directory(item, new_prefix)
            elif item.suffix == ".mdx":
                name = item.stem
                if prefix:
                    items.add(f"{prefix}.{name}")
                else:
                    items.add(name)
    
    # Start scanning from core-api
    core_api = mintlify_sdk / "core-api"
    if core_api.exists():
        scan_directory(core_api, "")
    
    return items

def compare_docs():
    """Compare mkdocs vs mintlifier coverage"""
    mkdocs_items = get_mkdocs_api_items()
    mintlifier_items = get_mintlifier_items()
    
    print("=" * 60)
    print("DOCUMENTATION COMPARISON: MkDocs vs Mintlifier")
    print("=" * 60)
    
    print(f"\n📚 MkDocs API items: {len(mkdocs_items)}")
    print(f"📄 Mintlifier items: {len(mintlifier_items)}")
    
    # Find differences
    only_mkdocs = mkdocs_items - mintlifier_items
    only_mintlifier = mintlifier_items - mkdocs_items
    common = mkdocs_items & mintlifier_items
    
    print(f"\n✅ Common items: {len(common)}")
    
    if only_mkdocs:
        print(f"\n⚠️  Only in MkDocs ({len(only_mkdocs)} items):")
        for item in sorted(only_mkdocs)[:10]:
            print(f"   - {item}")
        if len(only_mkdocs) > 10:
            print(f"   ... and {len(only_mkdocs) - 10} more")
    
    if only_mintlifier:
        print(f"\n⚠️  Only in Mintlifier ({len(only_mintlifier)} items):")
        for item in sorted(only_mintlifier)[:20]:
            print(f"   - {item}")
        if len(only_mintlifier) > 20:
            print(f"   ... and {len(only_mintlifier) - 20} more")
    
    # Check the OPML whitelist
    print("\n" + "=" * 60)
    print("CHECKING OVERPRODUCTION")
    print("=" * 60)
    
    # Load OPML to see what's whitelisted
    opml_path = Path("/Users/lux/repos/pixeltable-docs/docs/mintlifier/mintlifier.opml")
    if opml_path.exists():
        content = opml_path.read_text()
        
        # Count different types
        import re
        modules = len(re.findall(r'text="module\|', content))
        classes = len(re.findall(r'text="class\|', content))
        functions = len(re.findall(r'text="func\|', content))
        methods = len(re.findall(r'text="method\|', content))
        
        print(f"\n📋 OPML Whitelist contains:")
        print(f"   - Modules: {modules}")
        print(f"   - Classes: {classes}")
        print(f"   - Functions: {functions}")
        print(f"   - Methods: {methods}")
        print(f"   - Total: {modules + classes + functions + methods}")
    
    # Sample comparison of actual content
    print("\n" + "=" * 60)
    print("SAMPLE CONTENT COMPARISON")
    print("=" * 60)
    
    # Compare create_table documentation
    print("\n📝 Comparing 'create_table' documentation:")
    
    # Check mintlifier version
    mintlify_create_table = Path("/Users/lux/repos/pixeltable-docs/docs/mintlify/docs/sdk/latest/core-api/pixeltable/create_table.mdx")
    if mintlify_create_table.exists():
        content = mintlify_create_table.read_text()
        has_description = "description:" in content
        has_signature = "## Signature" in content or "```python" in content
        has_args = "## Args" in content
        has_examples = "## Example" in content
        has_warnings = "## ⚠️" in content or "<Warning>" in content
        
        print(f"   Mintlifier version:")
        print(f"     ✓ Description: {has_description}")
        print(f"     ✓ Signature: {has_signature}")
        print(f"     ✓ Arguments: {has_args}")
        print(f"     {'✓' if has_examples else '✗'} Examples: {has_examples}")
        print(f"     {'⚠️' if has_warnings else '✓'} Has warnings: {has_warnings}")

if __name__ == "__main__":
    compare_docs()