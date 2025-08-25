#!/usr/bin/env python3
"""Test that OPML filtering is working correctly."""

from pathlib import Path
from opml_reader import OPMLReader

def test_opml_filtering():
    """Test that children are properly extracted from OPML."""
    
    opml_path = Path("mintlifier.opml")
    reader = OPMLReader(opml_path)
    
    # Load without creating backup
    reader._backup_file = lambda: None  # Skip backup for testing
    reader.tree = ET.parse(reader.opml_path)
    reader.root = reader.tree.getroot()
    reader.structure = reader._process_root()
    
    all_pages = reader.get_all_pages()
    
    print("Testing OPML child filtering:\n")
    print("=" * 60)
    
    # Find modules with children
    modules_with_children = [p for p in all_pages if p.item_type == 'module' and p.children]
    print(f"\nModules with specific children: {len(modules_with_children)}")
    
    for page in modules_with_children[:3]:  # Show first 3
        print(f"\n  Module: {page.module_path}")
        print(f"  Children ({len(page.children)}): {', '.join(page.children[:5])}")
        if len(page.children) > 5:
            print(f"    ... and {len(page.children) - 5} more")
    
    # Find classes with children
    classes_with_children = [p for p in all_pages if p.item_type == 'class' and p.children]
    print(f"\n\nClasses with specific methods: {len(classes_with_children)}")
    
    for page in classes_with_children[:3]:  # Show first 3
        print(f"\n  Class: {page.module_path}")
        print(f"  Methods ({len(page.children)}): {', '.join(page.children[:5])}")
        if len(page.children) > 5:
            print(f"    ... and {len(page.children) - 5} more")
    
    # Find modules/classes without children (should document everything)
    modules_without = [p for p in all_pages if p.item_type == 'module' and not p.children]
    classes_without = [p for p in all_pages if p.item_type == 'class' and not p.children]
    
    print(f"\n\nModules that will document everything: {len(modules_without)}")
    if modules_without:
        print(f"  Examples: {', '.join([p.module_path for p in modules_without[:3]])}")
    
    print(f"\nClasses that will document everything: {len(classes_without)}")
    if classes_without:
        print(f"  Examples: {', '.join([p.module_path for p in classes_without[:3]])}")
    
    print("\n" + "=" * 60)
    print("✅ Filtering logic test complete")

if __name__ == "__main__":
    import xml.etree.ElementTree as ET
    test_opml_filtering()