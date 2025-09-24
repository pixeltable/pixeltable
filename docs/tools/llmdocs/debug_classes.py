#!/usr/bin/env python3
"""Debug why classes aren't being recognized."""

import xml.etree.ElementTree as ET
from pathlib import Path

def debug_opml():
    """Check what's happening with class entries in OPML."""
    
    opml_path = Path("mintlifier.opml")
    tree = ET.parse(opml_path)
    root = tree.getroot()
    
    print("Searching for class entries in OPML...\n")
    print("=" * 60)
    
    # Find all entries with 'class|' 
    for outline in root.iter('outline'):
        text = outline.get('text', '')
        if 'class|' in text:
            print(f"\nFound class entry: {text}")
            
            # Check for children
            children = []
            for child in outline:
                child_text = child.get('text', '')
                if child_text:
                    children.append(child_text)
            
            if children:
                print(f"  Children ({len(children)}):")
                for child in children[:5]:
                    print(f"    - {child}")
                if len(children) > 5:
                    print(f"    ... and {len(children) - 5} more")
            else:
                print("  No children specified")
    
    print("\n" + "=" * 60)
    
    # Now check what OPMLReader does with them
    from opml_reader import OPMLReader
    
    reader = OPMLReader(opml_path)
    reader._backup_file = lambda: None  # Skip backup
    reader.tree = tree
    reader.root = root
    reader.structure = reader._process_root()
    
    all_pages = reader.get_all_pages()
    
    print(f"\nTotal pages from OPMLReader: {len(all_pages)}")
    
    # Find class pages
    class_pages = [p for p in all_pages if p.item_type == 'class']
    print(f"Class pages found: {len(class_pages)}")
    
    for page in class_pages:
        print(f"\n  Class: {page.module_path}")
        print(f"  Parent groups: {page.parent_groups}")
        if page.children:
            print(f"  Children ({len(page.children)}): {', '.join(page.children[:5])}")
            if len(page.children) > 5:
                print(f"    ... and {len(page.children) - 5} more")
    
    # Check if they're being misclassified
    print("\n\nChecking for pixeltable.Table and pixeltable.DataFrame...")
    for page in all_pages:
        if 'Table' in page.module_path or 'DataFrame' in page.module_path:
            print(f"  Found: {page.module_path} (type: {page.item_type})")

if __name__ == "__main__":
    debug_opml()