#!/usr/bin/env python3
"""Test the flattened navigation structure."""

import json
from pathlib import Path

# Read the generated docs.json
docs_json_path = Path("../mintlify/docs.json")
if not docs_json_path.exists():
    print("docs.json not found. Run mintlifier.py first.")
    exit(1)

with open(docs_json_path) as f:
    docs = json.load(f)

# Find the SDK tab
sdk_tab = None
for tab in docs.get('tabs', []):
    if 'Pixeltable SDK' in str(tab):
        sdk_tab = tab
        break

if not sdk_tab:
    print("SDK tab not found")
    exit(1)

def print_structure(item, level=0):
    """Recursively print the navigation structure."""
    indent = "  " * level
    
    if isinstance(item, str):
        # It's a page
        name = item.split('/')[-1]
        print(f"{indent}📄 {name}")
    elif isinstance(item, dict):
        if 'group' in item:
            # It's a group/folder
            print(f"{indent}📁 {item['group']}")
            for page in item.get('pages', []):
                print_structure(page, level + 1)
        elif 'dropdown' in item:
            # It's a dropdown
            print(f"{indent}▼ {item['dropdown']}")
            for group in item.get('groups', []):
                print_structure(group, level + 1)
            for page in item.get('pages', []):
                print_structure(page, level + 1)

print("SDK Navigation Structure:")
print("=" * 60)

# Check the latest dropdown
dropdowns = sdk_tab.get('dropdowns', [])
for dropdown in dropdowns:
    if dropdown.get('dropdown') == 'latest':
        print_structure(dropdown)
        break

print("\n" + "=" * 60)
print("\nChecking for intermediate groups like 'Functions', 'Methods', 'Classes'...")

# Search for these unwanted intermediate groups
json_str = json.dumps(docs)
issues = []
if '"group": "Functions"' in json_str:
    issues.append("❌ Found 'Functions' group - should be flattened")
if '"group": "Methods"' in json_str:
    issues.append("❌ Found 'Methods' group - should be flattened")
if '"group": "Classes"' in json_str:
    issues.append("❌ Found 'Classes' group - should be flattened")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ No intermediate grouping folders found!")