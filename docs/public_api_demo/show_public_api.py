#!/usr/bin/env python3
"""
Print the public API registry in a readable format.

Usage:
    python scripts/show_public_api.py
"""

from pprint import pprint
import pixeltable as pxt
from pixeltable.func.public_api import get_public_api_registry

registry = get_public_api_registry()

print(f"Public API Registry - {len(registry)} total APIs")
print("=" * 80)
print()

for func_name, metadata in sorted(registry.items()):
    # Remove the function object itself (not printable)
    printable = {k: v for k, v in metadata.items() if k != 'object'}

    print(f"{func_name}:")
    pprint(printable, width=80, indent=2)
    print()
