#!/usr/bin/env python3
"""Test DataFrame items to see what type they are."""

import inspect
import sys
sys.path.insert(0, '/Users/lux/repos/pixeltable')

try:
    import pixeltable
    from pixeltable import DataFrame
    
    # Test items that have missing icons
    test_items = [
        'parameters',
        'to_coco_dataset', 
        'validate_constant_type',
        'where',
        # Compare with working ones
        'sample',
        'show',
        'tail'
    ]
    
    print("DataFrame item inspection:")
    print("=" * 60)
    
    for item_name in test_items:
        if hasattr(DataFrame, item_name):
            item = getattr(DataFrame, item_name)
            print(f"\n{item_name}:")
            print(f"  Type: {type(item)}")
            print(f"  Is method: {inspect.ismethod(item)}")
            print(f"  Is function: {inspect.isfunction(item)}")
            print(f"  Is property: {isinstance(item, property)}")
            print(f"  Is callable: {callable(item)}")
            
            # Check if it's a descriptor
            if hasattr(item, '__get__'):
                print(f"  Is descriptor: True")
            
            # Check module
            if hasattr(item, '__module__'):
                print(f"  Module: {item.__module__}")
                
        else:
            print(f"\n{item_name}: NOT FOUND")
    
    print("\n" + "=" * 60)
    
except ImportError as e:
    print(f"Could not import pixeltable: {e}")
    print("Please ensure pixeltable is installed in the current environment")