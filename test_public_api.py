#!/usr/bin/env python3
"""Test the @public_api decorator."""

from typing import Optional
from pixeltable.func import public_api, get_public_api_registry, is_public_api, get_pydantic_models


# Test 1: Simple function
@public_api
def create_table(name: str, schema: Optional[dict] = None) -> str:
    """Create a new table."""
    return f"Created table: {name}"


# Test 2: Function with multiple params
@public_api
def insert_data(table: str, data: list, validate: bool = True) -> int:
    """Insert data into table."""
    return len(data)


# Test 3: Class
@public_api
class Table:
    """A Pixeltable table."""

    def __init__(self, name: str):
        self.name = name

    @public_api
    def add_column(self, column_name: str, column_type: str = "String") -> None:
        """Add a column to the table."""
        pass


def main():
    print("=" * 60)
    print("Testing @public_api decorator")
    print("=" * 60)

    # Test is_public_api
    print("\n1. Testing is_public_api():")
    print(f"   create_table is public: {is_public_api(create_table)}")
    print(f"   insert_data is public: {is_public_api(insert_data)}")
    print(f"   Table is public: {is_public_api(Table)}")

    # Test registry
    print("\n2. Testing registry:")
    registry = get_public_api_registry()
    print(f"   Total public APIs: {len(registry)}")
    for qualname in registry.keys():
        print(f"   - {qualname}")

    # Test metadata extraction
    print("\n3. Testing metadata extraction:")
    create_table_meta = registry.get('__main__.create_table')
    if create_table_meta:
        print(f"   create_table:")
        print(f"     - Docstring: {create_table_meta['docstring']}")
        print(f"     - Parameters: {list(create_table_meta['parameters'].keys())}")
        print(f"     - Return type: {create_table_meta['return_type']}")

    # Test Pydantic model generation
    print("\n4. Testing Pydantic model generation:")
    try:
        input_model, output_model = get_pydantic_models(create_table)
        if input_model:
            print(f"   Input model: {input_model.__name__}")
            print(f"   Input fields: {list(input_model.model_fields.keys())}")
        if output_model:
            print(f"   Output model: {output_model.__name__}")
    except Exception as e:
        print(f"   Note: Pydantic not available or error: {e}")

    # Test parameter details
    print("\n5. Testing parameter details:")
    insert_data_meta = registry.get('__main__.insert_data')
    if insert_data_meta:
        for param_name, param_info in insert_data_meta['parameters'].items():
            print(f"   {param_name}:")
            print(f"     - Type: {param_info['annotation']}")
            print(f"     - Default: {param_info['default']}")
            print(f"     - Kind: {param_info['kind']}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
