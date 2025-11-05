#!/usr/bin/env python3
"""
Generate TypeScript SDK from Pixeltable's public API.

This script:
1. Imports pixeltable to trigger @public_api decorator registration
2. Retrieves the public API registry
3. Generates TypeScript type definitions and SDK
4. Leverages Pixeltable's built-in serialize() methods for type conversion

Usage:
    # Option 1: Using conda environment
    conda activate pxt
    python scripts/generate_typescript_sdk.py

    # Option 2: Using the repo in development mode
    PYTHONPATH=. python scripts/generate_typescript_sdk.py

Output:
    docs/typescript_sdk/pixeltable.d.ts - Type definitions
    docs/typescript_sdk/pixeltable.ts - SDK implementation

Features:
    - Automatically discovers all @public_api decorated functions/classes
    - Uses ColumnType.as_dict() to serialize Pixeltable types
    - Maps Python types to TypeScript equivalents
    - Generates interfaces for Pixeltable classes (Table, DataFrame, etc.)
    - Creates TypeScript definitions for media types (Image, Video, Audio, Document)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional, get_origin, get_args

# Import pixeltable to populate the public API registry
import pixeltable as pxt
from pixeltable.func.public_api import get_public_api_registry
from pixeltable.type_system import ColumnType


class TypeScriptGenerator:
    """Generates TypeScript SDK from Pixeltable's public API."""

    # Map Python built-in types to TypeScript
    PYTHON_TO_TS_TYPES = {
        'str': 'string',
        'int': 'number',
        'float': 'number',
        'bool': 'boolean',
        'None': 'null',
        'NoneType': 'null',
        'Any': 'any',
        'dict': 'Record<string, any>',
        'list': 'any[]',
        'tuple': 'any[]',
        'datetime': 'Date',
        'date': 'Date',
    }

    # Map Pixeltable ColumnType to TypeScript
    PIXELTABLE_TYPE_TO_TS = {
        'StringType': 'string',
        'IntType': 'number',
        'FloatType': 'number',
        'BoolType': 'boolean',
        'TimestampType': 'Date',
        'DateType': 'Date',
        'JsonType': 'Record<string, any>',
        'ArrayType': 'Array<any>',
        'ImageType': 'PixeltableImage',
        'VideoType': 'PixeltableVideo',
        'AudioType': 'PixeltableAudio',
        'DocumentType': 'PixeltableDocument',
    }

    def __init__(self):
        self.registry = get_public_api_registry()
        self.custom_types: set[str] = set()
        self.interfaces: dict[str, str] = {}

    def python_type_to_typescript(self, py_type: Any) -> str:
        """
        Convert Python type annotation to TypeScript type.
        Handles Optional, List, Dict, Union, and Pixeltable types.
        """
        if py_type is None or py_type == inspect.Parameter.empty:
            return 'any'

        # Handle string type annotations (for forward references)
        if isinstance(py_type, str):
            # Check if it's a known Pixeltable type
            if py_type in self.PIXELTABLE_TYPE_TO_TS:
                return self.PIXELTABLE_TYPE_TO_TS[py_type]
            # Check if it's a class we know about
            if py_type in self.registry:
                self.custom_types.add(py_type)
                return py_type.split('.')[-1]
            return 'any'

        # Handle Pixeltable ColumnType instances
        if isinstance(py_type, ColumnType):
            # Use the serialize method to get type info
            type_dict = py_type.as_dict()
            classname = type_dict.get('_classname', '')
            ts_type = self.PIXELTABLE_TYPE_TO_TS.get(classname, 'any')

            # Handle nullability
            if type_dict.get('nullable', False):
                return f'{ts_type} | null'
            return ts_type

        # Get the type name
        type_name = getattr(py_type, '__name__', str(py_type))

        # Handle built-in types
        if type_name in self.PYTHON_TO_TS_TYPES:
            return self.PYTHON_TO_TS_TYPES[type_name]

        # Handle typing generics
        origin = get_origin(py_type)
        args = get_args(py_type)

        if origin is not None:
            # Optional[T] -> T | null
            if origin is type(None) or (hasattr(origin, '__name__') and origin.__name__ == 'UnionType'):
                if args:
                    # Union types
                    ts_types = [self.python_type_to_typescript(arg) for arg in args]
                    # Remove duplicates and 'null'
                    unique_types = []
                    has_null = False
                    for t in ts_types:
                        if t == 'null':
                            has_null = True
                        elif t not in unique_types:
                            unique_types.append(t)
                    result = ' | '.join(unique_types)
                    if has_null:
                        result += ' | null'
                    return result

            # Handle typing.Union (Python 3.9 style)
            origin_name = getattr(origin, '__name__', str(origin))
            if 'Union' in str(origin):
                ts_types = [self.python_type_to_typescript(arg) for arg in args]
                return ' | '.join(ts_types)

            # List[T] -> T[]
            if origin_name in ('list', 'List', 'Sequence'):
                if args:
                    element_type = self.python_type_to_typescript(args[0])
                    return f'{element_type}[]'
                return 'any[]'

            # Dict[K, V] -> Record<K, V>
            if origin_name in ('dict', 'Dict', 'Mapping'):
                if len(args) >= 2:
                    key_type = self.python_type_to_typescript(args[0])
                    val_type = self.python_type_to_typescript(args[1])
                    return f'Record<{key_type}, {val_type}>'
                return 'Record<string, any>'

            # Tuple -> array
            if origin_name in ('tuple', 'Tuple'):
                if args:
                    types = [self.python_type_to_typescript(arg) for arg in args]
                    return f'[{", ".join(types)}]'
                return 'any[]'

        # Check if it's a class in our registry (custom Pixeltable type)
        if hasattr(py_type, '__module__') and hasattr(py_type, '__name__'):
            full_name = f"{py_type.__module__}.{py_type.__name__}"
            if full_name in self.registry:
                self.custom_types.add(full_name)
                return py_type.__name__

        # Default to any
        return 'any'

    def generate_pixeltable_media_types(self) -> str:
        """Generate TypeScript interfaces for Pixeltable media types."""
        return '''
// Pixeltable Media Types
// These represent the serialized form of Pixeltable media objects

export interface PixeltableImage {
  /** Base64-encoded image data or URL */
  data?: string;
  /** Image width in pixels */
  width?: number;
  /** Image height in pixels */
  height?: number;
  /** Image mode (e.g., 'RGB', 'RGBA', 'L') */
  mode?: string;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableVideo {
  /** Base64-encoded video data or URL */
  data?: string;
  /** Video duration in seconds */
  duration?: number;
  /** Video width in pixels */
  width?: number;
  /** Video height in pixels */
  height?: number;
  /** Frame rate */
  fps?: number;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableAudio {
  /** Base64-encoded audio data or URL */
  data?: string;
  /** Audio duration in seconds */
  duration?: number;
  /** Sample rate in Hz */
  sampleRate?: number;
  /** Number of channels */
  channels?: number;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableDocument {
  /** Base64-encoded document data or URL */
  data?: string;
  /** Document text content */
  text?: string;
  /** MIME type */
  mimeType?: string;
  /** Number of pages (for PDFs) */
  pages?: number;
}

export interface PixeltableColumnType {
  /** The type class name */
  _classname: string;
  /** Whether the column is nullable */
  nullable: boolean;
  /** Additional type-specific properties */
  [key: string]: any;
}
'''

    def generate_function_signature(
        self,
        name: str,
        metadata: dict[str, Any],
        is_method: bool = False
    ) -> str:
        """Generate TypeScript function signature from metadata."""
        params = metadata.get('parameters', {})
        return_type = metadata.get('return_type')

        # Build parameter list
        # TypeScript requires required parameters before optional ones
        required_params = []
        optional_params = []

        for param_name, param_info in params.items():
            # Skip 'self' and 'cls' for methods
            if is_method and param_name in ('self', 'cls'):
                continue

            param_type = self.python_type_to_typescript(param_info['annotation'])
            has_default = param_info.get('default') is not None

            if has_default:
                optional_params.append(f'{param_name}?: {param_type}')
            else:
                required_params.append(f'{param_name}: {param_type}')

        # Combine: required first, then optional
        ts_params = required_params + optional_params
        param_str = ', '.join(ts_params)

        # Convert return type
        ts_return_type = self.python_type_to_typescript(return_type)

        # Add documentation
        docstring = metadata.get('docstring', '')
        doc_lines = []
        if docstring:
            doc_lines.append('  /**')
            for line in docstring.split('\n'):
                doc_lines.append(f'   * {line}')
            doc_lines.append('   */')

        func_signature = f'  {name}({param_str}): Promise<{ts_return_type}>;'

        if doc_lines:
            return '\n'.join(doc_lines) + '\n' + func_signature
        return func_signature

    def generate_class_interface(self, class_name: str, metadata: dict[str, Any]) -> str:
        """Generate TypeScript interface for a Pixeltable class."""
        docstring = metadata.get('docstring', '')

        lines = []
        if docstring:
            lines.append('/**')
            for line in docstring.split('\n')[:5]:  # First 5 lines of docstring
                lines.append(f' * {line}')
            lines.append(' */')

        lines.append(f'export interface {class_name} {{')

        # Find all methods for this class
        class_qualname = metadata['qualname']
        for full_name, item_meta in self.registry.items():
            if item_meta.get('is_method') and full_name.startswith(f"{metadata['module']}.{class_qualname}."):
                method_name = item_meta['name']
                method_sig = self.generate_function_signature(method_name, item_meta, is_method=True)
                lines.append(method_sig)

        lines.append('}')
        return '\n'.join(lines)

    def generate_type_definitions(self) -> str:
        """Generate complete TypeScript type definitions (.d.ts)."""
        lines = [
            '// Generated TypeScript definitions for Pixeltable',
            '// This file is auto-generated from the Pixeltable public API',
            '',
            self.generate_pixeltable_media_types(),
            '',
        ]

        # Generate interfaces for classes
        for full_name, metadata in sorted(self.registry.items()):
            if metadata.get('is_class'):
                class_name = metadata['name']
                interface = self.generate_class_interface(class_name, metadata)
                lines.append(interface)
                lines.append('')

        # Generate function declarations (not methods)
        lines.append('// Pixeltable Public API Functions')
        lines.append('export interface Pixeltable {')

        for full_name, metadata in sorted(self.registry.items()):
            if metadata.get('is_function') and not metadata.get('is_method'):
                func_name = metadata['name']
                func_sig = self.generate_function_signature(func_name, metadata)
                lines.append(func_sig)
                lines.append('')

        lines.append('}')
        lines.append('')
        lines.append('declare const pixeltable: Pixeltable;')
        lines.append('export default pixeltable;')

        return '\n'.join(lines)

    def generate_sdk_implementation(self) -> str:
        """Generate TypeScript SDK implementation (.ts)."""
        lines = [
            '// Generated TypeScript SDK for Pixeltable',
            '// This file is auto-generated from the Pixeltable public API',
            '',
            '// This is a placeholder implementation.',
            '// You will need to implement the actual API client logic',
            '// (e.g., HTTP requests to a Pixeltable server)',
            '',
            'class PixeltableClient {',
            '  private baseUrl: string;',
            '  private apiKey?: string;',
            '',
            '  constructor(baseUrl: string, apiKey?: string) {',
            '    this.baseUrl = baseUrl;',
            '    this.apiKey = apiKey;',
            '  }',
            '',
            '  private async request<T>(endpoint: string, method: string, body?: any): Promise<T> {',
            '    const headers: Record<string, string> = {',
            '      "Content-Type": "application/json",',
            '    };',
            '',
            '    if (this.apiKey) {',
            '      headers["Authorization"] = `Bearer ${this.apiKey}`;',
            '    }',
            '',
            '    const response = await fetch(`${this.baseUrl}${endpoint}`, {',
            '      method,',
            '      headers,',
            '      body: body ? JSON.stringify(body) : undefined,',
            '    });',
            '',
            '    if (!response.ok) {',
            '      throw new Error(`Pixeltable API error: ${response.statusText}`);',
            '    }',
            '',
            '    return response.json();',
            '  }',
            '',
        ]

        # Generate method stubs for each public function
        for full_name, metadata in sorted(self.registry.items()):
            if metadata.get('is_function') and not metadata.get('is_method'):
                func_name = metadata['name']
                params = metadata.get('parameters', {})

                # Build parameter list
                param_names = [p for p in params.keys() if p not in ('self', 'cls')]
                param_str = ', '.join([f'{p}: any' for p in param_names])

                lines.append(f'  async {func_name}({param_str}): Promise<any> {{')
                lines.append(f'    // TODO: Implement {func_name}')
                lines.append(f'    return this.request(`/api/{func_name}`, "POST", {{ {", ".join(param_names)} }});')
                lines.append('  }')
                lines.append('')

        lines.append('}')
        lines.append('')
        lines.append('export default PixeltableClient;')

        return '\n'.join(lines)


def main():
    """Generate TypeScript SDK files."""
    print("Generating TypeScript SDK from Pixeltable public API...")

    generator = TypeScriptGenerator()

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'docs' / 'typescript_sdk'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate type definitions
    type_defs = generator.generate_type_definitions()
    type_defs_path = output_dir / 'pixeltable.d.ts'
    type_defs_path.write_text(type_defs)
    print(f"✓ Generated type definitions: {type_defs_path}")

    # Generate SDK implementation
    sdk_impl = generator.generate_sdk_implementation()
    sdk_impl_path = output_dir / 'pixeltable.ts'
    sdk_impl_path.write_text(sdk_impl)
    print(f"✓ Generated SDK implementation: {sdk_impl_path}")

    # Generate a README
    readme_content = '''# Pixeltable TypeScript SDK

This directory contains auto-generated TypeScript SDK for Pixeltable.

## Files

- `pixeltable.d.ts` - TypeScript type definitions
- `pixeltable.ts` - SDK implementation (placeholder)

## Usage

```typescript
import PixeltableClient from './pixeltable';

const client = new PixeltableClient('http://localhost:8000', 'your-api-key');

// Use the client
const result = await client.create_table('my_table', { schema: { /* ... */ } });
```

## Note

This SDK is generated from Pixeltable's `@public_api` decorator registry.
The implementation is a placeholder - you'll need to implement the actual
API communication logic based on your Pixeltable server setup.

## Regenerating

To regenerate this SDK:

```bash
python scripts/generate_typescript_sdk.py
```
'''

    readme_path = output_dir / 'README.md'
    readme_path.write_text(readme_content)
    print(f"✓ Generated README: {readme_path}")

    print(f"\n✅ TypeScript SDK generated successfully!")
    print(f"   Total APIs: {len(generator.registry)}")
    print(f"   Custom types referenced: {len(generator.custom_types)}")


if __name__ == '__main__':
    main()
