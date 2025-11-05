# TypeScript SDK Design

## Overview

The TypeScript SDK is auto-generated from Pixeltable's `@public_api` decorator registry. This ensures the SDK always stays in sync with the Python API.

## How It Works

### 1. Public API Discovery

The `@public_api` decorator in `pixeltable/func/public_api.py` automatically registers:
- Function signatures with parameter types and defaults
- Return types
- Docstrings
- Source file locations

### 2. Type Serialization

Pixeltable's `ColumnType` class provides built-in serialization:

```python
# Python
column_type = ImageType(width=640, height=480)
serialized = column_type.as_dict()
# {'_classname': 'ImageType', 'width': 640, 'height': 480, 'nullable': False}
```

This serialization is used to generate TypeScript type definitions:

```typescript
// TypeScript
interface PixeltableImage {
  width?: number;
  height?: number;
  data?: string;  // Base64 or URL
  mimeType?: string;
}
```

### 3. Type Mapping

The generator maps Python types to TypeScript:

| Python Type | TypeScript Type |
|-------------|----------------|
| `str` | `string` |
| `int`, `float` | `number` |
| `bool` | `boolean` |
| `Optional[T]` | `T \| null` |
| `List[T]` | `T[]` |
| `Dict[K, V]` | `Record<K, V>` |
| `ImageType` | `PixeltableImage` |
| `VideoType` | `PixeltableVideo` |
| `AudioType` | `PixeltableAudio` |
| `DocumentType` | `PixeltableDocument` |
| `JsonType` | `Record<string, any>` |
| `ArrayType` | `Array<any>` |

### 4. Generated Files

- **`pixeltable.d.ts`** - TypeScript type definitions
  - Interfaces for Pixeltable classes (Table, DataFrame, etc.)
  - Media type interfaces (Image, Video, Audio, Document)
  - Function signatures

- **`pixeltable.ts`** - SDK implementation (placeholder)
  - HTTP client wrapper
  - Method stubs for all public API functions
  - You need to implement the actual server communication

## Usage Example

```typescript
import PixeltableClient from './pixeltable';

const client = new PixeltableClient('http://localhost:8000', 'api-key');

// Create a table
const table = await client.create_table('video_analysis', {
  schema: {
    video: { type: 'Video' },
    frame_count: { type: 'Int' }
  }
});

// Insert data
await client.insert(table, {
  video: { data: 'base64_encoded_or_url' },
  frame_count: 120
});

// Query
const results = await client.query(table, { limit: 10 });
```

## Leveraging Pixeltable's Serialization

The SDK generator uses Pixeltable's built-in methods:

1. **`ColumnType.as_dict()`** - Serialize type metadata to JSON
2. **`ColumnType.serialize()`** - Full JSON serialization
3. **`get_public_api_registry()`** - Get all public APIs with metadata

This means:
- ✅ Type information is always accurate
- ✅ No manual type mapping needed
- ✅ SDK stays in sync with Python API
- ✅ Can leverage Pixeltable's type system directly

## Future Enhancements

### Server-Side
To make this SDK useful, you'll need a Pixeltable HTTP server that:
- Accepts JSON requests
- Uses `ColumnType.deserialize()` to reconstruct types
- Executes Pixeltable operations
- Returns JSON responses

### Client-Side
- Add retry logic and error handling
- Support streaming for large media files
- Add authentication methods (OAuth, JWT, etc.)
- Generate SDK for other languages (Go, Rust, etc.)

### Code Generation
- Generate method implementations from docstrings
- Add validation based on type constraints
- Generate tests from examples
- Add JSDoc comments from Python docstrings

## Regenerating

To regenerate the SDK after API changes:

```bash
# Activate conda environment with pixeltable installed
conda activate pxt

# Run the generator
python scripts/generate_typescript_sdk.py
```

The generator will:
1. Import pixeltable (triggering `@public_api` registration)
2. Call `get_public_api_registry()` to get all APIs
3. Use `ColumnType.as_dict()` for type serialization
4. Generate TypeScript files in `docs/typescript_sdk/`
