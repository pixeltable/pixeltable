# Pixeltable TypeScript SDK

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
