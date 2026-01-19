# Microsoft Fabric Integration for Pixeltable

This PR adds Microsoft Fabric integration to Pixeltable, enabling seamless access to Azure OpenAI models within Fabric notebook environments.

## Overview

Microsoft Fabric is a SaaS analytics platform with auto-provisioned Azure OpenAI endpoints and token-based authentication for each customer. This integration allows Fabric users to leverage Pixeltable's AI data infrastructure without managing API keys.

## What's Included

### 1. Core Integration (`pixeltable/functions/fabric.py`)
- **`chat_completions()`**: Chat completions with automatic reasoning model detection
  - Standard models (gpt-4.1, gpt-4.1-mini) use `max_tokens` and `temperature`
  - Reasoning models (gpt-5) use `max_completion_tokens`, no `temperature`
  - Configurable API version with smart defaults

- **`embeddings()`**: Embedding generation with automatic batching (batch_size=32)
  - Supports text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large

- **`FabricRateLimitsInfo`**: Azure OpenAI rate limit handling
  - Parses Azure OpenAI rate limit headers
  - Implements retry logic with exponential backoff

- **Helper functions**:
  - `_get_fabric_config()`: Gets Fabric environment config and auth token
  - `_is_reasoning_model()`: Detects reasoning models dynamically (no hardcoding)
  - `_get_header_info()`: Parses Azure OpenAI rate limit headers

### 2. Tests (`tests/functions/test_fabric.py`)
- Live tests that run only in Fabric environments
- Tests for standard models, reasoning models, embeddings
- Tests for batching, API version overrides, and parameter handling
- Uses `@pytest.mark.remote_api` and environment detection

### 3. Tutorial Notebook (`docs/release/howto/providers/working-with-fabric.ipynb`)
- Comprehensive tutorial with real-world examples:
  - Customer support ticket automation
  - Reasoning tasks with gpt-5
  - Semantic search with embeddings
  - RAG pattern combining embeddings + chat
- Follows existing Pixeltable tutorial patterns

### 4. Configuration (`pyproject.toml`)
- **No dependency changes** - `synapse-ml-fabric` is NOT added to dependencies
- Rationale:
  - Fabric users already have it in their runtime
  - Non-Fabric users can't use it anyway
  - Follows Pixeltable pattern (provider SDKs not in base deps)
  - Runtime check via `env.Env.get().require_package()` provides clear error

### 5. Module Exports (`pixeltable/functions/__init__.py`)
- Added `fabric` to module imports (alphabetically ordered)

## Design Decisions

### 1. No Model Hardcoding
- Models are detected dynamically rather than validated against a hardcoded list
- `_is_reasoning_model()` uses pattern matching (`startswith('gpt-5')` or `'reasoning' in model`)
- Future-proof: supports new Fabric models without code changes

### 2. No Client Registration
- Unlike other providers (OpenAI, Anthropic), Fabric doesn't use `@env.register_client()`
- Fabric auth is per-request token-based, not a persistent client object
- `_get_fabric_config()` is called directly in each function

### 3. Configurable API Versions
- Default API versions:
  - Reasoning models: `2025-04-01-preview`
  - Standard models: `2024-02-15-preview`
- Users can override via `api_version` parameter

### 4. Following Pixeltable Patterns
- Async functions with `@pxt.udf` decorator
- Rate limiting via `resource_pool` and `RateLimitsInfo`
- Same parameter patterns as other providers (`model_kwargs`, `_runtime_ctx`)
- Comprehensive docstrings with examples
- Live testing only (following maintainer guidance)

## Testing Instructions

### Testing in Fabric Notebook

1. **Install Pixeltable from your branch:**
   ```python
   %pip install git+https://github.com/<your-username>/pixeltable.git@add_fabric
   ```

2. **Run basic chat completion test:**
   ```python
   import pixeltable as pxt
   from pixeltable.functions import fabric

   pxt.drop_dir('test_fabric', force=True)
   pxt.create_dir('test_fabric')

   t = pxt.create_table('test_fabric.chat_test', {'input': pxt.String})
   messages = [{'role': 'user', 'content': t.input}]
   t.add_computed_column(
       output=fabric.chat_completions(messages, model='gpt-4.1')
   )
   t.insert(input="What is 2+2?")
   print(t.select(t.input, t.output).collect())
   ```

3. **Test reasoning model (gpt-5):**
   ```python
   t2 = pxt.create_table('test_fabric.reasoning_test', {'input': pxt.String})
   messages = [{'role': 'user', 'content': t2.input}]
   t2.add_computed_column(
       output=fabric.chat_completions(
           messages,
           model='gpt-5',
           model_kwargs={'max_completion_tokens': 500}
       )
   )
   t2.insert(input="Explain recursion in programming.")
   print(t2.select(t2.input, t2.output).collect())
   ```

4. **Test embeddings:**
   ```python
   t3 = pxt.create_table('test_fabric.embed_test', {'text': pxt.String})
   t3.add_computed_column(embed=fabric.embeddings(t3.text))
   t3.insert([
       {'text': 'Hello world'},
       {'text': 'AI is transforming industries'}
   ])
   results = t3.select(t3.text, t3.embed).collect()
   print(f"Embedding dimensions: {len(results['embed'][0])}")
   ```

5. **Test batching (insert 50+ rows):**
   ```python
   texts = [{'text': f'Sample text {i}'} for i in range(60)]
   t3.insert(texts)
   print(f"Total rows: {t3.count()}")
   ```

6. **Run the tutorial notebook:**
   - Copy `docs/release/howto/providers/working-with-fabric.ipynb` to your Fabric workspace
   - Run all cells to verify end-to-end functionality

### Local Testing (Limited)

Tests will be skipped locally since they require Fabric environment:
```bash
pytest tests/functions/test_fabric.py -v
# Expected: All tests skipped with "Not running in Fabric environment"
```

## Files Changed

- **New Files:**
  - `pixeltable/functions/fabric.py` (489 lines)
  - `tests/functions/test_fabric.py` (153 lines)
  - `docs/release/howto/providers/working-with-fabric.ipynb`

- **Modified Files:**
  - `pixeltable/functions/__init__.py` (added `fabric` import)
  - `pyproject.toml` (no dependency changes - see rationale above)

## Alignment with Pixeltable Conventions

✅ Follows existing provider integration patterns (OpenAI, Anthropic, Gemini)
✅ Async-first with proper rate limiting
✅ Comprehensive docstrings with examples
✅ Type hints and error handling
✅ No hardcoded models or API versions (configurable)
✅ Live testing approach (as per maintainer guidance)
✅ Tutorial notebook following established format
✅ Dev dependencies only (no bloat to base install)

## Breaking Changes

None. This is a new integration with no impact on existing functionality.

## Future Enhancements (Not in This PR)

- Tool calling support (will follow OpenAI pattern in future PR)
- Additional Azure OpenAI features as they become available in Fabric
- Conditional return types for embeddings (if needed)

## Questions for Reviewers

1. Should we add any additional validation for model names?
2. Is the rate limiting implementation sufficient for Fabric's Azure OpenAI endpoints?
3. Any preferences on the tutorial notebook structure or examples?

## Related Issues

Closes #985

## Checklist

- [x] Code follows Pixeltable conventions
- [x] Comprehensive docstrings with examples
- [x] Tests created (live tests for Fabric environment)
- [x] Tutorial notebook created
- [x] No unnecessary dependencies added (synapse-ml-fabric already in Fabric runtime)
- [x] Module exports updated
- [ ] Tested in Fabric notebook (pending user testing)
- [ ] CI passes (tests will skip outside Fabric)

---

**Note:** This integration is designed to work exclusively in Microsoft Fabric notebook environments where `synapse-ml-fabric` is available and authentication is handled automatically.
