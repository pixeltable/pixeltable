# AGENTS.md

Instructions for AI coding agents working with the Pixeltable codebase.

## Project Overview

Pixeltable is an open-source Python library providing declarative data infrastructure for building multimodal AI applications. It enables incremental storage, transformation, indexing, retrieval, and orchestration of data across images, video, audio, and documents.

**Key concepts:**
- **Tables**: Store structured and unstructured data with native multimodal column types (`pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`)
- **Computed Columns**: Declaratively define transformations that run automatically on new/updated data
- **Views**: Virtual tables with iterators for efficient data processing (e.g., document chunking, video frame extraction)
- **Embedding Indexes**: Built-in vector search with automatic index maintenance
- **UDFs**: User-defined functions with `@pxt.udf` and `@pxt.query` decorators
- **AI Integrations**: Built-in functions for OpenAI, Anthropic, Hugging Face, and many more

**Documentation**: https://docs.pixeltable.com/
**SDK Reference**: https://docs.pixeltable.com/sdk/latest/pixeltable

## Repository Structure

```
pixeltable/
├── pixeltable/           # Main library source code
│   ├── catalog/          # Table, View, Column metadata and operations
│   ├── exec/             # Query execution engine
│   ├── exprs/            # Expression types and operators
│   ├── func/             # UDF/query function infrastructure
│   ├── functions/        # Built-in AI provider integrations (openai, anthropic, etc.)
│   ├── index/            # Embedding index implementations
│   ├── io/               # Import/export (CSV, Parquet, Hugging Face, etc.)
│   ├── iterators/        # View iterators (DocumentSplitter, FrameIterator, etc.)
│   ├── metadata/         # Schema migration and persistence
│   ├── share/            # Data sharing (publish/replicate)
│   └── utils/            # Utilities
├── tests/                # Test suite
│   ├── functions/        # Tests for AI integrations
│   ├── io/               # Tests for import/export
│   └── data/             # Test fixtures (images, videos, documents)
├── docs/
│   ├── release/          # Mintlify documentation source (notebooks, MDX)
│   ├── _guidelines/      # Documentation style guides
│   └── sample-apps/      # Example applications
└── tool/                 # Development utilities
```

## Setup Commands

**Prerequisites:** Miniconda with a dedicated environment (not `base`)

```bash
# Create and activate conda environment
conda create --name pxt python=3.10
conda activate pxt

# Install development dependencies
make install

# Run tests (excludes expensive/remote API tests by default)
make test

# Run full test suite including notebooks
make fulltest

# Run minimal test suite for quick checks
make slimtest
```

## Development Workflow

### Code Style

- **Line length**: 120 characters
- **Quotes**: Single quotes (`'`) preferred
- **Formatter**: ruff (`make format`)
- **Type hints**: Required for all functions (mypy strict mode)
- **Imports**: Group by standard library → third-party → pixeltable

```bash
# Format code
make format

# Run static checks (mypy + ruff)
make check

# Individual checks
make typecheck    # mypy
make lint         # ruff check
make formatcheck  # ruff format --check
```

### Testing

```bash
# Run pytest (excludes expensive/remote_api tests)
make pytest

# Run full pytest including expensive tests
make fullpytest

# Run specific test file
pytest tests/test_table.py -v

# Run tests matching a pattern
pytest -k "test_insert" -v

# Run with remote API tests (requires credentials)
pytest -m "remote_api" tests/functions/test_openai.py
```

**Test markers:**
- `@pytest.mark.expensive` - Long-running tests
- `@pytest.mark.remote_api` - Tests calling external APIs
- `@pytest.mark.corrupts_db` - Tests that modify database state destructively

### Creating a Pull Request

1. Create a branch from `main`
2. Make changes and add tests in `tests/`
3. Run `make format` to format code
4. Run `make check` to verify static checks pass
5. Run `make test` to run the test suite
6. Push and create PR via GitHub

## Code Conventions

### Adding a New UDF

UDFs go in `pixeltable/functions/`. Each provider has its own module (e.g., `openai.py`, `anthropic.py`).

```python
# pixeltable/functions/my_provider.py
import pixeltable as pxt

@pxt.udf
def my_function(input_text: str, model: str = 'default-model') -> str:
    """
    Brief description of what this function does.

    Args:
        input_text: The input text to process.
        model: The model to use for processing.

    Returns:
        The processed output text.

    Example:
        ```python
        t.add_computed_column(result=my_function(t.text, model='advanced'))
        ```
    """
    # Implementation
    pass
```

**Important patterns:**
- Use `@pxt.udf` for scalar functions, `@pxt.uda` for aggregates
- Use `.using()` for model parameterization in embedding functions
- Add tests in `tests/functions/test_my_provider.py`

### Adding a New Iterator

Iterators go in `pixeltable/iterators/`. They split rows into multiple output rows.

```python
# pixeltable/iterators/my_iterator.py
from pixeltable.iterators import ComponentIterator
import pixeltable.type_system as ts

class MyIterator(ComponentIterator):
    """Iterator that splits X into multiple rows."""

    def __init__(self, input_data: SomeType):
        # Initialize iteration state
        pass

    def __next__(self) -> dict[str, Any]:
        # Return next row as dict
        pass

    def close(self) -> None:
        pass

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {'input_data': ts.SomeType()}

    @classmethod
    def output_schema(cls, *args, **kwargs) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {'output_field': ts.SomeType()}, []
```

### Working with Tables

```python
import pixeltable as pxt

# Create table with schema
t = pxt.create_table('my_dir.my_table', {
    'text': pxt.String,
    'image': pxt.Image,
    'metadata': pxt.Json,
})

# Add computed columns
t.add_computed_column(embedding=some_embedding_fn(t.text))
t.add_computed_column(analysis=some_analysis_fn(t.image))

# Add embedding index
t.add_embedding_index('text', embedding=embed_fn)

# Insert data
t.insert([{'text': 'hello', 'image': 'path/to/image.jpg'}])

# Query with similarity search
sim = t.text.similarity(string='search query')
results = t.order_by(sim, asc=False).limit(10).select(t.text, sim).collect()
```

### Error Handling

- Use `pixeltable.exceptions` for custom exceptions
- Validate inputs early and provide clear error messages
- Use `exn.Error` for user-facing errors

## Documentation

### Notebooks

Documentation notebooks are in `docs/release/`. Follow `docs/_guidelines/GUIDELINES_FOR_NOTEBOOKS.md`:

- Start with YAML frontmatter in a **Raw cell** (not Markdown)
- No H1 headers in markdown (title comes from frontmatter)
- Use `##` for main sections, `###` for subsections
- Clear outputs before committing unless output is instructive
- Use `raw.githubusercontent.com` for GitHub raw links

### Docstrings

Follow `docs/_guidelines/GUIDELINES_FOR_DOCSTRINGS.md`:

- Code fences must be on their own lines
- All code examples must be in fenced code blocks
- Backticks must be properly paired
- HTML tags must be self-closing

### Building Docs

```bash
# Build documentation
make docs

# Serve locally for development
make docs-serve

# Deploy to staging
make docs-deploy TARGET=stage
```

## Testing Against Remote APIs

For tests that call external APIs (OpenAI, Anthropic, etc.):

1. Set appropriate environment variables (e.g., `OPENAI_API_KEY`)
2. Run with the `remote_api` marker: `pytest -m "remote_api" tests/functions/test_openai.py`
3. These tests are excluded from CI by default

## Database and Storage

- Pixeltable uses embedded PostgreSQL at `~/.pixeltable/pgdata`
- Generated media stored at `~/.pixeltable/media`
- File cache at `~/.pixeltable/file_cache`
- **Never directly modify** files in `~/.pixeltable`; use the SDK

To reset the database for testing:
```bash
scripts/drop-pxt-db.sh
```

## Common Patterns

### Idempotent Operations

Use `if_exists='ignore'` or `if_not_exists=True` for idempotent schema operations:

```python
pxt.create_dir('my_dir', if_exists='ignore')
pxt.create_table('my_dir.table', schema, if_exists='ignore')
t.add_computed_column(col=expr, if_exists='ignore')
t.add_embedding_index('col', embedding=fn, if_not_exists=True)
```

### Query Functions

Encapsulate complex queries as reusable functions:

```python
@pxt.query
def search_documents(query_text: str, limit: int = 10):
    sim = docs.text.similarity(string=query_text)
    return docs.order_by(sim, asc=False).limit(limit).select(docs.text, sim)
```

### Handling Nullable Columns

Check for null values when processing data:

```python
@pxt.udf
def safe_process(value: Optional[str]) -> str:
    if value is None:
        return ''
    return process(value)
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `pixeltable/__init__.py` | Public API exports |
| `pixeltable/catalog/table.py` | Table class implementation |
| `pixeltable/catalog/view.py` | View class implementation |
| `pixeltable/func/udf.py` | UDF decorator implementation |
| `pixeltable/functions/` | AI provider integrations |
| `pixeltable/io/` | Import/export functionality |
| `pyproject.toml` | Dependencies and tool config |
| `Makefile` | Build and test commands |

## Getting Help

- **Documentation**: https://docs.pixeltable.com/
- **GitHub Issues**: https://github.com/pixeltable/pixeltable/issues
- **Discord**: https://discord.gg/QPyqFYx2UN
- **Discussions**: https://github.com/pixeltable/pixeltable/discussions
