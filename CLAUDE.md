# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pixeltable is an AI data infrastructure platform that provides declarative, multimodal, and incremental data processing capabilities. It combines database functionality with AI/ML model orchestration in a unified table interface.

## Development Commands

### Setup and Installation

- `conda create --name pxt python=3.10` - Create conda environment (Python 3.10 required for dev)
- `conda activate pxt` - Activate environment
- `make install` - Install development environment (includes uv, ffmpeg, dependencies, jupyter kernel)
- `make clean` - Remove generated files and temp files

### Testing

- `make test` - Run pytest, stresstest, and check (recommended for most changes)
- `make fulltest` - Run fullpytest, nbtest, stresstest, and check (comprehensive testing)
- `make pytest` - Run pytest only (excludes expensive tests)
- `make fullpytest` - Run pytest including expensive tests
- `make nbtest` - Run pytest on notebooks
- `make stresstest` - Run stress tests (random table operations)

**Testing Tips:**
- Tests use parallel execution (`-n auto --dist loadgroup --maxprocesses 6`)
- Each test worker gets its own database (`test_{worker_id}`)
- Tests automatically retry on conflicts (`--reruns 2`)
- Use markers: `-m "not remote_api and not expensive"` to exclude API/expensive tests

### Code Quality

- `make check` - Run typecheck, docscheck, lint, and formatcheck
- `make typecheck` - Run mypy type checking
- `make lint` - Run ruff linting
- `make format` - Run ruff formatting (modifies files)
- `make formatcheck` - Check formatting without modifying
- `make docscheck` - Run mkdocs build --strict

### Documentation

- `mkdocs serve` - Serve docs locally for development
- `make release-docs` - Build and deploy API documentation
- **API docs**: Generated from code at `/docs/api/`
- **User docs**: Mintlify-based docs at `/docs/mintlify/`
- **Notebooks**: Jupyter notebooks at `/docs/notebooks/`

### Release

- `make release` - Create PyPI release and post to GitHub

## Architecture Overview

Pixeltable is built around a unified table interface that handles both structured and unstructured multimodal data with automatic AI/ML processing.

### Core Components

- **Tables & Views** (`pixeltable/catalog/`) - Primary data abstraction with versioning and lineage
- **Type System** (`pixeltable/type_system.py`) - Specialized column types (Image, Video, Audio, Document, etc.)
- **Expression System** (`pixeltable/exprs/`) - SQL-like expressions with AI/ML function composition
- **UDF Framework** (`pixeltable/func/`) - User-defined functions with @pxt.udf decorator
- **AI/ML Functions** (`pixeltable/functions/`) - Built-in integrations (OpenAI, Anthropic, HuggingFace, etc.)
- **Storage Layer** (`pixeltable/store.py`) - PostgreSQL backend with media file management
- **Execution Engine** (`pixeltable/exec/`) - Incremental computation and batch processing

### Data Model

```python
# Tables support multimodal column types
t = pxt.create_table('media', {
    'img': pxt.Image,           # PIL.Image.Image in memory, stored as file URL + metadata  
    'video': pxt.Video,         # Local path or URL
    'text': pxt.String,         # Standard string
    'embeddings': pxt.Array     # NumPy array
})

# Computed columns define processing pipelines
t.add_computed_column(
    classification=huggingface.vit_for_image_classification(t.img)
)

# Embedding indexes enable similarity search
t.add_embedding_index('img', embedding=clip.using(model_id='openai/clip-vit-base-patch32'))
sim = t.img.similarity("cat playing with yarn")
```

### Key Architectural Patterns

1. **Declarative Processing** - Define computations once, they run automatically on new data
2. **Incremental Computation** - Only recompute what's necessary when data/code changes
3. **Unified Multimodal Interface** - Same API for text, images, video, audio, documents
4. **Type-Safe Expressions** - Strong typing throughout the expression system
5. **Resource Pooling** - Automatic rate limiting and resource management for AI APIs
6. **Versioning & Lineage** - Full history tracking for tables, schemas, and data

## Development Guidelines

### Code Quality Standards

**All Python code MUST include type hints and return types.**

```python
# Good
def process_media(img: PIL.Image.Image, threshold: float = 0.5) -> dict[str, Any]:
    """Process image with object detection."""
    pass

# Bad  
def process_media(img, threshold=0.5):
    pass
```

**Follow established patterns:**
- Use descriptive variable names
- Prefer composition over inheritance
- Follow the existing module structure
- Use dataclasses for structured data

### UDF Development

User-defined functions are the primary extensibility mechanism:

```python
@pxt.udf
def my_transform(text: str, multiplier: int = 2) -> str:
    """Transform text by repeating it."""
    return text * multiplier

# Batched processing for efficiency
@pxt.udf(batch_size=32)
def batch_process(inputs: Batch[str]) -> Batch[dict]:
    """Process multiple inputs efficiently."""
    return [{'result': inp.upper()} for inp in inputs]

# Async for I/O operations
@pxt.udf
async def api_call(prompt: str) -> dict:
    """Make async API call."""
    async with httpx.AsyncClient() as client:
        response = await client.post('/api', json={'prompt': prompt})
        return response.json()
```

### AI/ML Integration Patterns

Functions in `pixeltable/functions/` follow consistent patterns:

```python
# Client registration
@env.register_client('provider_name')
def _(api_key: str, base_url: Optional[str] = None) -> 'provider.Client':
    return provider.Client(api_key=api_key, base_url=base_url)

# Rate-limited UDF with proper typing
@pxt.udf(resource_pool='request-rate:provider:chat')
async def chat_completions(
    messages: list[dict[str, str]], 
    *, 
    model: str, 
    model_kwargs: Optional[dict[str, Any]] = None
) -> dict:
    """Provider chat completions with automatic rate limiting."""
    client = env.Env.get().get_client('provider_name')
    result = await client.chat.completions.create(
        messages=messages, 
        model=model, 
        **(model_kwargs or {})
    )
    return result.dict()
```

### Testing Patterns

- **Fixtures** - Use provided fixtures like `test_tbl`, `img_tbl`, `reset_db`
- **Test Structure** - Follow `test_*.py` naming in `tests/` directory
- **Mocking** - Mock external APIs in unit tests, use `@pytest.mark.remote_api` for integration tests
- **Cleanup** - Tests automatically clean up database state between runs

```python
def test_udf_functionality(test_tbl):
    """Test custom UDF behavior."""
    @pxt.udf
    def double_value(x: int) -> int:
        return x * 2
    
    test_tbl.add_computed_column(doubled=double_value(test_tbl.c2))
    results = test_tbl.select(test_tbl.doubled).collect()
    assert len(results) > 0
```

### Expression Development

When adding new expression types in `pixeltable/exprs/`:

```python
class MyExpr(Expr):
    def __init__(self, operand: Expr, param: int):
        super().__init__(operand.col_type)  # Inherit input type
        self.operand = operand
        self.param = param
        self.components = [operand]  # Dependencies
    
    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """Return SQL if expressible in SQL, None otherwise."""
        if self.operand.sql_expr() is not None:
            return sql.func.my_function(self.operand.sql_expr(), self.param)
        return None
    
    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> Any:
        """Evaluate in Python if not expressible in SQL."""
        operand_val = data_row[self.operand.slot_idx]
        return my_python_function(operand_val, self.param)
```

## Database and Storage

- **Backend**: PostgreSQL with pgvector extension
- **Media Storage**: Files stored separately in `~/.pixeltable/media/`, referenced by URL
- **Metadata**: Schema versioning and lineage tracking in metadata tables
- **Transactions**: ACID compliance with proper isolation levels

## Common Development Patterns

### Working with Tables

```python
# Create table with schema
t = pxt.create_table('my_table', {
    'id': pxt.Int,
    'content': pxt.String,
    'image': pxt.Image
})

# Add computed column
t.add_computed_column(
    processed=my_udf(t.content, param=123)
)

# Create views for iteration
frames = pxt.create_view(
    'video_frames', videos,
    iterator=FrameIterator.create(video=videos.video, fps=1.0)
)
```

### Error Handling

- Use `pixeltable.exceptions` for custom errors
- Preserve stack traces with proper exception chaining
- Handle media file access failures gracefully
- Provide meaningful error messages with context

### Performance Considerations

- **Batching** - Use `@pxt.udf(batch_size=N)` for expensive operations
- **SQL Pushdown** - Implement `sql_expr()` when possible to avoid Python evaluation
- **Resource Pools** - Use appropriate resource pools for rate limiting
- **Media Caching** - Leverage built-in file caching for external URLs

## Package Structure

```
pixeltable/
├── __init__.py              # Public API exports
├── catalog/                 # Table/view metadata and operations
├── exec/                    # Query execution engine  
├── exprs/                   # Expression system
├── func/                    # UDF framework
├── functions/               # Built-in AI/ML functions
├── io/                      # Import/export functionality
├── iterators/               # Data iteration (frames, chunks, etc.)
├── metadata/                # Database schema definitions
├── utils/                   # Utility modules
├── env.py                   # Environment and configuration
├── store.py                 # Storage abstraction
└── type_system.py           # Column type system

tests/                       # Test suite
├── conftest.py              # Pytest configuration and fixtures
├── test_*.py                # Unit tests
└── utils.py                 # Test utilities

docs/                        # Documentation
├── api/                     # API reference (auto-generated)
├── mintlify/                # User documentation
├── notebooks/               # Tutorial notebooks  
└── sample-apps/             # Example applications
```

## Dependencies and Environment

- **Python**: 3.10+ (dev environment uses 3.10 for compatibility testing)
- **Package Manager**: uv for fast dependency resolution and management
- **Core Dependencies**: PostgreSQL, SQLAlchemy, NumPy, Pandas, PIL, Pydantic
- **AI/ML**: OpenAI, Anthropic, HuggingFace Transformers, Sentence Transformers
- **Media**: Pillow, ffmpeg (via conda), PyMuPDF, BeautifulSoup
- **Development**: mypy, ruff, pytest, mkdocs

## Quick Reference

### Adding a New Function

1. Choose location: `pixeltable/functions/` for integrations, local for UDFs
2. Use appropriate decorators: `@pxt.udf`, `@pxt.uda` for aggregates
3. Add type hints and docstrings following Google style
4. Register with function registry if module-level
5. Add comprehensive tests in `tests/`

### Adding a New Column Type

1. Define in `pixeltable/type_system.py`
2. Add serialization/deserialization logic
3. Update expression system for operations
4. Add SQL type mapping
5. Test with various operations

### Debugging Tips

- Set `PIXELTABLE_LOGLEVEL=DEBUG` for verbose logging
- Use `pxt.configure_logging(level=logging.DEBUG, to_stdout=True)`  
- Check `~/.pixeltable/` for database and media files
- Use `table.collect()` to force computation and see errors
- Examine SQL with query explain plans

Remember: Pixeltable emphasizes incremental, type-safe, multimodal data processing. Follow existing patterns and maintain backward compatibility in public APIs.
