> **Audience: Pixeltable contributors.** If you're building an *application* with Pixeltable, use the [Agent Skill](https://github.com/pixeltable/pixeltable-skill) instead (`npx skills add pixeltable/pixeltable-skill`).

# AGENTS.md

Instructions for AI coding agents working with the Pixeltable codebase. This is the single developer
guide; `CLAUDE.md` imports it.

## Where the other instructions live

| File | Governs | Read it when |
|---|---|---|
| `.github/copilot-instructions.md` | Copilot code review on pull requests | Changing what a reviewer flags |
| `docs/_guidelines/GUIDELINES_FOR_PROSE.md` | Prose in docs, notebooks, READMEs | Writing anything a user reads |
| `docs/_guidelines/GUIDELINES_FOR_DOCSTRINGS.md` | Docstrings, which ship as SDK reference | Adding or editing a docstring |
| `docs/_guidelines/GUIDELINES_FOR_NOTEBOOKS.md` | Notebook structure and conversion | Touching `docs/release/**/*.ipynb` |
| `docs/_guidelines/GUIDELINES_FOR_COOKBOOKS.md` | Cookbook recipe structure | Adding a recipe |
| `dashboard/DESIGN.md`, `dashboard/ARCHITECTURE.md` | The local dashboard UI | Touching `dashboard/` or its server APIs |
| `docs/release/skill.md` | The user-facing agent skill | Changing what app builders are told |
| `CONTRIBUTING.md` | Branching, review, merge process | Opening or merging a PR |

## Protected Configuration

Never change `integrations.telemetry.enabled` in `docs/release/docs.json`. It must remain set to `true` in every pull request.

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
│   └── sample-apps/      # Older showcases; examples live in pixeltable-starter-kit
└── tool/                 # Development utilities
```

## Setup Commands

**Prerequisites:** Miniforge with a dedicated environment (not `base`)

```bash
# Create and activate conda environment
mamba create --name pxt python=3.11
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

Exercise behavior through public SDK, CLI, or HTTP APIs and assert on public results, metadata, or errors:
use `Table.get_metadata()`, `t.describe()`, or queries rather than `col.stored`, `ColumnRef`, or
`TableVersion` internals. Reach for an internal API only to test an internal component directly, or to set up
a fixture no public API can build.

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

### Required After Every Code Change

After every code change, before reporting it done:

1. `make format`: auto-formats code.
2. `make check`: mypy + ruff static checks; both must pass.
3. `git add` any new source file, then review the whole change with `git diff HEAD` (no pathspec: source
   and tests), reading every comment, docstring, and string you added. A diff narrowed to one file does not
   count, and an unstaged new file does not appear in it. A comment must describe only the code at hand
   (never a caller's intent or a called function's internals) and must not state behavior you have not
   verified.
4. Delete before rewording: cover each comment and read only the identifier, the signature, and the code
   below it. If those carry the same fact, delete the comment rather than improving it. See
   [Prose and grammar](#prose-and-grammar).
5. Check every sentence that survived against [Prose and grammar](#prose-and-grammar): straight word
   order, no banned construction, no vacuous or informal term. Fix every violation from steps 3 to 5
   before proceeding.

Skip only if explicitly directed or if the environment makes it impossible.

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

        >>> t.add_computed_column(result=my_function(t.text, model='advanced'))
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

Application schema is a `TableModel` class in `app.py`. `pxt schema update app.py my_app` creates those tables. Put `FastAPIRouter` routes in that same `app.py`. In tests, notebooks, and a REPL, keep using `pxt.create_table()`; do not require a project file there.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)
    summary = excerpt(title)


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.id, Docs.title_upper, Docs.summary]
)
ingest.add_update_route(
    Docs, path='/docs/update', inputs=[Docs.title], outputs=[Docs.id, Docs.title_upper]
)
ingest.add_compute_route(Docs, path='/titles', inputs=[Docs.title], outputs=[Docs.title_upper])
```

```bash
pxt schema update app.py my_app
pxt service update app.py my_app
```

After `pxt schema update`, open the table with `t = pxt.get_table('my_app.docs')`, then `t.insert()` / `.select()` / `.collect()`. On a `TableModel`, put indexes in `__indexes__`. Do not call `add_embedding_index()` in application code that you later create with `pxt schema update`.

Tests and notebooks (not app files):

```python
import pixeltable as pxt

t = pxt.create_table('my_dir.my_table', {
    'text': pxt.String,
    'image': pxt.Image,
    'metadata': pxt.Json,
})
t.add_computed_column(embedding=some_embedding_fn(t.text))
t.add_embedding_index('text', embedding=embed_fn)
t.insert([{'text': 'hello', 'image': 'path/to/image.jpg'}])
```

Examples: [pixeltable-starter-kit](https://github.com/pixeltable/pixeltable-starter-kit).

### Error Handling

- Use `pixeltable.exceptions` for custom exceptions
- Validate inputs early and provide clear error messages
- Raise a subclass of `Error`, never `Error` itself: its `__init__` asserts
  `raise a subclass of Error, not Error itself`. Every instance carries an `ErrorCode`, and the code
  determines the class, so `RequestError` takes a request code and `NotFoundError` a not-found one.
- `UserError` is the subclass for a user error with no more specific code:
  `raise pxt.UserError(pxt.ErrorCode.GENERIC_USER_ERROR, 'message')`. Reach for a specific subclass
  first (`RequestError`, `NotFoundError`, `AlreadyExistsError`, `AuthorizationError`,
  `ExternalServiceError`, `ServiceUnavailableError`, `ConcurrencyError`).

## Prose and grammar

These rules cover every sentence we write: code comments, docstrings, error messages, CLI help, MDX docs,
commit messages and PR descriptions. `docs/_guidelines/GUIDELINES_FOR_PROSE.md` governs what an MDX page
says; this section governs how a sentence is built.

Write the shortest sentence that states the fact, in straight word order.

### Four principles

1. **Use the shortest form for the relation.** A relative clause expressing only possession or
   attribution is a possessive with extra steps: "the models a service serves" -> "the service's models";
   "the file a binding came from" -> "the binding's source file". The tell is a verb at the end of a noun
   phrase. Such a clause has no relative pronoun, so scanning for "that" or "which" misses every instance.
2. **Cut a qualifier already established by the context.** Name the thing plainly and put the
   distinguishing fact in the predicate: "a file the served project does not hold cannot be imported" ->
   "loading a file outside the project is refused".
3. **Name the referent, not its category.** A sentence built out of category nouns can only be skimmed.
   Keep one concrete term, the method or the class or the field: not "refuse a Query member that needs a
   table, naming the model that has none", but "raise an error for an attribute only a bound query has,
   such as collect()".
4. **Straight word order.** No preposition stranded at the end of a clause, no noun-phrase pileup ("the X
   a Y is Z to"), no fused emphatic ("X is what makes Y work" -> "X makes Y work").

### Banned constructions

| Instead of | Write |
|---|---|
| `<noun> holds <x>` | has, contains, stores, or the relation itself: the image *contains* the file |
| `<x> names <y>` | specifies, points at, is set to |
| `<x> carries <y>` | has, contains, includes |
| `the <noun> the <other> <verbs>` | the possessive, or a clause with the verb in it: "the docstring states this rule" |
| "... has none", "... holds none", "... declares none" | say what is there, or state what is missing |
| "resolve against", "runs against" | to, with, or according to |

### Banned terms

- **Vacuous**: footgun, load-bearing, happy path, self-heal(ing), envelope, leaf. Name the behavior, the
  constraint, or the failure mode instead. "Default `mix_duration='first'` truncates the output when the
  audio is shorter than the video" informs; "is a footgun" does not.
- **Informal**, in anything a user reads: knob -> setting, magic -> the actual behavior, under the hood ->
  internally, kicks in -> applies, gotcha -> the specific failure mode.
- **slug**: `org_slug` and `db_slug` store a user-supplied name, so prose says the org's name, the
  database's name. Naming the identifier is accurate where the code is the subject: "`db_slug` was empty".
- **Non-ASCII typography**, in every file: no em or en dash, smart quotes, arrows, ellipsis, or math
  symbols. Write `-`, `"`, `->`, `...`, `>=`. Leave unicode already in a file alone.

### Economy

One fact per comment, on one line where possible, stated once at the site that does the thing. Cut the
"so that ..." clause when the code shows it.

Delete before rewording: cover the comment and read only the identifier, the signature, and the code below
it. If those carry the same fact, delete the comment rather than improve it. Deletion is the default
outcome. A docstring paraphrasing the name, an "or None if ..." for a `| None` annotation, and a fact
already stated elsewhere all go. Keep what the reader cannot recover: a constraint imposed by a callee, the
reason for a surprising choice, an invariant that would silently break.

### How to check

Negative pattern-matching is not enough, since it passes any sentence whose shape is new. Run these in
order on every sentence added or edited:

1. Read it aloud. If you cannot say what happens in one breath, rewrite it rather than reflow it.
2. Look for a noun phrase ending in a verb, and rewrite it as a possessive or a prepositional phrase.
3. Delete any qualifier already established by the surrounding text.
4. Check that at least one noun is the concrete referent.
5. Then the surface checks: stranded preposition, pileup, fused emphatic, and the constructions and terms
   banned above.

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

- Code examples must use `>>>` prompts, not fenced code blocks
- Backticks must be properly paired
- HTML tags must be self-closing
- When describing what a function does, focus on the behavior of the function itself, not its callers

### Code Comments

- Keep code comments succinct; avoid unnecessarily verbose comments.
- Always use parens to denote functions: in code comments, it's `my_func()`, not `my_func`.

### Building Docs

```bash
# Build documentation
make docs

# Serve locally for development
make docs-serve

# Deploy to the dev environment for preview
make docs-deploy TARGET=dev
```

`TARGET=dev` is the only deploy target an agent may run or suggest. `stage` and `prod` are for humans.

### Local Dashboard UI

Before changing `dashboard/` or dashboard-facing APIs in `pixeltable_cli/server/`, read [`dashboard/DESIGN.md`](dashboard/DESIGN.md) (required) and [`dashboard/ARCHITECTURE.md`](dashboard/ARCHITECTURE.md). Follow DESIGN.md; justify any visual departure explicitly.

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
| `pixeltable_cli/` | CLI + daemon (serves dashboard API and static SPA) |
| `dashboard/` | Local web UI frontend (React/Vite); see `dashboard/DESIGN.md` |
| `dashboard/DESIGN.md` | Dashboard visual/UX source of truth (required for UI changes) |
| `dashboard/ARCHITECTURE.md` | Dashboard stack and API map |
| `pyproject.toml` | Dependencies and tool config |
| `Makefile` | Build and test commands |

## Getting Help

- **Documentation**: https://docs.pixeltable.com/
- **GitHub Issues**: https://github.com/pixeltable/pixeltable/issues
- **Discord**: https://discord.gg/QPyqFYx2UN
- **Discussions**: https://github.com/pixeltable/pixeltable/discussions
