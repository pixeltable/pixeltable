# UDFs

Defining UDFs and aggregates, how they treat nulls, batches and media files, and how they become LLM tools.

## Defining a UDF

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.func import Batch

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[:n] + '...'


@pxt.udf
def severity_label(severity: str | None) -> str:
    return 'unknown' if severity is None else severity.upper()


@pxt.udf(batch_size=16)
def word_counts(texts: Batch[str]) -> Batch[int]:
    return [len(text.split()) for text in texts]


class Tickets(TableModel, name='tickets'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String | None
    severity: pxt.String | None
    preview = excerpt(body, n=20)
    label = severity_label(severity)
    words = word_counts(body)
```

- Every parameter and the return value need a type hint. Without one, `@pxt.udf` raises
  `Cannot infer pixeltable type for parameter 'x'` or `Cannot infer pixeltable return type`.
- The return hint sets the column type: `str` is `String`, `PIL.Image.Image` is `Image`, `pxt.Video` is `Video`,
  a TypedDict or `list[str]` is typed Json, and `dict` is untyped Json.
- Pixeltable records a UDF by its module path relative to the project root, such as `app.excerpt`.
- The project root is the directory of the project configuration, `pixeltable.toml`, which `pxt init` writes.
- The daemon and the services of `pxt service update` import the UDF's module by that path (`pxt guide serving`).
- `@pxt.udf` on a function in the main module of a script raises `Defining the UDF 'f' directly in the global
  namespace of a Python script is not allowed`.
- `@pxt.udf` on a function nested in another function raises `nested functions are not supported`.
- When the UDF's module can no longer be imported, Pixeltable warns `The computed column 'preview' in table
  'tickets' is no longer valid`.
- The invalid column's stored values stay readable.
- An insert into its table fails with `the column 'preview' is currently invalid`.
- In a notebook, a UDF defined in a cell works for that session only; another process cannot load it.
- `pxt.eval_udf(excerpt, 'a long text', n=5)` calls a UDF on Python values, without a table (`pxt guide
  expressions`).
- `pxt.get_table()` inside a UDF raises `Catalog functions such as pxt.get_table() cannot run while a query or update
  executes`.
- Known issue (PXT-1711): `add_computed_column()` accepts a column whose UDF calls `pxt.get_table()`, and the next
  insert or recompute fails.

## Nulls

- A parameter typed without `| None`, such as `pxt.String` or `str`, is non-nullable: for a null argument,
  Pixeltable skips the call and sets the cell to null without recording an error.
- A nullable argument for such a parameter makes the column nullable: `preview` is `String | None` even though
  `excerpt()` returns `str`.
- A parameter typed with `| None`, such as `pxt.String | None` or `str | None`, receives the null, and the UDF
  handles it: `label` is `String`.
- Built-in functions skip calls with null arguments the same way.
- An aggregate skips a row with a null argument for a non-nullable `update()` parameter.

## Batched and async UDFs

- With `batch_size=N`, each `Batch[...]` parameter receives a list of up to N values, and the function returns a
  list of equal length.
- In a batched UDF, a parameter without `Batch[...]` takes a constant; passing a column fails with `parameter offset
  must be a constant value`.
- Pixeltable runs an `async def` UDF as one task per row, concurrently.
- `@pxt.udf(resource_pool='request-rate:my_api')` limits the calls of an `async def` UDF to the number of requests
  per minute in the environment variable `MY_API_RATE_LIMIT`, 600 by default.
- A UDF with `resource_pool='request-rate:my_api'` retries 408, 429, 503 and 504 errors up to 3 times.
- Pixeltable fails with `Unrecognized option 'pixeltable.my_api'` on a `[my_api]` section in
  `$PIXELTABLE_HOME/config.toml`.
- `resource_pool=` on a synchronous UDF fails with `resource_pool requires an async function`.

## Media results

- A UDF with an image return type returns a PIL image, and Pixeltable stores it as a file in its media store.
- A UDF with a `pxt.Video`, `pxt.Audio` or `pxt.Document` return type returns a file path or URL.
- Pixeltable copies a returned file in a system temp directory, such as one from `tempfile.mkdtemp()`, into its
  media store.
- Pixeltable records any other returned local path as is, so the file must stay at that path.
- When the column sets `destination=`, Pixeltable copies the file there.

## Aggregates

```python
@pxt.uda
class mean_length(pxt.Aggregator):
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def update(self, text: str) -> None:
        self.lengths.append(len(text))

    def value(self) -> float | None:
        return sum(self.lengths) / len(self.lengths) if len(self.lengths) > 0 else None


by_severity = Tickets.group_by(Tickets.severity).select(Tickets.severity, avg_length=mean_length(Tickets.body))
```

- `__init__()` takes the constant arguments, `update()` takes one row's values, and the return hint of `value()`
  sets the result type.
- Aggregates run in queries; an aggregate in a computed column raises `aggregate functions are not supported in
  computed columns`.
- An aggregate declared with `requires_order_by=True`, such as `pxtf.video.make_video()`, takes the ordering
  expression as its first positional argument: `make_video(frames.pos, frames.frame, fps=25)`.
- Passing `order_by=` to such an aggregate raises `order_by invalid, this function requires the first argument to be
  the ordering expression`.

## Tools

```python
@pxt.udf
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f'{city}: 22C, sunny'


tools = pxt.tools(get_weather, pxt.tool(excerpt, name='shorten', description='Shortens a text.'))


class Assistant(TableModel, name='assistant'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    message: pxt.String
    response = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': message}], model='gpt-4o-mini', tools=tools
    )
    tool_output = pxtf.openai.invoke_tools(tools, response)
```

- A tool's name is the UDF's name, and its description is the UDF's whole docstring; `pxt.tool()` replaces either.
- A tool's non-nullable parameters are required.
- `pxt.tools()` takes UDFs, `@pxt.query` functions and `pxt.retrieval_udf()` results; passing an aggregate raises
  `Aggregator UDFs cannot be used as tools`.
- A tool's parameters can be `String`, `Int`, `Float`, `Bool`, `UUID` or Json; `pxt.tool()` and `pxt.tools()`
  refuse a UDF with a parameter of another type: `parameter 'when' has unsupported type`.
- `openai`, `anthropic`, `gemini`, `bedrock` and `groq` each have an `invoke_tools()` for their own replies.
- `invoke_tools()` returns a dict with one key per tool; the value is the tool's list of results, or null if the
  model did not call the tool, as in `{'get_weather': ['Paris: 22C, sunny'], 'shorten': None}`.
- The UDF from `pxt.retrieval_udf(table, parameters=['sku'])` returns the rows whose `sku` equals its argument, up to
  `limit` rows (10 by default), as a list of dicts with every column.
- With `limit=1`, the UDF from `pxt.retrieval_udf()` returns one dict, or null.
- `pxt.retrieval_udf()` takes a `Table` or a model class; over a model in `app.py`, the returned UDF can define
  another model's computed column.
- `pxt.mcp_udfs(url)` connects to the MCP server at `url` over streamable HTTP and returns a UDF for each of its tools.

```bash
pxt errors my_app/tickets --col preview
pxt recompute my_app/tickets preview --errors-only -f
```
