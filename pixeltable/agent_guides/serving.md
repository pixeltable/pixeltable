# Serving

Serving tables over HTTP with `FastAPIRouter` routes, run by `pxt service update` or mounted in a FastAPI app.

## Declaring routes

An application file declares routers next to its models. `pxt service update app.py TARGET` binds the models to the
tables in the catalog directory `TARGET` and runs each router as a separate service on its own port.

```python
import pydantic

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    id = pxt.Column(type=pxt.String, primary_key=True)
    title: pxt.String
    body: pxt.String | None
    cover: pxt.Image | None
    title_upper = pxtf.string.upper(title)
    thumbnail = cover.resize((64, 64))


ingest = FastAPIRouter(name='ingest', prefix='/api')
ingest.add_insert_route(Docs, path='/docs', inputs=['id', 'title', 'body'], outputs=['id', 'title_upper'])
ingest.add_insert_route(
    Docs, path='/upload', inputs=['id', 'title'], uploadfile_inputs=['cover'], outputs=['thumbnail'], background=True
)
ingest.add_update_route(Docs, path='/docs/update', inputs=['body'], outputs=['id', 'body'])
ingest.add_delete_route(Docs, path='/docs/delete')
ingest.add_compute_route(Docs, path='/preview', inputs=['title'], outputs=['title_upper'])


class Headline(pydantic.BaseModel):
    headline: str


@ingest.insert_route(Docs, path='/docs/headline', inputs=['id', 'title'], outputs=['title_upper'])
def headline(*, title_upper: str) -> Headline:
    return Headline(headline=f'{title_upper}!')
```

- Serving needs `pip install 'pixeltable[serve]'`.
- Without the `serve` extra, route definitions do nothing, so the file still imports and `pxt schema update` works.
- `name=` is the service name; without it, the service takes the name of the router's variable, such as `ingest` in
  `ingest = FastAPIRouter()`.
- `prefix=` is prepended to every path of the router, including the media and job paths.
- An insert, update, delete or compute route is a POST with one JSON field per input.
- The body may leave out a nullable input.
- With `uploadfile_inputs`, the body is multipart form data: one file per listed media column and one form field per
  other input.
- An insert route stores one row and returns the `outputs`.
- The `inputs` of an insert route must include every non-nullable input column.
- An update route finds the row by its primary key; the body always has the primary key fields.
- The `inputs` of an update route cannot include primary key or media columns.
- For an unknown primary key, an update route returns 404 `row not found`.
- A delete route deletes the rows whose primary key columns, or the columns in `match_columns=`, match the body's
  values, and returns `{"num_rows": n}`.
- A compute route runs `Table.compute()` and stores nothing.
- On a view with an iterator, a compute route returns a JSON array. On any other table or view, it returns one object,
  or `null` when a view's filter drops the row.
- The decorators `insert_route()`, `update_route()` and `compute_route()` pass each output column to the function as a
  keyword-only argument; a media column arrives as a URL `str`.
- A decorated route sends the function's pydantic model as the response.
- A route path cannot contain a `{parameter}`.
- The path `/_pxt` is reserved.
- `add_insert_route()`, `add_update_route()` and `add_compute_route()` refuse Array and Binary columns in `outputs`.

## Query routes

```python
@pxt.query
def titles() -> pxt.Query:
    return Docs.order_by(Docs.title).select(Docs.title)


@pxt.query
def find_doc(title: str) -> pxt.Query:
    return Docs.where(Docs.title == title).select(Docs.id, Docs.body).limit(1)


ingest.add_query_route(path='/titles', query=titles, method='get')
ingest.add_query_route(path='/find', query=find_doc)
```

- The query's parameters are the route's inputs: JSON fields for POST (the default) and query-string parameters with
  `method='get'`.
- A query that aggregates without `group_by()` or has a literal `limit(1)` is a one-row query: it returns at most one
  row. Any other query is a many-row query.
- `limit(n)` with a parameter `n` makes a many-row query, even when `n` is 1.
- For a many-row query that selects one unnamed expression, the route returns a JSON array of values (`/titles`); for
  other many-row queries, it returns `{"rows": [...]}`.
- For a one-row query that selects one unnamed expression, the route returns the value itself; for other one-row
  queries, it returns an object (`/find`).
- `select(title=Docs.title)` gives the expression a name.
- When a one-row query finds no row, the route returns 404 `query returned no rows`.

## Responses

- With `background=True`, a route answers at once with `{"id": ..., "job_url": ...}` and runs the request in a
  background thread.
- `job_url` is `{prefix}/_pxt/jobs/{name}/{id}`, or `{prefix}/_pxt/jobs/{id}` for a router without `name=`.
- A GET on `job_url` returns `status`: `pending`, with a `Retry-After: 2` header; `done`, with the route's response in
  `result`; or `error`, with the error's `detail` object in `error_detail`.
- A finished job is kept for 10 minutes, or for 60 seconds after the first poll that returns its final status,
  whichever ends first.
- A GET on the `job_url` of an expired job returns 404 `unknown or expired job`.
- After the service instance that ran the job stops, a GET on its `job_url` returns 410.
- A router accepts at most 64 unfinished jobs per service instance.
- Beyond that limit, the router returns 503 `JOB_REFUSED` with `retryable` true and a `Retry-After: 5` header.
- A route returns a media output as a URL.
- A local service serves the file under `{prefix}/_pxt/media/...`; a hosted service returns a signed URL that is valid
  for one hour.
- With `return_fileresponse=True`, a route returns the file of its only output instead of JSON; that output must be a
  media column.
- On a query route, `return_fileresponse=True` needs a one-row query.
- `return_fileresponse=True` cannot be combined with `background=True`.
- On a Pixeltable error, a route returns the HTTP status of the error code and
  `{"detail": {"error_code": ..., "message": ..., "retryable": ...}}`.
- When the error has a retry delay, the response also has `retry_after` and a `Retry-After` header.
- On a duplicate primary key, a route returns 422 `CONSTRAINT_VIOLATION`.
- When a computed column's expression raises an exception, the route returns 400 and stores nothing; `message`
  contains the exception.
- A service binds its tables when it starts. After `pxt schema update` changes a table, the insert, update, compute
  and delete routes on that table answer 409 `table schema changed since route was registered; please restart the
  service` until `pxt service update` restarts the service. Query routes keep answering; their select list is fixed
  when the service starts.
- On an insert, update or compute route, `export_sql=SqlExport(...)` writes each response to an existing table in
  another SQL database after Pixeltable commits the change. Import `SqlExport` from `pixeltable.serving`.
- If the `export_sql` write fails, the route returns 500, and the Pixeltable change stays committed.

## A FastAPI app of your own

```python
import fastapi

app = fastapi.FastAPI()
app.include_router(ingest)


@app.get('/api/stats')
def stats() -> dict[str, int]:
    return {'docs': Docs.count()}
```

- A file with a `fastapi.FastAPI` object is one service, named after its module (`app` for `app.py`).
- The app must call `include_router()` for every router of the file.
- Of the file's routers, at most one with background routes can omit `name=`.
- The service binds the file's models to the tables under `TARGET` before it starts, so a handler can use `Docs`
  directly.
- Declare a handler with `def`, not `async def`: FastAPI runs a `def` handler on a thread pool, and an `async def`
  handler that calls Pixeltable blocks the event loop.
- To include a router in an app that `pxt service` does not run, call `ingest.bind('my_app')` after
  `pxt schema update`, then `app.include_router(ingest)`.
- `bind()` fails if a table is missing or differs from its model.
- An app with an unbound router fails at startup.

```bash
pxt service check app.py
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list my_app --json
pxt service restart my_app/app
```
