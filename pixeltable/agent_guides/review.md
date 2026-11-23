# Review

Rules that `pxt lint` applies: Python code that redoes Pixeltable's work, or that uses Pixeltable wrongly.

Each rule says what to look for, why it matters, what to write instead, and which cases are not findings. A place
in the code is a finding only if it matches what a rule looks for and none of its exceptions.

The rules apply only to the data and logic that the project implements with Pixeltable, not to its interfaces with
external systems, such as a Redis store that other services read or the format of an input file.

## R01 A provider's SDK called directly

Look for: a provider's client, such as `openai.OpenAI()`, `anthropic.Anthropic()` or `genai.Client()`, created in the
project to call a model on table data, or to try a prompt.

Why: Pixeltable's provider functions read the configured keys, throttle and retry requests, and store each row's
result or error. An SDK call does none of this, and a script that loops over rows with it calls the model for every
row again on its next run.

Instead: a computed column that calls the provider function. To try a prompt on a few values, call the same function
with `pxt.eval_udf()`:

    reply = pxt.eval_udf(
        pxtf.openai.chat_completions, messages=[{'role': 'user', 'content': prompt}], model='gpt-4o-mini'
    )

Not a finding: a provider feature that no Pixeltable function covers, called in a UDF. `pxt functions openai` lists the
`openai` functions.

Guide: providers

## R02 A model or a UDF called in a loop over rows

Look for: a loop over `collect()` results, or over a DataFrame, that calls a model, a UDF or an API for each row.

Why: the results are not stored with the rows, a failure is not recorded with its row, and the next run repeats every
call.

Instead: a computed column. `pxt recompute PATH COLUMN --errors-only` reruns only the failed cells.

Not a finding: a look at a few rows while exploring.

Guide: tables

## R03 Values written back by a loop

Look for: `update()` or `batch_update()` inside a loop over `collect()` results, or a script that computes a value for
every row and writes it back.

Why: the stored values are stale until the script runs again, and nothing recomputes them when their inputs change.

Instead: a computed column for a value derived from the row's own data, or a query for a value that depends on other
rows, another table or the current time. A value from other rows that is fixed when the row arrives, such as an
earlier near-duplicate or the best match in a label table, is a column that calls a `@pxt.query`; it runs on insert
and can search an embedding index with `similarity()`.

Not a finding: writing values that come from outside the tables, such as a person's decision or an external system's
state.

Guide: tables, queries

## R04 Similarity scored in Python

Look for: vectors or embeddings read with `collect()` and compared in Python: dot products, cosine similarity, or a
sort by such a score.

Why: every call scores every row, and the work grows with the table.

Instead: an embedding index on the column, and `similarity()` in a query; the index embeds each new row on insert:

    __indexes__ = [pxt.EmbeddingIndex(description, embedding=embed_text, name='desc_idx')]

    sim = Items.description.similarity(string=text)
    Items.where(Items.sku != sku).order_by(sim, asc=False).limit(3).select(Items.sku).collect()

Not a finding: a comparison of two values passed in by the caller.

Guide: indexes

## R05 A value from other rows computed in a Python loop

Look for: rows read with `collect()` and combined in Python into a value per row or per group: duplicates, the first
row per key, counts, rankings, totals. Also a `@pxt.query` called with a column argument in `where()` or `select()`
to compute such a value: it runs one query per row.

Why: a `group_by()` query computes per-group values in the database and returns one row per group. A loop recomputes
them in Python from every row, and a `@pxt.query` called per row runs one query for each row.

Instead: compute the per-group value with a `group_by()` query, then look it up by key for each row in Python.
`join()` takes a table, not a query, so the grouped query cannot be joined back to its table:

    first = signups.group_by(signups.email_key).select(signups.email_key, first_at=pxtf.min(signups.signed_up_at))
    first_at = {r['email_key']: r['first_at'] for r in first.collect()}
    rows = signups.select(signups.signup_id, signups.email_key, signups.signed_up_at).collect()
    duplicates = [r['signup_id'] for r in rows if r['signed_up_at'] > first_at[r['email_key']]]

A window aggregate, such as `pxtf.min(t.at, group_by=t.key)` in `select()`, is not the fix: in this release it
returns a running value (PXT-1801, `pxt guide queries`).

A value that is fixed once the row arrives, such as an earlier near-duplicate found with `similarity()`, does not need
a query per row on every read: make the `@pxt.query` call a column, which runs once per row, on insert.

Not a finding:

- a loop that only formats or prints rows;
- grouping done in a Pixeltable query and looked up by key in Python.

Guide: queries

## R06 A copy refreshed on a schedule

Look for: a script that cron or another scheduler runs to rebuild a table or a column from other tables: a metrics
table, a summary, an enriched copy, precomputed recommendations.

Why: the copy is stale between runs, and a new row is missing from it until the next run.

Instead: a query that computes the value when it is read, run by the report or served with `add_query_route`.
Per-row work on new data is a computed column, which runs when the row is inserted.

Not a finding: a scheduled script that inserts new input data.

Guide: queries, serving

## R07 Workers, threads and status columns

Look for: `threading`, `asyncio.create_task()`, FastAPI `BackgroundTasks`, `while True` with `sleep()`, a FastAPI
`lifespan` hook, a jobs table, or a `status` column that code updates as work progresses.

Why: Pixeltable computes a row's computed columns when it is inserted, records each failed cell, and a route with
`background=True` returns a job URL to poll.

Instead: `add_insert_route(..., background=True)`. Derive a status from the row: its computed columns have values, or
`col.errortype != None`.

Not a finding: a status that records a person's decision.

Guide: serving

## R08 A UDF that only extracts, filters or formats

Look for: a `@pxt.udf` whose body only indexes into a dict or list argument, filters or counts a list, or does
arithmetic or string concatenation.

Why: Pixeltable cannot see into a UDF, but it can parallelize an expression, which is part of the column's
definition. A UDF is also more code to maintain.

Instead: Json paths and Json methods with lambdas (`pxt guide json`), operators, and the built-in functions
(`pxt functions --on String`).

Not a finding: logic with no expression form, such as a branch on the value, a regex parser or a scoring rule.

Guide: json, expressions

## R09 A UDF that reads a table

Look for: `pxt.get_table()`, a model class, `collect()` or `where()` inside a `@pxt.udf` body.

Why: nothing recomputes the column when the other table changes.

Instead: a join at query time, or the looked-up value as an input column.

Guide: queries

## R10 A model loaded inside a UDF

Look for: `from_pretrained()`, `load_model()`, `torch.load()` or `pipeline()` inside a `@pxt.udf` body.

Why: the weights load again for every row.

Instead: the shipped function for the model (`pxt functions --search whisper`). For a model without one, load it in a
module-level function decorated with `@functools.cache`, and call that function from the UDF.

Not a finding: a load behind `@functools.cache` or a module-level variable.

Guide: udfs, providers

## R11 Data kept outside the tables

Look for: SQLite, Redis, JSON or pickle files, or DataFrames that a script or a service reads and writes between
runs; a separate vector database such as Chroma, FAISS, Pinecone or pgvector.

Why: a second store is neither versioned nor updated in the same transaction as the tables, and it duplicates a
table, an embedding index or a stored computed column.

Instead: a table; an embedding index; a stored computed column as the cache of a model call. Embeddings of a fixed
set, such as label texts, go in a table with an embedding index, and a column that calls a `@pxt.query` finds the
best match.

Not a finding: an export at the end of a pipeline, such as `collect().to_pandas()` written to a file for someone else.

Guide: tables, indexes

## R12 Types that hide the data's structure

Look for: a timestamp or a date stored as a String, a list stored as a comma-separated String, a value used for
filtering or grouping kept inside a Json column, or a media file's path stored as a String.

Why: strings sort and compare as text, every reader parses the list again, and a field inside Json is not a column.

Instead: `pxt.Timestamp` or `pxt.Date`, `pxt.Json[list[str]]`, a typed column for each value used in a query, and
`pxt.Image`, `pxt.Video`, `pxt.Audio` or `pxt.Document` for media. Convert an input format, such as a comma-separated
field in a CSV file, in the code that loads it.

Guide: tables

## R13 Tables created by code in an application

Look for: `pxt.create_table()`, `drop_table()` or `add_computed_column()` in an application file or an "init" script
of a project that declares `TableModel` classes.

Why: `pxt schema update` creates and changes an application's tables from its models; code that also creates them
drifts from the models, and a drop-and-recreate deletes the data.

Instead: the column or index on the model, then `pxt schema update app.py TARGET`.

Not a finding: tests, notebooks and one-off scripts outside the application.

Guide: tables
