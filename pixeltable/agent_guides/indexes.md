# Indexes

Embedding indexes for similarity search, B-tree indexes for lookups, and how both stay current.

## Declaring indexes

An application lists a model's indexes in `__indexes__`; `pxt schema update` creates them with the table.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()
embed_text = pxtf.openai.embeddings.using(model='text-embedding-3-small')


class Articles(TableModel, name='articles'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String
    category: pxt.String
    __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed_text, name='body_idx'), pxt.BtreeIndex(category)]


@pxt.query
def search(q: str, k: int = 5) -> pxt.Query:
    sim = Articles.body.similarity(string=q)
    return Articles.where(sim > 0.3).order_by(sim, asc=False).select(Articles.id, score=sim).limit(k)
```

In a notebook, `add_embedding_index()` and `add_btree_index()` add indexes to a table, with or without rows:

```python
import numpy as np


@pxt.udf
def embed(text: str) -> pxt.Array[(16,), pxt.Float]:
    # a stand-in for a real embedding model
    vec = np.zeros(16, dtype=np.float32)
    for word in text.lower().split():
        vec[sum(word.encode()) % 16] += 1.0
    return vec


pxt.create_dir('kb', if_exists='ignore')
notes = pxt.create_table('kb.notes', {'id': pxt.Int, 'text': pxt.String, 'tag': pxt.String})
notes.insert([{'id': 1, 'text': 'red apple pie', 'tag': 'food'}, {'id': 2, 'text': 'blue sky', 'tag': 'sky'}])
notes.add_embedding_index('text', idx_name='text_idx', embedding=embed)
notes.add_embedding_index('text', idx_name='text_l2', embedding=embed, metric='l2')
notes.add_btree_index('tag')

sim = notes.text.similarity(string='red apple', idx='text_idx')
top = notes.order_by(sim, asc=False).limit(3).select(notes.text, score=sim).collect()
```

- `name=` sets an `EmbeddingIndex`'s name in a model; `add_embedding_index()` takes `idx_name=` instead.
- `EmbeddingIndex(idx_name=...)` raises `TypeError`.
- Pixeltable generates a name for an unnamed index, such as `idx0` or `idx1`.
- Index names are case-insensitive.
- In a model, two embedding indexes on one column need names: `column 'body' has multiple embedding indexes; they
  must be given explicit names`.
- `pxt.BtreeIndex` takes no name.
- A column has at most one B-tree index.
- `add_embedding_index()` with the definition of an existing index raises `An identical embedding index already
  exists`; with `if_exists='ignore'`, it leaves the existing index in place.

## Embedding functions

- An embedding function is a module-level UDF that takes one required parameter of the indexed column's type
  (`String`, `Image`, `Audio`, `Video` or `Document`) and returns a fixed-length array, such as
  `pxt.Array[(16,), pxt.Float]`.
- Passing any other function raises `The function ... is not a valid embedding`, with the reason.
- `.using(...)` binds the other parameters to constants:
  `pxtf.huggingface.sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')`,
  `pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32')`,
  `pxtf.openai.embeddings.using(model='text-embedding-3-small')`.
- `clip()` without `.using()` is not a valid embedding.
- `sentence_transformer.using()` and `clip.using()` load the model to find the embedding's length, so importing a
  file that calls them loads the model.
- `embedding=` applies to every parameter type in the function's signatures: a CLIP index on an `Image` column
  supports both `similarity(string=...)` and `similarity(image=...)`.
- `string_embed=`, `image_embed=`, `audio_embed=`, `video_embed=` and `document_embed=` each set the function for
  one type and override `embedding=` for that type.
- Passing one of these parameters a function that does not take the parameter's type raises an error.
- `similarity()` accepts a query value only if the index has a function for that value's type: with only
  `image_embed=`, `similarity(string=...)` raises `does not have a string embedding and does not support string
  queries`.
- An index on an `Array[(N,), Float]` column of precomputed vectors needs no function.
- Such an index supports only `similarity(vector=...)`.

## Searching

- `similarity()` takes the query value as one keyword argument: `string=`, `image=` (a path, URL or
  `PIL.Image.Image`), `audio=`, `video=` and `document=` (a path or URL), or `vector=` (a 1-dimensional float NumPy
  array of the embedding's length).
- A positional query value is deprecated.
- The query value is a constant or a `@pxt.query` parameter; passing a column raises `similarity(): requires a
  value, not an expression`.
- With several embedding indexes on the column, `idx=` picks one; without it, `similarity()` raises `Column 'text'
  has multiple embedding indices`.
- `metric='cosine'` (the default) scores by cosine similarity: higher is closer.
- The metric `'ip'` scores by inner product: higher is closer.
- With `'cosine'` or `'ip'`, `order_by(sim, asc=False)` lists the closest rows first.
- The metric `'l2'` scores by Euclidean distance: lower is closer.
- `similarity()` cannot be the value of a computed column: evaluating it raises `similarity(): cannot be used in a
  computed column`.
- A computed column that calls a `@pxt.query` can search: the query passes its parameter to `similarity()`, and the
  column runs the search once per row, on insert. A label table's best match for each image, or an earlier copy of
  the same photo, is such a column (`pxt guide queries` says which rows the search sees).
- A snapshot has no indexes: `similarity()` on a snapshot raises `Snapshot does not support indices`.
- `notes.text.embedding(idx='text_idx')` selects the stored vectors.

## How an index stays current

- An insert computes the new rows' embeddings in the same operation.
- An embedding function that raises fails the insert; with `on_error='ignore'`, the row is stored, and its
  similarity is null.
- An update of the indexed column recomputes the row's embedding.
- Adding an index to a table with rows embeds every row in the same call, so a paid embedding model runs for each
  row.
- An index on a view column, such as `text` of a `document_splitter()` view, embeds the new view rows during each
  insert into the base.
- `pxt schema update` treats a new entry in `__indexes__` as an additive change (`pxt guide tables`): it creates the
  index and embeds the existing rows.
- `pxt schema update` replaces the index of a changed entry and drops the index of a removed entry; both need
  `--allow-destructive`.
- `drop_embedding_index()` and `drop_index()` take `idx_name=`, or `column=` for a column with one index.

## B-tree indexes

- A column has a B-tree index when one is declared, or when the table sets `has_default_idxs=True`, which adds one
  to each eligible column.
- A B-tree index serves `where()` comparisons and `order_by()` on its column.
- A B-tree index requires a stored column: a scalar column other than `Bool`, or a media input column.
- The primary key has its own unique index, not listed in `TableMetadata.indexes`.

```bash
pxt idxs my_app/articles
pxt idxs my_app --embedding
```
