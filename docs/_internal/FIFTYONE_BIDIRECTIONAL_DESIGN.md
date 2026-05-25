# FiftyOne Bidirectional Integration Design

## Current State

Pixeltable currently has a **one-way export** to FiftyOne:

```python
from pixeltable.io.fiftyone import export_images_as_fo_dataset

dataset = export_images_as_fo_dataset(
    table, image=table.image,
    detections={'predictions': table.detections},
)
```

This uses `PxtImageDatasetImporter` which implements FiftyOne's `LabeledImageDatasetImporter`.

## Goal

Create a **bidirectional** integration where:
1. Pixeltable tables can be exported to FiftyOne datasets (existing)
2. FiftyOne datasets can be imported into Pixeltable tables (new)
3. FiftyOne Brain results (embeddings, clusters, uniqueness) can flow back to Pixeltable (new)
4. Pixeltable embedding indexes serve as a FiftyOne Brain backend (new, stretch)

## Proposed New Functions

### Import from FiftyOne

```python
# New function in pixeltable/io/fiftyone.py
def import_fo_dataset(
    table_path: str,
    dataset: fo.Dataset,
    *,
    image_column: str = 'image',
    include_labels: bool = True,
    include_metadata: bool = True,
) -> pxt.Table:
    """Import a FiftyOne dataset into a Pixeltable table."""
    schema = {'filepath': pxt.String, image_column: pxt.Image}
    if include_metadata:
        schema['metadata'] = pxt.Json
    if include_labels:
        for field_name, field in dataset.get_field_schema().items():
            if isinstance(field, fo.EmbeddedDocumentField):
                schema[field_name] = pxt.Json

    t = pxt.create_table(table_path, schema, if_exists='ignore')

    rows = []
    for sample in dataset:
        row = {'filepath': sample.filepath, image_column: sample.filepath}
        if include_metadata:
            row['metadata'] = sample.metadata.to_dict() if sample.metadata else {}
        if include_labels:
            for field_name in dataset.get_field_schema():
                if hasattr(sample, field_name) and sample[field_name] is not None:
                    try:
                        row[field_name] = sample[field_name].to_dict()
                    except Exception:
                        pass
        rows.append(row)

    t.insert(rows)
    return t
```

### FiftyOne Brain Backend

This is a stretch goal. LanceDB already has a FiftyOne Brain backend (`fiftyone.brain.compute_similarity`).
We would implement `PixeltableSimilarityBackend`:

```python
class PixeltableSimilarityBackend(SimilarityBackend):
    """Use Pixeltable's embedding index as a FiftyOne Brain similarity backend."""

    def __init__(self, table_name: str, embedding_column: str, **kwargs):
        self._table = pxt.get_table(table_name)
        self._embed_col = embedding_column

    def add_to_index(self, embeddings, sample_ids):
        # Map FiftyOne sample IDs to Pixeltable rows
        pass

    def query_by_id(self, sample_id, k=10):
        # Use Pixeltable similarity search
        pass

    def query_by_vector(self, vector, k=10):
        embed_col = getattr(self._table, self._embed_col)
        sim = embed_col.similarity(vector=vector)
        return self._table.order_by(sim, asc=False).limit(k).collect()
```

## Competitive Landscape

| Backend | Supported by FiftyOne | Bidirectional |
|---------|----------------------|---------------|
| LanceDB | Yes (via plugin) | Yes |
| Pinecone | Yes (via plugin) | Partial |
| Qdrant | Yes (via plugin) | Partial |
| **Pixeltable** | **No (proposed)** | **Yes** |

## Differentiation

Pixeltable's advantage over other backends:
- **Multimodal native**: Image, video, audio columns alongside embeddings
- **Computed columns**: Auto-generate labels, embeddings, and metadata on insert
- **Version control**: Track annotation changes over time
- **Incremental**: Only new samples get processed

## Effort Estimate

| Component | Effort |
|-----------|--------|
| Import from FiftyOne | 2-3 days |
| Sync annotations back | 3-5 days |
| Brain similarity backend | 5-7 days |
| Tests + docs | 2-3 days |

## Prerequisites

- FiftyOne must be installable alongside Pixeltable (dependency conflict check needed)
- FiftyOne Brain backend plugin API documentation review
- Test with FiftyOne 1.x (current stable)

## Next Steps

1. Implement `import_fo_dataset` in `pixeltable/io/fiftyone.py`
2. Add integration test with a small FiftyOne dataset
3. Explore FiftyOne Brain plugin API for similarity backend
4. Write a cookbook notebook demonstrating the round-trip workflow
