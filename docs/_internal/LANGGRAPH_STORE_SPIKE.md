# LangGraph Store Backend Spike

## Goal

Implement Pixeltable as a **LangGraph BaseStore** backend, positioning it as "LangGraph for control flow, Pixeltable for multimodal memory."

## LangGraph Store Interface

LangGraph's `BaseStore` (from `langgraph.store.base`) provides a key-value store that agents use for cross-thread memory. The interface:

```python
class BaseStore(ABC):
    async def aget(self, namespace: tuple[str, ...], key: str) -> Optional[Item]
    async def aput(self, namespace: tuple[str, ...], key: str, value: dict, index: Optional[list[str]] = None) -> None
    async def adelete(self, namespace: tuple[str, ...], key: str) -> None
    async def asearch(self, namespace_prefix: tuple[str, ...], /, *, query: Optional[str] = None, ...) -> list[Item]
    async def alist_namespaces(self, *, prefix: Optional[NamespacePath] = None, ...) -> list[tuple[str, ...]]
```

## Pixeltable Mapping

| LangGraph Concept | Pixeltable Equivalent |
|---|---|
| namespace | Directory hierarchy (`pxt.create_dir`) |
| key | Row ID (`pxt.String` column) |
| value | `pxt.Json` column |
| index fields | Computed columns with embedding indexes |
| search | `similarity(string=query)` on indexed fields |

## Proposed Schema

```python
store_table = pxt.create_table('langgraph_store.items', {
    'namespace': pxt.String,    # dot-joined namespace tuple
    'key': pxt.String,
    'value': pxt.Json,
    'created_at': pxt.Timestamp,
    'updated_at': pxt.Timestamp,
})

# Index fields extracted from value dict
store_table.add_computed_column(
    searchable_text=extract_index_fields(store_table.value),
    if_exists='ignore',
)

store_table.add_embedding_index(
    'searchable_text',
    string_embed=embeddings.using(model='text-embedding-3-small'),
    if_not_exists=True,
)
```

## Differentiation from InMemoryStore / PostgresStore

| Feature | InMemoryStore | PostgresStore | **PixeltableStore** |
|---------|---------------|---------------|---------------------|
| Persistence | No | Yes | Yes + versioned |
| Multimodal values | No | No | Yes (Image, Video, Audio) |
| Incremental embedding | No | Manual | Automatic via computed columns |
| Cross-thread search | Text only | Text only | Text + Image + Video |
| History | No | No | Full version history |

## Implementation Plan

1. Create `pixeltable/langchain-pixeltable` package extension or separate `langgraph-pixeltable` package
2. Implement `PixeltableStore(BaseStore)` with async get/put/delete/search/list_namespaces
3. Map namespace tuples to dot-joined strings for Pixeltable table organization
4. Use `pxt.Json` for value storage, extract indexable fields via UDF
5. Add multimodal index support (store Image/Video references in value, index them separately)
6. Test with LangGraph's `create_react_agent` + `MemorySaver` pattern

## Effort Estimate

- Core implementation: 2-3 days
- Tests + docs: 1-2 days
- Multimodal extension: 3-5 days (stretch goal)

## Next Steps

- Review LangGraph's `BaseStore` test suite for compliance requirements
- Prototype against `langgraph>=0.4` (current stable)
- Decide: extend `langchain-pixeltable` or create `langgraph-pixeltable`
