# Pixeltable LLM Developer Guide

> AI-ready documentation for building multimodal data workflows with Pixeltable

Generated: 2025-08-27T01:09:59.093837

## Quick Start

Pixeltable unifies data operations, ML models, and orchestration into a declarative framework optimized for AI workloads.

### Install
```python
pip install pixeltable
```

### Basic Example
```python
import pixeltable as pxt

# Create a table
table = pxt.create_table('media_analysis', {
    'image': pxt.Image,
    'description': pxt.String
})

# Add AI-powered computed column
table.add_computed_column(
    analysis=openai.vision(table.image, "Describe this image")
)

# Query results
table.select(table.image, table.analysis).show()
```

## Available Resources

### 📚 Complete API Reference
- **File**: `llm_map.jsonld`
- **Contents**: 0 functions, 0 classes
- **Format**: JSON-LD for semantic understanding
- **Usage**: Load this for comprehensive API details

### 🎯 Pattern Library  
- **File**: `llm_patterns.json`
- **Contents**: 27 notebook examples
- **Format**: Structured patterns with context
- **Usage**: Find working examples for common tasks

### 🚀 Key Concepts

1. **Tables & Schemas** - Structured multimodal data storage
2. **Computed Columns** - Automatic transformation pipelines  
3. **UDFs** - Custom Python functions as data operations
4. **Iterators** - Frame extraction, chunking, splitting
5. **Embeddings** - Built-in vector search and similarity
6. **Tool Calling** - Agentic workflows with LLM orchestration

## Common Patterns

### Multimodal Pipeline
```python
# Process video → frames → analysis
video_table = pxt.create_table('videos', {'video': pxt.Video})
frames = pxt.create_view('frames', video_table, 
    iterator=FrameIterator(video=video_table.video, fps=1))
frames.add_computed_column(
    objects=yolox.detect(frames.frame)
)
```

### RAG System
```python  
# Documents → chunks → embeddings → search
docs = pxt.create_table('documents', {'doc': pxt.Document})
chunks = pxt.create_view('chunks', docs,
    iterator=DocumentSplitter(doc=docs.doc, chunk_size=500))
chunks.add_embedding_index('text', embedding=openai.embeddings)
similar = chunks.order_by(chunks.text.similarity("query")).limit(5)
```

### Tool-Calling Agent
```python
@pxt.udf
def search_knowledge(query: str) -> str:
    return db.search(query)

tools = pxt.tools(search_knowledge)
agent = pxt.create_table('agent', {'prompt': pxt.String})
agent.add_computed_column(
    response=openai.chat(agent.prompt, tools=tools)
)
agent.add_computed_column(
    result=invoke_tools(tools, agent.response)
)
```

## Integration with LLMs

### For Code Generation
When generating Pixeltable code:
1. Reference patterns from `llm_patterns.json`
2. Use only public API from `llm_map.jsonld`
3. Follow the declarative pattern: table → computed columns → query

### For Tool Use
Pixeltable functions can be exposed as tools:
- Use `@pxt.udf` decorated functions
- Register with `pxt.tools()`
- Invoke with `invoke_tools()`

### For RAG Systems
Pixeltable provides built-in RAG primitives:
- Document chunking via iterators
- Embedding indexes for similarity search
- Metadata filtering with SQL-like queries

## File Manifest

```
llm_docs/
├── llm_quick_guide.md      # This file
├── llm_map.jsonld          # Complete API reference
├── llm_patterns.json       # Notebook patterns
└── llm_patterns_summary.md # Pattern overview
```

## Version Information

- Pixeltable Version: main
- Documentation Generated: 2025-08-27T01:09:59.093843
- API Elements: 0 total

## Learn More

- Repository: https://github.com/pixeltable/pixeltable
- Documentation: https://docs.pixeltable.com
- Examples: See llm_patterns.json for 27 working examples

---

*This documentation is optimized for consumption by large language models and AI coding assistants.*
