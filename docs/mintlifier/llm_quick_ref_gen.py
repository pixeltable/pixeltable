#!/usr/bin/env python3
"""
Generate complete LLM documentation suite.

This script coordinates generation of:
1. llm_map.jsonld - API reference from OPML
2. llm_patterns.json - Extracted notebook patterns  
3. llm_quick_guide.md - Entry point documentation
"""

import json
from pathlib import Path
from datetime import datetime
from llm_dev_pattern_gen import NotebookPatternExtractor
from llm_map_gen import LLMMapGenerator


def generate_quick_guide(api_path: str, patterns_path: str, output_path: str):
    """Generate the LLM quick guide that ties everything together."""
    
    # Load the data
    with open(api_path, 'r') as f:
        api_data = json.load(f)
    
    with open(patterns_path, 'r') as f:
        patterns_data = json.load(f)
    
    # Count resources
    num_functions = len([p for p in api_data.get('hasPart', []) 
                         if p.get('@type') == 'Function'])
    num_classes = len([p for p in api_data.get('hasPart', []) 
                       if p.get('@type') == 'Class'])
    num_patterns = len(patterns_data.get('notebooks', []))
    
    guide_content = f"""# Pixeltable LLM Developer Guide

> AI-ready documentation for building multimodal data workflows with Pixeltable

Generated: {datetime.now().isoformat()}

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
table = pxt.create_table('media_analysis', {{
    'image': pxt.Image,
    'description': pxt.String
}})

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
- **Contents**: {num_functions} functions, {num_classes} classes
- **Format**: JSON-LD for semantic understanding
- **Usage**: Load this for comprehensive API details

### 🎯 Pattern Library  
- **File**: `llm_patterns.json`
- **Contents**: {num_patterns} notebook examples
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
video_table = pxt.create_table('videos', {{'video': pxt.Video}})
frames = pxt.create_view('frames', video_table, 
    iterator=FrameIterator(video=video_table.video, fps=1))
frames.add_computed_column(
    objects=yolox.detect(frames.frame)
)
```

### RAG System
```python  
# Documents → chunks → embeddings → search
docs = pxt.create_table('documents', {{'doc': pxt.Document}})
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
agent = pxt.create_table('agent', {{'prompt': pxt.String}})
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

- Pixeltable Version: {api_data.get('version', 'unknown')}
- Documentation Generated: {datetime.now().isoformat()}
- API Elements: {num_functions + num_classes} total

## Learn More

- Repository: https://github.com/pixeltable/pixeltable
- Documentation: https://docs.pixeltable.com
- Examples: See llm_patterns.json for {num_patterns} working examples

---

*This documentation is optimized for consumption by large language models and AI coding assistants.*
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(guide_content)
    
    print(f"Generated LLM quick guide at {output_path}")


def main():
    """Generate complete LLM documentation suite."""
    
    print("=== Pixeltable LLM Documentation Generator ===\n")
    
    # Paths
    mintlifier_dir = Path(__file__).parent
    output_dir = mintlifier_dir / 'llm_output'
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate LLM map from OPML
    print("1. Generating LLM map from OPML...")
    llm_map_gen = LLMMapGenerator(mintlifier_dir, version='main')
    
    # Load and process OPML to build the map
    from opml_reader import OPMLReader
    opml_reader = OPMLReader(mintlifier_dir / 'mintlifier.opml')
    tab_structure = opml_reader.load()
    all_pages = opml_reader.get_all_pages()
    
    # Process each page through the LLM map generator
    for page in all_pages:
        if page.item_type == 'module':
            llm_map_gen.add_module(page.module_path, page.children)
        elif page.item_type == 'class':
            llm_map_gen.add_class(page.module_path, page.children)
        elif page.item_type == 'func':
            llm_map_gen.add_function(page.module_path)
        elif page.item_type == 'type':
            llm_map_gen.add_type(page.module_path)
    
    # Save the generated map
    llm_map_gen.save()
    llm_map_path = mintlifier_dir / 'llm_map.jsonld'
    
    # Step 2: Extract patterns from notebooks
    print("2. Extracting patterns from notebooks...")
    extractor = NotebookPatternExtractor(
        opml_path=str(mintlifier_dir / 'mintlifier.opml'),
        notebooks_dir=str(mintlifier_dir.parent / 'notebooks')
    )
    patterns_path = output_dir / 'llm_patterns.json'
    extractor.save_patterns(str(patterns_path))
    
    # Step 3: Generate quick guide
    print("3. Generating LLM quick guide...")
    generate_quick_guide(
        api_path=str(llm_map_path),
        patterns_path=str(patterns_path),
        output_path=str(output_dir / 'llm_quick_guide.md')
    )
    
    print(f"\n✅ Complete! LLM docs generated in {output_dir}")
    print("\nFiles created:")
    for file in output_dir.glob('*'):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()