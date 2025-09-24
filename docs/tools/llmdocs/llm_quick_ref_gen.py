#!/usr/bin/env python3
"""
Generate LLM-ready documentation for Pixeltable.

Creates 3 files in llm_output/:
1. llm_map.jsonld - Complete public API reference
2. llm_dev_patterns.jsonld - Developer patterns from notebooks
3. llm_quick_reference.md - Guide explaining how to use the files
"""

import json
from pathlib import Path
from datetime import datetime
from llm_dev_pattern_gen import NotebookPatternExtractor
from llm_map_gen import LLMMapGenerator


def generate_quick_reference(output_dir: Path):
    """Generate the quick reference guide that explains how to use the LLM docs."""
    
    guide_content = f"""# Pixeltable LLM Quick Reference

Generated: {datetime.now().isoformat()}

## What is Pixeltable?

Pixeltable is a declarative framework for multimodal data operations that unifies:
- **Tables & Schemas**: Structured storage for images, video, audio, documents, and data
- **Computed Columns**: Automatic transformation pipelines with caching
- **AI Integrations**: Direct access to OpenAI, Anthropic, HuggingFace, and 20+ model providers
- **Incremental Updates**: Changes propagate automatically through your pipeline

## Available Documentation Files

This directory contains 3 files with complete Pixeltable documentation:

### 1. `llm_map.jsonld` - Public API Reference
Complete reference of all public functions, classes, and methods with signatures.

**How to use with jq:**
```bash
# List all modules
jq '.hasPart[] | select(."@type" == "SoftwareSourceCode") | .name' llm_map.jsonld

# Find functions in pixeltable module
jq '.hasPart[] | select(.name == "pixeltable") | .hasPart[] | select(."@type" == "Function") | {{name, signature}}' llm_map.jsonld

# Search for specific function
jq '.. | select(."@id"? == "pxt:pixeltable.create_table")' llm_map.jsonld

# Get formatted signature for a function
jq '.. | select(.name? == "create_table") | .signature.formatted' llm_map.jsonld
```

**How to use with grep -A (simpler for quick lookups):**
```bash
# Find a function and see its signature (use -A 20 for most functions)
grep -A 20 '"name": "create_table"' llm_map.jsonld

# Find a class and see its methods (use -A 50 for classes)
grep -A 50 '"name": "Table"' llm_map.jsonld

# Find formatted signature (use -A 15 to see full signature)
grep -A 15 '"formatted":' llm_map.jsonld | grep -A 15 'create_table'

# Find all functions in a module (use -A 5 per function)
grep -B 2 -A 5 '"@type": "Function"' llm_map.jsonld

# Pro tip: Use -A 20 as a good default for functions, -A 50 for classes
```

### 2. `llm_dev_patterns.jsonld` - Developer Patterns
27 working examples from notebooks showing real Pixeltable usage patterns.

**How to use with jq:**
```bash
# List all example notebooks
jq '.dataset[].name' llm_dev_patterns.jsonld

# Find examples using specific concepts
jq '.dataset[] | select(.keywords | contains(["embedding"])) | .name' llm_dev_patterns.jsonld

# Get GitHub URL for full notebook
jq '.dataset[] | select(.name == "Pixeltable Basics") | .url' llm_dev_patterns.jsonld

# Extract code samples from patterns
jq '.dataset[0].hasPart[0].step[].codeSample.text' llm_dev_patterns.jsonld
```

**How to use with grep -A:**
```bash
# Find examples about a topic (use -A 10 to see description and keywords)
grep -A 10 '"embedding"' llm_dev_patterns.jsonld

# Find a specific notebook (use -A 30 to see its patterns)
grep -A 30 '"name": "RAG Operations in Pixeltable"' llm_dev_patterns.jsonld

# Find code examples (use -A 5 to see the code)
grep -A 5 '"text": "import pixeltable"' llm_dev_patterns.jsonld

# Find GitHub URLs for notebooks
grep '"url":' llm_dev_patterns.jsonld

# Pro tip: Use -A 10 for descriptions, -A 30 for full patterns
```

### 3. `llm_quick_reference.md` - This File
Quick reference and guide to using the documentation.

## Common Tasks

### Creating Tables and Adding Data
```python
import pixeltable as pxt

# Create a table with schema
table = pxt.create_table('my_data', {{
    'image': pxt.Image,
    'text': pxt.String
}})

# Insert data
table.insert({{'image': 'path/to/image.jpg', 'text': 'description'}})
```

### Adding AI-Powered Transformations
```python
from pixeltable.functions import openai

# Add computed column with AI model
table.add_computed_column(
    analysis=openai.vision(
        prompt="Describe this image",
        image=table.image
    )
)
```

### Working with Video
```python
from pixeltable.iterators import FrameIterator

# Create video table
videos = pxt.create_table('videos', {{'video': pxt.Video}})

# Extract frames as a view
frames = pxt.create_view('frames', videos,
    iterator=FrameIterator(video=videos.video, fps=1))
```

### RAG and Embeddings
```python
from pixeltable.iterators import DocumentSplitter

# Create document table
docs = pxt.create_table('docs', {{'document': pxt.Document}})

# Chunk documents
chunks = pxt.create_view('chunks', docs,
    iterator=DocumentSplitter(doc=docs.document, chunk_size=500))

# Add embedding index
chunks.add_embedding_index('text', embedding=openai.embeddings)

# Similarity search
similar = chunks.order_by(chunks.text.similarity("query")).limit(5)
```

## Key Concepts

- **Computed Columns**: Columns that automatically compute values using functions
- **Iterators**: Split data into chunks (frames from video, text chunks, tiles from images)
- **UDFs**: Custom Python functions decorated with `@pxt.udf`
- **Incremental Updates**: New data automatically flows through computed columns
- **Multimodal**: Native support for images, video, audio, documents, and structured data

## Finding More Information

1. **For API details**: Search `llm_map.jsonld` for function signatures and parameters
2. **For examples**: Browse `llm_dev_patterns.jsonld` for working code patterns
3. **For concepts**: Look for keywords in patterns: `computed column`, `embedding`, `iterator`, etc.

## Integration Providers

Pixeltable includes integrations with:
- **LLMs**: OpenAI, Anthropic, Gemini, Mistral, Together, Fireworks, Bedrock, Ollama
- **Vision**: YOLOX, DETR, ResNet
- **Audio**: Whisper, WhisperX
- **Local Models**: HuggingFace, llama.cpp

## Repository

- GitHub: https://github.com/pixeltable/pixeltable
- Documentation: https://docs.pixeltable.com
- Examples: See llm_dev_patterns.jsonld for 27 working notebooks
"""
    
    output_path = output_dir / "llm_quick_reference.md"
    with open(output_path, 'w') as f:
        f.write(guide_content)
    
    print(f"Generated LLM quick reference at {output_path}")


def main():
    """Generate complete LLM documentation suite."""
    
    print("=== Pixeltable LLM Documentation Generator ===\n")
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / 'llm_output'
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate LLM map from OPML
    print("1. Generating llm_map.jsonld from OPML...")
    from opml_reader import OPMLReader
    
    mintlifier_dir = Path(__file__).parent
    llm_map_gen = LLMMapGenerator(mintlifier_dir, version='main')
    
    # Load and process OPML to build the map
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
    
    # Save the map to llm_output with correct name
    llm_map_gen.save(output_dir / 'llm_map.jsonld', flatten=False)
    
    # Step 2: Extract patterns from notebooks
    print("2. Generating llm_dev_patterns.jsonld from notebooks...")
    extractor = NotebookPatternExtractor(
        opml_path=str(mintlifier_dir / 'mintlifier.opml'),
        notebooks_dir=str(mintlifier_dir.parent / 'notebooks')
    )
    extractor.save_patterns(str(output_dir / 'llm_dev_patterns.jsonld'))
    
    # Step 3: Generate quick reference
    print("3. Generating llm_quick_reference.md...")
    generate_quick_reference(output_dir)
    
    print(f"\n✅ Complete! LLM docs generated in {output_dir}")
    print("\nFiles created:")
    print("  - llm_map.jsonld          # Public API reference")
    print("  - llm_dev_patterns.jsonld # Developer patterns from notebooks")
    print("  - llm_quick_reference.md  # Guide to using the files")
    print("\nLLMs can now use these files to understand Pixeltable completely.")


if __name__ == "__main__":
    main()