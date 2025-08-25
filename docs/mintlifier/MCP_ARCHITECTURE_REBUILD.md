# Pixeltable MCP Server - Architecture Rebuild

## Overview

The MCP server should be Claude's interface to Pixeltable, hiding complexity while exposing full capability. Claude should be able to understand user intent and generate complete, working Pixeltable workflows without the user needing any Pixeltable knowledge.

## Core Design Principles

1. **Zero Pixeltable knowledge required** - Users describe what they want, not how
2. **Claude does the reasoning** - MCP provides capabilities, Claude orchestrates
3. **Examples over documentation** - Show working patterns, not abstract concepts
4. **Progressive disclosure** - Simple tasks simple, complex tasks possible

## What Claude Needs from the MCP

### 1. Understanding Pixeltable's Paradigm

```python
@mcp_function
def get_pixeltable_concepts() -> ConceptMap:
    """
    Returns the mental model of Pixeltable:
    - Tables are not just storage, they're computation graphs
    - Computed columns are the workflow (DAG nodes)
    - Everything is lazy until queried
    - Media files stay external, referenced by path
    - Models download automatically on first use
    """
    return {
        "core_concept": "Computation as data - schemas define workflows",
        "data_flow": "Input → Table → Computed Columns → Results",
        "key_insight": "The schema IS the workflow",
        "examples": [...]
    }
```

### 2. Model Discovery & Capabilities

```python
@mcp_function
def list_available_models(
    task: str = None,  # "object_detection", "transcription", "NER", etc.
    modality: str = None  # "image", "video", "audio", "text"
) -> ModelRegistry:
    """
    Returns what models are available for a task:
    - HuggingFace models that work with Pixeltable
    - Built-in Pixeltable functions
    - Required dependencies and auto-install capability
    """
    return {
        "huggingface": [
            {
                "name": "yolox",
                "task": "object_detection", 
                "modality": "image",
                "usage": "yolox(image_column, model='yolox_s')",
                "auto_download": True
            },
            {
                "name": "whisperx",
                "task": "transcription",
                "modality": "audio",
                "usage": "whisperx.transcribe(audio_column)",
                "requires_install": "pixeltable-whisperx"
            }
        ],
        "builtin": [
            {
                "name": "image.resize",
                "task": "transformation",
                "modality": "image",
                "usage": "image.resize(img_col, width=224, height=224)"
            }
        ]
    }
```

### 3. Workflow Patterns Library

```python
@mcp_function
def get_workflow_patterns() -> PatternLibrary:
    """
    Returns proven patterns for common tasks.
    Claude can adapt these rather than creating from scratch.
    """
    return {
        "video_analysis": {
            "pattern": "video → frames → detection → aggregation",
            "code_template": """
                t = pxt.create_table('analysis', {'video': pxt.Video})
                t.add_computed_column(frames=FrameIterator(t.video, fps=1))
                t.add_computed_column(detections=yolox(t.frames))
                t.add_computed_column(summary=aggregate_detections(t.detections))
            """,
            "variations": ["with_transcription", "with_thumbnails"]
        },
        "rag_pipeline": {...},
        "image_classification": {...}
    }
```

### 4. Schema Builder with Validation

```python
@mcp_function
def build_schema(
    description: str,
    input_data: dict = None,  # Sample of user's data
    requirements: list = None  # ["detect objects", "extract text", etc.]
) -> SchemaDefinition:
    """
    Helps Claude build valid Pixeltable schemas.
    Returns both the schema and the code to create it.
    """
    # Analyzes requirements and suggests schema
    return {
        "schema": {
            "columns": {
                "image_path": "String",
                "image": {"type": "Image", "computed": "load_from_path(image_path)"},
                "objects": {"type": "JSON", "computed": "yolox(image)"},
                "species": {"type": "String", "computed": "classify_species(objects)"}
            }
        },
        "code": "# Complete Pixeltable code...",
        "validation": {
            "valid": True,
            "warnings": [],
            "suggestions": ["Consider adding index on species column"]
        }
    }
```

### 5. Expression Syntax Helper

```python
@mcp_function
def validate_expression(
    expression: str,
    table_schema: dict,
    context: str = None
) -> ExpressionValidation:
    """
    Validates Pixeltable expressions and provides corrections.
    Critical for Claude to write correct computed columns.
    """
    # Example: "Image.from_path(t.path)" -> "pxt.functions.image.load(t.path)"
    return {
        "original": expression,
        "corrected": "t.image.resize(224, 224)",
        "explanation": "Image methods are called on column references",
        "alternatives": [...]
    }
```

### 6. Data Type Understanding

```python
@mcp_function
def get_type_system() -> TypeSystem:
    """
    Returns Pixeltable's type system with examples.
    """
    return {
        "media_types": {
            "Image": {
                "creation": "pxt.Image",
                "storage": "external_file",
                "methods": ["resize", "crop", "rotate"],
                "computed_example": "t.add_computed_column(thumb=t.image.resize(128))"
            },
            "Video": {...},
            "Audio": {...}
        },
        "structured_types": {
            "String": "pxt.String",
            "Int": "pxt.Int",
            "JSON": "pxt.Json"
        },
        "conversion_rules": [
            "String paths can become Images via computed columns",
            "Videos can produce frame iterators",
            "JSON can store detection results"
        ]
    }
```

### 7. Execution & Testing

```python
@mcp_function
def test_workflow(
    code: str,
    sample_data: dict = None,
    dry_run: bool = True
) -> TestResults:
    """
    Tests generated workflow before full execution.
    """
    if dry_run:
        # Just validate syntax and schema
        return validate_only(code)
    else:
        # Actually run with sample data
        return execute_with_sample(code, sample_data)
```

### 8. The Master Orchestrator

```python
@mcp_function
def generate_workflow(
    user_request: str,
    data_sample: dict = None,
    constraints: dict = None  # {"max_cost": 100, "latency": "real-time"}
) -> CompleteWorkflow:
    """
    The main function Claude calls to generate complete workflows.
    This orchestrates all the other functions internally.
    """
    # This is where the magic happens
    # 1. Understand intent
    # 2. Check available models
    # 3. Build schema
    # 4. Generate code
    # 5. Validate
    # 6. Return complete solution
    
    return {
        "understanding": "User wants to classify wildlife photos by species",
        "workflow_code": """
            import pixeltable as pxt
            
            # Create table for wildlife monitoring
            wildlife = pxt.create_table('wildlife_photos', {
                'image_path': pxt.String,
                'location': pxt.String,
                'timestamp': pxt.Timestamp
            })
            
            # Add computed columns for the workflow
            wildlife.add_computed_column(
                image=pxt.functions.image.load(wildlife.image_path)
            )
            wildlife.add_computed_column(
                detections=pxt.functions.yolox(wildlife.image)
            )
            wildlife.add_computed_column(
                species=pxt.functions.classify_species(wildlife.detections)
            )
            
            # Import data
            wildlife.insert(data)
            
            # Query results
            results = wildlife.select(wildlife.species, wildlife.location).collect()
        """,
        "explanation": "This workflow loads images, detects objects, and classifies species",
        "models_used": ["yolox", "species_classifier"],
        "estimated_cost": "$0.02 per 1000 images",
        "alternatives": ["Use CLIP for zero-shot classification", "Add confidence filtering"]
    }
```

## Parquet Generation Strategy

### Option A: External Generation (Preferred)
Generate Parquet files outside Pixeltable with embedded metadata:

```python
@mcp_function
def generate_workflow_parquet(
    workflow: CompleteWorkflow,
    output_path: str
) -> ParquetFile:
    """
    Creates a Parquet file that Pixeltable can import.
    No Pixeltable dependency required for generation.
    """
    # Create schema metadata
    metadata = {
        'pixeltable_schema': workflow.schema,
        'computed_columns': workflow.computed_columns,
        'udfs': workflow.custom_functions
    }
    
    # Create Parquet with PyArrow
    table = pa.Table.from_pydict(workflow.initial_data)
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output_path)
    
    return {
        "path": output_path,
        "import_command": f"pxt.import_parquet('{output_path}')",
        "schema_preserved": True
    }
```

### Option B: Use Pixeltable API
If external generation proves too complex:

```python
@mcp_function
def generate_via_pixeltable(
    workflow: CompleteWorkflow
) -> str:
    """
    Uses Pixeltable directly to create and export workflow.
    """
    # Execute workflow creation
    exec(workflow.code)
    
    # Export to Parquet
    table = pxt.get_table(workflow.table_name)
    pxt.io.export_parquet(table, 'workflow.parquet')
    
    return 'workflow.parquet'
```

**Recommendation**: Start with Option A (external generation) for portability, fall back to Option B if needed.

## What NOT to Include

1. **Individual table manipulation functions** - These are internal details
2. **Complex configuration options** - Use smart defaults
3. **Direct SQL/SPARQL access** - Too low level
4. **Manual dependency management** - Should be automatic
5. **Separate functions for every operation** - One smart function is better than 50 specific ones

## Success Metrics

Claude can successfully:
1. Take a user request like "analyze my drone footage for cars"
2. Generate complete, working Pixeltable code
3. Handle errors and suggest fixes
4. Explain what the workflow does in user terms
5. Export as shareable Parquet file

## Example User Session

```
User: "I have wildlife camera photos in /photos and want to track different species over time"

Claude: [Calls generate_workflow()]
"I'll create a wildlife monitoring system for you. This will:
1. Load your photos from /photos
2. Detect animals using YOLOX
3. Classify species using a wildlife model
4. Track sightings over time

Here's the complete code: [shows code]

Shall I run this on a sample of your photos first?"

User: "Yes, test it"

Claude: [Calls test_workflow()]
"Test successful! Found 3 species in 10 sample photos: deer (5), raccoon (3), fox (2).
Ready to process all photos?"
```

## Implementation Priority

1. **Phase 1 - Core** (Must have for demo)
   - `generate_workflow()` - Main orchestrator
   - `list_available_models()` - Model discovery
   - `get_pixeltable_concepts()` - Basic understanding

2. **Phase 2 - Enhancement** (Nice to have)
   - `test_workflow()` - Testing capability
   - `get_workflow_patterns()` - Pattern library
   - `validate_expression()` - Expression helper

3. **Phase 3 - Advanced** (Future)
   - `generate_workflow_parquet()` - External generation
   - Custom UDF generation
   - Cost optimization

## Key Insight

The MCP should be Claude's "Pixeltable brain" - it provides the knowledge and capabilities, but Claude does the reasoning and user interaction. The MCP doesn't need to be smart about user intent; it needs to be smart about Pixeltable and expose that smartness in a way Claude can use.

Think of it as:
- **Claude**: Understands what users want
- **MCP**: Understands what Pixeltable can do
- **Together**: Magic happens