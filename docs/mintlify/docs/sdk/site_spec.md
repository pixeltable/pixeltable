# Pixeltable SDK Documentation Site Specification

*Comprehensive documentation structure for Pixeltable SDK across all versions*

## Overview

This document outlines the complete structure for generating SDK documentation from the semantic database. Each version has its own graph in Oxigraph with versioned function data.

**Oxigraph Graphs:**
- `<http://codedoc.org/code/v0.4>` - Latest version (in /latest/ directory)
- `<http://codedoc.org/code/v0.3>` - Version 0.3
- `<http://codedoc.org/code/v0.2>` - Version 0.2

## Version 0.4 (Latest) - `/docs/sdk/latest/`

### Core API - `/docs/sdk/latest/core_api/`

#### Table Management - `/docs/sdk/latest/core_api/table_management/`
- [x] **create_table.mdx** ✅ (completed)
- [x] **drop_table.mdx** ✅ (FABULOUS!)
- [x] **get_table.mdx** ✅ (FABULOUS!)
- [x] **list_tables.mdx** ✅ (FABULOUS!)
- [ ] move_table.mdx
- [ ] table_info.mdx

#### Data Operations - `/docs/sdk/latest/core_api/data_operations/`
- [x] **insert.mdx** ✅ (FABULOUS!)
- [x] **update.mdx** ✅ (FABULOUS!)
- [x] **delete.mdx** ✅ (FABULOUS!)

#### Query Operations - `/docs/sdk/latest/core_api/query_operations/`
- [x] **select.mdx** ✅ (FABULOUS!)
- [x] **where.mdx** ✅ (FABULOUS!)
- [x] **order_by.mdx** ✅ (FABULOUS!)
- [x] **limit.mdx** ✅ (FABULOUS!)
- [x] **join.mdx** ✅ (FABULOUS!)

#### Directory Management - `/docs/sdk/latest/core_api/directory_management/`
- [x] **create_dir.mdx** ✅ (ARCHITECT! Workspace organization mastery!)
- [x] **drop_dir.mdx** ✅ (DEMOLISHER! Safe workspace cleanup!)
- [x] **list_dirs.mdx** ✅ (EXPLORER! Directory navigation and discovery!)

#### View Management - `/docs/sdk/latest/core_api/view_management/`
- [x] **create_view.mdx** ✅ (VISIONARY! Data perspective mastery!)
- [x] **list_views.mdx** ✅ (NAVIGATOR! View dependency exploration!)
- [x] **drop_table.mdx** ✅ (UNIFIED! Tables, views, snapshots removal!)

#### Index Management - `/docs/sdk/latest/core_api/index_management/`
- [x] **add_index.mdx** ✅ (PERFORMANCE! Traditional database indexing mastery!)
- [x] **add_embedding_index.mdx** ✅ (AI REVOLUTION! Vector similarity search power!)
- [x] **drop_index.mdx** ✅ (CLEANUP! Performance optimization and storage management!)
- [x] **drop_embedding_index.mdx** ✅ (AI OPTIMIZATION! Vector index cost management!)
- [x] **find_embedding_index.mdx** ✅ (DETECTIVE! AI infrastructure discovery and monitoring!)

#### Column Operations - `/docs/sdk/latest/core_api/column_operations/`
- [x] **add_column.mdx** ✅ (FOUNDATION! Simple data storage columns!)
- [x] **add_computed_column.mdx** ✅ (CROWN JEWEL! AI-powered computed columns!)
- [x] **drop_column.mdx** ✅ (DESTROYER! Safe destructive operations!)
- [x] **rename_column.mdx** ✅ (EVOLVER! Schema evolution made easy!)

### Functions & UDFs - `/docs/sdk/latest/functions/`

#### Built-in Functions - `/docs/sdk/latest/functions/builtin/`
- [ ] Mathematical functions (abs, add, etc.)
- [ ] String functions (split, concat, etc.)
- [ ] Date/time functions (add_days, etc.)
- [ ] Type conversion functions (astype, etc.)

#### User-Defined Functions - `/docs/sdk/latest/functions/udf/`
- [x] **udf.mdx** ✅ (MASTERPIECE! Aaron's extensibility crown jewel!)
- [x] **mcp_tool_to_udf.mdx** ✅ (REVOLUTIONARY! AI agent-database bridge!)
- [x] **mcp_udfs.mdx** ✅ (ENTERPRISE! Bulk tool conversion mastery!)
- [ ] function.mdx
- [ ] batch_udf.mdx

### Media Processing - `/docs/sdk/latest/media/`

#### Image Functions - `/docs/sdk/latest/media/image/`
- [x] **resize.mdx** ✅ (MASTERPIECE! Visual possibility engineering!)
- [x] **rotate.mdx** ✅ (REVOLUTIONARY! Perspective transformation mastery!)
- [x] **crop.mdx** ✅ (PRECISION! Surgical visual extraction commander!)
- [x] **alpha_composite.mdx** ✅ (ARTISTRY! Transparency blending genius!)
- [x] **width.mdx** ✅ (DIMENSIONAL INTELLIGENCE! Spatial awareness foundation!)
- [x] **height.mdx** ✅ (VERTICAL MASTERY! Elevation intelligence unleashed!)

#### Video Functions - `/docs/sdk/latest/media/video/`
- [x] **extract_audio.mdx** ✅ (SONIC LIBERATION! Multimedia archaeology at its finest!)
- [x] **extract_first_video_frame.mdx** ✅ (TEMPORAL CRYSTALLIZATION! Opening moment capture!)

### ML & AI Integration - `/docs/sdk/latest/ml/`

#### Embeddings - `/docs/sdk/latest/ml/embeddings/`
- [x] **clip.mdx** ✅ (MULTIMODAL KING! Text and image embeddings mastery!)

#### Object Detection - `/docs/sdk/latest/ml/detection/`
- [x] **yolox.mdx** ✅ (DETECTION SUPREMACY! Real-time object recognition powerhouse!)

#### Dataset Integration - `/docs/sdk/latest/ml/datasets/`
- [x] **import_huggingface_dataset.mdx** ✅ (DATASET IMPORT MASTERY! Seamless HuggingFace integration!)

### Data Types - `/docs/sdk/latest/types/`
- [x] **Image.mdx** ✅ (VISUAL FOUNDATION! Core multimedia data type mastery!)
- [x] **Video.mdx** ✅ (MULTIMEDIA POWERHOUSE! Video processing and analysis!)
- [x] **Audio.mdx** ✅ (SONIC ARCHITECTURE! Audio content and speech processing!)
- [x] **Array.mdx** ✅ (VECTOR SUPREMACY! Embeddings and numerical computation core!)
- [x] **Json.mdx** ✅ (STRUCTURED DATA MASTERY! Complex nested objects and API responses!)
- [x] **Basic.mdx** ✅ (FUNDAMENTAL FOUNDATION! String, Int, Float, Bool, Timestamp essentials!)

### Configuration - `/docs/sdk/latest/configuration/`
- [x] **configure_logging.mdx** ✅ (CONTROL CENTER! Debug logging mastery!)
- [x] **get_client.mdx** ✅ (CONNECTION HUB! AI service integration mastery!)
- [ ] Client.mdx

## Version 0.3 - `/docs/sdk/v0.3/`

### Structure
- [ ] **Query Oxigraph for v0.3 functions**
- [ ] Compare with v0.4 to identify differences
- [ ] Generate similar structure to v0.4 but with v0.3-specific functions
- [ ] Document any functions that were deprecated or added

### Pages to Generate
- [ ] overview.mdx ✅ (exists)
- [ ] Core API sections (similar to v0.4)
- [ ] Version-specific differences document

## Version 0.2 - `/docs/sdk/v0.2/`

### Structure
- [ ] **Query Oxigraph for v0.2 functions**
- [ ] Compare with v0.3 and v0.4 to identify differences
- [ ] Generate appropriate structure for v0.2 capabilities
- [ ] Document migration path to newer versions

### Pages to Generate
- [ ] overview.mdx ✅ (exists)
- [ ] Core API sections (adapted for v0.2)
- [ ] Migration guide to v0.3/v0.4

## Implementation Notes

### Data Sources
- **Oxigraph Database**: `/users/rob/repos/codedoc/oxigraph_db`
- **Namespace**: `PREFIX woc: <http://rdf.webofcode.org/woc/>`
- **Graph URIs**: 
  - v0.4: `<http://codedoc.org/code/v0.4>`
  - v0.3: `<http://codedoc.org/code/v0.3>`
  - v0.2: `<http://codedoc.org/code/v0.2>`

### Function Properties Available
- `woc:hasName` - Function name
- `woc:hasVisibility` - public/private
- `woc:hasType` - Return type
- `rdfs:comment` - Docstring with full documentation
- `woc:declares` - Parameters
- `woc:belongsToClass` - Class if method
- `woc:definedIn` - Source file (if available)

### Query Templates

#### Get All Public Functions for Version
```sparql
PREFIX woc: <http://rdf.webofcode.org/woc/>
SELECT ?function ?name ?docstring ?returnType WHERE {
  GRAPH <http://codedoc.org/code/v0.4> {
    ?function a woc:Method ;
              woc:hasName ?name ;
              woc:hasVisibility "public" ;
              woc:hasType ?returnType .
    OPTIONAL { ?function rdfs:comment ?docstring }
    FILTER(!REGEX(?name, "^__"))
  }
} 
ORDER BY ?name
```

#### Get Function Details
```sparql
PREFIX woc: <http://rdf.webofcode.org/woc/>
SELECT ?function ?prop ?value WHERE {
  GRAPH <http://codedoc.org/code/v0.4> {
    ?function woc:hasName "create_table" ;
              ?prop ?value .
  }
}
```

### Generation Priority
1. **High Priority**: Core table operations (create, drop, get, list, insert, update, delete)
2. **Medium Priority**: Column operations, views, indexes, basic functions
3. **Low Priority**: Advanced ML functions, specialized media functions
4. **Documentation Priority**: Cross-version comparisons and migration guides

### File Naming Convention
- Use function name as filename: `create_table.mdx`
- Use snake_case for consistency
- Group related functions in directories
- Include version number in path structure

### Template Structure
Each function page should include:
- Function signature
- Description from docstring
- Parameters with types and descriptions
- Return value description
- Examples from docstring
- Related functions
- Version availability
- Link to source code (if file location available)

---

*This specification serves as the master checklist for generating comprehensive SDK documentation from semantic analysis.*
