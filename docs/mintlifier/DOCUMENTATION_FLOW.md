# Documentation Generation Flow

## Overview
This diagram shows how Pixeltable's codebase is transformed into both human-readable and LLM-readable documentation through the Mintlifier system.

## Flow Diagram

```mermaid
graph TD
    %% Input Sources
    PXT[Pixeltable Codebase<br/>Python SDK] 
    OPML[OPML Whitelist<br/>API Surface Definition]
    
    %% Processing Layer
    PXT --> MINT[Mintlifier<br/>Documentation Generator]
    OPML --> MINT
    
    %% Mintlifier Details
    MINT --> INTR[Runtime Introspection<br/>inspect, importlib]
    MINT --> AST[AST Parsing<br/>docstring_parser]
    
    INTR --> PROC[Process & Structure]
    AST --> PROC
    
    %% Output Fork
    PROC --> FORK{Generate Output}
    
    %% Human-Readable Branch
    FORK -->|MDX Pages| MINTLY[Mintlify Docs<br/>Human-Readable SDK Documentation<br/>Per Release Version]
    
    %% LLM-Readable Branch
    FORK -->|JSON-LD| LLM[llm_map.jsonld<br/>LLM-Readable Documentation<br/>GitHub-Linked Sources]
    
    %% Styling
    style PXT fill:#e1f5fe
    style OPML fill:#e1f5fe
    style MINT fill:#fff3e0
    style INTR fill:#f3e5f5
    style AST fill:#f3e5f5
    style PROC fill:#fff3e0
    style FORK fill:#ffecb3
    style MINTLY fill:#c8e6c9
    style LLM fill:#c8e6c9
    
    %% Annotations
    MINTLY -.->|Published to| WEB[docs.pixeltable.com]
    LLM -.->|Used by| AI[LLMs for Code Generation]
    
    style WEB fill:#b3e5fc,stroke-dasharray: 5 5
    style AI fill:#b3e5fc,stroke-dasharray: 5 5
```

## Process Description

### Input Layer
- **Pixeltable Codebase**: The Python SDK source code
- **OPML Whitelist**: Defines which APIs to document (constrains the surface area)

### Processing Layer (Mintlifier)
1. **Runtime Introspection**: Uses Python's `inspect` and `importlib` to examine live objects
2. **AST Parsing**: Uses `docstring_parser` to extract structured documentation
3. **Process & Structure**: Combines runtime and static information

### Output Layer (T-Branch)
The processed information branches into two outputs:

#### Human-Readable Documentation
- **Format**: MDX files with YAML frontmatter
- **Platform**: Mintlify
- **Features**: Navigation, search, versioning
- **Audience**: Developers using the SDK

#### LLM-Readable Documentation  
- **Format**: JSON-LD structured data
- **Features**: GitHub source links, semantic types
- **Audience**: LLMs for code generation and assistance

## Key Components

### Mintlifier Components
- `page_module.py` - Module documentation
- `page_class.py` - Class documentation  
- `page_function.py` - Function documentation
- `page_method.py` - Method documentation
- `page_llmmap.py` - JSON-LD generation

### Output Characteristics

**Mintlify Docs**
- Hierarchical navigation (mod|, class| prefixes)
- Consistent icons (circle-m, square-c, etc.)
- Version dropdown for releases
- GitHub source links

**LLM Map (JSON-LD)**
- Stable `pxt:` identifiers
- Complete signatures with types
- Category inference
- ~400KB flattened size