# Guidelines for Writing Pixeltable Notebooks

**Purpose**: Best practices for creating educational and functional Jupyter notebooks for Pixeltable documentation

---

## Overview

These guidelines ensure that Jupyter notebooks convert properly to Mintlify MDX format using Quarto. The conversion process preserves YAML frontmatter and converts markdown/code cells to MDX.

## Required: YAML Frontmatter

Every notebook MUST start with a **raw cell** (not markdown) containing YAML frontmatter.

### How to Add YAML Frontmatter in Jupyter

1. Insert a new cell at the **very top** of the notebook
2. Change cell type to **Raw** (not Markdown, not Code)
   - In Jupyter: Cell → Cell Type → Raw
   - In JupyterLab: Click cell type dropdown and select "Raw"
3. Add the YAML frontmatter block

### Exact Frontmatter Template

**This is the exact format to use for every notebook.** Simply replace:
- `Your Notebook Title` with your notebook's title
- `path/to/your-notebook.ipynb` with the actual path (e.g., `use-cases/rag-operations.ipynb`)

```yaml
---
title: "Your Notebook Title"
icon: "notebook"
description: "[Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/pixeltable/pixeltable/blob/release/docs/notebooks/path/to/your-notebook.ipynb) | [Open in Colab](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/path/to/your-notebook.ipynb) | [View on GitHub](https://github.com/pixeltable/pixeltable/blob/release/docs/notebooks/path/to/your-notebook.ipynb)"
---
```

**Example for a notebook at `docs/notebooks/use-cases/rag-operations.ipynb`:**

```yaml
---
title: "RAG Operations"
icon: "notebook"
description: "[Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/rag-operations.ipynb) | [Open in Colab](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/rag-operations.ipynb) | [View on GitHub](https://github.com/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/rag-operations.ipynb)"
---
```

### Frontmatter Fields

- **title**: The display title for the notebook page (required)
  - Use title case (e.g., "Working with OpenAI")
  - This becomes the H1 heading in the rendered documentation
- **icon**: Always use `"notebook"` for consistency across all notebooks
- **description**: Contains three links separated by ` | ` (space-pipe-space)
  - **Kaggle link**: Opens notebook in Kaggle kernel
  - **Colab link**: Opens notebook in Google Colab
  - **GitHub link**: Views notebook source on GitHub
  - All three links use the `release` branch for stability
  - Path must be relative to `docs/notebooks/` directory

## Required: Remove H1 Headers from Markdown

**Do NOT use H1 headers (`#`) in markdown cells.** The title comes from the YAML frontmatter.

### ❌ Wrong
```markdown
# Pixeltable Basics

Welcome to this tutorial...
```

### ✅ Correct
```markdown
Welcome to this tutorial...

## Section Title

Content here...
```

### Header Hierarchy

- **Frontmatter `title`**: Acts as the H1 (page title)
- **`##` (H2)**: Main sections
- **`###` (H3)**: Subsections
- **`####` (H4)**: Sub-subsections

## Required: No Download or Badge Links

**Do NOT include "Download Notebook" badges or similar HTML badge images in markdown cells.**

These cause MDX parsing errors because HTML `<img>` tags with self-closing syntax are not compatible with MDX.

### ❌ Wrong
```html
<a href="..."><img src="https://img.shields.io/badge/..." alt="Download Notebook"></a>
```

The Kaggle/Colab/GitHub links should only appear in the frontmatter `description` field.

## Required: Use Full GitHub URLs

Use `raw.githubusercontent.com` (not `raw.github.com`) for any GitHub raw content links.

### ❌ Wrong
```python
image_url = 'https://raw.github.com/pixeltable/pixeltable/release/docs/resources/images/example.jpg'
```

### ✅ Correct
```python
image_url = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images/example.jpg'
```

## Required: Use Current Documentation Links

Link to current Pixeltable documentation structure, not old readme.io links.

### ❌ Wrong
```markdown
Check out the [tutorial](https://pixeltable.readme.io/docs/tutorial)
```

### ✅ Correct
```markdown
Check out the [tutorial](https://docs.pixeltable.com/tutorials/getting-started)
```

Or use relative links to other notebooks:
```markdown
See the [Object Detection](../use-cases/object-detection-in-videos.ipynb) tutorial
```

## Recommended: Code Style

Follow these conventions for consistency:

### Code Formatting
- Use clear, descriptive variable names
- Add comments for complex operations
- Keep code cells focused (one concept per cell when possible)

### Imports
- Group imports at the top of the notebook
- Standard library first, then third-party, then pixeltable

```python
import os
from pathlib import Path

import pandas as pd

import pixeltable as pxt
from pixeltable.functions import openai
```

### Output Display
- Clear output before committing notebooks (when appropriate)
- Keep meaningful outputs that help explain concepts
- For long outputs, consider using `head()` or limiting results

## Recommended: Markdown Style

### Explanatory Text
- Start with a brief introduction explaining what the notebook covers
- Use clear section headers to organize content
- Explain WHY before showing HOW
- Include context for code examples

### Links and References
- Provide links to relevant documentation
- Reference prerequisite knowledge when needed
- Link to related notebooks for deeper exploration

### Code Comments vs Markdown
- Use markdown cells for explanations and concepts
- Use code comments for implementation details
- Don't duplicate information between markdown and comments

## Testing Your Notebook

Before committing, verify your notebook:

1. **Run all cells** from a fresh kernel to ensure reproducibility
2. **Check frontmatter** is in a Raw cell at the very top
3. **Verify no H1 headers** (`#`) in markdown cells (use `##` and below)
4. **Test links** to ensure they point to correct locations
5. **Review output** to ensure it's appropriate for documentation

## Example Notebook Structure

```
┌─────────────────────────────────────┐
│ RAW CELL (Cell Type: Raw)          │
│ ---                                 │
│ title: "Example Notebook"           │
│ icon: "notebook"                    │
│ description: "..."                  │
│ ---                                 │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ MARKDOWN CELL                       │
│ Brief introduction to the notebook  │
│                                     │
│ ## Prerequisites                    │
│ What you need to know...            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ CODE CELL                           │
│ # Imports                           │
│ import pixeltable as pxt            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ MARKDOWN CELL                       │
│ ## First Section                    │
│ Explanation of what we'll do...     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ CODE CELL                           │
│ # Create table                      │
│ t = pxt.create_table(...)           │
└─────────────────────────────────────┘

... continue pattern ...
```

## Common Issues and Solutions

### Issue: "Double titles" in rendered docs
**Cause**: H1 header in markdown cell when frontmatter already has title
**Solution**: Remove `# Title` from markdown, use frontmatter title only

### Issue: Kaggle/Colab badges don't render
**Cause**: Badge links in markdown body instead of frontmatter
**Solution**: Move links to frontmatter `description` field

### Issue: GitHub raw links broken
**Cause**: Using `raw.github.com` instead of `raw.githubusercontent.com`
**Solution**: Update to full `raw.githubusercontent.com` URLs

### Issue: Parsing errors in Mintlify
**Cause**: Missing or malformed YAML frontmatter
**Solution**: Ensure frontmatter is in a Raw cell with proper YAML syntax

## Questions?

If you encounter issues not covered here, check:
- [Quarto documentation](https://quarto.org/docs/reference/formats/markdown/docusaurus.html)
- [Mintlify documentation](https://mintlify.com/docs/)
- Existing notebooks that render correctly for examples
