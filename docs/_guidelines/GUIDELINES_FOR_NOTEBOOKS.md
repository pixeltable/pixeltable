# Guidelines for Writing Pixeltable Notebooks

**Purpose**: Best practices for creating educational and functional Jupyter notebooks for Pixeltable documentation

---

## Overview

The docs build converts each notebook to a Mintlify page with Quarto. The page title comes from a
leading H1 or a raw YAML frontmatter cell. The build adds Kaggle, Colab, and notebook download links;
do not hand-write them in cells.

## Required: Title in the first cell

Use one title source. By default, start with a **markdown** cell whose first line is the notebook's
only H1. Quarto converts that H1 to the page title:

```markdown
# Build a RAG pipeline

Create a retrieval-augmented generation system that answers questions using your documents as context.
```

Use `##` for sections and `###` for subsections below it. Do not add a second `#` anywhere in the
notebook.

Alternatively, start with a **raw** cell containing YAML frontmatter with a `title` field, enclosed
by `---` lines. In that form, do not include an H1 in any markdown cell.

## Required: No Download or Badge Links

**Do NOT include "Download Notebook" badges or similar HTML badge images in markdown cells.**

The docs build generates these links. Hand-written badges duplicate them.

### ❌ Wrong
```html
<a href="..."><img src="https://img.shields.io/badge/..." alt="Download Notebook"></a>
```

Do not add these links to frontmatter `description` either. The build generates them from the notebook path.

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
- Keep outputs: `tool/check_notebooks.py` requires them on at least 50% of code cells
- Clear only outputs that are noise, such as progress bars or warnings
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
2. **Check the title source**: a leading markdown H1, or a first raw cell with YAML `title`, but not both
3. **Check heading levels**: use `##` and below after the title; frontmatter-based notebooks must have no H1
4. **Test links** to ensure they point to correct locations
5. **Review output** to ensure it's appropriate for documentation

## Example Notebook Structure

The default structure starts with a markdown title cell. For the frontmatter alternative, add a raw
YAML title cell before it and omit `# Example Notebook` from the markdown.

**Markdown cell:**

```markdown
# Example Notebook

Brief introduction to the notebook.

## Prerequisites

What you need to know...
```

**Code cell:**

```python
import pixeltable as pxt
```

**Markdown cell:**

```markdown
## First section

Explanation of what we'll do...
```

Continue with code and markdown cells for each step.

## Common Issues and Solutions

### Issue: "Double titles" in rendered docs
**Cause**: H1 header in markdown cell when frontmatter already has title
**Solution**: Keep one title source: remove the raw title cell to use the leading H1, or remove the H1 to keep frontmatter

### Issue: Duplicate notebook links
**Cause**: Hand-written links duplicate the generated links
**Solution**: Remove hand-written badges and open-in links; the docs build adds them

### Issue: GitHub raw links broken
**Cause**: Using `raw.github.com` instead of `raw.githubusercontent.com`
**Solution**: Update to full `raw.githubusercontent.com` URLs

### Issue: Malformed YAML frontmatter
**Solution**: If you use frontmatter, put valid YAML between `---` lines in the first raw cell. Frontmatter is not required for the leading-H1 form.

## Questions?

If you encounter issues not covered here, check:
- [Quarto documentation](https://quarto.org/docs/reference/formats/markdown/docusaurus.html)
- [Mintlify documentation](https://mintlify.com/docs/)
- Existing notebooks that render correctly for examples
