# Contributing recipes to the Pixeltable cookbook

Thanks for helping make Pixeltable more accessible! Here's how to write a recipe that helps users succeed.

## Quick start

1. **Read [STYLE_GUIDE.md](STYLE_GUIDE.md)** - Writing standards based on Diátaxis and Intuit's content design system
2. **Follow the recipe structure**:
   - **## Problem** - User's situation (no Pixeltable mentions)
   - **## Solution** - How to solve it
     - Optional intro: "Without Pixeltable... With Pixeltable..."
     - **### Setup** (pip + imports + directory setup)
     - **### Task subsections** (action-focused: "Load images", "Apply filters")
   - **## Explanation** (optional - when to use, how it works)
   - **## See also** (≤ 3 bullets, include attributions)

## Recipe checklist

Before you submit, make sure your recipe includes:

- [ ] `## Problem` section describing the user's situation (no Pixeltable mentions outside code)
- [ ] `## Solution` section showing how to solve it
- [ ] Problem comes before Solution
- [ ] Setup includes pip, imports, and `drop_dir`/`create_dir`
- [ ] Table creation is separate from setup (in its own "Load images" or similar section)
- [ ] **Query-and-Commit pattern**: Every transformation uses `.select()` with `.collect()` (or `.head(n)` to collect only first n rows) to preview, then `add_computed_column()` to save (same expression)
- [ ] For UDF recipes: define → query with `.select().collect()` (or `.head(n)`) → commit with `add_computed_column()`
- [ ] Action-focused subsection headings ("Load images" not "Create a table")
- [ ] Direct, respectful tone (no "we", no "let's")
- [ ] See also has ≤ 3 bullets
- [ ] All code runs successfully

Avoid these section names: "Overview", "What you'll learn", "Next steps", "Learn more"

## Directory structure

Place your recipe in the appropriate directory:

Recipes use prefixes to indicate their category:

- `doc-` - Document processing (office files, PDFs, text extraction)
- `img-` - Image processing (PIL/Pillow operations and custom UDFs)
- `vision-` - AI vision analysis (Anthropic, OpenAI)  
- `workflow-` - Common workflows (JSON extraction, API keys)
- `iteration/` - Development patterns (refining columns, testing, caching)

## Example recipes

These recipes demonstrate good structure and style:

- [img-rgb-to-grayscale.ipynb](img-rgb-to-grayscale.ipynb) - Clear Problem → Solution flow, comparison table, Query-and-Commit pattern
- [img-pil-transforms.ipynb](img-pil-transforms.ipynb) - Clean structure, multiple techniques in one recipe
- [img-brightness-contrast.ipynb](img-brightness-contrast.ipynb) - UDF examples with Query-and-Commit workflow

## Check your recipe structure

The STYLE_GUIDE.md includes a helpful script that shows how your recipe matches the expected structure:

```bash
cd docs/cookbook
python3 << 'EOF'
import re
with open('STYLE_GUIDE.md', 'r') as f:
    content = f.read()
    match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    if match:
        exec(match.group(1))
EOF
```

The script highlights which sections are present and points out anything that might be missing.

## Submitting your recipe

1. Make sure all cells run successfully
2. Run the structure check script above - it shows what's working and what needs adjustment
3. Review the checklist - does your recipe include everything?
4. Submit a pull request with your recipe in the appropriate directory
5. In your PR description, mention that you followed STYLE_GUIDE.md

## Questions?

- [Discord](https://discord.gg/QPyqFYx2UN) - Ask in #documentation
- [GitHub Issues](https://github.com/pixeltable/pixeltable/issues) - Report bugs or suggest improvements
