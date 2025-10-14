# Pixeltable cookbook style guide

Write clear, practical how-to guides that help users solve problems with Pixeltable.

**Based on:**
- [Diátaxis framework](https://diataxis.fr/how-to-guides/) for how-to guides
- [Intuit Content Design System](https://contentdesign.intuit.com/) for tone

**Style models:**
- [calmcode.io](https://calmcode.io/) - "Code. Simply. Clearly. Calmly."
- [Python Developer Tooling Handbook](https://pydevtools.com/handbook/)

---

## Quick reference

**Directory structure:**
- `vision/` - AI vision analysis (Anthropic, OpenAI)
- `images/` - Image processing (PIL operations)
- `workflows/` - Common workflows (JSON extraction, API keys)

**Every cookbook has:**
1. Title + one-sentence description
2. **What's in this recipe:** (2-3 bullets)
3. **## Problem** (user's situation)
4. **## Solution**
   - Optional intro: "Without Pixeltable... With Pixeltable..."
   - **### Setup** (pip + imports + API keys)
   - **### Task subsections**
5. **## Explanation** (optional)
6. **## See also** (2-3 bullets + attribution)

**Tone:**
- Direct, not chatty
- Problem-first, not tool-first
- Frame as normal workflows, not error recovery
- No "we" or "let's"
- Assume developer competence

---

## Validation script

Use this script to audit all cookbooks for style guide compliance:

```python
# Save as: docs/cookbook/audit_recipes.py
import json
import re
from pathlib import Path

def audit_notebook(nb_path):
    """Audit a notebook against style guide requirements."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    issues = []
    sections = []
    see_also_bullets = 0
    has_problem = False
    has_solution = False
    in_problem_section = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])
            
            if content.startswith('## Problem'):
                has_problem = True
                in_problem_section = True
                sections.append('Problem')
            elif content.startswith('## Solution'):
                has_solution = True
                in_problem_section = False
                sections.append('Solution')
            elif content.startswith('## Explanation'):
                in_problem_section = False
                sections.append('Explanation')
            elif content.startswith('## See also'):
                in_problem_section = False
                sections.append('See also')
                # Count actual bullet lines (lines starting with '- ')
                bullet_lines = [line for line in cell['source'] if line.strip().startswith('- ')]
                see_also_bullets = len(bullet_lines)
            elif content.startswith('##'):
                in_problem_section = False
            
            # Check for Pixeltable in Problem (allow in code blocks/error messages)
            if in_problem_section and 'pixeltable' in content.lower():
                clean_content = re.sub(r'```[\s\S]*?```', '', content)
                clean_content = re.sub(r'`[^`]+`', '', clean_content)
                if 'pixeltable' in clean_content.lower():
                    issues.append("Pixeltable in Problem section (outside code)")
            
            # Check for banned section names
            banned = ['## Overview', '## What you', '## Next steps', '## Learn more']
            for banned_name in banned:
                if banned_name in content:
                    issues.append(f"Banned section: {banned_name}")
    
    # Validate required sections
    if not has_problem:
        issues.append("Missing '## Problem' section")
    if not has_solution:
        issues.append("Missing '## Solution' section")
    
    # Check section order
    if 'Problem' in sections and 'Solution' in sections:
        prob_idx = sections.index('Problem')
        sol_idx = sections.index('Solution')
        if sol_idx < prob_idx:
            issues.append("Solution before Problem")
    
    # Check See also bullets
    if 'See also' in sections and see_also_bullets > 3:
        issues.append(f"See also has {see_also_bullets} bullets (max 3)")
    
    return issues, sections

# Run audit
recipe_dirs = ['vision', 'images', 'workflows', 'iteration']
all_results = {}

for dir_name in recipe_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        for nb_file in dir_path.glob('*.ipynb'):
            issues, sections = audit_notebook(nb_file)
            all_results[str(nb_file)] = {'issues': issues, 'sections': sections}

# Print report
total_recipes = len(all_results)
clean_recipes = sum(1 for r in all_results.values() if not r['issues'])

for recipe, data in sorted(all_results.items()):
    status = "✅" if not data['issues'] else "❌"
    print(f"{status} {recipe}")
    if data['issues']:
        for issue in data['issues']:
            print(f"   ⚠️  {issue}")

print(f"\n{clean_recipes}/{total_recipes} recipes pass all checks")
```

**Checks:**
- ✅ Has `## Problem` and `## Solution` sections
- ✅ Problem comes before Solution
- ✅ No Pixeltable mentions in Problem (except in code blocks/error messages)
- ✅ No banned section names (Overview, What you, Next steps, Learn more)
- ✅ See also has ≤ 3 bullets

---

## Problem-first, not tool-first

Users arrive with a problem, not a desire to learn Pixeltable. Show them the solution.

**Good titles:**
- "Analyze images with custom prompts"
- "Extract clean data from JSON responses"
- "Set up API keys for AI services"

**Bad titles:**
- "Using Pixeltable's vision integration"
- "Working with computed columns"
- "Fix Pixeltable errors"

**Frame as normal workflows:**
- ✓ "Iterate on your data pipeline" (development is iterative)
- ✗ "Fix incorrect computed columns" (sounds like error recovery)
- "Configure API keys" (tool-focused)

**Good intro:**
> AI services require API keys. You need a way to provide them without hardcoding credentials.

**Bad intro:**
> Here's how Pixeltable discovers credentials and manages API keys for you.

---

## Tone: Direct and respectful

**Assume developer competence:**
- No security lectures
- No cost warnings
- No condescending language
- Trust developers know their domain

**Be direct:**
- ✓ "You need API keys. Your Python runtime needs to find them."
- ✗ "You need to use API keys to access AI services, but hardcoding them is a security disaster that could cost thousands."

**No "we" or "let's":**
- ✓ "Examine the response structure"
- ✗ "Let's check out the response"
- ✓ "The response contains nested JSON"
- ✗ "As we'll see, the response has nested JSON"

**Speak directly:**
- Use "you" to address the reader
- Be prescriptive, not descriptive
- Give clear guidance

**When multiple Pixeltable approaches exist:**
Don't just list options - tell users exactly when to use each one.

✓ Good:
```markdown
### Option 1: Config file

**Use when:** Local development, credentials for all projects

Create `~/.pixeltable/config.toml`:
...
```

✗ Bad:
```markdown
Here are 3 ways to configure API keys:
1. Config file
2. Environment variables
3. Getpass

Choose the one that works for you.
```

---

## Structure

### Title and intro

**Title:** What the user needs to accomplish (not the tool)
- "Batch process images with a common prompt"
- "Convert color images to grayscale"

**One-sentence description:** State the problem from user's perspective
- "Run the same AI vision prompt against multiple images automatically."
- "Transform RGB images to grayscale for analysis or model inputs."

**Never mention Pixeltable in the title or intro.**

### What's in this recipe

**2-3 bullets, action-focused:**
```markdown
**What's in this recipe:**
- Apply one prompt to an entire column of images
- Insert images in batch and get all results at once
- No loops or manual API calls
```

Not:
- ✗ "Learn how to..."
- ✗ "Understand the concepts of..."
- ✗ "Master the technique of..."

### Problem section

Describe the user's situation. No Pixeltable, no solutions yet.

```markdown
## Problem

You have multiple images that need the same analysis—like "Is this product damaged?", "Write a haiku", or "What's the dominant color?".

| Image | Prompt | Result |
|-------|--------|--------|
| sunset.jpg | "Write a haiku" | *Golden rays descend...* |
| mountains.jpg | "Write a haiku" | *Ancient stones stand tall...* |
```

**Not:**
- ✗ Security lectures
- ✗ "Why this matters" paragraphs
- ✗ Pixeltable explanations

### Solution section

**Optional intro** contrasts with non-Pixeltable approaches:

```markdown
## Solution

**Without Pixeltable:** Loop through images, call API for each, collect responses.

**With Pixeltable:** Store images in a table. Add a computed column with your prompt. All images processed automatically.
```

**Then Setup subsection:**

```markdown
### Setup
```

Contains ALL prerequisites in code cells:
- Package installation (`%pip install -qU pixeltable ...`)
- All imports
- API key configuration (if needed)
- Directory setup (`pxt.drop_dir()` / `pxt.create_dir()`)

✓ Good:
```python
# Cell 1: Install
%pip install -qU pixeltable

# Cell 2: Setup
import pixeltable as pxt
from PIL import Image

# Create a fresh directory (drop existing if present)
pxt.drop_dir('demo', force=True)
pxt.create_dir('demo')
```

✗ Bad:
```python
# Cell 1: Everything including table creation
import pixeltable as pxt
pxt.drop_dir('demo', force=True)
pxt.create_dir('demo')
t = pxt.create_table('demo.images', {'image': pxt.Image})  # ❌ Not setup
t.insert([...])
```

**Separate table creation from setup:**

Table creation (`create_table` + `insert`) goes in its own section after Setup (e.g., "### Load images")

**Then task subsections:**

```markdown
### Load images

### Basic image properties

### Resize images
```

Use `###` for all subsections under Solution.

**Subsection headings describe user goals:**
- ✓ "Analyze each image with its specific prompt"
- ✗ "Create a computed column with anthropic.messages"

**When solution requires UDFs, use test-then-apply pattern:**

If the recipe requires custom UDFs (e.g., watermarks, filters, enhancements), show the workflow:

1. **Define the UDF**
2. **Test with `.select().head()`** (preview on sample data)
3. **Apply with `add_computed_column()`** (process all data)

✓ Good:
```markdown
### Define watermark UDF

### Test the watermark

Preview the result on one image before applying to all.

### Apply to all images

Add watermarked images as a computed column.
```

Code cells:
```python
# Cell 1: Define UDF
@pxt.udf
def add_watermark(img: Image.Image, text: str) -> Image.Image:
    ...

# Cell 2: Test on sample
t.select(t.image, add_watermark(t.image, '© 2024')).head(1)

# Cell 3: Apply to all
t.add_computed_column(watermarked=add_watermark(t.image, '© 2024'))
t.select(t.image, t.watermarked).show()
```

This reinforces the fast-feedback pattern from `iteration/fast-feedback-loops.ipynb`.

### Explanation section (optional)

Only include if you need to explain:
- Trade-offs between approaches
- How discovery/lookup works
- Why certain patterns matter

Most cookbooks skip this.

**Keep it brief:**
```markdown
## Explanation

**Discovery order:**
1. Environment variable (e.g., `OPENAI_API_KEY`)
2. Config file `~/.pixeltable/config.toml`
3. Raises error if not found

**Config file is global** (all projects on your machine)
**Getpass is per-session** (better for project-specific notebooks)
```

### See also section

**2-3 related cookbooks + attribution:**

```markdown
## See also

- [Analyze images with custom prompts](./vision-custom-prompts.ipynb)
- [Extract structured data from images](./vision-structured-output.ipynb)
- *Adapted from [Anthropic's vision cookbook](https://github.com/anthropics/claude-cookbooks)*
```

Maximum 3 bullets total.

---

## Formatting

### Headings

**Use sentence case:**
- ✓ "What you've built"
- ✓ "Insert images in batch"
- ✗ "What You've Built"
- ✗ "Insert Images In Batch"

**Be descriptive and action-oriented:**
- ✓ "Extract specific fields from JSON responses"
- ✗ "JSON"

### Lists

**Numbered lists** when order matters (steps in a process)
**Bulleted lists** when order doesn't matter (features, benefits)

### Code comments

Explain **why**, not just what:

```python
# Good: explains reasoning
t.add_computed_column(
    # Use haiku model for concise responses
    response=anthropic.messages(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='claude-3-haiku-20240307'
    )
)

# Bad: just describes what the code does
t.add_computed_column(
    # Add a computed column with anthropic.messages
    response=anthropic.messages(...)
)
```

**Never add emojis** unless user explicitly requests them.

### Delivering bad news or limitations

Inspired by [Intuit's bad news guidelines](https://contentdesign.intuit.com/talking-to-customers/bad-news/):

**Deliver upfront:**
- ✓ "Drop operations cannot be undone—data is permanently deleted."
- ✗ "Well, there's this thing you should know... it might be important..."

**Focus on the solution:**
- ✓ "Use `if_exists='replace'` to fix incorrect columns."
- ✗ "You can't just drop columns because they have dependents and..."

**Stay calm, not alarmist:**
- ✓ "Drop operations cannot be undone"
- ✗ "WARNING! DANGEROUS! This will destroy everything!"
- ✗ "This is a disaster waiting to happen"

**Keep language positive:**
Focus on what users CAN do, not what they can't:
- ✓ "Use `if_exists='replace'` to update columns"
- ✗ "You can't drop columns without..."

**Don't detail the problem:**
- ✓ "Drop operations cannot be undone. Use `if_exists='replace'` instead."
- ✗ "The issue here is that Pixeltable's internal implementation doesn't track history and there's no undo mechanism because..."

---

## Code patterns

### Show Pixeltable patterns

**Use batch operations:**
```python
# Good
t.insert([{'image': url1}, {'image': url2}, {'image': url3}])

# Avoid
for url in urls:
    t.insert([{'image': url}])
```

**Display results with `.show()`:**
```python
# Good
t.select(t.image, t.response).show()

# Avoid
for row in t.select(t.image, t.response):
    print(row)
```

### Use the correct API

**`add_computed_column()` for computed columns:**
```python
# Correct
t.add_computed_column(response=anthropic.messages(...))

# Wrong
t.add_column(response=anthropic.messages(...))
```

`add_column()` is for ordinary (non-computed) columns with just a type.

---

## Template

```markdown
# [What the user needs to accomplish]

[One sentence: what problem this solves]

**What's in this recipe:**
- [Capability 1]
- [Capability 2]
- [Capability 3]

*[Attribution if adapted]*

## Problem

[Describe user's situation in 2-4 sentences]

[Show example table or data structure]

## Solution

**Without Pixeltable:** [Alternative approach]

**With Pixeltable:** [Pixeltable approach]

### Setup

### Install required packages

```python
%pip install -qU pixeltable
```

```python
import pixeltable as pxt
```

### [First task subsection]

[Brief explanation]

```python
# Code example
```

### [Second task subsection]

...

## Explanation

[OPTIONAL. Only include if needed for trade-offs, alternatives, or discovery]

## See also

- [Link to related cookbook]
- [Link to related cookbook]
- *[Attribution if adapted from external source]*
```

---

## For LLMs

This guide uses consistent patterns. Look for "✓" and "✗" examples.

**Validation snippets:**

Check heading format:
```python
import re
def check_heading_case(heading):
    # Should be "Word word word" not "Word Word Word"
    words = heading.split()
    proper_nouns = ['Pixeltable', 'JSON', 'API', 'RGB', 'PIL', 'UDF']
    return words[0][0].isupper() and all(
        w[0].isupper() if w in proper_nouns else w[0].islower()
        for w in words[1:]
    )
```

Check for banned sections:
```python
banned = ['Overview', 'What you\'ve built', 'Next steps', 
          'What you\'ll learn', 'Summary', 'Conclusion']
```

---

**Quick transformation functions:**

Fix heading case:
```python
def fix_heading_case(heading):
    proper_nouns = ['Pixeltable', 'JSON', 'API', 'RGB', 'PIL', 'UDF']
    words = heading.split()
    return ' '.join([
        w if w in proper_nouns else w.lower() if i > 0 else w.capitalize()
        for i, w in enumerate(words)
    ])
```

Remove emojis:
```python
import re
def remove_emojis(text):
    return re.sub(r'[^\w\s,.-]', '', text)
```
