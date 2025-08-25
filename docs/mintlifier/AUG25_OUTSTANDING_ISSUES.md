# Outstanding Documentation Issues - Aug 25

## 1. Examples Section Not Parsed by docstring_parser

### Problem
The `docstring_parser` library is not properly parsing the Examples sections in Pixeltable docstrings, causing warnings to appear in the generated documentation even though the examples ARE being extracted manually and displayed.

### Root Cause
The `docstring_parser` library's Google parser treats the `Examples` section as a `SINGULAR` type section (see [documentation](https://rr-.github.io/docstring_parser/docstring_parser.google.html#DEFAULT_SECTIONS)). This means:

1. The entire Examples section is treated as a single text block
2. The parser does NOT attempt to extract individual code snippets
3. Everything under `Examples:` gets put into the `description` field of `DocstringExample` objects
4. The `snippet` field remains `None` (it's not used for Google-style docstrings)

This is by design - the Google parser in `docstring_parser` is not meant to extract executable code from Examples sections.

### Google Style Guide Reference
According to the [Google Python Style Guide - Section 3.8.5](https://google.github.io/styleguide/pyguide.html#385-docstring-example):

> Examples should be written in doctest format, and should illustrate how to use the function.

### Correct Format
```python
"""
Examples:
    Description of first example:

    >>> print([i for i in example_generator(4)])
    [0, 1, 2, 3]
    
    Description of second example:
    
    >>> print(another_example())
    output
"""
```

Key requirements:
1. **Blank line** between description and code block
2. Code blocks start with `>>>`
3. Continuation lines use `...`
4. Expected output (optional) follows the code

### Common Issues in Pixeltable Docstrings

#### Issue 1: Missing blank line between description and code
**WRONG:**
```python
Examples:
    Drop the index `idx1` of the table `my_table` by index name:
    >>> tbl = pxt.get_table('my_table')
    ... tbl.drop_index(idx_name='idx1')
```

**CORRECT:**
```python
Examples:
    Drop the index `idx1` of the table `my_table` by index name:

    >>> tbl = pxt.get_table('my_table')
    ... tbl.drop_index(idx_name='idx1')
```

#### Issue 2: Multi-line descriptions might need better formatting
**CURRENT:**
```python
Examples:
    Add a computed column that applies the model `claude-3-5-sonnet-20241022`
    to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

    >>> msgs = [{'role': 'user', 'content': tbl.prompt}]
```

**MIGHT BE BETTER:**
```python
Examples:
    Add a computed column that applies the model to an existing column:

    >>> msgs = [{'role': 'user', 'content': tbl.prompt}]
    >>> tbl.add_computed_column(response=messages(msgs, model='claude-3-5-sonnet-20241022'))
```

### Real Example from Codebase
File: `pixeltable/catalog/table.py` - `drop_index` method

This docstring has multiple examples where the third one is missing the blank line:
```python
Examples:
    Drop the index on the `img` column of the table `my_table` by column name:

    >>> tbl = pxt.get_table('my_table')
    ... tbl.drop_index(column_name='img')

    Drop the index on the `img` column of the table `my_table` by column reference:

    >>> tbl = pxt.get_table('my_table')
    ... tbl.drop_index(tbl.img)

    Drop the index `idx1` of the table `my_table` by index name:
    >>> tbl = pxt.get_table('my_table')  # <-- MISSING BLANK LINE ABOVE
    ... tbl.drop_index(idx_name='idx1')
```

### Recommended Actions

1. **Short-term fix**: 
   - Set `show_errors=False` in the documentation generation to hide warnings
   - The manual extraction is working, so examples ARE being displayed

2. **Medium-term fix**:
   - Create a linting rule or pre-commit hook to validate Examples sections
   - Fix all existing docstrings to follow consistent format

3. **Long-term fix**:
   - Consider contributing to `docstring_parser` to better handle these cases
   - OR switch to a different docstring parser that's more forgiving
   - OR create custom parser for Examples sections

### Test Script
To validate docstring format:
```python
from docstring_parser import parse

def check_examples(docstring):
    parsed = parse(docstring)
    if 'Examples:' in docstring or 'Example:' in docstring:
        if not parsed.examples or not any(ex.snippet for ex in parsed.examples):
            return False, "Examples not properly parsed"
    return True, "OK"
```

### Files to Check
Run this to find all files with Examples sections:
```bash
grep -r "Examples:" pixeltable/ --include="*.py" | wc -l
```

Currently there are many files with Examples sections that may need formatting fixes.

## Current Workaround in Documentation Generator

The `page_function.py` and `page_method.py` files currently:
1. Try to parse with `docstring_parser` 
2. Show a warning if `parsed.examples` is empty or None
3. **BUT** then manually extract examples with `_document_examples()` method
4. The manual extraction WORKS and displays the examples correctly

**The issue**: The warning is misleading because examples ARE being shown, just not via the parser.

### To disable warnings before demo:
In `page_function.py` and `page_method.py`, set `show_errors=False` when instantiating the page objects, or modify the check:

```python
# Current code that shows warning even when manual extraction works:
if self.show_errors and ('Examples:' in doc or 'Example:' in doc):
    if not hasattr(parsed, 'examples') or not parsed.examples:
        content += "## ⚠️ Documentation Issues\n\n"
        content += "<Warning>\n- Examples section exists in docstring but was not parsed by docstring_parser\n</Warning>\n\n"
```

**Better approach**: Only show warning if manual extraction also fails:
```python
# Check if manual extraction will succeed
examples_content = self._document_examples(doc)
if self.show_errors and ('Examples:' in doc or 'Example:' in doc):
    if (not hasattr(parsed, 'examples') or not parsed.examples) and not examples_content:
        content += "## ⚠️ Documentation Issues\n\n"
        content += "<Warning>\n- Examples section could not be extracted\n</Warning>\n\n"
```