# Guidelines for Writing MDX-Compatible Python Docstrings

**Purpose**: Prevent Python docstrings from generating invalid MDX that breaks Mintlify deployment

---

## Why This Matters

Python docstrings are extracted by `mintlifier` and converted to MDX for Mintlify documentation. While Python accepts many formatting patterns, **MDX has stricter parsing rules**. Invalid MDX causes:

- Silent deployment failures (pages don't render)
- Acorn parser errors
- Missing documentation
- Late discovery (issues found after release)

---

## Critical Rules

### 1. Code Fence Placement

Code fences must be on their own lines, not attached to code.

**❌ WRONG:**
```python
"""
```python
my_function(
    arg1='value',
    arg2='value'
)```
"""
```

**✅ CORRECT:**
```python
"""
```python
my_function(
    arg1='value',
    arg2='value'
)
```
"""
```

**Patterns to avoid:** `)```  `}```  `]```

---

### 2. Code Fences Must Be Complete

Don't close code fences mid-example.

**❌ WRONG:**
```python
"""
```python
tbl.update(
```
[{'id': 1, 'name': 'Alice'}],
if_not_exists='insert')
"""
```

**✅ CORRECT:**
```python
"""
```python
tbl.update(
    [{'id': 1, 'name': 'Alice'}],
    if_not_exists='insert'
)
```
"""
```

---

### 3. Backtick Pairing

Each backtick must have a matching pair. No escaped backticks.

**❌ WRONG:**
```python
"""
Store error info in `tbl.col.errormsg` tbl.col.errortype\` fields.
"""
```

**✅ CORRECT:**
```python
"""
Store error info in `tbl.col.errormsg` and `tbl.col.errortype` fields.
"""
```

---

### 4. HTML Tags Must Be Self-Closing

**❌ WRONG:**
```python
"""
Here's an image: <img src="diagram.png">
"""
```

**✅ CORRECT:**
```python
"""
Here's an image: <img src="diagram.png" />
"""
```

---

### 5. Markdown Links (Not Escaped Brackets)

**❌ WRONG:**
```python
"""
Uses \[CLIP embedding\]\[pixeltable.functions.huggingface.clip\] for indexing.
"""
```

**✅ CORRECT:**
```python
"""
Uses [CLIP embedding][pixeltable.functions.huggingface.clip] for indexing.
"""
```

---

### 6. All Code Must Be in Fenced Code Blocks

**CRITICAL**: All Python code, including REPL examples, **MUST** be enclosed in fenced code blocks.

**❌ WRONG:**
```python
"""
Example:

>>> from module import func
... result = func()
"""
```

**✅ CORRECT:**
```python
"""
Example:

```python
>>> from module import func
... result = func()
```
"""
```

**Important**:
- Use `python` language tag for Python code blocks (including REPL examples)
- Use `>>>` for primary REPL prompts
- Use `...` for continuation lines

---

## Best Practices

### Use Code Blocks for All Examples

```python
"""
Example:

```python
result = my_function(arg1, arg2)
```
"""
```

### Keep Code Blocks Self-Contained

Don't split function calls across code blocks:

```python
# ❌ WRONG
"""
```python
my_function(
```
arg1, arg2)
"""

# ✅ CORRECT
"""
```python
my_function(arg1, arg2)
```
"""
```

### Use Inline Code for Short References

```python
"""
The `column_name` parameter specifies which column to update.
"""
```

### Link to Other Documentation

```python
"""
See [`Table.add_column()`][pixeltable.Table.add_column] for details.
"""
```

---

## Common Error Messages

**"Could not parse expression with acorn"**
- **Cause**: Code or data structures outside code blocks
- **Fix**: Wrap all code in ` ```python ... ``` ` blocks

**"Unexpected closing tag"**
- **Cause**: Unclosed HTML elements
- **Fix**: Use self-closing tags: `<img />` not `<img>`

**"Could not parse import/exports with acorn"**
- **Cause**: Import statements outside code blocks
- **Fix**: Wrap imports in code blocks

---

## Quick Checklist

When writing docstrings with code examples:

- [ ] Code fences (` ``` `) are on their own lines
- [ ] No `)```  or `}```  or `]```  patterns
- [ ] All backticks are properly paired
- [ ] HTML tags are self-closing (`<img />`)
- [ ] Markdown links use `[text][ref]`, not `\[text\]`
- [ ] REPL examples are in code blocks with `>>>` and `...`
- [ ] Code blocks are complete and self-contained

**Remember**: Valid Python docstrings can generate invalid MDX. Always validate before release!
