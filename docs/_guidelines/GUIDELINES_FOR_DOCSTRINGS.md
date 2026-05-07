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

### 1. Code Examples Must Use `>>>` Prompts

Code examples must use `>>>` prompts with `...` continuation lines, not fenced code blocks.

**❌ WRONG:**
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

**✅ CORRECT:**
```python
"""
>>> my_function(
...     arg1='value',
...     arg2='value'
... )
"""
```

---

### 2. Multi-Line Examples Must Use `...` Continuations

Don't break examples across separate `>>>` prompts.

**❌ WRONG:**
```python
"""
>>> tbl.update(
>>> [{'id': 1, 'name': 'Alice'}],
>>> if_not_exists='insert')
"""
```

**✅ CORRECT:**
```python
"""
>>> tbl.update(
...     [{'id': 1, 'name': 'Alice'}],
...     if_not_exists='insert'
... )
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

### 6. Code Examples Must Use `>>>` Prompts

**CRITICAL**: All Python code examples **MUST** use `>>>` prompts, not fenced code blocks.

**❌ WRONG:**
```python
"""
Example:

```python
result = my_function(arg1, arg2)
```
"""
```

**✅ CORRECT:**
```python
"""
Example:

>>> result = my_function(arg1, arg2)
"""
```

**Important**:
- Use `>>>` for primary REPL prompts
- Use `...` for continuation lines
- Do not wrap `>>>` examples in fenced code blocks

---

## Best Practices

### Use `>>>` Prompts for All Code Examples

```python
"""
Example:

>>> result = my_function(arg1, arg2)
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
- **Cause**: Code or data structures not properly formatted
- **Fix**: Use `>>>` prompts for code examples

**"Unexpected closing tag"**
- **Cause**: Unclosed HTML elements
- **Fix**: Use self-closing tags: `<img />` not `<img>`

**"Could not parse import/exports with acorn"**
- **Cause**: Import statements not properly formatted
- **Fix**: Use `>>>` prompts for import examples

---

## Quick Checklist

When writing docstrings with code examples:

- [ ] Code fences (` ``` `) are on their own lines
- [ ] No `)```  or `}```  or `]```  patterns
- [ ] All backticks are properly paired
- [ ] HTML tags are self-closing (`<img />`)
- [ ] Markdown links use `[text][ref]`, not `\[text\]`
- [ ] Code examples use `>>>` and `...` prompts, not fenced code blocks
- [ ] Code blocks are complete and self-contained

**Remember**: Valid Python docstrings can generate invalid MDX. Always validate before release!
